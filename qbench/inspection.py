"""Runtime-first model inspection and support resolution."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn

from .capture import capture_scenario, clone_invocation
from .registry import (
    KERNEL_SPECS,
    STRUCTURAL_CAPABILITY_VERSION,
    STRUCTURAL_SCHEMAS,
    find_kernel,
    module_semantic_configuration,
    schema_route_key,
)
from .provenance import model_provenance
from .schemas import (
    InspectionConfig,
    InspectionResult,
    OperationRecord,
    QBenchError,
    Scenario,
    SimulationPlan,
    VerificationResult,
    normalize_scenarios,
    redacted_exception,
)


def _ensure_registrations() -> str | None:
    try:
        import qbench.ops  # noqa: F401

        return None
    except Exception as exc:
        # Dispatcher inspection remains useful without optional simulator deps.
        return f"Simulator module registry unavailable: {redacted_exception(exc)}"


def _standard_module_kernel(module: nn.Module):
    # Semantic collapse is only sound for an untouched stock module call.
    # Instance-level forward replacement and user hooks can execute arbitrary
    # operations outside the native forward body, so those cases must remain
    # visible in the dispatcher ledger.
    if "forward" in getattr(module, "__dict__", {}):
        return None
    if getattr(module, "_forward_pre_hooks", None) or getattr(
        module, "_forward_hooks", None
    ):
        return None
    module_runtime = torch.nn.modules.module
    if getattr(module_runtime, "_global_forward_pre_hooks", None) or getattr(
        module_runtime, "_global_forward_hooks", None
    ):
        return None
    for spec in KERNEL_SPECS:
        for native in spec.module_types:
            if type(module) is native and getattr(
                type(module), "forward", None
            ) is getattr(native, "forward", None):
                return spec, native
    return None


def _argument_metadata(value: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    metadata = value if isinstance(value, dict) else {}
    return tuple(metadata.get("args", ())), dict(metadata.get("kwargs", {}))


def _base_module_kernel(module: nn.Module, module_arguments: Any = None):
    semantic = _standard_module_kernel(module)
    if semantic is None:
        return None
    spec, native = semantic
    if not spec.ready:
        return None
    args, kwargs = _argument_metadata(module_arguments)
    registered = spec.implementation_for(native)
    if registered is not None and spec.accepts_module(module, args, kwargs):
        return spec, registered
    return None


def _kernel_gap_reason(
    schema: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str:
    candidates = [spec for spec in KERNEL_SPECS if schema in spec.schemas]
    if not candidates:
        if not schema.startswith("aten::"):
            return "custom namespace has no exact ready KernelSpec"
        return "unknown schema or overload"
    if not any(spec.ready for spec in candidates):
        return "known KernelSpec is not ready"
    if not any(spec.matches(schema, args, kwargs) for spec in candidates if spec.ready):
        return "argument, shape, dtype, or dimension constraints not satisfied"
    if not any(spec.handler is not None for spec in candidates if spec.ready):
        return "known KernelSpec has no ready runtime handler"
    return "no exact ready KernelSpec"


def _snapshot_runtime(model: nn.Module):
    return {
        "training": {path: module.training for path, module in model.named_modules()},
        "cpu_rng": torch.random.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        ),
    }


def _restore_runtime(model: nn.Module, state):
    for path, module in model.named_modules():
        if path in state["training"]:
            module.training = state["training"][path]
    torch.random.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def _graph_diagnostics(
    model: nn.Module, scenario: Scenario, config: InspectionConfig
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if config.enable_fx:
        try:
            # Graph tools are optional enrichment and may execute arbitrary
            # model code.  Give each tool its own clone so mutations performed
            # while tracing cannot alter the model used by authoritative eager
            # capture, capability resolution, or verification.
            graph = torch.fx.symbolic_trace(copy.deepcopy(model))
            node_count = len(list(graph.graph.nodes))
            del graph
            result["fx"] = {"succeeded": True, "nodes": node_count}
        except Exception as exc:
            result["fx"] = {
                "succeeded": False,
                "error": redacted_exception(exc),
            }
    else:
        result["fx"] = {"succeeded": False, "disabled": True}
    if config.enable_export:
        try:
            args, kwargs = clone_invocation(scenario)
            exported = torch.export.export(copy.deepcopy(model), args, kwargs=kwargs)
            node_count = len(list(exported.graph.nodes))
            del exported
            result["export"] = {
                "succeeded": True,
                "nodes": node_count,
            }
        except Exception as exc:
            result["export"] = {
                "succeeded": False,
                "error": redacted_exception(exc),
            }
    else:
        result["export"] = {"succeeded": False, "disabled": True}
    return result


def _resolve_operations(
    model: nn.Module,
    operations: list[OperationRecord],
    *,
    allow_fp32_fallback: bool = False,
):
    named_modules = list(model.named_modules(remove_duplicate=False))
    modules = dict(named_modules)
    aliases_by_identity: dict[int, list[str]] = defaultdict(list)
    for path, module in named_modules:
        aliases_by_identity[id(module)].append(path)
    shared_module_ids = {
        identity
        for identity, aliases in aliases_by_identity.items()
        if len(aliases) > 1
    }
    decisions: dict[str, str] = {}
    kernel_rows: dict[str, dict[str, Any]] = {}
    gaps: Counter[tuple[str, str]] = Counter()
    gap_reasons: dict[tuple[str, str], str] = {}
    by_module: dict[str, list[OperationRecord]] = defaultdict(list)
    seen_module_invocations: set[tuple[str, str, int]] = set()

    def add_kernel(
        key: str,
        row: dict[str, Any],
        scenario: str,
        module_path: str,
        module_invocation: Mapping[str, Any] | None = None,
    ) -> None:
        entry = kernel_rows.setdefault(key, row)
        entry["source_count"] = int(entry.get("source_count", 0)) + 1
        scenario_counts = entry.setdefault("scenario_counts", {})
        scenario_counts[scenario] = int(scenario_counts.get(scenario, 0)) + 1
        module_paths = entry.setdefault("module_paths", [])
        if module_path not in module_paths:
            module_paths.append(module_path)
        module_path_counts = entry.setdefault("module_path_counts", {})
        module_path_counts[module_path] = (
            int(module_path_counts.get(module_path, 0)) + 1
        )
        invocation_rows = entry.setdefault("module_invocations", {})
        if module_invocation is not None:
            invocation_rows.setdefault(module_path, []).append(
                {
                    "scenario": scenario,
                    "invocation": int(module_invocation.get("invocation", 0)),
                    "args": copy.deepcopy(module_invocation.get("args", ())),
                    "kwargs": copy.deepcopy(module_invocation.get("kwargs", {})),
                    "module_configuration": module_semantic_configuration(
                        modules[module_path]
                    ),
                }
            )

    for operation in operations:
        by_module[operation.module_path].append(operation)
        module = modules.get(operation.module_path)
        semantic = (
            _base_module_kernel(module, operation.module_arguments)
            if module is not None and id(module) not in shared_module_ids
            else None
        )
        if semantic is not None:
            spec, replacement = semantic
            operation.classification = "composite_detail"
            operation.kernel = spec.name
            implementation_path = spec.implementation_path_for(type(module))
            if implementation_path is None:
                raise QBenchError("Maintained module implementation path is missing")
            decisions[operation.module_path] = f"module:{implementation_path}"
            invocation = (
                operation.module_arguments.get("invocation", operation.sequence)
                if isinstance(operation.module_arguments, dict)
                else operation.sequence
            )
            invocation_key = (
                operation.module_path,
                operation.scenario,
                int(invocation),
            )
            if invocation_key not in seen_module_invocations:
                seen_module_invocations.add(invocation_key)
                add_kernel(
                    f"module:{operation.module_path}",
                    spec.to_dict(),
                    operation.scenario,
                    operation.module_path,
                    (
                        operation.module_arguments
                        if isinstance(operation.module_arguments, Mapping)
                        else {}
                    ),
                )
            continue
        standard_semantic = (
            _standard_module_kernel(module)
            if module is not None and id(module) not in shared_module_ids
            else None
        )
        if standard_semantic is not None:
            _spec, native = standard_semantic
            operation.classification = (
                "fp32_fallback" if allow_fp32_fallback else "unsupported"
            )
            gaps[(operation.schema, operation.scenario)] += 1
            replacement = _spec.implementation_for(native)
            module_args, module_kwargs = _argument_metadata(operation.module_arguments)
            if not _spec.ready:
                reason = "maintained module KernelSpec is not ready"
            elif not _spec.accepts_module(module, module_args, module_kwargs):
                reason = "module invocation constraints not satisfied"
            elif replacement is None:
                reason = "maintained KernelSpec has no pinned simulator replacement"
            else:
                reason = "maintained simulator replacement is not ready"
            gap_reasons[(operation.schema, operation.scenario)] = reason
            continue
        if operation.schema in STRUCTURAL_SCHEMAS:
            operation.classification = "structural"
            operation.kernel = "structural-v1"
            add_kernel(
                f"schema:{operation.schema}",
                {
                    "name": "structural-v1",
                    "classification": "structural",
                    "ready": True,
                    "schema": operation.schema,
                    "conversion": "structural",
                    "argument_constraints": {},
                    "handler_quantized": False,
                    "counts_as_quantized": False,
                    "quantizes_weights": False,
                    "weight_operand": None,
                    "weight_argument": None,
                    "activation_policy": False,
                    "schemas": [operation.schema],
                    "module_types": [],
                    "module_implementations": [],
                    "input_operands": {},
                    "policy_overrides": {},
                    "module_invocations": {},
                    "conformance": {"kind": "bit_exact", "evidence": "missing"},
                },
                operation.scenario,
                operation.module_path,
            )
            continue
        args, kwargs = _argument_metadata(operation.arguments)
        spec = find_kernel(operation.schema, args, kwargs)
        if spec is not None and spec.handler is not None:
            operation.classification = spec.classification
            operation.kernel = spec.name
            add_kernel(
                schema_route_key(operation.schema, spec),
                spec.to_dict(),
                operation.scenario,
                operation.module_path,
            )
        else:
            operation.classification = (
                "fp32_fallback" if allow_fp32_fallback else "unsupported"
            )
            gaps[(operation.schema, operation.scenario)] += 1
            gap_reasons[(operation.schema, operation.scenario)] = _kernel_gap_reason(
                operation.schema, args, kwargs
            )
    return decisions, kernel_rows, gaps, gap_reasons, by_module


def _support_report(
    model,
    scenarios,
    scenario_status,
    operations,
    gaps,
    gap_reasons,
    by_module,
    invoked_by_module,
    config,
):
    module_rows = []
    scenario_names = [scenario.name for scenario in scenarios]
    for path, module in model.named_modules(remove_duplicate=False):
        records = by_module.get(path, [])
        aggregate = Counter(record.schema for record in records)
        classes = Counter(record.classification for record in records)
        semantic_counts = Counter(record.kernel for record in records if record.kernel)
        semantic_scenarios: dict[str, set[str]] = defaultdict(set)
        for record in records:
            if record.kernel:
                semantic_scenarios[record.kernel].add(record.scenario)
        invoked_scenarios = sorted(invoked_by_module.get(path, set()))
        if not invoked_scenarios:
            status = "not_assessed"
        elif classes["unsupported"] or classes["fp32_fallback"]:
            status = "unsupported"
        else:
            status = "supported"
        module_rows.append(
            {
                "path": path,
                "type": f"{type(module).__module__}.{type(module).__qualname__}",
                "status": status,
                "operation_count": len(records),
                "operations": dict(sorted(aggregate.items())),
                "semantic_kernels": {
                    kernel: {
                        "count": count,
                        "example_scenarios": sorted(semantic_scenarios[kernel])[:3],
                    }
                    for kernel, count in sorted(semantic_counts.items())
                },
                "scenarios": invoked_scenarios,
                "aliases": next(
                    (
                        list(record.module_aliases)
                        for record in records
                        if len(record.module_aliases) > 1
                    ),
                    [path],
                ),
            }
        )
    gap_rows = [
        {
            "schema": schema,
            "scenario": scenario,
            "count": count,
            "reason": gap_reasons[(schema, scenario)],
        }
        for (schema, scenario), count in sorted(gaps.items())
    ]
    captured = [name for name, status in scenario_status.items() if status["succeeded"]]
    return {
        "schema_version": 3,
        "qualification": "captured scenarios only",
        "fully_supported": False,
        "capture_complete": len(captured) == len(scenario_names),
        "scenario_coverage": scenario_status,
        "captured_scenarios": captured,
        "module_summary": module_rows,
        "not_assessed_modules": [
            row["path"] for row in module_rows if row["status"] == "not_assessed"
        ],
        "gaps": gap_rows,
        "operation_count": len(operations),
        "structural_capability_version": STRUCTURAL_CAPABILITY_VERSION,
        "replacement_coverage": not gap_rows,
        "strict_realization": False,
        "routing_dry_run_verified": False,
        "quantized_execution_verified": False,
        "hardware_fidelity": {"status": "missing_evidence"},
        "allow_fp32_fallback": config.allow_fp32_fallback,
        "outside_v1": [
            "training/backward",
            "opaque TorchScript",
            "compiled-only models",
            "distributed/FSDP",
        ],
    }


def _hardware_fidelity(plan: SimulationPlan, directory: str | None) -> dict[str, Any]:
    used_capabilities: dict[str, str] = {}
    for route, row in plan.kernels.items():
        kernel = row.get("name")
        if row.get("classification") == "structural" or not kernel:
            continue
        if route.startswith("schema:"):
            capability = route
        elif route.startswith("module:"):
            path = route.removeprefix("module:")
            decision = plan.module_decisions.get(path)
            capability = (
                decision
                if isinstance(decision, str) and decision.startswith("module:")
                else f"module:<unresolved:{path}>"
            )
        else:
            continue
        used_capabilities[capability] = str(kernel)
    if not used_capabilities:
        return {"status": "not_applicable", "kernels": {}, "capabilities": {}}
    used = set(used_capabilities.values())
    if directory is None:
        return {
            "status": "missing_evidence",
            "kernels": {name: "missing_evidence" for name in sorted(used)},
            "capabilities": {
                capability: "missing_evidence"
                for capability in sorted(used_capabilities)
            },
        }

    from .kernels import _capability_policy_key, _policy_sha256, verify_kernels

    report = verify_kernels(directory)
    policy_sha256 = _policy_sha256(plan.quantization_policy.to_dict())
    evidence_keys = {
        capability: _capability_policy_key(capability, policy_sha256)
        for capability in used_capabilities
    }
    available = report.get("capability_policy_statuses", {})
    capability_statuses = {
        capability: str(available.get(evidence_keys[capability], "missing_evidence"))
        for capability in sorted(used_capabilities)
    }
    precedence = {
        "passed": 0,
        "missing_evidence": 1,
        "failed": 2,
        "configuration_error": 3,
    }
    statuses: dict[str, str] = {}
    for capability, value in capability_statuses.items():
        kernel = used_capabilities[capability]
        previous = statuses.get(kernel)
        if previous is None or precedence.get(value, 3) > precedence.get(previous, 3):
            statuses[kernel] = value
    values = set(capability_statuses.values())
    result_rows = report.get("kernels", [])
    global_configuration_error = bool(
        report.get("status") == "configuration_error" and not result_rows
    )
    if "configuration_error" in values or global_configuration_error:
        status = "configuration_error"
    elif "failed" in values:
        status = "failed"
    elif values == {"passed"}:
        status = "passed"
    else:
        status = "missing_evidence"
    scoped_errors = [
        str(row["error"])
        for row in result_rows
        if row.get("evidence_key") in set(evidence_keys.values()) and row.get("error")
    ]
    if global_configuration_error:
        scoped_errors.extend(str(error) for error in report.get("errors", []))
    return {
        "status": status,
        "kernels": statuses,
        "capabilities": capability_statuses,
        "quantization_policy_sha256": policy_sha256,
        "bundle_status": report.get("status", "missing_evidence"),
        "errors": scoped_errors,
    }


def inspect_model(
    model: nn.Module, scenarios, config: InspectionConfig | dict[str, Any] | None = None
) -> InspectionResult:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    config = InspectionConfig.coerce(config)
    scenarios = normalize_scenarios(scenarios)
    _validate_capture_device(model, scenarios, config.device)
    state = _snapshot_runtime(model)
    registry_warning = _ensure_registrations()
    operations: list[OperationRecord] = []
    scenario_status: dict[str, dict[str, Any]] = {}
    invoked_by_module: dict[str, set[str]] = defaultdict(set)
    callsite_keys: set[tuple[str, str]] = set()
    try:
        model.eval()
        for scenario in scenarios:
            invoked_modules: set[str] = set()
            scenario_records: list[OperationRecord] = []
            try:
                capture_scenario(
                    model,
                    scenario,
                    capture_callsites=config.capture_callsites,
                    invoked_modules=invoked_modules,
                    operation_sink=scenario_records,
                    callsite_keys=callsite_keys,
                )
                scenario_status[scenario.name] = {
                    "succeeded": True,
                    "operation_count": len(scenario_records),
                }
            except Exception as exc:
                scenario_status[scenario.name] = {
                    "succeeded": False,
                    "operation_count": len(scenario_records),
                    "error": redacted_exception(exc),
                }
            finally:
                for index, record in enumerate(scenario_records, start=len(operations)):
                    record.sequence = index
                operations.extend(scenario_records)
                for path in invoked_modules:
                    invoked_by_module[path].add(scenario.name)
        diagnostics = _graph_diagnostics(model, scenarios[0], config)
        diagnostics["environment"] = {
            "torch_version": str(torch.__version__),
            "inspection_config": asdict(config),
        }
        try:
            diagnostics["provenance"] = model_provenance(model)
        except Exception as exc:
            diagnostics["provenance"] = {
                "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
                "state_digest_error": redacted_exception(exc),
            }
    finally:
        _restore_runtime(model, state)
    if registry_warning:
        diagnostics["registry"] = {"succeeded": False, "error": registry_warning}
    decisions, kernels, gaps, gap_reasons, by_module = _resolve_operations(
        model, operations, allow_fp32_fallback=config.allow_fp32_fallback
    )
    for scenario in scenarios:
        status = scenario_status[scenario.name]
        scenario_gaps = {
            schema: count
            for (schema, gap_scenario), count in sorted(gaps.items())
            if gap_scenario == scenario.name
        }
        status["gap_count"] = sum(scenario_gaps.values())
        status["unresolved_schemas"] = list(scenario_gaps)
        status["supported"] = bool(status["succeeded"] and not scenario_gaps)
    registry_gap = "qbench::simulator_registry"
    unresolved = sorted({schema for schema, _scenario in gaps})
    if registry_warning:
        unresolved.append(registry_gap)
        for status in scenario_status.values():
            status["supported"] = False
            status["unresolved_schemas"] = sorted(
                set(status["unresolved_schemas"]) | {registry_gap}
            )
            status["gap_count"] += 1
    plan = SimulationPlan(
        kernels=kernels,
        module_decisions=decisions,
        unresolved_schemas=unresolved,
        allow_fp32_fallback=config.allow_fp32_fallback,
        quantization_enabled=config.quantization_enabled,
        quantization_policy=config.quantization_policy,
        scenarios=scenarios,
    )
    plan.validate_policy_routes()
    support = _support_report(
        model,
        scenarios,
        scenario_status,
        operations,
        gaps,
        gap_reasons,
        by_module,
        invoked_by_module,
        config,
    )
    support["hardware_fidelity"] = _hardware_fidelity(
        plan, config.conformance_directory
    )
    support["registry_ready"] = registry_warning is None
    if registry_warning:
        support["replacement_coverage"] = False
        support["gaps"].append(
            {
                "schema": registry_gap,
                "scenario": "*",
                "count": 1,
                "reason": registry_warning,
            }
        )
    verification = VerificationResult(
        attempted=False, strict=not config.allow_fp32_fallback
    )
    result = InspectionResult(support, operations, plan, verification, diagnostics)
    if config.verify and support["capture_complete"]:
        from .conversion import build_simulator

        verification_state = _snapshot_runtime(model)
        cpu_routing_only = bool(
            plan.quantization_enabled
            and _inspection_device_type(model, scenarios, config.device) == "cpu"
        )
        verification_plan = plan
        if cpu_routing_only:
            # CPU inspection proves the exact conversion/routing plan without
            # invoking CUDA-only quantizers.  The requested plan remains
            # quantization-enabled, so the missing GPU execution evidence is
            # still reflected as a partial verdict on its own report axis.
            verification_plan = copy.deepcopy(plan)
            verification_plan.quantization_enabled = False
        try:
            simulator = build_simulator(
                model, verification_plan, strict=not config.allow_fp32_fallback
            )
            verification = simulator.verify(scenarios)
        except Exception as exc:
            verification = VerificationResult(
                attempted=True,
                succeeded=False,
                strict=not config.allow_fp32_fallback,
                errors=[redacted_exception(exc)],
            )
        finally:
            _restore_runtime(model, verification_state)
        result.verification = verification
        diagnostics["verification"] = {
            "mode": (
                "quantization_disabled_routing_dry_run"
                if cpu_routing_only
                else (
                    "quantized_with_independent_routing_dry_run"
                    if plan.quantization_enabled
                    else "quantization_disabled_routing_dry_run"
                )
            ),
            "actual_quantized_execution_attempted": bool(
                plan.quantization_enabled and not cpu_routing_only
            ),
        }
    support["strict_realization"] = bool(verification.succeeded and verification.strict)
    support["routing_dry_run_verified"] = bool(
        verification.succeeded
        and verification.strict
        and verification.output_equivalence
    )
    support["quantized_execution_verified"] = bool(verification.quantized_execution)
    support["fully_supported"] = bool(
        support["capture_complete"]
        and support["replacement_coverage"]
        and verification.succeeded
        and verification.strict
        and not config.allow_fp32_fallback
        and (not config.quantization_enabled or verification.quantized_execution)
    )
    if not support["capture_complete"]:
        support["verdict"] = "capture_failed"
    elif support["fully_supported"]:
        support["verdict"] = "fully_supported"
    else:
        support["verdict"] = "partial_or_unsupported"
    return result


def _iter_tensors(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def _inspection_device_type(
    model: nn.Module, scenarios: list[Scenario], configured: str
) -> str:
    if configured != "auto":
        return torch.device(configured).type
    tensors = list(model.parameters()) + list(model.buffers())
    for scenario in scenarios:
        tensors.extend(_iter_tensors((scenario.args, scenario.kwargs)))
    device_types = {tensor.device.type for tensor in tensors}
    return "cpu" if not device_types or device_types == {"cpu"} else "cuda"


def _validate_capture_device(
    model: nn.Module, scenarios: list[Scenario], configured: str
) -> None:
    """Reject misleading device labels; providers own any device movement."""
    if configured == "auto":
        return
    expected = torch.device(configured)
    tensors = list(model.parameters()) + list(model.buffers())
    for scenario in scenarios:
        tensors.extend(_iter_tensors((scenario.args, scenario.kwargs)))

    def matches(device: torch.device) -> bool:
        return device.type == expected.type and (
            expected.index is None or device.index == expected.index
        )

    mismatches = sorted(
        {str(tensor.device) for tensor in tensors if not matches(tensor.device)}
    )
    if mismatches:
        raise QBenchError(
            f"InspectionConfig.device={configured!r}, but model/scenario tensors are on "
            + ", ".join(mismatches)
            + ". Move them in the trusted provider or use device='auto'."
        )
    if expected.type == "cuda" and not torch.cuda.is_available():
        raise QBenchError("CUDA inspection was requested, but CUDA is unavailable")


def inspect_provider(
    provider, config: InspectionConfig | dict[str, Any] | None = None
) -> InspectionResult:
    model = provider.build_model()
    clone = provider.clone_model(model)
    if clone is model:
        raise QBenchError("ModelProvider.clone_model must return an isolated model")
    result = inspect_model(clone, provider.capture_scenarios(), config)
    from .providers import provider_provenance

    result.diagnostics.setdefault("provenance", {}).update(
        provider_provenance(provider)
    )
    return result
