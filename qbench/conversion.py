"""Strict simulator construction and audited runtime routing."""

from __future__ import annotations

import copy
import io
from collections import Counter, OrderedDict
from collections.abc import Mapping
from contextlib import redirect_stdout
from typing import Any

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from .capture import clone_invocation, module_ownership, schema_name, value_metadata
from .registry import (
    STRUCTURAL_SCHEMAS,
    find_kernel,
    module_semantic_configuration,
    schema_from_route_key,
    schema_route_key,
)
from .runtime import (
    observe_quantization,
    simulation_input_quantization,
    simulation_quantization,
    simulation_route,
    simulator_implementation,
    simulator_implementation_active,
)
from .schemas import (
    QBenchError,
    QuantizationPolicy,
    Scenario,
    SimulationPlan,
    VerificationResult,
    redacted_exception,
)


def _is_qbench_module(module: nn.Module | None) -> bool:
    if module is None:
        return False
    namespace = type(module).__module__
    return (
        namespace == "qbench.ops"
        or namespace.startswith("qbench.ops.")
        or namespace == "qbench.simulator_kernels"
        or namespace.startswith("qbench.simulator_kernels.")
        or namespace == "qbench.ops"
        or namespace.startswith("qbench.ops.")
        or namespace == "qbench.ops"
        or namespace.startswith("qbench.ops.")
    )


def _matches_planned_module(module: nn.Module, decision: str | None) -> bool:
    if (
        not decision
        or not decision.startswith("module:")
        or not _is_qbench_module(module)
    ):
        return False
    expected = decision.removeprefix("module:")
    actual = f"{type(module).__module__}.{type(module).__qualname__}"
    if actual == expected:
        return True
    # The repository historically supports both import roots. They bind the
    # same canonical registry, but Python may still report either class module.
    return type(module).__qualname__ == expected.rsplit(".", 1)[-1]


def _module_has_quantized_weight(module: nn.Module) -> bool:
    """Detect materialized weight-quantization state on an executed module."""
    if not bool(getattr(module, "weight_quantization", False)):
        return False
    q_type = getattr(module, "weight_q_type", getattr(module, "q_type", None))
    return str(q_type).strip().lower() != "fp32" and torch.is_tensor(
        getattr(module, "weight_fp8", None)
    )


_ACTIVATION_BOUNDARY_ATTRIBUTES = {
    "input_quantization": (
        ("input_q_type", "q_type"),
        ("input_chunk_size", "chunk_size"),
    ),
    "output_quantization": (
        ("output_q_type", "q_type"),
        ("output_chunk_size", "chunk_size"),
    ),
}


def _activation_transport_configuration(
    model: nn.Module,
) -> dict[str, dict[str, dict[str, Any]]] | None:
    """Resolve exact per-boundary transport policy from converted modules."""
    boundaries: dict[str, dict[str, dict[str, Any]]] = {}
    for module_path, module in model.named_modules():
        for flag, (
            format_attrs,
            chunk_attrs,
        ) in _ACTIVATION_BOUNDARY_ATTRIBUTES.items():
            if not bool(getattr(module, flag, False)):
                continue
            q_type = next(
                (
                    str(value).strip().lower()
                    for attr in format_attrs
                    if (value := getattr(module, attr, None)) is not None
                ),
                None,
            )
            if q_type == "fp32":
                continue
            if not q_type:
                location = module_path or "<root>"
                raise QBenchError(
                    f"Activation boundary {location}.{flag} has no declared format"
                )
            chunk_size = next(
                (
                    int(value)
                    for attr in chunk_attrs
                    if (value := getattr(module, attr, None)) is not None
                ),
                128,
            )
            mode_attr = "input_mode" if flag == "input_quantization" else "output_mode"
            fallback_attr = "quant_mode" if flag == "input_quantization" else None
            mode = getattr(module, mode_attr, None)
            if mode is None and fallback_attr is not None:
                mode = getattr(module, fallback_attr, None)
            mode = str(mode or "tensor").strip().lower()
            if mode not in {"tensor", "channel", "chunk"}:
                location = module_path or "<root>"
                raise QBenchError(
                    f"Activation boundary {location}.{flag} has unsupported mode {mode!r}"
                )
            if mode == "chunk" and chunk_size < 1:
                location = module_path or "<root>"
                raise QBenchError(
                    f"Activation boundary {location}.{flag} has invalid chunk size"
                )
            boundaries.setdefault(module_path, {})[flag] = {
                "q_type": q_type,
                "mode": mode,
                "chunk_size": int(chunk_size),
            }

    return boundaries or None


def _contains_cuda_tensor(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(value.is_cuda)
    if isinstance(value, Mapping):
        return any(_contains_cuda_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_cuda_tensor(item) for item in value)
    return False


_MISSING = object()


class _EagerActivationTransport:
    """Dynamic-control-flow-safe activation transport for planned modules.

    The workbench's historical transport instruments an FX graph.  Canonical
    simulation cannot specialize an eager branch, so this runtime transmits
    operands at the executed module boundary instead.  It retains no tensor
    values and reports hardware transport separately from support coverage.
    """

    def __init__(self, simulator, model, plan, transport, boundaries):
        from qbench.quantization.activation_transport import (
            ActivationTransport,
        )

        self.simulator = simulator
        self.model = model
        self.plan = plan
        self.transport_mode = transport
        self._transport_type = ActivationTransport
        self._transports: dict[int, Any] = {}
        self.boundaries = dict(boundaries)
        self._transport_runtime = self
        self._handles = []
        self._saved: dict[int, tuple[nn.Module, dict[str, Any]]] = {}
        self.transmission_count = 0
        self.packet_count = 0
        self.decode_reads = 0
        self.encoded_bytes = 0
        self.stage_transmissions: Counter[str] = Counter()

    def _save_and_activate(
        self, module: nn.Module, *, disable: frozenset[str] = frozenset()
    ) -> None:
        existing = self._saved.get(id(module))
        if existing is None:
            saved = {
                "_qbench_activation_transport_active": getattr(
                    module, "_qbench_activation_transport_active", _MISSING
                )
            }
            module._qbench_activation_transport_active = True
            self._saved[id(module)] = (module, saved)
        else:
            saved = existing[1]
        if disable:
            for name in disable:
                if hasattr(module, name) and name not in saved:
                    saved[name] = getattr(module, name)
                    setattr(module, name, False)

    def _transmit_tensor(
        self,
        tensor: torch.Tensor,
        stage_id: str,
        owner_path: str,
        configuration: Mapping[str, Any],
        operand_indices: tuple[int, ...] = (),
    ) -> torch.Tensor:
        if tensor.dtype != torch.float32 or tensor.ndim == 0:
            return tensor
        from qbench.quantization.activation_transport import ActivationPacket
        from qbench.quantization.chunking import count_context_chunks
        from .runtime import quantization_stage, record_quantization

        q_type = str(configuration["q_type"])
        mode = str(configuration["mode"])
        chunk_size = int(configuration["chunk_size"])
        stage = "output" if stage_id.endswith(":output") else "input"

        with (
            simulator_implementation(),
            quantization_stage(
                stage,
                policy_q_type=q_type,
                policy_mode=mode,
                policy_chunk_size=chunk_size,
                operand_indices=list(operand_indices),
            ),
        ):
            if mode == "chunk":
                transport = self._transports.get(chunk_size)
                if transport is None:
                    transport = self._transport_type(
                        mode=self.transport_mode,
                        chunk_size=chunk_size,
                    )
                    self._transports[chunk_size] = transport
                transmitted = transport.transmit_uniform(
                    tensor,
                    q_type,
                    producer_id=stage_id,
                )
                decoded = transport.decode(transmitted)
                chunks = (
                    transmitted.num_chunks
                    if isinstance(transmitted, ActivationPacket)
                    else count_context_chunks(tensor, chunk_size)
                )
                record_quantization(
                    kind="activation_transport_codec",
                    q_type=q_type,
                    mode=mode,
                    shape=tuple(int(dim) for dim in tensor.shape),
                    dtype=str(tensor.dtype),
                    device=str(tensor.device),
                )
            else:
                if mode == "channel" and tensor.dim() < 2:
                    raise QBenchError(
                        "Channel activation quantization requires rank at least two"
                    )
                if tensor.is_cuda:
                    from qbench.ops.quant_base import quantize_tensor

                    decoded, _maximum = quantize_tensor(
                        tensor,
                        q_type=q_type,
                        mode=mode,
                        chunk_size=chunk_size,
                    )
                    event_kind = "activation_transport_cuda_codec"
                else:
                    from qbench.ops.quant_base import calculate_scale
                    from qbench.quantization.quantizer import quantize

                    reduce_dims = (
                        tuple(range(tensor.dim()))
                        if mode == "tensor"
                        else tuple(dim for dim in range(tensor.dim()) if dim != 1)
                    )
                    maximum = (
                        tensor.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-5)
                    )
                    scale = calculate_scale(maximum, q_type)
                    decoded = quantize(tensor / scale, q_type=q_type) * scale
                    event_kind = "activation_transport_cpu_reference"
                transmitted = decoded
                chunks = 1 if mode == "tensor" else int(tensor.shape[1])
                record_quantization(
                    kind=event_kind,
                    q_type=q_type,
                    mode=mode,
                    shape=tuple(int(dim) for dim in tensor.shape),
                    dtype=str(tensor.dtype),
                    device=str(tensor.device),
                )
        self.transmission_count += 1
        self.decode_reads += 1
        self.stage_transmissions[stage_id] += 1
        if isinstance(transmitted, ActivationPacket):
            self.packet_count += 1
            self.encoded_bytes += transmitted.encoded_nbytes
        record_quantization(
            kind="eager_activation_transport",
            stage_id=stage_id,
            module_paths=[owner_path],
            stage=stage,
            policy_q_type=q_type,
            policy_mode=mode,
            policy_chunk_size=chunk_size,
            operand_indices=list(operand_indices),
            q_type=q_type,
            mode=mode,
            transport=self.transport_mode,
            execution_backend=(
                "encoded_cuda"
                if tensor.is_cuda and self.transport_mode == "encoded"
                else "cpu_reference"
            ),
            chunks=int(chunks),
        )
        return decoded

    def _transmit_value(
        self,
        value: Any,
        stage_id: str,
        owner_path: str,
        configuration: Mapping[str, Any],
        memo: dict[int, Any],
        operand_indices: tuple[int, ...] = (),
    ):
        if torch.is_tensor(value):
            identity = id(value)
            if identity not in memo:
                memo[identity] = self._transmit_tensor(
                    value,
                    stage_id,
                    owner_path,
                    configuration,
                    operand_indices,
                )
            return memo[identity]
        if isinstance(value, Mapping):
            items = [
                (
                    key,
                    self._transmit_value(
                        item,
                        stage_id,
                        owner_path,
                        configuration,
                        memo,
                        operand_indices,
                    ),
                )
                for key, item in value.items()
            ]
            try:
                return type(value)(items)
            except (TypeError, ValueError):
                return dict(items)
        if isinstance(value, tuple):
            items = [
                self._transmit_value(
                    item,
                    stage_id,
                    owner_path,
                    configuration,
                    memo,
                    operand_indices,
                )
                for item in value
            ]
            if hasattr(value, "_fields"):
                return type(value)(*items)
            return tuple(items)
        if isinstance(value, list):
            return [
                self._transmit_value(
                    item,
                    stage_id,
                    owner_path,
                    configuration,
                    memo,
                    operand_indices,
                )
                for item in value
            ]
        return value

    def _pre_hook(
        self,
        stage_id: str,
        owner_path: str,
        boundary_path: str,
        configuration: Mapping[str, Any],
    ):
        def transmit(module, args, kwargs):
            # The first-layer rule is an execution property, not a module-list
            # property.  Ownership hooks run before this transport hook and
            # mark the first meaningful semantic route for this invocation.
            # Only that route's public boundary is skipped; child boundaries
            # introduced by a composite implementation remain eligible.
            if (
                boundary_path == owner_path
                and self.simulator._input_quantization_skipped(owner_path)
            ):
                return args, kwargs
            memo: dict[int, Any] = {}
            positional = list(args)
            keywords = dict(kwargs)
            # Quantization applies only to declared activation operands.  In
            # particular, attention masks carry additive infinities or boolean
            # control data and must never be passed through the activation
            # codec.  Other maintained module replacements are unary at their
            # public boundary.
            is_attention = type(module).__qualname__ == ("DecomposedMultiheadAttention")
            positional_limit = 3 if is_attention else 1
            alias_indices: dict[int, list[int]] = {}
            for index in range(min(positional_limit, len(positional))):
                operand = positional[index]
                if torch.is_tensor(operand):
                    alias_indices.setdefault(id(operand), []).append(index)
            operand_names = ("query", "key", "value") if is_attention else ("input",)
            for index, name in enumerate(operand_names):
                operand = keywords.get(name, _MISSING)
                if torch.is_tensor(operand):
                    alias_indices.setdefault(id(operand), []).append(index)
            for index in range(min(positional_limit, len(positional))):
                positional[index] = self._transmit_value(
                    positional[index],
                    stage_id,
                    owner_path,
                    configuration,
                    memo,
                    tuple(
                        dict.fromkeys(alias_indices.get(id(positional[index]), [index]))
                    ),
                )
            for index, name in enumerate(operand_names):
                if name in keywords:
                    keywords[name] = self._transmit_value(
                        keywords[name],
                        stage_id,
                        owner_path,
                        configuration,
                        memo,
                        tuple(
                            dict.fromkeys(
                                alias_indices.get(id(keywords[name]), [index])
                            )
                        ),
                    )
            return tuple(positional), keywords

        return transmit

    def _post_hook(
        self, stage_id: str, owner_path: str, configuration: Mapping[str, Any]
    ):
        def transmit(_module, _args, _kwargs, output):
            return self._transmit_value(output, stage_id, owner_path, configuration, {})

        return transmit

    def _planned_owner(self, boundary_path: str) -> str:
        candidates = [
            path
            for path in self.plan.module_decisions
            if not path or boundary_path == path or boundary_path.startswith(f"{path}.")
        ]
        if not candidates:
            raise QBenchError(
                f"Activation boundary {boundary_path!r} has no planned module owner"
            )
        return max(candidates, key=len)

    def register_hooks(self) -> None:
        modules = dict(self.model.named_modules())
        # Satisfy the adapter guard without enabling operator-owned hardware
        # arithmetic on an otherwise unconfigured root module.
        root_marker = getattr(self.model, "_qbench_eager_transport_active", _MISSING)
        self.model._qbench_eager_transport_active = True
        self._root_marker = root_marker
        for path, flag_configurations in self.boundaries.items():
            module = modules.get(path)
            if module is None:
                raise QBenchError(
                    f"Activation transport cannot find planned module {path!r}"
                )
            owner_path = self._planned_owner(path)
            flags = frozenset(flag_configurations)
            self._save_and_activate(module, disable=flags)
            if "input_quantization" in flag_configurations:
                self._handles.append(
                    module.register_forward_pre_hook(
                        self._pre_hook(
                            f"module:{path}:input",
                            owner_path,
                            path,
                            flag_configurations["input_quantization"],
                        ),
                        with_kwargs=True,
                    )
                )
            if "output_quantization" in flag_configurations:
                self._handles.append(
                    module.register_forward_hook(
                        self._post_hook(
                            f"module:{path}:output",
                            owner_path,
                            flag_configurations["output_quantization"],
                        ),
                        with_kwargs=True,
                    )
                )

    def cleanup(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module, saved in self._saved.values():
            for name, value in saved.items():
                if value is _MISSING:
                    delattr(module, name)
                else:
                    setattr(module, name, value)
        self._saved.clear()
        if hasattr(self, "_root_marker"):
            if self._root_marker is _MISSING:
                delattr(self.model, "_qbench_eager_transport_active")
            else:
                self.model._qbench_eager_transport_active = self._root_marker
            del self._root_marker

    def transport_stats(self) -> dict[str, Any]:
        activation_plan = {
            stage_id: {
                "layer_name": stage_id.removeprefix("module:") or "<root>",
                "kind": "eager_module_boundary",
                "transmissions": int(count),
            }
            for stage_id, count in sorted(self.stage_transmissions.items())
        }
        return {
            "transport": self.transport_mode,
            "runtime": "eager",
            "transmission_count": int(self.transmission_count),
            "packet_count": int(self.packet_count),
            "decode_reads": int(self.decode_reads),
            "encoded_bytes": int(self.encoded_bytes),
            "planner_version": 1,
            "stage_count": sum(len(flags) for flags in self.boundaries.values()),
            "activation_plan": activation_plan,
        }


class _RoutingMode(TorchDispatchMode):
    def __init__(self, simulator: "Simulator", active):
        super().__init__()
        self.simulator = simulator
        self.active = active

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if simulator_implementation_active():
            return func(*args, **kwargs)
        _namespace, schema, _overload = schema_name(func)
        path, module = self.active[-1] if self.active else ("", None)
        base_key = schema_route_key(schema)
        for owner_path, owner_module in reversed(self.active):
            owner_key = f"module:{owner_path}"
            if owner_key not in self.simulator.plan.kernels:
                continue
            decision = self.simulator.plan.module_decisions.get(owner_path)
            if _matches_planned_module(owner_module, decision):
                # Low-level operations inside an active QBench composite are
                # implementation details, including operations owned by child
                # modules introduced during decomposition.
                pass
            else:
                self.simulator._unexpected.append(
                    {
                        "schema": schema,
                        "module_path": owner_path,
                        "reason": "native module bypassed planned replacement",
                    }
                )
            return func(*args, **kwargs)

        if _is_qbench_module(module):
            self.simulator._unexpected.append(
                {
                    "schema": schema,
                    "module_path": path,
                    "reason": "unplanned QBench composite operation",
                }
            )
            return func(*args, **kwargs)

        if (
            base_key in self.simulator.plan.kernels
            and schema in STRUCTURAL_SCHEMAS
            and self.simulator.plan.kernels[base_key].get("classification")
            == "structural"
        ):
            self.simulator._realized[base_key] += 1
            return func(*args, **kwargs)

        metadata = value_metadata(args)
        kw_metadata = value_metadata(kwargs)
        spec = find_kernel(schema, tuple(metadata), dict(kw_metadata))
        key = schema_route_key(schema, spec)
        if key in self.simulator.plan.kernels:
            row = self.simulator.plan.kernels[key]
            if spec is None or spec.name != row.get("name") or spec.handler is None:
                if self.simulator.plan.allow_fp32_fallback:
                    self.simulator._fallbacks[schema] += 1
                self.simulator._unexpected.append(
                    {
                        "schema": schema,
                        "module_path": path,
                        "reason": "planned schema did not resolve to its ready runtime handler",
                    }
                )
                return func(*args, **kwargs)
            self.simulator._realized[key] += 1
            self.simulator._active_schema_routes.append(key)
            skip_input = self.simulator._begin_semantic_route(key, path)
            weight_operand = row.get("weight_operand")
            if weight_operand is not None:
                weight_value = (
                    args[weight_operand]
                    if weight_operand < len(args)
                    else kwargs.get(row.get("weight_argument"))
                )
                self.simulator._quantization_events.append(
                    {
                        "route": key,
                        "module_path": path,
                        "stage": "source_weight",
                        "present": torch.is_tensor(weight_value),
                    }
                )
            if row.get("classification") == "composite":
                self.simulator._record_composite_execution(key, path)
            try:
                with (
                    simulation_route(path),
                    simulation_input_quantization(not skip_input),
                ):
                    return spec.handler(func, args, kwargs)
            finally:
                self.simulator._active_schema_routes.pop()

        if schema in STRUCTURAL_SCHEMAS:
            # Structural operations do not create an FP32 compute island even
            # when introduced as an implementation detail by conversion.
            return func(*args, **kwargs)
        if self.simulator.plan.allow_fp32_fallback:
            self.simulator._fallbacks[schema] += 1
        else:
            self.simulator._unexpected.append(
                {
                    "schema": schema,
                    "module_path": path,
                    "reason": "unplanned native operation",
                }
            )
        return func(*args, **kwargs)


class _RootContainer(nn.Module):
    """Expose the source root as a child so GenericAdapter can replace it."""

    def __init__(self, root: nn.Module):
        super().__init__()
        self.qbench_root = root

    def forward(self, *args, **kwargs):
        return self.qbench_root(*args, **kwargs)


def _validate_plan_rows(plan: SimulationPlan) -> None:
    """Validate runtime-sensitive route metadata before it can drive an audit."""
    valid_classifications = {"quantized", "composite", "structural"}
    plan.validate_policy_routes()
    for route, row in plan.kernels.items():
        if not isinstance(route, str) or not route.startswith(("schema:", "module:")):
            raise QBenchError(f"Invalid simulation route {route!r}")
        if not isinstance(row, Mapping):
            raise QBenchError(f"Simulation route {route!r} must be a mapping")
        if not isinstance(row.get("name"), str) or not row["name"]:
            raise QBenchError(f"Simulation route {route!r} must declare a kernel name")
        if row.get("classification") not in valid_classifications:
            raise QBenchError(
                f"Simulation route {route!r} has an invalid classification"
            )
        for field_name in (
            "ready",
            "handler_quantized",
            "counts_as_quantized",
            "quantizes_weights",
            "activation_policy",
        ):
            if type(row.get(field_name)) is not bool:
                raise QBenchError(
                    f"Simulation route {route!r} {field_name} must be a boolean"
                )
        weight_operand = row.get("weight_operand")
        if weight_operand is not None and (
            isinstance(weight_operand, bool)
            or not isinstance(weight_operand, int)
            or weight_operand < 0
        ):
            raise QBenchError(f"Simulation route {route!r} has invalid weight_operand")
        weight_argument = row.get("weight_argument")
        if weight_argument is not None and (
            not isinstance(weight_argument, str) or not weight_argument
        ):
            raise QBenchError(f"Simulation route {route!r} has invalid weight_argument")
        if not row["ready"]:
            raise QBenchError(f"Simulation route {route!r} is not ready")
        scenario_counts = row.get("scenario_counts")
        if not isinstance(scenario_counts, Mapping) or not scenario_counts:
            raise QBenchError(
                f"Simulation route {route!r} requires non-empty scenario_counts"
            )
        if not all(
            isinstance(name, str)
            and bool(name)
            and not isinstance(count, bool)
            and isinstance(count, int)
            and count > 0
            for name, count in scenario_counts.items()
        ):
            raise QBenchError(f"Simulation route {route!r} has invalid scenario_counts")
        declared_scenarios = set(plan.scenario_names)
        undeclared_scenarios = set(scenario_counts) - declared_scenarios
        if declared_scenarios and undeclared_scenarios:
            raise QBenchError(
                f"Simulation route {route!r} references scenarios not declared "
                "by the plan: " + ", ".join(sorted(undeclared_scenarios))
            )
        source_count = row.get("source_count")
        if (
            isinstance(source_count, bool)
            or not isinstance(source_count, int)
            or source_count != sum(scenario_counts.values())
        ):
            raise QBenchError(
                f"Simulation route {route!r} source_count does not match scenario_counts"
            )
        module_paths = row.get("module_paths")
        module_path_counts = row.get("module_path_counts")
        if module_paths is not None and (
            not isinstance(module_paths, list)
            or not all(isinstance(path, str) for path in module_paths)
            or len(set(module_paths)) != len(module_paths)
        ):
            raise QBenchError(f"Simulation route {route!r} has invalid module_paths")
        if module_path_counts is not None and (
            not isinstance(module_path_counts, Mapping)
            or not all(
                isinstance(path, str)
                and not isinstance(count, bool)
                and isinstance(count, int)
                and count > 0
                for path, count in module_path_counts.items()
            )
            or sum(module_path_counts.values()) != source_count
            or (
                module_paths is not None
                and set(module_path_counts) != set(module_paths)
            )
        ):
            raise QBenchError(
                f"Simulation route {route!r} has invalid module_path_counts"
            )
        for field_name in ("schemas", "module_types", "module_implementations"):
            field_value = row.get(field_name)
            if (
                not isinstance(field_value, (tuple, list))
                or not all(isinstance(item, str) and item for item in field_value)
                or len(set(field_value)) != len(field_value)
            ):
                raise QBenchError(
                    f"Simulation route {route!r} has invalid {field_name}"
                )
        if not isinstance(row.get("policy_overrides"), Mapping):
            raise QBenchError(
                f"Simulation route {route!r} has invalid policy_overrides"
            )
        input_operands = row.get("input_operands")
        if not isinstance(input_operands, Mapping) or not all(
            isinstance(route_name, str)
            and route_name
            and isinstance(indices, (tuple, list))
            and all(
                not isinstance(index, bool) and isinstance(index, int) and index >= 0
                for index in indices
            )
            and len(set(indices)) == len(indices)
            for route_name, indices in input_operands.items()
        ):
            raise QBenchError(f"Simulation route {route!r} has invalid input_operands")
        module_invocations = row.get("module_invocations")
        if not isinstance(module_invocations, Mapping):
            raise QBenchError(
                f"Simulation route {route!r} has invalid module_invocations"
            )
        invocation_scenarios: Counter[str] = Counter()
        invocation_paths: Counter[str] = Counter()
        invocation_keys: set[tuple[str, str, int]] = set()
        for invocation_path, invocations in module_invocations.items():
            if (
                not isinstance(invocation_path, str)
                or not isinstance(invocations, list)
                or not all(
                    isinstance(invocation, Mapping)
                    and isinstance(invocation.get("scenario"), str)
                    and bool(invocation.get("scenario"))
                    and not isinstance(invocation.get("invocation"), bool)
                    and isinstance(invocation.get("invocation"), int)
                    and isinstance(invocation.get("args"), (tuple, list))
                    and isinstance(invocation.get("kwargs"), Mapping)
                    and (
                        "module_configuration" not in invocation
                        or isinstance(invocation["module_configuration"], Mapping)
                    )
                    for invocation in invocations
                )
            ):
                raise QBenchError(
                    f"Simulation route {route!r} has invalid module invocation metadata"
                )
            for invocation in invocations:
                scenario_name = invocation["scenario"]
                invocation_id = invocation["invocation"]
                invocation_scenarios[scenario_name] += 1
                invocation_paths[invocation_path] += 1
                invocation_key = (invocation_path, scenario_name, invocation_id)
                if invocation_key in invocation_keys:
                    raise QBenchError(
                        f"Simulation route {route!r} has duplicate module invocation metadata"
                    )
                invocation_keys.add(invocation_key)
        if module_invocations and dict(invocation_scenarios) != dict(scenario_counts):
            raise QBenchError(
                f"Simulation route {route!r} module invocation scenarios do not "
                "match scenario_counts"
            )
        if (
            module_invocations
            and module_path_counts is not None
            and dict(invocation_paths) != dict(module_path_counts)
        ):
            raise QBenchError(
                f"Simulation route {route!r} module invocation paths do not "
                "match module_path_counts"
            )


def _validate_plan_against_registry(model: nn.Module, plan: SimulationPlan) -> None:
    """Reject tampered route claims that differ from maintained capabilities."""
    from . import registry as maintained

    # Registration is lazy so CPU import/inspection never imports CUDA codecs.
    try:
        import qbench.ops  # noqa: F401
    except Exception as exc:
        raise QBenchError(
            "Simulator module registry unavailable: " + redacted_exception(exc)
        ) from exc

    _validate_plan_rows(plan)
    modules = dict(model.named_modules())
    module_routes: set[str] = set()
    for route, row in plan.kernels.items():
        name = row["name"]
        if route.startswith("schema:"):
            schema = schema_from_route_key(route)
            if name == "structural-v1":
                if (
                    schema not in maintained.STRUCTURAL_SCHEMAS
                    or row["classification"] != "structural"
                    or row["counts_as_quantized"]
                    or row["handler_quantized"]
                    or row["quantizes_weights"]
                    or row["weight_operand"] is not None
                    or row["weight_argument"] is not None
                    or row["activation_policy"]
                    or row.get("conversion") != "structural"
                    or row["schemas"] != [schema]
                    or row["module_types"]
                    or row["module_implementations"]
                    or row["input_operands"]
                    or row["policy_overrides"]
                ):
                    raise QBenchError(
                        f"Structural route {route!r} does not match capability list v1"
                    )
                continue
            candidates = [
                spec
                for spec in maintained.KERNEL_SPECS
                if spec.name == name and schema in spec.schemas
            ]
            if not candidates:
                raise QBenchError(
                    f"Schema route {route!r} does not match maintained kernel {name!r}"
                )
            spec = candidates[0]
            if not spec.ready or spec.handler is None:
                raise QBenchError(
                    f"Schema route {route!r} has no ready maintained runtime handler"
                )
            expected_route = schema_route_key(schema, spec)
            if route != expected_route:
                raise QBenchError(
                    f"Schema route {route!r} must use maintained variant key "
                    f"{expected_route!r}"
                )
        else:
            path = route.removeprefix("module:")
            module_routes.add(path)
            module = modules.get(path)
            if module is None:
                raise QBenchError(f"Planned module path {path!r} does not exist")
            module_runtime = torch.nn.modules.module
            if (
                "forward" in getattr(module, "__dict__", {})
                or getattr(module, "_forward_pre_hooks", None)
                or getattr(module, "_forward_hooks", None)
                or getattr(module_runtime, "_global_forward_pre_hooks", None)
                or getattr(module_runtime, "_global_forward_hooks", None)
            ):
                raise QBenchError(
                    f"Planned module path {path!r} no longer has an untouched forward"
                )
            candidates = [
                spec
                for spec in maintained.KERNEL_SPECS
                if spec.name == name and type(module) in spec.module_types
            ]
            if not candidates:
                raise QBenchError(
                    f"Module route {route!r} does not match maintained kernel {name!r}"
                )
            spec = candidates[0]
            if not spec.ready:
                raise QBenchError(f"Module route {route!r} is not maintained as ready")
            replacement = spec.implementation_for(type(module))
            if replacement is None:
                raise QBenchError(
                    f"Module route {route!r} has no pinned maintained replacement"
                )
            invocation_map = row.get("module_invocations", {})
            invocations = invocation_map.get(path, [])
            if len(invocations) != int(row["source_count"]):
                raise QBenchError(
                    f"Module route {route!r} invocation metadata is incomplete"
                )
            for invocation in invocations:
                captured_configuration = invocation.get("module_configuration")
                if captured_configuration is not None and dict(
                    captured_configuration
                ) != module_semantic_configuration(module):
                    raise QBenchError(
                        f"Module route {route!r} no longer satisfies captured constraints"
                    )
                if not spec.accepts_module(
                    module,
                    tuple(invocation["args"]),
                    dict(invocation["kwargs"]),
                ):
                    raise QBenchError(
                        f"Module route {route!r} no longer satisfies captured constraints"
                    )
            implementation_path = spec.implementation_path_for(type(module))
            expected_decision = f"module:{implementation_path}"
            if plan.module_decisions.get(path) != expected_decision:
                raise QBenchError(
                    f"Module route {route!r} replacement decision is not authoritative"
                )

        expected_fields = {
            "ready": spec.ready,
            "conversion": spec.conversion,
            "classification": spec.classification,
            "counts_as_quantized": spec.counts_as_quantized,
            "handler_quantized": spec.handler_quantized,
            "quantizes_weights": spec.quantizes_weights,
            "weight_operand": spec.weight_operand,
            "weight_argument": spec.weight_argument,
            "activation_policy": spec.activation_policy,
            "schemas": list(spec.schemas),
            "module_types": [
                f"{native.__module__}.{native.__qualname__}"
                for native in spec.module_types
            ],
            "input_operands": {
                route_name: list(indices)
                for route_name, indices in spec.input_operands.items()
            },
            "policy_overrides": spec.policy_overrides,
            "route_variant": spec.route_variant,
        }
        for field_name, expected in expected_fields.items():
            actual = row.get(field_name)
            if field_name in {"schemas", "module_types"}:
                actual = list(actual)
            if actual != expected:
                raise QBenchError(
                    f"Simulation route {route!r} {field_name} disagrees with "
                    f"maintained kernel {name!r}"
                )
        if list(row.get("module_implementations", ())) != list(
            spec.module_implementations
        ):
            raise QBenchError(
                f"Simulation route {route!r} implementation identity disagrees "
                f"with maintained kernel {name!r}"
            )

    if set(plan.module_decisions) != module_routes:
        raise QBenchError(
            "Every module replacement decision must have exactly one module route"
        )


def _guard_simulator_model(model: nn.Module) -> nn.Module:
    """Reject execution of the private converted model outside Simulator.run."""
    if getattr(model, "_qbench_simulator_guard_handle", None) is not None:
        return model

    def require_simulator_run(guarded_model, _args):
        if not bool(getattr(guarded_model, "_qbench_simulator_active", False)):
            raise QBenchError(
                "Direct converted-model execution is disabled; use Simulator.run()"
            )

    model._qbench_simulator_guard_handle = model.register_forward_pre_hook(
        require_simulator_run
    )
    return model


def _convert_modules(
    model: nn.Module,
    *,
    module_paths: tuple[str, ...] = (),
    structural_module_paths: tuple[str, ...] = (),
    expected_module_types: Mapping[str, type] | None = None,
    quantization_enabled: bool = False,
    quantization_policy: QuantizationPolicy | Mapping[str, Any] | None = None,
) -> tuple[nn.Module, bool, list[str]]:
    """Use the maintained adapter for safe module swaps."""
    policy = QuantizationPolicy.coerce(quantization_policy)
    if not module_paths:
        converted = copy.deepcopy(model)
        converted.eval()
        return _guard_simulator_model(converted), True, []
    try:
        from qbench.adapters.generic_adapter import GenericAdapter

        class StrictPlanAdapter(GenericAdapter):
            _enable_timm_decomposition = False
            _target_module_paths_exact = True

            def _prune_noop_layers(self, model, context=""):
                return 0

        wrapped = _RootContainer(copy.deepcopy(model))
        wrapped.eval()
        target_prefixes = [
            "qbench_root" if not path else f"qbench_root.{path}"
            for path in module_paths
        ]
        with redirect_stdout(io.StringIO()):
            adapter_layer_config = {}
            for path, raw_row in policy.layer_config.items():
                row = dict(raw_row)
                route_q_type = row.get(
                    "type", row.get("format", policy.quantization_type)
                )
                # Canonical precedence is input/output override, then the
                # route's format.  The legacy adapter otherwise falls back to
                # its global activation formats here.
                row.setdefault("input_format", route_q_type)
                resolved_output = policy.resolve(path)["output_quantization"]
                row.setdefault("output_quantization", resolved_output)
                row.setdefault("output_format", route_q_type)
                adapter_path = "qbench_root" if not path else f"qbench_root.{path}"
                adapter_layer_config[adapter_path] = row
            adapter = StrictPlanAdapter(
                model_name=type(model).__name__,
                model=wrapped,
                model_source="custom",
                quantization_type=policy.quantization_type,
                quantization_bias=policy.quantization_bias,
                input_quantization=(quantization_enabled and policy.input_quantization),
                weight_quantization=(
                    quantization_enabled and policy.weight_quantization
                ),
                output_quantization=(
                    quantization_enabled and policy.output_quantization
                ),
                quantize_first_layer=policy.quantize_first_layer,
                input_quantization_type=policy.quantization_type,
                output_quantization_type=policy.quantization_type,
                quant_mode=policy.quant_mode,
                chunk_size=policy.chunk_size,
                input_chunk_size=policy.chunk_size,
                weight_mode=policy.weight_mode,
                weight_chunk_size=policy.weight_chunk_size,
                act_mode=policy.act_mode,
                act_chunk_size=policy.act_chunk_size,
                output_mode=policy.output_mode,
                output_chunk_size=policy.output_chunk_size,
                rounding=policy.rounding,
                layer_config=adapter_layer_config,
                fold_layers=False,
                fold_input_norm=False,
                skip_calibration=not (
                    quantization_enabled and policy.weight_quantization
                ),
                enable_fx_quantization=False,
                quantized_ops=["all"],
                target_module_prefixes=target_prefixes,
            )
        converted = adapter.model.qbench_root
        converted_modules = dict(converted.named_modules())
        for path, expected_type in (expected_module_types or {}).items():
            actual = converted_modules.get(path)
            if actual is None or type(actual) is not expected_type:
                actual_name = (
                    "missing"
                    if actual is None
                    else f"{type(actual).__module__}.{type(actual).__qualname__}"
                )
                raise QBenchError(
                    f"Converted module {path!r} is {actual_name}, expected exact "
                    f"{expected_type.__module__}.{expected_type.__qualname__}"
                )
        for module in converted_modules.values():
            # Canonical first-layer handling is execution ordered and applies
            # equally to module and functional routes.  Disable the legacy
            # Conv-only static marker so a later convolution cannot be skipped
            # merely because it was first in the adapter's traversal order.
            if _is_qbench_module(module) and hasattr(module, "is_first_layer"):
                module.is_first_layer = False
                module.quantize_first_layer = True
            if _is_qbench_module(module):
                module._qbench_hardware_arithmetic_enabled = bool(quantization_enabled)
            weight = getattr(module, "weight", None)
            if (
                torch.is_tensor(weight)
                and weight.dim() < 2
                and getattr(module, "weight_mode", None) == "channel"
            ):
                module.weight_mode = "tensor"
        for path in structural_module_paths:
            structural = converted_modules.get(path)
            if structural is None:
                raise QBenchError(
                    f"Converted structural module path {path!r} does not exist"
                )
            # Structural routes must preserve identity and alias behavior.  A
            # following compute kernel owns any activation boundary.
            structural.input_quantization = False
            structural.output_quantization = False
        # GenericAdapter guards its temporary root container. Preserve the
        # activation-boundary invariant on the extracted public simulator root.
        adapter._install_activation_transport_guard(converted)
        _guard_simulator_model(converted)
        converted.eval()
        return converted, True, []
    except Exception as exc:
        fallback = copy.deepcopy(model)
        fallback.eval()
        return (
            _guard_simulator_model(fallback),
            False,
            ["Module conversion unavailable: " + redacted_exception(exc)],
        )


def _rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    return (
        torch.random.get_rng_state(),
        (torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None),
    )


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.random.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def _assert_outputs_close(
    reference: Any,
    simulated: Any,
    path: str = "output",
    *,
    compare_values: bool = True,
    compare_nonfinite: bool = False,
) -> None:
    """Recursively enforce output structure and optional value equivalence."""
    if torch.is_tensor(reference) or torch.is_tensor(simulated):
        if not (torch.is_tensor(reference) and torch.is_tensor(simulated)):
            raise AssertionError(f"{path}: tensor/non-tensor type mismatch")
        if tuple(reference.shape) != tuple(simulated.shape):
            raise AssertionError(
                f"{path}: shape mismatch {tuple(reference.shape)} != {tuple(simulated.shape)}"
            )
        if reference.dtype != simulated.dtype:
            raise AssertionError(
                f"{path}: dtype mismatch {reference.dtype} != {simulated.dtype}"
            )
        if compare_nonfinite and (
            reference.is_floating_point() or reference.is_complex()
        ):
            for label, left, right in (
                ("NaN", torch.isnan(reference), torch.isnan(simulated)),
                (
                    "positive infinity",
                    torch.isposinf(reference),
                    torch.isposinf(simulated),
                ),
                (
                    "negative infinity",
                    torch.isneginf(reference),
                    torch.isneginf(simulated),
                ),
            ):
                if not torch.equal(left, right):
                    raise AssertionError(f"{path}: {label} pattern differs")
        if not compare_values and (
            reference.is_floating_point() or reference.is_complex()
        ):
            return
        if reference.is_floating_point() or reference.is_complex():
            torch.testing.assert_close(reference, simulated, rtol=1e-5, atol=1e-6)
        elif not torch.equal(reference, simulated):
            raise AssertionError(f"{path}: tensor values differ")
        return

    if isinstance(reference, Mapping) or isinstance(simulated, Mapping):
        if not (isinstance(reference, Mapping) and isinstance(simulated, Mapping)):
            raise AssertionError(f"{path}: mapping/non-mapping type mismatch")
        if type(reference) is not type(simulated):
            raise AssertionError(
                f"{path}: mapping type mismatch {type(reference).__name__} != {type(simulated).__name__}"
            )
        if list(reference.keys()) != list(simulated.keys()):
            raise AssertionError(f"{path}: mapping keys differ")
        for key in reference:
            _assert_outputs_close(
                reference[key],
                simulated[key],
                f"{path}[{key!r}]",
                compare_values=compare_values,
                compare_nonfinite=compare_nonfinite,
            )
        return

    if isinstance(reference, (tuple, list)) or isinstance(simulated, (tuple, list)):
        if type(reference) is not type(simulated):
            raise AssertionError(
                f"{path}: sequence type mismatch {type(reference).__name__} != {type(simulated).__name__}"
            )
        if len(reference) != len(simulated):
            raise AssertionError(f"{path}: sequence length mismatch")
        for index, (left, right) in enumerate(zip(reference, simulated)):
            _assert_outputs_close(
                left,
                right,
                f"{path}[{index}]",
                compare_values=compare_values,
                compare_nonfinite=compare_nonfinite,
            )
        return

    if type(reference) is not type(simulated):
        raise AssertionError(
            f"{path}: leaf type mismatch {type(reference).__name__} != {type(simulated).__name__}"
        )
    try:
        equal = reference == simulated
        if torch.is_tensor(equal):
            equal = bool(torch.all(equal).item())
        else:
            equal = bool(equal)
    except Exception:
        equal = False
    if not equal:
        raise AssertionError(f"{path}: non-floating leaf value mismatch")


def _quantization_policy_evidence(
    model: nn.Module,
    plan: SimulationPlan,
    realized: Counter[str],
    events: list[dict[str, Any]],
) -> tuple[list[str], bool]:
    """Validate requested stages and return whether CUDA execution proved them."""

    errors: list[str] = []
    all_required_actual = True
    saw_required_stage = False
    for route, row in plan.kernels.items():
        if row.get("classification") == "structural" or not realized.get(route, 0):
            continue
        route_paths = row.get("module_path_counts")
        if not isinstance(route_paths, Mapping) or not route_paths:
            default_path = (
                route.removeprefix("module:") if route.startswith("module:") else ""
            )
            route_paths = {default_path: int(realized[route])}
        elif sum(route_paths.values()) != int(realized[route]):
            # Non-strict callers may verify a subset of captured scenarios.  A
            # single path can still be checked exactly from realized counts.
            if len(route_paths) == 1:
                route_paths = {next(iter(route_paths)): int(realized[route])}

        for path, expected_invocations in route_paths.items():
            expected_invocations = int(expected_invocations)
            is_activation = bool(row.get("activation_policy"))
            settings = plan.quantization_policy.resolve(
                str(path), activation=is_activation
            )
            if row.get("name") in {"normalization", "batch_norm"} and (
                settings["weight_mode"] == "channel"
            ):
                # Rank-one affine normalization parameters have no channel
                # axis in the maintained codec.  This conditional effective
                # override is declared by the KernelSpec and shared by module
                # and functional routes.
                settings["weight_mode"] = "tensor"

            required: dict[str, tuple[tuple[str, str, int], int]] = {}
            input_enabled = bool(settings["input_quantization"])
            path_events = [
                event
                for event in events
                if event.get("route") == route
                and str(event.get("module_path", "")) == str(path)
            ]
            if row.get("classification") == "composite":
                composite_events = [
                    event for event in path_events if event.get("stage") == "composite"
                ]
                if len(composite_events) < expected_invocations:
                    errors.append(
                        "Composite simulator execution evidence missing for "
                        f"{route} at {path!r}"
                    )
                    all_required_actual = False
            skipped_inputs = sum(
                event.get("stage") == "input_skipped" for event in path_events
            )
            expected_inputs = max(0, expected_invocations - skipped_inputs)
            if input_enabled:
                required["input"] = (
                    (
                        str(settings["input_q_type"]),
                        str(settings["input_mode"]),
                        int(settings["input_chunk_size"]),
                    ),
                    expected_inputs,
                )
            if settings["output_quantization"]:
                required["output"] = (
                    (
                        str(settings["output_q_type"]),
                        str(settings["output_mode"]),
                        int(settings["output_chunk_size"]),
                    ),
                    expected_invocations,
                )
            if settings["weight_quantization"] and bool(row.get("quantizes_weights")):
                if route.startswith("module:"):
                    runtime_module = dict(model.named_modules()).get(str(path))
                    weight_present = bool(
                        runtime_module is not None
                        and any(
                            parameter is not None
                            and name.rsplit(".", 1)[-1] == "weight"
                            for name, parameter in runtime_module.named_parameters()
                        )
                    )
                    expected_weights = expected_invocations if weight_present else 0
                else:
                    expected_weights = sum(
                        event.get("stage") == "source_weight"
                        and bool(event.get("present"))
                        for event in path_events
                    )
                required["weight"] = (
                    (
                        str(settings["q_type"]),
                        str(settings["weight_mode"]),
                        int(settings["weight_chunk_size"]),
                    ),
                    expected_weights,
                )
            operand_route = (
                "module"
                if route.startswith("module:")
                else schema_from_route_key(route)
            )
            required_input_operands = list(
                row.get("input_operands", {}).get(operand_route, (0,))
            )
            for stage, (expected_policy, expected_count) in required.items():
                if expected_count == 0:
                    continue
                saw_required_stage = True
                stage_events = [
                    event for event in path_events if event.get("stage") == stage
                ]
                matching = [
                    event
                    for event in stage_events
                    if (
                        str(event.get("policy_q_type")),
                        str(event.get("policy_mode")),
                        int(event.get("policy_chunk_size") or 0),
                    )
                    == expected_policy
                ]
                if len(matching) < expected_count:
                    detail = "missing" if not stage_events else "policy mismatch"
                    errors.append(
                        f"Quantization evidence {detail} for {route} at {path!r} "
                        f"stage {stage}: expected {expected_policy}"
                    )
                    all_required_actual = False
                    continue
                if stage == "input":
                    for operand_index in required_input_operands:
                        operand_events = [
                            event
                            for event in matching
                            if (
                                event.get("operand_index") == operand_index
                                or operand_index in (event.get("operand_indices") or ())
                                or (
                                    len(required_input_operands) == 1
                                    and "operand_index" not in event
                                    and "operand_indices" not in event
                                )
                            )
                        ]
                        if len(operand_events) < expected_count:
                            errors.append(
                                "Quantization evidence missing for "
                                f"{route} at {path!r} input operand "
                                f"{operand_index}"
                            )
                            all_required_actual = False
                            continue
                actual = [
                    event
                    for event in matching
                    if str(event.get("device", "")).startswith("cuda")
                    or str(event.get("execution_backend", "")).endswith("cuda")
                    or event.get("execution_backend") == "materialized_cuda_codec"
                ]
                if len(actual) < expected_count:
                    all_required_actual = False
                if stage == "input":
                    for operand_index in required_input_operands:
                        actual_operand_events = [
                            event
                            for event in actual
                            if (
                                event.get("operand_index") == operand_index
                                or operand_index in (event.get("operand_indices") or ())
                                or (
                                    len(required_input_operands) == 1
                                    and "operand_index" not in event
                                    and "operand_indices" not in event
                                )
                            )
                        ]
                        if len(actual_operand_events) < expected_count:
                            all_required_actual = False

    return errors, bool(saw_required_stage and all_required_actual)


_SIMULATOR_CONSTRUCTION_KEY = object()


class Simulator:
    def __init__(
        self,
        model: nn.Module,
        plan: SimulationPlan,
        *,
        strict: bool,
        converted: bool,
        build_errors: list[str],
        reference_model: nn.Module | None = None,
        equivalence_simulator: "Simulator | None" = None,
        _construction_key=None,
    ):
        if _construction_key is not _SIMULATOR_CONSTRUCTION_KEY:
            raise QBenchError(
                "Simulator cannot be constructed directly; use build_simulator()"
            )
        self._model = model
        self._model.eval()
        self._reference_model = (
            reference_model if reference_model is not None else copy.deepcopy(model)
        )
        self._reference_model.eval()
        self.plan = plan
        self.strict = strict
        self.converted = converted
        self.build_errors = build_errors
        # Quantized verification is intentionally two-stage.  This companion
        # realizes the identical plan with quantization disabled so conversion
        # correctness is established independently from quantized execution.
        self._equivalence_simulator = equivalence_simulator
        self._realized: Counter[str] = Counter()
        self._fallbacks: Counter[str] = Counter()
        self._quantized_evidence: Counter[str] = Counter()
        self._quantization_events: list[dict[str, Any]] = []
        self._active_schema_routes: list[str] = []
        self._unexpected: list[dict[str, Any]] = []
        self._activation_quantizer = None
        self._semantic_route_started = False
        self._input_skip_paths: set[str] = set()

    def _begin_semantic_route(self, route: str, path: str) -> bool:
        """Apply the first-layer rule to the first executed semantic route."""

        row = self.plan.kernels.get(route, {})
        if row.get("classification") == "structural":
            return False
        if self._semantic_route_started:
            return False
        self._semantic_route_started = True
        if (
            not self.plan.quantization_enabled
            or self.plan.quantization_policy.quantize_first_layer
        ):
            return False
        normalized_path = str(path)
        self._input_skip_paths.add(normalized_path)
        self._quantization_events.append(
            {
                "route": route,
                "module_path": normalized_path,
                "stage": "input_skipped",
                "reason": "quantize_first_layer_disabled",
            }
        )
        return True

    def _input_quantization_skipped(self, path: str) -> bool:
        return str(path) in self._input_skip_paths

    def _record_composite_execution(self, route: str, path: str) -> None:
        self._quantization_events.append(
            {
                "route": route,
                "module_path": str(path),
                "stage": "composite",
                "kind": "qbench_composite_kernel",
            }
        )

    @property
    def model(self):
        raise QBenchError(
            "Direct simulator model access is disabled; use Simulator.run(invocation)"
        )

    def state_dict(self) -> OrderedDict[str, Any]:
        """Return an isolated snapshot of the converted simulator state."""
        state = OrderedDict()
        for key, value in self._model.state_dict().items():
            state[key] = (
                value.detach().clone()
                if torch.is_tensor(value)
                else copy.deepcopy(value)
            )
        return state

    def _ensure_activation_transport(self, args, kwargs) -> None:
        """Install an eager activation path before a quantized run."""
        if not self.plan.quantization_enabled or self._activation_quantizer is not None:
            return
        configuration = _activation_transport_configuration(self._model)
        if configuration is None:
            return
        model_uses_cuda = any(
            tensor.is_cuda
            for tensor in (*self._model.parameters(), *self._model.buffers())
        )
        transport = (
            "encoded"
            if model_uses_cuda
            or _contains_cuda_tensor(args)
            or _contains_cuda_tensor(kwargs)
            else "reference"
        )
        try:
            quantizer = _EagerActivationTransport(
                self,
                self._model,
                self.plan,
                transport,
                configuration,
            )
            quantizer.register_hooks()
        except Exception as exc:
            if "quantizer" in locals():
                quantizer.cleanup()
            raise QBenchError(
                "Eager activation transport installation failed: "
                + redacted_exception(exc)
            ) from exc
        self._activation_quantizer = quantizer

    def activation_transport_stats(self) -> dict[str, Any]:
        """Return JSON-safe producer-transport counters for the active simulator."""
        quantizer = self._activation_quantizer
        runtime = getattr(quantizer, "_transport_runtime", None)
        if runtime is None:
            return {"active": False, "transmission_count": 0}
        return {"active": True, **runtime.transport_stats()}

    def close(self) -> None:
        """Restore module boundary flags and release transport instrumentation."""
        if self._activation_quantizer is not None:
            self._activation_quantizer.cleanup()
            self._activation_quantizer = None
        if self._equivalence_simulator is not None:
            self._equivalence_simulator.close()

    def run(self, invocation: Scenario | tuple[Any, ...] | Any):
        if not isinstance(invocation, Scenario):
            invocation = Scenario(
                "run",
                invocation if isinstance(invocation, tuple) else (invocation,),
                {},
            )
        args, kwargs = clone_invocation(invocation)
        self._realized.clear()
        self._fallbacks.clear()
        self._quantized_evidence.clear()
        self._quantization_events.clear()
        self._unexpected.clear()
        self._semantic_route_started = False
        self._input_skip_paths.clear()
        self._ensure_activation_transport(args, kwargs)

        runtime_modules = dict(self._model.named_modules())

        def module_enter(path, module, active):
            route_key = f"module:{path}"
            if route_key in self.plan.kernels:
                decision = self.plan.module_decisions.get(path)
                if _matches_planned_module(module, decision):
                    self._realized[route_key] += 1
                    self._begin_semantic_route(route_key, path)
                    if (
                        self.plan.kernels[route_key].get("classification")
                        == "composite"
                    ):
                        self._record_composite_execution(route_key, path)
                else:
                    self._unexpected.append(
                        {
                            "schema": route_key,
                            "module_path": path,
                            "reason": "native module bypassed planned replacement",
                        }
                    )
            if not self.plan.quantization_enabled or not _is_qbench_module(module):
                return
            if not _module_has_quantized_weight(module):
                return
            owner_key = quantized_owner(active)
            if owner_key is not None:
                self._quantized_evidence[owner_key] += 1
                weight = getattr(module, "weight_fp8", None)
                self._quantization_events.append(
                    {
                        "route": owner_key,
                        "module_path": owner_key.removeprefix("module:"),
                        "stage": "weight",
                        "policy_q_type": str(
                            getattr(
                                module,
                                "weight_q_type",
                                getattr(module, "q_type", ""),
                            )
                        ),
                        "policy_mode": str(getattr(module, "weight_mode", "channel")),
                        "policy_chunk_size": getattr(module, "weight_chunk_size", None),
                        "device": str(weight.device),
                        "execution_backend": (
                            "materialized_cuda_codec"
                            if weight.is_cuda
                            else "materialized_cpu_reference"
                        ),
                    }
                )

        def quantized_owner(active):
            for owner_path, owner_module in reversed(active):
                owner_key = f"module:{owner_path}"
                if owner_key not in self.plan.kernels:
                    continue
                decision = self.plan.module_decisions.get(owner_path)
                if _matches_planned_module(owner_module, decision):
                    return owner_key
                break
            return None

        with module_ownership(self._model, on_enter=module_enter) as active:

            def quantization_event(metadata):
                route_keys: list[str] = []
                if self._active_schema_routes:
                    route_keys.append(self._active_schema_routes[-1])
                else:
                    if isinstance(metadata, Mapping):
                        for module_path in dict.fromkeys(
                            metadata.get("module_paths", ())
                        ):
                            route_key = f"module:{module_path}"
                            module = runtime_modules.get(module_path)
                            decision = self.plan.module_decisions.get(module_path)
                            if (
                                route_key in self.plan.kernels
                                and module is not None
                                and _matches_planned_module(module, decision)
                            ):
                                route_keys.append(route_key)
                    if not route_keys:
                        owner_key = quantized_owner(active)
                        if owner_key is not None:
                            route_keys.append(owner_key)
                for route_key in dict.fromkeys(route_keys):
                    self._quantized_evidence[route_key] += 1
                    event = {
                        "route": route_key,
                        "module_path": (
                            route_key.removeprefix("module:")
                            if route_key.startswith("module:")
                            else (active[-1][0] if active else "")
                        ),
                    }
                    if isinstance(metadata, Mapping):
                        event.update(dict(metadata))
                        owner_paths = metadata.get("module_paths", ())
                        matching_paths = [
                            str(owner_path)
                            for owner_path in owner_paths
                            if f"module:{owner_path}" == route_key
                        ]
                        if matching_paths:
                            event["module_path"] = matching_paths[0]
                    self._quantization_events.append(event)

            previous_active = bool(
                getattr(self._model, "_qbench_simulator_active", False)
            )
            self._model._qbench_simulator_active = True
            try:
                with (
                    torch.inference_mode(),
                    _RoutingMode(self, active),
                    observe_quantization(quantization_event),
                    simulation_quantization(
                        self.plan.quantization_enabled,
                        self.plan.quantization_policy.to_dict(),
                    ),
                ):
                    output = self._model(*args, **kwargs)
            finally:
                self._model._qbench_simulator_active = previous_active
        if self.strict and self._unexpected:
            first = self._unexpected[0]
            raise QBenchError(
                f"Strict simulation encountered {first['schema']} at {first['module_path']!r}: "
                f"{first['reason']}"
            )
        return output

    def verify(self, scenarios) -> VerificationResult:
        scenarios = list(scenarios)
        aggregate = Counter()
        quantized_aggregate = Counter()
        fallback_aggregate = Counter()
        unexpected: list[dict[str, Any]] = []
        quantization_events: list[dict[str, Any]] = []
        errors: list[str] = []
        comparisons = 0
        value_comparisons = 0
        planned_keys = set(self.plan.kernels)
        equivalence: VerificationResult | None = None

        try:
            _validate_plan_rows(self.plan)
        except QBenchError as exc:
            return VerificationResult(
                attempted=True,
                succeeded=False,
                strict=self.strict,
                planned_operations=len(planned_keys),
                errors=[str(exc)],
            )

        scenario_names = [scenario.name for scenario in scenarios]
        if len(set(scenario_names)) != len(scenario_names):
            errors.append("Verification scenario names must be unique")
        if self.strict and not scenarios:
            errors.append("Strict verification requires at least one scenario")
        declared_scenarios = set(self.plan.scenario_names)
        if not declared_scenarios:
            declared_scenarios = {
                name
                for row in self.plan.kernels.values()
                for name in row["scenario_counts"]
            }
        if (
            self.strict
            and declared_scenarios
            and declared_scenarios != set(scenario_names)
        ):
            errors.append(
                "Verification scenarios do not match the captured plan: expected "
                + ", ".join(sorted(declared_scenarios))
                + "; received "
                + ", ".join(sorted(set(scenario_names)))
            )

        if self.plan.quantization_enabled:
            if self._equivalence_simulator is None:
                if self.strict:
                    errors.append(
                        "Strict quantized verification requires the independent "
                        "quantization-disabled simulator created by build_simulator()"
                    )
            else:
                equivalence = self._equivalence_simulator.verify(scenarios)
                if not equivalence.succeeded or not equivalence.output_equivalence:
                    errors.extend(
                        f"quantization-disabled dry run: {error}"
                        for error in equivalence.errors
                    )
                    if not equivalence.errors:
                        errors.append(
                            "quantization-disabled dry run did not establish output equivalence"
                        )
                    unexpected.extend(
                        {**row, "verification_phase": "quantization_disabled"}
                        for row in equivalence.unexpected_operations
                    )

        if self.strict and self.plan.unresolved_schemas:
            return VerificationResult(
                attempted=True,
                succeeded=False,
                strict=True,
                planned_operations=len(planned_keys),
                errors=[
                    "Unresolved operations: " + ", ".join(self.plan.unresolved_schemas)
                ],
            )
        if self.strict and self.plan.module_decisions and not self.converted:
            return VerificationResult(
                attempted=True,
                succeeded=False,
                strict=True,
                planned_operations=len(planned_keys),
                errors=list(self.build_errors),
            )

        for scenario in scenarios:
            scenario_rng = _rng_state()
            try:
                reference_args, reference_kwargs = clone_invocation(scenario)
                with torch.inference_mode():
                    reference_output = self._reference_model(
                        *reference_args, **reference_kwargs
                    )
                _restore_rng(scenario_rng)
                simulated_output = self.run(scenario)
                aggregate.update(self._realized)
                quantized_aggregate.update(self._quantized_evidence)
                quantization_events.extend(
                    {**event, "scenario": scenario.name}
                    for event in self._quantization_events
                )
                fallback_aggregate.update(self._fallbacks)
                unexpected.extend(self._unexpected)
                for route_key, row in self.plan.kernels.items():
                    scenario_counts = row.get("scenario_counts")
                    expected = int(scenario_counts.get(scenario.name, 0))
                    actual = int(self._realized.get(route_key, 0))
                    if actual != expected:
                        errors.append(
                            f"{scenario.name}: realization count mismatch for {route_key}: "
                            f"expected {expected}, got {actual}"
                        )
                # Establish structure independently so a numerical mismatch
                # does not erase valid structure evidence.
                _assert_outputs_close(
                    reference_output,
                    simulated_output,
                    compare_values=False,
                    compare_nonfinite=self.plan.quantization_enabled,
                )
                comparisons += 1
                if not self.plan.quantization_enabled:
                    _assert_outputs_close(reference_output, simulated_output)
                    value_comparisons += 1
            except Exception as exc:
                unexpected.extend(self._unexpected)
                errors.append(f"{scenario.name}: {redacted_exception(exc)}")
            finally:
                _restore_rng(scenario_rng)

        unrealized = sorted(key for key in planned_keys if aggregate[key] == 0)
        if unrealized:
            errors.append("Unrealized mappings: " + ", ".join(unrealized))
        if self.strict and fallback_aggregate:
            errors.append("FP32 fallbacks: " + ", ".join(sorted(fallback_aggregate)))

        policy_actual = False
        if self.plan.quantization_enabled:
            policy_errors, policy_actual = _quantization_policy_evidence(
                self._model,
                self.plan,
                aggregate,
                quantization_events,
            )
            errors.extend(policy_errors)

        succeeded = not errors and (not self.strict or not unexpected)
        output_structure = comparisons == len(scenarios) and (
            equivalence is None or equivalence.output_structure
        )
        output_equivalence = bool(
            (
                output_structure
                and not self.plan.quantization_enabled
                and value_comparisons == len(scenarios)
            )
            or (
                self.plan.quantization_enabled
                and equivalence is not None
                and equivalence.succeeded
                and equivalence.output_equivalence
            )
        )
        quantized_execution = bool(
            succeeded and self.plan.quantization_enabled and policy_actual
        )
        return VerificationResult(
            attempted=True,
            succeeded=succeeded,
            strict=self.strict,
            quantized_execution=quantized_execution,
            output_structure=output_structure,
            output_equivalence=output_equivalence,
            planned_operations=len(planned_keys),
            realized_operations=sum(aggregate.values()),
            quantized_routes=dict(sorted(quantized_aggregate.items())),
            unexpected_operations=unexpected,
            fp32_fallbacks=dict(sorted(fallback_aggregate.items())),
            errors=errors,
        )


def _create_simulator(
    model: nn.Module,
    plan: SimulationPlan,
    *,
    strict: bool,
    converted: bool,
    build_errors: list[str],
    reference_model: nn.Module | None = None,
    equivalence_simulator: Simulator | None = None,
) -> Simulator:
    """Internal constructor used after the public builder has validated a plan."""
    return Simulator(
        model,
        plan,
        strict=strict,
        converted=converted,
        build_errors=build_errors,
        reference_model=reference_model,
        equivalence_simulator=equivalence_simulator,
        _construction_key=_SIMULATOR_CONSTRUCTION_KEY,
    )


def build_simulator(
    model: nn.Module, plan: SimulationPlan | dict[str, Any], strict: bool = True
) -> Simulator:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    if not isinstance(plan, SimulationPlan):
        plan = SimulationPlan.from_dict(plan)
    if plan.schema_version != 3:
        raise QBenchError(
            "Legacy SimulationPlan objects must be upgraded through "
            "SimulationPlan.from_dict() before conversion"
        )
    if strict and plan.allow_fp32_fallback:
        raise QBenchError("A fallback-enabled plan cannot build a strict simulator")
    if strict and plan.unresolved_schemas:
        raise QBenchError(
            "Strict conversion rejected unresolved operations: "
            + ", ".join(plan.unresolved_schemas)
        )
    _validate_plan_against_registry(model, plan)

    construction_rng = _rng_state()
    try:
        reference = copy.deepcopy(model)
        reference.eval()
        equivalence_simulator = None
        structural_module_paths = tuple(
            route.removeprefix("module:")
            for route, row in plan.kernels.items()
            if route.startswith("module:") and row.get("classification") == "structural"
        )
        source_modules = dict(model.named_modules())
        expected_module_types = {}
        from .registry import KERNEL_SPECS

        for path in plan.module_decisions:
            source_module = source_modules[path]
            row = plan.kernels[f"module:{path}"]
            spec = next(
                candidate
                for candidate in KERNEL_SPECS
                if candidate.name == row["name"]
                and type(source_module) in candidate.module_types
            )
            expected_module_types[path] = spec.implementation_for(type(source_module))
        if plan.quantization_enabled:
            equivalence_plan = copy.deepcopy(plan)
            equivalence_plan.quantization_enabled = False
            equivalence_model, equivalence_converted, equivalence_errors = (
                _convert_modules(
                    model,
                    module_paths=tuple(plan.module_decisions),
                    structural_module_paths=structural_module_paths,
                    expected_module_types=expected_module_types,
                    quantization_enabled=False,
                    quantization_policy=plan.quantization_policy,
                )
            )
            equivalence_simulator = _create_simulator(
                equivalence_model,
                equivalence_plan,
                strict=strict,
                converted=equivalence_converted,
                build_errors=equivalence_errors,
                reference_model=copy.deepcopy(reference),
            )
        converted, did_convert, errors = _convert_modules(
            model,
            module_paths=tuple(plan.module_decisions),
            structural_module_paths=structural_module_paths,
            expected_module_types=expected_module_types,
            quantization_enabled=plan.quantization_enabled,
            quantization_policy=plan.quantization_policy,
        )
    finally:
        _restore_rng(construction_rng)
    return _create_simulator(
        converted,
        plan,
        strict=strict,
        converted=did_convert,
        build_errors=errors,
        reference_model=reference,
        equivalence_simulator=equivalence_simulator,
    )
