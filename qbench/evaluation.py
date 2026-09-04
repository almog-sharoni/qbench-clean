"""Low-overhead dual-forward evaluation."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, asdict
from typing import Any

import torch
import torch.nn as nn

from .capture import clone_invocation
from .schemas import QBenchError, Scenario, strict_json_safe


_FAST_CPU_METRIC_MAX_ELEMENTS = 262_144
_FAST_PENDING_METRIC_MAX_BYTES = 16 * 1024 * 1024


@dataclass
class EvaluationConfig:
    metrics: str = "fast"
    max_batches: int | None = None
    task: str = "generic"
    latency_repetitions: int = 1
    retain_activations: bool = False
    activation_retention_max_elements: int = 4096
    compliance_scan: bool = True

    def __post_init__(self) -> None:
        self.metrics = str(self.metrics).strip().lower()
        self.task = str(self.task).strip().lower().replace("-", "_")
        if self.metrics not in {"fast", "detailed"}:
            raise QBenchError("metrics must be 'fast' or 'detailed'")
        if self.task not in {
            "generic",
            "classification",
            "language_modeling",
            "feature_matching",
        }:
            raise QBenchError(
                "task must be 'generic', 'classification', 'language_modeling', "
                "or 'feature_matching'"
            )
        if self.max_batches is not None:
            if isinstance(self.max_batches, bool) or not isinstance(
                self.max_batches, int
            ):
                raise QBenchError("max_batches must be a non-negative integer or None")
            if self.max_batches < 0:
                raise QBenchError("max_batches must be a non-negative integer or None")
        if (
            isinstance(self.latency_repetitions, bool)
            or not isinstance(self.latency_repetitions, int)
            or self.latency_repetitions < 1
        ):
            raise QBenchError("latency_repetitions must be a positive integer")
        if type(self.retain_activations) is not bool:
            raise QBenchError("retain_activations must be a boolean")
        if (
            isinstance(self.activation_retention_max_elements, bool)
            or not isinstance(self.activation_retention_max_elements, int)
            or self.activation_retention_max_elements < 1
        ):
            raise QBenchError(
                "activation_retention_max_elements must be a positive integer"
            )
        if type(self.compliance_scan) is not bool:
            raise QBenchError("compliance_scan must be a boolean")
        if self.metrics != "detailed" and (
            self.latency_repetitions != 1 or self.retain_activations
        ):
            raise QBenchError(
                "latency repetitions and activation retention require metrics='detailed'"
            )

    @classmethod
    def coerce(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return cls(
                metrics=value.metrics,
                max_batches=value.max_batches,
                task=value.task,
                latency_repetitions=value.latency_repetitions,
                retain_activations=value.retain_activations,
                activation_retention_max_elements=(
                    value.activation_retention_max_elements
                ),
                compliance_scan=value.compliance_scan,
            )
        if isinstance(value, Mapping):
            unknown = sorted(set(value) - set(cls.__dataclass_fields__))
            if unknown:
                raise QBenchError(
                    "Unknown evaluation config fields: " + ", ".join(unknown)
                )
            return cls(**dict(value))
        raise QBenchError("evaluation config must be a mapping or EvaluationConfig")


@dataclass
class EvaluationReport:
    metrics: dict[str, Any]
    batches: int
    reference_forwards: int
    simulator_forwards: int
    timing: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return strict_json_safe(asdict(self))


def _tensors(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, Mapping):
        result = []
        for key in sorted(value, key=str):
            result.extend(_tensors(value[key]))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_tensors(item))
        return result
    return []


def _tensor_pairs(
    reference: Any,
    simulated: Any,
    path: str = "output",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pair tensor leaves only after validating their nested structure."""
    if torch.is_tensor(reference) or torch.is_tensor(simulated):
        if not (torch.is_tensor(reference) and torch.is_tensor(simulated)):
            raise QBenchError(f"{path}: tensor/non-tensor output mismatch")
        return [(reference, simulated)]

    if isinstance(reference, Mapping) or isinstance(simulated, Mapping):
        if not (isinstance(reference, Mapping) and isinstance(simulated, Mapping)):
            raise QBenchError(f"{path}: mapping/non-mapping output mismatch")
        reference_keys = set(reference)
        simulated_keys = set(simulated)
        if reference_keys != simulated_keys:
            raise QBenchError(f"{path}: mapping keys differ")
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for key in sorted(reference_keys, key=repr):
            pairs.extend(
                _tensor_pairs(reference[key], simulated[key], f"{path}[{key!r}]")
            )
        return pairs

    sequence_types = (tuple, list)
    if isinstance(reference, sequence_types) or isinstance(simulated, sequence_types):
        if type(reference) is not type(simulated):
            raise QBenchError(
                f"{path}: sequence type mismatch "
                f"{type(reference).__name__} != {type(simulated).__name__}"
            )
        if len(reference) != len(simulated):
            raise QBenchError(f"{path}: sequence lengths differ")
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, (left, right) in enumerate(zip(reference, simulated)):
            pairs.extend(_tensor_pairs(left, right, f"{path}[{index}]"))
        return pairs

    if type(reference) is not type(simulated):
        raise QBenchError(
            f"{path}: leaf type mismatch "
            f"{type(reference).__name__} != {type(simulated).__name__}"
        )
    return []


def _prepare_metric_pair(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.device | None]:
    """Use CPU reductions for small CUDA outputs to avoid many tiny kernels."""
    if left.shape != right.shape:
        raise QBenchError(
            f"Output shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    if (
        left.is_cuda
        and right.is_cuda
        and left.device == right.device
        and left.numel() + right.numel() <= _FAST_CPU_METRIC_MAX_ELEMENTS
    ):
        # Reductions are deferred across batches. A model may return a reusable
        # buffer, so keep snapshots rather than aliases to its next output.
        return left.detach().clone(), right.detach().clone(), left.device
    return left.detach(), right.detach(), None


def _prepare_metric_target(
    target: Any,
) -> tuple[Any, torch.device | None, int]:
    if not torch.is_tensor(target) or not target.is_cuda:
        return target, None, 0
    copied = torch.empty_like(target, device="cpu", pin_memory=True)
    with torch.cuda.device(target.device):
        copied.copy_(target.detach(), non_blocking=True)
    return copied, target.device, copied.numel() * copied.element_size()


def _rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cuda_is_initialized = getattr(torch.cuda, "is_initialized", None)
    return (
        torch.random.get_rng_state(),
        (
            torch.cuda.get_rng_state_all()
            if callable(cuda_is_initialized) and cuda_is_initialized()
            else None
        ),
    )


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.random.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


class _ActivationStats:
    """Streaming summaries used only by the opt-in detailed bundle."""

    _BOUNDARIES = (-8.0, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0)

    def __init__(self) -> None:
        self.layers: dict[str, dict[str, Any]] = {}

    def observe(self, path: str, output: Any) -> None:
        row = self.layers.setdefault(
            path or "<root>",
            {
                "calls": 0,
                "tensor_outputs": 0,
                "elements": 0,
                "nonfinite": 0,
                "minimum": None,
                "maximum": None,
                "sum": 0.0,
                "sum_squares": 0.0,
                "histogram_counts": [0] * (len(self._BOUNDARIES) + 1),
            },
        )
        row["calls"] += 1
        for tensor in _tensors(output):
            # Detailed hooks execute inside Simulator's routing mode.  The
            # statistics are observability work, not simulated model ops, so
            # keep them outside dispatch auditing and retain no activation.
            with torch._C._DisableTorchDispatch():
                values = tensor.detach()
                if values.is_complex():
                    values = values.abs()
                values = values.float().reshape(-1)
                row["tensor_outputs"] += 1
                row["nonfinite"] += int((~torch.isfinite(values)).sum().item())
                values = values[torch.isfinite(values)]
                count = int(values.numel())
                row["elements"] += count
                if not count:
                    continue
                minimum = float(values.min().item())
                maximum = float(values.max().item())
                row["minimum"] = (
                    minimum if row["minimum"] is None else min(row["minimum"], minimum)
                )
                row["maximum"] = (
                    maximum if row["maximum"] is None else max(row["maximum"], maximum)
                )
                row["sum"] += float(values.sum().item())
                row["sum_squares"] += float(values.square().sum().item())
                boundaries = torch.tensor(
                    self._BOUNDARIES, dtype=values.dtype, device=values.device
                )
                buckets = torch.bucketize(values, boundaries)
                counts = torch.bincount(
                    buckets, minlength=len(self._BOUNDARIES) + 1
                ).to(device="cpu", dtype=torch.int64)
                for index, count_value in enumerate(counts.tolist()):
                    row["histogram_counts"][index] += int(count_value)

    def result(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for path, source in sorted(self.layers.items()):
            row = dict(source)
            count = int(row.pop("elements"))
            total = float(row.pop("sum"))
            square_total = float(row.pop("sum_squares"))
            histogram_counts = row.pop("histogram_counts")
            row.update(
                {
                    "elements": count,
                    "mean": total / count if count else None,
                    "rms": math.sqrt(square_total / count) if count else None,
                    "histogram": {
                        "boundaries": list(self._BOUNDARIES),
                        "counts": histogram_counts,
                    },
                }
            )
            result[path] = row
        return result


def _json_scalar(value: Any) -> Any:
    """Convert one retained tensor scalar to a strict-JSON-safe value."""

    if isinstance(value, complex):
        return {"real": _json_scalar(value.real), "imag": _json_scalar(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


class _ActivationRetention:
    """Explicitly retain a globally capped sample of detailed activations."""

    def __init__(self, enabled: bool, max_elements: int) -> None:
        self.enabled = bool(enabled)
        self.max_elements = int(max_elements)
        self.captured_elements = 0
        self.truncated = False
        self.records: dict[str, dict[str, list[dict[str, Any]]]] = {
            "reference": {},
            "simulator": {},
        }

    def observe(self, side: str, path: str, output: Any) -> None:
        if not self.enabled:
            return
        display_path = path or "<root>"
        for tensor_index, tensor in enumerate(_tensors(output)):
            available = self.max_elements - self.captured_elements
            elements = int(tensor.numel())
            if elements == 0:
                continue
            if available <= 0:
                self.truncated = True
                return
            retained = min(elements, available)
            try:
                with torch._C._DisableTorchDispatch():
                    values = (
                        tensor.detach().reshape(-1)[:retained].to(device="cpu").tolist()
                    )
            except Exception as exc:
                self.records[side].setdefault(display_path, []).append(
                    {
                        "tensor_index": tensor_index,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "capture_error": type(exc).__name__,
                    }
                )
                continue
            if not isinstance(values, list):
                values = [values]
            self.records[side].setdefault(display_path, []).append(
                {
                    "tensor_index": tensor_index,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "total_elements": elements,
                    "retained_elements": retained,
                    "values": [_json_scalar(value) for value in values],
                }
            )
            self.captured_elements += retained
            if retained < elements:
                self.truncated = True

    def result(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "enabled": self.enabled,
            "max_elements": self.max_elements,
            "captured_elements": self.captured_elements,
            "truncated": self.truncated,
        }
        if self.enabled:
            result["activations"] = self.records
        return result


class _ComplianceScanner:
    """Aggregate value-level checks for values emitted by quantizer stages."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._checker = None

    def _row(self, path: str, stage: str, q_type: str) -> dict[str, Any]:
        return self._rows.setdefault(
            (path or "<root>", stage, q_type),
            {
                "checks": 0,
                "tensors": 0,
                "elements": 0,
                "invalid_values": 0,
                "nonfinite_values": 0,
                "errors": {},
            },
        )

    def observe(
        self,
        path: str,
        stage: str,
        q_type: Any,
        tensor: Any,
    ) -> None:
        if not self.enabled or not torch.is_tensor(tensor):
            return
        q_type = str(q_type or "")
        if not q_type or q_type.lower() == "fp32":
            return
        row = self._row(path, stage, q_type)
        row["checks"] += 1
        row["tensors"] += 1
        try:
            with torch._C._DisableTorchDispatch():
                values = tensor.detach().reshape(-1)
                elements = int(values.numel())
                row["elements"] += elements
                if not elements:
                    return
                if values.is_complex():
                    raise TypeError("complex quantized values are not supported")
                finite = torch.isfinite(values)
                nonfinite = int((~finite).sum().item())
                row["nonfinite_values"] += nonfinite
                row["invalid_values"] += nonfinite
                finite_values = values[finite]
                if finite_values.numel():
                    if self._checker is None:
                        from qbench.quantization.quantizer import (
                            check_fp8_compliance,
                        )

                        self._checker = check_fp8_compliance
                    _passed, invalid, _examples = self._checker(
                        finite_values, q_type=q_type
                    )
                    row["invalid_values"] += int(invalid)
        except Exception as exc:
            name = type(exc).__name__
            row["errors"][name] = int(row["errors"].get(name, 0)) + 1

    def observe_module(self, path: str, module: nn.Module) -> None:
        if not self.enabled:
            return
        q_type = getattr(module, "q_type", None)
        multi_inputs = getattr(module, "last_quant_inputs_unscaled", None)
        if isinstance(multi_inputs, (tuple, list)) and multi_inputs:
            raw_formats = getattr(module, "last_quant_input_formats", None)
            formats = raw_formats if isinstance(raw_formats, (tuple, list)) else []
            for index, tensor in enumerate(multi_inputs):
                input_type = formats[index] if index < len(formats) else q_type
                self.observe(path, "input", input_type, tensor)
        else:
            self.observe(
                path,
                "input",
                getattr(module, "input_q_type", q_type),
                getattr(module, "last_quant_input_unscaled", None),
            )
        if getattr(module, "output_quantization", False):
            self.observe(
                path,
                "output",
                getattr(module, "output_q_type", q_type),
                getattr(module, "last_quant_output_unscaled", None),
            )

    def observe_weight(self, path: str, module: nn.Module) -> None:
        if not self.enabled or not getattr(module, "weight_quantization", True):
            return
        self.observe(
            path,
            "weight",
            getattr(module, "weight_q_type", getattr(module, "q_type", None)),
            getattr(module, "weight_fp8", None),
        )

    def result(self) -> dict[str, Any]:
        modules: dict[str, dict[str, Any]] = {}
        checks = tensors = elements = invalid = nonfinite = errors = 0
        for (path, stage, q_type), source in sorted(self._rows.items()):
            row = dict(source)
            row["passed"] = not row["invalid_values"] and not row["errors"]
            modules.setdefault(path, {}).setdefault(stage, {})[q_type] = row
            checks += int(row["checks"])
            tensors += int(row["tensors"])
            elements += int(row["elements"])
            invalid += int(row["invalid_values"])
            nonfinite += int(row["nonfinite_values"])
            errors += sum(int(count) for count in row["errors"].values())
        if not self.enabled:
            status = "disabled"
            passed: bool | None = None
        elif not checks:
            status = "not_assessed"
            passed = None
        elif invalid or errors:
            status = "failed"
            passed = False
        else:
            status = "passed"
            passed = True
        return {
            "enabled": self.enabled,
            "status": status,
            "passed": passed,
            "checks": checks,
            "tensors": tensors,
            "elements": elements,
            "invalid_values": invalid,
            "nonfinite_values": nonfinite,
            "errors": errors,
            "modules": modules,
        }


def _module_inventory(model: nn.Module | None) -> dict[str, Any]:
    if model is None:
        return {
            "total_modules": 0,
            "quantized_modules": 0,
            "compliance": {},
            "modules": [],
        }

    from .registry import OpRegistry

    rows = []
    compliance_counts: dict[str, int] = {}
    for path, module in model.named_modules():
        module_type = type(module)
        namespace = module_type.__module__
        name = OpRegistry.get_registration_name(module_type)
        is_qbench = namespace == "qbench" or namespace.startswith(
            ("qbench.", "qbench.ops.", "qbench.ops.")
        )
        if not is_qbench and name is None:
            continue
        if name is None:
            compliance = "unknown"
            quantized = False
            under_construction = False
        else:
            compliance = OpRegistry.get_compliance_status(name) or "not_declared"
            quantized = OpRegistry.is_quantized(name)
            under_construction = OpRegistry.is_under_construction(name)
        compliance_counts[compliance] = compliance_counts.get(compliance, 0) + 1
        rows.append(
            {
                "path": path,
                "type": f"{namespace}.{module_type.__qualname__}",
                "registry_name": name,
                "quantized": quantized,
                "under_construction": under_construction,
                "compliance": compliance,
            }
        )
    return {
        "total_modules": sum(1 for _ in model.modules()),
        "quantized_modules": sum(bool(row["quantized"]) for row in rows),
        "compliance": dict(sorted(compliance_counts.items())),
        "modules": rows,
    }


class _DetailedCollector:
    _MISSING = object()
    _CAPTURE_ATTRIBUTES = (
        "last_pre_quant_input",
        "last_quant_input",
        "last_quant_input_unscaled",
        "last_quant_inputs_unscaled",
        "last_quant_input_formats",
        "last_quant_input_max",
        "last_quant_input_scale",
        "last_quant_input_scale_packed",
        "last_quant_input_dequant_max",
        "last_natural_output",
        "last_quant_output",
        "last_quant_output_unscaled",
        "last_quant_output_max",
        "last_quant_output_scale",
        "last_quant_output_scale_packed",
        "last_quant_weight",
        "last_quant_weight_scale",
    )

    def __init__(
        self,
        reference: nn.Module,
        simulator: Any,
        config: EvaluationConfig,
    ) -> None:
        self.reference = reference
        candidate = getattr(simulator, "_model", None)
        self.simulator_model = candidate if isinstance(candidate, nn.Module) else None
        self.reference_stats = _ActivationStats()
        self.simulator_stats = _ActivationStats()
        self.retention = _ActivationRetention(
            config.retain_activations,
            config.activation_retention_max_elements,
        )
        self.compliance = _ComplianceScanner(config.compliance_scan)
        self.handles: list[Any] = []
        self.cuda_devices: set[torch.device] = set()
        self.cuda_baselines: dict[torch.device, int] = {}
        self._capture_state: list[tuple[nn.Module, bool, Any, dict[str, Any]]] = []
        self._persistent_module_ids: set[int] = set()
        self._transient_capture_state: dict[
            int, tuple[nn.Module, bool, Any, dict[str, Any]]
        ] = {}

    def _install_model(
        self,
        model: nn.Module | None,
        stats: _ActivationStats,
        side: str,
    ) -> None:
        if model is None:
            return
        for path, module in model.named_modules():
            if any(module.children()):
                continue

            def hook(_module, _args, output, *, path=path, stats=stats, side=side):
                stats.observe(path, output)
                self.retention.observe(side, path, output)
                for tensor in _tensors(output):
                    if tensor.is_cuda:
                        self._track_cuda_device(tensor.device)

            self.handles.append(module.register_forward_hook(hook))

    @staticmethod
    def _is_quantized_module(module: nn.Module) -> bool:
        from .registry import OpRegistry

        name = OpRegistry.get_registration_name(type(module))
        if name is not None:
            return OpRegistry.is_quantized(name)
        namespace = type(module).__module__
        return bool(
            hasattr(module, "capture_activations")
            and hasattr(module, "q_type")
            and (
                namespace == "qbench"
                or namespace.startswith(("qbench.", "qbench.ops.", "qbench.ops."))
            )
        )

    def _clear_capture_values(self, module: nn.Module) -> None:
        for name in self._CAPTURE_ATTRIBUTES:
            if not hasattr(module, name):
                continue
            setattr(module, name, [] if name.endswith("s_unscaled") else None)

    def _capture_module_state(
        self, module: nn.Module
    ) -> tuple[nn.Module, bool, Any, dict[str, Any]]:
        capture_existed = hasattr(module, "capture_activations")
        capture_value = getattr(module, "capture_activations", self._MISSING)
        previous = {
            name: getattr(module, name, self._MISSING)
            for name in self._CAPTURE_ATTRIBUTES
        }
        module.capture_activations = True
        self._clear_capture_values(module)
        return module, capture_existed, capture_value, previous

    def _restore_module_state(
        self,
        state: tuple[nn.Module, bool, Any, dict[str, Any]],
    ) -> None:
        module, capture_existed, capture_value, previous = state
        if capture_existed:
            module.capture_activations = capture_value
        elif hasattr(module, "capture_activations"):
            delattr(module, "capture_activations")
        for name, value in previous.items():
            if value is self._MISSING:
                if hasattr(module, name):
                    delattr(module, name)
            else:
                setattr(module, name, value)

    def _install_compliance(self) -> None:
        if self.simulator_model is None or not self.compliance.enabled:
            return
        for path, module in self.simulator_model.named_modules():
            self._persistent_module_ids.add(id(module))
            if not self._is_quantized_module(module):
                continue
            self._capture_state.append(self._capture_module_state(module))
            self.compliance.observe_weight(path, module)

            def pre_hook(active_module, _args):
                self._clear_capture_values(active_module)

            def post_hook(active_module, _args, _output, *, path=path):
                try:
                    self.compliance.observe_module(path, active_module)
                finally:
                    self._clear_capture_values(active_module)

            self.handles.append(module.register_forward_pre_hook(pre_hook))
            self.handles.append(module.register_forward_hook(post_hook))

        # Runtime handlers instantiate short-lived QBench modules that are not
        # children of the converted clone.  Global hooks make those semantic
        # kernels visible while a simulator implementation context is active;
        # the context-local predicate avoids observing unrelated model work in
        # other threads.
        from torch.nn.modules.module import (
            register_module_forward_hook,
            register_module_forward_pre_hook,
        )

        def transient_pre_hook(module, _args):
            from .runtime import simulator_implementation_active

            if (
                id(module) in self._persistent_module_ids
                or not simulator_implementation_active()
                or not self._is_quantized_module(module)
            ):
                return
            self._transient_capture_state[id(module)] = self._capture_module_state(
                module
            )
            self.compliance.observe_weight("<runtime>", module)

        def transient_post_hook(module, _args, output):
            state = self._transient_capture_state.get(id(module))
            if state is None:
                return
            try:
                from .runtime import simulation_route_path

                route_path = simulation_route_path() or "<runtime>"
                self.compliance.observe_module(route_path, module)
                detail_path = f"{route_path}::{type(module).__name__}"
                self.simulator_stats.observe(detail_path, output)
                self.retention.observe("simulator", detail_path, output)
                for tensor in _tensors(output):
                    if tensor.is_cuda:
                        self._track_cuda_device(tensor.device)
            finally:
                self._transient_capture_state.pop(id(module), None)
                self._restore_module_state(state)

        self.handles.append(register_module_forward_pre_hook(transient_pre_hook))
        self.handles.append(
            register_module_forward_hook(transient_post_hook, always_call=True)
        )

    def install(self) -> None:
        self._install_model(self.reference, self.reference_stats, "reference")
        self._install_model(
            self.simulator_model,
            self.simulator_stats,
            "simulator",
        )
        self._install_compliance()
        for model in (self.reference, self.simulator_model):
            if model is None:
                continue
            for tensor in list(model.parameters()) + list(model.buffers()):
                if tensor.is_cuda:
                    self._track_cuda_device(tensor.device)

    def _track_cuda_device(self, device: torch.device) -> None:
        device = torch.device(device)
        if device in self.cuda_devices:
            return
        torch.cuda.reset_peak_memory_stats(device)
        self.cuda_devices.add(device)
        self.cuda_baselines[device] = int(torch.cuda.memory_allocated(device))

    def observe_invocation(self, scenario: Scenario) -> None:
        for tensor in _tensors((scenario.args, scenario.kwargs)):
            if tensor.is_cuda:
                self._track_cuda_device(tensor.device)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for state in self._capture_state:
            self._restore_module_state(state)
        self._capture_state.clear()
        for state in self._transient_capture_state.values():
            self._restore_module_state(state)
        self._transient_capture_state.clear()
        self._persistent_module_ids.clear()

    def result(self) -> dict[str, Any]:
        cuda_rows = {}
        for device in sorted(self.cuda_devices, key=str):
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            cuda_rows[str(device)] = {
                "baseline_allocated_bytes": self.cuda_baselines[device],
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "peak_additional_allocated_bytes": max(
                    0, peak_allocated - self.cuda_baselines[device]
                ),
            }
        return {
            "activation_statistics": {
                "reference": self.reference_stats.result(),
                "simulator": self.simulator_stats.result(),
            },
            "activation_retention": self.retention.result(),
            "quantized_value_compliance": self.compliance.result(),
            "module_inventory": _module_inventory(self.simulator_model),
            "cuda_peak_memory": {
                "measured": bool(cuda_rows),
                "devices": cuda_rows,
            },
        }


@dataclass(frozen=True)
class _CudaElapsed:
    start: Any
    end: Any
    device: torch.device


def _timed_call(call, invocation_values: Any) -> tuple[Any, float | _CudaElapsed, str]:
    cuda_tensors = [tensor for tensor in _tensors(invocation_values) if tensor.is_cuda]
    if cuda_tensors:
        device = cuda_tensors[0].device
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(device):
            start.record()
            output = call()
            end.record()
        # Resolving every event here serializes the dual-forward loop and makes
        # event/synchronization overhead part of the fast metric bundle.  Keep
        # the events alive and resolve all timings once the batch loop has
        # finished instead.
        return output, _CudaElapsed(start, end, device), "cuda_event"
    start_time = time.perf_counter()
    output = call()
    return output, time.perf_counter() - start_time, "cpu_wall"


def _resolve_timings(*groups: list[float | _CudaElapsed]) -> tuple[float, ...]:
    devices = {
        timing.device
        for group in groups
        for timing in group
        if isinstance(timing, _CudaElapsed)
    }
    for device in devices:
        torch.cuda.synchronize(device)
    return tuple(
        sum(
            (
                float(timing.start.elapsed_time(timing.end)) / 1000.0
                if isinstance(timing, _CudaElapsed)
                else timing
            )
            for timing in group
        )
        for group in groups
    )


def _sum_integer_vectors(rows: list[torch.Tensor], width: int) -> tuple[int, ...]:
    if not rows:
        return (0,) * width
    per_device: dict[torch.device, list[torch.Tensor]] = {}
    for row in rows:
        per_device.setdefault(row.device, []).append(row)
    totals = [
        torch.stack(device_rows).sum(dim=0).to(device="cpu")
        for device_rows in per_device.values()
    ]
    if len(totals) > 1:
        total = torch.stack(totals).sum(dim=0)
    else:
        total = totals[0]
    return tuple(int(value) for value in total.tolist())


def _sum_float_vectors(rows: list[torch.Tensor], width: int) -> tuple[float, ...]:
    if not rows:
        return (0.0,) * width
    per_device: dict[torch.device, list[torch.Tensor]] = {}
    for row in rows:
        per_device.setdefault(row.device, []).append(row)
    totals = [
        torch.stack(device_rows).sum(dim=0).to(device="cpu")
        for device_rows in per_device.values()
    ]
    if len(totals) > 1:
        total = torch.stack(totals).sum(dim=0)
    else:
        total = totals[0]
    return tuple(float(value) for value in total.tolist())


def _batch_cardinality(scenario: Scenario, target: Any) -> int | None:
    if torch.is_tensor(target) and target.ndim:
        return int(target.shape[0])
    if isinstance(target, (tuple, list)):
        return len(target)
    for tensor in _tensors((scenario.args, scenario.kwargs)):
        if tensor.ndim:
            return int(tensor.shape[0])
    return None


def evaluate(
    reference: nn.Module, simulator, provider, config=None
) -> EvaluationReport:
    config = EvaluationConfig.coerce(config)
    sums = CounterMetrics()
    batches = ref_forwards = sim_forwards = 0
    agreement_rows: list[torch.Tensor] = []
    classification_rows: list[torch.Tensor] = []
    language_modeling_rows: list[torch.Tensor] = []
    ref_timings: list[float | _CudaElapsed] = []
    sim_timings: list[float | _CudaElapsed] = []
    coarse_cuda_start = None
    coarse_cuda_elapsed: _CudaElapsed | None = None
    coarse_cuda_device: torch.device | None = None
    evaluated_examples = 0
    cardinality_complete = True
    timing_kinds: set[str] = set()
    training_state = {
        path: module.training for path, module in reference.named_modules()
    }
    initial_rng = _rng_state()
    detailed = (
        _DetailedCollector(reference, simulator, config)
        if config.metrics == "detailed"
        else None
    )
    pending_metric_work: list[tuple[list[tuple[torch.Tensor, torch.Tensor]], Any]] = []
    pending_metric_devices: set[torch.device] = set()
    pending_metric_bytes = 0

    def accumulate_metrics(
        pairs: list[tuple[torch.Tensor, torch.Tensor]], target: Any
    ) -> None:
        left = [pair[0] for pair in pairs]
        right = [pair[1] for pair in pairs]
        for lhs, rhs in pairs:
            sums.add(lhs, rhs)
        if left and right and left[0].ndim >= 2 and left[0].shape == right[0].shape:
            reference_predictions = left[0].argmax(-1)
            simulator_predictions = right[0].argmax(-1)
            agreement_rows.append(
                torch.stack(
                    (
                        (reference_predictions == simulator_predictions).sum(),
                        reference_predictions.new_tensor(
                            reference_predictions.numel(), dtype=torch.int64
                        ),
                    )
                )
            )
        if target is None or not left or not right:
            return
        labels = torch.as_tensor(target, device=left[0].device).long()
        if config.task == "classification":
            labels = labels.reshape(-1)
            ref_logits = left[0]
            sim_logits = right[0]
            if (
                ref_logits.ndim != 2
                or ref_logits.shape != sim_logits.shape
                or ref_logits.shape[0] != labels.numel()
            ):
                raise QBenchError(
                    "Classification logits/target shapes are incompatible"
                )
            reference_predictions = ref_logits.argmax(-1)
            simulator_predictions = sim_logits.argmax(-1)
            k = min(5, ref_logits.shape[-1])
            classification_rows.append(
                torch.stack(
                    (
                        (reference_predictions == labels).sum(),
                        (simulator_predictions == labels).sum(),
                        (ref_logits.topk(k, -1).indices == labels[:, None])
                        .any(-1)
                        .sum(),
                        (sim_logits.topk(k, -1).indices == labels[:, None])
                        .any(-1)
                        .sum(),
                        labels.new_tensor(labels.numel(), dtype=torch.int64),
                    )
                )
            )
        elif config.task == "language_modeling":
            import torch.nn.functional as F

            ref_logits = left[0].reshape(-1, left[0].shape[-1])
            sim_logits = right[0].reshape(-1, right[0].shape[-1])
            labels = labels.reshape(-1)
            if ref_logits.shape[0] != labels.numel():
                raise QBenchError(
                    "Language-model logits/target shapes are incompatible"
                )
            language_modeling_rows.append(
                torch.stack(
                    (
                        F.cross_entropy(
                            ref_logits,
                            labels,
                            ignore_index=-100,
                            reduction="sum",
                        ),
                        F.cross_entropy(
                            sim_logits,
                            labels,
                            ignore_index=-100,
                            reduction="sum",
                        ),
                        (labels != -100).sum().to(dtype=ref_logits.dtype),
                    )
                )
            )

    def flush_pending_metrics(*, synchronize: bool) -> None:
        nonlocal pending_metric_bytes
        metric_work = pending_metric_work
        flattened = [
            tensor
            for pairs, _target in pending_metric_work
            for pair in pairs
            for tensor in pair
        ]
        if flattened:
            first = flattened[0]
            packable = all(
                tensor.device == first.device
                and tensor.dtype == first.dtype
                and tensor.shape == first.shape
                for tensor in flattened[1:]
            )
            if packable:
                with torch.cuda.device(first.device):
                    packed_device = torch.stack(flattened)
                    packed = torch.empty_like(
                        packed_device, device="cpu", pin_memory=True
                    )
                    packed.copy_(packed_device, non_blocking=True)
                iterator = iter(packed)
                metric_work = [
                    (
                        [(next(iterator), next(iterator)) for _pair in pairs],
                        target,
                    )
                    for pairs, target in pending_metric_work
                ]
            else:
                copied_work = []
                for pairs, target in pending_metric_work:
                    copied_pairs = []
                    for left, right in pairs:
                        dtype = torch.promote_types(left.dtype, right.dtype)
                        with torch.cuda.device(left.device):
                            packed_device = torch.stack(
                                (left.to(dtype=dtype), right.to(dtype=dtype))
                            )
                            packed = torch.empty_like(
                                packed_device, device="cpu", pin_memory=True
                            )
                            packed.copy_(packed_device, non_blocking=True)
                        copied_pairs.append((packed[0], packed[1]))
                    copied_work.append((copied_pairs, target))
                metric_work = copied_work
        if synchronize:
            for device in pending_metric_devices:
                torch.cuda.synchronize(device)
        for pairs, target in metric_work:
            accumulate_metrics(pairs, target)
        pending_metric_work.clear()
        pending_metric_devices.clear()
        pending_metric_bytes = 0

    reference.eval()
    try:
        if detailed is not None:
            detailed.install()
        for index, batch in enumerate(provider.evaluation_loader()):
            if config.max_batches is not None and index >= config.max_batches:
                break
            prepared = provider.prepare_evaluation_batch(batch)
            target = None
            if (
                isinstance(prepared, (tuple, list))
                and len(prepared) == 2
                and isinstance(prepared[0], Scenario)
            ):
                scenario, target = prepared
            else:
                scenario = prepared
            if not isinstance(scenario, Scenario):
                raise QBenchError("prepare_evaluation_batch must return Scenario")
            cardinality = _batch_cardinality(scenario, target)
            if cardinality is None:
                cardinality_complete = False
            else:
                evaluated_examples += cardinality
            if detailed is not None:
                detailed.observe_invocation(scenario)
            repetitions = (
                config.latency_repetitions if config.metrics == "detailed" else 1
            )
            args, kwargs = clone_invocation(scenario)
            batch_rng = _rng_state()
            invocation_cuda_tensors = [
                tensor for tensor in _tensors((args, kwargs)) if tensor.is_cuda
            ]
            coarse_cuda = config.metrics == "fast" and bool(invocation_cuda_tensors)
            if coarse_cuda:
                device = invocation_cuda_tensors[0].device
                if coarse_cuda_device is None:
                    coarse_cuda_device = device
                    coarse_cuda_start = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.device(device):
                        coarse_cuda_start.record()
                elif device != coarse_cuda_device:
                    raise QBenchError(
                        "fast CUDA timing requires all batches on one device"
                    )
                with torch.inference_mode():
                    ref_output = reference(*args, **kwargs)
                timing_kind = "cuda_event_coarse"
            else:
                with torch.inference_mode():
                    ref_output, elapsed, timing_kind = _timed_call(
                        lambda: reference(*args, **kwargs), (args, kwargs)
                    )
                ref_timings.append(elapsed)
            timing_kinds.add(timing_kind)
            ref_forwards += 1
            _restore_rng(batch_rng)
            if coarse_cuda:
                sim_output = simulator.run(scenario)
                timing_kind = "cuda_event_coarse"
            else:
                sim_output, elapsed, timing_kind = _timed_call(
                    lambda: simulator.run(scenario),
                    (scenario.args, scenario.kwargs),
                )
                sim_timings.append(elapsed)
            timing_kinds.add(timing_kind)
            sim_forwards += 1
            ref_selected = provider.select_metric_output(ref_output)
            sim_selected = provider.select_metric_output(sim_output)
            prepared_pairs = [
                _prepare_metric_pair(lhs, rhs)
                for lhs, rhs in _tensor_pairs(ref_selected, sim_selected)
            ]
            copied_devices = {
                device for _lhs, _rhs, device in prepared_pairs if device is not None
            }
            pairs = [(lhs, rhs) for lhs, rhs, _device in prepared_pairs]
            if prepared_pairs and all(
                device is not None for _lhs, _rhs, device in prepared_pairs
            ):
                prepared_target, target_device, target_bytes = _prepare_metric_target(
                    target
                )
                if target_device is not None:
                    copied_devices.add(target_device)
                pending_metric_work.append((pairs, prepared_target))
                pending_metric_devices.update(copied_devices)
                pending_metric_bytes += target_bytes + sum(
                    (lhs.numel() * lhs.element_size())
                    + (rhs.numel() * rhs.element_size())
                    for lhs, rhs in pairs
                )
                if pending_metric_bytes >= _FAST_PENDING_METRIC_MAX_BYTES:
                    flush_pending_metrics(synchronize=True)
            else:
                for device in copied_devices:
                    torch.cuda.synchronize(device)
                accumulate_metrics(pairs, target)
            # Additional detailed-mode calls exist only for timing.  Small CUDA
            # outputs have already been copied into private device buffers, so a
            # stateful model cannot mutate the values used for metrics below.
            for _ in range(1, repetitions):
                args, kwargs = clone_invocation(scenario)
                batch_rng = _rng_state()
                with torch.inference_mode():
                    _output, elapsed, timing_kind = _timed_call(
                        lambda: reference(*args, **kwargs), (args, kwargs)
                    )
                ref_timings.append(elapsed)
                timing_kinds.add(timing_kind)
                ref_forwards += 1
                _restore_rng(batch_rng)
                _output, elapsed, timing_kind = _timed_call(
                    lambda: simulator.run(scenario),
                    (scenario.args, scenario.kwargs),
                )
                sim_timings.append(elapsed)
                timing_kinds.add(timing_kind)
                sim_forwards += 1
            batches += 1
        if coarse_cuda_start is not None and coarse_cuda_device is not None:
            coarse_cuda_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(coarse_cuda_device):
                coarse_cuda_end.record()
            coarse_cuda_elapsed = _CudaElapsed(
                coarse_cuda_start, coarse_cuda_end, coarse_cuda_device
            )
    finally:
        if detailed is not None:
            detailed.close()
        for path, module in reference.named_modules():
            if path in training_state:
                module.training = training_state[path]
        _restore_rng(initial_rng)
    ref_time, sim_time, paired_time = _resolve_timings(
        ref_timings,
        sim_timings,
        [coarse_cuda_elapsed] if coarse_cuda_elapsed is not None else [],
    )
    flush_pending_metrics(synchronize=True)
    metrics = sums.result()
    agreements, agreement_total = _sum_integer_vectors(agreement_rows, 2)
    (
        reference_correct1,
        simulator_correct1,
        top5_reference,
        top5_simulator,
        target_total,
    ) = _sum_integer_vectors(classification_rows, 5)
    lm_loss_reference, lm_loss_simulator, lm_tokens = _sum_float_vectors(
        language_modeling_rows, 3
    )
    metrics["prediction_agreement"] = (
        agreements / agreement_total if agreement_total else None
    )
    if config.task == "classification":
        metrics.update(
            {
                "reference_top1": reference_correct1 / target_total
                if target_total
                else None,
                "simulator_top1": simulator_correct1 / target_total
                if target_total
                else None,
                "reference_top5": top5_reference / target_total
                if target_total
                else None,
                "simulator_top5": top5_simulator / target_total
                if target_total
                else None,
            }
        )
    elif config.task == "language_modeling":
        ref_loss = lm_loss_reference / lm_tokens if lm_tokens else None
        sim_loss = lm_loss_simulator / lm_tokens if lm_tokens else None
        metrics.update(
            {
                "reference_loss": ref_loss,
                "simulator_loss": sim_loss,
                "reference_perplexity": math.exp(ref_loss)
                if ref_loss is not None and ref_loss < 700
                else None,
                "simulator_perplexity": math.exp(sim_loss)
                if sim_loss is not None and sim_loss < 700
                else None,
            }
        )
    individual_timing = paired_time <= 0
    repetitions_per_batch = (
        config.latency_repetitions if config.metrics == "detailed" else 1
    )
    timing = {
        "reference_seconds": ref_time if individual_timing else None,
        "simulator_seconds": sim_time if individual_timing else None,
        "paired_seconds": paired_time if paired_time > 0 else None,
        "individual_model_timing": individual_timing,
        "kind": "+".join(sorted(timing_kinds)) or "unmeasured",
        "repetitions_per_batch": repetitions_per_batch,
        "reference_batches_per_second": (
            ref_forwards / ref_time if individual_timing and ref_time > 0 else None
        ),
        "simulator_batches_per_second": (
            sim_forwards / sim_time if individual_timing and sim_time > 0 else None
        ),
        "paired_batches_per_second": (
            batches * repetitions_per_batch / paired_time if paired_time > 0 else None
        ),
        "reference_examples_per_second": (
            evaluated_examples * repetitions_per_batch / ref_time
            if individual_timing and cardinality_complete and ref_time > 0
            else None
        ),
        "simulator_examples_per_second": (
            evaluated_examples * repetitions_per_batch / sim_time
            if individual_timing and cardinality_complete and sim_time > 0
            else None
        ),
        "paired_examples_per_second": (
            evaluated_examples * repetitions_per_batch / paired_time
            if cardinality_complete and paired_time > 0
            else None
        ),
    }
    return EvaluationReport(
        metrics,
        batches,
        ref_forwards,
        sim_forwards,
        timing,
        detailed.result() if detailed is not None else {},
    )


class CounterMetrics:
    def __init__(self):
        self._integer_rows: list[torch.Tensor] = []
        self._float_rows: list[torch.Tensor] = []
        self._cpu_integer_totals = [0, 0, 0]
        self._cpu_float_totals = [0.0] * 6

    def add(self, left, right):
        if left.shape != right.shape:
            raise QBenchError(
                f"Output shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}"
            )
        left = left.detach().float()
        right = right.detach().float()
        if left.device.type == right.device.type == "cpu":
            element_count = left.numel()
            finite_left = torch.isfinite(left)
            finite_right = torch.isfinite(right)
            finite_left_count = int(finite_left.sum())
            finite_right_count = int(finite_right.sum())
            if (
                finite_left_count == element_count
                and finite_right_count == element_count
            ):
                finite_count = element_count
            else:
                finite = finite_left & finite_right
                finite_count = int(finite.sum())
                left = left[finite]
                right = right[finite]
            delta = left - right
            delta_square, left_square, right_square = (
                torch.linalg.vector_norm(
                    torch.stack((delta, left, right)).reshape(3, -1),
                    ord=2,
                    dim=1,
                )
                .square()
                .tolist()
            )
            float_totals = (
                float(delta.abs().sum()),
                float(delta_square),
                float(left_square),
                float((left * right).sum()),
                float(left_square),
                float(right_square),
            )
            integer_totals = (
                finite_count,
                element_count - finite_left_count,
                element_count - finite_right_count,
            )
            for index, value in enumerate(integer_totals):
                self._cpu_integer_totals[index] += value
            for index, value in enumerate(float_totals):
                self._cpu_float_totals[index] += float(value)
            return
        finite_left = torch.isfinite(left)
        finite_right = torch.isfinite(right)
        finite = finite_left & finite_right
        left = left[finite]
        right = right[finite]
        delta = left - right
        self._integer_rows.append(
            torch.stack(
                (
                    finite.sum(),
                    (~finite_left).sum(),
                    (~finite_right).sum(),
                )
            )
        )
        left_square = left.square().sum()
        self._float_rows.append(
            torch.stack(
                (
                    delta.abs().sum(),
                    delta.square().sum(),
                    left_square,
                    (left * right).sum(),
                    left_square,
                    right.square().sum(),
                )
            )
        )

    def result(self):
        integer_totals = _sum_integer_vectors(self._integer_rows, 3)
        count, nonfinite_reference, nonfinite_simulator = (
            gpu + cpu for gpu, cpu in zip(integer_totals, self._cpu_integer_totals)
        )
        (
            abs_sum,
            square_sum,
            signal_square_sum,
            dot,
            left_square,
            right_square,
        ) = (
            gpu + cpu
            for gpu, cpu in zip(
                _sum_float_vectors(self._float_rows, 6), self._cpu_float_totals
            )
        )
        mse = square_sum / count if count else None
        cosine_den = math.sqrt(left_square * right_square)
        return {
            "mae": abs_sum / count if count else None,
            "mse": mse,
            "cosine_similarity": dot / cosine_den if cosine_den else None,
            # JSON has no portable representation for infinity.  ``None`` plus
            # ``perfect_match`` expresses the zero-noise case without emitting
            # non-standard JSON tokens.
            "sqnr_db": 10 * math.log10(signal_square_sum / square_sum)
            if square_sum > 0 and signal_square_sum > 0
            else None,
            "perfect_match": bool(count and square_sum == 0),
            "nonfinite_reference": nonfinite_reference,
            "nonfinite_simulator": nonfinite_simulator,
        }
