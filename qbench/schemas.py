"""Public, JSON-serializable schemas used by QBench inspection."""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 3


class QBenchError(RuntimeError):
    """Base error for configuration, capture, and simulation failures."""


def _safe_json_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    if key is None:
        return "null"
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, int):
        return str(key)
    if isinstance(key, float) and math.isfinite(key):
        return repr(key)
    raise TypeError("mapping keys must be JSON scalar values")


def strict_json_safe(value: Any) -> Any:
    """Recursively normalize a public payload for strict JSON encoding.

    Non-finite floats use stable string spellings because RFC-compliant JSON
    has no NaN or infinity literals. Unsupported objects fail closed rather
    than leaking a value whose later serialization depends on ``repr``.
    """

    if is_dataclass(value) and not isinstance(value, type):
        return strict_json_safe(asdict(value))
    if isinstance(value, Enum):
        return strict_json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _safe_json_key(key)
            if normalized in result:
                raise TypeError("mapping keys collide after normalization")
            result[normalized] = strict_json_safe(item)
        return result
    if isinstance(value, (tuple, list)):
        return [strict_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [strict_json_safe(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, allow_nan=False),
        )
    # NumPy scalar metrics expose ``item`` without requiring NumPy as a public
    # dependency. Tensor-like objects are intentionally not accepted here.
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        return strict_json_safe(value.item())
    raise TypeError(
        f"{type(value).__module__}.{type(value).__qualname__} is not JSON serializable"
    )


def redacted_exception(exc: BaseException) -> str:
    """Return an artifact-safe exception summary without user-controlled text.

    Model, provider, and graph exceptions can interpolate tensor contents (or
    arbitrary private objects) into their message.  Inspection results are
    persistable artifacts, so only the exception class crosses that boundary.
    """

    return f"{type(exc).__name__}: details redacted"


_FORMAT_PATTERN = re.compile(
    r"^(?P<prefix>u?fp)(?P<bits>\d+)_e(?P<exp>\d+)m(?P<mant>\d+)$"
)
_QUANTIZATION_MODES = frozenset({"tensor", "chunk", "channel"})


def _validate_format(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QBenchError(f"{field_name} must be a supported format string")
    normalized = value.strip().lower()
    match = _FORMAT_PATTERN.fullmatch(normalized)
    if match is None:
        raise QBenchError(f"{field_name} must use fpN_eXmY or ufpN_eXmY syntax")
    bits = int(match.group("bits"))
    exponent = int(match.group("exp"))
    mantissa = int(match.group("mant"))
    expected = exponent + mantissa + (0 if match.group("prefix") == "ufp" else 1)
    if bits != expected or not 2 <= bits <= 16 or exponent < 1:
        raise QBenchError(f"{field_name} declares an inconsistent or unsupported width")
    return normalized


def _validate_mode(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or value.strip().lower() not in _QUANTIZATION_MODES:
        raise QBenchError(f"{field_name} must be tensor, chunk, or channel")
    return value.strip().lower()


def _validate_chunk_size(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise QBenchError(f"{field_name} must be a positive integer")
    return value


@dataclass
class QuantizationPolicy:
    """Validated, serializable configuration for maintained simulator kernels."""

    quantization_type: str = "fp8_e4m3"
    quantization_bias: int | None = None
    input_quantization: bool = True
    weight_quantization: bool = True
    output_quantization: bool = False
    quantize_first_layer: bool = False
    quant_mode: str = "tensor"
    chunk_size: int = 128
    weight_mode: str = "channel"
    weight_chunk_size: int = 128
    act_mode: str = "tensor"
    act_chunk_size: int = 128
    output_mode: str = "tensor"
    output_chunk_size: int = 128
    rounding: str = "nearest"
    layer_config: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.quantization_type = _validate_format(
            self.quantization_type, field_name="quantization_type"
        )
        if self.quantization_bias is not None:
            raise QBenchError(
                "quantization_bias is not implemented by the maintained codec; use None"
            )
        for name in (
            "input_quantization",
            "weight_quantization",
            "output_quantization",
            "quantize_first_layer",
        ):
            if type(getattr(self, name)) is not bool:
                raise QBenchError(f"{name} must be a boolean")
        for name in ("quant_mode", "weight_mode", "act_mode", "output_mode"):
            setattr(self, name, _validate_mode(getattr(self, name), field_name=name))
        for name in (
            "chunk_size",
            "weight_chunk_size",
            "act_chunk_size",
            "output_chunk_size",
        ):
            setattr(
                self,
                name,
                _validate_chunk_size(getattr(self, name), field_name=name),
            )
        for mode_name, chunk_name in (
            ("quant_mode", "chunk_size"),
            ("weight_mode", "weight_chunk_size"),
            ("act_mode", "act_chunk_size"),
            ("output_mode", "output_chunk_size"),
        ):
            if getattr(self, mode_name) == "chunk" and getattr(self, chunk_name) != 128:
                raise QBenchError(f"{chunk_name} must be 128 when {mode_name}='chunk'")
        if self.rounding != "nearest":
            raise QBenchError("rounding currently supports only 'nearest'")
        if not isinstance(self.layer_config, Mapping):
            raise QBenchError("layer_config must be a mapping")
        allowed = {
            "type",
            "format",
            "bias",
            "input_format",
            "mode",
            "chunk_size",
            "weight_mode",
            "weight_chunk_size",
            "act_mode",
            "act_chunk_size",
            "output_quantization",
            "output_format",
            "output_mode",
            "output_chunk_size",
            "rounding",
        }
        normalized_layers: dict[str, dict[str, Any]] = {}
        for path, raw in self.layer_config.items():
            if not isinstance(path, str) or not path:
                raise QBenchError("layer_config paths must be non-empty strings")
            if not isinstance(raw, Mapping):
                raise QBenchError(f"layer_config[{path!r}] must be a mapping")
            unknown = sorted(set(raw) - allowed)
            if unknown:
                raise QBenchError(
                    f"layer_config[{path!r}] has unknown fields: {', '.join(unknown)}"
                )
            row = copy.deepcopy(dict(raw))
            for field_name in ("type", "format", "input_format", "output_format"):
                if field_name in row:
                    row[field_name] = _validate_format(
                        row[field_name],
                        field_name=f"layer_config[{path!r}].{field_name}",
                    )
            for field_name in ("mode", "weight_mode", "act_mode", "output_mode"):
                if field_name in row:
                    row[field_name] = _validate_mode(
                        row[field_name],
                        field_name=f"layer_config[{path!r}].{field_name}",
                    )
            for field_name in (
                "chunk_size",
                "weight_chunk_size",
                "act_chunk_size",
                "output_chunk_size",
            ):
                if field_name in row:
                    row[field_name] = _validate_chunk_size(
                        row[field_name],
                        field_name=f"layer_config[{path!r}].{field_name}",
                    )
            if (
                "output_quantization" in row
                and type(row["output_quantization"]) is not bool
            ):
                raise QBenchError(
                    f"layer_config[{path!r}].output_quantization must be a boolean"
                )
            if "bias" in row:
                raise QBenchError(
                    f"layer_config[{path!r}].bias is not implemented by the "
                    "maintained codec"
                )
            if "rounding" in row and row["rounding"] != "nearest":
                raise QBenchError(
                    f"layer_config[{path!r}].rounding supports only 'nearest'"
                )
            for mode_name, chunk_name, global_mode, global_chunk in (
                ("mode", "chunk_size", self.quant_mode, self.chunk_size),
                (
                    "weight_mode",
                    "weight_chunk_size",
                    self.weight_mode,
                    self.weight_chunk_size,
                ),
                ("act_mode", "act_chunk_size", self.act_mode, self.act_chunk_size),
                (
                    "output_mode",
                    "output_chunk_size",
                    self.output_mode,
                    self.output_chunk_size,
                ),
            ):
                effective_mode = row.get(mode_name, global_mode)
                effective_chunk = row.get(chunk_name, global_chunk)
                if effective_mode == "chunk" and effective_chunk != 128:
                    raise QBenchError(
                        f"layer_config[{path!r}].{chunk_name} must be 128 "
                        f"when {mode_name}='chunk'"
                    )
            normalized_layers[path] = row
        self.layer_config = normalized_layers

    @classmethod
    def coerce(
        cls, value: "QuantizationPolicy | Mapping[str, Any] | None"
    ) -> "QuantizationPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return cls(**asdict(value))
        if isinstance(value, Mapping):
            unknown = sorted(set(value) - set(cls.__dataclass_fields__))
            if unknown:
                raise QBenchError(
                    f"Unknown quantization policy fields: {', '.join(unknown)}"
                )
            return cls(**copy.deepcopy(dict(value)))
        raise QBenchError("quantization_policy must be a mapping or QuantizationPolicy")

    def to_dict(self) -> dict[str, Any]:
        return strict_json_safe(asdict(self))

    def resolve(self, path: str, *, activation: bool = False) -> dict[str, Any]:
        """Resolve one route using the canonical global/per-layer precedence."""

        layer = self.layer_config.get(path, {})
        q_type = layer.get("type", layer.get("format", self.quantization_type))
        input_mode = (
            layer.get("act_mode", self.act_mode)
            if activation
            else layer.get("mode", self.quant_mode)
        )
        input_chunk_size = (
            layer.get("act_chunk_size", self.act_chunk_size)
            if activation
            else layer.get("chunk_size", self.chunk_size)
        )
        if "output_quantization" in layer:
            output_quantization = layer["output_quantization"]
        elif any(
            name in layer
            for name in ("output_format", "output_mode", "output_chunk_size")
        ):
            output_quantization = True
        else:
            output_quantization = self.output_quantization
        return {
            "q_type": q_type,
            "input_q_type": layer.get("input_format", q_type),
            "input_quantization": self.input_quantization,
            "input_mode": input_mode,
            "input_chunk_size": input_chunk_size,
            "weight_quantization": self.weight_quantization,
            "weight_mode": layer.get("weight_mode", self.weight_mode),
            "weight_chunk_size": layer.get("weight_chunk_size", self.weight_chunk_size),
            "act_mode": layer.get("act_mode", self.act_mode),
            "act_chunk_size": layer.get("act_chunk_size", self.act_chunk_size),
            "output_q_type": layer.get("output_format", q_type),
            "output_quantization": output_quantization,
            "output_mode": layer.get("output_mode", self.output_mode),
            "output_chunk_size": layer.get("output_chunk_size", self.output_chunk_size),
            "rounding": layer.get("rounding", self.rounding),
        }


@dataclass(frozen=True)
class Scenario:
    """One named eager invocation.

    ``args`` and ``kwargs`` are deliberately not serialized into artifacts: they
    can contain private model inputs.  Only metadata observed by the dispatcher
    is written to the operation ledger.
    """

    name: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Scenario.name must be non-empty")
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", dict(self.kwargs))


@dataclass
class InspectionConfig:
    allow_fp32_fallback: bool = False
    verify: bool = True
    capture_callsites: bool = True
    enable_fx: bool = True
    enable_export: bool = True
    quantization_enabled: bool = False
    device: str = "cpu"
    conformance_directory: str | None = None
    quantization_policy: QuantizationPolicy = field(default_factory=QuantizationPolicy)

    def __post_init__(self) -> None:
        self.quantization_policy = QuantizationPolicy.coerce(self.quantization_policy)
        for name in (
            "allow_fp32_fallback",
            "verify",
            "capture_callsites",
            "enable_fx",
            "enable_export",
            "quantization_enabled",
        ):
            if type(getattr(self, name)) is not bool:
                raise QBenchError(f"{name} must be a boolean")
        if not isinstance(self.device, str) or not self.device.strip():
            raise QBenchError("device must be 'auto' or a valid torch device string")
        self.device = self.device.strip().lower()
        if self.device != "auto":
            try:
                import torch

                parsed = torch.device(self.device)
            except (RuntimeError, TypeError) as exc:
                raise QBenchError(
                    "device must be 'auto' or a valid torch device string"
                ) from exc
            if parsed.type not in {"cpu", "cuda"}:
                raise QBenchError("inspection supports only CPU or CUDA devices")
        if self.conformance_directory is not None:
            if (
                not isinstance(self.conformance_directory, str)
                or not self.conformance_directory.strip()
            ):
                raise QBenchError("conformance_directory must be a non-empty path")
            self.conformance_directory = self.conformance_directory.strip()

    @classmethod
    def coerce(
        cls, value: "InspectionConfig | Mapping[str, Any] | None"
    ) -> "InspectionConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return cls(
                **{name: getattr(value, name) for name in cls.__dataclass_fields__}
            )
        if isinstance(value, Mapping):
            fields = cls.__dataclass_fields__
            unknown = sorted(set(value) - set(fields))
            if unknown:
                raise QBenchError(
                    f"Unknown inspection config fields: {', '.join(unknown)}"
                )
            return cls(**dict(value))
        raise QBenchError("config must be InspectionConfig, a mapping, or None")


@dataclass
class OperationRecord:
    sequence: int
    scenario: str
    namespace: str
    schema: str
    overload: str
    module_path: str
    module_type: str | None
    arguments: Any
    output: Any
    module_aliases: list[str] = field(default_factory=list)
    module_stack: list[dict[str, Any]] = field(default_factory=list)
    module_arguments: Any = None
    callsite: str | None = None
    classification: str = "unresolved"
    kernel: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return strict_json_safe(asdict(self))


@dataclass
class VerificationResult:
    attempted: bool = False
    succeeded: bool = False
    strict: bool = True
    quantized_execution: bool = False
    output_structure: bool = False
    output_equivalence: bool = False
    planned_operations: int = 0
    realized_operations: int = 0
    quantized_routes: dict[str, int] = field(default_factory=dict)
    unexpected_operations: list[dict[str, Any]] = field(default_factory=list)
    fp32_fallbacks: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return strict_json_safe(asdict(self))


@dataclass
class SimulationPlan:
    schema_version: int = SCHEMA_VERSION
    kernels: dict[str, dict[str, Any]] = field(default_factory=dict)
    module_decisions: dict[str, str] = field(default_factory=dict)
    unresolved_schemas: list[str] = field(default_factory=list)
    allow_fp32_fallback: bool = False
    quantization_enabled: bool = False
    quantization_policy: QuantizationPolicy = field(default_factory=QuantizationPolicy)
    scenario_names: list[str] = field(default_factory=list)
    scenarios: list[Scenario] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version not in {1, 2, SCHEMA_VERSION}
        ):
            raise QBenchError(f"Unsupported plan version {self.schema_version!r}")
        if type(self.allow_fp32_fallback) is not bool:
            raise QBenchError("allow_fp32_fallback must be a boolean")
        if type(self.quantization_enabled) is not bool:
            raise QBenchError("quantization_enabled must be a boolean")
        self.quantization_policy = QuantizationPolicy.coerce(self.quantization_policy)
        attached_names = [scenario.name for scenario in self.scenarios]
        if not self.scenario_names:
            if attached_names:
                self.scenario_names = attached_names
            else:
                inferred_names: list[str] = []
                seen_names: set[str] = set()
                for row in self.kernels.values():
                    counts = (
                        row.get("scenario_counts") if isinstance(row, Mapping) else None
                    )
                    if not isinstance(counts, Mapping):
                        continue
                    for name in counts:
                        if isinstance(name, str) and name and name not in seen_names:
                            seen_names.add(name)
                            inferred_names.append(name)
                self.scenario_names = inferred_names
        if (
            not isinstance(self.scenario_names, list)
            or not all(isinstance(name, str) and name for name in self.scenario_names)
            or len(set(self.scenario_names)) != len(self.scenario_names)
        ):
            raise QBenchError("simulation plan scenario_names must be unique strings")
        if attached_names and self.scenario_names != attached_names:
            raise QBenchError(
                "simulation plan scenario_names do not match attached scenarios"
            )

    @property
    def strict_ready(self) -> bool:
        return not self.unresolved_schemas and not self.allow_fp32_fallback

    def validate_policy_routes(self) -> None:
        """Reject layer overrides that were not exercised by the captured plan."""

        valid_paths = set(self.module_decisions)
        for row in self.kernels.values():
            module_paths = row.get("module_paths", ())
            if isinstance(module_paths, (tuple, list)):
                valid_paths.update(
                    path for path in module_paths if isinstance(path, str) and path
                )
        unused = sorted(set(self.quantization_policy.layer_config) - valid_paths)
        if unused:
            raise QBenchError(
                "Quantization layer_config paths were not exercised by the "
                "captured plan: " + ", ".join(unused)
            )

    def to_dict(self) -> dict[str, Any]:
        return strict_json_safe(
            {
                "schema_version": self.schema_version,
                "kernels": copy.deepcopy(self.kernels),
                "module_decisions": dict(self.module_decisions),
                "unresolved_schemas": list(self.unresolved_schemas),
                "allow_fp32_fallback": self.allow_fp32_fallback,
                "quantization_enabled": self.quantization_enabled,
                "quantization_policy": self.quantization_policy.to_dict(),
                "scenario_names": list(self.scenario_names),
            }
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SimulationPlan":
        if not isinstance(value, Mapping):
            raise QBenchError("simulation plan must be a mapping")
        raw_version = value.get("schema_version", value.get("version", 1))
        if isinstance(raw_version, bool) or not isinstance(raw_version, int):
            raise QBenchError("simulation plan version must be an integer")
        version = raw_version
        if version not in {1, 2, SCHEMA_VERSION}:
            raise QBenchError(f"Unsupported plan version {version}")
        kernels = value.get("kernels", {})
        decisions = value.get("module_decisions", value.get("decisions", {}))
        unresolved = value.get("unresolved_schemas", [])
        scenario_names = value.get("scenario_names", [])
        if not isinstance(kernels, Mapping):
            raise QBenchError("simulation plan kernels must be a mapping")
        if not isinstance(decisions, Mapping):
            raise QBenchError("simulation plan module_decisions must be a mapping")
        if isinstance(unresolved, (str, bytes)) or not isinstance(unresolved, list):
            raise QBenchError("simulation plan unresolved_schemas must be a list")
        if (
            isinstance(scenario_names, (str, bytes))
            or not isinstance(scenario_names, list)
            or not all(isinstance(name, str) and name for name in scenario_names)
            or len(set(scenario_names)) != len(scenario_names)
        ):
            raise QBenchError("simulation plan scenario_names must be unique strings")
        if not all(
            isinstance(key, str) and isinstance(row, Mapping)
            for key, row in kernels.items()
        ):
            raise QBenchError("simulation plan kernel entries must be mappings")
        if not all(
            isinstance(key, str) and isinstance(decision, str)
            for key, decision in decisions.items()
        ):
            raise QBenchError("simulation plan module decisions must be strings")
        if not all(isinstance(schema, str) for schema in unresolved):
            raise QBenchError("simulation plan unresolved schemas must be strings")
        normalized_kernels = copy.deepcopy(dict(kernels))
        if version == SCHEMA_VERSION:
            valid_classifications = {"quantized", "composite", "structural"}
            for route, row in normalized_kernels.items():
                if not route.startswith(("schema:", "module:")):
                    raise QBenchError(
                        f"simulation plan kernel route {route!r} is invalid"
                    )
                if not isinstance(row.get("name"), str) or not row["name"]:
                    raise QBenchError(f"kernel {route!r} must declare a name")
                if row.get("classification") not in valid_classifications:
                    raise QBenchError(f"kernel {route!r} has an invalid classification")
                for field_name in (
                    "ready",
                    "handler_quantized",
                    "counts_as_quantized",
                    "quantizes_weights",
                    "activation_policy",
                ):
                    if type(row.get(field_name)) is not bool:
                        raise QBenchError(
                            f"kernel {route!r} {field_name} must be a boolean"
                        )
                weight_operand = row.get("weight_operand")
                if weight_operand is not None and (
                    isinstance(weight_operand, bool)
                    or not isinstance(weight_operand, int)
                    or weight_operand < 0
                ):
                    raise QBenchError(
                        f"kernel {route!r} weight_operand must be a non-negative integer"
                    )
                weight_argument = row.get("weight_argument")
                if weight_argument is not None and (
                    not isinstance(weight_argument, str) or not weight_argument
                ):
                    raise QBenchError(
                        f"kernel {route!r} weight_argument must be a string"
                    )
                source_count = row.get("source_count")
                if (
                    isinstance(source_count, bool)
                    or not isinstance(source_count, int)
                    or source_count < 1
                ):
                    raise QBenchError(
                        f"kernel {route!r} source_count must be a positive integer"
                    )
                scenario_counts = row.get("scenario_counts")
                if not isinstance(scenario_counts, Mapping) or not scenario_counts:
                    raise QBenchError(
                        f"kernel {route!r} scenario_counts must be a non-empty mapping"
                    )
                if not all(
                    isinstance(name, str)
                    and bool(name)
                    and not isinstance(count, bool)
                    and isinstance(count, int)
                    and count > 0
                    for name, count in scenario_counts.items()
                ):
                    raise QBenchError(
                        f"kernel {route!r} scenario_counts must contain positive integer counts"
                    )
                if sum(scenario_counts.values()) != source_count:
                    raise QBenchError(
                        f"kernel {route!r} source_count does not match scenario_counts"
                    )
                for field_name in (
                    "schemas",
                    "module_types",
                    "module_implementations",
                ):
                    field_value = row.get(field_name)
                    if (
                        not isinstance(field_value, (tuple, list))
                        or not all(
                            isinstance(item, str) and item for item in field_value
                        )
                        or len(set(field_value)) != len(field_value)
                    ):
                        raise QBenchError(
                            f"kernel {route!r} {field_name} must be a unique string list"
                        )
                if not isinstance(row.get("policy_overrides"), Mapping):
                    raise QBenchError(
                        f"kernel {route!r} policy_overrides must be a mapping"
                    )
                input_operands = row.get("input_operands")
                if not isinstance(input_operands, Mapping):
                    raise QBenchError(
                        f"kernel {route!r} input_operands must be a mapping"
                    )
                if not isinstance(row.get("module_invocations"), Mapping):
                    raise QBenchError(
                        f"kernel {route!r} module_invocations must be a mapping"
                    )
                module_paths = row.get("module_paths")
                if module_paths is not None and (
                    not isinstance(module_paths, list)
                    or not all(isinstance(path, str) for path in module_paths)
                    or len(set(module_paths)) != len(module_paths)
                ):
                    raise QBenchError(
                        f"kernel {route!r} module_paths must be a unique string list"
                    )
                module_path_counts = row.get("module_path_counts")
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
                        f"kernel {route!r} module_path_counts must match source_count"
                    )
        allow_fp32_fallback = value.get("allow_fp32_fallback", False)
        quantization_enabled = value.get("quantization_enabled", False)
        quantization_policy = QuantizationPolicy.coerce(
            value.get("quantization_policy")
        )
        if type(allow_fp32_fallback) is not bool:
            raise QBenchError("allow_fp32_fallback must be a boolean")
        if type(quantization_enabled) is not bool:
            raise QBenchError("quantization_enabled must be a boolean")
        return cls(
            kernels=normalized_kernels,
            module_decisions=dict(decisions),
            unresolved_schemas=list(unresolved),
            allow_fp32_fallback=allow_fp32_fallback,
            quantization_enabled=quantization_enabled,
            quantization_policy=quantization_policy,
            scenario_names=list(scenario_names),
        )


@dataclass
class InspectionResult:
    support: dict[str, Any]
    operations: list[OperationRecord]
    plan: SimulationPlan
    verification: VerificationResult
    diagnostics: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def fully_supported(self) -> bool:
        return bool(self.support.get("fully_supported", False))

    def to_dict(self, *, include_operations: bool = False) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "support": copy.deepcopy(self.support),
            "plan": self.plan.to_dict(),
            "verification": self.verification.to_dict(),
            "diagnostics": copy.deepcopy(self.diagnostics),
        }
        if include_operations:
            result["operations"] = [op.to_dict() for op in self.operations]
        return strict_json_safe(result)


def normalize_scenarios(
    value: Scenario | Iterable[Scenario] | Mapping[str, Any],
) -> list[Scenario]:
    if isinstance(value, Scenario):
        scenarios = [value]
    elif isinstance(value, Mapping):
        scenarios = []
        for name, invocation in value.items():
            if isinstance(invocation, Scenario):
                scenarios.append(invocation)
            elif isinstance(invocation, Mapping) and (
                "args" in invocation or "kwargs" in invocation
            ):
                scenarios.append(
                    Scenario(
                        str(name),
                        tuple(invocation.get("args", ())),
                        invocation.get("kwargs", {}),
                    )
                )
            elif isinstance(invocation, tuple):
                scenarios.append(Scenario(str(name), invocation, {}))
            else:
                scenarios.append(Scenario(str(name), (invocation,), {}))
    else:
        scenarios = list(value)
    if not scenarios or not all(isinstance(item, Scenario) for item in scenarios):
        raise QBenchError("scenarios must contain at least one Scenario")
    names = [item.name for item in scenarios]
    if len(names) != len(set(names)):
        raise QBenchError("scenario names must be unique")
    return scenarios
