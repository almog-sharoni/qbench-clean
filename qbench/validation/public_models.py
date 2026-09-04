"""Reproducible, opt-in acceptance harness for public vision models.

The default command only prints the pinned matrix and never imports torchvision
or timm.  A live run requires all of ``--run``, ``--pretrained``, a CUDA device,
and ``QBENCH_RUN_PUBLIC_MODEL_ACCEPTANCE=1``.  This makes checkpoint downloads
and ImageNet/GPU use deliberate rather than a side effect of ordinary CI.

Live runs write ``acceptance.json``, ``subset_manifest.json``, per-case QBench
schema-v3 artifacts, and ``acceptance_diff.json`` when a baseline directory is
supplied.  Timing is retained in per-case evaluation artifacts but deliberately
excluded from regression decisions. Each case keeps strict modern routing
separate from a pinned, module-only legacy quantized comparison, and gates the
median overhead of the fast metric bundle against the same dual-forward loop.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


HARNESS_SCHEMA_VERSION = 1
SUBSET_SCHEMA_VERSION = 1
SUBSET_ALGORITHM = "sha256-rank-v1"
RUN_ENVIRONMENT_VARIABLE = "QBENCH_RUN_PUBLIC_MODEL_ACCEPTANCE"
DEFAULT_SUBSET_SEED = 20250317
DEFAULT_SAMPLE_COUNT = 128
DEFAULT_IMAGE_SIZES = (224, 256)
IMAGENET_VALIDATION_SIZE = 50_000
IMAGENET_CLASS_COUNT = 1_000
EQUIVALENCE_RTOL = 1e-5
EQUIVALENCE_ATOL = 1e-6
MAX_FAST_METRIC_OVERHEAD = 0.10
MIN_OVERHEAD_BATCH_INVOCATIONS = 8
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
STRICT_PHASE_NAME = "strict_quantization_disabled"
LEGACY_PHASE_NAME = "legacy_quantized_gpu"

# This intentionally spells out every field instead of inheriting evolving
# defaults. It is the compatibility configuration used by the pre-public-API
# workbench. The explicit fallback is part of that historical comparison and
# therefore cannot confer a fully-supported verdict.
LEGACY_QUANTIZATION_POLICY = {
    "quantization_type": "fp8_e4m3",
    "quantization_bias": None,
    "input_quantization": True,
    "weight_quantization": True,
    "output_quantization": False,
    "quantize_first_layer": False,
    "quant_mode": "tensor",
    "chunk_size": 128,
    "weight_mode": "channel",
    "weight_chunk_size": 128,
    "act_mode": "tensor",
    "act_chunk_size": 128,
    "output_mode": "tensor",
    "output_chunk_size": 128,
    "rounding": "nearest",
    "layer_config": {},
}


def strict_phase_configuration() -> dict[str, Any]:
    """Return the immutable-by-convention strict equivalence phase config."""
    return {
        "name": STRICT_PHASE_NAME,
        "quantization_enabled": False,
        "allow_fp32_fallback": False,
        "simulator_strict": True,
        "comparison_scope": [
            "output_structure",
            "rtol_atol",
            "prediction_parity",
            "top1_top5_parity",
        ],
    }


def legacy_quantized_configuration() -> dict[str, Any]:
    """Return the pinned legacy-compatibility quantized GPU configuration."""
    return {
        "name": LEGACY_PHASE_NAME,
        "quantization_enabled": True,
        "allow_fp32_fallback": True,
        "simulator_strict": False,
        "eligible_for_fully_supported_verdict": False,
        "requires_actual_quantized_execution": True,
        "routing_scope": "module_routes_only",
        "functional_routes": "captured_but_not_executed_for_legacy_comparison",
        "comparison_scope": ["prediction_sha256", "top1", "top5"],
        "quantization_policy": copy.deepcopy(LEGACY_QUANTIZATION_POLICY),
    }


class AcceptanceError(RuntimeError):
    """An invalid harness configuration or failed environment prerequisite."""


@dataclass(frozen=True)
class PublicModelSpec:
    key: str
    source: str
    model_name: str
    weights_id: str
    weights_enum: str | None
    model_seed: int


@dataclass(frozen=True)
class AcceptanceCase:
    model: PublicModelSpec
    image_size: int
    capture_seed: int

    @property
    def case_id(self) -> str:
        return f"{self.model.source}-{self.model.key}-{self.image_size}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source": self.model.source,
            "model_name": self.model.model_name,
            "weights_id": self.model.weights_id,
            "weights_enum": self.model.weights_enum,
            "model_seed": self.model.model_seed,
            "capture_seed": self.capture_seed,
            "capture_shape": [1, 3, self.image_size, self.image_size],
        }


PUBLIC_MODELS = (
    PublicModelSpec(
        key="resnet18",
        source="torchvision",
        model_name="resnet18",
        weights_id="IMAGENET1K_V1",
        weights_enum="ResNet18_Weights",
        model_seed=18001,
    ),
    PublicModelSpec(
        key="vit_b_16",
        source="torchvision",
        model_name="vit_b_16",
        weights_id="IMAGENET1K_V1",
        weights_enum="ViT_B_16_Weights",
        model_seed=16001,
    ),
    PublicModelSpec(
        key="mobilevit_s",
        source="timm",
        # The tagged name pins timm's concrete CVNets ImageNet checkpoint
        # instead of following a potentially changing unqualified default.
        model_name="mobilevit_s.cvnets_in1k",
        weights_id="cvnets_in1k",
        weights_enum=None,
        model_seed=26001,
    ),
)


def acceptance_matrix(
    image_sizes: Sequence[int] = DEFAULT_IMAGE_SIZES,
) -> tuple[AcceptanceCase, ...]:
    """Return the maintained three-model by two-shape acceptance matrix."""
    if any(isinstance(size, bool) or not isinstance(size, int) for size in image_sizes):
        raise AcceptanceError("image sizes must be positive integers")
    normalized = tuple(int(size) for size in image_sizes)
    if not normalized or any(size <= 0 for size in normalized):
        raise AcceptanceError("image sizes must be positive integers")
    if len(set(normalized)) != len(normalized):
        raise AcceptanceError("image sizes must be unique")
    return tuple(
        AcceptanceCase(
            model=model,
            image_size=size,
            capture_seed=model.model_seed + size,
        )
        for model in PUBLIC_MODELS
        for size in normalized
    )


PUBLIC_MODEL_MATRIX = acceptance_matrix()


@dataclass
class AcceptanceConfig:
    output_directory: str
    imagenet_directory: str
    baseline_directory: str | None = None
    subset_manifest: str | None = None
    explanations_file: str | None = None
    sample_count: int = DEFAULT_SAMPLE_COUNT
    subset_seed: int = DEFAULT_SUBSET_SEED
    batch_size: int = 8
    num_workers: int = 0
    device: str = "cuda"
    pretrained: bool = False
    enable_fx: bool = True
    enable_export: bool = True
    require_baseline: bool = False
    accuracy_tolerance: float = 0.0
    overhead_batches: int = 2
    overhead_repetitions: int = 7
    overhead_warmups: int = 1

    def __post_init__(self) -> None:
        for name in ("output_directory", "imagenet_directory", "device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise AcceptanceError(f"{name} must be a non-empty string")
            setattr(self, name, value.strip())
        for name in (
            "baseline_directory",
            "subset_manifest",
            "explanations_file",
        ):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise AcceptanceError(f"{name} must be a non-empty string or None")
                setattr(self, name, value.strip())
        for name, minimum in (
            ("sample_count", 1),
            ("subset_seed", 0),
            ("batch_size", 1),
            ("num_workers", 0),
            ("overhead_batches", 1),
            ("overhead_repetitions", 3),
            ("overhead_warmups", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "positive" if minimum else "non-negative"
                raise AcceptanceError(f"{name} must be a {qualifier} integer")
        for name in (
            "pretrained",
            "enable_fx",
            "enable_export",
            "require_baseline",
        ):
            if type(getattr(self, name)) is not bool:
                raise AcceptanceError(f"{name} must be a boolean")
        if (
            isinstance(self.accuracy_tolerance, bool)
            or not isinstance(self.accuracy_tolerance, (int, float))
            or not math.isfinite(float(self.accuracy_tolerance))
            or float(self.accuracy_tolerance) < 0
        ):
            raise AcceptanceError("accuracy_tolerance must be non-negative")
        self.accuracy_tolerance = float(self.accuracy_tolerance)

    def reproducibility_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "subset_seed": self.subset_seed,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "device": self.device,
            "pretrained": self.pretrained,
            "enable_fx": self.enable_fx,
            "enable_export": self.enable_export,
            "equivalence_rtol": EQUIVALENCE_RTOL,
            "equivalence_atol": EQUIVALENCE_ATOL,
            "accuracy_tolerance": self.accuracy_tolerance,
            "overhead_batches": self.overhead_batches,
            "overhead_repetitions": self.overhead_repetitions,
            "overhead_warmups": self.overhead_warmups,
            "minimum_overhead_batch_invocations": MIN_OVERHEAD_BATCH_INVOCATIONS,
            "max_fast_metric_overhead": MAX_FAST_METRIC_OVERHEAD,
            "deterministic_algorithms": True,
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            "allow_tf32": False,
        }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AcceptanceError(
            f"Could not read {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise AcceptanceError(f"{path} must contain a JSON object")
    return value


def deterministic_subset_indices(
    dataset_size: int,
    sample_count: int,
    seed: int = DEFAULT_SUBSET_SEED,
) -> list[int]:
    """Select stable indices without depending on PyTorch RNG implementation."""
    for name, value, minimum in (
        ("dataset_size", dataset_size, 1),
        ("sample_count", sample_count, 1),
        ("seed", seed, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            qualifier = "positive" if minimum else "non-negative"
            raise AcceptanceError(f"{name} must be a {qualifier} integer")
    if sample_count > dataset_size:
        raise AcceptanceError("sample_count cannot exceed dataset_size")
    seed_bytes = str(seed).encode("ascii")

    def rank(index: int) -> tuple[bytes, int]:
        digest = hashlib.sha256(
            b"qbench-imagenet-subset-v1\0"
            + seed_bytes
            + b"\0"
            + str(index).encode("ascii")
        ).digest()
        return digest, index

    return sorted(range(dataset_size), key=rank)[:sample_count]


def build_subset_manifest(
    dataset_size: int,
    sample_count: int,
    seed: int = DEFAULT_SUBSET_SEED,
    *,
    sample_identities: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    indices = deterministic_subset_indices(dataset_size, sample_count, seed)
    selection = {
        "algorithm": SUBSET_ALGORITHM,
        "dataset_size": dataset_size,
        "sample_count": sample_count,
        "seed": seed,
        "indices": indices,
    }
    manifest: dict[str, Any] = {
        "schema_version": SUBSET_SCHEMA_VERSION,
        "dataset": "ImageNet-1K-validation",
        **selection,
        "selection_sha256": hashlib.sha256(
            _canonical_json_bytes(selection)
        ).hexdigest(),
    }
    if sample_identities is not None:
        missing = [index for index in indices if index not in sample_identities]
        if missing:
            raise AcceptanceError(
                "sample identities are missing selected indices: "
                + ", ".join(str(index) for index in missing[:5])
            )
        manifest["samples"] = [
            {"index": index, **dict(sample_identities[index])} for index in indices
        ]
    return manifest


def validate_subset_manifest(
    manifest: Mapping[str, Any],
    *,
    dataset_size: int | None = None,
    sample_count: int | None = None,
    seed: int | None = None,
    sample_identities: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a persisted selection before reusing it for another run."""
    if not isinstance(manifest, Mapping):
        raise AcceptanceError("subset manifest must be a JSON object")
    required = {
        "schema_version",
        "algorithm",
        "dataset_size",
        "sample_count",
        "seed",
        "indices",
        "selection_sha256",
    }
    missing_fields = sorted(required - set(manifest))
    if missing_fields:
        raise AcceptanceError(
            "subset manifest is missing: " + ", ".join(missing_fields)
        )
    if manifest["schema_version"] != SUBSET_SCHEMA_VERSION:
        raise AcceptanceError("unsupported subset manifest schema version")
    if manifest["algorithm"] != SUBSET_ALGORITHM:
        raise AcceptanceError("unsupported subset selection algorithm")
    declared_size = manifest["dataset_size"]
    declared_count = manifest["sample_count"]
    declared_seed = manifest["seed"]
    expected_indices = deterministic_subset_indices(
        declared_size, declared_count, declared_seed
    )
    if list(manifest["indices"]) != expected_indices:
        raise AcceptanceError("subset indices do not match their seed/configuration")
    selection = {
        "algorithm": manifest["algorithm"],
        "dataset_size": declared_size,
        "sample_count": declared_count,
        "seed": declared_seed,
        "indices": expected_indices,
    }
    digest = hashlib.sha256(_canonical_json_bytes(selection)).hexdigest()
    if manifest["selection_sha256"] != digest:
        raise AcceptanceError("subset selection checksum mismatch")
    for name, expected, actual in (
        ("dataset_size", dataset_size, declared_size),
        ("sample_count", sample_count, declared_count),
        ("seed", seed, declared_seed),
    ):
        if expected is not None and expected != actual:
            raise AcceptanceError(
                f"subset manifest {name}={actual} does not match requested {expected}"
            )
    samples = manifest.get("samples")
    if samples is not None:
        if not isinstance(samples, list) or len(samples) != declared_count:
            raise AcceptanceError("subset samples must match sample_count")
        if [
            row.get("index") for row in samples if isinstance(row, Mapping)
        ] != expected_indices:
            raise AcceptanceError("subset sample identities are out of order")
        if sample_identities is not None:
            for row in samples:
                index = row["index"]
                actual = {key: value for key, value in row.items() if key != "index"}
                expected = dict(sample_identities.get(index, {}))
                if actual != expected:
                    raise AcceptanceError(
                        f"ImageNet sample identity changed at index {index}"
                    )
    return copy.deepcopy(dict(manifest))


def load_subset_manifest(path: str | os.PathLike[str], **expected) -> dict[str, Any]:
    return validate_subset_manifest(_read_json(Path(path)), **expected)


_ACCURACY_FIELDS = (
    "reference_top1",
    "reference_top5",
    "simulator_top1",
    "simulator_top5",
    "prediction_agreement",
)


def _case_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise AcceptanceError("acceptance report must contain a cases list")
    result: dict[str, Mapping[str, Any]] = {}
    for row in cases:
        if not isinstance(row, Mapping) or not isinstance(row.get("case_id"), str):
            raise AcceptanceError("each acceptance case must have a string case_id")
        if row["case_id"] in result:
            raise AcceptanceError(f"duplicate acceptance case {row['case_id']!r}")
        result[row["case_id"]] = row
    return result


def _explanation(
    explanations: Mapping[str, Any], case_id: str, code: str
) -> str | None:
    section = explanations.get(case_id, {})
    if not isinstance(section, Mapping):
        return None
    value = section.get(code)
    return str(value).strip() if isinstance(value, str) and value.strip() else None


def _record_explainable_change(
    *,
    regressions: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    explanations: Mapping[str, Any],
    case_id: str,
    code: str,
    before: Any,
    after: Any,
) -> None:
    reason = _explanation(explanations, case_id, code)
    row = {
        "case_id": case_id,
        "code": code,
        "before": before,
        "after": after,
        "explained": bool(reason),
    }
    if reason:
        row["explanation"] = reason
    else:
        regressions.append(row)
    changes.append(row)


def _legacy_phase(case: Mapping[str, Any]) -> Mapping[str, Any] | None:
    phases = case.get("phases")
    if not isinstance(phases, Mapping):
        return None
    phase = phases.get(LEGACY_PHASE_NAME)
    return phase if isinstance(phase, Mapping) else None


def _compare_legacy_quantized_phase(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    case_id: str,
    accuracy_tolerance: float,
    explanations: Mapping[str, Any],
    regressions: list[dict[str, Any]],
    changes: list[dict[str, Any]],
) -> None:
    """Compare legacy quantized results independently from strict coverage."""
    left = _legacy_phase(before)
    right = _legacy_phase(after)
    if left is None and right is None:
        return
    if left is None:
        changes.append({"case_id": case_id, "code": "legacy_quantized.added"})
        return
    if right is None:
        regressions.append({"case_id": case_id, "code": "legacy_quantized.missing"})
        return
    if right.get("status") != "passed":
        regressions.append(
            {
                "case_id": case_id,
                "code": "legacy_quantized.failed",
                "error": right.get("error"),
            }
        )
    if left.get("configuration") != right.get("configuration"):
        _record_explainable_change(
            regressions=regressions,
            changes=changes,
            explanations=explanations,
            case_id=case_id,
            code="legacy_quantized.configuration",
            before=left.get("configuration"),
            after=right.get("configuration"),
        )

    left_predictions = left.get("predictions", {})
    right_predictions = right.get("predictions", {})
    for role in ("reference", "simulator"):
        for field in ("sha256", "predictions"):
            before_value = (
                left_predictions.get(role, {}).get(field)
                if isinstance(left_predictions, Mapping)
                and isinstance(left_predictions.get(role), Mapping)
                else None
            )
            after_value = (
                right_predictions.get(role, {}).get(field)
                if isinstance(right_predictions, Mapping)
                and isinstance(right_predictions.get(role), Mapping)
                else None
            )
            if before_value != after_value:
                _record_explainable_change(
                    regressions=regressions,
                    changes=changes,
                    explanations=explanations,
                    case_id=case_id,
                    code=f"legacy_quantized.predictions.{role}.{field}",
                    before=before_value,
                    after=after_value,
                )

    left_accuracy = left.get("accuracy", {})
    right_accuracy = right.get("accuracy", {})
    for field in (
        "reference_top1",
        "reference_top5",
        "simulator_top1",
        "simulator_top5",
        "prediction_agreement",
    ):
        before_value = (
            left_accuracy.get(field) if isinstance(left_accuracy, Mapping) else None
        )
        after_value = (
            right_accuracy.get(field) if isinstance(right_accuracy, Mapping) else None
        )
        if before_value == after_value:
            continue
        if (
            before_value is not None
            and after_value is not None
            and abs(float(after_value) - float(before_value)) <= accuracy_tolerance
        ):
            continue
        _record_explainable_change(
            regressions=regressions,
            changes=changes,
            explanations=explanations,
            case_id=case_id,
            code=f"legacy_quantized.accuracy.{field}",
            before=before_value,
            after=after_value,
        )


def compare_acceptance_reports(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    accuracy_tolerance: float = 0.0,
    explanations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare support coverage and accuracy while allowing additive discovery."""
    if (
        isinstance(accuracy_tolerance, bool)
        or not isinstance(accuracy_tolerance, (int, float))
        or not math.isfinite(float(accuracy_tolerance))
        or accuracy_tolerance < 0
    ):
        raise AcceptanceError("accuracy_tolerance must be non-negative")
    explanations = {} if explanations is None else explanations
    if not isinstance(explanations, Mapping):
        raise AcceptanceError("explanations must be a JSON object")
    baseline_cases = _case_map(baseline)
    current_cases = _case_map(current)
    regressions: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []

    baseline_subset = baseline.get("subset", {})
    current_subset = current.get("subset", {})
    subset_fields = (
        "algorithm",
        "dataset_size",
        "sample_count",
        "seed",
        "indices",
        "samples",
        "selection_sha256",
    )
    if any(
        baseline_subset.get(key) != current_subset.get(key) for key in subset_fields
    ):
        code = "subset.changed"
        reason = _explanation(explanations, "__global__", code)
        row = {"case_id": "__global__", "code": code, "explained": bool(reason)}
        if reason:
            row["explanation"] = reason
        else:
            regressions.append(row)
        changes.append(row)

    for field in ("matrix", "config", "class_index_sha256"):
        if baseline.get(field) == current.get(field):
            continue
        code = f"run.{field}_changed"
        reason = _explanation(explanations, "__global__", code)
        row = {
            "case_id": "__global__",
            "code": code,
            "explained": bool(reason),
        }
        if reason:
            row["explanation"] = reason
        else:
            regressions.append(row)
        changes.append(row)

    for case_id in sorted(set(baseline_cases) | set(current_cases)):
        before = baseline_cases.get(case_id)
        after = current_cases.get(case_id)
        if before is None:
            changes.append({"case_id": case_id, "code": "case.added"})
            continue
        if after is None:
            regressions.append({"case_id": case_id, "code": "case.missing"})
            continue
        case_changes: dict[str, Any] = {"case_id": case_id}
        if after.get("status") != "passed":
            regressions.append(
                {
                    "case_id": case_id,
                    "code": "case.failed",
                    "error": after.get("error"),
                }
            )

        before_support = before.get("support", {})
        after_support = after.get("support", {})
        before_mappings = set(before_support.get("planned_mappings", ()))
        after_mappings = set(after_support.get("planned_mappings", ()))
        missing_mappings = sorted(before_mappings - after_mappings)
        added_mappings = sorted(after_mappings - before_mappings)
        if missing_mappings:
            regressions.append(
                {
                    "case_id": case_id,
                    "code": "support.mappings_removed",
                    "values": missing_mappings,
                }
            )
        if missing_mappings or added_mappings:
            case_changes["mappings"] = {
                "removed": missing_mappings,
                "added": added_mappings,
            }

        before_schemas = set(before_support.get("operation_schemas", ()))
        after_schemas = set(after_support.get("operation_schemas", ()))
        missing_schemas = sorted(before_schemas - after_schemas)
        added_schemas = sorted(after_schemas - before_schemas)
        if missing_schemas:
            regressions.append(
                {
                    "case_id": case_id,
                    "code": "support.operations_disappeared",
                    "values": missing_schemas,
                }
            )
        if missing_schemas or added_schemas:
            case_changes["operations"] = {
                "removed": missing_schemas,
                "added": added_schemas,
            }
        before_counts = before_support.get("operation_counts", {})
        after_counts = after_support.get("operation_counts", {})
        if isinstance(before_counts, Mapping) and isinstance(after_counts, Mapping):
            count_changes = {
                schema: {
                    "before": int(before_counts.get(schema, 0)),
                    "after": int(after_counts.get(schema, 0)),
                }
                for schema in sorted(set(before_counts) | set(after_counts))
                if int(before_counts.get(schema, 0)) != int(after_counts.get(schema, 0))
            }
            decreases = {
                schema: values
                for schema, values in count_changes.items()
                if values["after"] < values["before"]
            }
            if decreases:
                regressions.append(
                    {
                        "case_id": case_id,
                        "code": "support.operation_counts_decreased",
                        "values": decreases,
                    }
                )
            if count_changes:
                case_changes["operation_count_changes"] = count_changes
        before_ledger = before_support.get("operation_ledger_counts", {})
        after_ledger = after_support.get("operation_ledger_counts", {})
        if isinstance(before_ledger, Mapping) and isinstance(after_ledger, Mapping):
            removed_ledger = {
                identity: {
                    "before": int(before_ledger.get(identity, 0)),
                    "after": int(after_ledger.get(identity, 0)),
                }
                for identity in sorted(set(before_ledger) | set(after_ledger))
                if int(after_ledger.get(identity, 0))
                < int(before_ledger.get(identity, 0))
            }
            added_ledger = {
                identity: {
                    "before": int(before_ledger.get(identity, 0)),
                    "after": int(after_ledger.get(identity, 0)),
                }
                for identity in sorted(set(before_ledger) | set(after_ledger))
                if int(after_ledger.get(identity, 0))
                > int(before_ledger.get(identity, 0))
            }
            if removed_ledger:
                regressions.append(
                    {
                        "case_id": case_id,
                        "code": "support.raw_operations_disappeared",
                        "values": removed_ledger,
                    }
                )
            if removed_ledger or added_ledger:
                case_changes["raw_operation_count_changes"] = {
                    "removed": removed_ledger,
                    "added": added_ledger,
                }

        before_digest = before.get("weights", {}).get("state_sha256")
        after_digest = after.get("weights", {}).get("state_sha256")
        if before_digest != after_digest:
            code = "weights.state_sha256"
            reason = _explanation(explanations, case_id, code)
            row = {
                "case_id": case_id,
                "code": code,
                "before": before_digest,
                "after": after_digest,
                "explained": bool(reason),
            }
            if reason:
                row["explanation"] = reason
            else:
                regressions.append(row)
            case_changes.setdefault("values", []).append(row)

        before_accuracy = before.get("accuracy", {})
        after_accuracy = after.get("accuracy", {})
        for field in _ACCURACY_FIELDS:
            left = before_accuracy.get(field)
            right = after_accuracy.get(field)
            if left == right:
                continue
            changed = (
                left is None
                or right is None
                or abs(float(right) - float(left)) > accuracy_tolerance
            )
            if not changed:
                continue
            code = f"accuracy.{field}"
            reason = _explanation(explanations, case_id, code)
            row = {
                "case_id": case_id,
                "code": code,
                "before": left,
                "after": right,
                "explained": bool(reason),
            }
            if reason:
                row["explanation"] = reason
            else:
                regressions.append(row)
            case_changes.setdefault("values", []).append(row)

        before_predictions = before.get("predictions")
        after_predictions = after.get("predictions")
        if not isinstance(before_predictions, Mapping):
            if isinstance(after_predictions, Mapping):
                case_changes["strict_predictions"] = "added"
        else:
            for role in ("reference", "simulator"):
                for field in ("sha256", "predictions"):
                    before_value = (
                        before_predictions.get(role, {}).get(field)
                        if isinstance(before_predictions.get(role), Mapping)
                        else None
                    )
                    after_value = (
                        after_predictions.get(role, {}).get(field)
                        if isinstance(after_predictions, Mapping)
                        and isinstance(after_predictions.get(role), Mapping)
                        else None
                    )
                    if before_value == after_value:
                        continue
                    code = f"strict.predictions.{role}.{field}"
                    reason = _explanation(explanations, case_id, code)
                    row = {
                        "case_id": case_id,
                        "code": code,
                        "before": before_value,
                        "after": after_value,
                        "explained": bool(reason),
                    }
                    if reason:
                        row["explanation"] = reason
                    else:
                        regressions.append(row)
                    case_changes.setdefault("values", []).append(row)

        _compare_legacy_quantized_phase(
            before,
            after,
            case_id=case_id,
            accuracy_tolerance=float(accuracy_tolerance),
            explanations=explanations,
            regressions=regressions,
            changes=changes,
        )

        # A newly discovered gap may conservatively change the verdict and is
        # recorded, but does not erase mappings or operations by itself.
        before_verdict = before_support.get("verdict")
        after_verdict = after_support.get("verdict")
        if before_verdict != after_verdict:
            case_changes["verdict"] = {
                "before": before_verdict,
                "after": after_verdict,
            }
        if len(case_changes) > 1:
            changes.append(case_changes)

    return {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "baseline_schema_version": baseline.get("schema_version"),
        "current_schema_version": current.get("schema_version"),
        "accuracy_tolerance": float(accuracy_tolerance),
        "passed": not regressions,
        "regression_count": len(regressions),
        "regressions": regressions,
        "changes": changes,
    }


def compare_acceptance_directories(
    baseline_directory: str | os.PathLike[str],
    current_directory: str | os.PathLike[str],
    *,
    accuracy_tolerance: float = 0.0,
    explanations: Mapping[str, Any] | None = None,
    output_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Compare two harness artifact directories without loading any models."""
    baseline = _read_json(Path(baseline_directory) / "acceptance.json")
    current = _read_json(Path(current_directory) / "acceptance.json")
    diff = compare_acceptance_reports(
        baseline,
        current,
        accuracy_tolerance=accuracy_tolerance,
        explanations=explanations,
    )
    if output_path is not None:
        _write_json(Path(output_path), diff)
    return diff


class _CanonicalTargetMap:
    def __init__(self, mapping: Mapping[int, int]):
        self.mapping = dict(mapping)

    def __call__(self, target: int) -> int:
        return self.mapping[int(target)]


class _AcceptanceProvider:
    """Small trusted provider that owns device movement and labeled batches."""

    def __init__(self, model, scenario, loader, device: str):
        self.model = model
        self.scenario = scenario
        self.loader = loader
        self.device = device
        self._prediction_capture = False
        self._selection_calls = 0
        self._prediction_hashers: dict[str, Any] = {}
        self._prediction_counts: Counter[str] = Counter()

    def build_model(self):
        return self.model

    def clone_model(self, model):
        return copy.deepcopy(model)

    def capture_scenarios(self):
        return [self.scenario]

    def evaluation_loader(self):
        return self.loader

    def prepare_evaluation_batch(self, batch):
        from qbench.schemas import Scenario

        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise AcceptanceError("ImageNet loader must yield (images, targets)")
        images, targets = batch
        images = images.to(self.device, non_blocking=True)
        return Scenario("evaluation", (images,), {}), targets

    def select_metric_output(self, output):
        if isinstance(output, Mapping) and "logits" in output:
            selected = output["logits"]
        elif hasattr(output, "logits"):
            selected = output.logits
        else:
            selected = output
        if self._prediction_capture:
            role = "reference" if self._selection_calls % 2 == 0 else "simulator"
            self._selection_calls += 1
            self._record_predictions(role, selected)
        return selected

    def begin_prediction_capture(self) -> None:
        self._prediction_capture = True
        self._selection_calls = 0
        self._prediction_hashers = {
            "reference": hashlib.sha256(),
            "simulator": hashlib.sha256(),
        }
        self._prediction_counts.clear()

    def end_prediction_capture(self, batches: int) -> dict[str, Any]:
        self._prediction_capture = False
        if self._selection_calls != batches * 2:
            raise AcceptanceError(
                "prediction capture did not observe one reference and simulator output "
                "per evaluation batch"
            )
        return {
            role: {
                "sha256": self._prediction_hashers[role].hexdigest(),
                "predictions": int(self._prediction_counts[role]),
            }
            for role in ("reference", "simulator")
        }

    def cancel_prediction_capture(self) -> None:
        self._prediction_capture = False

    def _record_predictions(self, role: str, output: Any) -> None:
        import torch

        if not torch.is_tensor(output) or output.ndim < 2:
            raise AcceptanceError(
                "classification metric output must be a logits tensor"
            )
        predictions = (
            output.detach()
            .argmax(dim=-1)
            .to(device="cpu", dtype=torch.int64)
            .contiguous()
        )
        self._prediction_hashers[role].update(
            _canonical_json_bytes({"shape": list(predictions.shape)})
        )
        self._prediction_hashers[role].update(b"\0")
        self._prediction_hashers[role].update(memoryview(predictions.numpy()).cast("B"))
        self._prediction_hashers[role].update(b"\0")
        self._prediction_counts[role] += int(predictions.numel())


def _default_class_index_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "imagenet_class_index.json"


def _resolve_imagenet_split(path: str | os.PathLike[str]) -> Path:
    root = Path(path).expanduser().resolve()
    split = root / "val"
    if split.is_dir():
        root = split
    if not root.is_dir():
        raise AcceptanceError(f"ImageNet validation directory does not exist: {root}")
    return root


def _load_class_index(path: Path) -> dict[str, int]:
    payload = _read_json(path)
    result = {}
    for key, value in payload.items():
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], str)
            and str(key).isdigit()
        ):
            result[value[0]] = int(key)
    if len(result) != IMAGENET_CLASS_COUNT:
        raise AcceptanceError(
            "ImageNet class index must define "
            f"{IMAGENET_CLASS_COUNT} WNIDs, found {len(result)}"
        )
    return result


def _imagenet_inventory(
    directory: Path,
    class_index_path: Path,
) -> tuple[int, dict[int, int], dict[int, dict[str, Any]]]:
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(directory)
    if len(dataset) != IMAGENET_VALIDATION_SIZE:
        raise AcceptanceError(
            "ImageNet-1K validation must contain exactly "
            f"{IMAGENET_VALIDATION_SIZE} samples, found {len(dataset)}"
        )
    if len(dataset.class_to_idx) != IMAGENET_CLASS_COUNT:
        raise AcceptanceError(
            "ImageNet-1K validation must contain exactly "
            f"{IMAGENET_CLASS_COUNT} classes, found {len(dataset.class_to_idx)}"
        )
    canonical = _load_class_index(class_index_path)
    missing = sorted(set(dataset.class_to_idx) - set(canonical))
    if missing:
        raise AcceptanceError(
            "ImageNet validation folders lack canonical target mappings: "
            + ", ".join(missing[:5])
        )
    target_map = {
        local_index: canonical[wnid]
        for wnid, local_index in dataset.class_to_idx.items()
    }
    identities = {
        index: {
            "relative_path": str(Path(filename).resolve().relative_to(directory)),
            "target": target_map[int(target)],
        }
        for index, (filename, target) in enumerate(dataset.samples)
    }
    return len(dataset), target_map, identities


def _seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass


def _seed_worker(worker_id: int) -> None:
    import random

    import torch

    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass


def _model_state_sha256(model) -> str:
    """Hash tensor names, metadata, and bytes for exact checkpoint provenance."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not torch.is_tensor(tensor):
            continue
        value = tensor.detach().to(device="cpu").contiguous()
        metadata = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        digest.update(_canonical_json_bytes(metadata))
        digest.update(b"\0")
        # The maintained public models use NumPy-compatible floating/integer
        # state.  memoryview avoids constructing one extra checkpoint-sized
        # bytes object while hashing ViT-B/16.
        digest.update(memoryview(value.numpy()).cast("B"))
        digest.update(b"\0")
    return digest.hexdigest()


def _build_public_model(
    case: AcceptanceCase, *, pretrained: bool
) -> tuple[Any, dict[str, Any]]:
    if not pretrained:
        raise AcceptanceError("public-model acceptance requires pretrained weights")
    _seed_everything(case.model.model_seed)
    if case.model.source == "torchvision":
        from torchvision import models

        if case.model.weights_enum is None:
            raise AcceptanceError(f"{case.case_id} has no pinned weights enum")
        weights_type = getattr(models, case.model.weights_enum)
        weights = getattr(weights_type, case.model.weights_id)
        builder = getattr(models, case.model.model_name)
        derivation = "native"
        if case.model.key == "vit_b_16" and case.image_size != 224:
            from torchvision.models.vision_transformer import interpolate_embeddings

            native = builder(weights=weights).cpu().eval()
            state = interpolate_embeddings(
                image_size=case.image_size,
                patch_size=16,
                model_state=native.state_dict(),
                interpolation_mode="bicubic",
                reset_heads=False,
            )
            del native
            model = builder(weights=None, image_size=case.image_size)
            model.load_state_dict(state, strict=True)
            derivation = "torchvision.interpolate_embeddings:bicubic"
        else:
            model = builder(weights=weights)
        provenance = {
            "source": "torchvision",
            "id": f"{case.model.weights_enum}.{case.model.weights_id}",
            "url": getattr(weights, "url", None),
            "derivation": derivation,
        }
    elif case.model.source == "timm":
        import timm

        model = timm.create_model(case.model.model_name, pretrained=True)
        pretrained_cfg = dict(getattr(model, "pretrained_cfg", {}) or {})
        provenance = {
            "source": "timm",
            "id": case.model.model_name,
            "url": pretrained_cfg.get("url"),
            "hf_hub_id": pretrained_cfg.get("hf_hub_id"),
            "derivation": "native",
        }
    else:  # pragma: no cover - maintained constants make this unreachable
        raise AcceptanceError(f"unsupported model source {case.model.source!r}")
    model.cpu().eval()
    provenance["state_sha256"] = _model_state_sha256(model)
    return model, provenance


def _validation_transform(case: AcceptanceCase, model):
    if case.model.source == "torchvision":
        from torchvision import models

        weights_type = getattr(models, case.model.weights_enum)
        weights = getattr(weights_type, case.model.weights_id)
        native = weights.transforms()
        native_crop = int(native.crop_size[0])
        native_resize = int(native.resize_size[0])
        resize = round(case.image_size * native_resize / native_crop)
        return weights.transforms(crop_size=case.image_size, resize_size=resize)
    import timm

    data_config = dict(timm.data.resolve_model_data_config(model))
    data_config["input_size"] = (3, case.image_size, case.image_size)
    return timm.data.create_transform(**data_config, is_training=False)


def _build_loader(
    case: AcceptanceCase,
    model,
    *,
    imagenet_directory: Path,
    target_map: Mapping[int, int],
    indices: Sequence[int],
    config: AcceptanceConfig,
):
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(
        imagenet_directory,
        transform=_validation_transform(case, model),
        target_transform=_CanonicalTargetMap(target_map),
    )
    subset = Subset(dataset, list(indices))
    generator = torch.Generator().manual_seed(config.subset_seed)
    return DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=_seed_worker if config.num_workers > 0 else None,
        generator=generator,
    )


def _support_summary(inspection) -> dict[str, Any]:
    operation_counts = Counter(operation.schema for operation in inspection.operations)
    ledger_counts = Counter(
        "|".join(
            (
                operation.scenario,
                operation.module_path,
                operation.schema,
                operation.overload,
                operation.kernel or "",
                operation.classification,
            )
        )
        for operation in inspection.operations
    )
    planned_mappings = [
        f"{route}=>{row.get('name', '<unnamed>')}"
        for route, row in sorted(inspection.plan.kernels.items())
    ]
    planned_mappings.extend(
        f"module:{path}=>{decision}"
        for path, decision in sorted(inspection.plan.module_decisions.items())
    )
    return {
        "verdict": inspection.support.get("verdict"),
        "fully_supported": inspection.fully_supported,
        "capture_complete": inspection.support.get("capture_complete"),
        "replacement_coverage": inspection.support.get("replacement_coverage"),
        "strict_realization": inspection.support.get("strict_realization"),
        "planned_mappings": sorted(set(planned_mappings)),
        "operation_schemas": sorted(operation_counts),
        "operation_counts": dict(sorted(operation_counts.items())),
        "operation_ledger_counts": dict(sorted(ledger_counts.items())),
        "gaps": copy.deepcopy(inspection.support.get("gaps", [])),
    }


def _accuracy_summary(evaluation) -> dict[str, Any]:
    return {
        key: evaluation.metrics.get(key)
        for key in (
            *_ACCURACY_FIELDS,
            "mae",
            "mse",
            "cosine_similarity",
            "sqnr_db",
            "nonfinite_reference",
            "nonfinite_simulator",
        )
    }


def _evaluate_with_prediction_digest(reference, simulator, provider):
    """Run the canonical fast bundle while hashing its selected predictions."""
    from qbench import EvaluationConfig, evaluate

    provider.begin_prediction_capture()
    try:
        evaluation = evaluate(
            reference,
            simulator,
            provider,
            EvaluationConfig(metrics="fast", task="classification"),
        )
        predictions = provider.end_prediction_capture(evaluation.batches)
    except Exception:
        provider.cancel_prediction_capture()
        raise
    return evaluation, predictions


def summarize_fast_metric_overhead(
    bare_seconds: Sequence[float],
    fast_seconds: Sequence[float],
    *,
    batches: int,
    bare_reference_forwards: Sequence[int],
    bare_simulator_forwards: Sequence[int],
    reference_forwards: Sequence[int],
    simulator_forwards: Sequence[int],
) -> dict[str, Any]:
    """Summarize the live timing gate without depending on model libraries."""
    bare = [float(value) for value in bare_seconds]
    fast = [float(value) for value in fast_seconds]
    if len(bare) != len(fast) or len(bare) < 3:
        raise AcceptanceError("overhead timing requires at least three paired samples")
    if any(not math.isfinite(value) or value <= 0 for value in (*bare, *fast)):
        raise AcceptanceError("overhead timing samples must be finite and positive")
    if isinstance(batches, bool) or not isinstance(batches, int) or batches <= 0:
        raise AcceptanceError("overhead batches must be a positive integer")
    count_samples = (
        bare_reference_forwards,
        bare_simulator_forwards,
        reference_forwards,
        simulator_forwards,
    )
    if any(len(values) != len(fast) for values in count_samples):
        raise AcceptanceError("forward-count samples must match timing samples")
    exact_forwards = all(
        bare_reference == bare_simulator == reference == simulator == batches
        for bare_reference, bare_simulator, reference, simulator in zip(
            bare_reference_forwards,
            bare_simulator_forwards,
            reference_forwards,
            simulator_forwards,
        )
    )
    bare_median = statistics.median(bare)
    fast_median = statistics.median(fast)
    overhead = (fast_median - bare_median) / bare_median
    return {
        "status": "passed"
        if exact_forwards and overhead <= MAX_FAST_METRIC_OVERHEAD
        else "failed",
        "repetitions": len(bare),
        "batches_per_repetition": batches,
        "bare_seconds": bare,
        "fast_seconds": fast,
        "bare_median_seconds": bare_median,
        "fast_median_seconds": fast_median,
        "overhead_fraction": overhead,
        "maximum_overhead_fraction": MAX_FAST_METRIC_OVERHEAD,
        "exactly_two_forwards_per_batch": exact_forwards,
        "forward_counts": {
            "bare_reference": list(bare_reference_forwards),
            "bare_simulator": list(bare_simulator_forwards),
            "fast_reference": list(reference_forwards),
            "fast_simulator": list(simulator_forwards),
        },
        "retains_intermediate_activations": False,
        "metrics": "fast",
    }


def _runtime_rng_state():
    import torch

    return (
        torch.random.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    )


def _restore_runtime_rng(state) -> None:
    import torch

    torch.random.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def _bare_dual_forward_loop(reference, simulator, provider) -> tuple[int, int]:
    import torch

    from qbench.capture import clone_invocation
    from qbench.schemas import Scenario

    reference_forwards = 0
    simulator_forwards = 0
    initial_rng = _runtime_rng_state()
    training = {path: module.training for path, module in reference.named_modules()}
    reference.eval()
    try:
        for batch in provider.evaluation_loader():
            prepared = provider.prepare_evaluation_batch(batch)
            scenario = prepared[0] if isinstance(prepared, (tuple, list)) else prepared
            if not isinstance(scenario, Scenario):
                raise AcceptanceError("performance provider did not return a Scenario")
            args, kwargs = clone_invocation(scenario)
            batch_rng = _runtime_rng_state()
            with torch.inference_mode():
                reference_output = reference(*args, **kwargs)
            reference_forwards += 1
            _restore_runtime_rng(batch_rng)
            simulator_output = simulator.run(scenario)
            simulator_forwards += 1
            del reference_output, simulator_output
    finally:
        for path, module in reference.named_modules():
            module.training = training[path]
        _restore_runtime_rng(initial_rng)
    return reference_forwards, simulator_forwards


def _timed_cuda_call(call, device: str) -> tuple[Any, float]:
    import torch

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = call()
    torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


def _fast_metric_overhead(
    reference,
    simulator,
    provider: _AcceptanceProvider,
    config: AcceptanceConfig,
) -> dict[str, Any]:
    import torch

    from qbench import EvaluationConfig, evaluate

    batches = []
    for index, batch in enumerate(provider.evaluation_loader()):
        if index >= config.overhead_batches:
            break
        images, targets = batch
        batches.append(
            (
                images.to(config.device, non_blocking=True),
                targets,
            )
        )
    if not batches:
        raise AcceptanceError("overhead benchmark received no ImageNet batches")
    timing_cycles = max(1, math.ceil(MIN_OVERHEAD_BATCH_INVOCATIONS / len(batches)))
    timed_batches = tuple(batches) * timing_cycles
    performance_provider = _AcceptanceProvider(
        reference,
        provider.scenario,
        timed_batches,
        config.device,
    )
    hook_snapshot = {
        ("reference", path): tuple(module._forward_hooks)
        for path, module in reference.named_modules()
    }
    converted_model = simulator._model
    hook_snapshot.update(
        {
            ("simulator", path): tuple(module._forward_hooks)
            for path, module in converted_model.named_modules()
        }
    )
    # The preceding accuracy evaluation can leave large transient blocks in
    # PyTorch's CUDA cache.  Normalize allocator state before both paths are
    # warmed and timed so allocation pressure cannot bias the overhead ratio.
    torch.cuda.synchronize(config.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(config.device)

    def run_fast():
        return evaluate(
            reference,
            simulator,
            performance_provider,
            EvaluationConfig(metrics="fast", task="classification"),
        )

    for _ in range(config.overhead_warmups):
        _timed_cuda_call(
            lambda: _bare_dual_forward_loop(reference, simulator, performance_provider),
            config.device,
        )
        _timed_cuda_call(run_fast, config.device)

    bare_seconds: list[float] = []
    fast_seconds: list[float] = []
    reference_forwards: list[int] = []
    simulator_forwards: list[int] = []
    bare_reference_forwards: list[int] = []
    bare_simulator_forwards: list[int] = []
    for repetition in range(config.overhead_repetitions):
        measurements = ("bare", "fast") if repetition % 2 == 0 else ("fast", "bare")
        for kind in measurements:
            if kind == "bare":
                forwards, elapsed = _timed_cuda_call(
                    lambda: _bare_dual_forward_loop(
                        reference, simulator, performance_provider
                    ),
                    config.device,
                )
                if forwards != (len(timed_batches), len(timed_batches)):
                    raise AcceptanceError(
                        "bare loop did not perform exactly two forwards"
                    )
                bare_seconds.append(elapsed)
                bare_reference_forwards.append(forwards[0])
                bare_simulator_forwards.append(forwards[1])
            else:
                report, elapsed = _timed_cuda_call(run_fast, config.device)
                if report.details:
                    raise AcceptanceError(
                        "fast metric bundle retained opt-in detailed state"
                    )
                fast_seconds.append(elapsed)
                reference_forwards.append(report.reference_forwards)
                simulator_forwards.append(report.simulator_forwards)
    hook_snapshot_after = {
        ("reference", path): tuple(module._forward_hooks)
        for path, module in reference.named_modules()
    }
    hook_snapshot_after.update(
        {
            ("simulator", path): tuple(module._forward_hooks)
            for path, module in converted_model.named_modules()
        }
    )
    if hook_snapshot_after != hook_snapshot:
        raise AcceptanceError("fast metrics left forward hooks installed")
    result = summarize_fast_metric_overhead(
        bare_seconds,
        fast_seconds,
        batches=len(timed_batches),
        bare_reference_forwards=bare_reference_forwards,
        bare_simulator_forwards=bare_simulator_forwards,
        reference_forwards=reference_forwards,
        simulator_forwards=simulator_forwards,
    )
    result["cuda_peak_allocated_bytes"] = int(
        torch.cuda.max_memory_allocated(config.device)
    )
    result["distinct_batches"] = len(batches)
    result["timing_cycles_per_repetition"] = timing_cycles
    result["fast_details_empty"] = True
    return result


_QBENCH_ARTIFACT_NAMES = (
    "manifest.json",
    "support.json",
    "operations.jsonl.gz",
    "plan.json",
    "evaluation.json",
    "state.pt",
)


def _prepare_artifact_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for stale_name in _QBENCH_ARTIFACT_NAMES:
        stale = directory / stale_name
        if stale.is_file() or stale.is_symlink():
            stale.unlink()


def _legacy_module_only_plan(plan):
    """Derive the pre-functional-routing plan used only for baseline parity."""
    legacy_plan = copy.deepcopy(plan)
    schema_rows = {
        route: row
        for route, row in legacy_plan.kernels.items()
        if route.startswith("schema:")
    }
    legacy_plan.kernels = {
        route: row
        for route, row in legacy_plan.kernels.items()
        if route.startswith("module:")
    }
    compute_schemas = {
        route.removeprefix("schema:")
        for route, row in schema_rows.items()
        if row.get("classification") != "structural"
    }
    structural_schemas = {
        route.removeprefix("schema:")
        for route, row in schema_rows.items()
        if row.get("classification") == "structural"
    }
    legacy_plan.unresolved_schemas = sorted(
        set(legacy_plan.unresolved_schemas) | compute_schemas
    )
    legacy_plan.allow_fp32_fallback = True
    return legacy_plan, {
        "derivation": "module-routes-only-v1",
        "excluded_schema_routes": sorted(schema_rows),
        "fp32_compute_schemas": sorted(compute_schemas),
        "native_structural_schemas": sorted(structural_schemas),
    }


def _run_legacy_quantized_phase(
    model,
    provider: _AcceptanceProvider,
    config: AcceptanceConfig,
    *,
    artifact_directory: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    """Run the compatibility phase independently of strict support status."""
    from qbench import InspectionConfig, build_simulator, inspect_provider
    from qbench.artifacts import write_artifacts

    configuration = legacy_quantized_configuration()
    inspection = simulator = evaluation = None
    predictions: dict[str, Any] = {}
    phase: dict[str, Any] = {
        "status": "not_run",
        "configuration": configuration,
        "artifacts": str(artifact_path),
    }
    try:
        inspection = inspect_provider(
            provider,
            InspectionConfig(
                allow_fp32_fallback=configuration["allow_fp32_fallback"],
                # Verification is run below after deriving the historical
                # module-only plan, so artifacts never claim the full modern
                # functional plan produced these legacy predictions.
                verify=False,
                enable_fx=config.enable_fx,
                enable_export=config.enable_export,
                quantization_enabled=configuration["quantization_enabled"],
                device=config.device,
                quantization_policy=configuration["quantization_policy"],
            ),
        )
        legacy_plan, derivation = _legacy_module_only_plan(inspection.plan)
        inspection.plan = legacy_plan
        inspection.support = copy.deepcopy(inspection.support)
        inspection.support.update(
            {
                "verdict": "partial_or_unsupported",
                "fully_supported": False,
                "legacy_compatibility": derivation,
            }
        )
        inspection.diagnostics = copy.deepcopy(inspection.diagnostics)
        inspection.diagnostics["legacy_compatibility"] = derivation
        simulator = build_simulator(
            model,
            legacy_plan,
            strict=configuration["simulator_strict"],
        )
        inspection.verification = simulator.verify(legacy_plan.scenarios)
        if (
            not inspection.verification.succeeded
            or not inspection.verification.quantized_execution
        ):
            details = "; ".join(inspection.verification.errors)
            if not inspection.verification.quantized_execution:
                details = (details + "; " if details else "") + (
                    "no complete quantized-execution evidence"
                )
            raise AcceptanceError(
                "legacy module-only plan verification failed: " + details
            )
        evaluation, predictions = _evaluate_with_prediction_digest(
            model, simulator, provider
        )
        forward_accounting = {
            "batches": evaluation.batches,
            "reference_forwards": evaluation.reference_forwards,
            "simulator_forwards": evaluation.simulator_forwards,
            "exactly_two_forwards_per_batch": (
                evaluation.reference_forwards
                == evaluation.simulator_forwards
                == evaluation.batches
            ),
        }
        phase.update(
            {
                "status": (
                    "passed"
                    if forward_accounting["exactly_two_forwards_per_batch"]
                    else "failed"
                ),
                "support": _support_summary(inspection),
                "verification": inspection.verification.to_dict(),
                "plan_derivation": derivation,
                "accuracy": _accuracy_summary(evaluation),
                "predictions": predictions,
                "forward_accounting": forward_accounting,
            }
        )
        if phase["status"] != "passed":
            phase["error"] = (
                "AcceptanceError: legacy quantized evaluation did not perform "
                "exactly two forwards per batch"
            )
    except Exception as exc:
        phase.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "support": (
                    _support_summary(inspection) if inspection is not None else {}
                ),
                "accuracy": (
                    _accuracy_summary(evaluation) if evaluation is not None else {}
                ),
                "predictions": predictions,
            }
        )
    finally:
        if inspection is not None:
            try:
                write_artifacts(
                    artifact_directory,
                    inspection,
                    evaluation=evaluation,
                )
            except Exception as exc:
                phase["status"] = "failed"
                phase["artifact_error"] = f"{type(exc).__name__}: {exc}"
        if simulator is not None:
            simulator.close()
    return phase


def _execute_case(
    case: AcceptanceCase,
    config: AcceptanceConfig,
    *,
    imagenet_directory: Path,
    target_map: Mapping[int, int],
    subset: Mapping[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    import torch

    from qbench import InspectionConfig, Scenario, build_simulator, inspect_provider
    from qbench.artifacts import write_artifacts

    case_directory = output_directory / "cases" / case.case_id
    strict_directory = case_directory / STRICT_PHASE_NAME
    legacy_directory = case_directory / LEGACY_PHASE_NAME
    for artifact_directory in (case_directory, strict_directory, legacy_directory):
        _prepare_artifact_directory(artifact_directory)

    model = provider = strict_simulator = strict_inspection = strict_evaluation = None
    strict_predictions: dict[str, Any] = {}
    performance: dict[str, Any] = {"status": "not_run"}
    strict_phase: dict[str, Any] = {
        "status": "not_run",
        "configuration": strict_phase_configuration(),
    }
    legacy_phase: dict[str, Any] = {
        "status": "not_run",
        "configuration": legacy_quantized_configuration(),
    }
    weights: dict[str, Any] = {
        "source": case.model.source,
        "id": case.model.weights_id,
        "state_sha256": None,
    }
    try:
        model, weights = _build_public_model(case, pretrained=config.pretrained)
        model.to(config.device).eval()
        generator = torch.Generator(device="cpu").manual_seed(case.capture_seed)
        capture_input = torch.randn(
            (1, 3, case.image_size, case.image_size), generator=generator
        ).to(config.device)
        scenario = Scenario(f"capture_{case.image_size}", (capture_input,), {})
        loader = _build_loader(
            case,
            model,
            imagenet_directory=imagenet_directory,
            target_map=target_map,
            indices=subset["indices"],
            config=config,
        )
        provider = _AcceptanceProvider(model, scenario, loader, config.device)

        strict_inspection = inspect_provider(
            provider,
            InspectionConfig(
                allow_fp32_fallback=False,
                verify=True,
                enable_fx=config.enable_fx,
                enable_export=config.enable_export,
                quantization_enabled=False,
                device=config.device,
            ),
        )
        write_artifacts(strict_directory, strict_inspection)
        strict_simulator = build_simulator(model, strict_inspection.plan, strict=True)
        strict_evaluation, strict_predictions = _evaluate_with_prediction_digest(
            model, strict_simulator, provider
        )
        write_artifacts(
            strict_directory,
            strict_inspection,
            evaluation=strict_evaluation,
        )
        accuracy = _accuracy_summary(strict_evaluation)
        equivalence = {
            "rtol": EQUIVALENCE_RTOL,
            "atol": EQUIVALENCE_ATOL,
            "output_structure": strict_inspection.verification.output_structure,
            "allclose": strict_inspection.verification.output_equivalence,
            "prediction_parity": accuracy["prediction_agreement"] == 1.0,
            "prediction_digest_parity": (
                strict_predictions.get("reference", {}).get("sha256")
                == strict_predictions.get("simulator", {}).get("sha256")
            ),
            "top1_parity": accuracy["reference_top1"] == accuracy["simulator_top1"],
            "top5_parity": accuracy["reference_top5"] == accuracy["simulator_top5"],
            "exactly_two_forwards_per_batch": (
                strict_evaluation.reference_forwards
                == strict_evaluation.simulator_forwards
                == strict_evaluation.batches
            ),
        }
        equivalence["passed"] = all(
            equivalence[key]
            for key in (
                "output_structure",
                "allclose",
                "prediction_parity",
                "prediction_digest_parity",
                "top1_parity",
                "top5_parity",
                "exactly_two_forwards_per_batch",
            )
        )
        strict_phase = {
            "status": "passed" if equivalence["passed"] else "failed",
            "configuration": strict_phase_configuration(),
            "support": _support_summary(strict_inspection),
            "accuracy": accuracy,
            "predictions": strict_predictions,
            "equivalence": equivalence,
            "artifacts": str(Path("cases") / case.case_id / STRICT_PHASE_NAME),
        }
        if not equivalence["passed"]:
            raise AcceptanceError("quantization-disabled equivalence gate failed")

        performance = _fast_metric_overhead(model, strict_simulator, provider, config)
        if performance["status"] != "passed":
            raise AcceptanceError(
                "fast metric bundle exceeded the 10% median-overhead gate"
            )

        # Release the second full model before constructing the independently
        # inspected legacy-quantized simulator.
        strict_simulator.close()
        strict_simulator = None
        torch.cuda.empty_cache()

        legacy_phase = _run_legacy_quantized_phase(
            model,
            provider,
            config,
            artifact_directory=legacy_directory,
            artifact_path=Path("cases") / case.case_id / LEGACY_PHASE_NAME,
        )
        if legacy_phase["status"] != "passed":
            raise AcceptanceError(legacy_phase.get("error", "legacy phase failed"))
        return {
            **case.to_dict(),
            "status": "passed",
            "weights": weights,
            # Compatibility aliases intentionally remain the strict,
            # quantization-disabled acceptance phase.
            "support": strict_phase["support"],
            "accuracy": accuracy,
            "equivalence": equivalence,
            "predictions": strict_predictions,
            "phases": {
                STRICT_PHASE_NAME: strict_phase,
                LEGACY_PHASE_NAME: legacy_phase,
            },
            "performance": {"fast_metric_overhead": performance},
            "artifacts": str(Path("cases") / case.case_id),
        }
    except Exception as exc:
        if strict_phase["status"] != "passed":
            strict_phase.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "support": (
                        _support_summary(strict_inspection)
                        if strict_inspection is not None
                        else {}
                    ),
                    "accuracy": (
                        _accuracy_summary(strict_evaluation)
                        if strict_evaluation is not None
                        else {}
                    ),
                    "predictions": strict_predictions,
                }
            )
        if strict_inspection is not None:
            try:
                write_artifacts(
                    strict_directory,
                    strict_inspection,
                    evaluation=strict_evaluation,
                )
            except Exception as artifact_exc:
                strict_phase["artifact_error"] = (
                    f"{type(artifact_exc).__name__}: {artifact_exc}"
                )
        if provider is not None and legacy_phase["status"] == "not_run":
            if strict_simulator is not None:
                strict_simulator.close()
                strict_simulator = None
                torch.cuda.empty_cache()
            legacy_phase = _run_legacy_quantized_phase(
                model,
                provider,
                config,
                artifact_directory=legacy_directory,
                artifact_path=Path("cases") / case.case_id / LEGACY_PHASE_NAME,
            )
        return {
            **case.to_dict(),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "weights": weights,
            "support": (
                _support_summary(strict_inspection)
                if strict_inspection is not None
                else {}
            ),
            "accuracy": (
                _accuracy_summary(strict_evaluation)
                if strict_evaluation is not None
                else {}
            ),
            "equivalence": strict_phase.get("equivalence", {"passed": False}),
            "predictions": strict_predictions,
            "phases": {
                STRICT_PHASE_NAME: strict_phase,
                LEGACY_PHASE_NAME: legacy_phase,
            },
            "performance": {"fast_metric_overhead": performance},
            "artifacts": str(Path("cases") / case.case_id),
        }
    finally:
        if strict_simulator is not None:
            strict_simulator.close()
        del (
            strict_simulator,
            strict_inspection,
            strict_evaluation,
            model,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_execution_gate(config: AcceptanceConfig) -> None:
    """Fail before importing optional model packages or loading checkpoints."""
    if os.environ.get(RUN_ENVIRONMENT_VARIABLE) != "1":
        raise AcceptanceError(
            f"Set {RUN_ENVIRONMENT_VARIABLE}=1 to enable the expensive live run"
        )
    if not config.pretrained:
        raise AcceptanceError("pass --pretrained to authorize checkpoint loading")
    if not config.device.startswith("cuda"):
        raise AcceptanceError("public-model acceptance requires --device cuda[:index]")
    if importlib.util.find_spec("torchvision") is None:
        raise AcceptanceError("torchvision is required for public-model acceptance")
    if importlib.util.find_spec("timm") is None:
        raise AcceptanceError("timm is required for public-model acceptance")
    import torch

    if not torch.cuda.is_available():
        raise AcceptanceError("CUDA is unavailable")
    try:
        device = torch.device(config.device)
        if device.type != "cuda":
            raise ValueError("not a CUDA device")
        if device.index is not None:
            torch.cuda.get_device_properties(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise AcceptanceError(f"invalid CUDA device {config.device!r}: {exc}") from exc
    _resolve_imagenet_split(config.imagenet_directory)
    if config.require_baseline and config.baseline_directory is None:
        raise AcceptanceError("--require-baseline needs --baseline-directory")


def _environment_manifest() -> dict[str, Any]:
    import torch

    result = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
    }
    import timm
    import torchvision

    result["torchvision"] = str(torchvision.__version__)
    result["timm"] = str(timm.__version__)
    return result


def _configure_determinism() -> None:
    # CUDA deterministic matmul requires this to be set before the first
    # cuBLAS workspace is created.  The harness calls this before model
    # construction or any CUDA forward.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _load_explanations(path: str | None) -> dict[str, Any]:
    return {} if path is None else _read_json(Path(path))


def _artifact_manifest(
    output_directory: Path,
    *,
    case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    paths = [
        output_directory / "acceptance.json",
        output_directory / "subset_manifest.json",
    ]
    diff = output_directory / "acceptance_diff.json"
    if diff.is_file():
        paths.append(diff)
    if case_ids is None:
        paths.extend(sorted((output_directory / "cases").glob("**/manifest.json")))
    else:
        for case_id in case_ids:
            for phase_name in (STRICT_PHASE_NAME, LEGACY_PHASE_NAME):
                manifest = (
                    output_directory / "cases" / case_id / phase_name / "manifest.json"
                )
                if manifest.is_file():
                    paths.append(manifest)
    files = {}
    for path in paths:
        payload = path.read_bytes()
        files[str(path.relative_to(output_directory))] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {"schema_version": HARNESS_SCHEMA_VERSION, "files": files}


def run_acceptance(
    config: AcceptanceConfig,
    *,
    cases: Sequence[AcceptanceCase] = PUBLIC_MODEL_MATRIX,
    case_runner: Callable[..., dict[str, Any]] = _execute_case,
) -> dict[str, Any]:
    """Execute the gated matrix and write reproducible acceptance artifacts."""
    if not isinstance(config, AcceptanceConfig):
        raise AcceptanceError("config must be an AcceptanceConfig")
    validate_execution_gate(config)
    output_directory = Path(config.output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    if config.baseline_directory is not None:
        baseline_directory = Path(config.baseline_directory).expanduser().resolve()
        if baseline_directory == output_directory:
            raise AcceptanceError("baseline and output directories must be different")
    else:
        stale_diff = output_directory / "acceptance_diff.json"
        if stale_diff.is_file() or stale_diff.is_symlink():
            stale_diff.unlink()
    imagenet_directory = _resolve_imagenet_split(config.imagenet_directory)
    class_index_path = _default_class_index_path()
    dataset_size, target_map, identities = _imagenet_inventory(
        imagenet_directory, class_index_path
    )
    _configure_determinism()
    if config.sample_count > dataset_size:
        raise AcceptanceError(
            f"sample_count {config.sample_count} exceeds dataset size {dataset_size}"
        )

    supplied_subset = config.subset_manifest
    if supplied_subset is None and config.baseline_directory is not None:
        baseline_subset = Path(config.baseline_directory) / "subset_manifest.json"
        if baseline_subset.is_file():
            supplied_subset = str(baseline_subset)
    if supplied_subset is None:
        subset = build_subset_manifest(
            dataset_size,
            config.sample_count,
            config.subset_seed,
            sample_identities=identities,
        )
    else:
        subset = load_subset_manifest(
            supplied_subset,
            dataset_size=dataset_size,
            sample_count=config.sample_count,
            seed=config.subset_seed,
            sample_identities=identities,
        )
    _write_json(output_directory / "subset_manifest.json", subset)

    rows = [
        case_runner(
            case,
            config,
            imagenet_directory=imagenet_directory,
            target_map=target_map,
            subset=subset,
            output_directory=output_directory,
        )
        for case in cases
    ]
    report: dict[str, Any] = {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "harness": "qbench-public-model-acceptance",
        "matrix": [case.to_dict() for case in cases],
        "config": config.reproducibility_dict(),
        "environment": _environment_manifest(),
        "class_index_sha256": hashlib.sha256(class_index_path.read_bytes()).hexdigest(),
        "subset": subset,
        "cases": rows,
        "all_cases_passed": all(row.get("status") == "passed" for row in rows),
    }

    diff = None
    if config.baseline_directory is not None:
        baseline_path = Path(config.baseline_directory) / "acceptance.json"
        baseline = _read_json(baseline_path)
        diff = compare_acceptance_reports(
            baseline,
            report,
            accuracy_tolerance=config.accuracy_tolerance,
            explanations=_load_explanations(config.explanations_file),
        )
        _write_json(output_directory / "acceptance_diff.json", diff)
    elif config.require_baseline:
        raise AcceptanceError("a baseline artifact directory is required")
    report["baseline_compared"] = diff is not None
    report["diff_passed"] = None if diff is None else diff["passed"]
    report["passed"] = report["all_cases_passed"] and (diff is None or diff["passed"])
    _write_json(output_directory / "acceptance.json", report)
    _write_json(
        output_directory / "acceptance_manifest.json",
        _artifact_manifest(
            output_directory,
            case_ids=[case.case_id for case in cases],
        ),
    )
    return report


def describe_harness() -> dict[str, Any]:
    return {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "harness": "qbench-public-model-acceptance",
        "live_gate": {
            "environment": f"{RUN_ENVIRONMENT_VARIABLE}=1",
            "pretrained_flag": True,
            "device": "cuda",
            "imagenet": "local ImageNet-1K validation directory",
        },
        "matrix": [case.to_dict() for case in PUBLIC_MODEL_MATRIX],
        "defaults": {
            "sample_count": DEFAULT_SAMPLE_COUNT,
            "subset_seed": DEFAULT_SUBSET_SEED,
            "subset_algorithm": SUBSET_ALGORITHM,
            "equivalence_rtol": EQUIVALENCE_RTOL,
            "equivalence_atol": EQUIVALENCE_ATOL,
            "imagenet_validation_size": IMAGENET_VALIDATION_SIZE,
            "imagenet_class_count": IMAGENET_CLASS_COUNT,
            "overhead_batches": 2,
            "overhead_repetitions": 7,
            "overhead_warmups": 1,
            "minimum_overhead_batch_invocations": MIN_OVERHEAD_BATCH_INVOCATIONS,
            "max_fast_metric_overhead": MAX_FAST_METRIC_OVERHEAD,
        },
        "phases": {
            STRICT_PHASE_NAME: strict_phase_configuration(),
            LEGACY_PHASE_NAME: legacy_quantized_configuration(),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute the gated pretrained GPU/ImageNet matrix",
    )
    parser.add_argument("--output-directory", default="public_model_acceptance")
    parser.add_argument(
        "--imagenet-directory",
        default=os.environ.get("QBENCH_IMAGENET_VAL", "/data/imagenet/val"),
    )
    parser.add_argument("--baseline-directory")
    parser.add_argument("--subset-manifest")
    parser.add_argument("--explanations-file")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--subset-seed", type=int, default=DEFAULT_SUBSET_SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--require-baseline", action="store_true")
    parser.add_argument("--accuracy-tolerance", type=float, default=0.0)
    parser.add_argument("--overhead-batches", type=int, default=2)
    parser.add_argument("--overhead-repetitions", type=int, default=7)
    parser.add_argument("--overhead-warmups", type=int, default=1)
    parser.add_argument("--no-fx", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.run:
        print(json.dumps(describe_harness(), indent=2, sort_keys=True))
        return 0
    try:
        config = AcceptanceConfig(
            output_directory=args.output_directory,
            imagenet_directory=args.imagenet_directory,
            baseline_directory=args.baseline_directory,
            subset_manifest=args.subset_manifest,
            explanations_file=args.explanations_file,
            sample_count=args.sample_count,
            subset_seed=args.subset_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            pretrained=args.pretrained,
            enable_fx=not args.no_fx,
            enable_export=not args.no_export,
            require_baseline=args.require_baseline,
            accuracy_tolerance=args.accuracy_tolerance,
            overhead_batches=args.overhead_batches,
            overhead_repetitions=args.overhead_repetitions,
            overhead_warmups=args.overhead_warmups,
        )
        report = run_acceptance(config)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
