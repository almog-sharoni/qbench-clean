"""Reproducible schema-v3 artifact writing."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from .provenance import package_version
from .schemas import InspectionResult, strict_json_safe


def _json_safe(value: Any) -> Any:
    """Keep the historical helper while sharing public schema normalization."""

    return strict_json_safe(value)


def _write_json(path: Path, value: Any):
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_artifacts(
    directory, result: InspectionResult, *, evaluation=None, state_dict=None
) -> Path:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for name, included in (
        ("evaluation.json", evaluation is not None),
        ("state.pt", state_dict is not None),
    ):
        stale = destination / name
        if not included and (stale.is_file() or stale.is_symlink()):
            stale.unlink()
    support_payload = dict(result.support)
    support_payload["schema_version"] = 3
    # Keep summary fields at the top level for v1/v2 readers while making the
    # authoritative dry-run and reproducibility data part of schema v3.
    support_payload["verification"] = result.verification.to_dict()
    support_payload["diagnostics"] = result.diagnostics
    plan_payload = result.plan.to_dict()
    plan_payload["schema_version"] = 3
    _write_json(destination / "support.json", support_payload)
    _write_json(destination / "plan.json", plan_payload)
    # ``gzip.open`` has no ``mtime`` argument.  Use GzipFile directly and omit
    # the source filename so identical ledgers produce identical bytes.
    with (destination / "operations.jsonl.gz").open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_stream, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                for operation in result.operations:
                    payload = _json_safe(operation.to_dict())
                    stream.write(
                        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
                    )
    if evaluation is not None:
        _write_json(
            destination / "evaluation.json",
            evaluation.to_dict() if hasattr(evaluation, "to_dict") else evaluation,
        )
    if state_dict is not None:
        import torch

        torch.save(state_dict, destination / "state.pt")
    names = ["support.json", "operations.jsonl.gz", "plan.json"]
    if evaluation is not None:
        names.append("evaluation.json")
    if state_dict is not None:
        names.append("state.pt")
    files = {}
    for name in names:
        payload = (destination / name).read_bytes()
        files[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    _write_json(
        destination / "manifest.json",
        {
            "schema_version": 3,
            "files": files,
            "fully_supported": result.fully_supported,
            "scenario_names": plan_payload["scenario_names"],
            "qbench_version": package_version(),
            "provenance": result.diagnostics.get("provenance", {}),
        },
    )
    return destination
