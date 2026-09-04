"""Command-line interface for the public QBench model workflow."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import write_artifacts
from .conversion import build_simulator
from .evaluation import EvaluationConfig, evaluate
from .inspection import inspect_model, inspect_provider
from .kernels import list_kernels, verify_kernels
from .providers import load_provider
from .schemas import (
    InspectionConfig,
    InspectionResult,
    QBenchError,
    strict_json_safe,
)


EXIT_SUPPORTED = 0
EXIT_ERROR = 1
EXIT_PARTIAL = 2


class _UsageError(QBenchError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    """Map invalid command lines to QBench's configuration-error exit code."""

    def error(self, message: str) -> None:
        raise _UsageError(message)


def _version() -> str:
    try:
        return metadata.version("qbench")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _read_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QBenchError(f"Could not read {label} {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise QBenchError(
            f"Invalid JSON in {label} {source} at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise QBenchError(f"{label} must contain a JSON object")
    return value


def _provider_spec(args: argparse.Namespace) -> str:
    positional = getattr(args, "provider_spec", None)
    option = getattr(args, "provider_option", None)
    if positional and option and positional != option:
        raise QBenchError(
            "Specify the provider once, either positionally or with --provider"
        )
    value = option or positional
    if not value:
        raise QBenchError("A provider in package.module:object form is required")
    return value


def _inspection_config(args: argparse.Namespace) -> InspectionConfig:
    values: dict[str, Any] = {}
    config_path = getattr(args, "config", None)
    if config_path:
        values.update(_read_object(config_path, label="inspection config"))
    overrides = {
        "allow_fp32_fallback": getattr(args, "allow_fp32_fallback", None),
        "verify": getattr(args, "verify", None),
        "capture_callsites": getattr(args, "capture_callsites", None),
        "enable_fx": getattr(args, "enable_fx", None),
        "enable_export": getattr(args, "enable_export", None),
        "quantization_enabled": getattr(args, "quantization_enabled", None),
        "device": getattr(args, "device", None),
        "conformance_directory": getattr(args, "conformance_directory", None),
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    policy = values.get("quantization_policy", {})
    if not isinstance(policy, Mapping):
        raise QBenchError(
            "quantization_policy in the inspection config must be an object"
        )
    policy = dict(policy)
    policy_overrides = {
        "quantization_type": getattr(args, "quantization_type", None),
        "quantization_bias": getattr(args, "quantization_bias", None),
        "input_quantization": getattr(args, "input_quantization", None),
        "weight_quantization": getattr(args, "weight_quantization", None),
        "output_quantization": getattr(args, "output_quantization", None),
        "quantize_first_layer": getattr(args, "quantize_first_layer", None),
        "quant_mode": getattr(args, "quant_mode", None),
        "chunk_size": getattr(args, "chunk_size", None),
        "weight_mode": getattr(args, "weight_mode", None),
        "weight_chunk_size": getattr(args, "weight_chunk_size", None),
    }
    policy.update(
        {key: value for key, value in policy_overrides.items() if value is not None}
    )
    if policy:
        values["quantization_policy"] = policy
    return InspectionConfig.coerce(values)


def _evaluation_config(args: argparse.Namespace) -> EvaluationConfig:
    values: dict[str, Any] = {}
    config_path = getattr(args, "evaluation_config", None)
    if config_path:
        values.update(_read_object(config_path, label="evaluation config"))
    overrides = {
        "metrics": getattr(args, "metrics", None),
        "max_batches": getattr(args, "max_batches", None),
        "task": getattr(args, "task", None),
        "latency_repetitions": getattr(args, "latency_repetitions", None),
        "retain_activations": getattr(args, "retain_activations", None),
        "activation_retention_max_elements": getattr(
            args, "activation_retention_max_elements", None
        ),
        "compliance_scan": getattr(args, "compliance_scan", None),
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    if isinstance(values.get("task"), str):
        values["task"] = values["task"].replace("-", "_")
    return EvaluationConfig.coerce(values)


def _result_exit(result: InspectionResult) -> int:
    if not result.support.get("capture_complete", False):
        return EXIT_ERROR
    fidelity = result.support.get("hardware_fidelity", {})
    if (
        isinstance(fidelity, Mapping)
        and fidelity.get("status") == "configuration_error"
    ):
        return EXIT_ERROR
    if (
        result.verification.attempted
        and not result.verification.succeeded
        and (
            result.support.get("replacement_coverage", False)
            or result.plan.allow_fp32_fallback
        )
    ):
        return EXIT_ERROR
    return EXIT_SUPPORTED if result.fully_supported else EXIT_PARTIAL


def _summary(
    command: str, directory: Path, result: InspectionResult, **extra: Any
) -> dict[str, Any]:
    payload = {
        "command": command,
        "artifact_directory": str(directory.resolve()),
        "fully_supported": result.fully_supported,
        "capture_complete": bool(result.support.get("capture_complete", False)),
        "replacement_coverage": bool(result.support.get("replacement_coverage", False)),
        "strict_realization": bool(result.support.get("strict_realization", False)),
        "operation_count": len(result.operations),
        "gap_count": len(result.support.get("gaps", [])),
    }
    payload.update(extra)
    return payload


def _emit(value: Any, destination: str | Path | None = None) -> None:
    text = (
        json.dumps(strict_json_safe(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    if destination is None:
        sys.stdout.write(text)
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _clone_provider_model(provider: Any, model: Any, *, purpose: str) -> Any:
    clone = provider.clone_model(model)
    if clone is model:
        raise QBenchError(f"ModelProvider.clone_model must isolate the {purpose} model")
    return clone


def _capture_inputs(provider: Any) -> list[Any]:
    scenarios = list(provider.capture_scenarios())
    if not scenarios:
        raise QBenchError("ModelProvider.capture_scenarios returned no scenarios")
    return scenarios


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _refresh_verdict(result: InspectionResult) -> None:
    verification = result.verification
    result.support["strict_realization"] = bool(
        verification.succeeded and verification.strict
    )
    result.support["routing_dry_run_verified"] = bool(
        verification.succeeded
        and verification.strict
        and verification.output_equivalence
    )
    result.support["quantized_execution_verified"] = bool(
        verification.quantized_execution
    )
    result.support["fully_supported"] = bool(
        result.support.get("capture_complete", False)
        and result.support.get("replacement_coverage", False)
        and verification.succeeded
        and verification.strict
        and not result.plan.allow_fp32_fallback
        and (not result.plan.quantization_enabled or verification.quantized_execution)
    )
    result.support["verdict"] = (
        "capture_failed"
        if not result.support.get("capture_complete", False)
        else (
            "fully_supported"
            if result.support["fully_supported"]
            else "partial_or_unsupported"
        )
    )


def _inspect(args: argparse.Namespace) -> int:
    # Some compatibility adapters still print progress messages.  Preserve
    # them for interactive users without corrupting the JSON result on stdout.
    with redirect_stdout(sys.stderr):
        provider_spec = _provider_spec(args)
        provider = load_provider(provider_spec)
        result = inspect_provider(provider, _inspection_config(args))
        result.diagnostics.setdefault("provenance", {})["provider_spec"] = provider_spec
    directory = write_artifacts(args.output_dir, result)
    _emit(_summary("inspect", directory, result))
    return _result_exit(result)


def _convert(args: argparse.Namespace) -> int:
    config = _inspection_config(args)
    with redirect_stdout(sys.stderr):
        provider_spec = _provider_spec(args)
        provider = load_provider(provider_spec)
        source = provider.build_model()
        scenarios = _capture_inputs(provider)
        inspection_model = _clone_provider_model(provider, source, purpose="inspection")
        result = inspect_model(inspection_model, scenarios, config)
        from .providers import provider_provenance

        result.diagnostics.setdefault("provenance", {}).update(
            provider_provenance(provider)
        )
        result.diagnostics["provenance"]["provider_spec"] = provider_spec

        simulator = None
        if result.support.get("capture_complete", False) and (
            config.allow_fp32_fallback or result.plan.strict_ready
        ):
            simulator_model = _clone_provider_model(
                provider, source, purpose="simulator"
            )
            simulator = build_simulator(
                simulator_model,
                result.plan,
                strict=not config.allow_fp32_fallback,
            )
            result.verification = simulator.verify(scenarios)
            _refresh_verdict(result)

    state_dict = None
    if args.export_state:
        if simulator is None or not result.verification.succeeded:
            # An unresolved strict plan is a partial analysis, not a state that
            # can truthfully be described as converted.
            directory = write_artifacts(args.output_dir, result)
            _emit(_summary("convert", directory, result, state_exported=False))
            return _result_exit(result)
        state_dict = simulator.state_dict()

    directory = write_artifacts(args.output_dir, result, state_dict=state_dict)
    _emit(_summary("convert", directory, result, state_exported=state_dict is not None))
    return _result_exit(result)


def _evaluate(args: argparse.Namespace) -> int:
    config = _inspection_config(args)
    with redirect_stdout(sys.stderr):
        provider_spec = _provider_spec(args)
        provider = load_provider(provider_spec)
        source = provider.build_model()
        scenarios = _capture_inputs(provider)
        inspection_model = _clone_provider_model(provider, source, purpose="inspection")
        result = inspect_model(inspection_model, scenarios, config)
        from .providers import provider_provenance

        result.diagnostics.setdefault("provenance", {}).update(
            provider_provenance(provider)
        )
        result.diagnostics["provenance"]["provider_spec"] = provider_spec

    if not result.support.get("capture_complete", False):
        directory = write_artifacts(args.output_dir, result)
        _emit(_summary("evaluate", directory, result, evaluated=False))
        return EXIT_ERROR
    if not config.allow_fp32_fallback and not result.plan.strict_ready:
        directory = write_artifacts(args.output_dir, result)
        _emit(_summary("evaluate", directory, result, evaluated=False))
        return EXIT_PARTIAL

    with redirect_stdout(sys.stderr):
        reference = _clone_provider_model(provider, source, purpose="reference")
        simulator_model = _clone_provider_model(provider, source, purpose="simulator")
        simulator = build_simulator(
            simulator_model,
            result.plan,
            strict=not config.allow_fp32_fallback,
        )
        result.verification = simulator.verify(scenarios)
        _refresh_verdict(result)
    if not result.verification.succeeded:
        directory = write_artifacts(args.output_dir, result)
        _emit(_summary("evaluate", directory, result, evaluated=False))
        return _result_exit(result)
    with redirect_stdout(sys.stderr):
        report = evaluate(reference, simulator, provider, _evaluation_config(args))
    directory = write_artifacts(args.output_dir, result, evaluation=report)
    _emit(
        _summary(
            "evaluate",
            directory,
            result,
            evaluated=True,
            batches=report.batches,
            metrics=report.metrics,
        )
    )
    return _result_exit(result)


def _kernels_list(args: argparse.Namespace) -> int:
    rows = list_kernels()
    payload = {"schema_version": 3, "count": len(rows), "kernels": rows}
    _emit(payload, args.output)
    return EXIT_SUPPORTED


def _kernels_verify(args: argparse.Namespace) -> int:
    report = verify_kernels(args.vector_directory)
    _emit(report, args.output)
    if report.get("status") == "passed":
        return EXIT_SUPPORTED
    if report.get("status") == "configuration_error":
        return EXIT_ERROR
    return EXIT_PARTIAL


def _add_provider(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "provider_spec",
        nargs="?",
        metavar="PROVIDER",
        help="trusted provider in package.module:object form",
    )
    parser.add_argument(
        "--provider",
        dest="provider_option",
        metavar="PACKAGE.MODULE:OBJECT",
        help="trusted provider (alternative to the positional PROVIDER)",
    )


def _add_inspection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", metavar="FILE", help="JSON InspectionConfig object")
    parser.add_argument(
        "--allow-fp32-fallback",
        action="store_true",
        default=None,
        help="allow unresolved operations as FP32 fallbacks (the verdict remains partial)",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run the converted routing dry run (default: enabled)",
    )
    parser.add_argument(
        "--capture-callsites",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="capture the first custom-code callsite (default: enabled)",
    )
    fx = parser.add_mutually_exclusive_group()
    fx.add_argument("--enable-fx", dest="enable_fx", action="store_true", default=None)
    fx.add_argument("--no-fx", dest="enable_fx", action="store_false")
    export = parser.add_mutually_exclusive_group()
    export.add_argument(
        "--enable-export", dest="enable_export", action="store_true", default=None
    )
    export.add_argument("--no-export", dest="enable_export", action="store_false")
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument(
        "--quantization-enabled",
        dest="quantization_enabled",
        action="store_true",
        default=None,
    )
    quantization.add_argument(
        "--quantization-disabled",
        dest="quantization_enabled",
        action="store_false",
    )
    parser.add_argument(
        "--quantization-type",
        metavar="FORMAT",
        help="simulator format such as fp8_e4m3",
    )
    parser.add_argument(
        "--quantization-bias",
        type=int,
        help="reserved schema field; custom exponent bias is not implemented",
    )
    parser.add_argument(
        "--input-quantization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable maintained input-boundary quantization",
    )
    parser.add_argument(
        "--weight-quantization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable maintained weight quantization",
    )
    parser.add_argument(
        "--output-quantization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable maintained output-boundary quantization",
    )
    parser.add_argument(
        "--quantize-first-layer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include the first maintained module input boundary",
    )
    parser.add_argument(
        "--quant-mode",
        choices=("tensor", "chunk", "channel"),
        help="input activation quantization mode",
    )
    parser.add_argument(
        "--chunk-size",
        type=_positive_int,
        help="input activation chunk size (chunk mode requires 128)",
    )
    parser.add_argument(
        "--weight-mode",
        choices=("tensor", "chunk", "channel"),
        help="weight quantization mode",
    )
    parser.add_argument(
        "--weight-chunk-size",
        type=_positive_int,
        help="weight chunk size (chunk mode requires 128)",
    )
    parser.add_argument(
        "--device", help="inspection device recorded in the config (default: cpu)"
    )
    parser.add_argument(
        "--conformance-vectors",
        dest="conformance_directory",
        metavar="DIRECTORY",
        help="portable hardware conformance-vector bundle",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="qbench-artifacts",
        metavar="DIRECTORY",
        help="artifact directory (default: qbench-artifacts)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="qbench",
        description="Inspect, convert, and evaluate eager PyTorch models with QBench.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser(
        "inspect", help="capture operations and write support artifacts"
    )
    _add_provider(inspect_parser)
    _add_inspection_options(inspect_parser)
    inspect_parser.set_defaults(handler=_inspect)

    convert_parser = commands.add_parser(
        "convert", help="build and verify a simulator plan"
    )
    _add_provider(convert_parser)
    _add_inspection_options(convert_parser)
    convert_parser.add_argument(
        "--export-state",
        action="store_true",
        help="write the converted simulator state as state.pt",
    )
    convert_parser.set_defaults(handler=_convert)

    evaluate_parser = commands.add_parser(
        "evaluate", help="run paired reference/simulator evaluation"
    )
    _add_provider(evaluate_parser)
    _add_inspection_options(evaluate_parser)
    evaluate_parser.add_argument(
        "--evaluation-config", metavar="FILE", help="JSON EvaluationConfig object"
    )
    evaluate_parser.add_argument(
        "--metrics", choices=("fast", "detailed"), default=None
    )
    evaluate_parser.add_argument("--max-batches", type=_positive_int, default=None)
    evaluate_parser.add_argument(
        "--latency-repetitions",
        type=_positive_int,
        default=None,
        help="paired timing forwards per batch in detailed mode",
    )
    evaluate_parser.add_argument(
        "--retain-activations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="retain a capped activation sample in detailed evaluation artifacts",
    )
    evaluate_parser.add_argument(
        "--activation-retention-max-elements",
        type=_positive_int,
        default=None,
        help="global retained-activation element cap",
    )
    evaluate_parser.add_argument(
        "--compliance-scan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="check quantized input, weight, and output values in detailed mode",
    )
    evaluate_parser.add_argument(
        "--task",
        choices=("generic", "classification", "language-modeling", "feature-matching"),
        default=None,
    )
    evaluate_parser.set_defaults(handler=_evaluate)

    kernels_parser = commands.add_parser(
        "kernels", help="inspect maintained kernel capabilities"
    )
    kernel_commands = kernels_parser.add_subparsers(
        dest="kernels_command", required=True
    )
    list_parser = kernel_commands.add_parser(
        "list", help="list maintained KernelSpec records"
    )
    list_parser.add_argument(
        "-o", "--output", metavar="FILE", help="write JSON instead of stdout"
    )
    list_parser.set_defaults(handler=_kernels_list)
    verify_parser = kernel_commands.add_parser(
        "verify", help="verify portable conformance-vector checksums"
    )
    verify_parser.add_argument("vector_directory", metavar="VECTOR_DIRECTORY")
    verify_parser.add_argument(
        "-o", "--output", metavar="FILE", help="write JSON instead of stdout"
    )
    verify_parser.set_defaults(handler=_kernels_verify)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return int(args.handler(args))
    except KeyboardInterrupt:
        sys.stderr.write("qbench: interrupted\n")
        return EXIT_ERROR
    except (QBenchError, OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"qbench: error: {exc}\n")
        return EXIT_ERROR
    except Exception as exc:  # CLI boundary: execution failures use exit code 1.
        sys.stderr.write(f"qbench: error: {type(exc).__name__}: {exc}\n")
        return EXIT_ERROR


__all__ = [
    "EXIT_SUPPORTED",
    "EXIT_ERROR",
    "EXIT_PARTIAL",
    "build_parser",
    "main",
]
