"""CLI contract tests for artifacts and documented exit statuses."""

from __future__ import annotations

import json

import torch

from qbench import cli
from qbench.providers import DirectObjectProvider
from qbench.schemas import QBenchError, Scenario, VerificationResult


def _provider(model: torch.nn.Module) -> DirectObjectProvider:
    return DirectObjectProvider(model, Scenario("capture", (torch.ones(2),)))


def test_inspect_cli_writes_required_artifacts_and_uses_supported_partial_codes(
    monkeypatch, tmp_path, capsys
):
    provider = _provider(torch.nn.ReLU())

    def noisy_provider(_spec):
        print("provider diagnostic")
        return provider

    monkeypatch.setattr(cli, "load_provider", noisy_provider)
    supported_dir = tmp_path / "supported"

    status = cli.main(
        [
            "inspect",
            "tests:provider",
            "--no-fx",
            "--no-export",
            "-o",
            str(supported_dir),
        ]
    )

    assert status == cli.EXIT_SUPPORTED
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert "provider diagnostic" not in captured.out
    assert "provider diagnostic" in captured.err
    assert summary["fully_supported"] is True
    assert {path.name for path in supported_dir.iterdir()} == {
        "manifest.json",
        "support.json",
        "operations.jsonl.gz",
        "plan.json",
    }

    class Unsupported(torch.nn.Module):
        def forward(self, value):
            return torch.sin(value)

    monkeypatch.setattr(cli, "load_provider", lambda _spec: _provider(Unsupported()))
    partial_dir = tmp_path / "partial"
    status = cli.main(
        [
            "inspect",
            "tests:provider",
            "--no-fx",
            "--no-export",
            "-o",
            str(partial_dir),
        ]
    )

    assert status == cli.EXIT_PARTIAL
    summary = json.loads(capsys.readouterr().out)
    assert summary["capture_complete"] is True
    assert summary["fully_supported"] is False


def test_cpu_quantized_inspect_cli_is_successfully_analyzed_partial(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        cli,
        "load_provider",
        lambda _spec: _provider(torch.nn.Linear(2, 2)),
    )
    destination = tmp_path / "cpu-quantized"

    status = cli.main(
        [
            "inspect",
            "tests:provider",
            "--no-fx",
            "--no-export",
            "--quantization-enabled",
            "--device",
            "cpu",
            "-o",
            str(destination),
        ]
    )

    assert status == cli.EXIT_PARTIAL
    summary = json.loads(capsys.readouterr().out)
    support = json.loads((destination / "support.json").read_text(encoding="utf-8"))
    assert summary["capture_complete"] is True
    assert summary["strict_realization"] is True
    assert support["verification"]["succeeded"] is True
    assert support["routing_dry_run_verified"] is True
    assert support["quantized_execution_verified"] is False


def test_cli_configuration_failure_uses_error_code(monkeypatch, capsys):
    def fail(_spec):
        raise QBenchError("invalid provider")

    monkeypatch.setattr(cli, "load_provider", fail)

    assert cli.main(["inspect", "broken:provider"]) == cli.EXIT_ERROR
    assert "invalid provider" in capsys.readouterr().err


def test_attempted_verification_failure_is_error_even_with_partial_coverage():
    class Unsupported(torch.nn.Module):
        def forward(self, value):
            return torch.sin(value)

    result = cli.inspect_model(
        Unsupported(),
        Scenario("fallback", (torch.ones(2),)),
        {
            "allow_fp32_fallback": True,
            "enable_fx": False,
            "enable_export": False,
        },
    )
    assert result.support["replacement_coverage"] is False
    assert result.verification.attempted is True
    result.verification.succeeded = False

    assert cli._result_exit(result) == cli.EXIT_ERROR


def test_inspect_conformance_configuration_error_uses_error_code(
    monkeypatch, tmp_path, capsys
):
    provider = _provider(torch.nn.ReLU())
    inspect_provider = cli.inspect_provider

    def inspect_with_bad_conformance(selected, config):
        result = inspect_provider(selected, config)
        result.support["hardware_fidelity"] = {
            "status": "configuration_error",
            "errors": ["invalid conformance manifest"],
        }
        return result

    monkeypatch.setattr(cli, "load_provider", lambda _spec: provider)
    monkeypatch.setattr(cli, "inspect_provider", inspect_with_bad_conformance)

    status = cli.main(
        [
            "inspect",
            "tests:provider",
            "--no-fx",
            "--no-export",
            "--conformance-vectors",
            str(tmp_path / "vectors"),
            "-o",
            str(tmp_path / "artifacts"),
        ]
    )

    assert status == cli.EXIT_ERROR
    assert json.loads(capsys.readouterr().out)["fully_supported"] is True


def test_kernel_manifest_configuration_error_uses_error_code(tmp_path, capsys):
    (tmp_path / "manifest.json").write_text("not-json", encoding="utf-8")

    assert cli.main(["kernels", "verify", str(tmp_path)]) == cli.EXIT_ERROR
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "configuration_error"


def test_convert_cli_execution_failure_uses_error_code(monkeypatch, tmp_path, capsys):
    provider = _provider(torch.nn.ReLU())

    class FailedSimulator:
        def verify(self, _scenarios):
            return VerificationResult(
                attempted=True,
                succeeded=False,
                strict=True,
                errors=["converted dry run exploded"],
            )

    monkeypatch.setattr(cli, "load_provider", lambda _spec: provider)
    monkeypatch.setattr(
        cli, "build_simulator", lambda *_args, **_kwargs: FailedSimulator()
    )

    status = cli.main(
        [
            "convert",
            "tests:provider",
            "--no-fx",
            "--no-export",
            "-o",
            str(tmp_path / "failed-conversion"),
        ]
    )

    assert status == cli.EXIT_ERROR
    assert json.loads(capsys.readouterr().out)["fully_supported"] is False


def test_convert_cli_success_exports_verified_state(monkeypatch, tmp_path, capsys):
    model = torch.nn.Linear(2, 2).eval()
    provider = _provider(model)
    monkeypatch.setattr(cli, "load_provider", lambda _spec: provider)
    destination = tmp_path / "converted"

    status = cli.main(
        [
            "convert",
            "tests:provider",
            "--no-fx",
            "--no-export",
            "--export-state",
            "-o",
            str(destination),
        ]
    )

    assert status == cli.EXIT_SUPPORTED
    summary = json.loads(capsys.readouterr().out)
    assert summary["command"] == "convert"
    assert summary["fully_supported"] is True
    assert summary["state_exported"] is True
    assert {path.name for path in destination.iterdir()} == {
        "manifest.json",
        "support.json",
        "operations.jsonl.gz",
        "plan.json",
        "state.pt",
    }
    support = json.loads((destination / "support.json").read_text(encoding="utf-8"))
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    state = torch.load(destination / "state.pt", map_location="cpu", weights_only=True)
    assert support["schema_version"] == 3
    assert support["verification"]["succeeded"] is True
    assert support["fully_supported"] is True
    assert set(manifest["files"]) == {
        "support.json",
        "operations.jsonl.gz",
        "plan.json",
        "state.pt",
    }
    assert set(state) >= {"weight", "bias"}
    torch.testing.assert_close(state["weight"], model.weight)
    torch.testing.assert_close(state["bias"], model.bias)


def test_evaluate_cli_success_writes_fast_metrics(monkeypatch, tmp_path, capsys):
    scenario = Scenario("capture", (torch.tensor([-1.0, 0.0, 2.0]),))
    provider = DirectObjectProvider(torch.nn.ReLU().eval(), scenario, loader=[scenario])
    monkeypatch.setattr(cli, "load_provider", lambda _spec: provider)
    destination = tmp_path / "evaluated"

    status = cli.main(
        [
            "evaluate",
            "tests:provider",
            "--no-fx",
            "--no-export",
            "--metrics",
            "fast",
            "-o",
            str(destination),
        ]
    )

    assert status == cli.EXIT_SUPPORTED
    summary = json.loads(capsys.readouterr().out)
    assert summary["command"] == "evaluate"
    assert summary["fully_supported"] is True
    assert summary["evaluated"] is True
    assert summary["batches"] == 1
    assert summary["metrics"]["perfect_match"] is True
    assert {path.name for path in destination.iterdir()} == {
        "manifest.json",
        "support.json",
        "operations.jsonl.gz",
        "plan.json",
        "evaluation.json",
    }
    evaluation = json.loads(
        (destination / "evaluation.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert evaluation["batches"] == 1
    assert evaluation["reference_forwards"] == 1
    assert evaluation["simulator_forwards"] == 1
    assert evaluation["metrics"]["mae"] == 0.0
    assert evaluation["metrics"]["mse"] == 0.0
    assert evaluation["metrics"]["perfect_match"] is True
    assert set(manifest["files"]) == {
        "support.json",
        "operations.jsonl.gz",
        "plan.json",
        "evaluation.json",
    }
