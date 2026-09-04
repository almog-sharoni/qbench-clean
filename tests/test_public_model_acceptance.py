# ruff: noqa: E402
from __future__ import annotations

import builtins
import copy
import hashlib
import json
import os

import pytest

import qbench.validation.public_models as acceptance
from qbench.validation.public_models import (
    DEFAULT_IMAGE_SIZES,
    LEGACY_PHASE_NAME,
    LEGACY_QUANTIZATION_POLICY,
    MAX_FAST_METRIC_OVERHEAD,
    MIN_OVERHEAD_BATCH_INVOCATIONS,
    PUBLIC_MODEL_MATRIX,
    RUN_ENVIRONMENT_VARIABLE,
    AcceptanceConfig,
    AcceptanceError,
    _default_class_index_path,
    _AcceptanceProvider,
    _artifact_manifest,
    build_subset_manifest,
    compare_acceptance_directories,
    compare_acceptance_reports,
    deterministic_subset_indices,
    legacy_quantized_configuration,
    load_subset_manifest,
    main,
    run_acceptance,
    strict_phase_configuration,
    summarize_fast_metric_overhead,
    validate_execution_gate,
    validate_subset_manifest,
)


def test_public_model_matrix_is_complete_pinned_and_json_serializable():
    assert len(PUBLIC_MODEL_MATRIX) == 6
    assert {(case.model.key, case.image_size) for case in PUBLIC_MODEL_MATRIX} == {
        (model, size)
        for model in ("resnet18", "vit_b_16", "mobilevit_s")
        for size in DEFAULT_IMAGE_SIZES
    }
    assert len({case.case_id for case in PUBLIC_MODEL_MATRIX}) == 6
    assert all(case.model.weights_id != "DEFAULT" for case in PUBLIC_MODEL_MATRIX)
    mobilevit = [
        case for case in PUBLIC_MODEL_MATRIX if case.model.key == "mobilevit_s"
    ]
    assert {case.model.model_name for case in mobilevit} == {"mobilevit_s.cvnets_in1k"}
    vit_256 = next(
        case
        for case in PUBLIC_MODEL_MATRIX
        if case.model.key == "vit_b_16" and case.image_size == 256
    )
    assert vit_256.to_dict()["capture_shape"] == [1, 3, 256, 256]
    json.dumps([case.to_dict() for case in PUBLIC_MODEL_MATRIX])

    class_index = json.loads(_default_class_index_path().read_text(encoding="utf-8"))
    assert len(class_index) == 1000


def test_imagenet_class_index_is_a_packaged_resource():
    from importlib.resources import files

    resource = files("qbench.data").joinpath("imagenet_class_index.json")
    payload = resource.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "b090d218c425be1c056c8cb9dae307b70e349714712013b5b11265bd2cb9d80a"
    )
    assert json.loads(payload)["0"][0] == "n01440764"


def test_subset_manifest_freezes_seed_indices_and_sample_identities(tmp_path):
    identities = {
        index: {"relative_path": f"n{index:08d}/sample.JPEG", "target": index % 5}
        for index in range(50)
    }
    first = build_subset_manifest(50, 8, 17, sample_identities=identities)
    second = build_subset_manifest(50, 8, 17, sample_identities=identities)
    assert first == second
    assert first["indices"] == deterministic_subset_indices(50, 8, 17)
    assert len(first["indices"]) == len(set(first["indices"])) == 8
    assert first["indices"] != deterministic_subset_indices(50, 8, 18)
    assert [row["index"] for row in first["samples"]] == first["indices"]

    manifest_path = tmp_path / "subset_manifest.json"
    manifest_path.write_text(json.dumps(first), encoding="utf-8")
    loaded = load_subset_manifest(
        manifest_path,
        dataset_size=50,
        sample_count=8,
        seed=17,
        sample_identities=identities,
    )
    assert loaded == first

    tampered = dict(first)
    tampered["indices"] = list(reversed(first["indices"]))
    with pytest.raises(AcceptanceError, match="indices"):
        validate_subset_manifest(tampered)

    changed_identities = dict(identities)
    changed_identities[first["indices"][0]] = {
        "relative_path": "changed.JPEG",
        "target": 0,
    }
    with pytest.raises(AcceptanceError, match="identity changed"):
        validate_subset_manifest(first, sample_identities=changed_identities)


def _case(
    *,
    mappings=("module:layer=>module:qbench.QuantLinear",),
    operations=("aten::linear",),
    verdict="fully_supported",
    digest="weights-a",
    reference_top1=0.75,
    simulator_top1=0.75,
    legacy=None,
):
    row = {
        "case_id": "torchvision-resnet18-224",
        "status": "passed",
        "weights": {"state_sha256": digest},
        "support": {
            "verdict": verdict,
            "planned_mappings": list(mappings),
            "operation_schemas": list(operations),
            "operation_counts": {schema: 1 for schema in operations},
            "operation_ledger_counts": {
                f"capture_224|layer|{schema}|default|kernel|supported": 1
                for schema in operations
            },
        },
        "accuracy": {
            "reference_top1": reference_top1,
            "reference_top5": 0.95,
            "simulator_top1": simulator_top1,
            "simulator_top5": 0.95,
            "prediction_agreement": 1.0,
        },
        "equivalence": {"passed": True},
    }
    if legacy is not None:
        row["phases"] = {LEGACY_PHASE_NAME: legacy}
    return row


def _legacy_result(
    *,
    simulator_digest="simulator-a",
    simulator_top1=0.70,
    simulator_top5=0.92,
    configuration=None,
):
    return {
        "status": "passed",
        "configuration": (
            legacy_quantized_configuration() if configuration is None else configuration
        ),
        "predictions": {
            "reference": {"sha256": "reference-a", "predictions": 128},
            "simulator": {"sha256": simulator_digest, "predictions": 128},
        },
        "accuracy": {
            "reference_top1": 0.75,
            "reference_top5": 0.95,
            "simulator_top1": simulator_top1,
            "simulator_top5": simulator_top5,
            "prediction_agreement": 0.80,
        },
    }


def _report(case, *, seed=17):
    return {
        "schema_version": 1,
        "subset": {
            "algorithm": "sha256-rank-v1",
            "dataset_size": 50_000,
            "sample_count": 128,
            "seed": seed,
            "indices": list(range(128)),
        },
        "cases": [case],
    }


def test_support_accuracy_diff_allows_additive_discovery_and_conservative_verdict():
    baseline = _report(_case())
    current = _report(
        _case(
            mappings=(
                "module:layer=>module:qbench.QuantLinear",
                "schema:aten::relu=>activation",
            ),
            operations=("aten::linear", "aten::relu"),
            verdict="partial_or_unsupported",
        )
    )
    diff = compare_acceptance_reports(baseline, current)
    assert diff["passed"] is True
    assert diff["regressions"] == []
    assert diff["changes"][0]["mappings"]["removed"] == []
    assert diff["changes"][0]["operations"]["added"] == ["aten::relu"]
    assert diff["changes"][0]["verdict"]["after"] == "partial_or_unsupported"


def test_support_accuracy_diff_rejects_disappearing_mappings_and_unexplained_accuracy(
    tmp_path,
):
    baseline = _report(_case())
    current = _report(
        _case(
            mappings=(),
            operations=(),
            digest="weights-b",
            reference_top1=0.70,
            simulator_top1=0.70,
        )
    )
    diff = compare_acceptance_reports(
        baseline,
        current,
        explanations={
            "torchvision-resnet18-224": {
                "weights.state_sha256": "documented checkpoint refresh",
                "accuracy.reference_top1": "documented checkpoint refresh",
                "accuracy.simulator_top1": "documented checkpoint refresh",
            }
        },
    )
    codes = {row["code"] for row in diff["regressions"]}
    assert codes == {
        "support.mappings_removed",
        "support.operations_disappeared",
        "support.operation_counts_decreased",
        "support.raw_operations_disappeared",
    }
    assert diff["passed"] is False

    baseline_dir = tmp_path / "baseline"
    current_dir = tmp_path / "current"
    baseline_dir.mkdir()
    current_dir.mkdir()
    (baseline_dir / "acceptance.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )
    (current_dir / "acceptance.json").write_text(json.dumps(current), encoding="utf-8")
    emitted = compare_acceptance_directories(
        baseline_dir, current_dir, output_path=current_dir / "acceptance_diff.json"
    )
    assert emitted["passed"] is False
    payload = (current_dir / "acceptance_diff.json").read_bytes()
    assert json.loads(payload)["regression_count"] == 7
    assert hashlib.sha256(payload).hexdigest()


def test_legacy_quantized_configuration_is_explicit_valid_and_isolated():
    from qbench import QuantizationPolicy

    configuration = legacy_quantized_configuration()
    assert configuration["name"] == LEGACY_PHASE_NAME
    assert configuration["quantization_enabled"] is True
    assert configuration["allow_fp32_fallback"] is True
    assert configuration["simulator_strict"] is False
    assert configuration["eligible_for_fully_supported_verdict"] is False
    assert configuration["requires_actual_quantized_execution"] is True
    assert configuration["quantization_policy"] == LEGACY_QUANTIZATION_POLICY
    assert (
        QuantizationPolicy.coerce(configuration["quantization_policy"]).to_dict()
        == LEGACY_QUANTIZATION_POLICY
    )
    configuration["quantization_policy"]["layer_config"]["changed"] = {}
    assert legacy_quantized_configuration()["quantization_policy"]["layer_config"] == {}

    strict = strict_phase_configuration()
    assert strict["quantization_enabled"] is False
    assert strict["allow_fp32_fallback"] is False
    assert strict["simulator_strict"] is True


def test_legacy_plan_derivation_excludes_modern_functional_routes():
    from types import SimpleNamespace

    plan = SimpleNamespace(
        kernels={
            "module:layer": {"classification": "quantized"},
            "schema:aten::add.Tensor": {"classification": "quantized"},
            "schema:aten::view": {"classification": "structural"},
        },
        unresolved_schemas=["custom::missing"],
        allow_fp32_fallback=False,
    )
    derived, metadata = acceptance._legacy_module_only_plan(plan)
    assert set(derived.kernels) == {"module:layer"}
    assert derived.unresolved_schemas == ["aten::add.Tensor", "custom::missing"]
    assert derived.allow_fp32_fallback is True
    assert metadata == {
        "derivation": "module-routes-only-v1",
        "excluded_schema_routes": ["schema:aten::add.Tensor", "schema:aten::view"],
        "fp32_compute_schemas": ["aten::add.Tensor"],
        "native_structural_schemas": ["aten::view"],
    }
    assert set(plan.kernels) == {
        "module:layer",
        "schema:aten::add.Tensor",
        "schema:aten::view",
    }
    assert plan.allow_fp32_fallback is False


def test_legacy_quantized_diff_is_separate_and_prediction_exact():
    baseline = _report(_case(legacy=_legacy_result()))
    current = copy.deepcopy(baseline)
    assert compare_acceptance_reports(baseline, current)["passed"] is True

    changed = _report(
        _case(
            legacy=_legacy_result(
                simulator_digest="simulator-b",
                simulator_top1=0.69,
            )
        )
    )
    diff = compare_acceptance_reports(baseline, changed)
    assert diff["passed"] is False
    codes = {row["code"] for row in diff["regressions"]}
    assert "legacy_quantized.predictions.simulator.sha256" in codes
    assert "legacy_quantized.accuracy.simulator_top1" in codes
    assert not any(code.startswith("support.") for code in codes)

    missing = compare_acceptance_reports(baseline, _report(_case()))
    assert "legacy_quantized.missing" in {row["code"] for row in missing["regressions"]}
    added = compare_acceptance_reports(_report(_case()), baseline)
    assert added["passed"] is True
    assert {row["code"] for row in added["changes"]} == {"legacy_quantized.added"}


def test_strict_prediction_digest_must_remain_exact_across_baseline():
    baseline_case = _case()
    baseline_case["predictions"] = {
        "reference": {"sha256": "reference-a", "predictions": 128},
        "simulator": {"sha256": "simulator-a", "predictions": 128},
    }
    current_case = copy.deepcopy(baseline_case)
    current_case["predictions"]["simulator"]["sha256"] = "simulator-b"
    diff = compare_acceptance_reports(
        _report(baseline_case),
        _report(current_case),
    )
    assert diff["passed"] is False
    assert {row["code"] for row in diff["regressions"]} == {
        "strict.predictions.simulator.sha256"
    }

    added_case = _case()
    added_case["predictions"] = baseline_case["predictions"]
    assert (
        compare_acceptance_reports(_report(_case()), _report(added_case))["passed"]
        is True
    )


def test_legacy_quantized_diff_can_explain_configuration_change():
    baseline = _report(_case(legacy=_legacy_result()))
    changed_configuration = legacy_quantized_configuration()
    changed_configuration["quantization_policy"]["output_quantization"] = True
    current = _report(_case(legacy=_legacy_result(configuration=changed_configuration)))
    code = "legacy_quantized.configuration"
    unexplained = compare_acceptance_reports(baseline, current)
    assert code in {row["code"] for row in unexplained["regressions"]}
    explained = compare_acceptance_reports(
        baseline,
        current,
        explanations={
            "torchvision-resnet18-224": {code: "intentional compatibility refresh"}
        },
    )
    assert explained["passed"] is True


def test_fast_metric_overhead_summary_enforces_median_and_forward_budget():
    counts = [2, 2, 2, 2, 2]
    passed = summarize_fast_metric_overhead(
        [1.00, 0.98, 1.02, 1.01, 0.99],
        [1.05, 1.03, 1.07, 1.06, 1.04],
        batches=2,
        bare_reference_forwards=counts,
        bare_simulator_forwards=counts,
        reference_forwards=counts,
        simulator_forwards=counts,
    )
    assert passed["status"] == "passed"
    assert passed["overhead_fraction"] == pytest.approx(0.05)
    assert passed["maximum_overhead_fraction"] == MAX_FAST_METRIC_OVERHEAD
    assert passed["exactly_two_forwards_per_batch"] is True
    assert passed["retains_intermediate_activations"] is False

    slow = summarize_fast_metric_overhead(
        [1.0, 1.0, 1.0],
        [1.11, 1.11, 1.11],
        batches=2,
        bare_reference_forwards=counts[:3],
        bare_simulator_forwards=counts[:3],
        reference_forwards=counts[:3],
        simulator_forwards=counts[:3],
    )
    assert slow["status"] == "failed"
    wrong_counts = summarize_fast_metric_overhead(
        [1.0, 1.0, 1.0],
        [1.01, 1.01, 1.01],
        batches=2,
        bare_reference_forwards=counts[:3],
        bare_simulator_forwards=counts[:3],
        reference_forwards=[2, 1, 2],
        simulator_forwards=counts[:3],
    )
    assert wrong_counts["status"] == "failed"


def test_prediction_digest_records_outputs_from_the_same_dual_forward_loop():
    torch = pytest.importorskip("torch")
    provider = _AcceptanceProvider(None, None, (), "cpu")
    provider.begin_prediction_capture()
    for _ in range(2):
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        provider.select_metric_output(logits)
        provider.select_metric_output(logits.clone())
    matching = provider.end_prediction_capture(2)
    assert matching["reference"] == matching["simulator"]
    assert matching["reference"]["predictions"] == 4

    provider.begin_prediction_capture()
    provider.select_metric_output(torch.tensor([[0.1, 0.9]]))
    provider.select_metric_output(torch.tensor([[0.9, 0.1]]))
    different = provider.end_prediction_capture(1)
    assert different["reference"]["sha256"] != different["simulator"]["sha256"]


def test_artifact_manifest_includes_nested_phase_manifests(tmp_path):
    (tmp_path / "acceptance.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "subset_manifest.json").write_text("{}\n", encoding="utf-8")
    strict = tmp_path / "cases" / "case" / "strict_quantization_disabled"
    legacy = tmp_path / "cases" / "case" / "legacy_quantized_gpu"
    strict.mkdir(parents=True)
    legacy.mkdir(parents=True)
    (strict / "manifest.json").write_text('{"phase": "strict"}\n', encoding="utf-8")
    (legacy / "manifest.json").write_text('{"phase": "legacy"}\n', encoding="utf-8")
    manifest = _artifact_manifest(tmp_path)
    assert set(manifest["files"]) == {
        "acceptance.json",
        "subset_manifest.json",
        "cases/case/strict_quantization_disabled/manifest.json",
        "cases/case/legacy_quantized_gpu/manifest.json",
    }
    stale = tmp_path / "cases" / "stale" / "strict_quantization_disabled"
    stale.mkdir(parents=True)
    (stale / "manifest.json").write_text("{}\n", encoding="utf-8")
    filtered = _artifact_manifest(tmp_path, case_ids=["case"])
    assert not any("stale" in path for path in filtered["files"])


def test_legacy_phase_still_runs_when_strict_phase_fails(tmp_path, monkeypatch):
    import torch

    import qbench

    monkeypatch.setattr(
        acceptance,
        "_build_public_model",
        lambda _case, pretrained: (
            torch.nn.Identity(),
            {"source": "test", "id": "fixed", "state_sha256": "fixed"},
        ),
    )
    monkeypatch.setattr(acceptance, "_build_loader", lambda *args, **kwargs: ())

    def fail_strict(*args, **kwargs):
        raise RuntimeError("strict gap")

    monkeypatch.setattr(qbench, "inspect_provider", fail_strict)
    legacy_calls = []

    def legacy_phase(*args, **kwargs):
        legacy_calls.append((args, kwargs))
        return {
            "status": "passed",
            "configuration": legacy_quantized_configuration(),
            "predictions": {},
            "accuracy": {},
        }

    monkeypatch.setattr(acceptance, "_run_legacy_quantized_phase", legacy_phase)
    result = acceptance._execute_case(
        PUBLIC_MODEL_MATRIX[0],
        AcceptanceConfig(
            output_directory=str(tmp_path),
            imagenet_directory=str(tmp_path),
            device="cpu",
            pretrained=True,
        ),
        imagenet_directory=tmp_path,
        target_map={},
        subset={"indices": []},
        output_directory=tmp_path,
    )
    assert result["status"] == "failed"
    assert result["phases"]["strict_quantization_disabled"]["status"] == "failed"
    assert result["phases"][LEGACY_PHASE_NAME]["status"] == "passed"
    assert len(legacy_calls) == 1


def test_offline_tiny_model_executes_strict_api_and_phase_contract(
    tmp_path, monkeypatch
):
    import torch

    torch.manual_seed(11)
    model = torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
    ).eval()
    monkeypatch.setattr(
        acceptance,
        "_build_public_model",
        lambda _case, pretrained: (
            model,
            {"source": "test", "id": "fixed", "state_sha256": "fixed"},
        ),
    )
    batches = [(torch.randn(2, 3, 4, 4), torch.tensor([0, 1]))]
    monkeypatch.setattr(
        acceptance,
        "_build_loader",
        lambda *args, **kwargs: batches,
    )
    monkeypatch.setattr(
        acceptance,
        "_fast_metric_overhead",
        lambda *args, **kwargs: {"status": "passed", "overhead_fraction": 0.0},
    )
    legacy_calls = []

    def legacy_phase(*args, **kwargs):
        legacy_calls.append((args, kwargs))
        return {
            "status": "passed",
            "configuration": legacy_quantized_configuration(),
            "support": {"planned_mappings": ["module:0=>activation"]},
            "accuracy": {},
            "predictions": {},
            "plan_derivation": {"derivation": "module-routes-only-v1"},
        }

    monkeypatch.setattr(acceptance, "_run_legacy_quantized_phase", legacy_phase)
    spec = acceptance.PublicModelSpec("tiny", "test", "tiny", "fixed", None, 11)
    case = acceptance.AcceptanceCase(spec, 4, 15)
    result = acceptance._execute_case(
        case,
        AcceptanceConfig(
            output_directory=str(tmp_path),
            imagenet_directory=str(tmp_path),
            device="cpu",
            pretrained=True,
            enable_fx=False,
            enable_export=False,
            overhead_repetitions=3,
        ),
        imagenet_directory=tmp_path,
        target_map={},
        subset={"indices": [0, 1]},
        output_directory=tmp_path,
    )
    assert result["status"] == "passed", result.get("error")
    strict = result["phases"]["strict_quantization_disabled"]
    legacy = result["phases"][LEGACY_PHASE_NAME]
    assert strict["status"] == legacy["status"] == "passed"
    assert strict["predictions"]["reference"] == strict["predictions"]["simulator"]
    assert legacy["configuration"] == legacy_quantized_configuration()
    assert legacy["plan_derivation"]["derivation"] == "module-routes-only-v1"
    assert all(
        mapping.startswith("module:")
        for mapping in legacy["support"]["planned_mappings"]
    )
    assert result["performance"]["fast_metric_overhead"]["status"] == "passed"
    assert len(legacy_calls) == 1
    assert (
        tmp_path
        / "cases"
        / case.case_id
        / "strict_quantization_disabled"
        / "manifest.json"
    ).is_file()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_count", 0),
        ("subset_seed", -1),
        ("batch_size", True),
        ("num_workers", -1),
        ("accuracy_tolerance", -0.1),
        ("overhead_batches", 0),
        ("overhead_repetitions", 2),
        ("overhead_warmups", -1),
        ("device", ""),
    ],
)
def test_acceptance_config_rejects_invalid_offline_values(tmp_path, field, value):
    values = {
        "output_directory": str(tmp_path / "output"),
        "imagenet_directory": str(tmp_path / "imagenet"),
    }
    values[field] = value
    with pytest.raises(AcceptanceError):
        AcceptanceConfig(**values)


def test_default_command_describes_matrix_without_importing_optional_models(
    monkeypatch, capsys
):
    real_import = builtins.__import__

    def reject_optional(name, *args, **kwargs):
        if name == "torchvision" or name.startswith("torchvision."):
            raise AssertionError("default command imported torchvision")
        if name == "timm" or name.startswith("timm."):
            raise AssertionError("default command imported timm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_optional)
    assert main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["matrix"]) == 6
    assert payload["live_gate"]["environment"].endswith("=1")
    assert payload["phases"][LEGACY_PHASE_NAME] == legacy_quantized_configuration()
    assert payload["defaults"]["max_fast_metric_overhead"] == 0.10
    assert payload["defaults"]["minimum_overhead_batch_invocations"] == (
        MIN_OVERHEAD_BATCH_INVOCATIONS
    )


def test_live_gate_is_explicit_and_fails_before_optional_imports(tmp_path, monkeypatch):
    monkeypatch.delenv(RUN_ENVIRONMENT_VARIABLE, raising=False)
    config = AcceptanceConfig(
        output_directory=str(tmp_path / "output"),
        imagenet_directory=str(tmp_path / "imagenet"),
        pretrained=True,
    )
    with pytest.raises(AcceptanceError, match=RUN_ENVIRONMENT_VARIABLE):
        validate_execution_gate(config)


def test_offline_orchestration_emits_reusable_artifacts_and_clean_diff(
    tmp_path, monkeypatch
):
    imagenet = tmp_path / "imagenet"
    imagenet.mkdir()
    identities = {
        index: {"relative_path": f"class/sample-{index}.JPEG", "target": index % 2}
        for index in range(10)
    }
    monkeypatch.setattr(acceptance, "validate_execution_gate", lambda _config: None)
    monkeypatch.setattr(
        acceptance,
        "_imagenet_inventory",
        lambda _directory, _class_index: (10, {0: 0, 1: 1}, identities),
    )
    monkeypatch.setattr(acceptance, "_configure_determinism", lambda: None)
    monkeypatch.setattr(
        acceptance,
        "_environment_manifest",
        lambda: {"torch": "offline-test"},
    )

    def fake_case_runner(case, _config, **_context):
        return {
            **case.to_dict(),
            "status": "passed",
            "weights": {"state_sha256": "fixed"},
            "support": {
                "verdict": "fully_supported",
                "planned_mappings": ["schema:aten::linear=>linear"],
                "operation_schemas": ["aten::linear"],
                "operation_counts": {"aten::linear": 1},
                "operation_ledger_counts": {"capture|root|aten::linear": 1},
            },
            "accuracy": {
                "reference_top1": 1.0,
                "reference_top5": 1.0,
                "simulator_top1": 1.0,
                "simulator_top5": 1.0,
                "prediction_agreement": 1.0,
            },
            "equivalence": {"passed": True},
        }

    first_directory = tmp_path / "first"
    first = run_acceptance(
        AcceptanceConfig(
            output_directory=str(first_directory),
            imagenet_directory=str(imagenet),
            sample_count=4,
            subset_seed=9,
        ),
        cases=PUBLIC_MODEL_MATRIX[:1],
        case_runner=fake_case_runner,
    )
    assert first["passed"] is True
    assert first["baseline_compared"] is False
    assert (first_directory / "acceptance.json").is_file()
    assert (first_directory / "subset_manifest.json").is_file()
    assert (first_directory / "acceptance_manifest.json").is_file()

    second_directory = tmp_path / "second"
    second = run_acceptance(
        AcceptanceConfig(
            output_directory=str(second_directory),
            imagenet_directory=str(imagenet),
            baseline_directory=str(first_directory),
            sample_count=4,
            subset_seed=9,
        ),
        cases=PUBLIC_MODEL_MATRIX[:1],
        case_runner=fake_case_runner,
    )
    assert second["passed"] is True
    assert second["diff_passed"] is True
    diff = json.loads(
        (second_directory / "acceptance_diff.json").read_text(encoding="utf-8")
    )
    assert diff["regression_count"] == 0


_LIVE_ENABLED = os.environ.get(RUN_ENVIRONMENT_VARIABLE) == "1"


@pytest.mark.skipif(
    not _LIVE_ENABLED,
    reason=f"set {RUN_ENVIRONMENT_VARIABLE}=1 for pretrained CUDA/ImageNet acceptance",
)
def test_public_model_acceptance_live(tmp_path):
    baseline = os.environ.get("QBENCH_PUBLIC_MODEL_BASELINE")
    imagenet = os.environ.get("QBENCH_IMAGENET_VAL", "/data/imagenet/val")
    if not baseline:
        pytest.fail("QBENCH_PUBLIC_MODEL_BASELINE is required for the live gate")
    report = run_acceptance(
        AcceptanceConfig(
            output_directory=str(tmp_path),
            imagenet_directory=imagenet,
            baseline_directory=baseline,
            pretrained=True,
            require_baseline=True,
        )
    )
    assert report["all_cases_passed"] is True
    assert report["diff_passed"] is True
    assert report["passed"] is True
