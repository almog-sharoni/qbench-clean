# ruff: noqa: E402
from __future__ import annotations

import builtins
import copy
import gzip
import hashlib
import io
import json
import sys
import types
from importlib import resources
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
nn = torch.nn

from qbench import cli
from qbench.artifacts import _json_safe, write_artifacts
from qbench.capture import clone_invocation
from qbench.evaluation import EvaluationConfig, EvaluationReport, evaluate
from qbench.inspection import inspect_model, inspect_provider
from qbench.kernels import list_kernels, verify_kernels
from qbench.providers import (
    DirectObjectProvider,
    LegacyAdapterProvider,
    load_provider,
)
from qbench.schemas import (
    InspectionConfig,
    InspectionResult,
    OperationRecord,
    QBenchError,
    QuantizationPolicy,
    Scenario,
    SimulationPlan,
    VerificationResult,
)


class _CountingSimulator:
    def __init__(self, model: nn.Module):
        self._model = model.eval()
        self.calls = 0

    def run(self, scenario: Scenario):
        self.calls += 1
        args, kwargs = clone_invocation(scenario)
        with torch.inference_mode():
            return self._model(*args, **kwargs)


class _EvaluationProvider:
    def __init__(self, scenarios):
        self.scenarios = list(scenarios)

    def evaluation_loader(self):
        return self.scenarios

    def prepare_evaluation_batch(self, batch):
        return batch

    def select_metric_output(self, output):
        return output


class _StochasticMutatingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = nn.Identity()
        self.calls = 0

    def forward(self, value, *, alias):
        assert value is alias
        self.calls += 1
        value.add_(1)
        random = torch.rand_like(value)
        value = self.probe(value)
        return {"primary": (value + random,), "nested": [value * 2]}


def test_fast_evaluation_is_exactly_two_forwards_and_restores_inputs_rng_and_training(
    monkeypatch,
):
    reference = _StochasticMutatingModel()
    reference.train()
    reference.probe.eval()
    simulated_model = copy.deepcopy(reference)
    simulator = _CountingSimulator(simulated_model)
    original = torch.zeros(4)
    scenario = Scenario("aliased", (original,), {"alias": original})
    provider = _EvaluationProvider([scenario, scenario])
    rng_before = torch.random.get_rng_state().clone()

    # The fast evaluator itself must not install optional metric hooks.
    def reject_optional_hook(*_args, **_kwargs):
        raise AssertionError("fast evaluation installed a detailed hook")

    monkeypatch.setattr(nn.Module, "register_forward_hook", reject_optional_hook)
    report = evaluate(reference, simulator, provider, {"metrics": "fast"})

    assert report.batches == 2
    assert report.reference_forwards == reference.calls == 2
    assert report.simulator_forwards == simulator.calls == 2
    assert simulated_model.calls == 2
    assert report.metrics["perfect_match"] is True
    assert report.details == {}
    assert report.timing["reference_batches_per_second"] > 0
    assert report.timing["simulator_batches_per_second"] > 0
    assert report.timing["reference_examples_per_second"] == pytest.approx(
        report.timing["reference_batches_per_second"] * 4
    )
    assert report.timing["simulator_examples_per_second"] == pytest.approx(
        report.timing["simulator_batches_per_second"] * 4
    )
    assert torch.equal(original, torch.zeros_like(original))
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert reference.training is True
    assert reference.probe.training is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fast_cuda_evaluation_uses_one_coarse_paired_timing_span():
    reference = nn.Linear(4, 3).cuda().eval()
    simulator = _CountingSimulator(copy.deepcopy(reference))
    scenarios = [
        Scenario(f"batch_{index}", (torch.randn(8, 4, device="cuda"),), {})
        for index in range(2)
    ]

    report = evaluate(
        reference,
        simulator,
        _EvaluationProvider(scenarios),
        {"metrics": "fast"},
    )

    assert report.reference_forwards == report.simulator_forwards == 2
    assert report.metrics["perfect_match"] is True
    assert report.details == {}
    assert report.timing["kind"] == "cuda_event_coarse"
    assert report.timing["individual_model_timing"] is False
    assert report.timing["reference_seconds"] is None
    assert report.timing["simulator_seconds"] is None
    assert report.timing["paired_seconds"] > 0
    assert report.timing["paired_batches_per_second"] > 0
    assert report.timing["paired_examples_per_second"] > 0


def test_fast_cpu_metric_aggregates_preserve_finite_filtering_and_statistics():
    class AddOne(nn.Module):
        def forward(self, value):
            return value + 1

    scenario = Scenario(
        "metrics", (torch.tensor([1.0, 2.0, float("nan"), float("inf")]),), {}
    )
    report = evaluate(
        nn.Identity(),
        _CountingSimulator(AddOne()),
        _EvaluationProvider([scenario]),
        {"metrics": "fast"},
    )

    assert report.metrics["mae"] == pytest.approx(1.0)
    assert report.metrics["mse"] == pytest.approx(1.0)
    assert report.metrics["cosine_similarity"] == pytest.approx(
        8.0 / np.sqrt(5.0 * 13.0)
    )
    assert report.metrics["sqnr_db"] == pytest.approx(10.0 * np.log10(2.5))
    assert report.metrics["perfect_match"] is False
    assert report.metrics["nonfinite_reference"] == 2
    assert report.metrics["nonfinite_simulator"] == 2


def test_nested_output_structure_is_validated_before_metrics():
    class Reference(nn.Module):
        def forward(self, value):
            return {"logits": (value,)}

    class Simulated(nn.Module):
        def forward(self, value):
            return {"features": (value,)}

    scenario = Scenario("nested", (torch.ones(2),))
    with pytest.raises(QBenchError, match="mapping keys differ"):
        evaluate(
            Reference(),
            _CountingSimulator(Simulated()),
            _EvaluationProvider([scenario]),
        )


def test_prediction_agreement_counts_every_sequence_prediction():
    class Reference(nn.Module):
        def forward(self, _value):
            return torch.tensor([[[2.0, 0.0], [2.0, 0.0]]])

    class Simulated(nn.Module):
        def forward(self, _value):
            return torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])

    scenario = Scenario("sequence", (torch.zeros(1),))
    report = evaluate(
        Reference(),
        _CountingSimulator(Simulated()),
        _EvaluationProvider([scenario]),
    )
    assert report.metrics["prediction_agreement"] == pytest.approx(0.5)


def test_detailed_evaluation_streams_layer_histograms_and_removes_hooks():
    reference = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    simulator = _CountingSimulator(copy.deepcopy(reference))
    scenario = Scenario("detail", (torch.tensor([[1.0, -2.0]]),))

    report = evaluate(
        reference,
        simulator,
        _EvaluationProvider([scenario]),
        {"metrics": "detailed"},
    )

    assert report.reference_forwards == report.simulator_forwards == 1
    stats = report.details["activation_statistics"]
    assert set(stats["reference"]) == {"0", "1"}
    assert set(stats["simulator"]) == {"0", "1"}
    relu = stats["reference"]["1"]
    assert sum(relu["histogram"]["counts"]) == relu["elements"]
    assert report.details["module_inventory"]["total_modules"] == 3
    assert report.details["cuda_peak_memory"] == {
        "measured": False,
        "devices": {},
    }
    assert report.details["activation_retention"] == {
        "enabled": False,
        "max_elements": 4096,
        "captured_elements": 0,
        "truncated": False,
    }
    assert report.details["quantized_value_compliance"]["status"] == "not_assessed"
    assert not reference[0]._forward_hooks
    assert not reference[1]._forward_hooks
    assert not simulator._model[0]._forward_hooks
    assert not simulator._model[1]._forward_hooks


def test_detailed_evaluation_repeats_timing_and_caps_explicit_retention():
    class CountingIdentity(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return value + 1

    reference = nn.Sequential(CountingIdentity())
    simulator = _CountingSimulator(copy.deepcopy(reference))
    scenario = Scenario("detail", (torch.arange(4.0),))

    report = evaluate(
        reference,
        simulator,
        _EvaluationProvider([scenario]),
        {
            "metrics": "detailed",
            "latency_repetitions": 3,
            "retain_activations": True,
            "activation_retention_max_elements": 3,
            "compliance_scan": False,
        },
    )

    assert report.batches == 1
    assert report.reference_forwards == report.simulator_forwards == 3
    assert reference[0].calls == simulator._model[0].calls == 3
    assert report.timing["repetitions_per_batch"] == 3
    assert report.details["activation_statistics"]["reference"]["0"]["calls"] == 3
    retained = report.details["activation_retention"]
    assert retained["enabled"] is True
    assert retained["captured_elements"] == retained["max_elements"] == 3
    assert retained["truncated"] is True
    record = retained["activations"]["reference"]["0"][0]
    assert record["values"] == [1.0, 2.0, 3.0]
    assert report.details["quantized_value_compliance"]["status"] == "disabled"


def test_detailed_value_compliance_scans_current_values_and_restores_capture_state():
    class FakeQuantized(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_type = "fp8_e4m3"
            self.input_quantization = True
            self.output_quantization = False
            self.weight_quantization = True
            self.capture_activations = False
            self.register_buffer("weight_fp8", torch.tensor([0.5]))

        def forward(self, value):
            if self.capture_activations:
                self.last_quant_input_unscaled = torch.tensor(
                    [0.5, 0.3], device=value.device
                )
            return value

    # The fallback recognizer is intentionally restricted to QBench-owned
    # implementations when no registry record exists.
    FakeQuantized.__module__ = "qbench.testing"
    reference = nn.Identity()
    simulator = _CountingSimulator(FakeQuantized())
    scenario = Scenario("compliance", (torch.ones(2),))

    report = evaluate(
        reference,
        simulator,
        _EvaluationProvider([scenario]),
        {"metrics": "detailed"},
    )

    compliance = report.details["quantized_value_compliance"]
    assert compliance["enabled"] is True
    assert compliance["status"] == "failed"
    assert compliance["passed"] is False
    assert compliance["checks"] == 2
    assert compliance["invalid_values"] >= 1
    assert compliance["modules"]["<root>"]["weight"]["fp8_e4m3"]["passed"]
    assert not compliance["modules"]["<root>"]["input"]["fp8_e4m3"]["passed"]
    assert simulator._model.capture_activations is False
    assert not hasattr(simulator._model, "last_quant_input_unscaled")


def test_detailed_compliance_observes_transient_functional_handler_modules():
    from qbench.runtime import simulation_route, simulator_implementation

    instances = []

    class FakeRuntimeQuantized(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_type = "fp8_e4m3"
            self.input_quantization = True
            self.output_quantization = False
            self.capture_activations = False
            instances.append(self)

        def forward(self, value):
            if self.capture_activations:
                self.last_quant_input_unscaled = torch.tensor(
                    [0.5], device=value.device
                )
            return value

    FakeRuntimeQuantized.__module__ = "qbench.testing"

    class RuntimeSimulator:
        def __init__(self):
            self._model = nn.Identity()

        def run(self, scenario):
            args, _kwargs = clone_invocation(scenario)
            with simulator_implementation(), simulation_route("functional"):
                return FakeRuntimeQuantized()(args[0])

    scenario = Scenario("functional", (torch.ones(1),))
    report = evaluate(
        nn.Identity(),
        RuntimeSimulator(),
        _EvaluationProvider([scenario]),
        {"metrics": "detailed"},
    )

    compliance = report.details["quantized_value_compliance"]
    assert compliance["status"] == "passed"
    assert compliance["modules"]["functional"]["input"]["fp8_e4m3"]["passed"]
    assert (
        "functional::FakeRuntimeQuantized"
        in report.details["activation_statistics"]["simulator"]
    )
    assert instances[0].capture_activations is False
    assert not hasattr(instances[0], "last_quant_input_unscaled")


def test_detailed_hooks_and_capture_state_are_restored_after_metric_failure():
    class FakeQuantized(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_type = "fp8_e4m3"
            self.input_quantization = True
            self.output_quantization = False
            self.capture_activations = False

        def forward(self, value):
            if self.capture_activations:
                self.last_quant_input_unscaled = torch.tensor(
                    [0.5], device=value.device
                )
            return {"unexpected": value}

    FakeQuantized.__module__ = "qbench.testing"
    simulator = _CountingSimulator(FakeQuantized())
    scenario = Scenario("failure", (torch.ones(1),))

    with pytest.raises(QBenchError, match="tensor/non-tensor output mismatch"):
        evaluate(
            nn.Identity(),
            simulator,
            _EvaluationProvider([scenario]),
            {"metrics": "detailed"},
        )

    assert simulator._model.capture_activations is False
    assert not hasattr(simulator._model, "last_quant_input_unscaled")
    assert not simulator._model._forward_hooks
    assert not simulator._model._forward_pre_hooks


def test_detailed_cli_options_coerce_to_evaluation_config():
    args = cli.build_parser().parse_args(
        [
            "evaluate",
            "package.module:provider",
            "--metrics",
            "detailed",
            "--latency-repetitions",
            "4",
            "--retain-activations",
            "--activation-retention-max-elements",
            "17",
            "--no-compliance-scan",
        ]
    )

    config = cli._evaluation_config(args)
    assert config.latency_repetitions == 4
    assert config.retain_activations is True
    assert config.activation_retention_max_elements == 17
    assert config.compliance_scan is False


@pytest.mark.parametrize(
    ("value", "task"),
    [
        ("language-modeling", "language_modeling"),
        ("feature-matching", "feature_matching"),
    ],
)
def test_evaluation_config_normalizes_public_task_aliases(value, task):
    assert EvaluationConfig.coerce({"task": value}).task == task


@pytest.mark.parametrize(
    "config",
    [
        {"metrics": "everything"},
        {"task": "segmentation"},
        {"max_batches": -1},
        {"max_batches": 1.5},
        {"latency_repetitions": 0},
        {"latency_repetitions": True},
        {"metrics": "fast", "latency_repetitions": 2},
        {"metrics": "fast", "retain_activations": True},
        {"metrics": "detailed", "retain_activations": 1},
        {"metrics": "detailed", "activation_retention_max_elements": 0},
        {"metrics": "detailed", "compliance_scan": "yes"},
        {"unknown": True},
    ],
)
def test_evaluation_config_rejects_invalid_public_values(config):
    with pytest.raises(QBenchError):
        EvaluationConfig.coerce(config)


class _LoadableProvider:
    def __init__(self):
        self.model = nn.Identity()

    def build_model(self):
        return self.model

    def clone_model(self, model):
        return copy.deepcopy(model)

    def capture_scenarios(self):
        return [Scenario("capture", (torch.ones(1),))]

    def evaluation_loader(self):
        return []

    def prepare_evaluation_batch(self, batch):
        return batch

    def select_metric_output(self, output):
        return output


def test_provider_loading_supports_objects_classes_factories_and_dotted_objects(
    monkeypatch,
):
    module_name = "_qbench_test_public_provider"
    module = types.ModuleType(module_name)
    module.provider_object = _LoadableProvider()
    module.ProviderClass = _LoadableProvider
    module.make_provider = _LoadableProvider
    module.container = types.SimpleNamespace(provider=_LoadableProvider())
    monkeypatch.setitem(sys.modules, module_name, module)

    assert load_provider(f"{module_name}:provider_object") is module.provider_object
    assert isinstance(load_provider(f"{module_name}:ProviderClass"), _LoadableProvider)
    assert isinstance(load_provider(f"{module_name}:make_provider"), _LoadableProvider)
    assert (
        load_provider(f"{module_name}:container.provider") is module.container.provider
    )
    with pytest.raises(QBenchError, match="missing"):
        load_provider(f"{module_name}:container")

    def fail_with_private_details():
        raise RuntimeError("private provider value 12345")

    module.fail_with_private_details = fail_with_private_details
    with pytest.raises(QBenchError) as caught:
        load_provider(f"{module_name}:fail_with_private_details")
    assert "details redacted" in str(caught.value)
    assert "12345" not in str(caught.value)


def test_direct_and_legacy_providers_clone_models_and_delegate_custom_behavior():
    scenario = Scenario("one", (torch.ones(1),))
    direct = DirectObjectProvider(nn.Linear(1, 1), scenario, loader=[scenario])
    cloned = direct.clone_model(direct.model)
    assert cloned is not direct.model
    assert cloned.weight.data_ptr() != direct.model.weight.data_ptr()
    assert direct.capture_scenarios() == [scenario]

    class Legacy:
        def __init__(self):
            self.model = nn.Identity()
            self.sample_input = torch.zeros(1)
            self.loader = ["batch"]
            self.clone_calls = 0

        def clone_model(self, model):
            self.clone_calls += 1
            return copy.deepcopy(model)

        def prepare_evaluation_batch(self, batch):
            return Scenario(f"legacy-{batch}", (torch.ones(1),))

        def select_metric_output(self, output):
            return output["selected"]

    legacy_object = Legacy()
    legacy = LegacyAdapterProvider(legacy_object)
    assert legacy.clone_model(legacy.build_model()) is not legacy_object.model
    assert legacy_object.clone_calls == 1
    assert legacy.capture_scenarios()[0].args == (legacy_object.sample_input,)
    assert legacy.evaluation_loader() == ["batch"]
    assert legacy.prepare_evaluation_batch("batch").name == "legacy-batch"
    selected = torch.tensor([3.0])
    assert legacy.select_metric_output({"selected": selected}) is selected

    legacy_object.sample_input = (torch.ones(1), torch.zeros(1))
    tuple_scenario = legacy.capture_scenarios()[0]
    assert len(tuple_scenario.args) == 2
    assert tuple_scenario.kwargs == {}

    legacy_object.sample_input = {"value": torch.ones(1), "scale": 2.0}
    mapping_scenario = legacy.capture_scenarios()[0]
    assert mapping_scenario.args == ()
    assert set(mapping_scenario.kwargs) == {"value", "scale"}


def _inspection_result_for_artifacts() -> InspectionResult:
    scenario = Scenario("artifact", (torch.tensor([123.0]),))
    operation = OperationRecord(
        sequence=0,
        scenario="artifact",
        namespace="aten",
        schema="aten::add.Tensor",
        overload="Tensor",
        module_path="",
        module_type="torch.nn.Identity",
        arguments={"nonfinite": float("nan"), "numpy": np.float32(1.25)},
        output={
            "kind": "tensor",
            "shape": [1],
            "dtype": "torch.float32",
            "device": "cpu",
        },
    )
    return InspectionResult(
        support={"schema_version": 1, "fully_supported": False, "tags": {"b", "a"}},
        operations=[operation],
        plan=SimulationPlan(schema_version=1, scenarios=[scenario]),
        verification=VerificationResult(),
    )


def test_artifacts_are_schema_v3_valid_json_deterministic_and_checksummed(tmp_path):
    result = _inspection_result_for_artifacts()
    evaluation = EvaluationReport(
        metrics={"finite": np.float64(2.5), "infinite": float("inf")},
        batches=1,
        reference_forwards=1,
        simulator_forwards=1,
    )
    state = {"weight": torch.tensor([1.0, 2.0])}
    first = write_artifacts(
        tmp_path / "first", result, evaluation=evaluation, state_dict=state
    )
    second = write_artifacts(
        tmp_path / "second", result, evaluation=evaluation, state_dict=state
    )

    expected_names = {
        "manifest.json",
        "support.json",
        "operations.jsonl.gz",
        "plan.json",
        "evaluation.json",
        "state.pt",
    }
    assert {path.name for path in first.iterdir()} == expected_names
    for name in expected_names:
        assert (first / name).read_bytes() == (second / name).read_bytes()

    support = json.loads((first / "support.json").read_text(encoding="utf-8"))
    plan = json.loads((first / "plan.json").read_text(encoding="utf-8"))
    evaluation_payload = json.loads(
        (first / "evaluation.json").read_text(encoding="utf-8")
    )
    assert support["schema_version"] == plan["schema_version"] == 3
    assert support["tags"] == ["a", "b"]
    assert support["verification"]["attempted"] is False
    assert support["diagnostics"] == {}
    assert evaluation_payload["metrics"] == {"finite": 2.5, "infinite": "Infinity"}

    with gzip.open(first / "operations.jsonl.gz", "rt", encoding="utf-8") as stream:
        operation = json.loads(stream.readline())
    assert operation["arguments"]["nonfinite"] == "NaN"
    assert operation["arguments"]["numpy"] == pytest.approx(1.25)

    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["scenario_names"] == ["artifact"]
    for name, metadata in manifest["files"].items():
        payload = (first / name).read_bytes()
        assert metadata == {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    # Reusing an artifact directory cannot leave stale optional payloads that
    # are absent from the new manifest.
    write_artifacts(first, result)
    assert {path.name for path in first.iterdir()} == {
        "manifest.json",
        "support.json",
        "operations.jsonl.gz",
        "plan.json",
    }


def test_captured_nonfinite_scalars_are_strict_json_safe_in_public_result():
    class NonfiniteArguments(nn.Module):
        def forward(self, value):
            return torch.nan_to_num(
                value,
                nan=float("nan"),
                posinf=float("inf"),
                neginf=float("-inf"),
            )

    result = inspect_model(
        NonfiniteArguments(),
        Scenario("nonfinite", (torch.tensor([0.0]),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )

    payload = result.to_dict(include_operations=True)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    operation = next(
        row for row in payload["operations"] if row["schema"] == "aten::nan_to_num"
    )
    operation_json = json.dumps(operation, sort_keys=True, allow_nan=False)
    for spelling in ('"NaN"', '"Infinity"', '"-Infinity"'):
        assert spelling in operation_json
        assert spelling in encoded


def test_simulation_plan_infers_omitted_scenario_names_from_kernel_ledgers():
    from qbench import KernelSpec

    first = KernelSpec("add", schemas=("aten::add.Tensor",)).to_dict()
    first.update(
        source_count=3,
        scenario_counts={"branch-b": 1, "shared": 2},
    )
    second = KernelSpec("mul", schemas=("aten::mul.Tensor",)).to_dict()
    second.update(
        source_count=2,
        scenario_counts={"branch-a": 1, "shared": 1},
    )
    plan = SimulationPlan.from_dict(
        {
            "schema_version": 3,
            "kernels": {
                "schema:aten::add.Tensor": first,
                "schema:aten::mul.Tensor": second,
            },
        }
    )

    assert plan.scenario_names == ["branch-b", "shared", "branch-a"]
    assert plan.to_dict()["scenario_names"] == [
        "branch-b",
        "shared",
        "branch-a",
    ]


def test_artifacts_record_privacy_safe_model_and_provider_provenance(tmp_path):
    scenario = Scenario("capture", (torch.ones(1, 2),))
    model = nn.Linear(2, 2)
    provider = DirectObjectProvider(model, scenario)
    result = inspect_provider(
        provider,
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )
    provenance = result.diagnostics["provenance"]

    assert provenance["model_type"] == "torch.nn.modules.linear.Linear"
    assert provenance["provider_type"].endswith("DirectObjectProvider")
    assert provenance["parameter_count"] == 6
    assert len(provenance["model_state_sha256"]) == 64
    int(provenance["model_state_sha256"], 16)

    before = provenance["model_state_sha256"]
    with torch.no_grad():
        model.weight.add_(1)
    changed = inspect_provider(
        provider,
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )
    assert changed.diagnostics["provenance"]["model_state_sha256"] != before

    directory = write_artifacts(tmp_path / "provenance", result)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["qbench_version"]
    assert manifest["provenance"] == provenance
    assert "weight" not in json.dumps(manifest["provenance"])


def test_model_state_digest_fallback_is_storage_independent_and_meta_safe(
    monkeypatch,
):
    from qbench.provenance import model_state_sha256

    torch.manual_seed(29)
    first = nn.Linear(3, 2)
    second = copy.deepcopy(first)
    assert first.weight.untyped_storage().data_ptr() != (
        second.weight.untyped_storage().data_ptr()
    )
    numpy_digest = model_state_sha256(first)

    def numpy_unavailable(_tensor):
        raise RuntimeError("NumPy unavailable")

    monkeypatch.setattr(torch.Tensor, "numpy", numpy_unavailable)
    assert model_state_sha256(first) == numpy_digest
    assert model_state_sha256(second) == numpy_digest

    with torch.no_grad():
        second.weight[0, 0].add_(1)
    assert model_state_sha256(second) != numpy_digest

    meta_first = nn.Linear(3, 2, device="meta")
    meta_second = copy.deepcopy(meta_first)
    assert model_state_sha256(meta_first) == model_state_sha256(meta_second)


def test_artifact_json_never_stringifies_arbitrary_mapping_keys():
    class PrivateKey:
        def __str__(self):
            raise AssertionError("private key was stringified")

        def __repr__(self):
            raise AssertionError("private key was represented")

    with pytest.raises(TypeError, match="JSON scalar"):
        _json_safe({PrivateKey(): "value"})


def _write_vector_manifest(root, rows, *, hardware_actual_results=True):
    manifest_rows = []
    for row in rows:
        row = dict(row)
        path = root / row["file"]
        row.setdefault("sha256", hashlib.sha256(path.read_bytes()).hexdigest())
        manifest_rows.append(row)
    manifest = {
        "schema_version": 1,
        "hardware_actual_results": hardware_actual_results,
        "vectors": manifest_rows,
    }
    if hardware_actual_results:
        manifest["hardware_evidence"] = {
            "producer": "qbench-test-fixture",
            "platform": "independent-test-device",
            "generated_at": "2026-01-01T00:00:00Z",
            "independent_of_simulator": True,
        }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_v2_vector_manifest(root, rows, *, hardware_actual_results=False):
    manifest_rows = []
    for row in rows:
        row = dict(row)
        path = root / row["file"]
        row.setdefault("sha256", hashlib.sha256(path.read_bytes()).hexdigest())
        row.setdefault("quantization_policy", "test-policy")
        manifest_rows.append(row)
    manifest = {
        "schema_version": 2,
        "hardware_actual_results": hardware_actual_results,
        "hardware_evidence": None,
        "quantization_policies": {
            "test-policy": QuantizationPolicy(
                quantize_first_layer=True,
                output_quantization=True,
            ).to_dict()
        },
        "vectors": manifest_rows,
    }
    if hardware_actual_results:
        manifest["hardware_evidence"] = {
            "producer": "qbench-test-fixture",
            "platform": "independent-test-device",
            "generated_at": "2026-01-01T00:00:00Z",
            "independent_of_simulator": True,
        }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_kernel_conformance_distinguishes_passed_failed_and_missing_evidence(tmp_path):
    passing = tmp_path / "passing"
    passing.mkdir()
    expected = np.array([-1.0, 2.0], dtype=np.float32)
    actual = expected.copy()
    np.savez(passing / "ulp.npz", expected=expected, actual=actual)
    _write_vector_manifest(
        passing,
        [{"kernel": "linear", "file": "ulp.npz"}],
    )
    passed = verify_kernels(passing)
    assert passed["status"] == "passed"
    assert passed["passed"] == 1
    assert passed["kernel_statuses"] == {"linear": "passed"}

    missing = tmp_path / "missing"
    missing.mkdir()
    np.savez(missing / "golden.npz", expected=np.array([1.0], dtype=np.float32))
    _write_vector_manifest(
        missing,
        [{"kernel": "softmax", "file": "golden.npz"}],
    )
    incomplete = verify_kernels(missing)
    assert incomplete["status"] == "missing_evidence"
    assert incomplete["passed"] == incomplete["failed"] == 0
    assert incomplete["missing"] == 1
    assert incomplete["kernel_statuses"] == {"softmax": "missing_evidence"}

    failing = tmp_path / "failing"
    failing.mkdir()
    np.savez(
        failing / "shape.npz",
        expected=np.array([1.0, 1.0], dtype=np.float32),
        actual=np.array([[1.0, 1.0]], dtype=np.float32),
    )
    _write_vector_manifest(
        failing,
        [{"kernel": "arithmetic", "file": "shape.npz"}],
    )
    failed = verify_kernels(failing)
    assert failed["status"] == "failed"
    assert failed["failed"] == 1
    assert failed["kernel_statuses"] == {"arithmetic": "failed"}
    assert "shape mismatch" in failed["errors"][0]


def test_kernel_catalog_exposes_declared_rules_used_when_manifest_omits_comparison(
    tmp_path,
):
    catalog = {row["name"]: row for row in list_kernels()}
    assert catalog
    assert all(isinstance(row["conformance"], dict) for row in catalog.values())

    expected = np.array([1.0], dtype=np.float32)
    actual = expected + np.float32(5e-7)
    np.savez(tmp_path / "linear.npz", expected=expected, actual=actual)
    _write_vector_manifest(
        tmp_path,
        [
            {
                "kernel": "linear",
                "file": "linear.npz",
            }
        ],
    )
    report = verify_kernels(tmp_path)
    assert report["status"] == "passed"
    assert report["kernel_statuses"] == {"linear": "passed"}


def test_kernel_conformance_checksum_failure_marks_kernel_failed(tmp_path):
    np.savez(
        tmp_path / "vector.npz",
        expected=np.array([1], dtype=np.int8),
        actual=np.array([1], dtype=np.int8),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "vectors": [
                    {
                        "kernel": "embedding",
                        "file": "vector.npz",
                        "sha256": "0" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = verify_kernels(tmp_path)
    assert report["status"] == "failed"
    assert report["kernel_statuses"] == {"embedding": "failed"}
    assert "checksum mismatch" in report["errors"][0]


def test_inspection_hardware_badge_uses_only_executed_kernel_evidence(
    tmp_path, monkeypatch
):
    from qbench import kernels

    np.savez(
        tmp_path / "arithmetic.npz",
        left=np.array([1.0], dtype=np.float32),
        right=np.array([1.0], dtype=np.float32),
        scalar=np.asarray(1.0, dtype=np.float32),
        actual=np.array([2.0], dtype=np.float32),
    )
    _write_v2_vector_manifest(
        tmp_path,
        [
            {
                "kernel": "arithmetic",
                "runner": "arithmetic",
                "file": "arithmetic.npz",
                "capability": {
                    "kind": "schema",
                    "target": "aten::add.Scalar",
                },
            }
        ],
        hardware_actual_results=True,
    )
    monkeypatch.setitem(
        kernels._CONFORMANCE_RUNNERS,
        "arithmetic",
        lambda _arrays, _capability, _policy: torch.tensor([1.0]),
    )

    class Add(nn.Module):
        def forward(self, value):
            return torch.ops.aten.add.Scalar(value, 1.0)

    result = inspect_model(
        Add(),
        Scenario("add", (torch.ones(2),)),
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            conformance_directory=str(tmp_path),
            quantization_policy=QuantizationPolicy(
                quantize_first_layer=True,
                output_quantization=True,
            ),
        ),
    )
    badge = result.support["hardware_fidelity"]
    assert badge["status"] == "failed"
    assert badge["kernels"] == {"arithmetic": "failed"}
    assert badge["capabilities"] == {"schema:aten::add.Scalar": "failed"}
    # Hardware evidence is an independent axis and cannot rewrite simulator
    # routing support for the captured scenario.
    assert result.fully_supported is True

    mismatched_policy = inspect_model(
        Add(),
        Scenario("add", (torch.ones(2),)),
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            conformance_directory=str(tmp_path),
        ),
    )
    assert mismatched_policy.support["hardware_fidelity"]["status"] == (
        "missing_evidence"
    )


def test_hardware_badge_scopes_grouped_kernel_evidence_to_exact_capability(
    monkeypatch,
):
    from qbench import kernels
    from qbench.inspection import _hardware_fidelity

    policy_sha256 = kernels._policy_sha256(QuantizationPolicy().to_dict())
    tensor_evidence = f"schema:aten::add.Tensor|policy:{policy_sha256}"
    scalar_evidence = f"schema:aten::add.Scalar|policy:{policy_sha256}"
    index_evidence = (
        f"schema:aten::add.Tensor#kernel:index_arithmetic|policy:{policy_sha256}"
    )
    monkeypatch.setattr(
        kernels,
        "verify_kernels",
        lambda _directory: {
            "status": "failed",
            "kernel_statuses": {
                "arithmetic": "failed",
                "index_arithmetic": "failed",
            },
            "capability_statuses": {
                "schema:aten::add.Tensor": "passed",
                "schema:aten::add.Scalar": "failed",
                "schema:aten::add.Tensor#kernel:index_arithmetic": "failed",
            },
            "capability_policy_statuses": {
                tensor_evidence: "passed",
                scalar_evidence: "failed",
                index_evidence: "failed",
            },
            "kernels": [
                {
                    "kernel": "arithmetic",
                    "capability_key": "schema:aten::add.Tensor",
                    "evidence_key": tensor_evidence,
                    "status": "passed",
                },
                {
                    "kernel": "arithmetic",
                    "capability_key": "schema:aten::add.Scalar",
                    "evidence_key": scalar_evidence,
                    "status": "failed",
                    "error": "unrelated overload failed",
                },
                {
                    "kernel": "index_arithmetic",
                    "capability_key": (
                        "schema:aten::add.Tensor#kernel:index_arithmetic"
                    ),
                    "evidence_key": index_evidence,
                    "status": "failed",
                    "error": "structural variant failed",
                },
            ],
            "errors": ["unrelated overload failed", "structural variant failed"],
        },
    )
    plan = SimulationPlan(
        kernels={
            "schema:aten::add.Tensor": {
                "name": "arithmetic",
                "classification": "quantized",
                "scenario_counts": {"capture": 1},
            },
            "schema:aten::add.Tensor#kernel:index_arithmetic": {
                "name": "index_arithmetic",
                "classification": "structural",
                "scenario_counts": {"capture": 1},
            },
        }
    )

    badge = _hardware_fidelity(plan, "unused-path")

    assert badge["status"] == "passed"
    assert badge["kernels"] == {"arithmetic": "passed"}
    assert badge["capabilities"] == {"schema:aten::add.Tensor": "passed"}
    assert badge["errors"] == []


def test_kernel_conformance_rejects_unknown_kernels_and_rule_overrides(tmp_path):
    values = np.array([1.0], dtype=np.float32)
    np.savez(tmp_path / "vector.npz", expected=values, actual=values)
    digest = hashlib.sha256((tmp_path / "vector.npz").read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "vectors": [
                    {
                        "kernel": "not-maintained",
                        "file": "vector.npz",
                        "sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert verify_kernels(tmp_path)["status"] == "configuration_error"

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "vectors": [
                    {
                        "kernel": "linear",
                        "file": "vector.npz",
                        "sha256": digest,
                        "comparison": {"kind": "bit_exact"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = verify_kernels(tmp_path)
    assert report["status"] == "configuration_error"
    assert report["configuration_errors"] == 1


def test_missing_numpy_is_configuration_error_and_cli_exit_one(
    tmp_path, monkeypatch, capsys
):
    values = np.array([1.0], dtype=np.float32)
    np.savez(tmp_path / "vector.npz", expected=values, actual=values)
    _write_vector_manifest(
        tmp_path,
        [{"kernel": "arithmetic", "file": "vector.npz"}],
    )

    real_import = builtins.__import__

    def import_without_numpy(name, *args, **kwargs):
        if name == "numpy" or name.startswith("numpy."):
            raise ModuleNotFoundError("No module named 'numpy'", name="numpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_numpy)

    report = verify_kernels(tmp_path)
    assert report["status"] == "configuration_error"
    assert report["configuration_errors"] == 1
    assert report["failed"] == 0
    assert report["kernel_statuses"] == {"arithmetic": "configuration_error"}
    assert "qbench[conformance]" in report["errors"][0]

    assert cli.main(["kernels", "verify", str(tmp_path)]) == cli.EXIT_ERROR
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["status"] == "configuration_error"
    assert "qbench[conformance]" in emitted["errors"][0]


def test_schema_v2_rejects_embedded_fp32_golden_output(tmp_path):
    np.savez(
        tmp_path / "relu.npz",
        input=np.array([-1.0, 1.0], dtype=np.float32),
        expected=np.array([0.0, 1.0], dtype=np.float32),
        actual=np.array([0.0, 1.0], dtype=np.float32),
    )
    _write_v2_vector_manifest(
        tmp_path,
        [
            {
                "kernel": "activation",
                "runner": "activation",
                "file": "relu.npz",
                "capability": {"kind": "schema", "target": "aten::relu"},
            }
        ],
    )

    report = verify_kernels(tmp_path)

    row = next(
        item
        for item in report["kernels"]
        if item.get("capability_key") == "schema:aten::relu"
    )
    assert report["status"] == "configuration_error"
    assert row["status"] == "configuration_error"
    assert "must not embed an FP32 expected output" in row["error"]


def test_schema_v2_requires_exact_pinned_capability_and_full_policy(tmp_path):
    np.savez(tmp_path / "relu.npz", input=np.array([-1.0, 1.0], dtype=np.float32))
    row = {
        "kernel": "activation",
        "runner": "activation",
        "file": "relu.npz",
        "capability": {
            "kind": "module",
            "target": "torch.nn.modules.activation.ReLU",
            "implementation": "qbench.ops.quant_activations.QuantGELU",
        },
    }
    _write_v2_vector_manifest(tmp_path, [row])
    wrong_implementation = verify_kernels(tmp_path)
    assert wrong_implementation["status"] == "configuration_error"
    assert "must pin maintained implementation" in wrong_implementation["errors"][0]

    row["capability"] = {"kind": "schema", "target": "aten::relu"}
    _write_v2_vector_manifest(tmp_path, [row])
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["quantization_policies"]["test-policy"]["rounding"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    incomplete_policy = verify_kernels(tmp_path)
    assert incomplete_policy["status"] == "configuration_error"
    assert "must declare every policy field exactly" in incomplete_policy["errors"][0]


def test_hardware_evidence_claim_requires_independent_provenance(tmp_path):
    np.savez(
        tmp_path / "vector.npz",
        expected=np.array([1.0], dtype=np.float32),
        actual=np.array([1.0], dtype=np.float32),
    )
    _write_vector_manifest(tmp_path, [{"kernel": "linear", "file": "vector.npz"}])
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hardware_evidence"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_kernels(tmp_path)

    assert report["status"] == "configuration_error"
    assert "requires hardware_evidence provenance" in report["errors"][0]


def test_schema_v2_hardware_uses_quantized_simulator_reference_and_provenance(
    tmp_path, monkeypatch
):
    from qbench import kernels

    np.savez(
        tmp_path / "relu.npz",
        input=np.array([-1.0, 1.0], dtype=np.float32),
        actual=np.array([0.0, 1.0], dtype=np.float32),
    )
    row = {
        "kernel": "activation",
        "runner": "activation",
        "file": "relu.npz",
        "capability": {"kind": "schema", "target": "aten::relu"},
    }
    monkeypatch.setitem(
        kernels._CONFORMANCE_RUNNERS,
        "activation",
        lambda _arrays, _capability, _policy: torch.tensor([0.0, 2.0]),
    )

    _write_v2_vector_manifest(tmp_path, [row], hardware_actual_results=True)
    measured = verify_kernels(tmp_path)
    measured_row = next(
        item
        for item in measured["kernels"]
        if item.get("capability_key") == "schema:aten::relu"
    )
    assert measured["status"] == "failed"
    assert measured_row["simulator_status"] == "passed"
    assert measured_row["status"] == "failed"
    assert "comparison failed" in measured_row["error"]
    assert measured_row["quantization_policy"] == "test-policy"
    assert len(measured_row["quantization_policy_sha256"]) == 64

    _write_v2_vector_manifest(tmp_path, [row], hardware_actual_results=False)
    untrusted = verify_kernels(tmp_path)
    untrusted_row = next(
        item
        for item in untrusted["kernels"]
        if item.get("capability_key") == "schema:aten::relu"
    )
    assert untrusted_row["status"] == "missing_evidence"
    assert "no trusted independent hardware evidence" in untrusted_row["error"]


def test_qenabled_conformance_runner_uses_public_strict_simulator(monkeypatch):
    from qbench import conversion, inspection, kernels

    observed = {}
    plan = types.SimpleNamespace(
        kernels={
            "schema:aten::relu": {
                "name": "activation",
                "counts_as_quantized": True,
            }
        },
        unresolved_schemas=[],
        module_decisions={},
    )

    def fake_inspect(model, scenario, config):
        observed["inspection"] = (model, scenario, config)
        return types.SimpleNamespace(plan=plan)

    class FakeSimulator:
        def verify(self, scenarios):
            observed["verified"] = scenarios
            return VerificationResult(
                attempted=True,
                succeeded=True,
                strict=True,
                quantized_execution=True,
            )

        def run(self, scenario):
            observed["run"] = scenario
            return torch.tensor([1.0])

        def close(self):
            observed["closed"] = True

    def fake_build(model, built_plan, strict):
        observed["build"] = (model, built_plan, strict)
        return FakeSimulator()

    monkeypatch.setattr(inspection, "inspect_model", fake_inspect)
    monkeypatch.setattr(conversion, "build_simulator", fake_build)
    model = nn.Identity()
    policy = QuantizationPolicy(
        quantize_first_layer=True, output_quantization=True
    ).to_dict()

    output = kernels._run_inspected(
        "activation",
        {"kind": "schema", "target": "aten::relu"},
        model,
        (torch.tensor([-1.0]),),
        {},
        policy,
    )

    config = observed["inspection"][2]
    assert config.quantization_enabled is True
    assert config.quantization_policy.to_dict() == policy
    assert config.device == "cuda"
    assert observed["build"][2] is True
    assert observed["verified"] == [observed["run"]]
    assert observed["closed"] is True
    assert torch.equal(output, torch.tensor([1.0]))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="q-enabled in-place conformance requires the documented GPU container",
)
def test_inplace_add_conformance_preserves_operand_zero_alias_and_mutation():
    from qbench.conformance_vectors.generate import QUANTIZATION_POLICY
    from qbench.conversion import build_simulator

    class InplaceAdd(nn.Module):
        def forward(self, left, right):
            before = left.clone()
            result = torch.ops.aten.add_.Tensor(left, right)
            return result, left, before

    device = torch.device("cuda")
    left = torch.tensor([1.0, -2.0, 4.0], device=device)
    right = torch.tensor([0.5, 1.0, -2.0], device=device)
    scenario = Scenario("inplace-add", (left, right))
    model = InplaceAdd().to(device).eval()
    result = inspect_model(
        model,
        scenario,
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            verify=True,
            device="cuda",
            quantization_enabled=True,
            quantization_policy=QUANTIZATION_POLICY,
        ),
    )

    assert result.plan.kernels["schema:aten::add_.Tensor"]["name"] == "inplace_add"
    assert result.verification.succeeded is True
    assert result.verification.quantized_execution is True
    simulator = build_simulator(model, result.plan, strict=True)
    try:
        output, mutated_left, before = simulator.run(scenario)
    finally:
        simulator.close()

    assert output is mutated_left
    assert output.data_ptr() == mutated_left.data_ptr()
    assert torch.equal(output, before + right)
    # Simulator.run isolates caller-owned invocation tensors.
    assert torch.equal(left, torch.tensor([1.0, -2.0, 4.0], device=device))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="q-enabled SDPA conformance requires the documented GPU container",
)
def test_scaled_dot_product_attention_conformance_is_numeric_and_quantized():
    from qbench.conformance_vectors.generate import (
        QUANTIZATION_POLICY,
        _scaled_dot_product_attention,
    )
    from qbench.conversion import build_simulator

    class FunctionalAttention(nn.Module):
        def forward(self, query, key, value):
            return torch.ops.aten.scaled_dot_product_attention.default(
                query,
                key,
                value,
                None,
                0.0,
                False,
                scale=None,
                enable_gqa=False,
            )

    device = torch.device("cuda")
    arrays = _scaled_dot_product_attention()
    inputs = tuple(
        torch.from_numpy(arrays[name]).to(device) for name in ("query", "key", "value")
    )
    scenario = Scenario("sdpa", inputs)
    model = FunctionalAttention().to(device).eval()
    result = inspect_model(
        model,
        scenario,
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            verify=True,
            device="cuda",
            quantization_enabled=True,
            quantization_policy=QUANTIZATION_POLICY,
        ),
    )

    route = result.plan.kernels["schema:aten::scaled_dot_product_attention"]
    assert route["name"] == "scaled_dot_product_attention"
    assert route["classification"] == "composite"
    assert result.verification.succeeded is True
    assert result.verification.output_equivalence is True
    assert result.verification.quantized_execution is True
    assert (
        result.verification.quantized_routes[
            "schema:aten::scaled_dot_product_attention"
        ]
        >= 1
    )
    simulator = build_simulator(model, result.plan, strict=True)
    try:
        simulated = simulator.run(scenario)
    finally:
        simulator.close()
    with torch.inference_mode():
        reference = model(*inputs)

    assert simulated.shape == reference.shape
    assert torch.isfinite(simulated).all()
    assert torch.mean(torch.abs(simulated - reference)).item() < 0.05


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="q-enabled dynamic-index conformance requires the documented GPU container",
)
def test_dynamic_index_and_clamp_conformance_is_exactly_routed_and_quantized():
    from qbench.conformance_vectors.generate import QUANTIZATION_POLICY
    from qbench.conversion import build_simulator

    class DynamicIndexKernels(nn.Module):
        def forward(self, value, integer_value, alias_input):
            indexes = torch.ops.aten.arange.default(
                value.shape[-1],
                dtype=torch.int64,
                layout=torch.strided,
                device=value.device,
                pin_memory=False,
            )
            coordinates = torch.ops.aten.to.dtype(
                indexes, torch.float32, False, False, None
            )
            roundtrip = torch.ops.aten.to.dtype(
                coordinates, torch.int64, False, False, None
            )
            float_clamp = torch.ops.aten.clamp.default(value, -1.0, 2.0)
            integer_clamp = torch.ops.aten.clamp.default(integer_value, -2, 5)
            alias_cast = torch.ops.aten.to.dtype(
                alias_input, alias_input.dtype, False, False, None
            )
            return (
                indexes,
                coordinates,
                roundtrip,
                float_clamp,
                integer_clamp,
                alias_input,
                alias_cast,
            )

    device = torch.device("cuda")
    value = torch.tensor([[-3.0, -0.5, 0.25, 4.0]], device=device)
    integer_value = torch.tensor([[-4, -1, 2, 7]], device=device)
    alias_input = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4)
    alias_input = alias_input.transpose(0, 2)
    assert alias_input.ndim == 3 and not alias_input.is_contiguous()
    scenario = Scenario("dynamic-index", (value, integer_value, alias_input))
    model = DynamicIndexKernels().to(device).eval()
    result = inspect_model(
        model,
        scenario,
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            verify=True,
            device="cuda",
            quantization_enabled=True,
            quantization_policy=QUANTIZATION_POLICY,
        ),
    )

    assert result.plan.kernels["schema:aten::arange"]["name"] == "index_arange"
    assert result.plan.kernels["schema:aten::arange"]["classification"] == "structural"
    assert result.plan.kernels["schema:aten::to.dtype"]["name"] == "index_dtype_cast"
    assert (
        result.plan.kernels["schema:aten::to.dtype"]["classification"] == "structural"
    )
    assert result.plan.kernels["schema:aten::clamp"]["name"] == "clamp"
    index_clamp = result.plan.kernels["schema:aten::clamp#kernel:index_clamp"]
    assert index_clamp["name"] == "index_clamp"
    assert index_clamp["classification"] == "structural"
    assert index_clamp["counts_as_quantized"] is False
    assert result.verification.succeeded is True
    assert result.verification.output_equivalence is True
    assert result.verification.quantized_execution is True
    assert result.verification.quantized_routes["schema:aten::clamp"] >= 1
    assert "schema:aten::clamp#kernel:index_clamp" not in (
        result.verification.quantized_routes
    )

    simulator = build_simulator(model, result.plan, strict=True)
    try:
        simulated = simulator.run(scenario)
    finally:
        simulator.close()
    with torch.inference_mode():
        reference = model(value, integer_value, alias_input)

    for actual, expected in zip(simulated, reference, strict=True):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected)
    assert simulated[-2] is simulated[-1]
    assert simulated[-2].data_ptr() == simulated[-1].data_ptr()
    assert simulated[-2].stride() == alias_input.stride()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="q-enabled mixed arithmetic conformance requires the documented GPU container",
)
def test_integer_and_mixed_add_sub_conformance_preserves_native_promotion():
    from qbench.conformance_vectors.generate import QUANTIZATION_POLICY
    from qbench.conversion import build_simulator

    class ArithmeticVariants(nn.Module):
        def forward(self, float_left, float_right, integer_left, integer_right):
            pairs = (
                (float_left, float_right),
                (integer_left, integer_right),
                (float_left, integer_right),
                (integer_left, float_right),
            )
            return tuple(
                result
                for left, right in pairs
                for result in (
                    torch.ops.aten.add.Tensor(left, right),
                    torch.ops.aten.sub.Tensor(left, right),
                )
            )

    device = torch.device("cuda")
    inputs = (
        torch.tensor([[-1.0, 0.5, 3.0], [2.0, -4.0, 0.25]], device=device),
        torch.tensor([[0.5, 1.25, -2.0], [-1.0, 2.0, 3.5]], device=device),
        torch.tensor([[-4, 0, 3], [7, -2, 1]], device=device),
        torch.tensor([[1, 5, -2], [-3, 2, 4]], device=device),
    )
    scenario = Scenario("arithmetic-variants", inputs)
    model = ArithmeticVariants().to(device).eval()
    result = inspect_model(
        model,
        scenario,
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            verify=True,
            device="cuda",
            quantization_enabled=True,
            quantization_policy=QUANTIZATION_POLICY,
        ),
    )

    for schema in ("aten::add.Tensor", "aten::sub.Tensor"):
        route = result.plan.kernels[f"schema:{schema}"]
        assert route["name"] == "arithmetic"
        assert route["counts_as_quantized"] is True
        assert result.verification.quantized_routes[f"schema:{schema}"] >= 3
        index_route_key = f"schema:{schema}#kernel:index_arithmetic"
        index_route = result.plan.kernels[index_route_key]
        assert index_route["name"] == "index_arithmetic"
        assert index_route["classification"] == "structural"
        assert index_route["counts_as_quantized"] is False
        assert index_route_key not in result.verification.quantized_routes
    assert result.verification.succeeded is True
    assert result.verification.output_equivalence is True
    assert result.verification.quantized_execution is True

    simulator = build_simulator(model, result.plan, strict=True)
    try:
        simulated = simulator.run(scenario)
    finally:
        simulator.close()
    with torch.inference_mode():
        reference = model(*inputs)

    for actual, expected in zip(simulated, reference, strict=True):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)


def test_packaged_conformance_corpus_covers_and_runs_every_ready_kernel():
    package = resources.files("qbench.conformance_vectors")
    manifest = json.loads(package.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["hardware_actual_results"] is False
    assert manifest["hardware_evidence"] is None

    ready_rows = [row for row in list_kernels() if row["ready"]]
    ready = {row["name"] for row in ready_rows}
    expected_capabilities = {
        (row["name"], "schema", target, None)
        for row in ready_rows
        for target in row["schemas"]
    } | {
        (row["name"], "module", target, implementation)
        for row in ready_rows
        for target, implementation in zip(
            row["module_types"], row["module_implementations"], strict=True
        )
    }
    declared_capabilities = {
        (
            row["kernel"],
            row["capability"]["kind"],
            row["capability"]["target"],
            row["capability"].get("implementation"),
        )
        for row in manifest["vectors"]
    }
    assert declared_capabilities == expected_capabilities
    assert len(expected_capabilities) == 55
    assert {row["kernel"] for row in manifest["vectors"]} == ready
    assert "embedding" not in ready

    for row in manifest["vectors"]:
        payload = package.joinpath(row["file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            assert "expected" not in arrays.files
            assert "actual" not in arrays.files

    from importlib.resources import files

    report = verify_kernels(files("qbench.conformance_vectors"))
    assert report["status"] == "missing_evidence"
    assert report["passed"] == report["failed"] == 0
    assert report["missing"] == len(expected_capabilities)
    assert set(report["kernel_statuses"]) == ready
    assert set(report["kernel_statuses"].values()) == {"missing_evidence"}
    expected_simulator_status = (
        "passed" if torch.cuda.is_available() else "not_assessed"
    )
    expected_simulator_passes = (
        len(expected_capabilities) if torch.cuda.is_available() else 0
    )
    assert report["simulator_status"] == expected_simulator_status
    assert report["simulator_passed"] == expected_simulator_passes
    assert report["simulator_failed"] == 0
    assert set(report["simulator_kernel_statuses"]) == ready
    assert set(report["simulator_kernel_statuses"].values()) == {
        expected_simulator_status
    }
    assert set(report["capability_statuses"].values()) == {"missing_evidence"}
    route_variant_keys = {
        "schema:aten::add.Tensor#kernel:index_arithmetic",
        "schema:aten::sub.Tensor#kernel:index_arithmetic",
        "schema:aten::clamp#kernel:index_clamp",
    }
    assert route_variant_keys <= set(report["capability_statuses"])
    assert {
        "schema:aten::add.Tensor",
        "schema:aten::sub.Tensor",
        "schema:aten::clamp",
    } <= set(report["capability_statuses"])
    assert len(report["capability_policy_statuses"]) == len(expected_capabilities)
    assert set(report["capability_policy_statuses"].values()) == {"missing_evidence"}
    assert set(report["simulator_capability_policy_statuses"].values()) == {
        expected_simulator_status
    }
