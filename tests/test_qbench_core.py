"""Focused regression coverage for the canonical QBench core."""

from __future__ import annotations

import copy
import json
import types

import pytest
import torch

from qbench import (
    InspectionConfig,
    QBenchError,
    QuantizationPolicy,
    Scenario,
    SimulationPlan,
)
from qbench.capture import capture_scenario
from qbench.conversion import Simulator, _create_simulator, build_simulator
from qbench.inspection import inspect_model
from qbench.registry import KernelSpec, OpRegistry

nn = torch.nn


def _route_row(
    name,
    scenario_counts,
    *,
    classification="quantized",
    counts_as_quantized=True,
    handler_quantized=False,
):
    return {
        "name": name,
        "classification": classification,
        "ready": True,
        "counts_as_quantized": counts_as_quantized,
        "handler_quantized": handler_quantized,
        "quantizes_weights": False,
        "weight_operand": None,
        "weight_argument": None,
        "activation_policy": False,
        "schemas": [],
        "module_types": [],
        "module_implementations": [],
        "input_operands": {},
        "policy_overrides": {},
        "module_invocations": {},
        "source_count": sum(scenario_counts.values()),
        "scenario_counts": dict(scenario_counts),
    }


class _KwargNestedModel(nn.Module):
    def forward(self, value, *, bias, alpha=2.0, dim=1):
        shifted = torch.add(value, bias, alpha=alpha)
        reduced = torch.sum(shifted, dim=dim)
        return {"reduced": reduced, "nested": (shifted, [reduced])}


def test_dispatch_capture_has_exact_schema_kwargs_nested_output_and_no_setup_ops():
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    bias = torch.ones_like(value)
    scenario = Scenario(
        "kwargs",
        (value,),
        {"bias": bias, "alpha": 3.0, "dim": 1},
    )

    records, output = capture_scenario(_KwargNestedModel(), scenario)

    assert list(output) == ["reduced", "nested"]
    assert isinstance(output["nested"], tuple)
    assert isinstance(output["nested"][1], list)
    assert [record.sequence for record in records] == list(range(len(records)))
    schemas = [record.schema for record in records]
    assert "aten::add.Tensor" in schemas
    assert "aten::sum.dim_IntList" in schemas
    assert not {"aten::clone", "aten::detach"}.intersection(schemas)

    add = next(record for record in records if record.schema == "aten::add.Tensor")
    assert add.namespace == "aten"
    assert add.overload == "Tensor"
    assert add.arguments["kwargs"]["alpha"] == 3.0
    tensor_arg = add.arguments["args"][0]
    assert tensor_arg == {
        "kind": "tensor",
        "shape": [2, 3],
        "dtype": "torch.float32",
        "device": "cpu",
        "requires_grad": False,
    }
    # Tensor contents are never placed in the raw operation record.
    assert "0." not in repr(add.arguments)


def test_dispatch_in_user_pre_hook_is_owned_by_deepest_active_module():
    class HookedIdentity(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = nn.Identity()
            self.child.register_forward_pre_hook(
                lambda _module, args: (torch.relu(args[0]),)
            )

        def forward(self, value):
            return self.child(value)

    records, _ = capture_scenario(
        HookedIdentity(), Scenario("pre-hook", (torch.tensor([-1.0, 1.0]),))
    )

    relu = next(record for record in records if record.schema == "aten::relu")
    assert relu.module_path == "child"
    assert relu.module_stack[-1]["path"] == "child"


def test_callsites_are_only_recorded_for_first_schema_occurrence():
    class RepeatedAdd(nn.Module):
        def forward(self, value):
            return (value + 1) + 2

    records, _ = capture_scenario(RepeatedAdd(), Scenario("repeat", (torch.ones(2),)))
    adds = [record for record in records if record.schema == "aten::add.Tensor"]
    assert len(adds) == 2
    assert adds[0].callsite is not None
    assert adds[1].callsite is None


def test_inplace_operation_is_captured_with_exact_aliasing_overload():
    class Inplace(nn.Module):
        def forward(self, value):
            local = value.clone()
            local.add_(2)
            return local

    records, output = capture_scenario(Inplace(), Scenario("inplace", (torch.ones(2),)))
    assert torch.equal(output, torch.full((2,), 3.0))
    assert "aten::clone" in [record.schema for record in records]
    inplace = next(record for record in records if record.schema == "aten::add_.Tensor")
    assert inplace.overload == "Tensor"


def test_inplace_add_is_strictly_routed_and_preserves_mutation_alias():
    class InplaceAdd(nn.Module):
        def forward(self, value, other):
            local = value.clone()
            alias = local
            returned = local.add_(other)
            return returned, alias

    scenario = Scenario("inplace-add", (torch.ones(2), torch.ones(2)))
    result = inspect_model(
        InplaceAdd(),
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )

    assert result.plan.kernels["schema:aten::add_.Tensor"]["name"] == "inplace_add"
    assert result.verification.succeeded is True
    assert result.fully_supported is True
    output = build_simulator(InplaceAdd(), result.plan).run(scenario)
    assert output[0] is output[1]
    torch.testing.assert_close(output[0], torch.full((2,), 2.0))


def test_dynamic_index_arithmetic_and_clamp_share_exact_runtime_routes():
    class DynamicIndexing(nn.Module):
        def forward(self, value):
            indexes = torch.ops.aten.arange.default(
                value.shape[-1],
                dtype=torch.int64,
                layout=torch.strided,
                device=value.device,
            )
            indexes = torch.ops.aten.add.Tensor(indexes, indexes)
            indexes = torch.ops.aten.clamp.default(indexes, None, value.shape[-1] - 1)
            coordinates = torch.ops.aten.to.dtype(indexes, torch.float32, False, False)
            shifted = torch.ops.aten.sub.Tensor(value, coordinates)
            reverse_mixed = torch.ops.aten.add.Tensor(indexes, coordinates)
            return (
                torch.ops.aten.clamp.default(shifted, 0.0, 1.0),
                indexes,
                reverse_mixed,
            )

    scenario = Scenario("dynamic-indexing", (torch.linspace(0.0, 4.0, 4),))
    result = inspect_model(
        DynamicIndexing(),
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )

    assert result.support["gaps"] == []
    assert result.verification.succeeded is True
    assert result.fully_supported is True
    assert result.plan.kernels["schema:aten::arange"]["name"] == "index_arange"
    assert result.plan.kernels["schema:aten::to.dtype"]["name"] == "index_dtype_cast"
    assert result.plan.kernels["schema:aten::add.Tensor"]["name"] == "arithmetic"
    assert (
        result.plan.kernels["schema:aten::add.Tensor#kernel:index_arithmetic"]["name"]
        == "index_arithmetic"
    )
    assert result.plan.kernels["schema:aten::sub.Tensor"]["name"] == "arithmetic"
    assert result.plan.kernels["schema:aten::clamp"]["name"] == "clamp"
    assert (
        result.plan.kernels["schema:aten::clamp#kernel:index_clamp"]["name"]
        == "index_clamp"
    )
    arange = next(
        operation
        for operation in result.operations
        if operation.schema == "aten::arange"
    )
    assert arange.arguments["kwargs"]["layout"] == "torch.strided"

    from qbench.capture import value_metadata
    from qbench.registry import find_kernel

    integer = torch.ones(2, dtype=torch.int64)
    integer_metadata = tuple(value_metadata((integer, integer)))
    integer_spec = find_kernel("aten::add.Tensor", integer_metadata, {})
    assert integer_spec is not None and integer_spec.name == "index_arithmetic"
    assert find_kernel("aten::mul.Tensor", integer_metadata, {}) is None


def test_large_integer_index_routes_remain_bit_exact_without_value_capture():
    class IndexMath(nn.Module):
        def forward(self, left, right):
            shifted = torch.ops.aten.add.Tensor(left, right)
            return torch.ops.aten.clamp.default(shifted, None, 2**55 + 9)

    scenario = Scenario(
        "large-index",
        (
            torch.tensor([2**40 + 1, 2**54 + 3], dtype=torch.int64),
            torch.tensor([2**40 + 7, 5], dtype=torch.int64),
        ),
    )
    result = inspect_model(
        IndexMath(),
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )

    assert result.support["gaps"] == []
    assert result.verification.succeeded is True
    assert result.verification.output_equivalence is True
    assert set(result.plan.kernels) == {
        "schema:aten::add.Tensor#kernel:index_arithmetic",
        "schema:aten::clamp#kernel:index_clamp",
    }
    output = build_simulator(IndexMath(), result.plan).run(scenario)
    expected = torch.clamp(scenario.args[0] + scenario.args[1], max=2**55 + 9)
    assert torch.equal(output, expected)

    tampered = result.plan.to_dict()
    row = tampered["kernels"].pop("schema:aten::add.Tensor#kernel:index_arithmetic")
    tampered["kernels"]["schema:aten::add.Tensor"] = row
    with pytest.raises(QBenchError, match="maintained variant key"):
        build_simulator(IndexMath(), tampered)


def test_input_cloning_preserves_shared_storage_between_distinct_views():
    class MutateAlias(nn.Module):
        def forward(self, left, right):
            left.add_(1)
            return right.clone()

    base = torch.zeros(4)
    left = base[:3]
    right = base[1:]

    _records, output = capture_scenario(MutateAlias(), Scenario("views", (left, right)))

    assert torch.equal(output, torch.tensor([1.0, 1.0, 0.0]))
    assert torch.equal(base, torch.zeros(4))


def test_raw_operation_records_include_the_full_active_module_stack():
    class Child(nn.Module):
        def forward(self, value):
            return torch.sin(value)

    class Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Child()

        def forward(self, value):
            return self.child(value)

    records, _output = capture_scenario(
        Parent(), Scenario("module-stack", (torch.ones(2),))
    )

    assert records[0].module_path == "child"
    assert [owner["path"] for owner in records[0].module_stack] == ["", "child"]
    assert records[0].module_stack[1]["type"].endswith(".Child")


def test_dynamic_scenarios_are_unioned_and_only_unexecuted_modules_not_assessed():
    class DynamicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.executed_identity = nn.Identity()
            self.never_called = nn.Identity()

        def forward(self, value, *, use_relu):
            value = self.executed_identity(value)
            return torch.relu(value) if use_relu else torch.sin(value)

    result = inspect_model(
        DynamicModel(),
        [
            Scenario("positive", (torch.ones(3),), {"use_relu": True}),
            Scenario("negative", (-torch.ones(3),), {"use_relu": False}),
        ],
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )

    assert result.support["capture_complete"] is True
    assert set(result.support["captured_scenarios"]) == {"positive", "negative"}
    assert {record.scenario for record in result.operations} == {"positive", "negative"}
    assert any(
        gap["schema"] == "aten::sin" and gap["scenario"] == "negative"
        for gap in result.support["gaps"]
    )
    coverage = result.support["scenario_coverage"]
    assert coverage["positive"]["supported"] is True
    assert coverage["positive"]["gap_count"] == 0
    assert coverage["negative"]["supported"] is False
    assert coverage["negative"]["unresolved_schemas"] == ["aten::sin"]
    rows = {row["path"]: row for row in result.support["module_summary"]}
    assert rows[""]["status"] == "unsupported"
    assert rows[""]["semantic_kernels"]["activation"] == {
        "count": 1,
        "example_scenarios": ["positive"],
    }
    assert rows["executed_identity"]["status"] == "supported"
    assert rows["executed_identity"]["operation_count"] == 0
    assert rows["never_called"]["status"] == "not_assessed"


def test_repeated_operations_remain_raw_and_are_aggregated_in_reports_and_plan():
    class Repeated(nn.Module):
        def forward(self, value):
            return (value + 1) + 2

    result = inspect_model(
        Repeated(),
        Scenario("repeat", (torch.ones(2),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )
    adds = [
        record for record in result.operations if record.schema == "aten::add.Tensor"
    ]
    assert len(adds) == 2
    root = result.support["module_summary"][0]
    assert root["operations"]["aten::add.Tensor"] == 2
    assert root["semantic_kernels"]["arithmetic"]["count"] == 2
    route = result.plan.kernels["schema:aten::add.Tensor"]
    assert route["source_count"] == 2
    assert route["scenario_counts"] == {"repeat": 2}


def test_repeated_module_invocations_are_counted_at_kernel_entry():
    class RepeatedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def forward(self, value):
            return self.relu(self.relu(value))

    result = inspect_model(
        RepeatedModule(),
        Scenario("twice", (-torch.ones(2),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )

    route = result.plan.kernels["module:relu"]
    assert route["source_count"] == 2
    assert route["scenario_counts"] == {"twice": 2}
    assert result.verification.succeeded is True
    assert result.verification.realized_operations == 2


def test_custom_linear_override_is_resolved_from_dispatch_not_class_name():
    class OverrideLinear(nn.Linear):
        def forward(self, value):
            return torch.sin(super().forward(value))

    class Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = OverrideLinear(4, 2)

        def forward(self, value):
            return self.layer(value)

    result = inspect_model(
        Parent(),
        Scenario("custom", (torch.randn(2, 4),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )

    schemas = {
        record.schema for record in result.operations if record.module_path == "layer"
    }
    assert "aten::linear" in schemas or "aten::addmm" in schemas
    assert "aten::sin" in schemas
    assert "layer" not in result.plan.module_decisions
    assert any(gap["schema"] == "aten::sin" for gap in result.support["gaps"])


def test_semantic_root_module_is_physically_replaced_and_verified():
    model = nn.Linear(4, 2)
    result = inspect_model(
        model,
        Scenario("root-linear", (torch.randn(3, 4),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )

    assert "" in result.plan.module_decisions
    assert "module:" in result.plan.kernels
    assert result.verification.succeeded is True
    assert result.verification.output_equivalence is True
    assert result.support["strict_realization"] is True
    assert result.fully_supported is True
    assert result.support["verdict"] == "fully_supported"

    simulator = build_simulator(model, result.plan)
    with pytest.raises(QBenchError, match="Simulator.run"):
        simulator._model(torch.randn(1, 4))


def test_attention_module_is_planned_as_one_composite_and_children_are_audited():
    model = nn.MultiheadAttention(4, 2, batch_first=True)
    value = torch.randn(2, 3, 4)
    unsupported = inspect_model(
        model,
        Scenario("mha-default", (value, value, value)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )
    assert unsupported.plan.module_decisions == {}
    assert "module invocation constraints" in unsupported.support["gaps"][0]["reason"]

    result = inspect_model(
        model,
        Scenario("mha", (value, value, value), {"need_weights": False}),
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )

    assert result.plan.module_decisions[""].endswith("DecomposedMultiheadAttention")
    assert result.plan.kernels["module:"]["name"] == "attention"
    assert result.verification.succeeded is True
    assert result.verification.output_equivalence is True

    positional = inspect_model(
        model,
        Scenario("mha-positional", (value, value, value, None, False)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )
    assert positional.plan.kernels["module:"]["name"] == "attention"


def test_attention_factory_preserves_source_placement_dtype_and_state():
    from qbench import conversion

    source = nn.MultiheadAttention(
        4,
        2,
        batch_first=True,
        dtype=torch.float64,
    ).train()
    converted, did_convert, errors = conversion._convert_modules(
        source,
        module_paths=("",),
        quantization_enabled=False,
    )

    assert did_convert is True
    assert errors == []
    assert converted.training is False  # Canonical conversion is inference-only.
    source_q, source_k, source_v = source.in_proj_weight.chunk(3, dim=0)
    assert converted.q_proj.weight.device == source.in_proj_weight.device
    assert converted.q_proj.weight.dtype == source.in_proj_weight.dtype
    torch.testing.assert_close(converted.q_proj.weight, source_q)
    torch.testing.assert_close(converted.k_proj.weight, source_k)
    torch.testing.assert_close(converted.v_proj.weight, source_v)
    torch.testing.assert_close(converted.out_proj.weight, source.out_proj.weight)


@pytest.mark.parametrize(
    ("module", "input_value"),
    [
        (nn.Linear(4, 2), torch.randn(3, 4)),
        (nn.Conv1d(2, 3, 3), torch.randn(1, 2, 6)),
        (nn.Conv2d(2, 3, 3), torch.randn(1, 2, 6, 6)),
        (nn.ReLU(), torch.randn(2, 3)),
        (nn.Softmax(dim=-1), torch.randn(2, 3)),
        (nn.LayerNorm(3), torch.randn(2, 3)),
        (nn.AvgPool2d(2), torch.randn(1, 2, 4, 4)),
    ],
)
def test_stock_module_semantics_accept_keyword_only_input(module, input_value):
    result = inspect_model(
        module.eval(),
        Scenario("keyword-input", (), {"input": input_value}),
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )

    assert result.support["gaps"] == []
    assert result.plan.kernels["module:"]["source_count"] == 1
    assert result.verification.succeeded is True
    assert result.verification.output_equivalence is True
    assert result.fully_supported is True


def test_attention_module_semantics_accept_keyword_only_operands():
    model = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    value = torch.randn(2, 3, 4)
    result = inspect_model(
        model,
        Scenario(
            "keyword-attention",
            (),
            {
                "query": value,
                "key": value,
                "value": value,
                "need_weights": False,
            },
        ),
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )

    assert result.support["gaps"] == []
    assert result.plan.kernels["module:"]["name"] == "attention"
    assert result.verification.succeeded is True
    assert result.verification.output_equivalence is True
    assert result.fully_supported is True


def test_functional_scaled_dot_product_attention_routes_as_composite():
    class FunctionalAttention(nn.Module):
        def forward(self, query, key, value):
            return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    query = torch.randn(2, 3, 4, 8)
    scenario = Scenario("sdpa", (query, query, query))
    result = inspect_model(
        FunctionalAttention(),
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )

    route = result.plan.kernels["schema:aten::scaled_dot_product_attention"]
    assert route["name"] == "scaled_dot_product_attention"
    assert route["classification"] == "composite"
    assert result.verification.succeeded is True
    assert result.verification.output_equivalence is True
    assert result.fully_supported is True


@pytest.mark.parametrize(
    ("model", "scenario"),
    [
        (
            nn.LayerNorm((2, 3)),
            Scenario("multi-dim-layer-norm", (torch.randn(1, 2, 3),)),
        ),
        (
            nn.LayerNorm(3, elementwise_affine=False),
            Scenario("non-affine-layer-norm", (torch.randn(1, 2, 3),)),
        ),
        (
            nn.Softmax(dim=None),
            Scenario("implicit-softmax-dim", (torch.randn(1, 2, 3),)),
        ),
        (
            nn.Conv2d(2, 2, 3, padding=1, padding_mode="reflect"),
            Scenario("reflect-convolution", (torch.randn(1, 2, 5, 5),)),
        ),
        (
            nn.MaxPool2d(2, return_indices=True),
            Scenario("pooling-indices", (torch.randn(1, 2, 4, 4),)),
        ),
        (
            nn.MultiheadAttention(4, 2, batch_first=True, add_zero_attn=True),
            Scenario(
                "zero-attention",
                (torch.randn(1, 2, 4),) * 3,
                {"need_weights": False},
            ),
        ),
    ],
)
def test_underconstrained_native_module_variants_remain_explicit_gaps(model, scenario):
    result = inspect_model(
        model,
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )

    assert result.plan.module_decisions == {}
    assert result.plan.unresolved_schemas
    assert any(
        "module invocation constraints" in gap["reason"]
        for gap in result.support["gaps"]
    )


def test_fx_and_export_failures_remain_diagnostics_not_capture_fallback(monkeypatch):
    class UnsupportedModel(nn.Module):
        def forward(self, value):
            return torch.sin(value)

    def fail_graph(*_args, **_kwargs):
        raise RuntimeError("forced graph failure")

    monkeypatch.setattr(torch.fx, "symbolic_trace", fail_graph)
    monkeypatch.setattr(torch.export, "export", fail_graph)
    result = inspect_model(
        UnsupportedModel(),
        Scenario("eager", (torch.ones(3),)),
        InspectionConfig(verify=False),
    )

    assert result.support["capture_complete"] is True
    assert [record.schema for record in result.operations] == ["aten::sin"]
    assert result.diagnostics["fx"]["succeeded"] is False
    assert result.diagnostics["fx"]["error"] == "RuntimeError: details redacted"
    assert result.diagnostics["export"]["succeeded"] is False
    assert result.fully_supported is False
    assert result.support["gaps"][0]["schema"] == "aten::sin"


def test_optional_graph_enrichment_cannot_mutate_authoritative_model_or_verdict():
    class TraceMutator(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU(inplace=False)

        def forward(self, value):
            if isinstance(value, torch.fx.Proxy):
                self.relu.inplace = True
            return self.relu(value)

    scenario = Scenario("relu", (-torch.ones(2),))
    baseline = inspect_model(
        TraceMutator(),
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False),
    )
    source = TraceMutator()
    enriched = inspect_model(
        source,
        scenario,
        InspectionConfig(enable_fx=True, enable_export=False),
    )

    assert enriched.diagnostics["fx"]["succeeded"] is True
    assert source.relu.inplace is False
    assert enriched.plan.module_decisions == baseline.plan.module_decisions
    assert enriched.support["replacement_coverage"] is True
    assert enriched.support["verdict"] == baseline.support["verdict"]
    assert enriched.fully_supported is baseline.fully_supported is True


def test_exact_constraint_matching_uses_captured_scalar_metadata(monkeypatch):
    from qbench import inspection, registry

    def redispatch(func, args, kwargs):
        return func(*args, **kwargs)

    constrained = KernelSpec(
        "sum-dim-one",
        schemas=("aten::sum.dim_IntList",),
        handler=redispatch,
        constraints=lambda args, kwargs: len(args) > 1 and args[1] == [1],
    )
    monkeypatch.setattr(registry, "KERNEL_SPECS", [constrained])
    monkeypatch.setattr(inspection, "KERNEL_SPECS", [constrained])

    class Sum(nn.Module):
        def forward(self, value, *, dim):
            return torch.sum(value, dim=dim)

    result = inspect_model(
        Sum(),
        [
            Scenario("accepted", (torch.ones(2, 3),), {"dim": 1}),
            Scenario("rejected", (torch.ones(2, 3),), {"dim": 0}),
        ],
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )

    accepted = next(
        record for record in result.operations if record.scenario == "accepted"
    )
    rejected = next(
        record for record in result.operations if record.scenario == "rejected"
    )
    assert accepted.classification == "quantized"
    assert accepted.kernel == "sum-dim-one"
    assert rejected.classification == "unsupported"
    assert any(gap["scenario"] == "rejected" for gap in result.support["gaps"])


def test_scalar_arithmetic_is_finite_and_declares_both_encoded_operands():
    class ScalarAdd(nn.Module):
        def __init__(self, scalar):
            super().__init__()
            self.scalar = scalar

        def forward(self, value):
            return torch.add(value, self.scalar)

    scenario = Scenario("scalar", (torch.ones(2),))
    finite = inspect_model(
        ScalarAdd(0.5),
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False),
    )
    route = next(iter(finite.plan.kernels))
    schema = route.removeprefix("schema:")
    assert finite.plan.kernels[route]["input_operands"][schema] == [0, 1]
    assert finite.fully_supported is True

    nonfinite = inspect_model(
        ScalarAdd(float("inf")),
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )
    assert nonfinite.support["replacement_coverage"] is False
    assert nonfinite.operations[0].classification == "unsupported"


def test_custom_autograd_and_registered_custom_operator_are_in_raw_ledger():
    class Multiply(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value * value

    namespace = "qbench_capture_test"
    library = torch.library.Library(namespace, "DEF")
    library.define("scale(Tensor value, float factor) -> Tensor")

    def scale(value, factor):
        return value * factor

    library.impl("scale", scale, "CompositeExplicitAutograd")

    class CustomOps(nn.Module):
        def forward(self, value):
            squared = Multiply.apply(value)
            return torch.ops.qbench_capture_test.scale.default(squared, 2.5)

    records, _ = capture_scenario(CustomOps(), Scenario("custom-op", (torch.ones(3),)))
    schemas = [record.schema for record in records]
    assert "aten::mul.Tensor" in schemas
    assert "qbench_capture_test::scale" in schemas
    custom = next(record for record in records if record.namespace == namespace)
    assert custom.overload == "default"
    assert custom.arguments["args"][1] == 2.5


def test_strict_unresolved_rejection_and_explicit_fallback_stays_partial(monkeypatch):
    from qbench import conversion

    class Sin(nn.Module):
        def forward(self, value):
            return torch.sin(value)

    scenario = Scenario("sin", (torch.ones(3),))
    strict_plan = SimulationPlan(unresolved_schemas=["aten::sin"], scenarios=[scenario])
    with pytest.raises(QBenchError, match="Strict conversion rejected unresolved"):
        build_simulator(Sin(), strict_plan, strict=True)

    fallback_plan = SimulationPlan(
        unresolved_schemas=["aten::sin"],
        allow_fp32_fallback=True,
        scenarios=[scenario],
    )
    with pytest.raises(QBenchError, match="fallback-enabled plan"):
        build_simulator(Sin(), fallback_plan, strict=True)

    monkeypatch.setattr(
        conversion,
        "_convert_modules",
        lambda model, **_kwargs: (copy.deepcopy(model), True, []),
    )
    simulator = build_simulator(Sin(), fallback_plan, strict=False)
    verification = simulator.verify([scenario])
    assert verification.succeeded is True
    assert verification.strict is False
    assert verification.quantized_execution is False
    assert verification.fp32_fallbacks == {"aten::sin": 1}

    result = inspect_model(
        Sin(),
        scenario,
        InspectionConfig(
            allow_fp32_fallback=True,
            enable_fx=False,
            enable_export=False,
            verify=False,
        ),
    )
    assert result.operations[0].classification == "fp32_fallback"
    assert result.fully_supported is False


def test_runtime_handler_is_invoked_and_missing_handler_fails_strictly(monkeypatch):
    from qbench import conversion, registry

    calls = []

    def handler(func, args, kwargs):
        calls.append(func)
        return func(*args, **kwargs)

    spec = KernelSpec("neg", schemas=("aten::neg",), handler=handler)
    monkeypatch.setattr(registry, "KERNEL_SPECS", [spec])
    monkeypatch.setattr(
        conversion,
        "_convert_modules",
        lambda model, **_kwargs: (copy.deepcopy(model), True, []),
    )

    class Neg(nn.Module):
        def forward(self, value):
            return torch.neg(value)

    scenario = Scenario("neg", (torch.ones(2),))
    row = spec.to_dict()
    row.update(source_count=1, scenario_counts={"neg": 1})
    plan = SimulationPlan(
        kernels={"schema:aten::neg": row},
        scenarios=[scenario],
    )
    simulator = build_simulator(Neg(), plan)
    verification = simulator.verify([scenario])
    assert verification.succeeded is True
    assert verification.output_equivalence is True
    assert len(calls) == 1

    missing = KernelSpec("neg", schemas=("aten::neg",), handler=None)
    monkeypatch.setattr(registry, "KERNEL_SPECS", [missing])
    missing_row = missing.to_dict()
    missing_row.update(source_count=1, scenario_counts={"neg": 1})
    broken_plan = SimulationPlan(
        kernels={"schema:aten::neg": missing_row},
        scenarios=[scenario],
    )
    with pytest.raises(QBenchError, match="ready maintained runtime handler"):
        build_simulator(Neg(), broken_plan)


def test_default_functional_add_uses_the_maintained_qbench_kernel(monkeypatch):
    from qbench.ops.quant_arithmetic import QuantAdd

    calls = 0
    original_forward = QuantAdd.forward

    def observed_forward(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(QuantAdd, "forward", observed_forward)

    class FunctionalAdd(nn.Module):
        def forward(self, value):
            return value + 2.0

    result = inspect_model(
        FunctionalAdd(),
        Scenario("add", (torch.ones(2),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )
    assert result.fully_supported is True
    assert result.verification.succeeded is True
    assert calls == 1


def test_functional_linear_and_addmm_use_exact_maintained_handlers(monkeypatch):
    from qbench.ops.quant_arithmetic import QuantAdd
    from qbench.ops.quant_linear import QuantLinear

    calls = {"linear": 0, "add": 0}

    def observe(cls, name):
        original = cls.forward

        def forward(self, *args, **kwargs):
            calls[name] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "forward", forward)

    observe(QuantLinear, "linear")
    observe(QuantAdd, "add")

    class FunctionalLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3, 4))
            self.bias = nn.Parameter(torch.randn(3))

        def forward(self, value):
            return torch.nn.functional.linear(value, self.weight, self.bias)

    class FunctionalAddmm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 3))
            self.bias = nn.Parameter(torch.randn(3))

        def forward(self, value):
            return torch.addmm(self.bias, value, self.weight)

    torch.manual_seed(17)
    linear = FunctionalLinear().eval()
    addmm = FunctionalAddmm().eval()
    linear_weight = linear.weight.detach().clone()
    addmm_weight = addmm.weight.detach().clone()
    linear_input = torch.randn(2, 4)
    addmm_input = torch.randn(2, 4)
    rng_before = torch.random.get_rng_state().clone()

    linear_result = inspect_model(
        linear,
        Scenario("linear", (linear_input,)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )
    addmm_result = inspect_model(
        addmm,
        Scenario("addmm", (addmm_input,)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )

    assert linear_result.operations[0].schema == "aten::linear"
    assert linear_result.operations[0].kernel == "linear"
    assert linear_result.fully_supported is True
    assert linear_result.verification.output_equivalence is True
    assert addmm_result.operations[0].schema == "aten::addmm"
    assert addmm_result.operations[0].kernel == "addmm"
    assert addmm_result.fully_supported is True
    assert addmm_result.verification.output_equivalence is True
    # addmm is deliberately decomposed through the weight-aware linear path,
    # so its right-hand matrix receives weight-policy calibration/evidence.
    assert calls == {"linear": 2, "add": 1}
    assert torch.equal(linear.weight, linear_weight)
    assert torch.equal(addmm.weight, addmm_weight)
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    simulator = build_simulator(linear, linear_result.plan)
    run_rng_before = torch.random.get_rng_state().clone()
    simulator.run(Scenario("standalone", (linear_input,)))
    assert calls["linear"] == 3
    assert torch.equal(torch.random.get_rng_state(), run_rng_before)


def test_weighted_functional_handler_honors_quantization_runtime_context(monkeypatch):
    from qbench.registry import find_kernel
    from qbench.runtime import simulation_quantization
    from qbench.ops.quant_linear import QuantLinear

    observed = {}

    def calibrate(self):
        observed["calibrated"] = True

    def forward(self, value):
        observed.update(
            input_quantization=self.input_quantization,
            output_quantization=self.output_quantization,
            weight_quantization=self.weight_quantization,
        )
        return torch.nn.functional.linear(value, self.weight, self.bias)

    monkeypatch.setattr(QuantLinear, "calibrate_weights", calibrate)
    monkeypatch.setattr(QuantLinear, "forward", forward)
    value = torch.randn(2, 4)
    weight = torch.randn(3, 4)
    bias = torch.randn(3)
    metadata = tuple(
        {
            "kind": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": False,
        }
        for tensor in (value, weight, bias)
    )
    spec = find_kernel("aten::linear", metadata, {})
    assert spec is not None and spec.handler is not None
    rng_before = torch.random.get_rng_state().clone()

    with simulation_quantization(True):
        output = spec.handler(
            torch.ops.aten.linear.default,
            (value, weight, bias),
            {},
        )

    torch.testing.assert_close(output, torch.nn.functional.linear(value, weight, bias))
    assert observed == {
        "calibrated": True,
        "input_quantization": True,
        "output_quantization": False,
        "weight_quantization": True,
    }
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize(
    ("schema", "operation"),
    [
        ("aten::layer_norm", "layer_norm"),
        ("aten::batch_norm", "batch_norm"),
    ],
)
def test_normalization_handlers_use_supported_rank_one_weight_mode(
    monkeypatch, schema, operation
):
    from qbench.capture import value_metadata
    from qbench.registry import find_kernel
    from qbench.runtime import simulation_quantization
    from qbench.ops.quant_bn import QuantBatchNorm2d
    from qbench.ops.quant_ln import QuantLayerNorm

    implementation = QuantLayerNorm if operation == "layer_norm" else QuantBatchNorm2d
    observed = {}

    def calibrate(self):
        observed["weight_mode"] = self.weight_mode

    monkeypatch.setattr(implementation, "calibrate_weights", calibrate)
    value = torch.randn(2, 3, 4, 4)
    weight = torch.randn(3 if operation == "batch_norm" else 4)
    bias = torch.randn_like(weight)
    running_mean = torch.randn(3)
    running_var = torch.rand(3) + 0.5
    if operation == "layer_norm":
        concrete = (value, [4], weight, bias, 1e-5, True)
    else:
        concrete = (
            value,
            weight,
            bias,
            running_mean,
            running_var,
            False,
            0.1,
            1e-5,
            True,
        )
    metadata = value_metadata(concrete)
    spec = find_kernel(schema, tuple(metadata), {})
    assert spec is not None and spec.handler is not None

    # Bypass the CUDA-only numeric quantizer: this test owns the maintained
    # kernel configuration boundary and checks that calibration receives a
    # rank-compatible mode.
    monkeypatch.setattr(implementation, "forward", lambda self, input: input)
    with simulation_quantization(True):
        spec.handler(getattr(torch.ops.aten, operation).default, concrete, {})

    assert observed == {"weight_mode": "tensor"}


def test_affine_free_batch_norm_does_not_require_weight_evidence():
    model = nn.BatchNorm1d(3, affine=False).eval()
    scenario = Scenario("batch-norm", (torch.randn(2, 3),))
    result = inspect_model(
        model,
        scenario,
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            quantization_enabled=True,
        ),
    )

    assert result.verification.succeeded is True
    assert not any("stage weight" in error for error in result.verification.errors)
    assert result.verification.quantized_execution is False


def test_cpu_quantized_inspection_runs_only_the_disabled_routing_dry_run(monkeypatch):
    from qbench import conversion
    from qbench.cli import EXIT_PARTIAL, _result_exit

    conversion_modes = []
    original_conversion = conversion._convert_modules

    def observe_conversion(model, *, quantization_enabled, **kwargs):
        conversion_modes.append(quantization_enabled)
        return original_conversion(
            model, quantization_enabled=quantization_enabled, **kwargs
        )

    monkeypatch.setattr(conversion, "_convert_modules", observe_conversion)
    result = inspect_model(
        nn.Linear(2, 2),
        Scenario("cpu", (torch.ones(1, 2),)),
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            quantization_enabled=True,
            device="cpu",
        ),
    )

    assert result.plan.quantization_enabled is True
    assert conversion_modes == [False]
    assert result.verification.succeeded is True
    assert result.verification.output_equivalence is True
    assert result.verification.quantized_execution is False
    assert result.support["strict_realization"] is True
    assert result.support["routing_dry_run_verified"] is True
    assert result.support["quantized_execution_verified"] is False
    assert result.support["verdict"] == "partial_or_unsupported"
    assert result.diagnostics["verification"] == {
        "mode": "quantization_disabled_routing_dry_run",
        "actual_quantized_execution_attempted": False,
    }
    assert _result_exit(result) == EXIT_PARTIAL


def test_normalization_handlers_preserve_explicit_cudnn_flag(monkeypatch):
    from qbench.registry import _batch_norm_handler, _layer_norm_handler
    from qbench.runtime import simulation_quantization
    from qbench.ops.quant_bn import QuantBatchNorm1d
    from qbench.ops.quant_ln import QuantLayerNorm

    observed = []
    original_layer_norm = QuantLayerNorm.forward
    original_batch_norm = QuantBatchNorm1d.forward

    def observe_layer_norm(self, value):
        observed.append(("layer_norm", torch.backends.cudnn.enabled))
        return original_layer_norm(self, value)

    def observe_batch_norm(self, value):
        observed.append(("batch_norm", torch.backends.cudnn.enabled))
        return original_batch_norm(self, value)

    monkeypatch.setattr(QuantLayerNorm, "forward", observe_layer_norm)
    monkeypatch.setattr(QuantBatchNorm1d, "forward", observe_batch_norm)
    requested = not torch.backends.cudnn.enabled
    value = torch.randn(2, 3)
    with simulation_quantization(False):
        _layer_norm_handler(
            torch.ops.aten.layer_norm.default,
            (value, [3], torch.ones(3), torch.zeros(3), 1e-5, requested),
            {},
        )
        _batch_norm_handler(
            torch.ops.aten.batch_norm.default,
            (
                value,
                torch.ones(3),
                torch.zeros(3),
                torch.zeros(3),
                torch.ones(3),
                False,
                0.1,
                1e-5,
                requested,
            ),
            {},
        )

    assert observed == [
        ("layer_norm", requested),
        ("batch_norm", requested),
    ]


@pytest.mark.parametrize(
    ("schema", "spatial_rank"),
    [
        ("aten::conv1d", 1),
        ("aten::conv2d", 2),
        ("aten::convolution", 2),
    ],
)
def test_functional_convolutions_use_exact_maintained_handlers(
    monkeypatch, schema, spatial_rank
):
    from qbench.ops.quant_conv import QuantConv2d
    from qbench.ops.quant_conv1d import QuantConv1d

    calls = 0
    implementation = QuantConv1d if spatial_rank == 1 else QuantConv2d
    original_forward = implementation.forward

    def observed_forward(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(implementation, "forward", observed_forward)

    class FunctionalConvolution(nn.Module):
        def __init__(self):
            super().__init__()
            kernel_shape = (3, 2, 3) if spatial_rank == 1 else (3, 2, 3, 2)
            self.weight = nn.Parameter(torch.randn(*kernel_shape))
            self.bias = nn.Parameter(torch.randn(3))

        def forward(self, value):
            if schema == "aten::conv1d":
                return torch.ops.aten.conv1d.default(
                    value, self.weight, self.bias, [2], [1], [1], 1
                )
            if schema == "aten::conv2d":
                return torch.ops.aten.conv2d.default(
                    value, self.weight, self.bias, [2, 1], [1, 0], [1, 1], 1
                )
            return torch.ops.aten.convolution.default(
                value,
                self.weight,
                self.bias,
                [2, 1],
                [1, 0],
                [1, 1],
                False,
                [0, 0],
                1,
            )

    model = FunctionalConvolution().eval()
    input_shape = (2, 2, 9) if spatial_rank == 1 else (2, 2, 9, 8)
    weight_before = model.weight.detach().clone()
    input_value = torch.randn(*input_shape)
    rng_before = torch.random.get_rng_state().clone()
    result = inspect_model(
        model,
        Scenario(schema, (input_value,)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )

    assert result.operations[0].schema == schema
    assert result.operations[0].kernel == schema.removeprefix("aten::")
    assert result.fully_supported is True
    assert result.verification.output_equivalence is True
    assert calls == 1
    assert torch.equal(model.weight, weight_before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize(
    "model,scenario,expected_schema",
    [
        (
            type(
                "Float64Linear",
                (nn.Module,),
                {
                    "forward": lambda self, value: torch.nn.functional.linear(
                        value, self.weight
                    ),
                },
            )(),
            Scenario("float64", (torch.ones(2, 3, dtype=torch.float64),)),
            "aten::linear",
        ),
        (
            type(
                "ScaledAddmm",
                (nn.Module,),
                {
                    "forward": lambda self, value: torch.addmm(
                        self.bias, value, self.weight, alpha=2
                    ),
                },
            )(),
            Scenario("scaled", (torch.ones(2, 3),)),
            "aten::addmm",
        ),
    ],
)
def test_unsupported_functional_linear_variants_remain_explicit_gaps(
    model, scenario, expected_schema
):
    if expected_schema == "aten::linear":
        model.weight = nn.Parameter(torch.ones(2, 3, dtype=torch.float64))
    else:
        model.weight = nn.Parameter(torch.ones(3, 2))
        model.bias = nn.Parameter(torch.ones(2))

    result = inspect_model(
        model,
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )

    assert result.plan.kernels == {}
    assert result.support["gaps"][0]["schema"] == expected_schema
    assert "constraints not satisfied" in result.support["gaps"][0]["reason"]


def test_safe_functional_norm_handlers_use_maintained_qbench_ops(monkeypatch):
    from qbench.ops.quant_bn import QuantBatchNorm2d
    from qbench.ops.quant_ln import QuantLayerNorm

    calls = {"layer_norm": 0, "batch_norm": 0}
    original_layer_norm = QuantLayerNorm.forward
    original_batch_norm = QuantBatchNorm2d.forward

    def layer_norm_forward(self, *args, **kwargs):
        calls["layer_norm"] += 1
        return original_layer_norm(self, *args, **kwargs)

    def batch_norm_forward(self, *args, **kwargs):
        calls["batch_norm"] += 1
        return original_batch_norm(self, *args, **kwargs)

    monkeypatch.setattr(QuantLayerNorm, "forward", layer_norm_forward)
    monkeypatch.setattr(QuantBatchNorm2d, "forward", batch_norm_forward)

    class FunctionalLayerNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4))
            self.bias = nn.Parameter(torch.randn(4))

        def forward(self, value):
            return torch.ops.aten.layer_norm.default(
                value, [4], self.weight, self.bias, 1e-5, True
            )

    class FunctionalBatchNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3))
            self.bias = nn.Parameter(torch.randn(3))
            self.register_buffer("running_mean", torch.randn(3))
            self.register_buffer("running_var", torch.rand(3) + 0.5)

        def forward(self, value):
            return torch.ops.aten.batch_norm.default(
                value,
                self.weight,
                self.bias,
                self.running_mean,
                self.running_var,
                False,
                0.1,
                1e-5,
                True,
            )

    cases = [
        (FunctionalLayerNorm(), torch.randn(2, 3, 4), "normalization"),
        (FunctionalBatchNorm(), torch.randn(2, 3, 4, 4), "batch_norm"),
    ]
    for index, (model, value, kernel) in enumerate(cases):
        result = inspect_model(
            model.eval(),
            Scenario(f"norm-{index}", (value,)),
            InspectionConfig(enable_fx=False, enable_export=False, verify=True),
        )
        assert result.operations[0].kernel == kernel
        assert result.fully_supported is True
        assert result.verification.output_equivalence is True

    assert calls == {"layer_norm": 1, "batch_norm": 1}


@pytest.mark.parametrize(
    ("schema", "expected_kernel"),
    [
        ("aten::max_pool2d_with_indices", "max_pool2d"),
        ("aten::avg_pool2d", "avg_pool2d"),
        ("aten::adaptive_avg_pool2d", "adaptive_avg_pool2d"),
        ("aten::dropout", "dropout"),
    ],
)
def test_safe_functional_pool_and_eval_dropout_handlers(schema, expected_kernel):
    class Functional(nn.Module):
        def forward(self, value):
            if schema == "aten::max_pool2d_with_indices":
                return torch.ops.aten.max_pool2d_with_indices.default(
                    value, [2, 2], [], [0, 0], [1, 1], False
                )
            if schema == "aten::avg_pool2d":
                return torch.ops.aten.avg_pool2d.default(
                    value, [2, 2], [], [0, 0], False, True, None
                )
            if schema == "aten::adaptive_avg_pool2d":
                return torch.ops.aten.adaptive_avg_pool2d.default(value, [2, 3])
            return torch.ops.aten.dropout.default(value, 0.5, False)

    result = inspect_model(
        Functional(),
        Scenario(schema, (torch.randn(2, 3, 8, 8),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )

    assert result.operations[0].schema == schema
    assert result.operations[0].kernel == expected_kernel
    assert result.fully_supported is True
    assert result.verification.output_equivalence is True


@pytest.mark.parametrize(
    ("schema", "invoke"),
    [
        (
            "aten::native_layer_norm",
            lambda value: torch.ops.aten.native_layer_norm.default(
                value, [4], None, None, 1e-5
            ),
        ),
        (
            "aten::native_dropout",
            lambda value: torch.ops.aten.native_dropout.default(value, 0.5, False),
        ),
    ],
)
def test_unimplemented_tuple_functional_variants_are_known_unready(schema, invoke):
    class Functional(nn.Module):
        def forward(self, value):
            return invoke(value)

    result = inspect_model(
        Functional(),
        Scenario(schema, (torch.randn(2, 3, 4),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )

    assert result.plan.kernels == {}
    assert result.support["gaps"][0]["schema"] == schema
    assert "KernelSpec is not ready" in result.support["gaps"][0]["reason"]


def test_planned_composite_suppresses_generated_child_internals():
    class GeneratedComposite(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = nn.Linear(2, 2, bias=False)

        def forward(self, value):
            return self.child(value)

    GeneratedComposite.__module__ = "qbench.ops.generated"
    model = GeneratedComposite()
    decision = f"module:{type(model).__module__}.{type(model).__qualname__}"
    plan = SimulationPlan(
        kernels={
            "module:": _route_row(
                "generated", {"composite": 1}, classification="composite"
            )
        },
        module_decisions={"": decision},
    )
    simulator = _create_simulator(
        model,
        plan,
        strict=True,
        converted=True,
        build_errors=[],
        reference_model=copy.deepcopy(model),
    )
    verification = simulator.verify([Scenario("composite", (torch.ones(1, 2),))])
    assert verification.succeeded is True
    assert verification.realized_operations >= 1
    assert verification.unexpected_operations == []


def test_quantized_verification_checks_structure_without_requiring_close_values():
    class QuantizedShift(nn.Module):
        def __init__(self):
            super().__init__()
            self.emit_quantization_evidence = True

        def forward(self, value):
            if self.emit_quantization_evidence:
                from qbench.runtime import record_quantization

                record_quantization(
                    q_type="fp8_e4m3",
                    stage="input",
                    policy_q_type="fp8_e4m3",
                    policy_mode="tensor",
                    policy_chunk_size=128,
                    device="cuda:0",
                )
            return {"prediction": value + 1}

    QuantizedShift.__module__ = "qbench.ops.generated"
    converted = QuantizedShift()
    decision = f"module:{type(converted).__module__}.{type(converted).__qualname__}"
    plan = SimulationPlan(
        kernels={"module:": _route_row("shift", {"quantized": 1, "unmarked": 1})},
        module_decisions={"": decision},
        quantization_enabled=True,
        quantization_policy=QuantizationPolicy(quantize_first_layer=True),
    )
    simulator = _create_simulator(
        converted,
        plan,
        strict=True,
        converted=True,
        build_errors=[],
        reference_model=nn.Identity(),
    )
    verification = simulator.verify([Scenario("quantized", (torch.ones(2),))])
    # The mapping output structure deliberately differs here (mapping vs tensor),
    # so even quantized verification must reject it.
    assert verification.succeeded is False
    assert verification.output_structure is False
    assert any(
        "independent quantization-disabled simulator" in error
        for error in verification.errors
    )

    class ReferenceMapping(nn.Module):
        def forward(self, value):
            return {"prediction": value}

    simulator = _create_simulator(
        converted,
        plan,
        strict=False,
        converted=True,
        build_errors=[],
        reference_model=ReferenceMapping(),
    )
    verification = simulator.verify([Scenario("quantized", (torch.ones(2),))])
    assert verification.succeeded is True
    assert verification.output_structure is True
    assert verification.output_equivalence is False
    assert verification.quantized_execution is True
    assert verification.quantized_routes == {"module:": 1}

    converted.emit_quantization_evidence = False
    simulator = _create_simulator(
        converted,
        plan,
        strict=False,
        converted=True,
        build_errors=[],
        reference_model=ReferenceMapping(),
    )
    verification = simulator.verify([Scenario("unmarked", (torch.ones(2),))])
    assert verification.succeeded is False
    assert verification.quantized_execution is False
    assert verification.quantized_routes == {}


def test_quantized_verification_requires_every_declared_input_operand():
    class PartialEvidence(nn.Module):
        def forward(self, value):
            from qbench.runtime import record_quantization

            record_quantization(
                stage="input",
                operand_index=0,
                policy_q_type="fp8_e4m3",
                policy_mode="tensor",
                policy_chunk_size=128,
                device="cuda:0",
            )
            return value

    PartialEvidence.__module__ = "qbench.ops.generated"
    converted = PartialEvidence()
    decision = f"module:{type(converted).__module__}.{type(converted).__qualname__}"
    row = _route_row("binary", {"binary": 1})
    row["input_operands"] = {"module": [0, 1]}
    simulator = _create_simulator(
        converted,
        SimulationPlan(
            kernels={"module:": row},
            module_decisions={"": decision},
            quantization_enabled=True,
            quantization_policy=QuantizationPolicy(quantize_first_layer=True),
        ),
        strict=False,
        converted=True,
        build_errors=[],
        reference_model=nn.Identity(),
    )

    verification = simulator.verify([Scenario("binary", (torch.ones(2),))])

    assert verification.succeeded is False
    assert verification.quantized_execution is False
    assert any("input operand 1" in error for error in verification.errors)
    assert any("evidence missing" in error for error in verification.errors)


def test_quantized_build_requires_a_separate_disabled_equivalence_pass(monkeypatch):
    from qbench import conversion

    scenario = Scenario("value-gate", (torch.ones(1, 2),))
    source = nn.Linear(2, 2)
    inspected = inspect_model(
        copy.deepcopy(source),
        scenario,
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            verify=False,
            quantization_enabled=True,
        ),
    )
    original_conversion = conversion._convert_modules

    def deliberately_wrong_conversion(model, *, quantization_enabled, **kwargs):
        # Use the maintained replacement in both phases, but corrupt only the
        # disabled companion.  The quantized phase's looser numeric comparison
        # must not hide this conversion error.
        converted, did_convert, errors = original_conversion(
            model, quantization_enabled=False, **kwargs
        )
        if not quantization_enabled:
            with torch.no_grad():
                converted.bias.add_(1)
        return converted, did_convert, errors

    monkeypatch.setattr(conversion, "_convert_modules", deliberately_wrong_conversion)
    simulator = build_simulator(source, inspected.plan)

    verification = simulator.verify([scenario])

    assert verification.succeeded is False
    assert verification.output_structure is True
    assert verification.output_equivalence is False
    assert verification.quantized_execution is False
    assert any(
        "quantization-disabled dry run" in error for error in verification.errors
    )


def test_simulator_constructor_is_private_to_the_validating_builder():
    with pytest.raises(QBenchError, match="use build_simulator"):
        Simulator(
            nn.Identity(),
            SimulationPlan(),
            strict=True,
            converted=True,
            build_errors=[],
        )


def test_build_rejects_stale_module_forward_hooks_and_constraints():
    scenario = Scenario("linear", (torch.ones(1, 3),))
    source = nn.Linear(3, 2)
    inspected = inspect_model(
        source,
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )

    source.forward = types.MethodType(
        lambda self, value: torch.sin(value[..., :2]), source
    )
    with pytest.raises(QBenchError, match="untouched forward"):
        build_simulator(source, inspected.plan)

    hooked = nn.Linear(3, 2)
    hooked_plan = inspect_model(
        hooked,
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    ).plan
    hooked.register_forward_pre_hook(lambda _module, args: args)
    with pytest.raises(QBenchError, match="untouched forward"):
        build_simulator(hooked, hooked_plan)

    double_model = nn.Linear(3, 2).double()
    with pytest.raises(QBenchError, match="captured constraints"):
        build_simulator(double_model, inspected.plan)

    relu_scenario = Scenario("relu", (torch.ones(2),))
    relu_plan = inspect_model(
        nn.ReLU(inplace=False),
        relu_scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    ).plan
    with pytest.raises(QBenchError, match="captured constraints"):
        build_simulator(nn.ReLU(inplace=True), relu_plan)


def test_build_rejects_counts_for_undeclared_scenarios_and_tampered_invocations():
    class FunctionalRelu(nn.Module):
        def forward(self, value):
            return torch.relu(value)

    scenario = Scenario("captured", (-torch.ones(2),))
    functional = FunctionalRelu()
    functional_plan = inspect_model(
        functional,
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    ).plan
    route = functional_plan.kernels["schema:aten::relu"]
    route["scenario_counts"]["phantom"] = 100
    route["source_count"] += 100
    route["module_path_counts"][""] += 100

    with pytest.raises(QBenchError, match="scenarios not declared"):
        build_simulator(functional, functional_plan)

    module = nn.Linear(2, 2)
    module_plan = inspect_model(
        module,
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    ).plan
    invocation = module_plan.kernels["module:"]["module_invocations"][""][0]
    invocation["scenario"] = "phantom"

    with pytest.raises(QBenchError, match="invocation scenarios"):
        build_simulator(module, module_plan)


def test_strict_verification_includes_zero_operation_scenarios():
    class DynamicIdentity(nn.Module):
        def forward(self, value, *, identity):
            return value if identity else torch.relu(value)

    scenarios = [
        Scenario("identity", (torch.tensor([-1.0, 1.0]),), {"identity": True}),
        Scenario("compute", (torch.tensor([-1.0, 1.0]),), {"identity": False}),
    ]
    result = inspect_model(
        DynamicIdentity(),
        scenarios,
        InspectionConfig(enable_fx=False, enable_export=False),
    )

    assert result.plan.scenario_names == ["identity", "compute"]
    assert result.verification.succeeded is True
    assert result.verification.output_equivalence is True
    assert result.support["scenario_coverage"]["identity"]["operation_count"] == 0
    assert result.fully_supported is True


def test_strict_verification_rejects_missing_or_wrong_zero_operation_scenarios():
    scenario = Scenario("captured", (torch.ones(2),))
    source = nn.Identity()
    result = inspect_model(
        source,
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )

    assert result.plan.kernels == {}
    assert result.plan.scenario_names == ["captured"]

    empty = build_simulator(source, result.plan).verify([])
    assert empty.succeeded is False
    assert any("at least one scenario" in error for error in empty.errors)
    assert any("do not match the captured plan" in error for error in empty.errors)

    wrong = build_simulator(source, result.plan).verify(
        [Scenario("different", (torch.ones(2),))]
    )
    assert wrong.succeeded is False
    assert any("do not match the captured plan" in error for error in wrong.errors)

    correct = build_simulator(source, result.plan).verify([scenario])
    assert correct.succeeded is True


def test_quantized_simulator_installs_producer_transport_on_extracted_root():
    from qbench import conversion

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 2)

        def forward(self, value):
            return self.proj(value)

    reference = Projection().eval()
    converted, did_convert, errors = conversion._convert_modules(
        reference,
        module_paths=("proj",),
        quantization_enabled=False,
    )
    assert did_convert is True
    assert errors == []
    assert getattr(converted, "_qbench_activation_transport_guarded", False)

    # Exercise the quantized public path without invoking the CUDA-only legacy
    # module boundary. CPU uses the same stage plan through reference transport.
    converted.proj.input_quantization = True
    converted.proj.input_q_type = "fp8_e4m3"
    converted.proj.input_chunk_size = 128
    decision = (
        f"module:{type(converted.proj).__module__}.{type(converted.proj).__qualname__}"
    )
    plan = SimulationPlan(
        kernels={"module:proj": _route_row("linear", {"transport": 1})},
        module_decisions={"proj": decision},
        quantization_enabled=True,
        quantization_policy=QuantizationPolicy(quantize_first_layer=True),
    )
    simulator = _create_simulator(
        converted,
        plan,
        strict=False,
        converted=True,
        build_errors=[],
        reference_model=copy.deepcopy(reference),
    )
    # Width three exercises a non-power-of-two operand while strict dispatch
    # auditing is active.
    scenario = Scenario("transport", (torch.randn(2, 3),))

    with pytest.raises(AssertionError, match="producer-stage hardware transport"):
        converted(*scenario.args)

    verification = simulator.verify([scenario])

    assert verification.succeeded is True
    # CPU transport proves exact routing/policy but is reference arithmetic;
    # only the CUDA codec can satisfy actual quantized-execution evidence.
    assert verification.quantized_execution is False
    assert verification.quantized_routes["module:proj"] >= 1
    assert verification.unexpected_operations == []
    stats = simulator.activation_transport_stats()
    assert stats["active"] is True
    assert stats["transport"] == "reference"
    assert stats["transmission_count"] >= 1
    assert converted.proj.input_quantization is False
    assert converted.proj._qbench_activation_transport_active is True

    with pytest.raises(QBenchError, match="Simulator.run"):
        converted(*scenario.args)

    simulator.close()
    assert converted.proj.input_quantization is True
    assert not hasattr(converted.proj, "_qbench_activation_transport_active")


@pytest.mark.parametrize("inplace", [False, True])
def test_structural_dropout_preserves_identity_during_quantized_verification(inplace):
    class AliasSensitiveDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(inplace=inplace)
            self.relu = nn.ReLU()

        def forward(self, value):
            dropped = self.drop(value)
            return dropped is value, self.relu(dropped)

    result = inspect_model(
        AliasSensitiveDropout(),
        Scenario("alias", (torch.ones(2),)),
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            quantization_enabled=True,
        ),
    )

    assert result.verification.succeeded is True
    assert result.verification.output_structure is True
    assert result.verification.quantized_execution is False
    assert result.fully_supported is False


def test_eager_transport_honors_first_layer_and_per_layer_boundary_policy():
    from qbench.conversion import _activation_transport_configuration
    from qbench.ops.quant_activations import QuantReLU
    from qbench.ops.quant_conv import QuantConv2d

    class Boundaries(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = QuantConv2d(3, 4, 3)
            self.first.is_first_layer = True
            self.first.quantize_first_layer = False
            self.first.input_quantization = True
            self.first.input_q_type = "fp8_e4m3"
            self.first.input_chunk_size = 4
            self.enabled = QuantReLU()
            self.enabled.input_quantization = True
            self.enabled.input_q_type = "fp8_e4m3"
            self.enabled.input_chunk_size = 4
            self.disabled = QuantReLU()
            self.disabled.input_quantization = False
            self.disabled.input_q_type = "fp8_e4m3"

    boundaries = _activation_transport_configuration(Boundaries())

    assert boundaries == {
        "first": {
            "input_quantization": {
                "q_type": "fp8_e4m3",
                "mode": "tensor",
                "chunk_size": 4,
            }
        },
        "enabled": {
            "input_quantization": {
                "q_type": "fp8_e4m3",
                "mode": "tensor",
                "chunk_size": 4,
            }
        },
    }


def test_first_semantic_functional_route_skips_only_input_boundary(monkeypatch):
    from qbench.ops.quant_softmax import QuantSoftmax

    hardware_paths = []
    original_forward = QuantSoftmax.forward

    def observed_forward(self, value):
        hardware_paths.append(self.hardware_arithmetic_enabled())
        return original_forward(self, value)

    monkeypatch.setattr(QuantSoftmax, "forward", observed_forward)

    class FunctionalSoftmax(nn.Module):
        def forward(self, value):
            return torch.softmax(value, dim=-1)

    scenario = Scenario("softmax", (torch.randn(2, 4),))
    result = inspect_model(
        FunctionalSoftmax(),
        scenario,
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            verify=False,
            quantization_enabled=True,
            quantization_policy=QuantizationPolicy(quantize_first_layer=False),
        ),
    )
    simulator = build_simulator(FunctionalSoftmax(), result.plan)
    verification = simulator.verify([scenario])

    assert verification.succeeded is True
    assert verification.quantized_execution is False
    # The independent disabled companion takes the native branch first; the
    # q-enabled phase must take the maintained hardware-arithmetic branch.
    assert hardware_paths[-1] is True
    assert any(
        event["route"].startswith("schema:") and event.get("stage") == "input_skipped"
        for event in simulator._quantization_events
    )
    assert any(
        event.get("stage") == "composite" for event in simulator._quantization_events
    )
    simulator.close()


def test_softmax_functional_evidence_uses_activation_policy_axis(monkeypatch):
    from qbench import inspection
    from qbench.runtime import record_quantization
    from qbench.ops.quant_softmax import QuantSoftmax

    def identity_codec(self, value, *args, **kwargs):
        if self.input_quantization:
            record_quantization(
                stage="input",
                policy_q_type=self.input_q_type,
                policy_mode=self.input_mode,
                policy_chunk_size=self.input_chunk_size,
                device="cuda:0",
            )
        return value

    monkeypatch.setattr(QuantSoftmax, "quantize_input", identity_codec)
    # This test isolates the quantized-evidence policy logic.  Stock CPU
    # inspection intentionally performs only a quantization-disabled routing
    # dry run and is covered separately.
    monkeypatch.setattr(inspection, "_inspection_device_type", lambda *_: "cuda")

    class FunctionalSoftmax(nn.Module):
        def forward(self, value):
            return torch.softmax(value, dim=-1)

    policy = QuantizationPolicy(
        quant_mode="channel",
        act_mode="tensor",
        quantize_first_layer=True,
    )
    scenario = Scenario("softmax", (torch.randn(2, 4),))
    result = inspect_model(
        FunctionalSoftmax(),
        scenario,
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            verify=True,
            quantization_enabled=True,
            quantization_policy=policy,
        ),
    )

    softmax_route = next(
        route for route, row in result.plan.kernels.items() if row["name"] == "softmax"
    )
    assert result.plan.kernels[softmax_route]["activation_policy"] is True
    assert result.verification.succeeded is True
    assert not any("policy mismatch" in error for error in result.verification.errors)
    assert result.verification.quantized_execution is True


def test_eager_attention_transport_never_quantizes_semantic_masks():
    from qbench import conversion

    reference = nn.MultiheadAttention(4, 2, batch_first=True).eval()
    converted, did_convert, errors = conversion._convert_modules(
        reference,
        module_paths=("",),
        quantization_enabled=False,
    )
    assert did_convert is True
    assert errors == []
    converted.input_quantization = True
    converted.input_q_type = "fp8_e4m3"
    converted.input_chunk_size = 128
    decision = f"module:{type(converted).__module__}.{type(converted).__qualname__}"
    simulator = _create_simulator(
        converted,
        SimulationPlan(
            kernels={
                "module:": _route_row(
                    "attention",
                    {"masked-attention": 1},
                    classification="composite",
                )
            },
            module_decisions={"": decision},
            quantization_enabled=True,
            quantization_policy=QuantizationPolicy(quantize_first_layer=True),
        ),
        strict=False,
        converted=True,
        build_errors=[],
        reference_model=copy.deepcopy(reference),
    )
    value = torch.randn(1, 2, 4)
    mask = torch.tensor([[0.0, float("-inf")], [0.0, 0.0]])
    scenario = Scenario(
        "masked-attention",
        (value, value, value),
        {"need_weights": False, "attn_mask": mask},
    )

    verification = simulator.verify([scenario])
    output, _weights = simulator.run(scenario)

    assert verification.succeeded is True
    assert verification.quantized_execution is False
    assert torch.isfinite(output).all()
    # Aliased q/k/v are transmitted once; the floating-point mask is excluded.
    assert simulator.activation_transport_stats()["transmission_count"] == 2
    input_events = [
        event
        for event in simulator._quantization_events
        if event.get("stage") == "input" and event.get("operand_indices")
    ]
    assert any(event["operand_indices"] == [0, 1, 2] for event in input_events)
    simulator.close()


def test_producer_transport_plans_a_quantized_semantic_root():
    from qbench import conversion

    reference = nn.Linear(4, 3).eval()
    converted, did_convert, errors = conversion._convert_modules(
        reference,
        module_paths=("",),
        quantization_enabled=False,
    )
    assert did_convert is True
    assert errors == []
    converted.input_quantization = True
    converted.input_q_type = "fp8_e4m3"
    converted.input_chunk_size = 128
    decision = f"module:{type(converted).__module__}.{type(converted).__qualname__}"
    simulator = _create_simulator(
        converted,
        SimulationPlan(
            kernels={"module:": _route_row("linear", {"root_transport": 1})},
            module_decisions={"": decision},
            quantization_enabled=True,
            quantization_policy=QuantizationPolicy(quantize_first_layer=True),
        ),
        strict=False,
        converted=True,
        build_errors=[],
        reference_model=copy.deepcopy(reference),
    )

    verification = simulator.verify([Scenario("root_transport", (torch.randn(2, 4),))])

    assert verification.succeeded is True
    assert verification.quantized_execution is False
    assert simulator.activation_transport_stats()["transmission_count"] >= 1
    simulator.close()


def test_eager_transport_preserves_dynamic_python_control_flow_without_fx():
    from qbench import conversion

    class DynamicProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 3)

        def forward(self, value, *, negate):
            if negate:
                # The branch is intentionally data-independent at execution,
                # but still forces an FX Proxy boolean conversion if traced.
                value = value
            return self.proj(value)

    reference = DynamicProjection().eval()
    converted, did_convert, errors = conversion._convert_modules(
        reference,
        module_paths=("proj",),
        quantization_enabled=False,
    )
    assert did_convert is True
    assert errors == []
    converted.proj.input_quantization = True
    converted.proj.input_q_type = "fp8_e4m3"
    converted.proj.input_chunk_size = 128
    decision = (
        f"module:{type(converted.proj).__module__}.{type(converted.proj).__qualname__}"
    )
    simulator = _create_simulator(
        converted,
        SimulationPlan(
            kernels={
                "module:proj": _route_row(
                    "linear", {"dynamic-true": 1, "dynamic-false": 1}
                )
            },
            module_decisions={"proj": decision},
            quantization_enabled=True,
            quantization_policy=QuantizationPolicy(quantize_first_layer=True),
        ),
        strict=False,
        converted=True,
        build_errors=[],
        reference_model=copy.deepcopy(reference),
    )

    verification = simulator.verify(
        [
            Scenario("dynamic-true", (torch.randn(2, 4),), {"negate": True}),
            Scenario("dynamic-false", (torch.randn(2, 4),), {"negate": False}),
        ]
    )

    assert verification.succeeded is True
    assert verification.quantized_execution is False
    assert simulator.activation_transport_stats()["runtime"] == "eager"
    assert simulator.activation_transport_stats()["transmission_count"] == 2
    assert converted.proj.input_quantization is False
    assert converted.proj._qbench_activation_transport_active is True
    simulator.close()
    assert converted.proj.input_quantization is True
    assert not hasattr(converted.proj, "_qbench_activation_transport_active")


def test_verification_rejects_nested_output_mismatch_and_state_export_is_isolated(
    monkeypatch,
):
    from qbench import conversion

    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("constant", torch.tensor([1.0]))

        def forward(self, value):
            return {"value": value, "nested": (self.constant,)}

    class WrongNested(Nested):
        def forward(self, value):
            return {"value": value, "nested": (self.constant + 1,)}

    monkeypatch.setattr(
        conversion,
        "_convert_modules",
        lambda model, **_kwargs: (WrongNested(), True, []),
    )
    simulator = build_simulator(Nested(), SimulationPlan())
    verification = simulator.verify([Scenario("nested", (torch.ones(1),))])
    assert verification.succeeded is False
    assert verification.output_equivalence is False

    exported = simulator.state_dict()
    exported["constant"].zero_()
    assert simulator.state_dict()["constant"].item() == 1.0


def test_canonical_registry_preserves_function_set_and_legacy_class_identity():
    from qbench.registry import OpRegistry as LegacyRegistry

    assert LegacyRegistry is OpRegistry
    supported = OpRegistry.get_supported_functions()
    assert torch.nn.functional.linear in supported
    assert torch.nn.functional.conv2d in supported
    assert torch.relu in supported


def test_every_ready_functional_schema_has_a_runtime_handler():
    from qbench.registry import KERNEL_SPECS

    missing = [
        (spec.name, schema)
        for spec in KERNEL_SPECS
        if spec.ready and spec.schemas and spec.handler is None
        for schema in spec.schemas
    ]
    assert missing == []


def test_quantization_flag_round_trips_through_plan_schema():
    plan = SimulationPlan(quantization_enabled=True)
    restored = SimulationPlan.from_dict(plan.to_dict())
    assert restored.quantization_enabled is True


def test_cli_verdict_refresh_requires_actual_quantized_execution():
    from qbench.cli import _refresh_verdict

    result = inspect_model(
        nn.ReLU(),
        Scenario("relu", (torch.ones(2),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )
    result.plan.quantization_enabled = True
    result.verification.quantized_execution = False

    _refresh_verdict(result)

    assert result.fully_supported is False
    assert result.support["verdict"] == "partial_or_unsupported"


def test_failed_scenario_retains_partial_dispatch_ledger():
    class FailsAfterOperation(nn.Module):
        def forward(self, value):
            value = value + 1
            raise RuntimeError("model failed after add")

    result = inspect_model(
        FailsAfterOperation(),
        Scenario("failure", (torch.ones(2),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )

    assert result.support["capture_complete"] is False
    assert result.support["verdict"] == "capture_failed"
    assert result.support["scenario_coverage"]["failure"]["operation_count"] == 1
    assert [record.schema for record in result.operations] == ["aten::add.Tensor"]
    assert result.operations[0].output is not None


def test_model_exception_text_is_never_retained_in_persistable_results():
    class LeakingModel(nn.Module):
        def forward(self, value):
            raise RuntimeError(f"private tensor was {value}")

    result = inspect_model(
        LeakingModel(),
        Scenario("failure", (torch.tensor([1234.5]),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )
    serialized = json.dumps(result.to_dict(), sort_keys=True)

    assert "1234.5" not in serialized
    assert "private tensor" not in serialized
    assert result.support["scenario_coverage"]["failure"]["error"] == (
        "RuntimeError: details redacted"
    )


def test_clone_invocation_isolates_mutable_objects_and_preserves_tensor_aliases():
    class Box:
        def __init__(self):
            self.values = []

    class MutatesInputs(nn.Module):
        def forward(self, left, *, right, box):
            assert left is right
            box.values.append("changed")
            return left + 1

    tensor = torch.ones(2)
    box = Box()
    capture_scenario(
        MutatesInputs(),
        Scenario("isolated", (tensor,), {"right": tensor, "box": box}),
    )
    assert box.values == []


def test_registry_load_failure_is_an_unresolved_capability_gap(monkeypatch):
    from qbench import inspection

    monkeypatch.setattr(
        inspection,
        "_ensure_registrations",
        lambda: "Simulator module registry unavailable: forced",
    )

    class Neg(nn.Module):
        def forward(self, value):
            return torch.neg(value)

    # Supply a ready functional spec so the registry failure, rather than the
    # operation itself, is what prevents a positive verdict.
    spec = KernelSpec(
        "neg",
        schemas=("aten::neg",),
        handler=lambda func, args, kwargs: func(*args, **kwargs),
    )
    monkeypatch.setattr(inspection, "KERNEL_SPECS", [spec])
    from qbench import registry

    monkeypatch.setattr(registry, "KERNEL_SPECS", [spec])

    result = inspect_model(
        Neg(),
        Scenario("registry", (torch.ones(2),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )
    assert result.support["registry_ready"] is False
    assert result.support["replacement_coverage"] is False
    assert "qbench::simulator_registry" in result.plan.unresolved_schemas
    assert result.support["scenario_coverage"]["registry"]["supported"] is False
    assert result.support["verdict"] == "partial_or_unsupported"


def test_nonquantized_module_registration_cannot_pass_as_semantic_support(monkeypatch):
    import qbench.ops  # noqa: F401

    implementation = OpRegistry.get_quantized_op(nn.Linear)
    assert implementation is not None
    name = OpRegistry.get_registration_name(implementation) or implementation.__name__
    monkeypatch.setattr(
        OpRegistry,
        "_unquantized_ops",
        set(OpRegistry._unquantized_ops) | {name},
    )

    result = inspect_model(
        nn.Linear(3, 2),
        Scenario("linear", (torch.ones(1, 3),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )
    assert result.plan.module_decisions == {
        "": "module:qbench.ops.quant_linear.QuantLinear"
    }
    assert result.operations[0].classification == "composite_detail"
    assert result.support["gaps"] == []
    assert result.fully_supported is False


def test_unready_module_spec_and_missing_functional_handler_fail_closed(monkeypatch):
    from qbench import inspection, registry

    unready = KernelSpec(
        "linear",
        schemas=("aten::linear", "aten::addmm"),
        module_types=(nn.Linear,),
        ready=False,
    )
    monkeypatch.setattr(inspection, "KERNEL_SPECS", [unready])
    monkeypatch.setattr(registry, "KERNEL_SPECS", [unready])
    module_result = inspect_model(
        nn.Linear(2, 2),
        Scenario("unready", (torch.ones(1, 2),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )
    assert module_result.plan.module_decisions == {}
    assert module_result.support["replacement_coverage"] is False
    assert "KernelSpec is not ready" in module_result.support["gaps"][0]["reason"]

    missing_handler = KernelSpec("neg", schemas=("aten::neg",), handler=None)
    monkeypatch.setattr(inspection, "KERNEL_SPECS", [missing_handler])
    monkeypatch.setattr(registry, "KERNEL_SPECS", [missing_handler])

    class Neg(nn.Module):
        def forward(self, value):
            return torch.neg(value)

    functional_result = inspect_model(
        Neg(),
        Scenario("missing-handler", (torch.ones(2),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )
    assert functional_result.plan.kernels == {}
    assert functional_result.support["replacement_coverage"] is False
    assert "no ready runtime handler" in functional_result.support["gaps"][0]["reason"]


def test_stock_module_hooks_and_instance_forward_overrides_are_not_collapsed():
    hooked = nn.Linear(2, 2)
    handle = hooked.register_forward_hook(
        lambda _module, _args, output: torch.sin(output)
    )
    try:
        hook_result = inspect_model(
            hooked,
            Scenario("hooked", (torch.ones(1, 2),)),
            InspectionConfig(enable_fx=False, enable_export=False, verify=False),
        )
    finally:
        handle.remove()
    assert hook_result.plan.module_decisions == {}
    assert any(gap["schema"] == "aten::sin" for gap in hook_result.support["gaps"])

    overridden = nn.ReLU()

    def custom_forward(self, value):
        return torch.sin(torch.relu(value))

    overridden.forward = types.MethodType(custom_forward, overridden)
    override_result = inspect_model(
        overridden,
        Scenario("overridden", (torch.ones(2),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )
    assert override_result.plan.module_decisions == {}
    assert any(gap["schema"] == "aten::sin" for gap in override_result.support["gaps"])
    assert (
        next(
            record
            for record in override_result.operations
            if record.schema == "aten::sin"
        ).callsite
        is not None
    )


def test_shared_module_aliases_route_functionally_without_wrong_path_conversion():
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            activation = nn.ReLU()
            self.a = activation
            self.b = activation

        def forward(self, value, *, use_b):
            return (self.b if use_b else self.a)(value)

    scenarios = [
        Scenario("through-b", (-torch.ones(2),), {"use_b": True}),
        Scenario("through-a", (-torch.ones(2),), {"use_b": False}),
    ]
    result = inspect_model(
        Shared(),
        scenarios,
        InspectionConfig(enable_fx=False, enable_export=False),
    )

    assert result.plan.module_decisions == {}
    assert result.plan.kernels["schema:aten::relu"]["scenario_counts"] == {
        "through-b": 1,
        "through-a": 1,
    }
    assert all(record.module_aliases == ["a", "b"] for record in result.operations)
    module_rows = {row["path"]: row for row in result.support["module_summary"]}
    assert module_rows["a"]["status"] == module_rows["b"]["status"] == "supported"
    assert result.verification.succeeded is True


def test_stock_and_functional_softmax_and_relu6_use_exact_ready_handlers():
    stock = inspect_model(
        nn.Softmax(dim=-1),
        Scenario("stock-softmax", (torch.randn(2, 3),)),
        InspectionConfig(enable_fx=False, enable_export=False),
    )
    assert stock.plan.module_decisions == {
        "": "module:qbench.ops.quant_softmax.QuantSoftmax"
    }
    assert stock.fully_supported is True

    class Functional(nn.Module):
        def forward(self, value):
            return torch.softmax(value, dim=-1), torch.nn.functional.relu6(value)

    functional = inspect_model(
        Functional(),
        Scenario("functional", (torch.randn(2, 3),)),
        InspectionConfig(enable_fx=False, enable_export=False),
    )
    schemas = {record.schema: record.kernel for record in functional.operations}
    assert schemas["aten::softmax.int"] == "softmax"
    assert schemas["aten::relu6"] == "activation"
    assert functional.fully_supported is True


def test_arithmetic_alpha_and_boolean_attention_masks_fail_closed():
    class AddAlpha(nn.Module):
        def forward(self, value, *, alpha):
            return torch.ops.aten.add.Scalar(value, 0.0, alpha)

    scenarios = [
        Scenario("supported", (torch.ones(2),), {"alpha": 1.0}),
        Scenario("fallback", (torch.ones(2),), {"alpha": 2.0}),
    ]
    result = inspect_model(
        AddAlpha(),
        scenarios,
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            allow_fp32_fallback=True,
        ),
    )
    assert result.support["gaps"][0]["scenario"] == "fallback"
    assert result.verification.fp32_fallbacks == {"aten::add.Scalar": 1}
    assert result.fully_supported is False

    attention = nn.MultiheadAttention(4, 2, batch_first=True)
    value = torch.randn(1, 2, 4)
    mask = torch.tensor([[False, True], [False, False]])
    masked = inspect_model(
        attention,
        Scenario(
            "bool-mask",
            (value, value, value),
            {"need_weights": False, "attn_mask": mask},
        ),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )
    assert masked.plan.module_decisions == {}
    assert masked.support["replacement_coverage"] is False


def test_quantization_policy_round_trips_and_reaches_module_and_functional_routes(
    monkeypatch,
):
    policy = QuantizationPolicy(
        quantization_type="fp8_e4m3",
        act_mode="chunk",
        act_chunk_size=128,
        layer_config={
            "0": {"format": "fp4_e2m1"},
            "functional": {
                "format": "fp6_e3m2",
                "act_mode": "tensor",
            },
        },
    )
    restored = SimulationPlan.from_dict(
        SimulationPlan(quantization_policy=policy).to_dict()
    )
    assert restored.quantization_policy == policy

    model = nn.Sequential(nn.Linear(2, 2))
    module_policy = QuantizationPolicy(
        quantization_type="fp8_e4m3",
        act_mode="chunk",
        act_chunk_size=128,
        layer_config={"0": {"format": "fp4_e2m1"}},
    )
    inspected = inspect_model(
        model,
        Scenario("linear", (torch.ones(1, 2),)),
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            verify=False,
            quantization_policy=module_policy,
        ),
    )
    simulator = build_simulator(model, inspected.plan)
    assert simulator._model[0].q_type == "fp4_e2m1"
    simulator.close()

    from qbench.registry import find_kernel
    from qbench.runtime import simulation_quantization, simulation_route
    from qbench.ops.quant_activations import QuantReLU

    observed = {}

    def observe_input(self, value):
        observed.update(
            q_type=self.q_type,
            bias=self.quantization_bias,
            mode=self.input_mode,
            chunk_size=self.input_chunk_size,
        )
        return value

    monkeypatch.setattr(QuantReLU, "quantize_input", observe_input)
    metadata = (
        {
            "kind": "tensor",
            "shape": [2],
            "dtype": "torch.float32",
            "device": "cpu",
            "requires_grad": False,
        },
    )
    spec = find_kernel("aten::relu", metadata, {})
    functional_policy = QuantizationPolicy(
        quantization_type="fp8_e4m3",
        act_mode="chunk",
        act_chunk_size=128,
        layer_config={
            "functional": {
                "format": "fp6_e3m2",
                "act_mode": "tensor",
            }
        },
    )
    with (
        simulation_quantization(True, functional_policy.to_dict()),
        simulation_route("functional"),
    ):
        spec.handler(torch.ops.aten.relu.default, (torch.ones(2),), {})
    assert observed == {
        "q_type": "fp6_e3m2",
        "bias": None,
        "mode": "tensor",
        "chunk_size": 128,
    }


def test_quantization_policy_rejects_invalid_layer_chunk_and_direct_plan_version():
    with pytest.raises(QBenchError, match="must be 128"):
        QuantizationPolicy(layer_config={"layer": {"mode": "chunk", "chunk_size": 64}})
    with pytest.raises(QBenchError, match="Unsupported plan version"):
        SimulationPlan(schema_version=999)
    with pytest.raises(QBenchError, match="not implemented"):
        QuantizationPolicy(quantization_bias=7)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("quantizes_weights", False, "quantizes_weights disagrees"),
        ("module_implementations", [], "implementation identity disagrees"),
        ("activation_policy", True, "activation_policy disagrees"),
    ],
)
def test_strict_builder_rejects_tampered_runtime_kernel_claims(field, value, match):
    model = nn.Linear(3, 2)
    scenario = Scenario("linear", (torch.ones(1, 3),))
    plan = inspect_model(
        model,
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    ).plan
    plan = SimulationPlan.from_dict(plan.to_dict())
    plan.kernels["module:"][field] = value

    with pytest.raises(QBenchError, match=match):
        build_simulator(model, plan)


def test_hardware_badge_ignores_unrelated_vector_configuration_errors(monkeypatch):
    from qbench import kernels
    from qbench.inspection import _hardware_fidelity

    policy_sha256 = kernels._policy_sha256(QuantizationPolicy().to_dict())
    linear_evidence = (
        "module:qbench.ops.quant_linear.QuantLinear|policy:" + policy_sha256
    )
    softmax_evidence = (
        "module:qbench.ops.quant_softmax.QuantSoftmax|policy:" + policy_sha256
    )
    monkeypatch.setattr(
        kernels,
        "verify_kernels",
        lambda _directory: {
            "status": "configuration_error",
            "kernel_statuses": {
                "linear": "passed",
                "softmax": "configuration_error",
            },
            "capability_statuses": {
                "module:qbench.ops.quant_linear.QuantLinear": "passed",
                "module:qbench.ops.quant_softmax.QuantSoftmax": "configuration_error",
            },
            "capability_policy_statuses": {
                linear_evidence: "passed",
                softmax_evidence: "configuration_error",
            },
            "kernels": [
                {
                    "kernel": "linear",
                    "capability_key": "module:qbench.ops.quant_linear.QuantLinear",
                    "evidence_key": linear_evidence,
                    "status": "passed",
                },
                {
                    "kernel": "softmax",
                    "capability_key": "module:qbench.ops.quant_softmax.QuantSoftmax",
                    "evidence_key": softmax_evidence,
                    "status": "configuration_error",
                    "error": "unrelated bad vector",
                },
            ],
            "errors": ["unrelated bad vector"],
        },
    )
    plan = SimulationPlan(
        kernels={"module:": _route_row("linear", {"linear": 1})},
        module_decisions={"": "module:qbench.ops.quant_linear.QuantLinear"},
    )

    badge = _hardware_fidelity(plan, "unused-path")

    assert badge["status"] == "passed"
    assert badge["kernels"] == {"linear": "passed"}
    assert badge["capabilities"] == {
        "module:qbench.ops.quant_linear.QuantLinear": "passed"
    }
    assert badge["bundle_status"] == "configuration_error"
    assert badge["errors"] == []
    with pytest.raises(QBenchError, match="not implemented"):
        QuantizationPolicy(layer_config={"layer": {"bias": 5}})


def test_mapping_key_metadata_never_stringifies_private_tensor_or_object_values():
    from qbench.capture import value_metadata

    class PrivateKey:
        def __str__(self):
            return "private-secret-from-str"

    metadata = value_metadata(
        {
            torch.tensor([123456.0]): torch.ones(1),
            PrivateKey(): torch.zeros(2),
        }
    )
    serialized = json.dumps(metadata, sort_keys=True)

    assert metadata["kind"] == "mapping"
    assert "123456" not in serialized
    assert "private-secret-from-str" not in serialized
    assert metadata["items"][0]["key"] == {
        "kind": "tensor",
        "shape": [1],
        "dtype": "torch.float32",
        "device": "cpu",
        "requires_grad": False,
    }


def test_inplace_activation_module_is_replaced_and_preserves_mutation_alias():
    class InplaceRelu(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU(inplace=True)

        def forward(self, value):
            local = value.clone()
            alias = local
            returned = self.relu(local)
            return returned, alias

    scenario = Scenario("inplace-relu", (-torch.ones(2),))
    result = inspect_model(
        InplaceRelu(),
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )
    assert [record.schema for record in result.operations] == [
        "aten::clone",
        "aten::relu_",
    ]
    assert result.plan.module_decisions["relu"].endswith("QuantReLU")
    assert result.verification.succeeded is True
    assert result.fully_supported is True
    output = build_simulator(InplaceRelu(), result.plan).run(scenario)
    assert output[0] is output[1]
    torch.testing.assert_close(output[0], torch.zeros(2))


def test_strict_conversion_does_not_replace_unplanned_module_ancestors():
    class FusedBatchNorm(nn.BatchNorm2d):
        def __init__(self):
            super().__init__(2)
            self.act = nn.SiLU(inplace=True)

        def forward(self, value):
            normalized = torch.nn.functional.batch_norm(
                value,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
            return self.act(normalized)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fused = FusedBatchNorm()

        def forward(self, value):
            return self.fused(value)

    scenario = Scenario("fused", (torch.randn(1, 2, 4, 4),))
    result = inspect_model(
        Model().eval(),
        scenario,
        InspectionConfig(enable_fx=False, enable_export=False, verify=True),
    )

    assert "fused" not in result.plan.module_decisions
    assert result.plan.module_decisions["fused.act"].endswith("QuantSiLU")
    assert result.plan.kernels["schema:aten::batch_norm"]["name"] == "batch_norm"
    assert result.verification.succeeded is True
    assert result.fully_supported is True


def test_diagnostics_include_reproducible_environment_and_config():
    config = InspectionConfig(
        allow_fp32_fallback=True,
        enable_fx=False,
        enable_export=False,
        verify=False,
        device="cpu",
    )
    result = inspect_model(
        nn.Identity(), Scenario("identity", (torch.ones(1),)), config
    )
    environment = result.diagnostics["environment"]
    assert environment["torch_version"] == str(torch.__version__)
    assert environment["inspection_config"]["device"] == "cpu"
    assert environment["inspection_config"]["allow_fp32_fallback"] is True


@pytest.mark.parametrize(
    "field",
    [
        "allow_fp32_fallback",
        "verify",
        "capture_callsites",
        "enable_fx",
        "enable_export",
        "quantization_enabled",
    ],
)
def test_inspection_config_rejects_string_boolean_safety_flags(field):
    with pytest.raises(QBenchError, match="must be a boolean"):
        InspectionConfig.coerce({field: "false"})


@pytest.mark.parametrize("field", ["allow_fp32_fallback", "quantization_enabled"])
def test_simulation_plan_rejects_string_boolean_safety_flags(field):
    with pytest.raises(QBenchError, match="must be a boolean"):
        SimulationPlan.from_dict({"schema_version": 3, field: "false"})


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"schema_version": 3, "kernels": []}, "kernels must be a mapping"),
        (
            {"schema_version": 3, "module_decisions": []},
            "module_decisions must be a mapping",
        ),
        (
            {"schema_version": 3, "unresolved_schemas": "aten::sin"},
            "unresolved_schemas must be a list",
        ),
    ],
)
def test_simulation_plan_rejects_malformed_containers(payload, message):
    with pytest.raises(QBenchError, match=message):
        SimulationPlan.from_dict(payload)


def test_inspection_device_must_match_model_and_scenario_tensors():
    with pytest.raises(QBenchError, match="Move them in the trusted provider"):
        inspect_model(
            nn.Identity(),
            Scenario("cpu", (torch.ones(1),)),
            InspectionConfig(
                device="cuda",
                enable_fx=False,
                enable_export=False,
                verify=False,
            ),
        )


def test_conv3d_does_not_match_the_conv1d_conv2d_kernel_spec():
    class FunctionalConv3d(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, 1, 1, 1, 1))

        def forward(self, value):
            return torch.ops.aten.convolution.default(
                value,
                self.weight,
                None,
                [1, 1, 1],
                [0, 0, 0],
                [1, 1, 1],
                False,
                [0, 0, 0],
                1,
            )

    result = inspect_model(
        FunctionalConv3d(),
        Scenario("conv3d", (torch.ones(1, 1, 2, 2, 2),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )

    assert result.fully_supported is False
    assert result.operations[0].schema == "aten::convolution"
    assert "constraints not satisfied" in result.support["gaps"][0]["reason"]


def test_transposed_and_string_padding_convolutions_remain_explicit_gaps():
    class TransposedConvolution(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 3, 3, 3))

        def forward(self, value):
            return torch.ops.aten.convolution.default(
                value,
                self.weight,
                None,
                [1, 1],
                [1, 1],
                [1, 1],
                True,
                [0, 0],
                1,
            )

    class StringPadding(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(3, 2, 3, 3))

        def forward(self, value):
            return torch.ops.aten.conv2d.padding(
                value,
                self.weight,
                None,
                [1, 1],
                "same",
                [1, 1],
                1,
            )

    transposed = inspect_model(
        TransposedConvolution(),
        Scenario("transposed", (torch.ones(1, 2, 6, 6),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )
    string_padding = inspect_model(
        StringPadding(),
        Scenario("string-padding", (torch.ones(1, 2, 6, 6),)),
        InspectionConfig(enable_fx=False, enable_export=False, verify=False),
    )

    assert transposed.plan.kernels == {}
    assert transposed.support["gaps"][0]["schema"] == "aten::convolution"
    assert "constraints not satisfied" in transposed.support["gaps"][0]["reason"]
    assert string_padding.plan.kernels == {}
    assert string_padding.support["gaps"][0]["schema"] == "aten::conv2d.padding"
    assert "unknown schema or overload" in string_padding.support["gaps"][0]["reason"]


def test_canonical_operation_registration_has_one_class_identity():
    import importlib

    canonical_module = importlib.import_module("qbench.ops.quant_linear")
    package = importlib.import_module("qbench.ops")
    canonical_registry = importlib.import_module("qbench.registry")
    assert package.QuantLinear is canonical_module.QuantLinear
    assert canonical_registry.OpRegistry is OpRegistry
    assert canonical_module.QuantLinear.__module__ == "qbench.ops.quant_linear"
