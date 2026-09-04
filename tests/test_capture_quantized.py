# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from qbench.capture import clone_inputs, clone_invocation
from qbench.schemas import Scenario


def test_clone_invocation_preserves_per_tensor_quantizer_and_object_aliases():
    source = torch.quantize_per_tensor(
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
        scale=0.125,
        zero_point=-3,
        dtype=torch.qint8,
    )
    original_bits = source.int_repr().clone()
    scenario = Scenario(
        "per-tensor",
        (source,),
        {"same": source, "nested": [source]},
    )

    args, kwargs = clone_invocation(scenario)
    cloned = args[0]

    assert cloned is kwargs["same"]
    assert cloned is kwargs["nested"][0]
    assert cloned is not source
    assert cloned.is_quantized
    assert cloned.qscheme() == source.qscheme() == torch.per_tensor_affine
    assert cloned.q_scale() == source.q_scale()
    assert cloned.q_zero_point() == source.q_zero_point()
    assert cloned.dtype == source.dtype
    assert cloned.shape == source.shape
    assert cloned.stride() == source.stride()
    assert torch.equal(cloned.int_repr(), original_bits)

    cloned[0, 0] = 4.0
    assert not torch.equal(cloned.int_repr(), original_bits)
    assert torch.equal(source.int_repr(), original_bits)


@pytest.mark.parametrize("float_zero_points", [False, True])
def test_clone_inputs_preserves_per_channel_parameters_and_shared_view_storage(
    float_zero_points,
):
    scales = torch.tensor([0.1, 0.2, 0.25], dtype=torch.float64)
    if float_zero_points:
        zero_points = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)
        dtype = torch.quint8
        expected_scheme = torch.per_channel_affine_float_qparams
    else:
        zero_points = torch.tensor([0, 2, -1], dtype=torch.int64)
        dtype = torch.qint8
        expected_scheme = torch.per_channel_affine
    source = torch.quantize_per_channel(
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
        scales,
        zero_points,
        axis=0,
        dtype=dtype,
    )
    # PyTorch does not currently permit views of float-qparam quantized
    # tensors, so that case exercises repeated-object aliasing instead.
    source_view = source if float_zero_points else source[1:]
    original_bits = source.int_repr().clone()

    cloned, cloned_view = clone_inputs((source, source_view))

    assert cloned.qscheme() == expected_scheme
    assert cloned_view.qscheme() == expected_scheme
    assert cloned.q_per_channel_axis() == source.q_per_channel_axis() == 0
    torch.testing.assert_close(
        cloned.q_per_channel_scales(), source.q_per_channel_scales()
    )
    torch.testing.assert_close(
        cloned.q_per_channel_zero_points(), source.q_per_channel_zero_points()
    )
    torch.testing.assert_close(
        cloned_view.q_per_channel_scales(), source_view.q_per_channel_scales()
    )
    torch.testing.assert_close(
        cloned_view.q_per_channel_zero_points(), source_view.q_per_channel_zero_points()
    )
    assert torch.equal(cloned.int_repr(), original_bits)
    assert torch.equal(cloned_view.int_repr(), source_view.int_repr())

    if float_zero_points:
        assert cloned is cloned_view
    else:
        assert cloned is not cloned_view
    assert cloned.untyped_storage()._cdata == cloned_view.untyped_storage()._cdata
    assert cloned.untyped_storage()._cdata != source.untyped_storage()._cdata
    if float_zero_points:
        cloned[0, 0] = 1.0
        assert cloned_view.int_repr()[0, 0] == cloned.int_repr()[0, 0]
    else:
        cloned[1, 0] = 1.0
        assert cloned_view.int_repr()[0, 0] == cloned.int_repr()[1, 0]
    assert torch.equal(source.int_repr(), original_bits)
