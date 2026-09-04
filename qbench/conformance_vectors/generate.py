"""Generate the deterministic canonical QBench conformance corpus.

The archives contain inputs only.  Verification obtains golden outputs from a
strict q-enabled simulator run, never from a native-FP32 precomputation.
Hardware teams may add independently measured ``actual`` arrays to a separate
bundle with explicit provenance; the packaged corpus is not hardware evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
TOLERANCE = {"kind": "tolerance", "rtol": 1e-5, "atol": 1e-6}
BIT_EXACT = {"kind": "bit_exact"}
POLICY_NAME = "fp8_e4m3_full_tensor"
QUANTIZATION_POLICY = {
    "quantization_type": "fp8_e4m3",
    "quantization_bias": None,
    "input_quantization": True,
    "weight_quantization": True,
    "output_quantization": True,
    "quantize_first_layer": True,
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


def _float(values):
    return np.asarray(values, dtype=np.float32)


def _linear():
    value = _float([[-1.0, 0.5, 2.0], [3.0, -2.0, 0.25]])
    weight = _float([[0.5, -1.0, 0.25], [1.5, 0.0, -0.5]])
    bias = _float([0.125, -0.25])
    return {"input": value, "weight": weight, "bias": bias}


def _convolution():
    value = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4) / 4.0
    weight = _float([[[[1.0, 0.0, -1.0], [0.5, 0.25, -0.5], [1.0, 0.0, -1.0]]]])
    bias = _float([0.125])
    return {"input": value, "weight": weight, "bias": bias}


def _conv1d():
    value = _float([[[0.0, 0.5, 1.0, -1.0, 2.0]]])
    weight = _float([[[1.0, -0.5, 0.25]]])
    bias = _float([0.125])
    return {"input": value, "weight": weight, "bias": bias}


def _normalization():
    value = _float([[1.0, -1.0, 0.5, 2.0], [3.0, 1.0, -2.0, 0.0]])
    weight = _float([1.0, 0.5, 1.5, -0.25])
    bias = _float([0.0, 0.25, -0.5, 1.0])
    epsilon = np.asarray(1e-5, dtype=np.float32)
    return {
        "input": value,
        "weight": weight,
        "bias": bias,
        "epsilon": epsilon,
    }


def _batch_norm():
    value = _float(
        [
            [[[1.0, -1.0], [0.5, 2.0]], [[3.0, 1.0], [-2.0, 0.0]]],
            [[[0.0, 2.0], [1.5, -0.5]], [[4.0, -1.0], [0.5, 2.5]]],
        ]
    )
    weight = _float([1.25, -0.5])
    bias = _float([0.125, 1.0])
    running_mean = _float([0.5, 1.5])
    running_var = _float([0.25, 2.0])
    epsilon = np.asarray(1e-5, dtype=np.float32)
    momentum = np.asarray(0.1, dtype=np.float32)
    return {
        "input": value,
        "weight": weight,
        "bias": bias,
        "running_mean": running_mean,
        "running_var": running_var,
        "epsilon": epsilon,
        "momentum": momentum,
    }


def _batch_norm_1d():
    arrays = _batch_norm()
    arrays["input"] = arrays["input"].reshape(2, 2, 4)
    return arrays


def _activation():
    value = _float([-3.0, -0.0, 0.25, 2.0])
    return {"input": value}


def _softmax():
    value = _float([[1.0, -2.0, 0.5], [4.0, 4.0, 1.0]])
    return {
        "input": value,
        "dim": np.asarray(-1, dtype=np.int64),
    }


def _attention():
    query = _float([[[1.0, 0.0, 0.5, -1.0], [0.5, 1.0, -0.25, 2.0]]])
    q_weight = np.eye(4, dtype=np.float32)
    k_weight = np.diag(_float([0.5, 1.0, -0.5, 0.25]))
    v_weight = _float(
        [
            [1.0, 0.25, 0.0, -0.5],
            [-0.5, 1.0, 0.25, 0.0],
            [0.0, -0.25, 1.0, 0.5],
            [0.5, 0.0, -0.5, 1.0],
        ]
    )
    in_proj_weight = np.concatenate([q_weight, k_weight, v_weight], axis=0)
    in_proj_bias = np.zeros(12, dtype=np.float32)
    out_proj_weight = np.eye(4, dtype=np.float32)
    out_proj_bias = np.zeros(4, dtype=np.float32)
    return {
        "query": query,
        "in_proj_weight": in_proj_weight,
        "in_proj_bias": in_proj_bias,
        "out_proj_weight": out_proj_weight,
        "out_proj_bias": out_proj_bias,
        "num_heads": np.asarray(2, dtype=np.int64),
    }


def _scaled_dot_product_attention():
    query = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4) / np.float32(
        8.0
    ) - np.float32(1.0)
    key = np.flip(query, axis=-2).copy()
    value = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4) / np.float32(
        16.0
    ) - np.float32(0.5)
    return {"query": query, "key": key, "value": value}


def _index_arange():
    return {"end": np.asarray(9, dtype=np.int64)}


def _index_dtype_cast():
    return {
        "float_input": _float([[-2.75, 0.0, 3.5], [4.0, -1.25, 2.0]]),
        "integer_input": np.asarray([[-3, 0, 5], [7, -2, 1]], dtype=np.int64),
        "alias_input": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
    }


def _clamp():
    return {
        "float_input": _float([[-3.0, -0.5, 0.25, 4.0]]),
        "float_minimum": np.asarray(-1.0, dtype=np.float32),
        "float_maximum": np.asarray(2.0, dtype=np.float32),
        "integer_input": np.asarray([[-4, -1, 2, 7]], dtype=np.int64),
        "integer_minimum": np.asarray(-2, dtype=np.int64),
        "integer_maximum": np.asarray(5, dtype=np.int64),
        "integer_boundary_input": np.asarray(
            [
                [2**24 + 1, -(2**24 + 1)],
                [np.iinfo(np.int64).max, np.iinfo(np.int64).min],
            ],
            dtype=np.int64,
        ),
        "integer_boundary_minimum": np.asarray(np.iinfo(np.int64).min, dtype=np.int64),
        "integer_boundary_maximum": np.asarray(np.iinfo(np.int64).max, dtype=np.int64),
    }


def _matmul():
    left = _float([[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]])
    right = _float([[2.0, -1.0], [0.0, 0.5], [4.0, 1.0]])
    return {"left": left, "right": right}


def _batched_matmul():
    arrays = _matmul()
    arrays["left"] = np.stack([arrays["left"], arrays["left"] * 0.5])
    arrays["right"] = np.stack([arrays["right"], arrays["right"] * -0.25])
    return arrays


def _arithmetic():
    left = _float([[-1.0, 0.5, 3.0], [2.0, -4.0, 0.25]])
    right = _float([[0.25, 1.5, -2.0], [1.0, 2.0, 0.75]])
    return {
        "left": left,
        "right": right,
        "scalar": np.asarray(0.75, dtype=np.float32),
        "integer_left": np.asarray([[-4, 0, 3], [7, -2, 1]], dtype=np.int64),
        "integer_right": np.asarray([[1, 5, -2], [-3, 2, 4]], dtype=np.int64),
        "mixed_left": _float([[-1.0, 0.5, 3.0], [2.0, -4.0, 0.25]]),
        "mixed_right": np.asarray([[1, 2, -2], [-1, 2, 3]], dtype=np.int64),
        "reverse_mixed_left": np.asarray([[-4, 0, 3], [7, -2, 1]], dtype=np.int64),
        "reverse_mixed_right": _float([[0.5, 1.25, -2.0], [-1.0, 2.0, 3.5]]),
        "integer_boundary_left": np.asarray(
            [
                [2**24 + 1, -(2**24 + 1)],
                [np.iinfo(np.int64).max, np.iinfo(np.int64).min],
            ],
            dtype=np.int64,
        ),
        "integer_boundary_right": np.asarray([[0, 0], [1, -1]], dtype=np.int64),
    }


def _pooling():
    value = _float(
        [
            [
                [
                    [1.0, -2.0, 3.0, 0.0],
                    [4.0, 5.0, -1.0, 2.0],
                    [0.5, 7.0, 6.0, -3.0],
                    [2.5, 1.0, 8.0, 4.0],
                ]
            ]
        ]
    )
    return {"input": value}


def _average_pooling():
    value = _float(
        [
            [
                [
                    [1.0, -2.0, 3.0, 0.0],
                    [4.0, 5.0, -1.0, 2.0],
                    [0.5, 7.0, 6.0, -3.0],
                    [2.5, 1.0, 8.0, 4.0],
                ]
            ]
        ]
    )
    return {"input": value}


def _dropout():
    value = _float([[-1.0, 0.0, 2.0], [4.0, -3.0, 0.5]])
    return {"input": value}


def _schema(target):
    return {"kind": "schema", "target": target}


def _module(target, implementation):
    return {
        "kind": "module",
        "target": target,
        "implementation": implementation,
    }


DEFINITIONS = (
    (
        "linear",
        "linear_basic.npz",
        "linear",
        _schema("aten::linear"),
        _linear,
        TOLERANCE,
    ),
    (
        "linear",
        "linear_module.npz",
        "linear",
        _module(
            "torch.nn.modules.linear.Linear",
            "qbench.ops.quant_linear.QuantLinear",
        ),
        _linear,
        TOLERANCE,
    ),
    ("addmm", "addmm_basic.npz", "addmm", _schema("aten::addmm"), _linear, TOLERANCE),
    (
        "convolution",
        "convolution_basic.npz",
        "convolution",
        _schema("aten::convolution"),
        _convolution,
        TOLERANCE,
    ),
    (
        "convolution",
        "convolution_module_conv1d.npz",
        "convolution",
        _module(
            "torch.nn.modules.conv.Conv1d",
            "qbench.ops.quant_conv1d.QuantConv1d",
        ),
        _conv1d,
        TOLERANCE,
    ),
    (
        "convolution",
        "convolution_module_conv2d.npz",
        "convolution",
        _module(
            "torch.nn.modules.conv.Conv2d",
            "qbench.ops.quant_conv.QuantConv2d",
        ),
        _convolution,
        TOLERANCE,
    ),
    (
        "conv1d",
        "conv1d_basic.npz",
        "conv1d",
        _schema("aten::conv1d"),
        _conv1d,
        TOLERANCE,
    ),
    (
        "conv2d",
        "conv2d_basic.npz",
        "conv2d",
        _schema("aten::conv2d"),
        _convolution,
        TOLERANCE,
    ),
    (
        "normalization",
        "normalization_layernorm.npz",
        "normalization",
        _schema("aten::layer_norm"),
        _normalization,
        TOLERANCE,
    ),
    (
        "normalization",
        "normalization_module_layernorm.npz",
        "normalization",
        _module(
            "torch.nn.modules.normalization.LayerNorm",
            "qbench.ops.quant_ln.QuantLayerNorm",
        ),
        _normalization,
        TOLERANCE,
    ),
    (
        "normalization",
        "normalization_module_batchnorm1d.npz",
        "normalization",
        _module(
            "torch.nn.modules.batchnorm.BatchNorm1d",
            "qbench.ops.quant_bn.QuantBatchNorm1d",
        ),
        _batch_norm_1d,
        TOLERANCE,
    ),
    (
        "normalization",
        "normalization_module_batchnorm2d.npz",
        "normalization",
        _module(
            "torch.nn.modules.batchnorm.BatchNorm2d",
            "qbench.ops.quant_bn.QuantBatchNorm2d",
        ),
        _batch_norm,
        TOLERANCE,
    ),
    (
        "batch_norm",
        "batch_norm_inference.npz",
        "batch_norm",
        _schema("aten::batch_norm"),
        _batch_norm,
        TOLERANCE,
    ),
    *tuple(
        (
            "activation",
            filename,
            "activation",
            _schema(target),
            _activation,
            TOLERANCE,
        )
        for target, filename in (
            ("aten::relu", "activation_relu.npz"),
            ("aten::relu6", "activation_relu6.npz"),
            ("aten::gelu", "activation_gelu.npz"),
            ("aten::silu", "activation_silu.npz"),
            ("aten::hardswish", "activation_hardswish.npz"),
            ("aten::hardsigmoid", "activation_hardsigmoid.npz"),
        )
    ),
    *tuple(
        (
            "activation",
            filename,
            "activation",
            _module(target, implementation),
            _activation,
            TOLERANCE,
        )
        for target, implementation, filename in (
            (
                "torch.nn.modules.activation.ReLU",
                "qbench.ops.quant_activations.QuantReLU",
                "activation_module_relu.npz",
            ),
            (
                "torch.nn.modules.activation.ReLU6",
                "qbench.ops.quant_activations.QuantReLU6",
                "activation_module_relu6.npz",
            ),
            (
                "torch.nn.modules.activation.GELU",
                "qbench.ops.quant_activations.QuantGELU",
                "activation_module_gelu.npz",
            ),
            (
                "torch.nn.modules.activation.SiLU",
                "qbench.ops.quant_activations.QuantSiLU",
                "activation_module_silu.npz",
            ),
            (
                "torch.nn.modules.activation.Hardswish",
                "qbench.ops.quant_activations.QuantHardswish",
                "activation_module_hardswish.npz",
            ),
            (
                "torch.nn.modules.activation.Hardsigmoid",
                "qbench.ops.quant_activations.QuantHardsigmoid",
                "activation_module_hardsigmoid.npz",
            ),
        )
    ),
    (
        "softmax",
        "softmax_last_dim.npz",
        "softmax",
        _schema("aten::_softmax"),
        _softmax,
        TOLERANCE,
    ),
    (
        "softmax",
        "softmax_schema_int.npz",
        "softmax",
        _schema("aten::softmax.int"),
        _softmax,
        TOLERANCE,
    ),
    (
        "softmax",
        "softmax_module.npz",
        "softmax",
        _module(
            "torch.nn.modules.activation.Softmax",
            "qbench.ops.quant_softmax.QuantSoftmax",
        ),
        _softmax,
        TOLERANCE,
    ),
    (
        "attention",
        "attention_mha.npz",
        "attention",
        _module(
            "torch.nn.modules.activation.MultiheadAttention",
            "qbench.ops.quant_mha.DecomposedMultiheadAttention",
        ),
        _attention,
        TOLERANCE,
    ),
    (
        "scaled_dot_product_attention",
        "scaled_dot_product_attention_basic.npz",
        "scaled_dot_product_attention",
        _schema("aten::scaled_dot_product_attention"),
        _scaled_dot_product_attention,
        TOLERANCE,
    ),
    (
        "index_arange",
        "index_arange_basic.npz",
        "index_arange",
        _schema("aten::arange"),
        _index_arange,
        BIT_EXACT,
    ),
    (
        "index_dtype_cast",
        "index_dtype_cast_basic.npz",
        "index_dtype_cast",
        _schema("aten::to.dtype"),
        _index_dtype_cast,
        BIT_EXACT,
    ),
    *tuple(
        (
            "index_arithmetic",
            filename,
            "index_arithmetic",
            _schema(target),
            _arithmetic,
            BIT_EXACT,
        )
        for target, filename in (
            ("aten::add.Tensor", "index_arithmetic_add.npz"),
            ("aten::sub.Tensor", "index_arithmetic_sub.npz"),
        )
    ),
    (
        "index_clamp",
        "index_clamp_integer.npz",
        "index_clamp",
        _schema("aten::clamp"),
        _clamp,
        BIT_EXACT,
    ),
    (
        "clamp",
        "clamp_float_integer.npz",
        "clamp",
        _schema("aten::clamp"),
        _clamp,
        TOLERANCE,
    ),
    (
        "matmul",
        "matmul_basic.npz",
        "matmul",
        _schema("aten::matmul"),
        _matmul,
        TOLERANCE,
    ),
    ("matmul", "matmul_mm.npz", "matmul", _schema("aten::mm"), _matmul, TOLERANCE),
    (
        "matmul",
        "matmul_bmm.npz",
        "matmul",
        _schema("aten::bmm"),
        _batched_matmul,
        TOLERANCE,
    ),
    *tuple(
        (
            "arithmetic",
            filename,
            "arithmetic",
            _schema(target),
            _arithmetic,
            TOLERANCE,
        )
        for target, filename in (
            ("aten::add.Tensor", "arithmetic_add.npz"),
            ("aten::add.Scalar", "arithmetic_add_scalar.npz"),
            ("aten::sub.Tensor", "arithmetic_sub.npz"),
            ("aten::mul.Tensor", "arithmetic_mul.npz"),
            ("aten::mul.Scalar", "arithmetic_mul_scalar.npz"),
            ("aten::div.Tensor", "arithmetic_div.npz"),
            ("aten::div.Scalar", "arithmetic_div_scalar.npz"),
        )
    ),
    (
        "inplace_add",
        "arithmetic_add_inplace.npz",
        "inplace_add",
        _schema("aten::add_.Tensor"),
        _arithmetic,
        TOLERANCE,
    ),
    (
        "pooling",
        "pooling_max2d.npz",
        "pooling",
        _module(
            "torch.nn.modules.pooling.MaxPool2d",
            "qbench.ops.quant_pooling.QuantMaxPool2d",
        ),
        _pooling,
        TOLERANCE,
    ),
    (
        "pooling",
        "pooling_avg2d.npz",
        "pooling",
        _module(
            "torch.nn.modules.pooling.AvgPool2d",
            "qbench.ops.quant_pooling.QuantAvgPool2d",
        ),
        _average_pooling,
        TOLERANCE,
    ),
    (
        "pooling",
        "pooling_adaptive_avg2d.npz",
        "pooling",
        _module(
            "torch.nn.modules.pooling.AdaptiveAvgPool2d",
            "qbench.ops.quant_pooling.QuantAdaptiveAvgPool2d",
        ),
        _average_pooling,
        TOLERANCE,
    ),
    (
        "max_pool2d",
        "max_pool2d_basic.npz",
        "max_pool2d",
        _schema("aten::max_pool2d_with_indices"),
        _pooling,
        TOLERANCE,
    ),
    (
        "avg_pool2d",
        "avg_pool2d_basic.npz",
        "avg_pool2d",
        _schema("aten::avg_pool2d"),
        _average_pooling,
        TOLERANCE,
    ),
    (
        "adaptive_avg_pool2d",
        "adaptive_avg_pool2d_basic.npz",
        "adaptive_avg_pool2d",
        _schema("aten::adaptive_avg_pool2d"),
        _average_pooling,
        TOLERANCE,
    ),
    (
        "dropout",
        "dropout_eval.npz",
        "dropout",
        _schema("aten::dropout"),
        _dropout,
        BIT_EXACT,
    ),
    (
        "dropout",
        "dropout_module.npz",
        "dropout",
        _module(
            "torch.nn.modules.dropout.Dropout",
            "qbench.ops.quant_dropout.QuantDropout",
        ),
        _dropout,
        BIT_EXACT,
    ),
)


def _archive_bytes(arrays) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as output:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(
                payload,
                np.ascontiguousarray(arrays[name]),
                allow_pickle=False,
                version=(1, 0),
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            output.writestr(
                info,
                payload.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return archive.getvalue()


def _corpus():
    vectors = []
    payloads = {}
    for kernel, filename, runner, capability, factory, comparison in DEFINITIONS:
        payload = _archive_bytes(factory())
        payloads[filename] = payload
        vectors.append(
            {
                "kernel": kernel,
                "file": filename,
                "runner": runner,
                "capability": capability,
                "quantization_policy": POLICY_NAME,
                "comparison": comparison,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 2,
        "corpus": "qbench-canonical-kernels",
        "hardware_actual_results": False,
        "hardware_evidence": None,
        "quantization_policies": {POLICY_NAME: QUANTIZATION_POLICY},
        "vectors": vectors,
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return payloads, manifest_payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors-only", action="store_true")
    parser.add_argument("--print-manifest", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payloads, manifest = _corpus()
    if args.print_manifest:
        sys.stdout.buffer.write(manifest)
        return 0
    if args.check:
        expected = {**payloads, "manifest.json": manifest}
        mismatches = [
            name
            for name, payload in expected.items()
            if not (ROOT / name).is_file() or (ROOT / name).read_bytes() != payload
        ]
        if mismatches:
            print(
                "Out-of-date conformance corpus: " + ", ".join(mismatches),
                file=sys.stderr,
            )
            return 1
        return 0
    for filename, payload in payloads.items():
        (ROOT / filename).write_bytes(payload)
    if not args.vectors_only:
        (ROOT / "manifest.json").write_bytes(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
