"""Public kernel catalog and portable conformance helpers."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path

from .registry import KERNEL_SPECS, list_kernel_specs, schema_route_key


class _ConformanceConfigurationError(ValueError):
    pass


class _SimulatorConformanceError(AssertionError):
    pass


class _SimulatorUnavailableError(RuntimeError):
    """The declared q-enabled simulator cannot run on this host."""


def list_kernels():
    return list_kernel_specs()


def _tensor(arrays, name, device):
    import torch

    return torch.from_numpy(arrays[name].copy()).to(device=device)


def _numpy_outputs(value):
    """Return every tensor output, including tuple auxiliaries such as indices."""

    outputs = {}

    def visit(item, suffix=""):
        import torch

        if torch.is_tensor(item):
            outputs[f"output{suffix}"] = item.detach().to(device="cpu").numpy()
        elif isinstance(item, (tuple, list)):
            for index, member in enumerate(item):
                visit(member, f"{suffix}_{index}" if suffix else f"_{index}")
        elif isinstance(item, Mapping):
            for index, member in enumerate(item.values()):
                visit(member, f"{suffix}_{index}" if suffix else f"_{index}")

    visit(value)
    if not outputs:
        raise RuntimeError("maintained simulator produced no tensor output")
    # Preserve the historical primary-array spelling while retaining all
    # additional tensor outputs under stable indexed names.
    first_key = next(iter(outputs))
    if first_key != "output":
        outputs = {"output": outputs.pop(first_key), **outputs}
    return outputs


def _cuda_device():
    import torch

    if not torch.cuda.is_available():
        raise _SimulatorUnavailableError(
            "q-enabled conformance requires a CUDA device and the documented GPU container"
        )
    return torch.device("cuda")


def _maintained_spec(kernel_name: str):
    matches = [spec for spec in KERNEL_SPECS if spec.name == kernel_name]
    if len(matches) != 1:
        raise _ConformanceConfigurationError(
            f"expected one maintained KernelSpec named {kernel_name!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _run_inspected(kernel_name, capability, model, args, kwargs, policy):
    """Run one exact capability through the public strict q-enabled simulator."""

    from .conversion import build_simulator
    from .inspection import inspect_model
    from .schemas import InspectionConfig, Scenario

    scenario = Scenario("conformance", tuple(args), dict(kwargs))
    inspected = inspect_model(
        model,
        scenario,
        InspectionConfig(
            enable_fx=False,
            enable_export=False,
            verify=False,
            device="cuda",
            quantization_enabled=True,
            quantization_policy=policy,
        ),
    )
    target = capability["target"]
    if capability["kind"] == "module":
        source_type = f"{type(model).__module__}.{type(model).__qualname__}"
        if source_type != target:
            raise RuntimeError(
                f"module conformance declared {target!r} but ran {source_type!r}"
            )
    route = (
        schema_route_key(target, _maintained_spec(kernel_name))
        if capability["kind"] == "schema"
        else "module:"
    )
    row = inspected.plan.kernels.get(route)
    if row is None or row.get("name") != kernel_name:
        raise RuntimeError(
            f"exact capability {capability['kind']}:{target} did not resolve to "
            f"maintained kernel {kernel_name!r}"
        )
    if capability["kind"] == "module":
        expected_implementation = f"module:{capability['implementation']}"
        if inspected.plan.module_decisions.get("") != expected_implementation:
            raise RuntimeError(
                "module capability did not select pinned implementation "
                f"{expected_implementation!r}"
            )
    if inspected.plan.unresolved_schemas:
        raise RuntimeError(
            "exact conformance route has unresolved operations: "
            + ", ".join(inspected.plan.unresolved_schemas)
        )
    simulator = build_simulator(model, inspected.plan, strict=True)
    try:
        verification = simulator.verify([scenario])
        if not verification.succeeded:
            raise RuntimeError(
                "strict q-enabled conformance verification failed: "
                + ("; ".join(verification.errors) or "unknown verification failure")
            )
        if row.get("counts_as_quantized") and not verification.quantized_execution:
            raise RuntimeError(
                "q-enabled conformance produced no actual quantized-execution evidence"
            )
        return simulator.run(scenario)
    finally:
        simulator.close()


def _run_handler(kernel_name, capability, func, args, policy, *, operator_kwargs=None):
    import torch

    operator_kwargs = {} if operator_kwargs is None else dict(operator_kwargs)

    class ExactOperator(torch.nn.Module):
        def forward(self, *runtime_args):
            return func(*runtime_args, **operator_kwargs)

    module = ExactOperator()
    placement = next(
        (value.device for value in args if torch.is_tensor(value)),
        operator_kwargs.get("device"),
    )
    if placement is not None:
        module.to(device=placement)
    return _run_inspected(
        kernel_name,
        capability,
        module,
        args,
        {},
        policy,
    )


def _run_converted_module(kernel_name, capability, module, args, kwargs, policy):
    return _run_inspected(kernel_name, capability, module, args, kwargs, policy)


def _runner_linear(arrays, capability, policy):
    import torch

    device = _cuda_device()
    args = (
        _tensor(arrays, "input", device),
        _tensor(arrays, "weight", device),
        _tensor(arrays, "bias", device),
    )
    if capability["kind"] == "schema":
        return _run_handler(
            "linear", capability, torch.ops.aten.linear.default, args, policy
        )
    module = torch.nn.Linear(arrays["weight"].shape[1], arrays["weight"].shape[0]).to(
        device
    )
    with torch.no_grad():
        module.weight.copy_(args[1])
        module.bias.copy_(args[2])
    return _run_converted_module("linear", capability, module, (args[0],), {}, policy)


def _runner_convolution(arrays, capability, policy):
    import torch

    device = _cuda_device()
    weight = arrays["weight"]
    spatial_rank = weight.ndim - 2
    input_value = _tensor(arrays, "input", device)
    weight_value = _tensor(arrays, "weight", device)
    bias_value = _tensor(arrays, "bias", device)
    if capability["kind"] == "schema":
        args = (
            input_value,
            weight_value,
            bias_value,
            [1] * spatial_rank,
            [0] * spatial_rank,
            [1] * spatial_rank,
            False,
            [0] * spatial_rank,
            1,
        )
        return _run_handler(
            "convolution", capability, torch.ops.aten.convolution.default, args, policy
        )
    implementation = torch.nn.Conv1d if spatial_rank == 1 else torch.nn.Conv2d
    module = implementation(
        weight.shape[1], weight.shape[0], tuple(weight.shape[2:]), bias=True
    ).to(device)
    with torch.no_grad():
        module.weight.copy_(weight_value)
        module.bias.copy_(bias_value)
    return _run_converted_module(
        "convolution", capability, module, (input_value,), {}, policy
    )


def _runner_addmm(arrays, capability, policy):
    import torch

    device = _cuda_device()
    return _run_handler(
        "addmm",
        capability,
        torch.ops.aten.addmm.default,
        (
            _tensor(arrays, "bias", device),
            _tensor(arrays, "input", device),
            _tensor(arrays, "weight", device).t().contiguous(),
        ),
        policy,
    )


def _runner_conv1d(arrays, capability, policy):
    import torch

    device = _cuda_device()
    return _run_handler(
        "conv1d",
        capability,
        torch.ops.aten.conv1d.default,
        (
            _tensor(arrays, "input", device),
            _tensor(arrays, "weight", device),
            _tensor(arrays, "bias", device),
            [1],
            [0],
            [1],
            1,
        ),
        policy,
    )


def _runner_conv2d(arrays, capability, policy):
    import torch

    device = _cuda_device()
    return _run_handler(
        "conv2d",
        capability,
        torch.ops.aten.conv2d.default,
        (
            _tensor(arrays, "input", device),
            _tensor(arrays, "weight", device),
            _tensor(arrays, "bias", device),
            [1, 1],
            [0, 0],
            [1, 1],
            1,
        ),
        policy,
    )


def _configure_batch_norm(module, arrays, device):
    import torch

    with torch.no_grad():
        module.weight.copy_(_tensor(arrays, "weight", device))
        module.bias.copy_(_tensor(arrays, "bias", device))
        module.running_mean.copy_(_tensor(arrays, "running_mean", device))
        module.running_var.copy_(_tensor(arrays, "running_var", device))
    return module


def _runner_normalization(arrays, capability, policy):
    import torch

    device = _cuda_device()
    input_value = _tensor(arrays, "input", device)
    target = capability["target"]
    if capability["kind"] == "schema":
        return _run_handler(
            "normalization",
            capability,
            torch.ops.aten.layer_norm.default,
            (
                input_value,
                list(arrays["weight"].shape),
                _tensor(arrays, "weight", device),
                _tensor(arrays, "bias", device),
                float(arrays["epsilon"].item()),
                True,
            ),
            policy,
        )
    if target == "torch.nn.modules.normalization.LayerNorm":
        module = torch.nn.LayerNorm(
            arrays["weight"].shape,
            eps=float(arrays["epsilon"].item()),
            elementwise_affine=True,
        ).to(device)
        with torch.no_grad():
            module.weight.copy_(_tensor(arrays, "weight", device))
            module.bias.copy_(_tensor(arrays, "bias", device))
    elif target == "torch.nn.modules.batchnorm.BatchNorm1d":
        module = _configure_batch_norm(
            torch.nn.BatchNorm1d(arrays["weight"].shape[0]).eval().to(device),
            arrays,
            device,
        )
    elif target == "torch.nn.modules.batchnorm.BatchNorm2d":
        module = _configure_batch_norm(
            torch.nn.BatchNorm2d(arrays["weight"].shape[0]).eval().to(device),
            arrays,
            device,
        )
    else:
        raise RuntimeError(f"unsupported normalization capability {target!r}")
    return _run_converted_module(
        "normalization", capability, module, (input_value,), {}, policy
    )


def _runner_batch_norm(arrays, capability, policy):
    import torch

    device = _cuda_device()
    return _run_handler(
        "batch_norm",
        capability,
        torch.ops.aten.batch_norm.default,
        (
            _tensor(arrays, "input", device),
            _tensor(arrays, "weight", device),
            _tensor(arrays, "bias", device),
            _tensor(arrays, "running_mean", device),
            _tensor(arrays, "running_var", device),
            False,
            float(arrays["momentum"].item()),
            float(arrays["epsilon"].item()),
            True,
        ),
        policy,
    )


def _runner_activation(arrays, capability, policy):
    import torch

    device = _cuda_device()
    input_value = _tensor(arrays, "input", device)
    target = capability["target"]
    if capability["kind"] == "schema":
        functions = {
            "aten::relu": torch.ops.aten.relu.default,
            "aten::relu6": torch.ops.aten.relu6.default,
            "aten::gelu": torch.ops.aten.gelu.default,
            "aten::silu": torch.ops.aten.silu.default,
            "aten::hardswish": torch.ops.aten.hardswish.default,
            "aten::hardsigmoid": torch.ops.aten.hardsigmoid.default,
        }
        return _run_handler(
            "activation", capability, functions[target], (input_value,), policy
        )
    modules = {
        "torch.nn.modules.activation.ReLU": torch.nn.ReLU,
        "torch.nn.modules.activation.ReLU6": torch.nn.ReLU6,
        "torch.nn.modules.activation.GELU": torch.nn.GELU,
        "torch.nn.modules.activation.SiLU": torch.nn.SiLU,
        "torch.nn.modules.activation.Hardswish": torch.nn.Hardswish,
        "torch.nn.modules.activation.Hardsigmoid": torch.nn.Hardsigmoid,
    }
    module = modules[target]().to(device)
    return _run_converted_module(
        "activation", capability, module, (input_value,), {}, policy
    )


def _runner_softmax(arrays, capability, policy):
    import torch

    device = _cuda_device()
    input_value = _tensor(arrays, "input", device)
    dim = int(arrays["dim"].item())
    target = capability["target"]
    if capability["kind"] == "schema":
        if target == "aten::_softmax":
            func = torch.ops.aten._softmax.default
            args = (input_value, dim, False)
        else:
            func = torch.ops.aten.softmax.int
            args = (input_value, dim, None)
        return _run_handler("softmax", capability, func, args, policy)
    return _run_converted_module(
        "softmax",
        capability,
        torch.nn.Softmax(dim=dim).to(device),
        (input_value,),
        {},
        policy,
    )


def _runner_attention(arrays, capability, policy):
    import torch

    device = _cuda_device()
    module = torch.nn.MultiheadAttention(
        embed_dim=arrays["query"].shape[-1],
        num_heads=int(arrays["num_heads"].item()),
        dropout=0.0,
        bias=True,
        batch_first=True,
    ).to(device)
    with torch.no_grad():
        module.in_proj_weight.copy_(_tensor(arrays, "in_proj_weight", device))
        module.in_proj_bias.copy_(_tensor(arrays, "in_proj_bias", device))
        module.out_proj.weight.copy_(_tensor(arrays, "out_proj_weight", device))
        module.out_proj.bias.copy_(_tensor(arrays, "out_proj_bias", device))
    value = _tensor(arrays, "query", device)
    return _run_converted_module(
        "attention",
        capability,
        module,
        (value, value, value),
        {"need_weights": False},
        policy,
    )


def _runner_scaled_dot_product_attention(arrays, capability, policy):
    import torch

    device = _cuda_device()
    return _run_handler(
        "scaled_dot_product_attention",
        capability,
        torch.ops.aten.scaled_dot_product_attention.default,
        (
            _tensor(arrays, "query", device),
            _tensor(arrays, "key", device),
            _tensor(arrays, "value", device),
            None,
            0.0,
            False,
        ),
        policy,
        operator_kwargs={"scale": None, "enable_gqa": False},
    )


def _runner_index_arange(arrays, capability, policy):
    import torch

    device = _cuda_device()
    return _run_handler(
        "index_arange",
        capability,
        torch.ops.aten.arange.default,
        (int(arrays["end"].item()),),
        policy,
        operator_kwargs={
            "dtype": torch.int64,
            "layout": torch.strided,
            "device": device,
            "pin_memory": False,
        },
    )


def _runner_index_dtype_cast(arrays, capability, policy):
    import torch

    device = _cuda_device()

    def convert(name, dtype):
        return _run_handler(
            "index_dtype_cast",
            capability,
            torch.ops.aten.to.dtype,
            (_tensor(arrays, name, device), dtype, False, False, None),
            policy,
        )

    class SameDtypeAlias(torch.nn.Module):
        def forward(self, input_value):
            converted = torch.ops.aten.to.dtype(
                input_value, input_value.dtype, False, False, None
            )
            return input_value, converted

    alias_input = _tensor(arrays, "alias_input", device).transpose(0, 2)
    alias_outputs = _run_inspected(
        "index_dtype_cast",
        capability,
        SameDtypeAlias().to(device),
        (alias_input,),
        {},
        policy,
    )
    if (
        alias_outputs[0] is not alias_outputs[1]
        or alias_outputs[0].data_ptr() != alias_outputs[1].data_ptr()
        or alias_outputs[0].stride() != alias_outputs[1].stride()
    ):
        raise RuntimeError(
            "same-dtype copy=False conformance did not preserve tensor identity, "
            "storage aliasing, and strides"
        )
    return (
        convert("float_input", torch.int64),
        convert("integer_input", torch.float32),
        alias_outputs[1],
    )


def _runner_clamp(arrays, capability, policy):
    import torch

    device = _cuda_device()

    return _run_handler(
        "clamp",
        capability,
        torch.ops.aten.clamp.default,
        (
            _tensor(arrays, "float_input", device),
            float(arrays["float_minimum"].item()),
            float(arrays["float_maximum"].item()),
        ),
        policy,
    )


def _runner_index_clamp(arrays, capability, policy):
    import torch

    device = _cuda_device()

    def clamp(name, minimum, maximum):
        return _run_handler(
            "index_clamp",
            capability,
            torch.ops.aten.clamp.default,
            (_tensor(arrays, name, device), minimum, maximum),
            policy,
        )

    return (
        clamp(
            "integer_input",
            int(arrays["integer_minimum"].item()),
            int(arrays["integer_maximum"].item()),
        ),
        clamp(
            "integer_boundary_input",
            int(arrays["integer_boundary_minimum"].item()),
            int(arrays["integer_boundary_maximum"].item()),
        ),
    )


def _runner_matmul(arrays, capability, policy):
    import torch

    device = _cuda_device()
    functions = {
        "aten::matmul": torch.ops.aten.matmul.default,
        "aten::mm": torch.ops.aten.mm.default,
        "aten::bmm": torch.ops.aten.bmm.default,
    }
    return _run_handler(
        "matmul",
        capability,
        functions[capability["target"]],
        (_tensor(arrays, "left", device), _tensor(arrays, "right", device)),
        policy,
    )


def _runner_arithmetic_kernel(kernel_name, arrays, capability, policy):
    import torch

    device = _cuda_device()
    target = capability["target"]
    functions = {
        "aten::add.Tensor": torch.ops.aten.add.Tensor,
        "aten::add_.Tensor": torch.ops.aten.add_.Tensor,
        "aten::add.Scalar": torch.ops.aten.add.Scalar,
        "aten::sub.Tensor": torch.ops.aten.sub.Tensor,
        "aten::mul.Tensor": torch.ops.aten.mul.Tensor,
        "aten::mul.Scalar": torch.ops.aten.mul.Scalar,
        "aten::div.Tensor": torch.ops.aten.div.Tensor,
        "aten::div.Scalar": torch.ops.aten.div.Scalar,
    }

    def execute(left_name, right_name="right"):
        other = (
            float(arrays["scalar"].item())
            if target.endswith(".Scalar")
            else _tensor(arrays, right_name, device)
        )
        return _run_handler(
            kernel_name,
            capability,
            functions[target],
            (_tensor(arrays, left_name, device), other),
            policy,
        )

    if kernel_name == "index_arithmetic":
        return (
            execute("integer_left", "integer_right"),
            execute("integer_boundary_left", "integer_boundary_right"),
        )
    primary = execute("left")
    if target not in {"aten::add.Tensor", "aten::sub.Tensor"}:
        return primary
    return (
        primary,
        execute("mixed_left", "mixed_right"),
        execute("reverse_mixed_left", "reverse_mixed_right"),
    )


def _runner_arithmetic(arrays, capability, policy):
    return _runner_arithmetic_kernel("arithmetic", arrays, capability, policy)


def _runner_index_arithmetic(arrays, capability, policy):
    return _runner_arithmetic_kernel("index_arithmetic", arrays, capability, policy)


def _runner_inplace_add(arrays, capability, policy):
    return _runner_arithmetic_kernel("inplace_add", arrays, capability, policy)


def _runner_pooling(arrays, capability, policy):
    import torch

    device = _cuda_device()
    target = capability["target"]
    modules = {
        "torch.nn.modules.pooling.MaxPool2d": lambda: torch.nn.MaxPool2d(2, 2),
        "torch.nn.modules.pooling.AvgPool2d": lambda: torch.nn.AvgPool2d(2, 2),
        "torch.nn.modules.pooling.AdaptiveAvgPool2d": lambda: (
            torch.nn.AdaptiveAvgPool2d((2, 2))
        ),
    }
    module = modules[target]().to(device)
    return _run_converted_module(
        "pooling",
        capability,
        module,
        (_tensor(arrays, "input", device),),
        {},
        policy,
    )


def _runner_max_pool2d(arrays, capability, policy):
    import torch

    device = _cuda_device()
    return _run_handler(
        "max_pool2d",
        capability,
        torch.ops.aten.max_pool2d_with_indices.default,
        (
            _tensor(arrays, "input", device),
            [2, 2],
            [2, 2],
            [0, 0],
            [1, 1],
            False,
        ),
        policy,
    )


def _runner_avg_pool2d(arrays, capability, policy):
    import torch

    device = _cuda_device()
    return _run_handler(
        "avg_pool2d",
        capability,
        torch.ops.aten.avg_pool2d.default,
        (
            _tensor(arrays, "input", device),
            [2, 2],
            [2, 2],
            [0, 0],
            False,
            True,
            None,
        ),
        policy,
    )


def _runner_adaptive_avg_pool2d(arrays, capability, policy):
    import torch

    device = _cuda_device()
    return _run_handler(
        "adaptive_avg_pool2d",
        capability,
        torch.ops.aten.adaptive_avg_pool2d.default,
        (_tensor(arrays, "input", device), [2, 2]),
        policy,
    )


def _runner_dropout(arrays, capability, policy):
    import torch

    device = _cuda_device()
    input_value = _tensor(arrays, "input", device)
    if capability["kind"] == "schema":
        return _run_handler(
            "dropout",
            capability,
            torch.ops.aten.dropout.default,
            (input_value, 0.25, False),
            policy,
        )
    module = torch.nn.Dropout(p=0.25).eval().to(device)
    return _run_converted_module(
        "dropout", capability, module, (input_value,), {}, policy
    )


_CONFORMANCE_RUNNERS = {
    "linear": _runner_linear,
    "addmm": _runner_addmm,
    "convolution": _runner_convolution,
    "conv1d": _runner_conv1d,
    "conv2d": _runner_conv2d,
    "normalization": _runner_normalization,
    "batch_norm": _runner_batch_norm,
    "activation": _runner_activation,
    "softmax": _runner_softmax,
    "attention": _runner_attention,
    "scaled_dot_product_attention": _runner_scaled_dot_product_attention,
    "index_arange": _runner_index_arange,
    "index_dtype_cast": _runner_index_dtype_cast,
    "index_arithmetic": _runner_index_arithmetic,
    "index_clamp": _runner_index_clamp,
    "clamp": _runner_clamp,
    "matmul": _runner_matmul,
    "arithmetic": _runner_arithmetic,
    "inplace_add": _runner_inplace_add,
    "pooling": _runner_pooling,
    "max_pool2d": _runner_max_pool2d,
    "avg_pool2d": _runner_avg_pool2d,
    "adaptive_avg_pool2d": _runner_adaptive_avg_pool2d,
    "dropout": _runner_dropout,
}


def _configuration_report(message: str) -> dict:
    return {
        "status": "configuration_error",
        "passed": 0,
        "failed": 0,
        "missing": 0,
        "configuration_errors": 1,
        "errors": [message],
        "kernels": [],
        "kernel_statuses": {},
        "capability_statuses": {},
        "capability_policy_statuses": {},
        "simulator_status": "configuration_error",
        "simulator_passed": 0,
        "simulator_failed": 0,
        "simulator_kernel_statuses": {},
        "simulator_capability_statuses": {},
        "simulator_capability_policy_statuses": {},
    }


def _capability_key(capability: Mapping, kernel_name: str | None = None) -> str:
    if capability["kind"] == "schema":
        spec = None if kernel_name is None else _maintained_spec(kernel_name)
        return schema_route_key(capability["target"], spec)
    return f"module:{capability['implementation']}"


def _policy_sha256(policy: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _capability_policy_key(capability_key: str, policy_sha256: str) -> str:
    return f"{capability_key}|policy:{policy_sha256}"


def _expected_capabilities(catalog: Mapping[str, Mapping]) -> dict[str, dict]:
    expected: dict[str, dict] = {}
    for kernel, row in catalog.items():
        if not row.get("ready"):
            continue
        for target in row.get("schemas", ()):
            capability = {"kind": "schema", "target": target}
            key = _capability_key(capability, kernel)
            if key in expected:
                raise _ConformanceConfigurationError(
                    f"duplicate maintained capability key {key!r}"
                )
            expected[key] = {
                "kernel": kernel,
                "capability": capability,
            }
        module_types = row.get("module_types", ())
        implementations = row.get("module_implementations", ())
        if len(module_types) != len(implementations):
            raise _ConformanceConfigurationError(
                f"maintained kernel {kernel!r} has an incomplete module implementation map"
            )
        for target, implementation in zip(module_types, implementations, strict=True):
            capability = {
                "kind": "module",
                "target": target,
                "implementation": implementation,
            }
            key = _capability_key(capability, kernel)
            if key in expected:
                raise _ConformanceConfigurationError(
                    f"duplicate maintained capability key {key!r}"
                )
            expected[key] = {"kernel": kernel, "capability": capability}
    return expected


def _validate_capability(item: Mapping, catalog_row: Mapping) -> tuple[dict, str]:
    value = item.get("capability")
    if not isinstance(value, Mapping):
        raise _ConformanceConfigurationError("capability must be an object")
    kind = value.get("kind")
    if kind == "schema":
        if set(value) != {"kind", "target"}:
            raise _ConformanceConfigurationError(
                "schema capability must contain exactly kind and target"
            )
        target = value.get("target")
        if not isinstance(target, str) or target not in catalog_row.get("schemas", ()):
            raise _ConformanceConfigurationError(
                f"schema capability {target!r} is not an exact matcher for this kernel"
            )
        capability = {"kind": "schema", "target": target}
    elif kind == "module":
        if set(value) != {"kind", "target", "implementation"}:
            raise _ConformanceConfigurationError(
                "module capability must contain exactly kind, target, and implementation"
            )
        target = value.get("target")
        implementation = value.get("implementation")
        module_types = list(catalog_row.get("module_types", ()))
        if not isinstance(target, str) or target not in module_types:
            raise _ConformanceConfigurationError(
                f"module capability {target!r} is not an exact matcher for this kernel"
            )
        expected = list(catalog_row.get("module_implementations", ()))[
            module_types.index(target)
        ]
        if not isinstance(implementation, str) or implementation != expected:
            raise _ConformanceConfigurationError(
                f"module capability must pin maintained implementation {expected!r}"
            )
        capability = {
            "kind": "module",
            "target": target,
            "implementation": implementation,
        }
    else:
        raise _ConformanceConfigurationError(
            "capability.kind must be 'schema' or 'module'"
        )
    return capability, _capability_key(capability, str(catalog_row["name"]))


def _load_quantization_policies(manifest: Mapping) -> dict[str, dict]:
    from .schemas import QuantizationPolicy

    value = manifest.get("quantization_policies")
    if not isinstance(value, Mapping) or not value:
        raise _ConformanceConfigurationError(
            "schema-v2 manifest must contain quantization_policies"
        )
    policies: dict[str, dict] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not name:
            raise _ConformanceConfigurationError(
                "quantization policy names must be non-empty strings"
            )
        if not isinstance(raw, Mapping):
            raise _ConformanceConfigurationError(
                f"quantization_policies[{name!r}] must be an object"
            )
        try:
            normalized = QuantizationPolicy.coerce(raw).to_dict()
        except Exception as exc:
            raise _ConformanceConfigurationError(
                f"invalid quantization policy {name!r}: {type(exc).__name__}: {exc}"
            ) from exc
        if dict(raw) != normalized:
            raise _ConformanceConfigurationError(
                f"quantization policy {name!r} must declare every policy field exactly"
            )
        if not normalized["quantize_first_layer"]:
            raise _ConformanceConfigurationError(
                f"quantization policy {name!r} must quantize the standalone first layer"
            )
        policies[name] = normalized
    return policies


def _hardware_evidence(manifest: Mapping) -> tuple[bool, dict | None]:
    declared = manifest.get("hardware_actual_results", False)
    if type(declared) is not bool:
        raise _ConformanceConfigurationError(
            "hardware_actual_results must be a boolean"
        )
    evidence = manifest.get("hardware_evidence")
    if not declared:
        return False, None
    if not isinstance(evidence, Mapping):
        raise _ConformanceConfigurationError(
            "hardware_actual_results=true requires hardware_evidence provenance"
        )
    required_strings = ("producer", "platform", "generated_at")
    for field in required_strings:
        if not isinstance(evidence.get(field), str) or not evidence[field].strip():
            raise _ConformanceConfigurationError(
                f"hardware_evidence.{field} must be a non-empty string"
            )
    if evidence.get("independent_of_simulator") is not True:
        raise _ConformanceConfigurationError(
            "hardware_evidence.independent_of_simulator must be true"
        )
    return True, dict(evidence)


def verify_kernels(vector_directory) -> dict:
    root = Path(vector_directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {
            "status": "missing_evidence",
            "passed": 0,
            "failed": 0,
            "missing": 0,
            "configuration_errors": 0,
            "errors": ["manifest.json not found"],
            "kernels": [],
            "kernel_statuses": {},
            "capability_statuses": {},
            "capability_policy_statuses": {},
            "simulator_status": "not_assessed",
            "simulator_passed": 0,
            "simulator_failed": 0,
            "simulator_kernel_statuses": {},
            "simulator_capability_statuses": {},
            "simulator_capability_policy_statuses": {},
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _configuration_report(f"manifest.json: {type(exc).__name__}: {exc}")
    if not isinstance(manifest, Mapping) or not isinstance(
        manifest.get("vectors", []), list
    ):
        return _configuration_report("manifest.json must contain a vectors list")

    schema_version = manifest.get("schema_version", 1)
    if type(schema_version) is not int or schema_version not in {1, 2}:
        return _configuration_report("schema_version must be integer 1 or 2")
    modern = schema_version == 2

    errors: list[str] = []
    passed = failed = missing = configuration_errors = 0
    simulator_passed = simulator_failed = simulator_not_assessed = 0
    results = []
    catalog = {row["name"]: row for row in list_kernel_specs()}
    try:
        expected_capabilities = _expected_capabilities(catalog) if modern else {}
        policies = _load_quantization_policies(manifest) if modern else {}
        trusted_hardware, hardware_provenance = _hardware_evidence(manifest)
    except _ConformanceConfigurationError as exc:
        return _configuration_report(f"manifest.json: {exc}")
    declared_capabilities: set[str] = set()
    declared_evidence: set[str] = set()
    root_resolved = root.resolve()
    for index, item in enumerate(manifest.get("vectors", [])):
        if not isinstance(item, Mapping):
            message = f"vectors[{index}] must be an object"
            errors.append(message)
            configuration_errors += 1
            results.append(
                {
                    "kernel": "unknown",
                    "status": "configuration_error",
                    "error": message,
                }
            )
            continue
        kernel = str(item.get("kernel", "unknown"))
        if kernel not in catalog:
            message = f"vectors[{index}] names unknown kernel {kernel!r}"
            errors.append(message)
            configuration_errors += 1
            results.append(
                {
                    "kernel": kernel,
                    "status": "configuration_error",
                    "error": message,
                }
            )
            continue
        capability = None
        capability_key = None
        policy_name = None
        policy = None
        policy_sha256 = None
        evidence_key = None
        if modern:
            try:
                capability, capability_key = _validate_capability(item, catalog[kernel])
                policy_name = item.get("quantization_policy")
                if not isinstance(policy_name, str) or policy_name not in policies:
                    raise _ConformanceConfigurationError(
                        "quantization_policy must name a declared policy"
                    )
                policy = policies[policy_name]
                policy_sha256 = _policy_sha256(policy)
                evidence_key = _capability_policy_key(capability_key, policy_sha256)
                if evidence_key in declared_evidence:
                    raise _ConformanceConfigurationError(
                        "duplicate capability and quantization-policy evidence "
                        f"{evidence_key!r}"
                    )
                runner_name = item.get("runner")
                if not isinstance(runner_name, str) or runner_name != kernel:
                    raise _ConformanceConfigurationError(
                        f"runner must be the maintained {kernel!r} runner"
                    )
                declared_capabilities.add(capability_key)
                declared_evidence.add(evidence_key)
            except _ConformanceConfigurationError as exc:
                message = f"vectors[{index}]: {exc}"
                errors.append(message)
                configuration_errors += 1
                row = {
                    "kernel": kernel,
                    "status": "configuration_error",
                    "simulator_status": "configuration_error",
                    "error": message,
                }
                if capability_key is not None:
                    row["capability"] = capability
                    row["capability_key"] = capability_key
                if evidence_key is not None:
                    row["evidence_key"] = evidence_key
                results.append(row)
                continue

        identity = (
            {
                "capability": capability,
                "capability_key": capability_key,
                "evidence_key": evidence_key,
                "quantization_policy": policy_name,
                "quantization_policy_sha256": policy_sha256,
            }
            if capability_key is not None
            else {}
        )

        filename = item.get("file")
        if not isinstance(filename, str) or not filename:
            message = f"vectors[{index}] has no file"
            errors.append(message)
            configuration_errors += 1
            results.append(
                {
                    "kernel": kernel,
                    "status": "configuration_error",
                    "error": message,
                    **identity,
                }
            )
            continue
        path = (root / filename).resolve()
        try:
            path.relative_to(root_resolved)
        except ValueError:
            message = f"vector path escapes its bundle: {filename}"
            errors.append(message)
            configuration_errors += 1
            results.append(
                {
                    "kernel": kernel,
                    "status": "configuration_error",
                    "error": message,
                    **identity,
                }
            )
            continue
        if not path.is_file():
            message = f"missing {filename}"
            missing += 1
            simulator_not_assessed += 1
            results.append(
                {
                    "kernel": kernel,
                    "file": filename,
                    "status": "missing_evidence",
                    "error": message,
                    **identity,
                }
            )
            continue
        expected_digest = item.get("sha256")
        if not isinstance(expected_digest, str) or not expected_digest:
            message = f"missing checksum: {filename}"
            missing += 1
            simulator_not_assessed += 1
            results.append(
                {
                    "kernel": kernel,
                    "file": filename,
                    "status": "missing_evidence",
                    "error": message,
                    **identity,
                }
            )
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest.lower():
            message = f"checksum mismatch: {filename}"
            errors.append(message)
            failed += 1
            simulator_not_assessed += 1
            results.append(
                {
                    "kernel": kernel,
                    "file": filename,
                    "status": "failed",
                    "error": message,
                    **identity,
                }
            )
            continue
        try:
            declared_value = catalog[kernel].get("conformance", {})
            if not isinstance(declared_value, Mapping):
                raise _ConformanceConfigurationError(
                    "maintained conformance rule must be an object"
                )
            comparison = {
                key: value
                for key, value in declared_value.items()
                if key not in {"evidence", "status"}
            }
            supplied_value = item.get("comparison")
            if supplied_value is not None and not isinstance(supplied_value, Mapping):
                raise _ConformanceConfigurationError("comparison must be an object")
            if supplied_value is not None:
                supplied = {
                    key: value
                    for key, value in supplied_value.items()
                    if key not in {"evidence", "status"}
                }
                if supplied != comparison:
                    raise _ConformanceConfigurationError(
                        "manifest comparison does not match the maintained KernelSpec rule"
                    )
            comparison.setdefault("kind", "bit_exact")
        except _ConformanceConfigurationError as exc:
            message = f"{filename}: {type(exc).__name__}: {exc}"
            errors.append(message)
            configuration_errors += 1
            results.append(
                {
                    "kernel": kernel,
                    "file": filename,
                    "status": "configuration_error",
                    "error": message,
                    **identity,
                }
            )
        except Exception as exc:
            message = f"{filename}: {type(exc).__name__}: {exc}"
            errors.append(message)
            failed += 1
            simulator_not_assessed += 1
            results.append(
                {
                    "kernel": kernel,
                    "file": filename,
                    "status": "failed",
                    "error": message,
                    **identity,
                }
            )
        else:
            runner_name = item.get("runner")
            simulator_status = "not_assessed"
            simulator_error = None
            simulated_outputs = None
            if modern:
                try:
                    simulated_outputs = _verify_simulator_npz(
                        path, runner_name, capability, policy
                    )
                except _SimulatorUnavailableError as exc:
                    simulator_not_assessed += 1
                    simulator_error = f"{filename}: simulator unavailable: {exc}"
                except _ConformanceConfigurationError as exc:
                    message = f"{filename}: {type(exc).__name__}: {exc}"
                    errors.append(message)
                    configuration_errors += 1
                    row = {
                        "kernel": kernel,
                        "file": filename,
                        "status": "configuration_error",
                        "simulator_status": "configuration_error",
                        "error": message,
                    }
                    if capability_key is not None:
                        row["capability"] = capability
                        row["capability_key"] = capability_key
                    if evidence_key is not None:
                        row["evidence_key"] = evidence_key
                    results.append(row)
                    continue
                except _SimulatorConformanceError as exc:
                    simulator_status = "failed"
                    simulator_error = f"{filename}: simulator: {exc}"
                    errors.append(simulator_error)
                    simulator_failed += 1
                else:
                    simulator_status = "passed"
                    simulator_passed += 1
            elif runner_name is not None:
                # Version 1 has no exact matcher or quantization-policy binding,
                # so it cannot make a simulator-conformance claim.
                simulator_not_assessed += 1
                simulator_error = (
                    f"{filename}: schema-v1 runner lacks exact q-enabled evidence"
                )
            else:
                simulator_not_assessed += 1

            try:
                if modern:
                    has_measurement = bool(
                        trusted_hardware
                        and simulated_outputs is not None
                        and _verify_hardware_npz(path, simulated_outputs, comparison)
                    )
                else:
                    has_measurement = bool(
                        trusted_hardware and _verify_npz(path, comparison)
                    )
            except _ConformanceConfigurationError as exc:
                message = f"{filename}: {type(exc).__name__}: {exc}"
                errors.append(message)
                configuration_errors += 1
                row = {
                    "kernel": kernel,
                    "file": filename,
                    "status": "configuration_error",
                    "simulator_status": simulator_status,
                    "error": message,
                }
            except Exception as exc:
                message = f"{filename}: {type(exc).__name__}: {exc}"
                errors.append(message)
                failed += 1
                row = {
                    "kernel": kernel,
                    "file": filename,
                    "status": "failed",
                    "simulator_status": simulator_status,
                    "error": message,
                }
            else:
                if has_measurement:
                    passed += 1
                    row = {"kernel": kernel, "file": filename, "status": "passed"}
                else:
                    missing += 1
                    reason = (
                        "no trusted independent hardware evidence"
                        if not trusted_hardware
                        else "no comparable actual hardware result"
                    )
                    row = {
                        "kernel": kernel,
                        "file": filename,
                        "status": "missing_evidence",
                        "error": f"{filename}: {reason}",
                    }
                row["simulator_status"] = simulator_status
            if capability_key is not None:
                row["capability"] = capability
                row["capability_key"] = capability_key
                row["evidence_key"] = evidence_key
                row["quantization_policy"] = policy_name
                row["quantization_policy_sha256"] = policy_sha256
            if simulator_error is not None:
                row["simulator_error"] = simulator_error
            results.append(row)

    if modern:
        for capability_key in sorted(
            set(expected_capabilities) - declared_capabilities
        ):
            expected = expected_capabilities[capability_key]
            missing += 1
            simulator_not_assessed += 1
            results.append(
                {
                    "kernel": expected["kernel"],
                    "status": "missing_evidence",
                    "simulator_status": "not_assessed",
                    "capability": expected["capability"],
                    "capability_key": capability_key,
                    "error": f"no vector for exact capability {capability_key}",
                }
            )

    if not manifest.get("vectors") and not modern:
        return {
            "status": "missing_evidence",
            "passed": 0,
            "failed": 0,
            "missing": 0,
            "configuration_errors": 0,
            "errors": ["manifest contains no vectors"],
            "kernels": [],
            "kernel_statuses": {},
        }

    precedence = {
        "passed": 0,
        "missing_evidence": 1,
        "failed": 2,
        "configuration_error": 3,
    }
    kernel_statuses: dict[str, str] = {}
    capability_statuses: dict[str, str] = {}
    capability_policy_statuses: dict[str, str] = {}
    for row in results:
        kernel = row["kernel"]
        status = row["status"]
        previous = kernel_statuses.get(kernel)
        if previous is None or precedence[status] > precedence[previous]:
            kernel_statuses[kernel] = status
        capability_key = row.get("capability_key")
        if capability_key is not None:
            previous = capability_statuses.get(capability_key)
            if previous is None or precedence[status] > precedence[previous]:
                capability_statuses[capability_key] = status
        evidence_key = row.get("evidence_key")
        if evidence_key is not None:
            previous = capability_policy_statuses.get(evidence_key)
            if previous is None or precedence[status] > precedence[previous]:
                capability_policy_statuses[evidence_key] = status
    simulator_precedence = {
        "passed": 0,
        "not_assessed": 1,
        "failed": 2,
        "configuration_error": 3,
    }
    simulator_kernel_statuses: dict[str, str] = {}
    simulator_capability_statuses: dict[str, str] = {}
    simulator_capability_policy_statuses: dict[str, str] = {}
    for row in results:
        kernel = row["kernel"]
        simulator_status = row.get("simulator_status", "not_assessed")
        previous = simulator_kernel_statuses.get(kernel)
        if (
            previous is None
            or simulator_precedence[simulator_status] > simulator_precedence[previous]
        ):
            simulator_kernel_statuses[kernel] = simulator_status
        capability_key = row.get("capability_key")
        if capability_key is not None:
            previous = simulator_capability_statuses.get(capability_key)
            if (
                previous is None
                or simulator_precedence[simulator_status]
                > simulator_precedence[previous]
            ):
                simulator_capability_statuses[capability_key] = simulator_status
        evidence_key = row.get("evidence_key")
        if evidence_key is not None:
            previous = simulator_capability_policy_statuses.get(evidence_key)
            if (
                previous is None
                or simulator_precedence[simulator_status]
                > simulator_precedence[previous]
            ):
                simulator_capability_policy_statuses[evidence_key] = simulator_status
    status = (
        "configuration_error"
        if configuration_errors
        else (
            "failed"
            if failed or simulator_failed
            else ("missing_evidence" if missing else "passed")
        )
    )
    if configuration_errors:
        simulator_status = "configuration_error"
    elif simulator_failed:
        simulator_status = "failed"
    elif simulator_passed and simulator_not_assessed:
        simulator_status = "partial"
    elif simulator_passed:
        simulator_status = "passed"
    else:
        simulator_status = "not_assessed"
    return {
        "status": status,
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "configuration_errors": configuration_errors,
        "errors": errors,
        "kernels": results,
        "kernel_statuses": dict(sorted(kernel_statuses.items())),
        "capability_statuses": dict(sorted(capability_statuses.items())),
        "capability_policy_statuses": dict(sorted(capability_policy_statuses.items())),
        "simulator_status": simulator_status,
        "simulator_passed": simulator_passed,
        "simulator_failed": simulator_failed,
        "simulator_kernel_statuses": dict(sorted(simulator_kernel_statuses.items())),
        "simulator_capability_statuses": dict(
            sorted(simulator_capability_statuses.items())
        ),
        "simulator_capability_policy_statuses": dict(
            sorted(simulator_capability_policy_statuses.items())
        ),
        "hardware_evidence": hardware_provenance if trusted_hardware else None,
    }


def _numpy():
    try:
        import numpy as np
    except (ImportError, ModuleNotFoundError) as exc:
        raise _ConformanceConfigurationError(
            "NumPy is required to verify conformance vectors; install it with "
            "`python -m pip install 'qbench[conformance]'`"
        ) from exc
    return np


def _assert_array_comparison(expected, actual, comparison: dict, np) -> None:
    if expected.shape != actual.shape:
        raise ValueError(
            f"shape mismatch: expected {expected.shape}, actual {actual.shape}"
        )
    kind = str(comparison.get("kind", "bit_exact"))
    if expected.dtype.kind not in {"f", "c"} or actual.dtype.kind not in {"f", "c"}:
        if expected.dtype != actual.dtype or not np.array_equal(expected, actual):
            raise AssertionError("non-floating outputs require bit-exact comparison")
        return
    if kind == "bit_exact":
        if expected.dtype != actual.dtype:
            raise ValueError("bit-exact comparison requires equal dtypes")
        matches = np.array_equal(
            np.ascontiguousarray(expected).view(np.uint8),
            np.ascontiguousarray(actual).view(np.uint8),
        )
    elif kind == "ulp":
        if expected.dtype != actual.dtype or expected.dtype.kind != "f":
            raise ValueError("ULP comparison requires equal floating dtypes")
        max_ulp = int(comparison.get("max_ulp", 0))
        if max_ulp < 0:
            raise ValueError("max_ulp must be non-negative")
        try:
            np.testing.assert_array_max_ulp(expected, actual, maxulp=max_ulp)
        except AssertionError:
            matches = False
        else:
            matches = True
    elif kind == "tolerance":
        matches = bool(
            np.allclose(
                expected,
                actual,
                rtol=float(comparison.get("rtol", 0.0)),
                atol=float(comparison.get("atol", 0.0)),
                equal_nan=bool(comparison.get("equal_nan", True)),
            )
        )
    else:
        raise ValueError(f"unknown comparison kind {kind!r}")
    if not matches:
        raise AssertionError(f"{kind} comparison failed")


def _verify_simulator_npz(
    path: Path,
    runner_name: str,
    capability: Mapping,
    policy: Mapping,
) -> dict:
    np = _numpy()
    runner = _CONFORMANCE_RUNNERS.get(runner_name)
    if runner is None:
        raise _ConformanceConfigurationError(
            f"unknown maintained conformance runner {runner_name!r}"
        )
    with np.load(path, allow_pickle=False) as vectors:
        if "expected" in vectors:
            raise _ConformanceConfigurationError(
                "schema-v2 archives must not embed an FP32 expected output"
            )
        arrays = {name: vectors[name].copy() for name in vectors.files}
    try:
        with redirect_stdout(io.StringIO()):
            simulated = runner(arrays, capability, policy)
        return _numpy_outputs(simulated)
    except _SimulatorUnavailableError:
        raise
    except _ConformanceConfigurationError:
        raise
    except Exception as exc:
        raise _SimulatorConformanceError(f"{type(exc).__name__}: {exc}") from exc


def _verify_hardware_npz(
    path: Path, expected: Mapping[str, object], comparison: dict
) -> bool:
    """Compare independently measured arrays with q-enabled simulator outputs."""

    np = _numpy()
    with np.load(path, allow_pickle=False) as vectors:
        actual_names = {name for name in vectors.files if name.startswith("actual")}
        if not actual_names:
            return False
        required = {
            ("actual" if name == "output" else name.replace("output", "actual", 1))
            for name in expected
        }
        if actual_names != required:
            raise _ConformanceConfigurationError(
                "actual hardware outputs must exactly match simulator tensor outputs: "
                f"expected {sorted(required)}, found {sorted(actual_names)}"
            )
        for name, golden in expected.items():
            actual_name = (
                "actual" if name == "output" else name.replace("output", "actual", 1)
            )
            _assert_array_comparison(golden, vectors[actual_name], comparison, np)
    return True


def _verify_npz(path: Path, comparison: dict) -> bool:
    """Compare portable ``expected``/``actual`` hardware arrays when present."""
    np = _numpy()

    with np.load(path, allow_pickle=False) as vectors:
        if "expected" not in vectors or "actual" not in vectors:
            # Input/golden-only bundles are portable vector definitions, but
            # are not evidence that a hardware implementation passed them.
            return False
        expected = vectors["expected"]
        actual = vectors["actual"]
    _assert_array_comparison(expected, actual, comparison, np)
    return True
