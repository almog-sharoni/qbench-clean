"""Canonical QBench operation and capability registries."""

from __future__ import annotations

import importlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .schemas import strict_json_safe


class OpRegistry:
    """Backwards-compatible module replacement registry.

    This is the authoritative class object used by both public and workbench APIs.
    """

    _registry: dict[str, type] = {}
    _supported_ops: dict[type, type] = {}
    _activation_ops: set[str] = set()
    _compliance_status: dict[str, str] = {}
    _supported_functions: set[Callable[..., Any]] = set()
    _under_construction_ops: set[str] = set()
    _replacements_by_name: dict[str, tuple[type, dict[str, Any]]] = {}
    _observed_variants: dict[str, dict[str, tuple[type, dict[str, Any]]]] = {}
    _unquantized_ops: set[str] = set()

    @classmethod
    def register(
        cls,
        op_name: str,
        original_cls=None,
        *,
        replaces=None,
        init_from_args=None,
        is_activation=False,
        compliance_status=None,
        under_construction=False,
        quantized=True,
        variant: str | None = None,
        default: bool = False,
    ):
        def decorator(cls_impl):
            cls._registry[op_name] = cls_impl
            if original_cls:
                cls._supported_ops[original_cls] = cls_impl
            if replaces is not None:
                entry = (cls_impl, dict(init_from_args or {}))
                if variant is not None:
                    cls._observed_variants.setdefault(replaces, {})[variant] = entry
                if default or variant is None:
                    existing = cls._replacements_by_name.get(replaces)
                    if (
                        existing is not None
                        and existing[0] is not cls_impl
                        and existing[0].__qualname__ != cls_impl.__qualname__
                    ):
                        raise ValueError(
                            f"Multiple default replacements registered for '{replaces}'"
                        )
                    cls._replacements_by_name[replaces] = entry
            if is_activation:
                cls._activation_ops.add(op_name)
            if compliance_status:
                cls._compliance_status[op_name] = compliance_status
            if under_construction:
                cls._under_construction_ops.add(op_name)
            if not quantized:
                cls._unquantized_ops.add(op_name)
            return cls_impl

        return decorator

    @classmethod
    def register_function(cls, func):
        cls._supported_functions.add(func)

    @classmethod
    def get(cls, op_name):
        if op_name not in cls._registry:
            raise ValueError(f"Operator {op_name} not found in registry.")
        return cls._registry[op_name]

    @classmethod
    def get_supported_ops(cls):
        return cls._supported_ops

    @classmethod
    def get_supported_functions(cls):
        return cls._supported_functions

    @classmethod
    def is_supported(cls, module_cls):
        return module_cls in cls._supported_ops

    @classmethod
    def get_quantized_op(cls, original_cls):
        return cls._supported_ops.get(original_cls)

    @classmethod
    def is_activation(cls, op_name):
        return op_name in cls._activation_ops

    @classmethod
    def get_compliance_status(cls, op_name):
        return cls._compliance_status.get(op_name)

    @classmethod
    def is_under_construction(cls, op_name):
        return op_name in cls._under_construction_ops

    @classmethod
    def get_replacement_by_name(cls, fn_name):
        return cls._replacements_by_name.get(fn_name)

    @classmethod
    def get_observed_variant(cls, fn_name, variant):
        return cls._observed_variants.get(fn_name, {}).get(variant)

    @classmethod
    def list_observed_variants(cls, fn_name):
        return list(cls._observed_variants.get(fn_name, {}))

    @classmethod
    def iter_observed_classes(cls, fn_name):
        seen = set()
        entries = (
            [cls._replacements_by_name[fn_name]]
            if fn_name in cls._replacements_by_name
            else []
        )
        entries += list(cls._observed_variants.get(fn_name, {}).values())
        for impl, _ in entries:
            if id(impl) not in seen:
                seen.add(id(impl))
                yield impl

    @classmethod
    def resolve_observed_class_from_config(cls, fn_name, cfg, parent_path=""):
        layers = (cfg or {}).get("layers", {}) if isinstance(cfg, dict) else {}
        qualified = f"{parent_path}.{fn_name}" if parent_path else fn_name
        layer_cfg = layers.get(qualified) or layers.get(fn_name)
        if layer_cfg is None:
            layer_cfg = next(
                (
                    v
                    for k, v in layers.items()
                    if isinstance(k, str) and k.endswith(f".{fn_name}")
                ),
                None,
            )
        variant = layer_cfg.get("variant") if isinstance(layer_cfg, dict) else None
        entry = (
            cls.get_observed_variant(fn_name, variant) if variant is not None else None
        )
        return (
            entry[0]
            if entry
            else (cls.get_replacement_by_name(fn_name) or (None, {}))[0]
        )

    @classmethod
    def is_quantized(cls, op_name):
        return op_name not in cls._unquantized_ops

    @classmethod
    def get_registration_name(cls, implementation):
        """Return the logical registry name for an implementation class."""
        for name, registered in cls._registry.items():
            if registered is implementation:
                return name
        return None

    @classmethod
    def is_ready_replacement(cls, original_cls):
        """Whether ``original_cls`` has a maintained, quantized replacement."""
        implementation = cls.get_quantized_op(original_cls)
        if implementation is None:
            return False
        name = cls.get_registration_name(implementation) or implementation.__name__
        return not cls.is_under_construction(name) and cls.is_quantized(name)


@dataclass(frozen=True)
class KernelSpec:
    name: str
    schemas: tuple[str, ...] = ()
    module_types: tuple[type, ...] = ()
    module_implementations: tuple[str, ...] = ()
    handler: Callable[..., Any] | None = field(default=None, compare=False, repr=False)
    conversion: str = "runtime_handler"
    classification: str = "quantized"
    ready: bool = True
    constraints: Callable[[tuple[Any, ...], dict[str, Any]], bool] | None = field(
        default=None, compare=False, repr=False
    )
    module_constraints: (
        Callable[[Any, tuple[Any, ...], dict[str, Any]], bool] | None
    ) = field(default=None, compare=False, repr=False)
    argument_constraints: dict[str, Any] = field(default_factory=dict)
    policy_overrides: dict[str, Any] = field(default_factory=dict)
    handler_quantized: bool = False
    counts_as_quantized: bool = True
    quantizes_weights: bool = False
    weight_operand: int | None = None
    weight_argument: str | None = None
    activation_policy: bool = False
    input_operands: dict[str, tuple[int, ...]] = field(default_factory=dict)
    conformance: dict[str, Any] = field(
        default_factory=lambda: {
            "kind": "tolerance",
            "rtol": 1e-5,
            "atol": 1e-6,
            "evidence": "missing",
        }
    )
    schema_constraints: dict[str, Callable[[tuple[Any, ...], dict[str, Any]], bool]] = (
        field(default_factory=dict, compare=False, repr=False)
    )
    route_variant: bool = False

    def accepts(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        if self.constraints is None:
            return True
        try:
            return bool(self.constraints(args, kwargs))
        except Exception:
            return False

    def accepts_module(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> bool:
        if self.module_constraints is None:
            return True
        try:
            return bool(self.module_constraints(module, args, kwargs))
        except Exception:
            return False

    def matches(
        self, schema: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> bool:
        """Match an exact dispatcher schema and privacy-safe argument metadata."""
        if not self.ready or schema not in self.schemas:
            return False
        constraint = self.schema_constraints.get(schema, self.constraints)
        if constraint is None:
            return True
        try:
            return bool(constraint(args, kwargs))
        except Exception:
            return False

    def implementation_for(self, native_type: type) -> type | None:
        """Resolve the exact implementation pinned by this maintained spec."""

        try:
            index = self.module_types.index(native_type)
        except ValueError:
            return None
        if index >= len(self.module_implementations):
            return None
        module_name, object_name = self.module_implementations[index].rsplit(".", 1)
        implementation = getattr(importlib.import_module(module_name), object_name)
        return implementation if isinstance(implementation, type) else None

    def implementation_path_for(self, native_type: type) -> str | None:
        try:
            index = self.module_types.index(native_type)
        except ValueError:
            return None
        if index >= len(self.module_implementations):
            return None
        return self.module_implementations[index]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("handler", None)
        data.pop("constraints", None)
        data.pop("schema_constraints", None)
        data.pop("module_constraints", None)
        data["module_types"] = [
            f"{t.__module__}.{t.__qualname__}" for t in self.module_types
        ]
        data["schemas"] = list(self.schemas)
        data["module_implementations"] = list(self.module_implementations)
        data["input_operands"] = {
            route: list(indices) for route, indices in self.input_operands.items()
        }
        # Populated with privacy-safe invocation metadata when a module route
        # is selected.  Keeping the empty container in the serialized spec
        # makes the schema explicit for functional-only kernels as well.
        data["module_invocations"] = {}
        return strict_json_safe(data)


def schema_route_key(schema: str, spec: KernelSpec | None = None) -> str:
    """Return the stable plan key for an exact schema capability variant."""

    base = f"schema:{schema}"
    if spec is not None and spec.route_variant:
        return f"{base}#kernel:{spec.name}"
    return base


def schema_from_route_key(route: str) -> str:
    """Recover the dispatcher schema from a validated schema route key."""

    if not route.startswith("schema:"):
        raise ValueError(f"Not a schema route: {route!r}")
    return route.removeprefix("schema:").split("#kernel:", 1)[0]


# Exact, versioned structural capability list. It is intentionally data, not a
# target-name heuristic. New overloads must be added explicitly.
STRUCTURAL_CAPABILITY_VERSION = 1
STRUCTURAL_SCHEMAS = frozenset(
    {
        "aten::alias",
        "aten::detach",
        "aten::clone",
        "aten::contiguous",
        "aten::view",
        "aten::_unsafe_view",
        "aten::reshape",
        "aten::flatten.using_ints",
        "aten::transpose.int",
        "aten::t",
        "aten::permute",
        "aten::squeeze",
        "aten::squeeze.dim",
        "aten::unsqueeze",
        "aten::expand",
        "aten::expand_as",
        "aten::slice.Tensor",
        "aten::select.int",
        "aten::_unsafe_index.Tensor",
        "aten::split.Tensor",
        "aten::split_with_sizes",
        "aten::chunk",
        "aten::unbind.int",
        "aten::cat",
        "aten::stack",
    }
)


_SEMANTIC_MODULE_ATTRIBUTES = (
    "training",
    "inplace",
    "dim",
    "approximate",
    "p",
    "in_features",
    "out_features",
    "in_channels",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "groups",
    "padding_mode",
    "output_padding",
    "normalized_shape",
    "eps",
    "momentum",
    "elementwise_affine",
    "affine",
    "track_running_stats",
    "ceil_mode",
    "return_indices",
    "count_include_pad",
    "divisor_override",
    "output_size",
    "embed_dim",
    "num_heads",
    "dropout",
    "batch_first",
    "add_zero_attn",
    "kdim",
    "vdim",
    "_qkv_same_embed_dim",
)


def module_semantic_configuration(module: torch.nn.Module) -> dict[str, Any]:
    """Return privacy-safe module attributes that affect eager semantics."""

    configuration: dict[str, Any] = {}
    for name in _SEMANTIC_MODULE_ATTRIBUTES:
        if not hasattr(module, name):
            continue
        value = getattr(module, name)
        if value is None or type(value) in {bool, int, float, str}:
            configuration[name] = value
        elif isinstance(value, (tuple, list)) and all(
            item is None or type(item) in {bool, int, float, str} for item in value
        ):
            configuration[name] = list(value)
    for name in (
        "weight",
        "bias",
        "in_proj_weight",
        "in_proj_bias",
        "q_proj_weight",
        "k_proj_weight",
        "v_proj_weight",
        "bias_k",
        "bias_v",
        "running_mean",
        "running_var",
    ):
        if hasattr(module, name):
            configuration[f"{name}_present"] = getattr(module, name) is not None
    return strict_json_safe(configuration)


def _runtime_quantization_enabled() -> bool:
    from .runtime import simulation_quantization_enabled

    return simulation_quantization_enabled()


def _configure_runtime_kernel(module: torch.nn.Module, value: Any) -> torch.nn.Module:
    """Configure a transient maintained kernel for one functional operation."""
    from .runtime import (
        simulation_input_quantization_allowed,
        simulation_quantization_policy,
        simulation_route_path,
    )

    enabled = _runtime_quantization_enabled()
    from .schemas import QuantizationPolicy

    policy = QuantizationPolicy.coerce(simulation_quantization_policy())
    route_path = simulation_route_path()
    module.eval()
    is_activation = any(
        spec.activation_policy
        and any(
            spec.implementation_for(native) is type(module)
            for native in spec.module_types
        )
        for spec in KERNEL_SPECS
    )
    settings = policy.resolve(route_path, activation=is_activation)
    module.input_quantization = (
        enabled
        and settings["input_quantization"]
        and simulation_input_quantization_allowed()
    )
    module.output_quantization = enabled and settings["output_quantization"]
    module.weight_quantization = enabled and settings["weight_quantization"]
    module.q_type = settings["q_type"]
    module.quantization_bias = None
    module.input_q_type = settings["input_q_type"]
    module.output_q_type = settings["output_q_type"]
    module.input_mode = settings["input_mode"]
    module.input_chunk_size = settings["input_chunk_size"]
    module.output_mode = settings["output_mode"]
    module.output_chunk_size = settings["output_chunk_size"]
    module.quant_mode = module.input_mode
    module.chunk_size = module.input_chunk_size
    module.weight_mode = str(settings["weight_mode"])
    module.weight_chunk_size = int(settings["weight_chunk_size"])
    module.act_mode = settings["act_mode"]
    module.act_chunk_size = settings["act_chunk_size"]
    module.rounding = settings["rounding"]
    module._qbench_hardware_arithmetic_enabled = enabled
    if torch.is_tensor(value):
        module.to(device=value.device)
    return module


def _exact_schema(func: Any) -> str:
    schema = getattr(func, "_schema", None)
    if schema is None:
        return str(func).replace("aten.", "aten::", 1)
    base = str(schema.name)
    overload = str(schema.overload_name or "default")
    return base if overload == "default" else f"{base}.{overload}"


_MISSING_ARGUMENT = object()


def _argument(args, kwargs, index: int, name: str, default=_MISSING_ARGUMENT):
    if index < len(args):
        return args[index]
    if name in kwargs:
        return kwargs[name]
    if default is not _MISSING_ARGUMENT:
        return default
    raise TypeError(f"Missing required argument {name!r}")


def _spatial_tuple(value, rank: int) -> tuple[int, ...]:
    if type(value) is int:
        return (value,) * rank
    values = tuple(value)
    if len(values) != rank or any(type(item) is not int for item in values):
        raise ValueError(f"Expected {rank} integer spatial values, got {value!r}")
    return values


def _construct_without_rng_advance(factory):
    """Construct a transient module without changing caller-visible RNG state."""
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        return factory()
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _bind_weighted_kernel(module, weight, bias, input_value):
    """Attach invocation tensors read-only and realize configured weight quantization."""
    module.weight = torch.nn.Parameter(weight.detach(), requires_grad=False)
    if bias is None:
        module.register_parameter("bias", None)
    else:
        module.bias = torch.nn.Parameter(bias.detach(), requires_grad=False)
    module = _configure_runtime_kernel(module, input_value)
    # The maintained channel codec requires rank >= 2.  Affine normalization
    # weights are one-dimensional, so realize those with the exact supported
    # tensor-mode path instead of advertising an impossible channel route.
    if weight.dim() < 2 and module.weight_mode == "channel":
        module.weight_mode = "tensor"
    if _runtime_quantization_enabled():
        module.calibrate_weights()
    return module


def _linear_handler(func, args, kwargs):
    """Execute exact ``aten::linear`` through the maintained QuantLinear."""
    if _exact_schema(func) != "aten::linear":
        raise RuntimeError(f"No maintained linear handler for {_exact_schema(func)}")
    from qbench.ops.quant_linear import QuantLinear

    input_value = _argument(args, kwargs, 0, "input")
    weight = _argument(args, kwargs, 1, "weight")
    bias = _argument(args, kwargs, 2, "bias", None)
    out_features, in_features = weight.shape
    kernel = _construct_without_rng_advance(
        lambda: QuantLinear(
            int(in_features),
            int(out_features),
            bias=bias is not None,
            device="meta",
            dtype=weight.dtype,
        )
    )
    return _bind_weighted_kernel(kernel, weight, bias, input_value)(input_value)


def _addmm_handler(func, args, kwargs):
    """Realize addmm as a weight-aware maintained linear plus final add."""
    if _exact_schema(func) != "aten::addmm":
        raise RuntimeError(f"No maintained addmm handler for {_exact_schema(func)}")
    from qbench.ops.quant_arithmetic import QuantAdd
    from qbench.ops.quant_linear import QuantLinear

    addend = _argument(args, kwargs, 0, "self")
    mat1 = _argument(args, kwargs, 1, "mat1")
    mat2 = _argument(args, kwargs, 2, "mat2")
    weight = mat2.transpose(0, 1).contiguous()
    product_kernel = _construct_without_rng_advance(
        lambda: QuantLinear(
            int(weight.shape[1]),
            int(weight.shape[0]),
            bias=False,
            device="meta",
            dtype=weight.dtype,
        )
    )
    product_kernel = _bind_weighted_kernel(product_kernel, weight, None, mat1)
    # addmm owns one public output boundary, after the addend is applied.
    product_kernel.output_quantization = False
    product = product_kernel(mat1)
    return _configure_runtime_kernel(QuantAdd(quant_mode="tensor"), addend)(
        addend, product
    )


def _convolution_handler(func, args, kwargs):
    """Execute supported 1D/2D non-transposed convolution via QBench modules."""
    schema = _exact_schema(func)
    input_value = _argument(args, kwargs, 0, "input")
    weight = _argument(args, kwargs, 1, "weight")
    bias = _argument(args, kwargs, 2, "bias", None)
    spatial_rank = int(weight.dim()) - 2
    defaults = [1] * spatial_rank
    stride = _spatial_tuple(
        _argument(args, kwargs, 3, "stride", defaults), spatial_rank
    )
    padding = _spatial_tuple(
        _argument(args, kwargs, 4, "padding", [0] * spatial_rank), spatial_rank
    )
    dilation = _spatial_tuple(
        _argument(args, kwargs, 5, "dilation", defaults), spatial_rank
    )
    if schema == "aten::convolution":
        transposed = _argument(args, kwargs, 6, "transposed")
        output_padding = _spatial_tuple(
            _argument(args, kwargs, 7, "output_padding"), spatial_rank
        )
        groups = _argument(args, kwargs, 8, "groups")
        if transposed or any(output_padding):
            raise RuntimeError("Maintained convolution handler is non-transposed only")
    elif schema in {"aten::conv1d", "aten::conv2d"}:
        groups = _argument(args, kwargs, 6, "groups", 1)
    else:
        raise RuntimeError(f"No maintained convolution handler for {schema}")

    if spatial_rank == 1:
        from qbench.ops.quant_conv1d import QuantConv1d as QuantConv
    elif spatial_rank == 2:
        from qbench.ops.quant_conv import QuantConv2d as QuantConv
    else:
        raise RuntimeError(
            f"Maintained convolution handler requires 1D/2D weights, got rank {weight.dim()}"
        )

    kernel = _construct_without_rng_advance(
        lambda: QuantConv(
            in_channels=int(weight.shape[1]) * int(groups),
            out_channels=int(weight.shape[0]),
            kernel_size=tuple(int(value) for value in weight.shape[2:]),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=int(groups),
            bias=bias is not None,
            device="meta",
            dtype=weight.dtype,
        )
    )
    return _bind_weighted_kernel(kernel, weight, bias, input_value)(input_value)


def _layer_norm_handler(func, args, kwargs):
    """Execute the tensor-output LayerNorm schema through QuantLayerNorm."""
    if _exact_schema(func) != "aten::layer_norm":
        raise RuntimeError(
            f"No maintained layer norm handler for {_exact_schema(func)}"
        )
    from qbench.ops.quant_ln import QuantLayerNorm

    input_value = _argument(args, kwargs, 0, "input")
    normalized_shape = tuple(_argument(args, kwargs, 1, "normalized_shape"))
    weight = _argument(args, kwargs, 2, "weight", None)
    bias = _argument(args, kwargs, 3, "bias", None)
    eps = float(_argument(args, kwargs, 4, "eps", 1e-5))
    cudnn_enabled = bool(_argument(args, kwargs, 5, "cudnn_enable", True))
    kernel = _construct_without_rng_advance(
        lambda: QuantLayerNorm(
            normalized_shape,
            eps=eps,
            elementwise_affine=True,
            device="meta",
            dtype=input_value.dtype,
        )
    )
    kernel = _bind_weighted_kernel(kernel, weight, bias, input_value)
    with torch.backends.cudnn.flags(enabled=cudnn_enabled):
        return kernel(input_value)


def _batch_norm_handler(func, args, kwargs):
    """Execute inference-only BatchNorm through the maintained QBench wrapper."""
    if _exact_schema(func) != "aten::batch_norm":
        raise RuntimeError(
            f"No maintained batch norm handler for {_exact_schema(func)}"
        )
    from qbench.ops.quant_bn import QuantBatchNorm1d, QuantBatchNorm2d

    input_value = _argument(args, kwargs, 0, "input")
    weight = _argument(args, kwargs, 1, "weight", None)
    bias = _argument(args, kwargs, 2, "bias", None)
    running_mean = _argument(args, kwargs, 3, "running_mean")
    running_var = _argument(args, kwargs, 4, "running_var")
    momentum = float(_argument(args, kwargs, 6, "momentum", 0.1))
    eps = float(_argument(args, kwargs, 7, "eps", 1e-5))
    cudnn_enabled = bool(_argument(args, kwargs, 8, "cudnn_enabled", True))
    implementation = QuantBatchNorm2d if input_value.dim() == 4 else QuantBatchNorm1d
    kernel = _construct_without_rng_advance(
        lambda: implementation(
            int(input_value.shape[1]),
            eps=eps,
            momentum=momentum,
            affine=weight is not None or bias is not None,
            track_running_stats=True,
            device=input_value.device,
            dtype=input_value.dtype,
        )
    )
    kernel.weight = (
        None
        if weight is None
        else torch.nn.Parameter(weight.detach(), requires_grad=False)
    )
    kernel.bias = (
        None if bias is None else torch.nn.Parameter(bias.detach(), requires_grad=False)
    )
    kernel.running_mean = running_mean.detach()
    kernel.running_var = running_var.detach()
    kernel = _configure_runtime_kernel(kernel, input_value)
    if weight is not None and weight.dim() < 2 and kernel.weight_mode == "channel":
        kernel.weight_mode = "tensor"
    if _runtime_quantization_enabled() and weight is not None:
        kernel.calibrate_weights()
    with torch.backends.cudnn.flags(enabled=cudnn_enabled):
        return kernel(input_value)


def _pool_pair(value, *, empty_default=None) -> tuple[int, int] | None:
    if type(value) is int:
        return (value, value)
    values = tuple(value)
    if not values and empty_default is not None:
        return empty_default
    if len(values) != 2 or any(type(item) is not int for item in values):
        raise ValueError(f"Expected two integer pooling values, got {value!r}")
    return values


def _max_pool2d_handler(func, args, kwargs):
    if _exact_schema(func) != "aten::max_pool2d_with_indices":
        raise RuntimeError(f"No maintained max-pool handler for {_exact_schema(func)}")
    from qbench.ops.quant_pooling import QuantMaxPool2d

    input_value = _argument(args, kwargs, 0, "self")
    kernel_size = _pool_pair(_argument(args, kwargs, 1, "kernel_size"))
    stride = _pool_pair(
        _argument(args, kwargs, 2, "stride", ()), empty_default=kernel_size
    )
    padding = _pool_pair(_argument(args, kwargs, 3, "padding", (0, 0)))
    dilation = _pool_pair(_argument(args, kwargs, 4, "dilation", (1, 1)))
    ceil_mode = _argument(args, kwargs, 5, "ceil_mode", False)
    kernel = QuantMaxPool2d(
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices=True,
        ceil_mode=ceil_mode,
    )
    return _configure_runtime_kernel(kernel, input_value)(input_value)


def _avg_pool2d_handler(func, args, kwargs):
    if _exact_schema(func) != "aten::avg_pool2d":
        raise RuntimeError(f"No maintained avg-pool handler for {_exact_schema(func)}")
    from qbench.ops.quant_pooling import QuantAvgPool2d

    input_value = _argument(args, kwargs, 0, "self")
    kernel_size = _pool_pair(_argument(args, kwargs, 1, "kernel_size"))
    stride = _pool_pair(
        _argument(args, kwargs, 2, "stride", ()), empty_default=kernel_size
    )
    padding = _pool_pair(_argument(args, kwargs, 3, "padding", (0, 0)))
    ceil_mode = _argument(args, kwargs, 4, "ceil_mode", False)
    count_include_pad = _argument(args, kwargs, 5, "count_include_pad", True)
    divisor_override = _argument(args, kwargs, 6, "divisor_override", None)
    kernel = QuantAvgPool2d(
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    return _configure_runtime_kernel(kernel, input_value)(input_value)


def _adaptive_avg_pool2d_handler(func, args, kwargs):
    if _exact_schema(func) != "aten::adaptive_avg_pool2d":
        raise RuntimeError(
            f"No maintained adaptive-pool handler for {_exact_schema(func)}"
        )
    from qbench.ops.quant_pooling import QuantAdaptiveAvgPool2d

    input_value = _argument(args, kwargs, 0, "self")
    output_size = _pool_pair(_argument(args, kwargs, 1, "output_size"))
    kernel = QuantAdaptiveAvgPool2d(output_size)
    return _configure_runtime_kernel(kernel, input_value)(input_value)


def _dropout_handler(func, args, kwargs):
    """Use the maintained dropout wrapper for the deterministic eval subset."""
    if _exact_schema(func) != "aten::dropout":
        raise RuntimeError(f"No maintained dropout handler for {_exact_schema(func)}")
    from qbench.ops.quant_dropout import QuantDropout

    input_value = _argument(args, kwargs, 0, "input")
    probability = float(_argument(args, kwargs, 1, "p"))
    kernel = _configure_runtime_kernel(
        QuantDropout(p=probability, inplace=False), input_value
    )
    # Eval dropout is structural and must preserve identity/alias behavior; it
    # is not an extra activation-quantization boundary.
    kernel.input_quantization = False
    kernel.output_quantization = False
    return kernel(input_value)


def _native_structural_handler(func, args, kwargs):
    """Execute an exactly constrained data-layout/index operation natively."""
    return func(*args, **kwargs)


def _clamp_handler(func, args, kwargs):
    if _exact_schema(func) != "aten::clamp":
        raise RuntimeError(f"No maintained clamp handler for {_exact_schema(func)}")
    from qbench.ops.quant_arithmetic import QuantClamp

    input_value = _argument(args, kwargs, 0, "self")
    minimum = _argument(args, kwargs, 1, "min", None)
    maximum = _argument(args, kwargs, 2, "max", None)
    output_dtype = input_value.dtype
    working_value = (
        input_value
        if input_value.dtype == torch.float32
        else input_value.to(dtype=torch.float32)
    )
    kernel = _configure_runtime_kernel(QuantClamp(quant_mode="tensor"), working_value)
    result = kernel(working_value, minimum, maximum)
    return result if result.dtype == output_dtype else result.to(dtype=output_dtype)


def _arithmetic_handler(func, args, kwargs):
    """Route supported arithmetic overloads through maintained QBench ops."""
    from qbench.ops.quant_arithmetic import (
        QuantAdd,
        QuantDiv,
        QuantMul,
        QuantSub,
    )

    schema = _exact_schema(func)
    implementation = {
        "aten::add.Tensor": QuantAdd,
        "aten::add_.Tensor": QuantAdd,
        "aten::add.Scalar": QuantAdd,
        "aten::sub.Tensor": QuantSub,
        "aten::mul.Tensor": QuantMul,
        "aten::mul.Scalar": QuantMul,
        "aten::div.Tensor": QuantDiv,
        "aten::div.Scalar": QuantDiv,
    }.get(schema)
    if implementation is None:
        raise RuntimeError(f"No maintained arithmetic handler for {schema}")
    if schema in {
        "aten::add.Tensor",
        "aten::add_.Tensor",
        "aten::add.Scalar",
        "aten::sub.Tensor",
    }:
        alpha = _argument(args, kwargs, 2, "alpha", 1)
        if isinstance(alpha, bool) or alpha != 1:
            raise RuntimeError(
                f"Maintained arithmetic handler requires alpha=1 for {schema}"
            )
    input_value = args[0]
    other = args[1]
    other_dtype = (
        other.dtype
        if torch.is_tensor(other)
        else (torch.float32 if type(other) is float else torch.int64)
    )
    output_dtype = (
        torch.float32
        if (
            schema.startswith("aten::div.")
            or input_value.dtype == torch.float32
            or other_dtype == torch.float32
        )
        else torch.int64
    )
    working_input = (
        input_value
        if input_value.dtype == torch.float32
        else input_value.to(dtype=torch.float32)
    )
    if not torch.is_tensor(other):
        other = torch.tensor(
            other, dtype=working_input.dtype, device=working_input.device
        )
    elif other.dtype != working_input.dtype:
        other = other.to(dtype=working_input.dtype)
    kernel = _configure_runtime_kernel(
        implementation(quant_mode="tensor"), working_input
    )
    result = kernel(working_input, other)
    if result.dtype != output_dtype:
        result = result.to(dtype=output_dtype)
    if schema == "aten::add_.Tensor":
        # Quantization may materialize a new result, but the source overload
        # promises to mutate and return operand zero. Copy the simulated value
        # back so callers observing an alias see the exact mutation.
        args[0].copy_(result)
        return args[0]
    return result


def _matmul_handler(func, args, kwargs):
    from qbench.ops.quant_matmul import QuantBMM, QuantMatMul

    implementation = QuantBMM if _exact_schema(func) == "aten::bmm" else QuantMatMul
    kernel = _configure_runtime_kernel(implementation(quant_mode="tensor"), args[0])
    return kernel(args[0], args[1])


def _scaled_dot_product_attention_handler(func, args, kwargs):
    if _exact_schema(func) != "aten::scaled_dot_product_attention":
        raise RuntimeError(
            "No maintained scaled-dot-product-attention handler for "
            f"{_exact_schema(func)}"
        )
    if not _runtime_quantization_enabled():
        # CUDA may select a fused SDPA algorithm whose rounding differs from a
        # Python-level matmul/softmax decomposition.  The routing-only dry run
        # must preserve the source operation exactly; the maintained
        # decomposition below is exercised when quantization is enabled.
        from .runtime import simulator_implementation

        with simulator_implementation():
            return func(*args, **kwargs)
    from qbench.ops.quant_arithmetic import QuantDiv
    from qbench.ops.quant_matmul import QuantMatMul
    from qbench.ops.quant_softmax import QuantSoftmax

    query = _argument(args, kwargs, 0, "query")
    key = _argument(args, kwargs, 1, "key")
    value = _argument(args, kwargs, 2, "value")

    scores_kernel = _configure_runtime_kernel(QuantMatMul(quant_mode="tensor"), query)
    scores_kernel.output_quantization = False
    scores = scores_kernel(query, key.transpose(-2, -1))

    divisor = torch.tensor(
        math.sqrt(float(query.shape[-1])),
        dtype=query.dtype,
        device=query.device,
    )
    scale_kernel = _configure_runtime_kernel(QuantDiv(quant_mode="tensor"), scores)
    # Q/K were already routed through the input boundary; scale is an exact
    # composite implementation detail rather than another public boundary.
    scale_kernel.input_quantization = False
    scale_kernel.output_quantization = False
    scores = scale_kernel(scores, divisor)

    softmax_kernel = _configure_runtime_kernel(QuantSoftmax(dim=-1), scores)
    softmax_kernel.output_quantization = False
    probabilities = softmax_kernel(scores)

    output_kernel = _configure_runtime_kernel(
        QuantMatMul(quant_mode="tensor"), probabilities
    )
    # The output matmul's second local operand is the public SDPA value input.
    # Probabilities are internal; do not misattribute them to query.
    output_kernel._qbench_source_operand_indices = {0: None, 1: 2}
    return output_kernel(probabilities, value)


def _activation_handler(func, args, kwargs):
    from qbench.ops.quant_activations import (
        QuantGELU,
        QuantHardsigmoid,
        QuantHardswish,
        QuantReLU,
        QuantReLU6,
        QuantSiLU,
    )

    schema = _exact_schema(func)
    if schema == "aten::gelu":
        approximate = args[1] if len(args) > 1 else kwargs.get("approximate", "none")
        kernel = QuantGELU(approximate=approximate, quant_mode="tensor")
    else:
        implementation = {
            "aten::relu": QuantReLU,
            "aten::relu6": QuantReLU6,
            "aten::silu": QuantSiLU,
            "aten::hardswish": QuantHardswish,
            "aten::hardsigmoid": QuantHardsigmoid,
        }.get(schema)
        if implementation is None:
            raise RuntimeError(f"No maintained activation handler for {schema}")
        kernel = implementation(quant_mode="tensor")
    return _configure_runtime_kernel(kernel, args[0])(args[0])


def _softmax_handler(func, args, kwargs):
    from qbench.ops.quant_softmax import QuantSoftmax

    schema = _exact_schema(func)
    dim = _argument(args, kwargs, 1, "dim")
    third = _argument(
        args,
        kwargs,
        2,
        "half_to_float" if schema == "aten::_softmax" else "dtype",
        False if schema == "aten::_softmax" else None,
    )
    if schema == "aten::_softmax":
        if third is not False:
            raise RuntimeError("Maintained _softmax requires half_to_float=False")
    elif schema == "aten::softmax.int":
        if third is not None:
            raise RuntimeError("Maintained softmax.int does not convert dtype")
    else:
        raise RuntimeError(f"No maintained softmax handler for {schema}")
    kernel = _configure_runtime_kernel(
        QuantSoftmax(dim=dim, quant_mode="tensor"), args[0]
    )
    return kernel(args[0])


def _mha_module_constraints(module, args, kwargs) -> bool:
    # The maintained decomposition assumes batch-major inputs and does not
    # reproduce native averaged attention-weight output.
    need_weights = kwargs.get("need_weights", args[4] if len(args) > 4 else True)
    is_causal = kwargs.get("is_causal", args[7] if len(args) > 7 else False)
    tensor_args = [
        _metadata_tensor(_argument(args, kwargs, index, name, None))
        for index, name in enumerate(("query", "key", "value"))
    ]
    if any(metadata is None for metadata in tensor_args):
        return False
    query, key, value = tensor_args
    query_shape = tuple(query.get("shape", ()))
    key_shape = tuple(key.get("shape", ()))
    value_shape = tuple(value.get("shape", ()))
    devices = {metadata.get("device") for metadata in tensor_args}
    key_padding_mask = _argument(args, kwargs, 3, "key_padding_mask", None)
    attn_mask = _argument(args, kwargs, 5, "attn_mask", None)
    if key_padding_mask is not None:
        if (
            not isinstance(key_padding_mask, dict)
            or key_padding_mask.get("kind") != "tensor"
            or key_padding_mask.get("dtype") != "torch.bool"
            or tuple(key_padding_mask.get("shape", ()))
            != (query_shape[0], key_shape[1])
            or key_padding_mask.get("device") not in devices
        ):
            return False
    if attn_mask is not None:
        # The maintained decomposition implements additive masks only. Native
        # bool masks require masked_fill and must not be certified by adding
        # their False/True values to scores.
        if (
            not isinstance(attn_mask, dict)
            or attn_mask.get("kind") != "tensor"
            or attn_mask.get("dtype") != "torch.float32"
            or tuple(attn_mask.get("shape", ())) != (query_shape[1], key_shape[1])
            or attn_mask.get("device") not in devices
        ):
            return False
    return (
        bool(getattr(module, "batch_first", False))
        and need_weights is False
        and is_causal is False
        and bool(getattr(module, "_qkv_same_embed_dim", False))
        and getattr(module, "in_proj_weight", None) is not None
        and getattr(module, "q_proj_weight", None) is None
        and getattr(module, "k_proj_weight", None) is None
        and getattr(module, "v_proj_weight", None) is None
        and getattr(module, "bias_k", None) is None
        and getattr(module, "bias_v", None) is None
        and not bool(getattr(module, "add_zero_attn", False))
        and all(
            metadata is not None
            and metadata.get("dtype") == "torch.float32"
            and len(metadata.get("shape", ())) == 3
            and metadata.get("shape", ())[-1] == module.embed_dim
            for metadata in tensor_args
        )
        and len(devices) == 1
        and query_shape[0] == key_shape[0] == value_shape[0]
        and key_shape[1] == value_shape[1]
    )


def _activation_module_constraints(module, args, kwargs) -> bool:
    # Maintained activation replacements explicitly copy quantized results
    # back into the source operand when ``inplace`` is enabled, preserving the
    # module's mutation and return-alias contract.
    return _float32_module_input(args, kwargs)


def _normalization_module_constraints(module, args, kwargs) -> bool:
    input_metadata = _module_input_metadata(args, kwargs)
    if input_metadata is None or input_metadata.get("dtype") != "torch.float32":
        return False
    if isinstance(module, torch.nn.LayerNorm):
        normalized_shape = tuple(module.normalized_shape)
        input_shape = tuple(input_metadata.get("shape", ()))
        # The maintained hardware arithmetic reduces over exactly the final
        # dimension and unconditionally applies gamma and beta.
        return (
            len(normalized_shape) == 1
            and bool(input_shape)
            and input_shape[-1] == normalized_shape[0]
            and bool(module.elementwise_affine)
            and torch.is_tensor(module.weight)
            and module.weight.dtype == torch.float32
            and torch.is_tensor(module.bias)
            and module.bias.dtype == torch.float32
        )
    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        return (
            bool(module.track_running_stats)
            and torch.is_tensor(module.running_mean)
            and torch.is_tensor(module.running_var)
            and module.running_mean.dtype == torch.float32
            and module.running_var.dtype == torch.float32
            and (module.weight is None or module.weight.dtype == torch.float32)
            and (module.bias is None or module.bias.dtype == torch.float32)
        )
    return False


def _linear_module_constraints(module, args, kwargs) -> bool:
    input_metadata = _module_input_metadata(args, kwargs)
    input_shape = (
        tuple(input_metadata.get("shape", ())) if input_metadata is not None else ()
    )
    return (
        input_metadata is not None
        and input_metadata.get("dtype") == "torch.float32"
        and bool(input_shape)
        and input_shape[-1] == module.in_features
        and module.weight.dtype == torch.float32
        and (module.bias is None or module.bias.dtype == torch.float32)
    )


def _softmax_module_constraints(module, args, kwargs) -> bool:
    input_metadata = _module_input_metadata(args, kwargs)
    rank = len(input_metadata.get("shape", ())) if input_metadata is not None else 0
    dim = getattr(module, "dim", None)
    return (
        input_metadata is not None
        and input_metadata.get("dtype") == "torch.float32"
        and type(dim) is int
        and rank > 0
        and -rank <= dim < rank
    )


def _pooling_module_constraints(module, args, kwargs) -> bool:
    input_metadata = _module_input_metadata(args, kwargs)
    if (
        input_metadata is None
        or input_metadata.get("dtype") != "torch.float32"
        or len(input_metadata.get("shape", ())) != 4
    ):
        return False
    if isinstance(module, torch.nn.MaxPool2d):
        return not bool(module.return_indices)
    if isinstance(module, torch.nn.AdaptiveAvgPool2d):
        output_size = module.output_size
        values = (output_size, output_size) if type(output_size) is int else output_size
        return (
            isinstance(values, (tuple, list))
            and len(values) == 2
            and all(type(value) is int and value > 0 for value in values)
        )
    return isinstance(module, torch.nn.AvgPool2d)


def _tensor_metadata_at(args: tuple[Any, ...], index: int) -> dict[str, Any] | None:
    if index >= len(args) or not isinstance(args[index], dict):
        return None
    return args[index] if args[index].get("kind") == "tensor" else None


def _float32_tensor_at(args: tuple[Any, ...], index: int) -> bool:
    metadata = _tensor_metadata_at(args, index)
    return metadata is not None and metadata.get("dtype") == "torch.float32"


def _metadata_tensor(value) -> dict[str, Any] | None:
    return value if isinstance(value, dict) and value.get("kind") == "tensor" else None


def _module_input_metadata(args, kwargs) -> dict[str, Any] | None:
    """Read a stock module's required input from either Python call form."""
    return _metadata_tensor(_argument(args, kwargs, 0, "input", None))


def _float32_module_input(args, kwargs) -> bool:
    metadata = _module_input_metadata(args, kwargs)
    return metadata is not None and metadata.get("dtype") == "torch.float32"


def _tensor_signature_is_supported(*metadata) -> bool:
    tensors = [value for value in metadata if value is not None]
    return (
        bool(tensors)
        and all(
            value.get("dtype") == "torch.float32"
            and isinstance(value.get("device"), str)
            and all(type(dim) is int and dim > 0 for dim in value.get("shape", ()))
            for value in tensors
        )
        and len({value["device"] for value in tensors}) == 1
    )


def _linear_constraints(args, kwargs) -> bool:
    input_metadata = _metadata_tensor(_argument(args, kwargs, 0, "input"))
    weight_metadata = _metadata_tensor(_argument(args, kwargs, 1, "weight"))
    bias_value = _argument(args, kwargs, 2, "bias", None)
    bias_metadata = None if bias_value is None else _metadata_tensor(bias_value)
    if input_metadata is None or weight_metadata is None:
        return False
    if bias_value is not None and bias_metadata is None:
        return False
    input_shape = tuple(input_metadata.get("shape", ()))
    weight_shape = tuple(weight_metadata.get("shape", ()))
    if len(input_shape) < 1 or len(weight_shape) != 2:
        return False
    if input_shape[-1] != weight_shape[1]:
        return False
    if bias_metadata is not None and tuple(bias_metadata.get("shape", ())) != (
        weight_shape[0],
    ):
        return False
    return _tensor_signature_is_supported(
        input_metadata, weight_metadata, bias_metadata
    )


def _addmm_constraints(args, kwargs) -> bool:
    addend = _metadata_tensor(_argument(args, kwargs, 0, "self"))
    mat1 = _metadata_tensor(_argument(args, kwargs, 1, "mat1"))
    mat2 = _metadata_tensor(_argument(args, kwargs, 2, "mat2"))
    if addend is None or mat1 is None or mat2 is None:
        return False
    addend_shape = tuple(addend.get("shape", ()))
    mat1_shape = tuple(mat1.get("shape", ()))
    mat2_shape = tuple(mat2.get("shape", ()))
    if len(mat1_shape) != 2 or len(mat2_shape) != 2:
        return False
    if mat1_shape[1] != mat2_shape[0]:
        return False
    if addend_shape not in {(mat2_shape[1],), (mat1_shape[0], mat2_shape[1])}:
        return False
    alpha = _argument(args, kwargs, 4, "alpha", 1)
    beta = _argument(args, kwargs, 3, "beta", 1)
    if alpha != 1 or beta != 1 or isinstance(alpha, bool) or isinstance(beta, bool):
        return False
    return _tensor_signature_is_supported(addend, mat1, mat2)


def _unary_float32_constraints(args, kwargs) -> bool:
    return _float32_tensor_at(args, 0)


def _matmul_constraints(args, kwargs) -> bool:
    return _float32_tensor_at(args, 0) and _float32_tensor_at(args, 1)


def _scaled_dot_product_attention_constraints(args, kwargs) -> bool:
    tensors = [
        _metadata_tensor(_argument(args, kwargs, index, name, None))
        for index, name in enumerate(("query", "key", "value"))
    ]
    if any(metadata is None for metadata in tensors):
        return False
    query, key, value = tensors
    query_shape = tuple(query.get("shape", ()))
    key_shape = tuple(key.get("shape", ()))
    value_shape = tuple(value.get("shape", ()))
    attn_mask = _argument(args, kwargs, 3, "attn_mask", None)
    dropout_p = _argument(args, kwargs, 4, "dropout_p", 0.0)
    is_causal = _argument(args, kwargs, 5, "is_causal", False)
    scale = _argument(args, kwargs, 6, "scale", None)
    enable_gqa = _argument(args, kwargs, 7, "enable_gqa", False)
    return (
        len(query_shape) >= 3
        and len(query_shape) == len(key_shape) == len(value_shape)
        and query_shape[:-2] == key_shape[:-2] == value_shape[:-2]
        and query_shape[-1] == key_shape[-1]
        and key_shape[-2] == value_shape[-2]
        and attn_mask is None
        and isinstance(dropout_p, (int, float))
        and not isinstance(dropout_p, bool)
        and float(dropout_p) == 0.0
        and is_causal is False
        and scale is None
        and enable_gqa is False
        and _tensor_signature_is_supported(query, key, value)
    )


def _arange_index_constraints(args, kwargs) -> bool:
    if len(args) != 1 or type(args[0]) is not int or args[0] < 0:
        return False
    dtype = kwargs.get("dtype")
    layout = kwargs.get("layout")
    pin_memory = kwargs.get("pin_memory", False)
    return (
        dtype in {None, "torch.int64"}
        and layout in {None, "torch.strided"}
        and pin_memory is False
    )


def _dtype_cast_constraints(args, kwargs) -> bool:
    source = _metadata_tensor(_argument(args, kwargs, 0, "self", None))
    target = _argument(args, kwargs, 1, "dtype", None)
    non_blocking = _argument(args, kwargs, 2, "non_blocking", False)
    copy_value = _argument(args, kwargs, 3, "copy", False)
    if (
        source is None
        or target not in {"torch.float32", "torch.int64"}
        or non_blocking is not False
        or copy_value is not False
    ):
        return False
    source_dtype = source.get("dtype")
    rank = len(source.get("shape", ()))
    return source_dtype == target or (
        rank <= 2 and source_dtype in {"torch.float32", "torch.int64"}
    )


def _integer_index_arithmetic_constraints(args, kwargs) -> bool:
    if len(args) < 2:
        return False
    left = _metadata_tensor(args[0])
    alpha = _argument(args, kwargs, 2, "alpha", 1)
    if (
        left is None
        or left.get("dtype") != "torch.int64"
        or len(left.get("shape", ())) > 2
        or isinstance(alpha, bool)
        or alpha != 1
    ):
        return False
    other = args[1]
    if type(other) is int:
        return True
    right = _metadata_tensor(other)
    return (
        right is not None
        and right.get("dtype") == "torch.int64"
        and right.get("device") == left.get("device")
        and len(right.get("shape", ())) <= 2
    )


def _integer_clamp_constraints(args, kwargs) -> bool:
    value = _metadata_tensor(_argument(args, kwargs, 0, "self", None))
    minimum = _argument(args, kwargs, 1, "min", None)
    maximum = _argument(args, kwargs, 2, "max", None)
    return (
        value is not None
        and value.get("dtype") == "torch.int64"
        and len(value.get("shape", ())) <= 2
        and any(bound is not None for bound in (minimum, maximum))
        and all(bound is None or type(bound) is int for bound in (minimum, maximum))
        and (minimum is None or maximum is None or minimum <= maximum)
    )


def _clamp_constraints(args, kwargs) -> bool:
    value = _metadata_tensor(_argument(args, kwargs, 0, "self", None))
    if value is None or value.get("dtype") != "torch.float32":
        return False
    minimum = _argument(args, kwargs, 1, "min", None)
    maximum = _argument(args, kwargs, 2, "max", None)
    bounds = (minimum, maximum)
    return (
        any(bound is not None for bound in bounds)
        and all(
            bound is None
            or (
                isinstance(bound, (int, float))
                and not isinstance(bound, bool)
                and math.isfinite(float(bound))
            )
            for bound in bounds
        )
        and (minimum is None or maximum is None or float(minimum) <= float(maximum))
    )


def _arithmetic_constraints(args, kwargs) -> bool:
    if len(args) < 2:
        return False
    left = _metadata_tensor(args[0])
    if left is None or left.get("dtype") not in {
        "torch.float32",
        "torch.int64",
    }:
        return False
    left_rank = len(left.get("shape", ()))
    if left.get("dtype") == "torch.int64" and left_rank > 2:
        return False
    other = args[1]
    if isinstance(other, dict):
        if other.get("kind") != "tensor" or other.get("dtype") not in {
            "torch.float32",
            "torch.int64",
        }:
            return False
        if other.get("device") != left.get("device"):
            return False
        if (
            left.get("dtype") == "torch.int64" or other.get("dtype") == "torch.int64"
        ) and (left_rank > 2 or len(other.get("shape", ())) > 2):
            return False
        if left.get("dtype") == "torch.int64" and other.get("dtype") == "torch.int64":
            return False
    elif (
        not isinstance(other, (int, float))
        or isinstance(other, bool)
        or not math.isfinite(float(other))
    ):
        return False
    elif left.get("dtype") == "torch.int64":
        return False
    alpha = _argument(args, kwargs, 2, "alpha", 1)
    return not isinstance(alpha, bool) and alpha == 1


def _float_arithmetic_constraints(args, kwargs) -> bool:
    if len(args) < 2 or not _float32_tensor_at(args, 0):
        return False
    left = _metadata_tensor(args[0])
    other = args[1]
    if isinstance(other, dict):
        right = _metadata_tensor(other)
        if (
            right is None
            or right.get("dtype") != "torch.float32"
            or right.get("device") != left.get("device")
        ):
            return False
    elif (
        not isinstance(other, (int, float))
        or isinstance(other, bool)
        or not math.isfinite(float(other))
    ):
        return False
    alpha = _argument(args, kwargs, 2, "alpha", 1)
    return not isinstance(alpha, bool) and alpha == 1


def _inplace_add_constraints(args, kwargs) -> bool:
    if len(args) < 2:
        return False
    left = _metadata_tensor(args[0])
    right = _metadata_tensor(args[1])
    alpha = _argument(args, kwargs, 2, "alpha", 1)
    return (
        left is not None
        and right is not None
        and tuple(left.get("shape", ())) == tuple(right.get("shape", ()))
        and _tensor_signature_is_supported(left, right)
        and not isinstance(alpha, bool)
        and alpha == 1
    )


def _softmax_constraints(args, kwargs) -> bool:
    if not _float32_tensor_at(args, 0):
        return False
    metadata = _tensor_metadata_at(args, 0)
    rank = len(metadata.get("shape", ()))
    dim = _argument(args, kwargs, 1, "dim", None)
    third = (
        args[2] if len(args) > 2 else kwargs.get("half_to_float", kwargs.get("dtype"))
    )
    return (
        type(dim) is int
        and rank > 0
        and -rank <= dim < rank
        and (third is None or third is False)
    )


def _convolution_constraints(args, kwargs, spatial_rank: int) -> bool:
    """Restrict functional convolution to exact maintained 1D/2D semantics."""
    input_metadata = _metadata_tensor(_argument(args, kwargs, 0, "input"))
    weight_metadata = _metadata_tensor(_argument(args, kwargs, 1, "weight"))
    bias_value = _argument(args, kwargs, 2, "bias", None)
    bias_metadata = None if bias_value is None else _metadata_tensor(bias_value)
    if input_metadata is None or weight_metadata is None:
        return False
    if bias_value is not None and bias_metadata is None:
        return False
    input_shape = tuple(input_metadata.get("shape", ()))
    weight_shape = tuple(weight_metadata.get("shape", ()))
    if len(input_shape) != spatial_rank + 2 or len(weight_shape) != spatial_rank + 2:
        return False
    stride = _spatial_tuple(
        _argument(args, kwargs, 3, "stride", [1] * spatial_rank), spatial_rank
    )
    padding = _spatial_tuple(
        _argument(args, kwargs, 4, "padding", [0] * spatial_rank), spatial_rank
    )
    dilation = _spatial_tuple(
        _argument(args, kwargs, 5, "dilation", [1] * spatial_rank), spatial_rank
    )
    groups = _argument(args, kwargs, 6, "groups", 1)
    if type(groups) is not int or groups <= 0:
        return False
    if any(value <= 0 for value in stride + dilation) or any(
        value < 0 for value in padding
    ):
        return False
    if input_shape[1] != weight_shape[1] * groups:
        return False
    if weight_shape[0] % groups:
        return False
    if bias_metadata is not None and tuple(bias_metadata.get("shape", ())) != (
        weight_shape[0],
    ):
        return False
    return _tensor_signature_is_supported(
        input_metadata, weight_metadata, bias_metadata
    )


def _conv1d_constraints(args, kwargs) -> bool:
    return _convolution_constraints(args, kwargs, 1)


def _conv2d_constraints(args, kwargs) -> bool:
    return _convolution_constraints(args, kwargs, 2)


def _aten_convolution_constraints(args, kwargs) -> bool:
    input_metadata = _metadata_tensor(_argument(args, kwargs, 0, "input"))
    weight_metadata = _metadata_tensor(_argument(args, kwargs, 1, "weight"))
    if input_metadata is None or weight_metadata is None:
        return False
    spatial_rank = len(weight_metadata.get("shape", ())) - 2
    if spatial_rank not in {1, 2}:
        return False
    transposed = _argument(args, kwargs, 6, "transposed")
    output_padding = _spatial_tuple(
        _argument(args, kwargs, 7, "output_padding"), spatial_rank
    )
    if transposed is not False or any(output_padding):
        return False
    # Rebind the convolution schema's groups position to the common helper.
    normalized_args = (
        _argument(args, kwargs, 0, "input"),
        _argument(args, kwargs, 1, "weight"),
        _argument(args, kwargs, 2, "bias"),
        _argument(args, kwargs, 3, "stride"),
        _argument(args, kwargs, 4, "padding"),
        _argument(args, kwargs, 5, "dilation"),
        _argument(args, kwargs, 8, "groups"),
    )
    return _convolution_constraints(normalized_args, {}, spatial_rank)


def _layer_norm_constraints(args, kwargs) -> bool:
    input_metadata = _metadata_tensor(_argument(args, kwargs, 0, "input"))
    normalized_shape = tuple(_argument(args, kwargs, 1, "normalized_shape"))
    weight = _metadata_tensor(_argument(args, kwargs, 2, "weight", None))
    bias = _metadata_tensor(_argument(args, kwargs, 3, "bias", None))
    eps = _argument(args, kwargs, 4, "eps", 1e-5)
    cudnn_enable = _argument(args, kwargs, 5, "cudnn_enable", True)
    if input_metadata is None or weight is None or bias is None:
        return False
    input_shape = tuple(input_metadata.get("shape", ()))
    return (
        len(normalized_shape) == 1
        and type(normalized_shape[0]) is int
        and normalized_shape[0] > 0
        and bool(input_shape)
        and input_shape[-1] == normalized_shape[0]
        and tuple(weight.get("shape", ())) == normalized_shape
        and tuple(bias.get("shape", ())) == normalized_shape
        and isinstance(eps, (int, float))
        and not isinstance(eps, bool)
        and math.isfinite(float(eps))
        and float(eps) > 0
        and type(cudnn_enable) is bool
        and _tensor_signature_is_supported(input_metadata, weight, bias)
    )


def _batch_norm_constraints(args, kwargs) -> bool:
    input_metadata = _metadata_tensor(_argument(args, kwargs, 0, "input"))
    weight_value = _argument(args, kwargs, 1, "weight", None)
    bias_value = _argument(args, kwargs, 2, "bias", None)
    running_mean = _metadata_tensor(_argument(args, kwargs, 3, "running_mean"))
    running_var = _metadata_tensor(_argument(args, kwargs, 4, "running_var"))
    training = _argument(args, kwargs, 5, "training")
    momentum = _argument(args, kwargs, 6, "momentum", 0.1)
    eps = _argument(args, kwargs, 7, "eps", 1e-5)
    cudnn_enabled = _argument(args, kwargs, 8, "cudnn_enabled", True)
    weight = None if weight_value is None else _metadata_tensor(weight_value)
    bias = None if bias_value is None else _metadata_tensor(bias_value)
    if (
        input_metadata is None
        or running_mean is None
        or running_var is None
        or (weight_value is not None and weight is None)
        or (bias_value is not None and bias is None)
    ):
        return False
    input_shape = tuple(input_metadata.get("shape", ()))
    if len(input_shape) not in {2, 3, 4}:
        return False
    channels = input_shape[1]
    channel_shape = (channels,)
    if any(
        tuple(metadata.get("shape", ())) != channel_shape
        for metadata in (running_mean, running_var, weight, bias)
        if metadata is not None
    ):
        return False
    scalars = (momentum, eps)
    return (
        training is False
        and type(cudnn_enabled) is bool
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in scalars
        )
        and float(eps) > 0
        and _tensor_signature_is_supported(
            input_metadata, running_mean, running_var, weight, bias
        )
    )


def _pool_input_constraints(args, kwargs) -> dict[str, Any] | None:
    input_metadata = _metadata_tensor(_argument(args, kwargs, 0, "self"))
    if input_metadata is None or len(input_metadata.get("shape", ())) != 4:
        return None
    return input_metadata if _tensor_signature_is_supported(input_metadata) else None


def _max_pool2d_constraints(args, kwargs) -> bool:
    if _pool_input_constraints(args, kwargs) is None:
        return False
    kernel = _pool_pair(_argument(args, kwargs, 1, "kernel_size"))
    stride = _pool_pair(_argument(args, kwargs, 2, "stride", ()), empty_default=kernel)
    padding = _pool_pair(_argument(args, kwargs, 3, "padding", (0, 0)))
    dilation = _pool_pair(_argument(args, kwargs, 4, "dilation", (1, 1)))
    ceil_mode = _argument(args, kwargs, 5, "ceil_mode", False)
    return (
        all(value > 0 for value in kernel + stride + dilation)
        and all(value >= 0 for value in padding)
        and type(ceil_mode) is bool
    )


def _avg_pool2d_constraints(args, kwargs) -> bool:
    if _pool_input_constraints(args, kwargs) is None:
        return False
    kernel = _pool_pair(_argument(args, kwargs, 1, "kernel_size"))
    stride = _pool_pair(_argument(args, kwargs, 2, "stride", ()), empty_default=kernel)
    padding = _pool_pair(_argument(args, kwargs, 3, "padding", (0, 0)))
    ceil_mode = _argument(args, kwargs, 4, "ceil_mode", False)
    count_include_pad = _argument(args, kwargs, 5, "count_include_pad", True)
    divisor_override = _argument(args, kwargs, 6, "divisor_override", None)
    return (
        all(value > 0 for value in kernel + stride)
        and all(value >= 0 for value in padding)
        and type(ceil_mode) is bool
        and type(count_include_pad) is bool
        and (
            divisor_override is None
            or (type(divisor_override) is int and divisor_override > 0)
        )
    )


def _adaptive_avg_pool2d_constraints(args, kwargs) -> bool:
    if _pool_input_constraints(args, kwargs) is None:
        return False
    output_size = _pool_pair(_argument(args, kwargs, 1, "output_size"))
    return all(value > 0 for value in output_size)


def _dropout_constraints(args, kwargs) -> bool:
    input_metadata = _metadata_tensor(_argument(args, kwargs, 0, "input"))
    probability = _argument(args, kwargs, 1, "p")
    training = _argument(args, kwargs, 2, "train")
    return (
        input_metadata is not None
        and _tensor_signature_is_supported(input_metadata)
        and isinstance(probability, (int, float))
        and not isinstance(probability, bool)
        and math.isfinite(float(probability))
        and 0 <= float(probability) <= 1
        and training is False
    )


def _convolution_module_constraints(module, args, kwargs) -> bool:
    weight = getattr(module, "weight", None)
    input_metadata = _module_input_metadata(args, kwargs)
    return (
        isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d))
        and torch.is_tensor(weight)
        and weight.dtype == torch.float32
        and getattr(module, "padding_mode", None) == "zeros"
        and input_metadata is not None
        and input_metadata.get("dtype") == "torch.float32"
        and len(input_metadata.get("shape", ()))
        == (3 if isinstance(module, torch.nn.Conv1d) else 4)
    )


def _default_specs() -> list[KernelSpec]:
    import torch.nn as nn

    def tolerance() -> dict[str, Any]:
        return {
            "kind": "tolerance",
            "rtol": 1e-5,
            "atol": 1e-6,
            "evidence": "missing",
        }

    def bit_exact() -> dict[str, Any]:
        return {"kind": "bit_exact", "evidence": "missing"}

    module_route = "module_swap_or_runtime_handler"
    return [
        KernelSpec(
            "linear",
            ("aten::linear",),
            (nn.Linear,),
            module_implementations=("qbench.ops.quant_linear.QuantLinear",),
            handler=_linear_handler,
            constraints=_linear_constraints,
            module_constraints=_linear_module_constraints,
            argument_constraints={
                "input_rank_min": 1,
                "weight_rank": 2,
                "bias_shape": "out_features_or_none",
                "dtype": "torch.float32",
                "same_device": True,
            },
            handler_quantized=True,
            quantizes_weights=True,
            weight_operand=1,
            weight_argument="weight",
            conversion=module_route,
            conformance=tolerance(),
        ),
        KernelSpec(
            "addmm",
            ("aten::addmm",),
            handler=_addmm_handler,
            constraints=_addmm_constraints,
            argument_constraints={
                "mat1_rank": 2,
                "mat2_rank": 2,
                "self_shape": ["out_features", "output_shape"],
                "alpha": 1,
                "beta": 1,
                "dtype": "torch.float32",
                "same_device": True,
            },
            handler_quantized=True,
            quantizes_weights=True,
            weight_operand=2,
            weight_argument="mat2",
            input_operands={"aten::addmm": (0, 1)},
            conformance=tolerance(),
        ),
        KernelSpec(
            "convolution",
            ("aten::convolution",),
            (nn.Conv1d, nn.Conv2d),
            module_implementations=(
                "qbench.ops.quant_conv1d.QuantConv1d",
                "qbench.ops.quant_conv.QuantConv2d",
            ),
            handler=_convolution_handler,
            constraints=_aten_convolution_constraints,
            module_constraints=_convolution_module_constraints,
            argument_constraints={
                "input_rank": [3, 4],
                "weight_rank_matches_input": True,
                "dtype": "torch.float32",
                "transposed": False,
                "output_padding": 0,
                "same_device": True,
            },
            handler_quantized=True,
            quantizes_weights=True,
            weight_operand=1,
            weight_argument="weight",
            conversion=module_route,
            conformance=tolerance(),
        ),
        KernelSpec(
            "conv1d",
            ("aten::conv1d",),
            handler=_convolution_handler,
            constraints=_conv1d_constraints,
            argument_constraints={
                "input_rank": 3,
                "weight_rank": 3,
                "dtype": "torch.float32",
                "padding": "non_negative_integer",
                "same_device": True,
            },
            handler_quantized=True,
            quantizes_weights=True,
            weight_operand=1,
            weight_argument="weight",
            conformance=tolerance(),
        ),
        KernelSpec(
            "conv2d",
            ("aten::conv2d",),
            handler=_convolution_handler,
            constraints=_conv2d_constraints,
            argument_constraints={
                "input_rank": 4,
                "weight_rank": 4,
                "dtype": "torch.float32",
                "padding": "non_negative_integer",
                "same_device": True,
            },
            handler_quantized=True,
            quantizes_weights=True,
            weight_operand=1,
            weight_argument="weight",
            conformance=tolerance(),
        ),
        KernelSpec(
            "normalization",
            ("aten::layer_norm",),
            (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d),
            module_implementations=(
                "qbench.ops.quant_ln.QuantLayerNorm",
                "qbench.ops.quant_bn.QuantBatchNorm1d",
                "qbench.ops.quant_bn.QuantBatchNorm2d",
            ),
            handler=_layer_norm_handler,
            constraints=_layer_norm_constraints,
            module_constraints=_normalization_module_constraints,
            argument_constraints={
                "normalized_shape_rank": 1,
                "affine_weight_and_bias": True,
                "dtype": "torch.float32",
                "same_device": True,
            },
            policy_overrides={
                "rank1_weight_channel": {
                    "effective_mode": "tensor",
                    "reason": "rank-one affine weights have no channel axis",
                }
            },
            handler_quantized=True,
            quantizes_weights=True,
            weight_operand=2,
            weight_argument="weight",
            conversion=module_route,
            classification="composite",
            conformance=tolerance(),
        ),
        KernelSpec(
            "batch_norm",
            ("aten::batch_norm",),
            handler=_batch_norm_handler,
            constraints=_batch_norm_constraints,
            argument_constraints={
                "input_rank": [2, 3, 4],
                "training": False,
                "running_statistics": "required",
                "dtype": "torch.float32",
                "same_device": True,
            },
            policy_overrides={
                "rank1_weight_channel": {
                    "effective_mode": "tensor",
                    "reason": "rank-one affine weights have no channel axis",
                }
            },
            handler_quantized=True,
            quantizes_weights=True,
            weight_operand=1,
            weight_argument="weight",
            classification="composite",
            conformance=tolerance(),
        ),
        KernelSpec(
            "native_layer_norm",
            ("aten::native_layer_norm",),
            ready=False,
            classification="composite",
            argument_constraints={"reason": "tuple statistics output not implemented"},
            conformance=tolerance(),
        ),
        KernelSpec(
            "native_batch_norm",
            ("aten::_native_batch_norm_legit_no_training",),
            ready=False,
            classification="composite",
            argument_constraints={"reason": "tuple statistics output not implemented"},
            conformance=tolerance(),
        ),
        KernelSpec(
            "activation",
            (
                "aten::relu",
                "aten::relu6",
                "aten::gelu",
                "aten::silu",
                "aten::hardswish",
                "aten::hardsigmoid",
            ),
            (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.Hardswish, nn.Hardsigmoid),
            module_implementations=(
                "qbench.ops.quant_activations.QuantReLU",
                "qbench.ops.quant_activations.QuantReLU6",
                "qbench.ops.quant_activations.QuantGELU",
                "qbench.ops.quant_activations.QuantSiLU",
                "qbench.ops.quant_activations.QuantHardswish",
                "qbench.ops.quant_activations.QuantHardsigmoid",
            ),
            handler=_activation_handler,
            constraints=_unary_float32_constraints,
            module_constraints=_activation_module_constraints,
            handler_quantized=True,
            activation_policy=True,
            conversion=module_route,
            conformance=tolerance(),
        ),
        KernelSpec(
            "softmax",
            ("aten::_softmax", "aten::softmax.int"),
            (nn.Softmax,),
            module_implementations=("qbench.ops.quant_softmax.QuantSoftmax",),
            handler=_softmax_handler,
            constraints=_softmax_constraints,
            module_constraints=_softmax_module_constraints,
            handler_quantized=True,
            activation_policy=True,
            conversion=module_route,
            classification="composite",
            conformance=tolerance(),
        ),
        KernelSpec(
            "attention",
            (),
            (nn.MultiheadAttention,),
            module_implementations=(
                "qbench.ops.quant_mha.DecomposedMultiheadAttention",
            ),
            module_constraints=_mha_module_constraints,
            argument_constraints={"batch_first": True, "need_weights": False},
            conversion=module_route,
            classification="composite",
            quantizes_weights=True,
            input_operands={"module": (0, 1, 2)},
            conformance=tolerance(),
        ),
        KernelSpec(
            "scaled_dot_product_attention",
            ("aten::scaled_dot_product_attention",),
            handler=_scaled_dot_product_attention_handler,
            constraints=_scaled_dot_product_attention_constraints,
            argument_constraints={
                "input_rank_min": 3,
                "dtype": "torch.float32",
                "same_device": True,
                "attn_mask": None,
                "dropout_p": 0.0,
                "is_causal": False,
                "scale": None,
                "enable_gqa": False,
            },
            handler_quantized=True,
            input_operands={"aten::scaled_dot_product_attention": (0, 1, 2)},
            classification="composite",
            conformance=tolerance(),
        ),
        KernelSpec(
            "scaled_dot_product_attention_backend",
            (
                "aten::_scaled_dot_product_attention_math",
                "aten::_scaled_dot_product_flash_attention",
                "aten::_scaled_dot_product_flash_attention_for_cpu",
                "aten::_scaled_dot_product_efficient_attention",
                "aten::_scaled_dot_product_cudnn_attention",
            ),
            ready=False,
            argument_constraints={"reason": "exact runtime handler not implemented"},
            classification="composite",
            conformance=tolerance(),
        ),
        KernelSpec(
            "index_arange",
            ("aten::arange",),
            handler=_native_structural_handler,
            constraints=_arange_index_constraints,
            argument_constraints={
                "signature": "integer_end",
                "dtype": "torch.int64",
                "layout": "torch.strided",
                "pin_memory": False,
            },
            classification="structural",
            counts_as_quantized=False,
            conformance=bit_exact(),
        ),
        KernelSpec(
            "index_dtype_cast",
            ("aten::to.dtype",),
            handler=_native_structural_handler,
            constraints=_dtype_cast_constraints,
            argument_constraints={
                "dtypes": ["torch.float32", "torch.int64"],
                "changed_dtype_rank_max": 2,
                "non_blocking": False,
                "copy": False,
            },
            classification="structural",
            counts_as_quantized=False,
            conformance=bit_exact(),
        ),
        KernelSpec(
            "index_arithmetic",
            ("aten::add.Tensor", "aten::sub.Tensor"),
            handler=_native_structural_handler,
            constraints=_integer_index_arithmetic_constraints,
            argument_constraints={
                "dtype": "torch.int64",
                "rank_max": 2,
                "same_device": True,
                "alpha": 1,
            },
            classification="structural",
            counts_as_quantized=False,
            conformance=bit_exact(),
            route_variant=True,
        ),
        KernelSpec(
            "index_clamp",
            ("aten::clamp",),
            handler=_native_structural_handler,
            constraints=_integer_clamp_constraints,
            argument_constraints={
                "dtype": "torch.int64",
                "rank_max": 2,
                "bounds": "integer_or_none",
            },
            classification="structural",
            counts_as_quantized=False,
            conformance=bit_exact(),
            route_variant=True,
        ),
        KernelSpec(
            "clamp",
            ("aten::clamp",),
            handler=_clamp_handler,
            constraints=_clamp_constraints,
            argument_constraints={
                "dtype": "torch.float32",
                "bounds": "finite_scalar",
            },
            handler_quantized=True,
            input_operands={"aten::clamp": (0,)},
            conformance=tolerance(),
        ),
        KernelSpec(
            "matmul",
            ("aten::matmul", "aten::mm", "aten::bmm"),
            handler=_matmul_handler,
            constraints=_matmul_constraints,
            handler_quantized=True,
            input_operands={
                "aten::matmul": (0, 1),
                "aten::mm": (0, 1),
                "aten::bmm": (0, 1),
            },
            conformance=tolerance(),
        ),
        KernelSpec(
            "arithmetic",
            (
                "aten::add.Tensor",
                "aten::add.Scalar",
                "aten::sub.Tensor",
                "aten::mul.Tensor",
                "aten::mul.Scalar",
                "aten::div.Tensor",
                "aten::div.Scalar",
            ),
            handler=_arithmetic_handler,
            constraints=_float_arithmetic_constraints,
            schema_constraints={
                "aten::add.Tensor": _arithmetic_constraints,
                "aten::sub.Tensor": _arithmetic_constraints,
            },
            argument_constraints={
                "default_dtype": "torch.float32",
                "add_sub_tensor_dtypes": [
                    "torch.float32",
                    "mixed torch.float32/torch.int64",
                ],
                "mixed_integer_operand_rank_max": 2,
                "same_device": True,
                "finite_scalar": True,
                "alpha": 1,
            },
            handler_quantized=True,
            input_operands={
                "aten::add.Tensor": (0, 1),
                "aten::add.Scalar": (0, 1),
                "aten::sub.Tensor": (0, 1),
                "aten::mul.Tensor": (0, 1),
                "aten::mul.Scalar": (0, 1),
                "aten::div.Tensor": (0, 1),
                "aten::div.Scalar": (0, 1),
            },
            conformance=tolerance(),
        ),
        KernelSpec(
            "inplace_add",
            ("aten::add_.Tensor",),
            handler=_arithmetic_handler,
            constraints=_inplace_add_constraints,
            argument_constraints={
                "operand_shapes": "exact_match",
                "alpha": 1,
                "dtype": "torch.float32",
                "same_device": True,
                "mutation": "operand_zero",
                "return_alias": "operand_zero",
            },
            handler_quantized=True,
            input_operands={"aten::add_.Tensor": (0, 1)},
            conformance=tolerance(),
        ),
        KernelSpec(
            "pooling",
            (),
            (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d),
            module_implementations=(
                "qbench.ops.quant_pooling.QuantMaxPool2d",
                "qbench.ops.quant_pooling.QuantAvgPool2d",
                "qbench.ops.quant_pooling.QuantAdaptiveAvgPool2d",
            ),
            module_constraints=_pooling_module_constraints,
            conversion=module_route,
            conformance=tolerance(),
        ),
        KernelSpec(
            "max_pool2d",
            ("aten::max_pool2d_with_indices",),
            handler=_max_pool2d_handler,
            constraints=_max_pool2d_constraints,
            argument_constraints={
                "input_rank": 4,
                "dtype": "torch.float32",
                "return_indices": True,
            },
            handler_quantized=True,
            conformance=tolerance(),
        ),
        KernelSpec(
            "avg_pool2d",
            ("aten::avg_pool2d",),
            handler=_avg_pool2d_handler,
            constraints=_avg_pool2d_constraints,
            argument_constraints={"input_rank": 4, "dtype": "torch.float32"},
            handler_quantized=True,
            conformance=tolerance(),
        ),
        KernelSpec(
            "adaptive_avg_pool2d",
            ("aten::adaptive_avg_pool2d",),
            handler=_adaptive_avg_pool2d_handler,
            constraints=_adaptive_avg_pool2d_constraints,
            argument_constraints={
                "input_rank": 4,
                "output_size": "two_positive_integers",
                "dtype": "torch.float32",
            },
            handler_quantized=True,
            conformance=tolerance(),
        ),
        KernelSpec(
            "embedding",
            ("aten::embedding",),
            (nn.Embedding,),
            conversion=module_route,
            ready=False,
            conformance=tolerance(),
        ),
        KernelSpec(
            "dropout",
            ("aten::dropout",),
            (nn.Dropout,),
            module_implementations=("qbench.ops.quant_dropout.QuantDropout",),
            handler=_dropout_handler,
            constraints=_dropout_constraints,
            argument_constraints={"training": False, "dtype": "torch.float32"},
            conversion=module_route,
            classification="structural",
            counts_as_quantized=False,
            conformance=bit_exact(),
        ),
        KernelSpec(
            "native_dropout",
            ("aten::native_dropout",),
            ready=False,
            classification="structural",
            counts_as_quantized=False,
            argument_constraints={"reason": "RNG mask tuple output not implemented"},
            conformance=bit_exact(),
        ),
    ]


KERNEL_SPECS: list[KernelSpec] = _default_specs()


def find_kernel(schema: str, args=(), kwargs=None) -> KernelSpec | None:
    kwargs = {} if kwargs is None else kwargs
    return next(
        (
            spec
            for spec in KERNEL_SPECS
            if spec.matches(schema, tuple(args), dict(kwargs))
        ),
        None,
    )


def list_kernel_specs() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in KERNEL_SPECS]


# Preserve the historical registry API with idempotent function registration.
standard_functions = [
    F.conv2d,
    F.linear,
    F.batch_norm,
    F.layer_norm,
    F.dropout,
    F.softmax,
    F.scaled_dot_product_attention,
    torch.softmax,
    torch.relu,
    F.relu,
    F.relu6,
    F.silu,
    F.gelu,
    F.hardswish,
]
for _function in standard_functions:
    OpRegistry.register_function(_function)
