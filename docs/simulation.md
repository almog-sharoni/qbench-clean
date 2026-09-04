# Conversion & simulation

```python
from qbench import inspect_model, build_simulator

result = inspect_model(model, scenarios)
simulator = build_simulator(model, result.plan, strict=True)
output = simulator.run(scenarios[0])
```

Inspection automatically builds and dry-runs a strict simulator clone by
default. Calling `build_simulator` gives you the executable simulator for later
evaluation. The source model is retained; conversion works on a clone.

## Strict runtime routing

Recognized native modules are safely swapped. Functional and dynamic operations
are routed by runtime handlers, so conversion does not require a successful FX
trace. Low-level calls inside an active composite simulator kernel are treated
as implementation details, preventing LayerNorm or Softmax internals from being
double-counted as independent source operations.

The audit checks that planned source operations reach their intended simulator
kernels. Unresolved operations, unexpected native work, or unrealized mappings
fail strict verification. New invocation shapes or branches can invalidate a
plan; capture them explicitly and inspect again.

Always execute through `Simulator.run(invocation)`. Direct model calls are
rejected where they would bypass required runtime routing. Exported state alone
does not encode that behavior and is not a standalone converted executable.

## Explicit fallback

```python
from qbench import InspectionConfig

result = inspect_model(
    model, scenarios,
    InspectionConfig(allow_fp32_fallback=True),
)
simulator = build_simulator(model, result.plan, strict=False)
```

Use fallback to investigate partial models, not to claim complete support. The
report remains partial even when a fallback path numerically matches PyTorch.

## Quantization-enabled execution

```python
from qbench import InspectionConfig, QuantizationPolicy

config = InspectionConfig(
    device="cuda",
    quantization_enabled=True,
    quantization_policy=QuantizationPolicy(quantization_type="fp8_e4m3"),
)
```

Use compatible CUDA inputs/model placement and the [GPU environment](environments.md).
Read quantization-stage evidence as well as replacement coverage. A routing dry
run with quantization disabled cannot establish quantization accuracy or hardware
fidelity. Match the full policy when comparing experiments, not only the format.
