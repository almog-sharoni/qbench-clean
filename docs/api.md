# Python API reference

Import public interfaces from `qbench`. Internal runtime modules may change;
prefer these entry points for application code.

## Inspection

```python
inspect_model(model, scenarios, config=None) -> InspectionResult
inspect_provider(provider, config=None) -> InspectionResult
```

`scenarios` accepts a `Scenario` or iterable of scenarios. `config` accepts an
`InspectionConfig` or compatible mapping. Provider inspection honors the trusted
provider's model-building and cloning contract.

`InspectionResult` exposes `support`, `operations`, `plan`, `verification`,
`diagnostics`, `fully_supported`, and serialization helpers. `operations` is the
raw ordered ledger; use the support summary for normal UI presentation.

### InspectionConfig defaults

| Field | Default | Purpose |
| --- | --- | --- |
| `device` | `"cpu"` | Inspection execution device |
| `quantization_enabled` | `False` | Enable actual quantization in verification |
| `allow_fp32_fallback` | `False` | Permit partial execution; never fully supported |
| `verify` | `True` | Automatic converted dry run |
| `capture_callsites` | `True` | Include fallback callsite ownership |
| `enable_fx` | `True` | Optional FX enrichment |
| `enable_export` | `True` | Optional export enrichment |
| `conformance_directory` | `None` | Optional external evidence bundle |
| `quantization_policy` | `QuantizationPolicy()` | Validated kernel configuration |

`QuantizationPolicy` controls format, bias, input/weight/output quantization,
first-layer behavior, tensor/chunk/channel modes, and chunk sizes. Invalid or
unsupported combinations fail validation; inspect `QuantizationPolicy()` or the
CLI help for the exact installed version's defaults.

## Simulation

```python
build_simulator(model, plan, strict=True) -> Simulator
simulator.run(invocation) -> object
simulator.verify(scenarios) -> VerificationResult
simulator.state_dict() -> dict
```

`plan` is a `SimulationPlan` emitted by inspection. The simulator owns a clone;
the source model is not the converted model. `run` takes a `Scenario` and preserves
the model's supported output structure. Do not bypass runtime routing by calling
the underlying model directly.

## Evaluation

```python
evaluate(reference, simulator, provider, config=None) -> EvaluationReport
```

`EvaluationConfig` defaults to `metrics="fast"`, `task="generic"`, all loader
batches, one timing repetition, and no retained activation samples. Extra
repetitions and activation retention require `metrics="detailed"`.

`EvaluationReport` contains `metrics`, `batches`, `reference_forwards`,
`simulator_forwards`, `timing`, `details`, and `to_dict()`.

## Schemas, capabilities and artifacts

- `Scenario(name, args=(), kwargs={})`: named invocation; tensor values are not
  included in inspection artifacts.
- `ModelProvider`: six-method trusted protocol; see [providers](providers.md).
- `KernelSpec`: maintained exact matchers, constraints, handler, strategy,
  classifications, quantized-counting and conformance metadata.
- `OpRegistry`: canonical registry shared by public runtime and workbench.
- `qbench.kernels.list_kernels()`: serializable maintained capability catalog.
- `qbench.kernels.verify_kernels(directory)`: portable vector verification report.
- `qbench.artifacts.write_artifacts(directory, result, evaluation=None,
  state_dict=None)`: schema-v3 artifact writer; optional arguments are keyword-only.
- `QBenchError`: configuration, capture, and simulation error base class.

The compatibility workbench API is available under
`qbench.quantization.model_workbench`; see [migration](migration.md).
