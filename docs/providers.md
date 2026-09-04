# Providers & scenarios

`Scenario(name, args=(), kwargs={})` represents a named eager invocation. Supply
positional arguments as a tuple and keyword arguments as a mapping. Nested
inputs and outputs are supported where their leaves can be safely handled by
the capture and evaluation runtime.

```python
from qbench import Scenario

scenario = Scenario("masked", (tokens,), {"attention_mask": mask})
```

Names must be nonempty. Choose distinct, meaningful names for distinct capture
paths. User tensor values are not serialized into inspection traces.

## Maintained providers

| Provider | Intended input |
| --- | --- |
| `DirectObjectProvider` | An existing model, scenarios, optional evaluation loader |
| `TorchvisionProvider` | A torchvision model name, scenarios, optional weights |
| `TimmProvider` | A timm model name, scenarios, optional pretrained weights |
| `LegacyAdapterProvider` | A trusted adapter exposing scenarios or `sample_input` |

Vision dependencies load only when used. When reproducibility matters, pin the
dependency version, exact model weights, preprocessing, seed, and sample list.
Providers are responsible for meaningful evaluation preprocessing and outputs.

## Protocol

A trusted `ModelProvider` implements six methods:

| Method | Contract |
| --- | --- |
| `build_model()` | Return the eager reference `nn.Module` |
| `clone_model(model)` | Return an independent clone without mutating the original |
| `capture_scenarios()` | Return a finite iterable of named `Scenario` objects |
| `evaluation_loader()` | Return evaluation batches |
| `prepare_evaluation_batch(batch)` | Return a scenario, or `(scenario, target)` |
| `select_metric_output(output)` | Select the task-relevant tensor or nested outputs |

`DirectObjectProvider` uses deep-copy cloning. For models that cannot be
deep-copied, implement cloning by rebuilding the architecture and loading an
independent state. Avoid sharing parameters, buffers, or mutable custom state.

## CLI loading

Use `package.module:object`. The object can be a provider instance, a provider
class with a zero-argument constructor, or a zero-argument factory. The bundled
`qbench.examples:tiny_provider` is a minimal executable example.

Provider modules must be installed or importable. Loading a provider executes
Python code; only use providers you trust. QBench does not sandbox model/provider
code or custom replacement classes. Do not use untrusted pickled models.
