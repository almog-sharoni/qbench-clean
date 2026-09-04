# Evaluation & metrics

```python
from qbench import EvaluationConfig, build_simulator, evaluate, inspect_provider
from qbench.examples import tiny_provider

provider = tiny_provider()
result = inspect_provider(provider)
reference = provider.build_model()
simulator = build_simulator(reference, result.plan)
report = evaluate(reference, simulator, provider, EvaluationConfig(metrics="fast"))
print(report.to_dict())
```

## Fast bundle

The default path performs one reference forward and one simulator forward per
batch. It retains no intermediate activations. Reference and simulator invocations
receive isolated inputs and aligned RNG state; training flags and RNG state are
restored after evaluation.

The output bundle includes MAE, MSE, cosine similarity, SQNR, non-finite counts,
and prediction agreement where an argmax is meaningful. Select a built-in task
with `EvaluationConfig(task=...)`:

| Task | Required provider outputs |
| --- | --- |
| `generic` | Corresponding tensor/nested outputs |
| `classification` | Class logits and integer target labels |
| `language_modeling` | Token logits and compatible token targets |
| `feature_matching` | Comparable task-selected feature outputs |

Use `select_metric_output` to avoid comparing unrelated auxiliary outputs. Output
structure and shape mismatches are errors, not silently dropped leaves. Metric
meaning depends on the provider's preprocessing and target alignment.

CPU timing is coarse wall time. CUDA uses event-based coarse timing; it is not a
kernel-level profiler. Small CUDA final outputs may be queued for batched CPU
reductions, with private snapshots so reused model buffers cannot corrupt metrics.
This bounded final-output queue is not intermediate activation retention.

## Detailed diagnostics

```python
config = EvaluationConfig(
    metrics="detailed",
    latency_repetitions=3,
    retain_activations=True,
    activation_retention_max_elements=4096,
    compliance_scan=True,
)
```

Detailed mode enables per-layer collection, activation summaries/histograms,
optional bounded samples, compliance scans, extra timing forwards, and memory
information where available. Expect overhead. Activation samples contain model
values and may be sensitive; do not publish them blindly.

No runtime-loaded metric plugin mechanism is included. Use the built-in tasks
and a trusted provider to select meaningful outputs.

## Performance and accuracy claims

The target is at most 10% median overhead compared with the same dual-forward
loop on the target GPU. That is an acceptance criterion, not a guarantee for all
models/devices. Re-run the [acceptance harness](validation.md) after metric or
kernel changes. Real accuracy needs identical weights, preprocessing, sample
selection, policy, and task labels—not just a random-input smoke test.
