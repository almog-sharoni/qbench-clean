# Migration from QBench-Release

The original checkout remains intact. This standalone repository has a fresh Git
history and physically owns its runtime under `qbench/`; it does not import from
`runspace` or rely on adding that directory to `sys.path`.

| Previous location | Clean distribution |
| --- | --- |
| `runspace.src.ops` / `src.ops` | `qbench.ops` |
| `runspace.src.registry.op_registry` | `qbench.registry` |
| `runspace.src.quantization.model_workbench` | `qbench.quantization.model_workbench` |
| Shared dashboard workbench tab | `qbench-dashboard` |
| `runspace.public_model_acceptance` | `qbench.validation.public_models` |

Legacy workbench functions including `analyze_model`, planning, `convert_model`,
sample inference, and classification benchmarking remain available under the
new module location. A single legacy sample input becomes one named scenario.
Recipe readers retain versions 1 and 2; newly emitted support/recipe data uses
version 3 with legacy-compatible fields where applicable.

Historical `src.*` and `runspace.*` import aliases remain in the original
repository, not in this clean package. Update application imports when moving.
Serialized Python pickles tied to historical module names are not a portable
migration format; rebuild models from trusted definitions and state dictionaries.

## Deliberately excluded

Experiment databases, pipeline/run orchestration, SLM/feature-matching experiment
adapters, PIM/cache studies, raw datasets, weights, logs, and external simulator
source trees are excluded. The generic adapter's old experiment `create_metrics`
hook directs users to `qbench.evaluate` instead of importing the removed
`MetricsEngine`. Built-in task metrics and trusted provider extension points remain.

Newly captured operations may make verdicts more conservative. That is different
from losing a previously working source-to-simulator mapping. Compare raw ledgers
and qualified reports when investigating a migration difference.
