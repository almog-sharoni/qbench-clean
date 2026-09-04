# Artifacts & privacy

Use a dedicated output directory for each run:

```python
from qbench.artifacts import write_artifacts

write_artifacts("artifacts/run-001", result, evaluation=report)
```

| File | Contents |
| --- | --- |
| `manifest.json` | Schema version, file sizes and SHA-256 checksums, scenario names, provenance |
| `support.json` | Schema-v3 qualified support, gaps, coverage, verification and diagnostics |
| `operations.jsonl.gz` | Ordered raw dispatcher ledger, one JSON object per invocation |
| `plan.json` | Serializable simulation plan and quantization policy |
| `evaluation.json` | Optional paired metrics, timing and detailed data |
| `state.pt` | Optional simulator state dictionary, not a standalone executable |

The writer replaces these named outputs in the requested directory, and removes
stale `evaluation.json`/`state.pt` when they are not supplied for the new run.
Keep output directories separate from source files and valuable unrelated data.

## Inspect a ledger

```python
import gzip
import json

with gzip.open("artifacts/run-001/operations.jsonl.gz", "rt") as stream:
    first_operation = json.loads(next(stream))
print(first_operation)
```

Compression omits timestamps and source filenames for reproducible ledger bytes.
JSON serialization is strict: non-finite metric floats use stable string
spellings such as `"Infinity"`, rather than invalid JSON numeric literals.

## Privacy boundaries

Inspection does not store user tensor values. It does record shapes, dtypes,
devices, relevant scalar arguments, module names, scenario names, and callsite
metadata. These can still reveal model structure or private paths. Model/provider
exception details are redacted in persistable diagnostics.

Detailed activation samples and exported state **do contain values**. Review
artifacts before sharing and disable optional retention for sensitive workloads.
Only load trusted state files and trusted provider code.

## Reproducing a result

Keep exact package versions, model/weights identity, preprocessing, scenario
definitions, dataset sample manifest, seed, device, and quantization policy.
The operation ledger intentionally cannot reconstruct private inputs. Supply
those again through the original trusted provider.
