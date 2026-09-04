# QBench

Runtime-first inspection and quantized simulation for eager PyTorch inference.

QBench records executed dispatcher operations, resolves maintained kernel capabilities,
and verifies a strict simulator clone against your named input scenarios. CPU inspection
does not compile CUDA. Quantization-enabled simulation uses the GPU runtime.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dashboard,conformance,test]"
qbench inspect qbench.examples:tiny_provider --output-dir artifacts/tiny
qbench-admin init --database .qbench/platform.sqlite3 --username owner
qbench-dashboard --database .qbench/platform.sqlite3
```

For CPU-only PyTorch, first install the appropriate CPU wheel following
[PyTorch's installation instructions](https://pytorch.org/get-started/locally/).
Install this repository locally; a public PyPI release is not assumed.

```python
import torch
from qbench import Scenario, inspect_model, build_simulator

model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU()).eval()
scenario = Scenario("batch-1", (torch.ones(1, 4),))
result = inspect_model(model, [scenario])
print(result.fully_supported)
simulator = build_simulator(model, result.plan)
output = simulator.run(scenario)
```

“Fully supported” means **fully supported for the captured scenarios**. It is not
a claim of full model-path coverage, quantization-enabled execution on CPU, or
hardware bit-exactness. Explicit FP32 fallback always remains partial.

## Clean layout

| Path | Purpose |
| --- | --- |
| `qbench/` | Public API, capture, conversion, evaluation, providers, registry |
| `qbench/ops/` | Maintained simulator implementations |
| `qbench/quantization/` | Runtime, lazy CUDA codecs, compatibility workbench APIs |
| `qbench/dashboard/` | Standalone Streamlit model workbench |
| `qbench/dashboard/platform/` | Accounts, admin panel, authorization and usage audit |
| `qbench/conformance_vectors/` | Portable vectors and checksums |
| `qbench/validation/` | Reproducible public-model acceptance harness |
| `tests/` | Unit, integration, dashboard and optional GPU tests |
| `docs/` | GitHub Pages documentation source |

No experiment database, run orchestration, datasets, pretrained weights, container
filesystem, or old Git history is required or included. The original repository
retains its historical import shims; this clean distribution uses `qbench.*` imports.

## Documentation and verification

```bash
python -m pip install -r docs/requirements.txt
mkdocs serve
mkdocs build --strict
python -m pytest -q
python -m qbench.validation.public_models --help
```

Start with the [quickstart](docs/quickstart.md), [dashboard guide](docs/dashboard.md),
[API reference](docs/api.md), and [validation record](docs/validation.md).
The [multi-user platform guide](docs/platform.md) covers administration, per-user
features, dataset allowlists, account recovery, and secure single-host deployment.
For the legacy trusted local workbench, use `qbench-dashboard --single-user`.
The Pages workflow builds this site and deploys it when Pages is configured for
GitHub Actions. See [deployment](docs/deployment.md).

## Scope and provenance

Initial guarantees cover eager `nn.Module` inference and supplied scenario paths.
Training/backward, opaque TorchScript, compiled models without an eager original,
and distributed/FSDP execution are not certified.

Extracted from the QBench-Release workspace. No new software license is assigned
by this extraction; see [NOTICE](NOTICE.md) before redistribution.
