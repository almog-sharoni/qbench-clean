# Installation & first inspection

## Install from this checkout

Use Python 3.10 or newer in an isolated environment. The declared PyTorch lower
bound is 2.1; the precise environment tested for this extraction is recorded in
[validation](validation.md). A full PyTorch-version matrix is not yet certified.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For CPU-only use, install CPU PyTorch first using the
[official selector](https://pytorch.org/get-started/locally/). QBench import and CPU
inspection do not load CUDA codecs or compile extensions.

Optional components:

```bash
python -m pip install -e ".[dashboard,conformance,test]"
```

`vision` adds torchvision/timm; `dashboard` also adds Streamlit and pandas;
`conformance` adds NumPy; `test` adds pytest. The CLI is installed by the base package.

## Inspect without downloads

```bash
qbench inspect qbench.examples:tiny_provider --output-dir artifacts/tiny
```

The built-in provider creates deterministic random weights and two input shapes.
No pretrained model or dataset is downloaded. Inspect `support.json` and
`manifest.json` in the output directory.

## Use Python directly

```python
import torch
from qbench import Scenario, inspect_model, build_simulator

model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU()).eval()
scenarios = [
    Scenario("single", (torch.ones(1, 4),)),
    Scenario("batch", (torch.ones(3, 4),)),
]
result = inspect_model(model, scenarios)
assert result.fully_supported
simulator = build_simulator(model, result.plan)
actual = simulator.run(scenarios[0])
with torch.inference_mode():
    expected = model(*scenarios[0].args)
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
```

This is a **quantization-disabled** routing check. It is intentionally not a claim
about low-precision accuracy. Continue with [simulation](simulation.md) for GPU use.

## Save evaluation artifacts

From the repository root after installation:

```bash
python examples/inspect_and_evaluate.py --output artifacts/example
qbench evaluate qbench.examples:tiny_provider --output-dir artifacts/evaluation
qbench-dashboard --single-user
```

If a command exits with `2`, read its gaps: analysis may have succeeded with
partial support. Exit `1` is a configuration or execution failure.

The command above opens the trusted local legacy workbench on loopback only.
For authenticated sharing, initialize accounts and use the [multi-user platform](platform.md).
