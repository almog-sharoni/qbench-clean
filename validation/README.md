# Validation records

`platform.json` records the subsequent multi-user platform source, installed-wheel,
GPU and real-browser checks. `summary.json` records the extraction validation gates and deliberately unverified
acceptance criteria. `extraction-comparison.json` compares the original workspace
and this standalone runtime on six deterministic random-weight model/shape cases.

The comparison excludes FX/export enrichment and enables strict runtime routing
with quantization disabled. Its hashes and classifications are regression evidence,
not a pretrained accuracy or hardware-fidelity certificate.

To rerun the maintained checks:

```bash
python -m pytest -q
python -m build --wheel
python -m pip install --force-reinstall --no-deps dist/qbench-0.1.0-py3-none-any.whl
# Run this from outside the source checkout, using an absolute script path:
python /path/to/qbench-clean/scripts/smoke_installed.py
mkdocs build --strict
python scripts/check_site.py
qbench kernels verify qbench/conformance_vectors
```

GPU vector verification returns code 2 when simulator checks pass but independent
hardware evidence is missing. Read `simulator_status` separately from `status`.
Use the public-model acceptance harness and a real baseline/dataset for accuracy
and performance claims. See `docs/validation.md`.
