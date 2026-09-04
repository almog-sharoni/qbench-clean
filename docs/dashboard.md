# Dashboard & model workbench

The dashboard is a client of the QBench runtime and compatibility workbench API.
It does not require the old experiment database, dashboard setup script, or any
other experiment tab.

## Launch

```bash
python -m pip install -e ".[dashboard]"
qbench-admin init --database .qbench/platform.sqlite3 --username owner
qbench-dashboard --database .qbench/platform.sqlite3 --host 127.0.0.1 --port 8501
```

Open the local URL printed by Streamlit and sign in with your chosen credentials.
The authenticated platform provides a guided inspect → convert → evaluate flow,
an admin panel, per-user feature access, private session workspaces and usage
metrics. Start with the download-free demo. Read the [platform guide](platform.md)
for account management, dataset allowlists, recovery and HTTPS deployment.

## Shared workflow improvements

Capture, routing, quantization and hardware evidence are displayed separately.
Support gaps are searchable, scenario coverage is explicit, and changed model
settings disable downstream actions until reinspection. Evaluation identifies
synthetic data versus approved labeled datasets. Users can export a private ZIP
if permitted and release their workspace to free server capacity.

## Legacy local mode

```bash
qbench-dashboard --single-user
```

This explicitly disables authentication and preserves the earlier full workbench,
including trusted Python factories, custom replacements and dataset path controls.
The launcher refuses non-loopback binding in this mode. Do not expose it through
a reverse proxy. Shared mode does not offer arbitrary-code or unrestricted-path
controls; it uses vetted model sources and operator-approved datasets.

## Legacy workbench workflow

1. Select a torchvision or timm model, device, weights, and capture shape.
2. Load and analyze the model. Expensive work is button-driven; model objects
   remain in Streamlit session state.
3. Read the scenario-qualified verdict, module summary, and unsupported gaps.
   Expand graph views only when you need operation-level detail.
4. Configure conversion and review the proposed replacements. Custom replacement
   targets are trusted Python code and must pass their validation step.
5. Convert and run sample inference. Use dataset benchmarking for classification
   accuracy when you have a compatible local validation dataset.
6. Download reports and recipes for reproducibility.

Model weights may download when requested. Dataset files are never bundled.
Use the built-in CLI provider from the [quickstart](quickstart.md) for a completely
download-free first check.

## Read the badges carefully

- A green scenario-qualified support verdict applies only to the paths actually
  captured and successfully routed through strict verification.
- Unassessed modules are shown separately; they are not silently certified.
- Missing hardware evidence is different from a failed simulator routing check.
- Explicit fallback is useful for investigation but cannot produce full support.

Changing device, shape, recipe, or weights invalidates assumptions behind earlier
results. Analyze again before interpreting the new conversion. For multiple
named scenarios and custom argument structures, use the [Python API](api.md).

!!! note "Pages hosts documentation, not Streamlit"
    GitHub Pages is static hosting. It publishes this guide, not a running model
    server. The dashboard runs locally or on your own GPU/CPU host.
