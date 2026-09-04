# Standalone model workbench

The dashboard is a client of the QBench runtime and compatibility workbench API.
It does not require the old experiment database, dashboard setup script, or any
other experiment tab.

## Launch

```bash
python -m pip install -e ".[dashboard]"
qbench-dashboard --server.address 127.0.0.1 --server.port 8501
```

Open the local URL printed by Streamlit. Use a trusted local machine: this is a
research workbench, not an authenticated multi-tenant service. Do not expose it
to the public internet without an appropriate authentication/reverse-proxy layer.

## Typical workflow

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
