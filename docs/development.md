# Architecture & contributing

## Runtime pipeline

```text
ModelProvider + named Scenarios
          |
     eager capture --------> optional FX/export diagnostics
          |
 semantic normalization + maintained capability resolution
          |
     SimulationPlan
          |
 strict clone + runtime handlers + routing audit
          |
 support artifacts / paired evaluation / standalone dashboard
```

| Module | Responsibility |
| --- | --- |
| `capture.py` | Dispatcher ledger, input cloning, module ownership |
| `inspection.py` | Scenario capture, normalization, coverage and automatic verification |
| `registry.py` | Exact capability records and canonical operation registry |
| `conversion.py`, `runtime.py` | Clone conversion, handlers, composite boundaries and audit |
| `schemas.py` | Public configuration/result schemas and strict serialization |
| `providers.py` | Trusted provider protocol and maintained implementations |
| `evaluation.py` | Paired metrics and opt-in detailed diagnostics |
| `artifacts.py`, `provenance.py` | Reproducible outputs and provenance |
| `kernels.py`, `conformance_vectors/` | Portable simulator/evidence checks |
| `dashboard/platform/` | Private accounts, authorization, usage metrics and public-API workbench |
| `dashboard/workbench.py` | Explicit loopback-only legacy UI |

The retained workbench module supports old recipe/report workflows; new programmatic
integration should use the public API. Do not add simulator logic to the dashboard.

## Local checks

```bash
python -m pip install -e ".[dashboard,conformance,test]"
python -m pytest -q
python -m pip install build
python -m build --wheel
python -m pip install -r docs/requirements.txt
mkdocs build --strict
```

The platform tests cover persistent authentication throttling, session revocation,
role/feature enforcement, cross-session workspace isolation, concurrent account
edits, approved dataset evaluation, and Streamlit interactions. The GPU suite
also exercises a quantization-enabled platform workflow.

For a real Chromium check using a disposable private database and loopback server:

```bash
python -m pip install -e ".[dashboard,browser-test]"
python -m playwright install --with-deps chromium
python scripts/smoke_platform_browser.py
```

This creates test accounts only in a temporary directory and writes screenshots,
an artifact ZIP, and a result summary under `artifacts/platform-browser/`. It checks
admin account creation, first-login password change, model actions, downloads,
revocation, and admin usage metrics in isolated browser contexts. CI runs it too.

CPU CI runs the full portable suite; GPU-only tests skip explicitly. GPU conformance
and pretrained ImageNet acceptance require a compatible machine and data. A passing
CPU workflow must not be presented as GPU/hardware certification.

## Change checklist

- Add conservative exact-schema/constraint tests for every new capability.
- Test capture completeness even when FX and export both fail.
- Check arbitrary args/kwargs, output structures, branches and ownership.
- Audit strict routing, composite boundaries, fallback and quantization evidence.
- Verify input/model/RNG isolation and reused output buffers.
- Keep import and installed-wheel tests independent of the original repository.
- Run dashboard tests when changing workbench report or recipe contracts.
- Update conformance provenance and documentation when numerical behavior changes.

Do not include datasets, weights, tokens, activation dumps, or experiment outputs
in commits. Review `NOTICE.md` in the repository before
redistributing or assigning a new project license.
