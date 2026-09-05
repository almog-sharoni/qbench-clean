# Troubleshooting

## CPU inspection tries to compile CUDA

Base import and quantization-disabled CPU inspection should not load the codec.
Check that `quantization_enabled` is false and that your application did not
explicitly import `qbench.quantization.cuda`. Run the standalone import regression
test and report the import traceback if the issue persists.

## FX or export failed

Read the diagnostics, but use the eager capture result for support. Try
`--no-fx --no-export` to inspect without enrichment. A dynamic model is not fully
supported merely because its modules have familiar class names.

## Exit code 2 / unsupported schema

Open `support.json` and inspect the exact schema, overload, and constraint reason.
Supply additional named scenarios for new branches. Explicit fallback can help
exploration, but leaves the verdict partial. Do not edit the plan to hide gaps.

## Strict execution rejects a new input

The new shape, dtype, kwargs, or branch may differ from captured scenarios.
Inspect all intended paths again. Execute with `Simulator.run`, not directly on
the converted model. Confirm model state and quantization policy match the plan.

## Torchvision import or operator registration fails

Install matching PyTorch and torchvision builds for your CPU/CUDA environment.
Do not mix incompatible wheel channels. The base API and the bundled tiny
provider do not require torchvision or timm.

## A pretrained model needs network access

Downloading weights is controlled by the provider/model-library configuration.
Use `weights=None` or the tiny provider for offline smoke checks; supply a trusted
local cache for pretrained acceptance. Random weights cannot establish accuracy.

## Model cloning fails

Implement the provider's `clone_model` using a fresh constructor and independent
state loading. Do not return the same model or share mutable buffers. Initial
support guarantees do not include opaque compiled or distributed wrappers.

## A legacy child replacement is hidden by an ancestor

Legacy recipe validation rejects replacements underneath an expanded, aliased,
or FP32 ancestor when that ancestor could overwrite or hide the selected child.
Graph-enrichment recommendations can differ between dependency versions. Resolve
the ancestor decisions before conversion; do not suppress this safety check.
The canonical scenario-based API is preferred for new integrations.

## CUDA build fails

Check GPU architecture, host driver, toolkit/compiler compatibility, writable
extension cache, and free disk space. The shipped code generation targets Ada
SM89 and Hopper SM90. See [environments](environments.md). Do not treat a build
failure as a passed quantization or hardware-fidelity check.

## Docs show a missing Pages site

An HTTP 404 from `configure-pages` means the site is not enabled or is unavailable
to the workflow token. Documentation builds no longer require a Pages site. For
publication, Pages must use GitHub Actions and `QBENCH_PAGES_ENABLED` must be `true`.
Check private-repository plan eligibility and intended website visibility before
enabling it. Do not auto-enable public hosting just to make a build pass.
A local `mkdocs build` or a skipped deployment only verifies static output.
See [deployment](deployment.md).
