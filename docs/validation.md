# Validation evidence

This page distinguishes completed checks from acceptance work that still needs
independent baselines or external data. It is not a hardware-fidelity certificate.

## Source baseline

Before extraction, the current QBench API and workbench/dashboard suites passed
together: **249 passed, 1 skipped**, in 32.45 seconds on the available CUDA host.
The skipped test is the opt-in pretrained public-model ImageNet acceptance run.
The original checkout contained uncommitted implementation work; its parent commit
was `08b42206f899c8e4e1e5091b332118f7f0ccb21d`.

## Extraction-specific regressions

The initial standalone suite exposed a label-resource path pointing at the old
tree. The clean runtime now loads its packaged canonical ImageNet class index.
A new stateful CUDA-output regression also reproduced incorrect deferred metrics:
MAE was 2.0 instead of 1.5 when a model reused a buffer across batches. Private
final-output snapshots fix that aliasing without retaining intermediate activations.

Standalone tests check optional-dependency laziness, absence of `runspace`/`src`
imports, and real dashboard startup. The full validation summary is saved in
`validation/summary.json` in the repository and updated after final checks.

| Final gate | Result |
| --- | --- |
| Standalone CUDA source suite | 262 passed, 1 opt-in skip |
| Installed-wheel stock-CPU suite | 256 passed, 7 GPU/opt-in skips |
| GPU simulator conformance | 55 passed, 0 failed; hardware evidence missing |
| Wheel/API/CLI isolation checks | Passed outside the checkout |
| Documentation | Strict build and all local links/anchors passed |

## Tested environment

| Component | Version/device |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.9.0a0+50eac811a6.nv25.09 |
| torchvision | 0.24.0a0+98f8b375 |
| timm | 1.0.26 |
| Streamlit | 1.55.0 |
| CUDA runtime | 13.0 |
| GPU | NVIDIA H100 NVL |

The installed wheel was also exercised in a separate stock-CPU virtual environment
with PyTorch `2.14.0+cpu`, torchvision `0.29.0+cpu`, timm `1.0.29`, and Streamlit
`1.63.0`. This environment has no CUDA PyTorch runtime. The export-specific timm
replacement test explicitly forces graph-enrichment fallback, rather than relying
on older dependency versions to fail FX incidentally.

Tests in this container do not prove all declared dependency versions work. CPU
CI and wheel-isolation checks are additional gates; see their recorded results.

## Before/after extraction matrix

The deterministic, random-weight comparison uses eager capture with FX/export
disabled, seed 42, batch size 1, and quantization disabled. All six source-output
hashes, ledgers/mappings, operation counts, and support verdicts are unchanged.

| Model | Shape | Dispatcher operations | Strict verification |
| --- | --- | --- | --- |
| ResNet-18 | 224 | 69 | Passed; identical predictions |
| ResNet-18 | 256 | 69 | Passed; identical predictions |
| ViT-B/16 | 224 | 142 | Passed; identical predictions |
| ViT-B/16 | 256 | 142 | Passed; identical predictions |
| MobileViT-S | 224 | 298 | Partial in both source and clean trees |
| MobileViT-S | 256 | 296 | Passed; identical predictions |

The five strict simulator outputs also have identical byte hashes before/after
extraction and match their reference outputs at `rtol=1e-5`, `atol=1e-6`.
Machine-readable results are in `validation/extraction-comparison.json` in the
repository. This is an extraction regression check, not pretrained accuracy.

## Reproduce public-model acceptance

```bash
python -m qbench.validation.public_models --help
QBENCH_RUN_PUBLIC_MODEL_ACCEPTANCE=1 python -m pytest \
  tests/test_public_model_acceptance.py -q
```

The harness supports ResNet-18, ViT-B/16, and MobileViT-S; 224/256 capture cases;
deterministic ImageNet subset manifests; source-to-simulator mapping comparisons;
and separate strict versus legacy-compatible evaluation phases. Consult `--help`
and the opt-in test for required dataset/configuration environment variables.

## Outstanding acceptance gates

- A true pre-refactor baseline and complete six-case pretrained ImageNet
  support/accuracy diff are not established by the extraction regression suite.
- The at-most-10% fast-metric overhead target must be remeasured on representative
  target-GPU workloads after the output-snapshot fix. Earlier reduced measurements
  are not a certification of the changed implementation.
- Included simulator vectors are not independent hardware captures. Hardware
  fidelity remains missing without authenticated external evidence.
- A newly rebuilt container and a broad Python/PyTorch matrix need separate checks.

Random inputs, random weights, and a green support badge are insufficient evidence
for unchanged top-1/top-5 accuracy on real ImageNet data.
