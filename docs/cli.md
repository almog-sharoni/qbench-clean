# Command-line reference

All provider commands accept a trusted `package.module:object`, either as the
positional argument or with `--provider`. Use `--help` on the installed command
for the complete, version-specific option list.

## Inspect

```bash
qbench inspect qbench.examples:tiny_provider --output-dir artifacts/inspect
qbench inspect qbench.examples:tiny_provider --no-fx --no-export -o artifacts/eager
qbench inspect qbench.examples:gap_provider -o artifacts/gaps
```

The last command intentionally returns `2` with a gap for `sin`. Graph enrichment
can be disabled without disabling runtime capture.

## Convert

```bash
qbench convert qbench.examples:tiny_provider --export-state -o artifacts/convert
```

Writes a verified plan and optionally `state.pt`. It is not a TorchScript/export
compiler; rebuild the simulator with its provider and plan for execution.

## Evaluate

```bash
qbench evaluate qbench.examples:tiny_provider --max-batches 2 -o artifacts/evaluate
qbench evaluate qbench.examples:tiny_provider --metrics detailed \
  --latency-repetitions 3 --retain-activations -o artifacts/detailed
```

Select `--task classification`, `language-modeling`, `feature-matching`, or
`generic` to match your provider. `--evaluation-config FILE` accepts the
evaluation configuration as JSON.

## Shared inspection options

| Option | Meaning |
| --- | --- |
| `--config FILE` | JSON InspectionConfig |
| `--output-dir DIRECTORY`, `-o` | Output artifacts; default `qbench-artifacts` |
| `--allow-fp32-fallback` | Explicitly partial conversion |
| `--verify` / `--no-verify` | Enable/disable dry-run verification |
| `--enable-fx` / `--no-fx` | Optional FX enrichment |
| `--enable-export` / `--no-export` | Optional export enrichment |
| `--device cpu` / `--device cuda` | Execution device |
| `--quantization-enabled` | Request actual quantization |
| `--quantization-type FORMAT` | Quantization format such as `fp8_e4m3` |
| `--conformance-vectors DIRECTORY` | Conformance evidence to evaluate |

Format bias, stage controls, quantization modes, and chunk sizes have explicit
flags as listed by `qbench inspect --help`. CLI values override config-file values.

## Kernel catalog and verification

```bash
qbench kernels list -o artifacts/kernel-catalog.json
qbench kernels verify qbench/conformance_vectors -o artifacts/conformance.json
```

The vector path above is relative to a source checkout. For installed packages,
resolve it using `importlib.resources.files("qbench.conformance_vectors")` in Python.
Quantization-enabled vectors need a compatible GPU; CPU unavailability is not
a successful conformance result.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Fully supported workflow, or successful kernel command |
| `2` | Successfully analyzed but partial/unsupported, or non-passing conformance |
| `1` | Configuration or execution failure |

Do not discard artifacts just because a command returns `2`. They explain the
support gaps and are often the most useful output of the inspection.
