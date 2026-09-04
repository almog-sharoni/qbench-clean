# Kernels & conformance

## Maintained capability records

Support comes from `KernelSpec` records in the canonical registry—not a naming
convention. Each record declares exact module/ATen matching, argument constraints,
implementation or handler, conversion strategy, structural/composite classification,
whether it counts as quantized execution, and conformance metadata.

Overloads are distinct. A ready implementation for one dimension or dtype does
not certify the others. Custom namespaces and unknown overloads remain explicit
gaps until a maintained handler and constraints exist.

```bash
qbench kernels list
```

The generated catalog is the source of truth for the installed version. Avoid
maintaining an independent hard-coded support table in downstream applications.

## Portable vectors

A bundle consists of `manifest.json` and `.npz` files containing tensors or raw
bit patterns, along with checksums. Vector metadata identifies the kernel,
arguments, configuration, expected outputs, and comparison rule. Rules include
bit-exact, ULP-bounded, and explicit absolute/relative tolerances.

```python
from importlib.resources import files
from qbench.kernels import verify_kernels

report = verify_kernels(files("qbench.conformance_vectors"))
print(report["status"])
```

The included vectors exercise simulator behavior. They are **not independent
hardware captures**. Their successful execution cannot manufacture hardware
fidelity evidence. Missing hardware evidence affects that badge; failed evidence
marks fidelity failed. Quantization-enabled vectors unavailable on CPU must be
reported as unavailable, not passed.

## Adding a maintained kernel

1. Translate the agreed kernel-team behavior into a simulator implementation.
2. Register exact matchers and conservative constraints in `KernelSpec`.
3. Add its runtime handler/conversion strategy and composite boundary when needed.
4. Test eager capture, routing, error paths, and quantization-stage evidence.
5. Add checksummed vectors and explicit comparison criteria with provenance.
6. Run CPU regression tests and GPU conformance; review support/accuracy changes.

Never loosen constraints to make a support badge green without implementing and
verifying the corresponding behavior. Updating vector expected outputs requires
review of the changed numerical contract, not just regenerating files.
