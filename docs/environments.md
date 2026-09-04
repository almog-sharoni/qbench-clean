# CPU & GPU environments

## CPU inspection

CPU is the default inspection device. Install stock CPU PyTorch and the base
QBench package to capture eager dispatcher operations and run quantization-disabled
routing verification. Vision libraries and Streamlit are optional. Neither CUDA
extension compilation nor an NVIDIA driver is needed for this workflow.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
qbench inspect qbench.examples:tiny_provider -o artifacts/cpu
```

Always use a compatible Python/PyTorch pair; consult the
[official installation guide](https://pytorch.org/get-started/locally/).

## Quantization-enabled GPU simulation

The extraction was validated in the existing NVIDIA PyTorch 25.09-based container
on an H100 NVL. The retained codec build targets SM89 and SM90. Other GPU
architectures are not validated by this extraction and may need maintained build
changes and their own conformance run.

A minimal reproducible container definition is included under
`containers/Apptainer.def`. The original multi-gigabyte sandbox is intentionally
not copied. Building a new image requires Apptainer, network access, appropriate
permissions, storage, and acceptance of upstream container terms.

```bash
apptainer build qbench.sif containers/Apptainer.def
apptainer exec --nv qbench.sif python -m pip install --user -e ".[dashboard,conformance,test]"
apptainer exec --nv qbench.sif python -m qbench inspect \
  qbench.examples:tiny_cuda_provider --device cuda --quantization-enabled \
  --output-dir artifacts/gpu
```

The definition matches the base used during validation, but a fresh image build
has not itself been certified by these tests. The host driver must support its
CUDA runtime. Dependency resolution may change; record the actual environment.

## Lazy extension build

The codec is compiled on first quantization-enabled use, not on base import.
Set `TORCH_EXTENSIONS_DIR` or `QBENCH_CUDA_BUILD_DIR` to a writable cache directory.
`QBENCH_CUDA_VERBOSE=1` enables compiler output for troubleshooting. Keep caches
out of the repository and allow enough space for compiler artifacts.

Do not infer quantization success from a CPU dry run, and do not infer hardware
fidelity from a successful simulator CUDA build. These are separate report axes.
