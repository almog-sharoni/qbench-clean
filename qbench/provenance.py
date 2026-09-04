"""Privacy-safe, reproducible identities for QBench artifacts."""

from __future__ import annotations

import hashlib
import platform
from importlib import metadata
from typing import Any

import torch
import torch.nn as nn


def package_version() -> str:
    try:
        return metadata.version("qbench")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def qualified_type(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach()
    # Meta tensors have no storage to copy.  Shape, dtype, layout, name, and
    # quantization metadata are already fed into the surrounding state digest,
    # so an empty value payload is the deterministic representation here.
    if value.device.type == "meta":
        return b""
    value = value.cpu()
    if value.is_quantized:
        value = value.int_repr()
    if value.layout != torch.strided:
        value = value.to_dense()
    value = value.contiguous()
    raw = value.view(torch.uint8).reshape(-1)
    try:
        return raw.numpy().tobytes(order="C")
    except Exception:
        # NumPy is optional.  Clone the logical uint8 view so its storage is
        # exactly the tensor payload (no storage offset or unrelated capacity),
        # then copy those bytes directly.  Unlike torch.save, this encoding has
        # no process-specific storage identifiers, so equal values in distinct
        # allocations hash identically.
        isolated = raw.clone(memory_format=torch.contiguous_format)
        return bytes(isolated.untyped_storage())


def model_state_sha256(model: nn.Module) -> str:
    """Hash model state without returning or retaining any state values."""

    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if torch.is_tensor(value):
            try:
                shape = tuple(int(dim) for dim in value.shape)
            except RuntimeError:
                digest.update(qualified_type(value).encode("utf-8"))
                digest.update(b"\0")
                continue
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(value.layout).encode("ascii"))
            digest.update(repr(shape).encode("ascii"))
            if value.is_quantized:
                digest.update(str(value.qscheme()).encode("ascii"))
                if value.qscheme() in {
                    torch.per_tensor_affine,
                    torch.per_tensor_symmetric,
                }:
                    digest.update(float(value.q_scale()).hex().encode("ascii"))
                    digest.update(str(int(value.q_zero_point())).encode("ascii"))
                else:
                    digest.update(str(int(value.q_per_channel_axis())).encode("ascii"))
                    digest.update(_tensor_bytes(value.q_per_channel_scales()))
                    digest.update(_tensor_bytes(value.q_per_channel_zero_points()))
            digest.update(_tensor_bytes(value))
        else:
            digest.update(qualified_type(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def model_provenance(model: nn.Module) -> dict[str, Any]:
    def count(values) -> int | None:
        try:
            return sum(int(value.numel()) for value in values)
        except (RuntimeError, TypeError):
            return None

    parameters = count(model.parameters())
    buffers = count(model.buffers())
    return {
        "model_type": qualified_type(model),
        "model_state_sha256": model_state_sha256(model),
        "parameter_count": parameters,
        "buffer_count": buffers,
        "qbench_version": package_version(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
    }


__all__ = [
    "model_provenance",
    "model_state_sha256",
    "package_version",
    "qualified_type",
]
