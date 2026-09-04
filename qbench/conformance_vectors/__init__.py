"""Packaged deterministic conformance vectors for maintained QBench kernels."""

from importlib.resources import files


def resources():
    """Return the installed conformance-corpus resource directory."""
    return files(__package__)


__all__ = ["resources"]
