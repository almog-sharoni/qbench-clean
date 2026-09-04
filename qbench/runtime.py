"""Internal runtime evidence hooks shared with maintained simulator kernels."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable


_QUANTIZATION_OBSERVER: ContextVar[Callable[[dict[str, Any]], None] | None] = (
    ContextVar("qbench_quantization_observer", default=None)
)
_SIMULATOR_IMPLEMENTATION_DEPTH: ContextVar[int] = ContextVar(
    "qbench_simulator_implementation_depth", default=0
)
_SIMULATION_QUANTIZATION_ENABLED: ContextVar[bool] = ContextVar(
    "qbench_simulation_quantization_enabled", default=False
)
_SIMULATION_QUANTIZATION_POLICY: ContextVar[dict[str, Any] | None] = ContextVar(
    "qbench_simulation_quantization_policy", default=None
)
_SIMULATION_ROUTE_PATH: ContextVar[str] = ContextVar(
    "qbench_simulation_route_path", default=""
)
_QUANTIZATION_STAGE: ContextVar[dict[str, Any] | None] = ContextVar(
    "qbench_quantization_stage", default=None
)
_SIMULATION_INPUT_QUANTIZATION_ALLOWED: ContextVar[bool] = ContextVar(
    "qbench_simulation_input_quantization_allowed", default=True
)


@contextmanager
def observe_quantization(callback: Callable[[dict[str, Any]], None]):
    token = _QUANTIZATION_OBSERVER.set(callback)
    try:
        yield
    finally:
        _QUANTIZATION_OBSERVER.reset(token)


@contextmanager
def simulation_quantization(enabled: bool, policy: dict[str, Any] | None = None):
    """Expose the active plan's quantization mode to stateless handlers."""
    token = _SIMULATION_QUANTIZATION_ENABLED.set(bool(enabled))
    policy_token = _SIMULATION_QUANTIZATION_POLICY.set(
        None if policy is None else dict(policy)
    )
    try:
        yield
    finally:
        _SIMULATION_QUANTIZATION_POLICY.reset(policy_token)
        _SIMULATION_QUANTIZATION_ENABLED.reset(token)


def simulation_quantization_enabled() -> bool:
    return _SIMULATION_QUANTIZATION_ENABLED.get()


def simulation_quantization_policy() -> dict[str, Any]:
    return dict(_SIMULATION_QUANTIZATION_POLICY.get() or {})


@contextmanager
def simulation_route(path: str):
    """Expose the canonical owning module path to a transient runtime handler."""
    token = _SIMULATION_ROUTE_PATH.set(str(path))
    try:
        yield
    finally:
        _SIMULATION_ROUTE_PATH.reset(token)


def simulation_route_path() -> str:
    return _SIMULATION_ROUTE_PATH.get()


@contextmanager
def simulation_input_quantization(allowed: bool):
    """Apply the per-invocation first-semantic-route input policy."""

    token = _SIMULATION_INPUT_QUANTIZATION_ALLOWED.set(bool(allowed))
    try:
        yield
    finally:
        _SIMULATION_INPUT_QUANTIZATION_ALLOWED.reset(token)


def simulation_input_quantization_allowed() -> bool:
    return _SIMULATION_INPUT_QUANTIZATION_ALLOWED.get()


@contextmanager
def quantization_stage(stage: str, **policy: Any):
    """Tag primitive codec evidence with its requested semantic boundary."""

    inherited = dict(_QUANTIZATION_STAGE.get() or {})
    token = _QUANTIZATION_STAGE.set({**inherited, "stage": str(stage), **dict(policy)})
    try:
        yield
    finally:
        _QUANTIZATION_STAGE.reset(token)


def record_quantization(**metadata: Any) -> None:
    """Record one completed quantization primitive without retaining tensors."""
    observer = _QUANTIZATION_OBSERVER.get()
    if observer is not None:
        event = dict(metadata)
        for name, value in (_QUANTIZATION_STAGE.get() or {}).items():
            event.setdefault(name, value)
        observer(event)


@contextmanager
def simulator_implementation():
    """Mark low-level dispatcher work performed inside a simulator kernel."""
    token = _SIMULATOR_IMPLEMENTATION_DEPTH.set(
        _SIMULATOR_IMPLEMENTATION_DEPTH.get() + 1
    )
    try:
        yield
    finally:
        _SIMULATOR_IMPLEMENTATION_DEPTH.reset(token)


def simulator_implementation_active() -> bool:
    return _SIMULATOR_IMPLEMENTATION_DEPTH.get() > 0
