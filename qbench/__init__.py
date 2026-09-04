"""Canonical public API for QBench model inspection and simulation."""

from .conversion import Simulator, build_simulator
from .evaluation import EvaluationConfig, EvaluationReport, evaluate
from .inspection import inspect_model, inspect_provider
from .providers import (
    DirectObjectProvider,
    LegacyAdapterProvider,
    ModelProvider,
    TimmProvider,
    TorchvisionProvider,
)
from .provenance import package_version
from .registry import KernelSpec, OpRegistry
from .schemas import (
    InspectionConfig,
    InspectionResult,
    QuantizationPolicy,
    QBenchError,
    Scenario,
    SimulationPlan,
)

__version__ = package_version()

__all__ = [
    "inspect_model",
    "inspect_provider",
    "build_simulator",
    "evaluate",
    "Simulator",
    "InspectionConfig",
    "InspectionResult",
    "QuantizationPolicy",
    "SimulationPlan",
    "EvaluationConfig",
    "EvaluationReport",
    "Scenario",
    "QBenchError",
    "ModelProvider",
    "DirectObjectProvider",
    "TorchvisionProvider",
    "TimmProvider",
    "LegacyAdapterProvider",
    "KernelSpec",
    "OpRegistry",
    "__version__",
]
