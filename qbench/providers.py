"""Trusted model-provider protocol and maintained provider implementations."""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, runtime_checkable

import torch.nn as nn

from .provenance import qualified_type
from .schemas import QBenchError, Scenario, redacted_exception


@runtime_checkable
class ModelProvider(Protocol):
    def build_model(self) -> nn.Module: ...
    def clone_model(self, model: nn.Module) -> nn.Module: ...
    def capture_scenarios(self) -> Iterable[Scenario]: ...
    def evaluation_loader(self) -> Iterable[Any]: ...
    def prepare_evaluation_batch(
        self, batch: Any
    ) -> Scenario | tuple[Scenario, Any]: ...
    def select_metric_output(self, output: Any) -> Any: ...


@dataclass
class DirectObjectProvider:
    model: nn.Module
    scenarios: Iterable[Scenario]
    loader: Iterable[Any] | None = None

    def __post_init__(self):
        self.scenarios = (
            (self.scenarios,)
            if isinstance(self.scenarios, Scenario)
            else tuple(self.scenarios)
        )

    def build_model(self):
        return self.model

    def clone_model(self, model):
        return copy.deepcopy(model)

    def capture_scenarios(self):
        return list(self.scenarios)

    def evaluation_loader(self):
        if self.loader is None:
            raise QBenchError("DirectObjectProvider has no evaluation loader")
        return self.loader

    def prepare_evaluation_batch(self, batch):
        if isinstance(batch, Scenario):
            return batch
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return Scenario("evaluation", (batch[0],), {}), batch[1]
        return Scenario("evaluation", (batch,), {})

    def select_metric_output(self, output):
        return output


@dataclass
class TorchvisionProvider:
    model_name: str
    scenarios: Iterable[Scenario]
    weights: Any = None
    loader: Iterable[Any] | None = None

    def __post_init__(self):
        self.scenarios = (
            (self.scenarios,)
            if isinstance(self.scenarios, Scenario)
            else tuple(self.scenarios)
        )

    def build_model(self):
        from torchvision import models

        return models.get_model(self.model_name, weights=self.weights)

    def clone_model(self, model):
        return copy.deepcopy(model)

    def capture_scenarios(self):
        return list(self.scenarios)

    def evaluation_loader(self):
        if self.loader is None:
            raise QBenchError("Configure an evaluation loader for TorchvisionProvider")
        return self.loader

    def prepare_evaluation_batch(self, batch):
        return Scenario("evaluation", (batch[0],), {}), batch[1]

    def select_metric_output(self, output):
        return output


@dataclass
class TimmProvider(TorchvisionProvider):
    def build_model(self):
        import timm

        return timm.create_model(self.model_name, pretrained=bool(self.weights))


class LegacyAdapterProvider:
    """Adapt a legacy object exposing model/sample/loader attributes."""

    def __init__(self, legacy: Any):
        self.legacy = legacy

    def build_model(self):
        value = getattr(self.legacy, "build_model", None)
        return value() if callable(value) else self.legacy.model

    def clone_model(self, model):
        value = getattr(self.legacy, "clone_model", None)
        return value(model) if callable(value) else copy.deepcopy(model)

    def capture_scenarios(self):
        value = getattr(self.legacy, "capture_scenarios", None)
        if callable(value):
            return value()
        sample = self.legacy.sample_input
        if isinstance(sample, Mapping):
            return [Scenario("sample", (), dict(sample))]
        if isinstance(sample, tuple):
            return [Scenario("sample", sample, {})]
        return [Scenario("sample", (sample,), {})]

    def evaluation_loader(self):
        value = getattr(self.legacy, "evaluation_loader", None)
        if callable(value):
            return value()
        if value is not None:
            return value
        loader = getattr(self.legacy, "loader", None)
        if loader is None:
            raise QBenchError("Legacy provider has no evaluation loader")
        return loader

    def prepare_evaluation_batch(self, batch):
        value = getattr(self.legacy, "prepare_evaluation_batch", None)
        if callable(value):
            return value(batch)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return Scenario("evaluation", (batch[0],), {}), batch[1]
        return Scenario("evaluation", (batch,), {})

    def select_metric_output(self, output):
        value = getattr(self.legacy, "select_metric_output", None)
        return value(output) if callable(value) else output


def provider_provenance(provider: Any) -> dict[str, Any]:
    """Return maintained, non-input provider identity fields."""

    result: dict[str, Any] = {"provider_type": qualified_type(provider)}
    if isinstance(provider, (TorchvisionProvider, TimmProvider)):
        result["model_name"] = provider.model_name
        weights = provider.weights
        if weights is None:
            result["weights"] = None
        elif isinstance(weights, (str, bool, int, float)):
            result["weights"] = weights
        else:
            name = getattr(weights, "name", None)
            result["weights"] = (
                str(name) if isinstance(name, str) else qualified_type(weights)
            )
    elif isinstance(provider, DirectObjectProvider):
        result["model_type"] = qualified_type(provider.model)
    elif isinstance(provider, LegacyAdapterProvider):
        result["legacy_type"] = qualified_type(provider.legacy)
    return result


def load_provider(spec: str) -> ModelProvider:
    if not isinstance(spec, str) or ":" not in spec:
        raise QBenchError("Provider must use package.module:object syntax")
    module_name, object_name = spec.split(":", 1)
    if not module_name or not object_name:
        raise QBenchError("Provider must use package.module:object syntax")
    try:
        value = importlib.import_module(module_name)
        for component in object_name.split("."):
            value = getattr(value, component)
        provider = (
            value()
            if isinstance(value, type)
            or (callable(value) and not isinstance(value, ModelProvider))
            else value
        )
    except Exception as exc:
        raise QBenchError(
            f"Could not load provider {spec!r}: {redacted_exception(exc)}"
        ) from exc
    required = (
        "build_model",
        "clone_model",
        "capture_scenarios",
        "evaluation_loader",
        "prepare_evaluation_batch",
        "select_metric_output",
    )
    missing = [name for name in required if not callable(getattr(provider, name, None))]
    if missing:
        raise QBenchError(f"Provider {spec!r} is missing: {', '.join(missing)}")
    return provider
