"""Small trusted providers; no downloads, external files, or global RNG changes."""
import torch
from torch import nn

from .providers import DirectObjectProvider
from .schemas import Scenario


def tiny_provider():
    """Deterministic CPU model suitable for CLI and dashboard smoke checks."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).eval()
        inputs = [torch.randn(batch, 4) for batch in (1, 2)]
    scenarios = [Scenario(f"batch-{value.shape[0]}", (value,)) for value in inputs]
    return DirectObjectProvider(model, scenarios, loader=scenarios)


def gap_provider():
    """An unsupported operation for demonstrating partial support artifacts."""
    class Sine(nn.Module):
        def forward(self, value):
            return value.sin()

    scenario = Scenario("sine", (torch.ones(1, 4),))
    return DirectObjectProvider(Sine().eval(), [scenario], loader=[scenario])


def tiny_cuda_provider():
    """The same deterministic example with explicit CUDA model/input placement."""
    provider = tiny_provider()
    model = provider.build_model().to("cuda")
    scenarios = [Scenario(s.name, tuple(value.to("cuda") for value in s.args), s.kwargs)
                 for s in provider.capture_scenarios()]
    return DirectObjectProvider(model, scenarios, loader=scenarios)
