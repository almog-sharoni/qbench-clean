"""Clean-distribution independence and stateful-output regressions."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from qbench import DirectObjectProvider, Scenario, evaluate


def test_cpu_import_does_not_load_legacy_or_optional_dependencies():
    program = '''
import sys
import torch
from qbench import inspect_model, Scenario
result = inspect_model(torch.nn.Linear(4, 2).eval(), Scenario("cpu", (torch.ones(1, 4),)))
assert result.fully_supported
assert not any(name == "runspace" or name.startswith(("runspace.", "src.")) for name in sys.modules)
assert not any(name.startswith(("torchvision", "timm", "streamlit", "qbench.quantization.cuda")) for name in sys.modules)
'''
    subprocess.run([sys.executable, "-c", program], check=True)


def test_standalone_dashboard_starts():
    from streamlit.testing.v1 import AppTest
    from qbench import dashboard

    app = AppTest.from_file(str(Path(dashboard.__file__).with_name("app.py")))
    app.run(timeout=90)
    assert not app.exception
    assert app.title[0].value == "QBench"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA output queue")
def test_fast_metrics_snapshot_reused_cuda_output_buffers():
    class Reusable(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.register_buffer("output", torch.empty(1, 2, device="cuda"))
            self.scale = scale

        def forward(self, value):
            self.output.copy_(value * self.scale)
            return self.output

    class Runner:
        def __init__(self):
            self.model = Reusable(2)

        def run(self, invocation):
            return self.model(*invocation.args, **invocation.kwargs)

    reference = Reusable(1)
    inputs = [Scenario(str(i), (torch.full((1, 2), float(i), device="cuda"),)) for i in (1, 2)]
    provider = DirectObjectProvider(reference, inputs, loader=inputs)
    report = evaluate(reference, Runner(), provider)
    assert report.metrics["mae"] == pytest.approx(1.5)
    assert report.metrics["mse"] == pytest.approx(2.5)
