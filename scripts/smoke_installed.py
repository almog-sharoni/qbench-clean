"""Verify an installed wheel from outside the checkout, with no model downloads."""
import gzip
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import qbench
from qbench import build_simulator, evaluate, inspect_provider
from qbench.examples import tiny_provider


def main():
    assert "site-packages" in qbench.__file__, qbench.__file__
    assert not any(name.startswith(("runspace", "src.")) for name in sys.modules)
    assert not any(name.startswith("qbench.quantization.cuda") for name in sys.modules)
    provider = tiny_provider()
    result = inspect_provider(provider)
    assert result.fully_supported
    model = provider.build_model()
    simulator = build_simulator(model, result.plan)
    report = evaluate(model, simulator, provider)
    assert report.metrics["mae"] < 1e-6
    assert report.reference_forwards == report.simulator_forwards == 2
    assert len(json.loads(files("qbench.data").joinpath("imagenet_class_index.json").read_text())) == 1000
    for name in ("ops_host.cpp", "ops_tensor.cu", "codec.cuh", "formats.def"):
        assert files("qbench.quantization").joinpath("cuda", name).is_file()
    assert files("qbench.dashboard").joinpath("app.py").is_file()
    assert files("qbench.conformance_vectors").joinpath("manifest.json").is_file()
    with tempfile.TemporaryDirectory(prefix="qbench-wheel-smoke-") as directory:
        for command, expected in [
            (["inspect", "qbench.examples:tiny_provider"], 0),
            (["convert", "qbench.examples:tiny_provider", "--export-state"], 0),
            (["evaluate", "qbench.examples:tiny_provider"], 0),
            (["inspect", "qbench.examples:gap_provider"], 2),
            (["convert", "qbench.examples:gap_provider", "--allow-fp32-fallback"], 2),
        ]:
            completed = subprocess.run(
                [sys.executable, "-m", "qbench", *command, "-o", directory],
                capture_output=True, text=True,
            )
            assert completed.returncode == expected, completed.stderr + completed.stdout
            for artifact in ("manifest.json", "support.json", "plan.json"):
                assert json.loads((Path(directory) / artifact).read_text())
            with gzip.open(Path(directory) / "operations.jsonl.gz", "rt") as ledger:
                assert all(json.loads(line) for line in ledger)
        bad = subprocess.run([sys.executable, "-m", "qbench", "invalid"], capture_output=True)
        assert bad.returncode == 1
    print(json.dumps({"status": "passed", "package": qbench.__file__,
                      "torch": torch.__version__, "cuda_build": torch.version.cuda,
                      "cli_exit_codes": [0, 2, 1], "paired_forwards": [2, 2]}))


if __name__ == "__main__":
    main()
