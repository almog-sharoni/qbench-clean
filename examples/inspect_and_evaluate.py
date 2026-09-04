"""Run from anywhere after installing QBench; writes only the chosen artifact directory."""
import argparse

from qbench import build_simulator, evaluate, inspect_provider
from qbench.artifacts import write_artifacts
from qbench.examples import tiny_provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/example")
    args = parser.parse_args()
    provider = tiny_provider()
    result = inspect_provider(provider)
    model = provider.build_model()
    simulator = build_simulator(model, result.plan)
    report = evaluate(model, simulator, provider)
    write_artifacts(args.output, result, evaluation=report)
    print(f"Fully supported for captured scenarios: {result.fully_supported}")
    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()
