# Know what your model actually executes.

<div class="hero" markdown>

**QBench** turns eager PyTorch inference into an auditable support report and a
verified simulator plan. Capture real execution paths, find capability gaps,
and compare reference and simulated outputs with one public API.

[Start on CPU](quickstart.md){ .md-button .md-button--primary }
[Open the workbench](dashboard.md){ .md-button }

</div>

<div class="grid cards" markdown>

- **Capture execution**

    Record exact dispatcher schemas, overloads, module ownership, and tensor
    metadata across named scenarios. FX/export enrich the graph, not the verdict.

    [Understand capture](support.md)

- **Verify routing**

    Build a strict clone and audit source-to-simulator mappings. Unsupported
    operations stay visible; explicit FP32 fallback always means partial support.

    [Build a simulator](simulation.md)

- **Measure differences**

    Run paired evaluation for output errors, agreement, task metrics, and coarse
    timing. Enable expensive per-layer diagnostics only when you need them.

    [Evaluate a model](evaluation.md)

- **Keep evidence**

    Save a checksummed manifest, support report, raw compressed operation ledger,
    and simulation plan. Inspect portable kernel conformance independently.

    [Read the artifact format](artifacts.md)

</div>

## Four different questions—not one badge

| Question | Evidence |
| --- | --- |
| Did the requested paths run? | Per-scenario capture completion |
| Did supported operations reach simulator kernels? | Replacement coverage and strict verification |
| Was quantization actually exercised? | Quantization-enabled execution verification |
| Does the simulator match hardware? | Independently sourced conformance evidence |

!!! warning "A qualified guarantee"
    Fully supported means **for the captured scenarios**. It does not certify
    unseen branches, training/backward, or hardware bit-exactness. CPU dry runs
    verify routing with quantization disabled. See [scope](support.md).

## A deliberately small repository

The clean distribution includes the Python API, CLI, simulator implementations,
standalone Streamlit workbench, tests, and this documentation. Experiments,
databases, model weights, datasets, and container filesystems are excluded.

Read the [validation record](validation.md) for tested behavior and open acceptance
gates. Existing research scripts remain in the original repository.
