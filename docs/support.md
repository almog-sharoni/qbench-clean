# Support verdicts & runtime capture

## What full support means

QBench reports full support **for captured scenarios** only when all scenarios
complete, every meaningful executed operation is covered by a ready simulator
implementation or explicit structural capability, the strict converted dry run
succeeds, and no FP32 fallback remains. Disabling verification prevents that
guarantee. Opting into fallback always leaves the result partial.

## Capture is authoritative

Eager dispatch capture records every observed PyTorch dispatcher invocation:
namespace, schema, overload, sequence number, scenario, module stack, scalar
argument metadata, and tensor shape/dtype/device metadata. Temporary module
hooks assign ownership to the deepest active module. Calls bypassing normal
module entry can receive a Python callsite.

Exact standard modules with unchanged forwards can form semantic operations.
A custom subclass overriding `forward` is analyzed from its executed operations,
not accepted just because its class name resembles a supported module.

FX and `torch.export` are optional graph-enrichment tools. Their errors are
retained as diagnostics, while eager capture and its gaps remain authoritative.

## Aggregation does not erase execution

The raw compressed ledger retains repeated invocations. Default summaries group
by module path and semantic kernel, with counts and example scenarios.
Unsupported schemas have a dedicated gaps table. Union coverage combines the
named paths; per-scenario coverage keeps path-specific outcomes distinguishable.

Structural operations come from an explicit versioned capability list. An
unknown overload, custom namespace, unsupported dtype/shape/dimension, or
under-construction implementation is a gap, even if a related operation works.

## Limits

An unexecuted module is `not_assessed`, not supported or unsupported. Capture
cannot prove anything about unseen inputs, branches, Python-only behavior, or
external work opaque to the dispatcher. Scenario names should communicate the
conditions they exercise, such as `image-224` and `image-256`.

The initial guarantee is eager `nn.Module` inference. Training/backward, opaque
TorchScript, already-compiled models without an eager original, and distributed
or FSDP execution are outside the certified scope. Hardware fidelity and actual
quantized execution are separate axes; see [conformance](kernels.md).
