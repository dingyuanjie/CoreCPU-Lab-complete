# Architecture: two orthogonal research factors

The experiment design is **organization** x **compute** (not a guarantee that all models have matched parameters or FLOPs):

```text
                         COMPUTE
                   conventional        SEC
ORGANIZATION  sequential      A              C
              corecpu         B              D
```

## Organization

SequentialRuntime allocates an independent cell per depth. In structured tasks, hidden computation is repeated per instruction; categorical output state is passed to the next instruction. In SEC, every depth has its own persistent state slots. In language A, each depth has separate causal attention and FFN.

CoreCPURuntime has E shared expert cells and a differentiable top-k router reused across depth and successive task steps. Only selected experts run, with per-expert state stored across time for SEC. Entire expert weights stay in memory. This is not disk offloading, autonomous component discovery, or optimized CUDA dispatch. For language B, a shared causal attention block and sparse FFN are reused across depth; for language D, a shared SEC expert registry runs sequentially across tokens.

## Compute

Conventional compute: MLP cell for structured execution; causal attention+FFN for language.

SEC compute: an input projection, a featurewise gate over S state slots, a candidate update, learnable decay, and a slot-pooled output projection. State has shape `[batch, slots, hidden_dim]`. C/D language models have no full-sequence self-attention. SEC is inspired by prior recurrent/memory designs; novelty is a research question, not a proved fact.

## Task boundaries

STRUCTURED: ExternalReader converts a per-sample key/value list into a differentiable probability distribution over 8 possible values; then each instruction recursively consumes a predicted state. Ground-truth intermediate labels, correct address locations and final labels are withheld from `forward` and only enter the loss. The same reader is used across all A-D (a controlled memory-reading prior).

LANGUAGE: byte vocabulary 257 (0..255 byte plus BOS 256). A uses independent causal attention blocks, B uses tied/shared attention and MoE, C/D scan bytes sequentially using SEC. All output the same next-byte predictions. C/D support passing state between chunks and are tested for causal consistency. A/B do not implement incremental KV cache.

## Interface contracts

- `ModelConfig` validates architecture and dimensions.
- `build_model(config)` constructs either task-specific model.
- `model.forward(...) -> ModelOutput` with `logits [B,T,V]`, optional `features [B,T,D]`, `auxiliary_loss` and diagnostics.
- `runtime.step(x, state) -> hidden, new_state, balance, counts, route_logits`.
- `learner.train_step(batch)` is independent of model construction.
- Model checkpoint contains configuration, model parameters, learner state, data RNG, torch RNG, and text SHA256.

## Correctness invariants

- Never pass `labels` or `qpos` to model forward.
- Residual and SEC state shapes must match.
- For language models, no future token may affect earlier logits.
- Sparse experts not selected must not execute or receive parameter gradients.
- For a fixed RNG, each model receives the same evaluation splits.
- Report real latency and memory instead of equating top-k sparsity to an overall compute reduction.
