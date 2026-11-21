# SEC: State Evolution Cell

Per token / instruction input `x_t [B,D]`; state `S_{t-1} [B,K,D]`.

```text
x_t → in_proj → u
S_prev → mean → h
concat(u,h) → sigmoid gate g [B,K,D]
u → candidate c [B,K,D]
c, S_prev, decay → tanh candidate
S_new = (1-g) * S_prev + g * candidate
S_new.mean(slots) → LayerNorm → out → delta
```

`SequentialRuntime` maintains one SEC per depth, with its independent state. `CoreCPURuntime` maintains state for each shared expert. Both return residual-updated hidden activations.

SEC does not use sequence self-attention. It still uses dense linear algebra and the ordinary BP learner by default. It can have slower wall-clock performance than optimized attention due to explicit recurrent loops and Python dispatch. Fixed slot capacity may lose information on long sequences. Its state is not a semantic database, and `num_slots` is not a direct measure of effective memory capacity.

## Discriminating experiments

- Length sweep: 8/16/32/64/128/256 tokens, under equal quality and memory limits.
- State capacity sweep: K=1/2/4/8 slots.
- Multiple initializations and seed means/standard deviations.
- Retention challenge: delayed-copy and associative recall. Not yet implemented; do not claim their result.
- With/without gate, learned decay and layerwise state ablations.
- Real GPU memory and tokens/second, including peak-reserved and peak-allocated.

The current implementation is the smallest causal, trainable SEC prototype. It does not yet implement disk paging, parallel scan, learned component life cycle or large-scale language pretraining.
