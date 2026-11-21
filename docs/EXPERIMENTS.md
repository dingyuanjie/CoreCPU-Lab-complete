# Reproducible experimental protocol

## Stage 0 (engineering)

```bash
python -m corecpu_lab selftest
python -m pytest
python legacy/v0.1/core_cpu.py selftest
```

Stage 0 does not establish any neural architecture advantage. Next, run 1–2 step smoke tests A/B/C/D for STRUCTURED and LANGUAGE. Generated language demo only tests code flow.

## Stage 1 (factorial)

Use four fixed seeds: 313, 101, 202, 303. Train STRUCTURED with matched train samples, splits and supervised losses. Use 6400+ examples per split for headline claims (e.g. `--eval-batches 50` and batch 128), after exploratory small runs. Record all failed runs, including OOM.

All four combinations must report the same split labels: `iid` for in-distribution; `pair` for unseen adjacent operator combinations; `long` for six-step extrapolation; `mod` for unseen modulus 5 and 7. Old v0.1 trained moduli 4/6/8 and lengths 1..3, so compare on same distribution. Accuracy on this small arithmetic task is not evidence of general reasoning ability.

## Stage 2 (statistical & efficiency)

Fix an outcome metric and define the 2x2 interaction in advance:

`I = performance(D) - performance(B) - performance(C) + performance(A)`

Use seed-paired experiments. Report both quality and resources, and include parameter-matched, approximate FLOP-matched and wall-clock-matched experiments. Do not assert experimental synergy from a single seed or from unmatched model resources.

Report parameter count, estimated storage, real peak memory when CUDA is used, latency, end-to-end throughput, total training time and failure cases. Note that current sparse expert dispatch has Python/control-flow overhead; actual speedup is never implied by `top_k/experts` alone.

## Stage 3 (language)

Train all four models on identical legally usable UTF-8 text, same byte vocabulary, same token budget and holdout policy. Report bits-per-byte and loss. Standardize length and warmups for speed comparisons, and measure inference, not just training. A/B are attention models while C/D are recurrent alternatives; optimization parity is nontrivial.

## Stage 4 (learning rules)

First compare BP vs LOCAL on structured tasks. Then test ELIG on SEC C/D as a limited manual-learning ablation; report exactly which parameters changed. A negative result is useful data. FF remains isolated until a protocol for end-to-end sequence modeling is explicitly designed.
