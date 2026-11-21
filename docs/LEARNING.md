# Learning mechanisms

| name | exact implemented meaning | task / compute | uses autograd? |
|---|---|---|---|
| `bp` | full supervised gradient, AdamW, gradient clipping | all 8 task-architecture combinations | yes |
| `local` | step-level supervised cross entropy; detach categorical state and SEC runtime state between instructions; gradients still exist *within* a step | structured A/B/C/D | yes, local/truncated only |
| `elig` | classifier head is updated with hand-computed cross-entropy gradient; SEC `out` uses scalar centered reward times per-sample activity eligibility; remaining parameters fixed | structured C/D | **no** |
| `ff-demo` | standalone positive/negative local goodness training | isolated toy 8D task | local backprop only |

Do not call this `elig` variant exact e-prop or full end-to-end non-gradient training. It intentionally has large fixed portions, intended to test feasibility and failure modes with honest baselines. Code caches eligibility-like activations only in the manual learner; no unbounded history is retained.

Future: add exact local eligibility propagation for each SEC parameter and router credit assignment, compare BP, truncated BP, and the fully local rule at matched training budgets. Add separate tests for long-term retention and continual learning, which are not part of v1.0.
