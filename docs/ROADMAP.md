# Roadmap

- [x] Preserve legacy v0.1 source in a separate folder.
- [x] Build four new modular A/B/C/D architectures with structured and byte-language tasks.
- [x] Shared configs, deterministic synthetic splits, training, checkpoint and eval CLI.
- [x] Preliminary BP, truncated local, limited manual eligibility learner and isolated FF demo.
- [x] Smoke and unit tests including causality, dispatch and resume.
- [ ] Validate CUDA/BF16 on RTX 5060 Laptop 8 GB.
- [ ] Reproduce legacy v0.1 actual training results on matched benchmark splits.
- [ ] Match parameter budget and real FLOPs across A/B/C/D with a bounded search.
- [ ] Add delayed recall, document-level language holdout and variable-length memory tests.
- [ ] Add profiler for independent batch/inference latency and reserved GPU memory.
- [ ] Add an exact and fully trainable local eligibility implementation.
- [ ] Add incremental KV cache for A/B; fused/parallel SEC scan for C/D.
- [ ] Study dynamic computation budgets, component lifecycle, memory paging.

**Do not overclaim**: no result here yet proves a new general architecture surpasses existing model families. The point is an open experimental foundation for successive AI/researcher iterations.
