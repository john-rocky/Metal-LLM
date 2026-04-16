# Benchmarks

**Status: placeholder.** Populated as each milestone lands.

Planned table shape:

| Milestone | Prefill tok/s | Decode tok/s (burst) | Decode tok/s (60s sustained) | Thermal |
|-----------|--------------:|---------------------:|-----------------------------:|---------|
| llama.cpp reference | — | — | — | — |
| Week 2 (baseline)   | — | — | — | — |
| Week 4 (Gemma tune) | — | — | — | — |
| Week 6 (spec decode)| — | — | — | — |
| Week 8 (ANE prefill)| — | — | — | — |
| Week 12 (NAX)       | — | — | — | — |

Device: iPhone 17 Pro, iOS 18+, Gemma 4 E2B (INT4 GGUF).

Reference point: LiteRT-LM reports **56.5 tok/s** on the same hardware / model.
