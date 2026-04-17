# Benchmarks

## iPhone 17 Pro (A19 Pro, iOS 26)

Model: Gemma 4 E2B Q4_K_M (2.89 GiB)

| Test | tok/s | Date | Notes |
|------|------:|------|-------|
| llama-bench greedy tg128 | 38.0 | 2026-04-13 | reference (prior measurement) |
| **Metal-LLM Engine greedy** | **35** | **2026-04-17** | **HUD measured, real chat** |
| LiteRT-LM (Google) | 56.5 | 2026-04 | Google公称, MTP込み |

Hardware:
- `has tensor = true` (NAX tensor API available)
- `use fusion = true`
- Memory: 2948 / 8192 MiB (36%)
- All 36 layers on GPU

## Mac Studio M4 Max

Model: Gemma 4 E2B Q4_K_M (2.88 GiB)

| Test | tok/s | Date | Notes |
|------|------:|------|-------|
| llama-bench tg128 | 144 | 2026-04-16 | reference |
| llama-bench tg256 | 49 | 2026-04-16 | longer seq = slower (full-attn layers) |
| Engine greedy tg128 | 147 | 2026-04-16 | 4% overhead vs llama-bench |
| Raw C API tg128 | 153 | 2026-04-16 | direct llama.cpp calls |
| Custom kernel 1-layer | 0.350 ms | 2026-04-16 | Q4_K real weights |

### Speculative decode attempts (Mac Studio)

| Drafter | tok/s | vs greedy | Acceptance |
|---------|------:|-----------|------------|
| N-gram (prompt lookup) | 47 | -31% | 0.9% |
| Gemma 3 1B (cross-arch) | 34 | -31% | low |
| IQ2_M (same model 2-bit) | 22 | -55% | low, draft too slow |
| Q3_K_S (same model 3-bit) | 20 | -59% | low, draft too slow |

Root cause: Gemma 4 E2B is 4.6B params — no smaller same-arch variant exists.
IQ2_M runs at 90% of Q4_K_M speed (117 vs 130 tok/s) — not enough speed gap for spec decode.
Google's MTP uses a 48M param drafter (100x smaller, 5x faster) — requires training.

## Target

| Milestone | tok/s | How |
|-----------|------:|-----|
| Current | 35 | greedy decode |
| Beat LiteRT-LM | 57+ | MTP drafter training |
| Stretch | 100+ | MTP + NAX optimization |
