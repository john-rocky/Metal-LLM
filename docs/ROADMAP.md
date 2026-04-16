# Roadmap — 12 weeks to 80–130 tok/s

Target: sustained decode of Gemma 4 E2B on iPhone 17 Pro, beating LiteRT-LM's public 56.5 tok/s number.

## Week 1–2 — llama.cpp baseline

Integrate llama.cpp via Swift Package (submodule or binary xcframework). Gemma 4 E2B runs on iPhone at ~38 tok/s, matching the existing llama.cpp measurement. Ship a minimal `MetalLLMEngine.generate` that round-trips prompt → tokens on-device. Baseline recorded in `docs/BENCHMARKS.md`.

## Week 3–4 — Gemma-specific kernel tuning

Gemma-specific optimisations inside the llama.cpp Metal kernels: KV-share across layers L15–L34 (the repeated-block exploit), sliding-window attention tuning for Gemma's 4K window. Target: **45 tok/s**.

## Week 5–6 — Speculative decoding stack

Wire MTP and EAGLE-3 drafters through llama.cpp's speculative API. Prompt-lookup drafter as the cheap fallback. Orchestrator picks per-step based on rolling acceptance rate. Target: **55–65 tok/s** — LiteRT parity.

## Week 7–8 — ANE prefill bridge

Reuse the chunk models from the sibling CoreML-LLM repo to offload prefill to the ANE while the GPU stays cool for decode. Hand the KV cache from Core ML to llama.cpp (IOSurface zero-copy where layouts match, staged copy otherwise). Target: **70–80 tok/s** with ANE prefill + Metal spec-decode.

## Week 9–10 — INT4 dequant audit

Verify llama.cpp's Metal kernels already dequant INT4 in-register (no FP16 round-trip through threadgroup memory). Measure effective bandwidth vs peak. If there is a gap — patch. LiteRT-LM pays a measurable FP16 upload tax here; we should not.

## Week 11–12 — A19 NAX kernels + Mirror SD

Custom Metal kernels that target A19's NAX tensor cores directly for the hot matmuls (QKV projection, FFN gate/up). Mirror-SD: GPU draft and ANE verify run in parallel so the drafter is free. Target: **90–120 tok/s** sustained, with a stretch goal of 130 tok/s burst.
