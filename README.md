# Metal-LLM — Fastest iPhone LLM Decode

**Beat Google LiteRT-LM's 56.5 tok/s** — Gemma 4 E2B on iPhone 17 Pro, Metal GPU + ANE hybrid, speculative decoding native.

Metal-LLM is a Swift engine built on top of llama.cpp's Metal kernels, extended with a speculative decoding stack and an ANE bridge that LiteRT-LM does not ship. The target is 80–130 tok/s sustained decode on iPhone 17 Pro, in the regime where LiteRT-LM currently holds the public crown at 56.5 tok/s.

## Strategy

- **llama.cpp Metal as the floor.** Start from a known-good Metal kernel set (INT4 matmul, flash-attention, KV cache management) and match its measured 38 tok/s baseline on Gemma 4 E2B.
- **Speculative decoding on top.** EAGLE-3, MTP, and prompt-lookup drafters share a single verifier. Google's 56.5 tok/s number is greedy decode — spec decode is the asymmetric advantage.
- **ANE + GPU hybrid.** Prefill on ANE (reuse the chunk models from the sibling CoreML-LLM repo), hand the KV cache to Metal for decode. Mirror SD runs GPU draft and ANE verify in parallel.
- **A19 tensor core kernels.** Hand-tuned kernels for A19's NAX tensor cores where llama.cpp's generic Metal path under-utilises the hardware. In-register INT4 dequant to avoid the FP16 bandwidth tax LiteRT pays.

## Why a separate repo

Sibling repo: [CoreML-LLM](../CoreML-LLM) — ANE-first, always-on, battery-friendly, shipped.

Metal-LLM is a different thesis. CoreML-LLM keeps the GPU free for the rest of the app; Metal-LLM assumes you want the GPU and are willing to trade some battery for throughput. Keeping the codebases separate lets each optimise for its own regime without the Core ML runtime constraints leaking into the Metal path (or vice versa). The ANE bridge here depends on CoreML-LLM's chunk models but not on its runtime.

## Status

**Week 0 — scaffold.** No kernels yet. See [docs/ROADMAP.md](docs/ROADMAP.md) for the 12-week plan.

## License

MIT (placeholder — to be confirmed once llama.cpp integration lands).
