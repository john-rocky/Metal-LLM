# Metal-LLM Architecture

**Status:** Design, Week 0. This document drives 3–6 months of implementation. It is deliberately opinionated; every speculative choice is flagged and linked to a prototype task in the "Open Questions" section. The target is **80–130 tok/s sustained decode on Gemma 4 E2B, iPhone 17 Pro**, beating LiteRT-LM's public 56.5 tok/s number.

**Audience:** the engineer writing the Metal kernels, the engineer writing the ANE bridge, and the reviewer in three months asking "why did we pick A over B".

---

## 1. Thesis

Gemma 4 E2B decode on A19 Pro is bandwidth-bound at the matmul and dispatch-bound at per-layer orchestration.

- **GPU (Metal)** — low dispatch latency (tens of µs/kernel), high sustained bandwidth. Wins the decode hot path: one token every ~15 ms, kernel-launch overhead disappears. llama.cpp Metal already lands ~38 tok/s on E2B.
- **ANE** — higher dispatch latency (~ms per `MLModel.prediction()`), power-efficient for large compute blocks, dedicated INT4/INT8 matmul. Sibling's 4-chunk prefill beats GPU prefill wall-clock for prompts longer than ~256 tokens.

We want both. Prefill on ANE, decode on Metal, Mirror Speculative Decoding with drafter on ANE and verifier on GPU running **in parallel on different silicon**. That is the asymmetric advantage LiteRT-LM doesn't have. The central technical bet is a shared KV cache across engines with no per-token copy. Section 2 answers that.

---

## 2. Shared KV Cache Design

This is the central technical challenge. Fail here and the hybrid collapses because the prefill-to-decode handoff and Mirror-SD verify both become bounded by a memcpy of ~50–200 MB of KV state.

### 2.1 What the two engines natively want

| Engine | Preferred backing |
|---|---|
| Metal decode | `MTLBuffer` with `MTLStorageModeShared` on Apple Silicon (unified memory, CPU and GPU both read/write). |
| ANE via Core ML | `MLMultiArray`. On ANE the compiler prefers buffers whose underlying storage is an `IOSurface` (64-byte row pitch, min size ≈ 49 KiB); otherwise Core ML silently copies the input to an ANE-owned scratch region on every prediction. |

An `IOSurface` is the OS-level primitive both are implemented on top of: `MTLDevice.makeBuffer(iosurface:plane:)` returns an `MTLBuffer` view of an IOSurface, and `MLMultiArray(pixelBuffer:shape:)` returns an `MLMultiArray` view of the IOSurface backing a `CVPixelBuffer`. **The same physical memory can be exposed as both.** This is a supported path since iOS 11 (the camera pipeline has used it for years).

The sibling CoreML-LLM repo already allocates its KV cache this way. It builds a `CVPixelBuffer` with `kCVPixelBufferIOSurfacePropertiesKey = {}` and `kCVPixelBufferMetalCompatibilityKey = true`, memsets to zero, and wraps via `MLMultiArray(pixelBuffer:shape:)`. See `Sources/CoreMLLLM/ChunkedEngine.swift:437-468` in the sibling repo; that exact allocator is lifted verbatim into Metal-LLM.

### 2.2 Three options considered

- **Option A — Single IOSurface, two views (chosen target).** Allocate KV slabs as IOSurface-backed CVPixelBuffers. Wrap as `MLMultiArray` for ANE prefill. Wrap the same IOSurface as an `MTLBuffer` via `MTLDevice.makeBuffer(iosurface:plane:)` for Metal decode. Zero copy on handoff, zero copy during Mirror SD.
- **Option B — Copy on handoff.** Run ANE prefill into its own buffers, `memcpy` to `MTLBuffer` before decode. ~50 MB at ~40 GB/s on-chip ≈ 1.2 ms. Acceptable for one-off prefill→decode handoff. **Unacceptable for Mirror SD**, which would pay the copy on every burst.
- **Option C — Final ANE forward commits KV, then read out.** Same problem as B, still one copy per handoff; no help for Mirror SD.

Option A is the only one that survives Mirror SD. Option B is the fallback: it ships first (Week 1) so we have a working end-to-end before the zero-copy bringup in Week 4.

### 2.3 Constraints Option A imposes

This is where the design gets rigorous and where most Week 4–6 bugs will come from.

- **Data type must match.** The sibling uses `kCVPixelFormatType_OneComponent16Half` (FP16). Matches Core ML's ANE native KV type. Matches llama.cpp's Metal KV when compiled with `GGML_USE_F16_KV=ON` (our default).
- **Layout must match.** Core ML's ANE prefill uses `(slots, 1, seq_len, head_dim_padded=512)`. llama.cpp's KV layout is `(layer, head, pos, head_dim)`. Before the two engines can share, we need a layout **view** translation — not a copy, a re-indexing on the same IOSurface. Plan: patch llama.cpp's `ggml_metal_kv_cache` to accept a user-provided `MTLBuffer` + stride tuple, then compute strides so Metal sees the ANE layout as equivalent. **Speculative.** See Open Question #1.
- **Alignment.** ANE requires 64-byte row alignment; Metal is happy with any alignment. Match 64 everywhere.
- **Synchronization.** We rely on the observed fact that `MLModel.prediction()` is blocking: by the time it returns, ANE has committed writes to the IOSurface and invalidated its caches. Metal's own command buffer ordering handles GPU-side coherence from there. No explicit `MTLSharedEvent` needed — unless Open Question #2 says otherwise.
- **Write-window collision on Mirror SD.** During a burst, ANE drafter reads frozen KV at position `p-1`; GPU verifier writes positions `p..p+K-1`. We must guarantee ANE is not reading positions GPU is writing. In practice the drafter KV is tiny (2 layers × small `head_dim` — the sibling's `kv13K/kv14K`) and lives in a separate slab from the main target KV, so there is no collision. This is a load-bearing **topology** fact, not a sync trick.

### 2.4 Fallback ladder

The Week-1 engine ships with Option B. Option A lights up in Week 4 behind a feature flag `kSharedKVZeroCopy = false`, enabled once the layout-view translation passes parity vs Option B bit-exactly on a 512-token prompt. Rolling back is a one-line flag flip.

---

## 3. Component Diagram

```
                                  ┌──────────────────────────────┐
                                  │     MetalLLMEngine (actor)   │
                                  │  - public async API          │
                                  │  - owns lifecycle, metrics   │
                                  └──┬────────────┬──────────────┘
                                     │            │
                ┌────────────────────┴──┐      ┌──┴───────────────────┐
                │  SpecDecodeOrch       │      │  MetricsCollector    │
                │  (Mirror SD sched.)   │      │  tok/s, TTFT,        │
                └──┬──────┬──────┬──────┘      │  thermal, power,     │
                   │      │      │             │  ANE/GPU busy frac.  │
                   ▼      ▼      ▼             └──────────────────────┘
        ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
        │ ANEBridge    │ │ LlamaBridge  │ │ SharedKVSlab     │
        │  Core ML     │ │ llama.cpp +  │ │  IOSurface-backed│
        │  chunk1-4    │ │ Metal kernels│ │  FP16, NHWC      │
        │  (prefill)   │ │  (decode,    │ │                  │
        │  + drafter   │ │   verify)    │ │ viewed as both   │
        │  (Mirror SD) │ │              │ │ MLMultiArray AND │
        └──────┬───────┘ └──────┬───────┘ │ MTLBuffer        │
               │                │         └──────────────────┘
               └────────┬───────┘
                        ▼
             ┌─────────────────────┐
             │ Shared KV IOSurface │
             │  (physical memory)  │
             └─────────────────────┘
```

`ANEBridge` consumes compiled `.mlmodelc` artifacts produced by the sibling CoreML-LLM conversion pipeline. `LlamaBridge` wraps a `llama_context*` built from a GGUF of the same checkpoint. Both view the same KV slab. See §7 on asset versioning.

---

## 4. Prefill → Decode Handoff

### 4.1 Sequence diagram

```
User         Engine           ANEBridge         KVSlab         LlamaBridge
 │  prompt     │                   │                │                │
 ├────────────▶│                   │                │                │
 │             │ tokenize ─────────────────────────────────────────▶│  llama_tokenize
 │             │                   │                │                │
 │             │ prefill(tokens)   │                │                │
 │             │──────────────────▶│                │                │
 │             │                   │ chunk1(batch of 128 tokens)     │
 │             │                   │───────────────▶│  write K/V@pos │
 │             │                   │ chunk2, 3, 4                     │
 │             │                   │───────────────▶│                 │
 │             │                   │   …repeat for next 128-batch    │
 │             │◀── last logits    │                │                │
 │             │                   │                │                │
 │             │ decode_loop(startPos)                               │
 │             │────────────────────────────────────────────────────▶│
 │             │                                    │  read K/V      │
 │             │                                    │◀───────────────│
 │             │                                    │  write K/V@pos │
 │             │                                    │───────────────▶│
 │◀── token    │◀── sampled token id                │                │
 │ token       │   ...                              │                │
```

### 4.2 Prefill batching

ANE has a practical IOSurface floor of 49 KiB per tensor and a dispatch overhead of ~2 ms per `MLModel.prediction()` call. Optimal prefill batch size is where `batch × per-token-compute ≈ 4–8 ms per dispatch`. Sibling measurements land at **128 tokens per prefill step** for Gemma 4 E2B chunks 1–4. Shorter prompts are padded to 128 and masked; longer prompts loop in 128-token windows, advancing `currentPosition` in the shared KV slab.

**Speculative:** A19 Pro may prefer 256. Open Question #3.

### 4.3 First-token latency budget (TTFT ≤ 500 ms, 512-token prompt)

| Stage | Budget |
|---|---|
| Tokenize | 2 ms |
| ANE prefill warm-up (one-time per session, cached across calls) | 0 ms amortized |
| ANE prefill, 4 × 128 tokens × ~15 ms/batch | ~240 ms |
| Handoff (Option A, zero-copy) | ~0 ms |
| First decode step (Metal) | 15 ms |
| Detokenize + yield | 1 ms |
| **Total** | **~260 ms** |

The sibling repo currently prefills 2048 tokens in ~13 s cold, ~2 s warm. 512 tokens warm is ~500 ms, so the budget is realistic but tight.

### 4.4 Swift async/await interface

```swift
public actor MetalLLMEngine {
    private let llama: LlamaBridge
    private let ane: ANEBridge?
    private let kv: SharedKVSlab
    private let spec: SpecDecodeOrchestrator
    private let metrics: MetricsCollector
    private var generation: Int = 0   // invalidates in-flight streams on cancel

    public func load(model: ModelBundle) async throws {
        try await llama.loadModel(path: model.ggufURL, contextLength: model.ctx)
        if let aneAssets = model.aneAssets {
            self.ane = try ANEBridge(chunks: aneAssets.chunks, kv: kv)
            try await ane!.warmCompilePlans()    // ANE prewarm (see sibling ChunkedEngine.swift:505)
        }
    }

    public func generate(prompt: String,
                         config: SamplingConfig) -> AsyncThrowingStream<TokenEvent, Error> {
        AsyncThrowingStream { continuation in
            let gen = self.generation
            Task.detached { [self] in
                do {
                    let tokens = try await llama.tokenize(prompt, addBOS: true)
                    let startPos: Int
                    if let ane = await self.ane {
                        startPos = try await ane.prefill(tokens: tokens)  // writes shared KV
                    } else {
                        startPos = try await llama.prefillFallback(tokens: tokens)
                    }
                    var pos = startPos
                    var next: Int32 = try await llama.sampleFirst(config: config)
                    while pos < config.maxTokens, await self.generation == gen {
                        let emitted = try await spec.step(
                            next: &next, position: &pos,
                            config: config, ane: self.ane, llama: self.llama)
                        for t in emitted { continuation.yield(.init(tokenId: t /* … */)) }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func cancel() { generation &+= 1 }
}
```

`SharedKVSlab` owns the IOSurface and hands out both `MLMultiArray` and `MTLBuffer` views; resets serialize via the actor mailbox. It does **not** gate per-token reads/writes — that would kill decode throughput. Coherence comes from the natural happens-before between ANE `prediction()` return and the subsequent Metal `commandBuffer.commit()`.

---

## 5. Mirror SD Scheduler

### 5.1 Topology

Mirror SD (arXiv 2510.13161) runs drafter and verifier on different silicon simultaneously so drafter latency overlaps verifier latency. The sibling repo's `MirrorSpeculativeLoop` runs draft-GPU + verify-ANE because its target is on ANE. We flip it: **draft-ANE + verify-GPU**, because our target is on GPU.

Why this direction wins: verifier is the bigger model (~1B params of activation) and benefits most from GPU tensor cores; drafter is tiny (~48M params per Apple Foundation Models precedent, or the MTP head in Gemma 4) and maps cleanly to ANE's INT4 matmul units. Drafter wall-clock ~3–5 ms roughly matches verify-per-token ~15 ms divided by acceptance-rate-weighted `K`, which is where the pipelining win comes from.

### 5.2 Burst sequence (v1: serial within burst, parallel across bursts)

```
time →
GPU:  verify[N-1] ──────────▶  verify[N] ──────────▶  verify[N+1] ─────▶
ANE:              draft[N] ──▶             draft[N+1] ──▶    draft[N+2]
                                         ▲
                             drafter reads KV at pos committed by verify[N-1]
```

Within one burst, drafter must finish before verifier runs — verify consumes proposals. Across bursts we start drafting for burst N+1 while verify N is still on GPU, because drafter only needs KV at burst N's commit boundary (known at burst boundary) and its 2-layer KV is a separate slab. This is the sibling repo's "v2" pattern and is where the Apple Mirror SD paper's +30% lands.

### 5.3 Commit and rollback

- **Accept:** greedy argmax match up to first disagreement, exactly as `MtpSpeculativeEngine.speculateStep` in the sibling repo (`Sources/CoreMLLLM/MtpSpeculativeEngine.swift:67`). Advance `kv.currentPosition` by `matchCount + 1`.
- **Reject (partial):** truncate KV via `llama_kv_cache_seq_rm(ctx, 0, acceptedPos, -1)`, continue from GPU-sampled correction token. Drafter re-bootstraps from new KV state next burst.
- **Full reject (drafter disagrees on position 0):** penalty is one wasted ANE dispatch (~3 ms). Rolling-acceptance EMA drops below `fallbackThreshold` (default 0.30, matching sibling's `MirrorSpeculativeLoop.fallbackThreshold`); scheduler falls back to GPU-only decode for N bursts before retrying the drafter.

### 5.4 Swift pseudocode

```swift
public struct SpecDecodeOrchestrator {
    let K: Int = 3
    private var rollingAccept: Double = 1.0
    private let α: Double = 0.05
    public var fallbackThreshold: Double = 0.30

    mutating func step(next: inout Int32, position: inout Int,
                       config: SamplingConfig,
                       ane: ANEBridge?, llama: LlamaBridge) async throws -> [Int32] {

        guard rollingAccept >= fallbackThreshold, let drafter = ane?.drafter else {
            // Fallback: vanilla decode on GPU
            let logits = try await llama.decode(tokens: [next],
                                                 positions: [Int32(position)],
                                                 returnAllLogits: false)
            let t = sample(logits, config)
            position += 1
            next = t
            return [t]
        }

        // Parallel launch: ANE draft; GPU verify waits on draft completion
        // within this burst. Cross-burst pipelining kicks in once burst N+1's
        // draft is enqueued before burst N's verify resolves on GPU.
        async let proposalsT = drafter.draft(seed: next, K: K, kvView: kv.aneView)
        let props = try await proposalsT                         // ~3–5 ms ANE

        let verifyTokens: [Int32] = [next] + props.dropLast()
        let argmax = try await llama.verify(tokens: verifyTokens,
                                             startPosition: Int32(position),
                                             K: K)               // ~15 ms GPU

        var matched = 0
        for k in 0..<(K - 1) where props[k] == argmax[k] { matched += 1 }
        var emitted: [Int32] = [next]
        emitted.append(contentsOf: props.prefix(matched))

        let carry = argmax[matched]
        position += matched + 1
        next = carry
        rollingAccept = α * Double(matched) / Double(K - 1) + (1 - α) * rollingAccept
        return emitted
    }
}
```

Cross-engine coordination overhead is bounded above by one `async let` hop plus one actor hop — budget **0.5 ms per burst**. Speculative; measure Week 5.

---

## 6. Failure Modes

Enumerated because each has observed precedent in the sibling repo.

### 6.1 ANE error -14 ("Failed to build the model execution plan")

Root cause in sibling `docs/SPLIT_ROTATE_FINDINGS.md`: ANE compiler rejects `coreml_update_state` MIL ops and certain stateful graphs. Mitigations:

- Metal-LLM does not use MLState at all for drafter or prefill, because our KV is external (shared IOSurface slab). Drafter Core ML model takes KV slices as normal inputs and returns proposals + carry hidden as normal outputs — the path already proven in sibling's `MtpSpeculativeEngine`.
- If a shipped `.mlmodelc` still fails ANE compile on a given device (e.g., A16 falling back to GPU), `ANEBridge.loadPlan` catches the error, retries with `computeUnits = .cpuAndGPU`, logs a warning, and disables Mirror SD (drafter would now fight verifier for GPU cycles, pointless). Vanilla prefill via ANE still runs on devices where only the drafter fails.

### 6.2 Thermal throttle

Poll `ProcessInfo.processInfo.thermalState` every decode step. At `.serious`, halve `K`. At `.critical`, disable spec decode entirely, fall back to GPU-only greedy.

Empirically the ANE throttles slower than GPU on A19 Pro (thermally less dense), so under sustained load the Mirror SD asymmetry may **grow**. That's a guess — verify on a 10-minute bench (Open Question #5).

### 6.3 Memory pressure

On `didReceiveMemoryWarning`: flush unused Mirror-SD compile plans, drop `PrefixKVCache` LRU to 8 entries, and — last resort — unload ANE prefill (Metal can prefill too, slower). Never unload the GPU context; that aborts decode.

Core ML `.mlmodelc` weights are mmap'd, so the OS can evict; they re-fault on next call. Fine for prefill (~200 ms fault per rare eviction). **Bad for drafter** which runs every ~30 ms — pin the drafter compile plan in memory via the `warmComputePlans` technique the sibling uses (`ChunkedEngine.swift:505`).

### 6.4 Mid-sequence prefill (RAG, continue-chat)

ANE prefill chunks are position-aware: they take `cos_full / cos_sliding / sin_full / sin_sliding` RoPE tables and a `causal_mask_full` indexed by absolute position. They do **not** assume fresh state. To continue from a snapshot, restore KV slab bytes (via `PrefixKVCache`, ported from sibling), set `currentPosition`, and prefill starting from that position.

One catch: sliding-window-attention KV is physically circular-buffered. Restoring a snapshot must also restore the write cursor, not just the bytes. Sibling handles this; the Week 4 `continueChat` fixture tests it.

---

## 7. Model Asset Versioning

### 7.1 Where assets come from

Sibling `conversion/` produces compiled `.mlmodelc`: `chunk1..4`, `prefill_chunk1..4`, `verify_chunk1..4`, plus RoPE tables (`cos_sliding.npy`, `sin_sliding.npy`, `cos_full.npy`, `sin_full.npy`), plus the MTP drafter `.mlpackage`. Those live under `CoreML-LLM/Examples/CoreMLLLMChat/CoreMLLLMChat/`.

Metal-LLM additionally needs a GGUF of the same checkpoint for llama.cpp. Separate conversion path, not gated on the sibling.

### 7.2 How Metal-LLM consumes them

**Build time:** `Models/` is not a copy but a `ModelBundle.resolvedURL` lookup in this order:

1. `$METAL_LLM_MODELS_DIR` env var (CI).
2. `Bundle.main.resourceURL/Models/` (iOS app bundle, production).
3. Relative path to sibling, resolved via `Models/bundle.json`: `{ "ane_source": "../CoreML-LLM/Examples/.../Models", "gguf_source": "Models/gemma4-e2b-q4_k.gguf" }`.

**Dev on Mac:** path 3 symlinks the sibling's model dir; sibling re-conversion picks up automatically.

**Prod on iPhone:** path 2; a Run Script rsyncs from path 3 at build time. `ane_assets_version.json` is embedded and validated on load against a content hash; mismatch fails loudly, never silently falls back.

### 7.3 When the sibling updates

Sibling conversion is slow (hours for 4-chunk Gemma 4 E2B). On publish of a new `Models/` set, Metal-LLM picks up on next `swift build` via the Run Script. No code change unless model schema shifted (new hidden dim, new layer count, etc.), in which case the hash mismatch fails-fast at load.

---

## 8. Metrics Surface

Emitted by `MetricsCollector`, accessed via `Engine.metricsSnapshot()`.

| Metric | Source | Cost |
|---|---|---|
| `tokPerSecondSteady` | rolling 64-token window around `mach_absolute_time()` | ~0 |
| `ttftMs` | `generate()` invocation to first yield | ~0 |
| `thermalState` | `ProcessInfo.processInfo.thermalState` | ~0 |
| `thermalStateChanges` | `NSProcessInfoThermalStateDidChangeNotification` | ~0 |
| `gpuBusyFraction` | `MTLCommandBuffer.gpuStartTime/gpuEndTime` / wall clock | negligible |
| `aneBusyFraction` | approximate as `sum(MLModel.prediction durations) / wallClock` — flag `estimate` | ~0 |
| `acceptanceRate` | `SpecDecodeOrchestrator.rollingAccept` | ~0 |
| `kvFootprintBytes` | `SharedKVSlab.byteCount` | ~0 |
| `estPowerMilliwatts` | see below | ~0.5 ms per sample |

**Power telemetry.** `IOKit` / `IOReport` power states are locked down on iOS (available on macOS via private IOKit calls; on iOS only exposed through Instruments). In-app we can only report proxies:

- *Battery delta*: sample `UIDevice.current.batteryLevel` pre/post a 60-second bench. 1% resolution. Useful for comparing runs, not for instantaneous power.
- *SoC energy via Instruments*: not from in-app code. We ship an Instruments template (`Bench/PowerBench.instruments`) for manual dev-side capture.

Surface `batteryDeltaPerMinute` (live, rough) plus document the Instruments template (exact, manual).

Speculative: Apple may expose `MXMetrics`-style ANE busy telemetry on iOS 19. Check WWDC 2026.

---

## 9. Open Questions Requiring Prototype

Each is blocking for the week listed. An open question here is one we refuse to resolve from docs alone.

1. **Layout-view translation between ANE and Metal KV.** Does llama.cpp's `ggml_metal_kv_cache` accept a user-provided `MTLBuffer` + stride? If not, how invasive is the patch? **Week 4, 2 days.**
2. **ANE → Metal coherence without explicit fence.** Is `MLModel.prediction()` return a hard memory barrier across engines, or do we need `MTLSharedEvent`? Checksum the IOSurface from GPU right after ANE return, 1M iterations, look for tearing. **Week 4, 1 day.**
3. **Optimal ANE prefill batch size on A19 Pro.** Sibling uses 128. Measure 64/128/256/512 on a 2K prompt; report wall-clock + dispatch overhead. **Week 3, 0.5 days.**
4. **Mirror SD cross-engine coordination overhead.** Target ≤0.5 ms per burst. Measure with `os_signpost` around `async let` launch/join. **Week 5, 0.5 days.**
5. **Thermal differential ANE vs GPU under sustained decode.** Does Mirror SD win widen or shrink at `.serious` thermal? **Week 6, 1 day bench.**
6. **Drafter KV vs target KV collision.** Confirm sibling's `kv13K/kv14K` slabs are physically separate from the main KV slab. Read sibling's allocator and diagram. **Week 4, 2 h reading.**
7. **INT4 in-register dequant kernel vs llama.cpp generic FP16 matmul.** If we only match and don't beat it, we still need it as prerequisite for the +20% goal. **Week 8, 3 days.**
8. **Can `MTLDevice.makeBuffer(iosurface:)` bind an IOSurface with pixel format `OneComponent16Half`?** Documented for `BGRA8Unorm`, not explicitly for FP16 half. Fallback: reinterpret via `MTLResourceOptions.storageModeShared` + manual stride math. **Week 4, 1 day. Single most load-bearing unknown in Option A.**

---

## 10. References

- Mirror Speculative Decoding — arXiv 2510.13161 (Apple ML Research, 2026). Cross-silicon draft/verify overlap pattern; we implement it in reverse direction to the paper.
- Apple, *Introducing Apple's On-Device and Server Foundation Models* (2024 Transformer report). Drafter size ≈ 48M params; precedent for Apple's own spec-decode on ANE.
- Apple, [Core ML — Stateful Models Guide](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html). MLState semantics. ANE compiler rejects `coreml_update_state` with error -14; Metal-LLM avoids MLState by keeping KV external.
- Apple, `IOSurface` developer docs + `MTLDevice.makeBuffer(iosurface:plane:)`. Zero-copy bridge, Metal side.
- Apple, `MLMultiArray(pixelBuffer:shape:)`. Zero-copy bridge, Core ML side.
- llama.cpp Metal backend — `ggml/src/ggml-metal.m`, `ggml_metal_kv_cache_*` and FP16 matmul path.
- Sibling repo **CoreML-LLM**:
  - `Sources/CoreMLLLM/ChunkedEngine.swift` (IOSurface KV allocator, prewarm, outputBackings pattern).
  - `Sources/CoreMLLLM/MirrorSpeculativeLoop.swift` (Mirror SD reference impl, opposite direction).
  - `Sources/CoreMLLLM/MtpSpeculativeEngine.swift` (commit/accept semantics, carry-state extraction).
  - `Sources/CoreMLLLM/PrefixKVCache.swift` (persistent-prefix protocol we reuse verbatim).
  - `docs/SPLIT_ROTATE_FINDINGS.md` (ANE error -14 root-cause analysis).
  - `docs/CONVERSION.md` (chunk layout, asset paths, compile recipe).
- LiteRT-LM public numbers — Google AI Edge blog, 2026. The 56.5 tok/s we are beating.

---

## 11. Non-Goals

Out of scope for the first 12 weeks: training (stays in sibling); multi-modal (sibling Phase 2); CPU-only fallback (engine refuses to load without Metal or ANE); Windows/Linux; models other than Gemma 4 E2B for v1 (GGUF loader is generic, but kernels target E2B shapes — hidden=1536, 35 layers, SWA every-5).

*Next review: end of Week 4 prototype bringup.*
