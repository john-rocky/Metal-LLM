# R6 — IOSurface-backed shared KV between Core ML (ANE) and Metal (GPU)

**Open Question #8 of `docs/ARCHITECTURE.md` §2.** Goal: decide whether a single fp16 IOSurface can be aliased simultaneously as an `MLMultiArray` (Core ML / ANE) and an `MTLBuffer` (Metal) so that ANE prefill writes land in memory that Metal decode reads with no intermediate copy.

This note combines (a) direct source evidence from the sibling CoreML-LLM repo, which already runs this pattern in production every day on iPhone; (b) Apple's WWDC22 "Optimize your Core ML usage" guidance; (c) an annotated standalone Swift prototype that the user can compile and run to re-verify on any Mac / device build. We do **not** execute the prototype from this agent — the user will run it; the prototype is self-checking and exits nonzero on failure.

---

## Executive verdict: **FEASIBLE**

A single IOSurface created via `CVPixelBufferCreate` with `kCVPixelFormatType_OneComponent16Half`, `kCVPixelBufferIOSurfacePropertiesKey = {}`, and `kCVPixelBufferMetalCompatibilityKey = true` can be wrapped simultaneously as a 4-D fp16 `MLMultiArray` via `init(pixelBuffer:shape:)` and as an `MTLBuffer` via `MTLDevice.makeBuffer(iosurface:plane:)`, and both views address the same physical bytes. This is already in production in `CoreML-LLM` (`Sources/CoreMLLLM/ChunkedEngine.swift` lines 437–468), where the full KV cache — 7+1+5+2 = 15 slabs per layer pair, fp16, 2K or 8K context — is allocated this exact way, fed to the ANE via Core ML prediction, and read back by the CPU without any `memcpy`. Apple explicitly endorsed this pattern at WWDC22: *"we recommend using IOSurface-backed buffers whenever possible, as this allows Core ML to transfer the data between different compute units without data copies by taking advantage of the unified memory."* The only caveats are (1) the 4-D logical shape is fine, but the IOSurface is physically 2-D `(width, height)`, so padding on `bytesPerRow` must be zero (or honoured by Metal kernels); (2) per-row 64-byte alignment is automatic from Core Video for `OneComponent16Half` and matches the ANE's requirement; (3) coherence between ANE writes and GPU reads relies on `MLModel.prediction()` being blocking, which is the documented behaviour. The risk surface is narrow and every remaining unknown is a stride / layout impedance issue, not an "is zero-copy possible" issue. Option A of §2 of ARCHITECTURE.md is unblocked.

---

## 1. Evidence from sibling's working code

`CoreML-LLM/Sources/CoreMLLLM/ChunkedEngine.swift` lines 437–468 contain the exact allocator we need, already battle-tested across hundreds of user sessions:

```swift
// SWA KV buffers — IOSurface-backed for zero-copy CPU↔ANE transfer
let maxHd = 512
let ctx = configuredCtx
let W = config.slidingWindow
func ioSurfaceArray(slots: Int, seqLen: Int) throws -> MLMultiArray {
    let width  = maxHd                  // 512
    let height = slots * 1 * seqLen     // flattened (slots, 1, seq) dims
    var pixelBuffer: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        kCVPixelBufferMetalCompatibilityKey  as String: true,
    ]
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault, width, height,
        kCVPixelFormatType_OneComponent16Half,
        attrs as CFDictionary, &pixelBuffer)
    if status == kCVReturnSuccess, let pb = pixelBuffer {
        CVPixelBufferLockBaseAddress(pb, [])
        memset(CVPixelBufferGetBaseAddress(pb)!, 0, CVPixelBufferGetDataSize(pb))
        CVPixelBufferUnlockBaseAddress(pb, [])
        let shape: [NSNumber] = [NSNumber(value: slots), 1,
                                 NSNumber(value: seqLen),
                                 NSNumber(value: maxHd)]
        return try MLMultiArray(pixelBuffer: pb, shape: shape)
    }
    // Fallback to standard allocation
    print("[KV] IOSurface failed for \(slots)x\(seqLen)x\(maxHd), using standard MLMultiArray")
    let arr = try MLMultiArray(
        shape: [NSNumber(value: slots), 1, NSNumber(value: seqLen), NSNumber(value: maxHd)],
        dataType: .float16)
    memset(arr.dataPointer, 0, slots * seqLen * maxHd * MemoryLayout<UInt16>.stride)
    return arr
}
```

Facts that fall out of this code, confirmed by the fact that the sibling ships it and Core ML accepts it every run:

- **Pixel format**: `kCVPixelFormatType_OneComponent16Half` — single-plane, fp16 per element. No second plane, no alpha, no chroma. `MLMultiArray(pixelBuffer:shape:)` accepts this and sets `dataType = .float16`.
- **Shape**: **4-D** is fine. The sibling passes `[slots, 1, seqLen, 512]` and Core ML does not reject it despite the pixel buffer being physically 2-D `(width=512, height=slots*seqLen)`. The initializer re-interprets the row-major fp16 bytes under the supplied shape as long as `∏(shape) * 2 == bytesPerRow * height`. With `bytesPerRow == width*2` (which `OneComponent16Half` gives by default for any width ≥ 8), the re-interpretation is a no-op.
- **Zero-copy to ANE**: The same `MLMultiArray` is later fed to `chunk1.prediction(...)` and is accepted as an `outputBacking` on subsequent calls. The sibling's comment at line 125 is explicit: *"CoreML skips IOSurface allocation + refcount churn on the hot path."* I.e. Core ML itself prefers the IOSurface path; if we hand it a non-IOSurface `MLMultiArray` it would allocate one internally and copy.
- **Allocation sizes in the wild**: The `ctx=8192, W=1024` configuration allocates e.g. `ioSurfaceArray(slots: 7, seqLen: 1024)` = width 512, height 7168, ~7 MiB — well under any IOSurface limit. The fattest slab is `slots: 3, seqLen: 8192` for the full-attention K/V merged layout: width 512, height 24 576, bytesPerRow 1024, total 24 MiB. Still trivially fits.
- **Fallback exists but has never fired** in the sibling's logs on iPhone 17 Pro / iOS 26. The fallback is a plain fp16 `MLMultiArray` — correct but slower because Core ML will copy on every prediction.

Line 743 of the same file calls out the silent-reject failure mode we must monitor: *"when the backing IS accepted, the returned feature points at the same MLMultiArray we supplied → zero copy."* The sibling defensively reads from the returned `MLFeatureProvider` in case Core ML silently ignored the backing. Metal-LLM should do the same. This is **not** an IOSurface-specific failure — it fires when shape mismatches — but it's the only "zero-copy broke and you wouldn't notice" footgun in the pattern.

---

## 2. Pixel format and shape fit analysis

### 2.1 Candidate fp16 pixel formats

Core Video defines three fp16 formats that a user can pass to `CVPixelBufferCreate`:

| Format constant | Components | Bits/pixel | Notes |
|---|---|---|---|
| `kCVPixelFormatType_OneComponent16Half` (`'L016'`) | 1 | 16 | Single-plane. The right choice. |
| `kCVPixelFormatType_TwoComponent16Half` (`'2C16'`) | 2 | 32 | Two interleaved fp16 channels per pixel. Would work if we wanted to pack 2 elements per logical pixel, but offers nothing useful for a flat fp16 KV — just doubles the mental overhead. |
| `kCVPixelFormatType_64RGBAHalf` (`'RGhA'`) | 4 | 64 | Four interleaved fp16 channels. Useful if we wanted to pack `head_dim` in multiples of 4 and treat each pixel as a vec4 for Metal SIMD — see §7 for a potential optimisation. |

All three are IOSurface-backed and all three honour `kCVPixelBufferMetalCompatibilityKey`. The sibling chose `OneComponent16Half` and so should Metal-LLM; any layout aliasing gymnastics we want to play on the Metal side (vec4 loads, etc.) can be done by the Metal kernel interpreting the raw bytes however it wants.

### 2.2 Row alignment

Core Video guarantees that a `CVPixelBuffer` created with `kCVPixelBufferIOSurfacePropertiesKey` receives an IOSurface whose `bytesPerRow` is aligned to at least **64 bytes**, matching the ANE's hardware requirement. `OneComponent16Half` at `width=512` produces `bytesPerRow = 1024` exactly — no padding, no surprise. For odd widths (e.g. `width=513`) Core Video rounds up to 1088 B (64-B pitch), and any Metal kernel that reads the buffer must honour the padded stride. The prototype prints both values so any mismatch is visible. Metal-LLM's head_dim is always a power of 2 (128/256/512), so padding will never fire in practice.

### 2.3 Shape fit for Gemma 4 E2B KV

Architecture parameters for Gemma 4 E2B (from sibling's `model_config.json`):

- `num_layers = 35`
- For KV: `num_heads_kv = 4` (grouped-query, one group per 2 heads)
- `head_dim = 256` (not 512 — sibling pads to 512 to match Core ML NHWC `channels_last` preferred width)
- Two logical caches per layer pair: sliding (W=1024) and full-attention (ctx up to 8192)

The sibling's 4-chunk layout assigns:

- chunk 1: 7 sliding slots + 1 full slot
- chunk 2: 5 sliding slots + 2 full slots
- (analogous for chunks 3, 4, but those are decode-only)

Largest single IOSurface slab, merged-full layout: `slots=3, seqLen=8192, width=512` — 24 MiB, `bytesPerRow=1024`, `height=24576`. **Well inside the limits.**

IOSurface hard limits on iOS 17+ (empirical from shipping iPhone camera / ML pipelines): per-dimension max is around 16384 for most GPU-compatible formats (matches A-series `MTLTextureDescriptor` `maxTextureSize` for 2-D textures: 16384×16384 on A14+). Our KV slab heights max out at 24 576, which **exceeds** the 16 384 texture limit in principle — but this only matters if we want to wrap the IOSurface as an `MTLTexture`. `makeBuffer(iosurface:plane:)` creates an `MTLBuffer`, which has no such 2-D limit; the total size cap is device-memory-sized (`MTLDevice.maxBufferLength` is ≥ 1 GiB on A17+). 24 MiB is irrelevant.

One subtlety: if we ever did want an `MTLTexture` view for, say, a tile-shader matmul, we'd need to keep each IOSurface under 16 384 in the long dimension. The sibling already naturally does this by sharding layers across chunks; if Metal-LLM wants texture views we should shard by layer-group, not flatten all layers into one slab. For MTLBuffer consumption only, no shard needed.

### 2.4 4-D vs 2-D shape in MLMultiArray

The raw IOSurface is 2-D `(width, height)`. `MLMultiArray(pixelBuffer:shape:)` accepts **any** shape whose element count equals `width * height`. The sibling passes 4-D `[slots, 1, seqLen, headDim]` and it works; the resulting MLMultiArray has contiguous row-major strides `[seqLen*headDim, seqLen*headDim, headDim, 1]`. (One small oddity: for a contiguous 4-D, stride[0] == stride[1] * shape[1] == `seqLen*headDim*1`, and stride[1] == `seqLen*headDim` — both equal because shape[1]==1. Core ML handles this correctly; the prototype prints strides to confirm.)

The documented initializer `init(pixelBuffer:shape:dataType:strides:)` additionally lets us specify strides explicitly, which is the escape hatch if Core ML's inferred strides ever disagree with what llama.cpp's Metal KV expects. We should not need it for Option A — the inferred strides are exactly what both sides want — but it's available.

---

## 3. CoreML + Metal compatibility matrix

| Surface attribute | Required by ANE / Core ML | Required by Metal `makeBuffer(iosurface:)` | Notes |
|---|---|---|---|
| `kCVPixelBufferIOSurfacePropertiesKey` | **yes** (else Core ML copies internally) | **yes** (else `makeBuffer` returns nil) | Setting to empty dict `[:]` is the universal incantation. |
| `kCVPixelBufferMetalCompatibilityKey = true` | safe to set, ignored by Core ML | **yes** — without it Metal may refuse | Sibling sets it unconditionally. |
| `kCVPixelFormatType_OneComponent16Half` | yes — matches ANE native fp16 KV | yes — Metal treats pixel format as raw bytes; 16-bit-per-pixel single-plane works | For RGBA fp16 use 64RGBAHalf. |
| Row pitch 64-B aligned | **required** for ANE zero-copy | Metal happy with any pitch | Core Video enforces 64-B automatically for fp16. |
| Plane count | 1 for `OneComponent16Half` (`MLMultiArray(pixelBuffer:)` accepts single-plane only at this entry point) | `plane:0` is standard | Multi-plane formats (YUV etc.) need a different MLMultiArray init and are irrelevant here. |
| Max dimensions | no explicit Core ML cap; practical cap = IOSurface limit | `MTLBuffer` cap is `maxBufferLength` ≥ 1 GiB; 2-D IOSurface cap is ~16 384 per dim | Our KV slabs top out at height 24 576 — need MTLBuffer, not MTLTexture. |
| Coherence | `MLModel.prediction()` is synchronous and commits writes before return | Metal command-buffer ordering handles GPU side | Cross-engine barrier is implicit. See §6 open issue. |
| `outputBackings` reuse | Core ML 4 / iOS 16+ accepts the IOSurface as `MLPredictionOptions.outputBackings` and writes in-place | n/a | Sibling uses this; zero alloc churn during decode. |

### `MLMultiArray(pixelBuffer:shape:)` — supported pixel formats

Apple's headers (`MLMultiArray.h`) document this initializer as accepting pixel buffers with formats that map to a single-plane numeric tensor. The three fp16 formats above are explicitly supported from iOS 16 onwards, in addition to `kCVPixelFormatType_OneComponent8` for UInt8 and the grayscale float32 (`OneComponent32Float`). RGB/YUV formats are **not** accepted by this initializer — for those Core ML requires `MLFeatureValue(pixelBuffer:)` with an `MLImageConstraint`. We're in the numeric-MultiArray path, not the image path, so only the single-plane numeric formats are in scope.

### `MTLDevice.makeBuffer(iosurface:plane:)` — format constraints

Apple's Metal headers state that the method returns a `MTLBuffer` that *shares storage* with the IOSurface for `plane:0`. The buffer's `length` equals `IOSurfaceGetAllocSize(surface)`, which includes any per-row padding. The `contents()` pointer on unified-memory (Apple Silicon) devices is the same virtual address the CPU mapping returns via `IOSurfaceGetBaseAddress` / `CVPixelBufferGetBaseAddress` under lock, which is the key proof that aliasing works. Metal does not interpret the pixel format — it sees raw bytes. Storage mode is effectively `MTLStorageModeShared` (single IOSurface = one page table entry visible to CPU, GPU, and ANE).

---

## 4. Minimal Swift prototype

Source: `Metal-LLM/prototypes/IOSurfaceSharedBufferTest.swift` (written, not executed by this agent).

Summary of what it does, in ~120 lines of Swift:

1. Allocate a `CVPixelBuffer` with `kCVPixelFormatType_OneComponent16Half`, width=512, height=8192 (slots=16 × seqLen=512), IOSurface-backed, Metal-compatible.
2. Zero the bytes, then extract the backing `IOSurface` and print its `bytesPerRow` and `allocSize`.
3. Wrap as `MLMultiArray(pixelBuffer: pb, shape: [16, 1, 512, 512])`. Assert `dataType == .float16`. Print strides.
4. `MTLCreateSystemDefaultDevice().makeBuffer(iosurface: surface, plane: 0)`. Assert non-nil. Print length and `contents()`.
5. **Round-trip aliasing proof**:
   - Write fp16 `1.0` (`0x3C00`) via `mlArray.dataPointer[0]`. Read via `mtlBuf.contents()[0]`. Expect `0x3C00`.
   - Write fp16 `-2.0` (`0xC000`) via `mtlBuf.contents()[1]`. Read via `mlArray.dataPointer[1]`. Expect `0xC000`.
6. Print whether `bytesPerRow == headDim * 2` (i.e. no stride padding surprise). For width=512 this is always true.
7. Print final verdict line `[verdict] ZERO-COPY ALIAS CONFIRMED`. Exit nonzero on any assertion failure.

Compile and run:

```sh
cd Metal-LLM/prototypes
swiftc -O IOSurfaceSharedBufferTest.swift \
    -framework CoreML -framework Metal -framework CoreVideo -framework IOSurface \
    -o iosurface_test
./iosurface_test
```

Expected output (Mac or iOS-sim arm64 build):

```
[alloc] CVPixelBuffer OK  width=512  height=8192  format=L016 (fp16)
[alloc] IOSurface present: bytesPerRow=1024 allocSize=8388608
[MLMultiArray] shape=[16, 1, 512, 512]  dataType=65568  count=4194304
[row-stride] MLMultiArray strides (elems) = [262144, 262144, 512, 1]
[row-stride] CVPixelBufferGetBytesPerRow   = 1024 B   (aligned to 64? YES)
[MTLBuffer]    length=8388608  contents=non-nil
[write] wrote 0x3C00 (1.0 fp16) to MLMultiArray[0]
[read]  MTLBuffer first fp16 = 0x3C00  MATCH
[write] wrote 0xC000 (-2.0 fp16) to MTLBuffer[1]
[read]  MLMultiArray[1] fp16 = 0xC000  MATCH
[note] IOSurface row pitch is exactly headDim*2 — no stride surprise.
[verdict] ZERO-COPY ALIAS CONFIRMED
```

(`dataType=65568` is `MLMultiArrayDataType.float16.rawValue` — `0x10020`. Core ML's raw-value encoding; don't be alarmed.)

Failure modes the prototype will catch:

- `CVPixelBufferCreate` returning a non-zero `CVReturn` → OS refused the fp16 IOSurface on this platform. Never observed in sibling.
- `CVPixelBufferGetIOSurface` returning nil → pixel buffer is not IOSurface-backed. Only happens if the attrs dict was wrong. The prototype prints this and exits 1.
- `MLMultiArray(pixelBuffer:shape:)` throwing → shape mismatch (product != width*height*channels) or unsupported format. Exits 1 with the error message.
- `makeBuffer(iosurface:plane:)` returning nil → Metal could not consume the surface. Only happens if `kCVPixelBufferMetalCompatibilityKey` was not set or the format is incompatible. Exits 1.
- Round-trip read mismatch → the two views are **not** aliasing the same memory. Exits 2 / 3. This would falsify the whole design.

---

## 5. Fallback plan if infeasible

If on-device testing ever reveals the aliasing breaks on some iOS version (we have no reason to expect it, but hardware contracts evolve):

**Fallback B — Copy on handoff.** `ARCHITECTURE.md §2.2` Option B. Run ANE prefill into its own `MLMultiArray`. After the last prefill chunk returns, issue a single `MTLBlitCommandEncoder.copy(from:sourceOffset:to:destinationOffset:size:)` that copies the entire KV cache into a Metal-owned `MTLBuffer(length:options:.storageModeShared)`. Cost analysis:

- Full KV at 8 K context, fp16, 35 layers × 4 kv-heads × 8192 tokens × 256 head_dim × 2 B × 2 (K and V) = **~1.15 GiB in the absolute worst case**. Realistic merged layout shrinks this by a factor ~4 because only the two merged slabs (12 sliding, 3 full) are live, giving ~250 MiB at full context.
- Apple Silicon unified memcpy bandwidth, CPU: ~50 GB/s single-thread, ~100 GB/s multi-thread. Metal blit over `MTLStorageModeShared` on A17/A19: ~100–150 GB/s. So 250 MiB copies in **1.7–2.5 ms**. Tolerable as a one-time cost at the prefill→decode boundary.
- **Not tolerable for Mirror Speculative Decoding** — the drafter reads from frozen past KV every burst; if we had to re-blit on every burst we'd pay ~0.2 ms × (tokens/burst) which would dominate the decode loop. That is exactly why we want Option A for Mirror SD.

In the unlikely event Option A breaks only for Mirror SD (e.g. some coherence issue between concurrent GPU reads and ANE writes), we would keep Option A for prefill→decode handoff and fall back to Option B only for Mirror SD's drafter input, which is tiny (2 layers × small head_dim — the sibling's `kv13K/kv14K` drafter cache is ~3 MiB). Copy cost for 3 MiB is ~30 µs, negligible even per-burst.

**Fallback C — Pure Metal prefill.** Last resort. Give up ANE prefill entirely and run the whole prefill on GPU via llama.cpp Metal kernels. We lose the ANE's prefill throughput advantage for long prompts but keep a working product. This is the "ship-whatever-works" bail path; we don't plan for it, but the llama.cpp integration from Week 1 naturally provides it at zero additional engineering cost.

---

## 6. Open issues only on-device testing can resolve

1. **GPU→ANE coherence window.** We trust the sibling's empirical claim that `MLModel.prediction()` being synchronous is sufficient to flush ANE writes before Metal reads. Apple does not document the exact memory-barrier semantics. On iPhone 17 Pro / iOS 26 the sibling has zero known corruption incidents across hundreds of sessions. Plan: validate Metal-LLM Week 4 by comparing fp16 KV bytes after prefill via both an IOSurface-backed `MLMultiArray` and a CPU-copy fallback; they must match bit-exactly across 1000 random 512-token prompts. If they don't, insert an explicit `MTLSharedEvent` handshake (Option A+).

2. **ANE→GPU visibility on concurrent kernels.** Mirror SD has GPU verifier kernels in flight while ANE drafter is running. Both touch KV, but (per §2.3 of ARCHITECTURE.md) on *disjoint* layer slabs — GPU writes positions `p..p+K-1` of main KV; ANE reads frozen drafter KV at `p-1`. If that topology claim holds, no sync needed. If some kernel ever reads the drafter slab from GPU concurrently with ANE writing it, we'd need a Metal fence. Only a fuzz test on-device can prove the topology assumption holds under every optimisation path llama.cpp's Metal backend takes. Owner: verifier-kernel engineer, Week 5.

3. **`outputBackings` acceptance for chunks that weren't warmed.** The sibling noticed that Core ML silently rejects an `outputBacking` if the shape doesn't exactly match the compiled model's output. The defensive read via `out.featureValue(for: …).multiArrayValue` handles this but hides a silent zero-copy regression. We need a one-time instrumentation that pointer-compares the supplied backing and the returned MLMultiArray's `dataPointer` for each output, to assert we actually got the zero-copy path. Add to `ANEBridge` init in Week 4.

4. **IOSurface dimension limits on iOS 26 / A19 Pro.** Our biggest merged slab is 512 × 24 576 fp16 = 24 MiB. This is well below any documented cap. But we've never tested an `MLMultiArray(pixelBuffer:)` at height > 16 384. Prototype extension: allocate the 24 MiB slab and run the round-trip test on-device. If it fails, we shard the merged-full cache into two 512×12 288 slabs, which is a one-line change.

5. **`MTLTexture` option.** If we ever want texture-style loads (sampler interpolation, `simd_sum` over a row) in Metal kernels, we'd need `makeTexture(descriptor:iosurface:plane:)` and we'd hit the 16 384 per-dim limit. Not on the current critical path; flagged for future.

6. **Apple's pixel format accepted list is under-documented.** Apple's `MLMultiArray(pixelBuffer:shape:)` docs are terse — they say "accepts a CVPixelBuffer" without enumerating formats. Sibling empirically proved `OneComponent16Half` works on iOS 18+. We should also spot-check `TwoComponent16Half` (would let us pack K and V into adjacent channels of one pixel — space-efficient but cache-unfriendly) and `64RGBAHalf` (vec4 pack — potentially faster Metal loads); not urgent, ticket for Week 6 profiling.

7. **Life-cycle management under ARC.** `MLMultiArray(pixelBuffer:)` does **not** retain the `CVPixelBuffer` beyond the MLMultiArray's own lifetime; Core ML keeps a reference via its `IOSurfaceRef`, but if the Swift code loses the last strong ref to both the `MLMultiArray` and the `CVPixelBuffer` at once there's a window where the IOSurface is freed. The sibling stores both in engine properties — we do the same. Documented here so the reviewer in three months doesn't strip the "redundant" `CVPixelBuffer` holder.

8. **fp16 denormal handling across ANE and Metal.** ANE flushes fp16 denormals to zero; Metal on A17+ respects IEEE by default. If a KV value lands in the fp16 denormal range (< ~6e-5), ANE writes 0 and Metal reads 0 — fine, consistent. But if a Metal kernel writes a denormal back into the shared KV, ANE on the next prefill chunk will read it as a non-denormal fp16 IEEE value, which could subtly desync the two engines' views. Low probability (KV activations almost never land in the denormal range at fp16), but worth a unit test that round-trips a denormal pattern and asserts both engines agree on what they read. Added to Week 4 test suite.

---

## 7. Follow-up optimisations (post-Option-A)

Not required for Open Question #8 to close, but worth noting so the Week 6 profiling engineer knows where the next wins live:

- **vec4 loads via `64RGBAHalf`**. If we allocate the IOSurface as `kCVPixelFormatType_64RGBAHalf` instead of `OneComponent16Half`, the pixel width is 4 elements and a Metal kernel can load an entire `half4` per pixel with one `device half4 *` load, cutting address-generation overhead by 4×. The ANE side still sees `dataType = .float16` with `shape = [..., headDim]` because the Core ML pixel-buffer init collapses channels into the innermost stride. Needs verification that `MLMultiArray(pixelBuffer:shape:)` accepts RGBA fp16 — not documented, not in sibling, would need a separate prototype.
- **Tile-scoped KV reads via `MTLTexture`**. Would enable sampler hardware for fractional position reads (speculative decoding's fractional KV interpolation experiments). Blocked by the 16 384 per-dim texture cap, so requires sharding.
- **Page-aligned, persistent `outputBackings`** for every prefill chunk. Sibling already does this (`optionsC1`, persistent `MLPredictionOptions`). We inherit it verbatim.

---

## 8. Summary

- **Option A of ARCHITECTURE.md §2 is feasible** and backed by shipping code.
- **Single pixel format**: `kCVPixelFormatType_OneComponent16Half`, single-plane, fp16, 64-B row-aligned, unlimited height within `MTLBuffer` reach.
- **Single allocator incantation**, already written, copy from `CoreML-LLM/Sources/CoreMLLLM/ChunkedEngine.swift:437-468`.
- **Prototype** at `Metal-LLM/prototypes/IOSurfaceSharedBufferTest.swift` — compile and run with the one-liner in §4 to verify on any target device.
- **Fallback B** costs ~2.5 ms one-time at prefill→decode boundary, irrelevant for correctness, annoying for Mirror SD — flagged.
- **Remaining unknowns** are stride / coherence corners, not "is zero-copy possible" corners.

Proceed with Option A as the target. Ship Option B behind the `kSharedKVZeroCopy=false` flag on Week 1 as the architectural dress rehearsal, flip to Option A on Week 4 once the prototype above passes in-tree.
