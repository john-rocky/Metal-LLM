# llama.cpp Integration Decision Document

Target: iOS 18+, iPhone 17 Pro, Swift-first codebase, Gemma 4 E2B via llama.cpp Metal backend.
Goal: 80–130 tok/s decode with EAGLE-3 / MTP / prompt-lookup speculative decoding orchestrated from Swift.
Date: 2026-04-15.

---

## 1. Recommendation Summary

Vendor llama.cpp via a **git submodule** at a pinned upstream SHA, build it ourselves with `build-xcframework.sh` into a local `llama.xcframework`, and consume it from Swift Package Manager as a **`binaryTarget`**. Wrap the C API in a thin Swift `actor` modeled on the upstream `examples/llama.swiftui/llama.cpp.swift/LibLlama.swift` reference (do **not** depend on third-party wrappers like `SwiftLlama`, `LocalLLMClient`, or `llama-cpp-swift` — none expose the speculative APIs we need, and all add a maintenance hop between us and upstream). For speculative decoding, use `common/speculative.{h,cpp}` plus the recently merged EAGLE-3 path (PR #18039, in `llama-speculative-simple`) as the starting point; expect to **fork llama.cpp on a feature branch** to (a) expose `common_speculative_*` from the public C ABI, (b) add an "external drafter" callback so we can plug MTP and prompt-lookup from Swift, and (c) add tree-attention sampling. Build flags: `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON`, `GGML_METAL_USE_BF16=ON`, `GGML_BLAS=ON` (Accelerate), everything else OFF, deployment target iOS 18.0 (residency sets). Use `Q4_K_M` GGUF for Gemma 4 E2B from `ggml-org/gemma-4-E2B-it-GGUF`, with `mlock` and `mmap` both on.

---

## 2. Integration Pattern Choice

### 2.1 Swift wrapper — winner: roll our own from `LibLlama.swift`

Comparison of the candidates we evaluated:

| Wrapper | Stars / Activity | Pinned llama.cpp? | Async API | Speculative? | Verdict |
|---|---|---|---|---|---|
| `ggml-org/llama.cpp` `examples/llama.swiftui` (`LibLlama.swift`) | Upstream, always current | n/a (in-tree) | `actor`, `AsyncStream` ready | No, but trivially extensible | **Use as template** |
| `ShenghaiWang/SwiftLlama` | 92★, last release 0.4.0 (Jul 2025) | Floating | async/await + Combine | No | Skip |
| `tattn/LocalLLMClient` | Active, 0.4.6 (Nov 2025) | Submodule, abstracts MLX too | async/await | No | Skip — extra abstraction layer hides KV cache |
| `srgtuszy/llama-cpp-swift` | Small, mostly demo | Floating | async/await | No | Skip |
| `mattt/llama.swift` | Mostly dormant, semver wrapper | Pinned | Limited | No | Skip |
| `alexrozanski/llama.swift` | Pre-ggml era fork | Stale | n/a | No | Skip |
| `StanfordBDHG` SpeziLLM XCFramework fork | Maintained, semver, GH Actions | Pinned via fork | actor-based | No | Useful as **build-script reference** only |

Rationale: every wrapper above hides exactly the pointers we need (`llama_context*`, `llama_batch`, `llama_kv_cache_*`, sampler chains). Speculative decoding requires direct access to those handles, plus a way to inject hidden states (EAGLE-3) and tree-shape sampling. Adopting a wrapper buys us a few hundred lines of glue code at the cost of forever fighting an opinionated abstraction. The official `LibLlama.swift` (~250 lines, MIT, kept in lock-step with upstream) is the right starting template — copy it into our `Sources/LlamaKernel/`, then extend.

For a Swift LLM harness design pattern (sessions, conversation state, tool-use, async streams), study `mlx-swift-examples/Libraries/LLM` (MLXLLM) and Apple's `AnyLanguageModel`. Don't depend on them; copy the **session orchestration shape** (input → token stream → cancellation token → telemetry callbacks) into our own `LlamaSession` actor that sits one layer above `LlamaContext`.

### 2.2 Binary distribution — winner: submodule + locally built XCFramework + SwiftPM `binaryTarget`

| Option | Sync with upstream | Build time per dev | Binary size | App Store validation | Notes |
|---|---|---|---|---|---|
| SwiftPM `binaryTarget` (XCFramework, hosted zip) | Manual rev bump | ~0 | OK | Best — single embed-and-sign | Recommended **distribution** form |
| SwiftPM source target with `unsafeFlags` | Auto with Package pin | ~3–5 min cold | OK | OK | Breaks semver on consumers, blocks downstream packaging |
| Git submodule + Xcode target compiling sources | Manual rev bump, full source visibility | ~3–5 min cold | OK | OK | Best for **iterating on the C++** (our case) |
| CocoaPods | Stale podspec community-maintained | n/a | n/a | n/a | Skip |

We need both: the submodule for development (we will be patching `common/speculative.cpp` and adding ggml hooks for EAGLE), and an XCFramework `binaryTarget` for shipping. Concretely:

- `External/llama.cpp/` — git submodule pinned to a known-good SHA (we will rebase weekly).
- `Scripts/build-xcframework.sh` — wrapper around upstream `build-xcframework.sh` with our flag set.
- `Frameworks/llama.xcframework` — checked into git-LFS (or rebuilt by CI) for use by `Package.swift`'s `binaryTarget(path:)`.
- App target depends on the local `binaryTarget`, never on raw sources, so iteration in the App stays fast and Xcode autocomplete sees a stable surface.

Upstream sync cadence: bump the submodule SHA on Mondays, run regression against our benchmark suite, ship the new XCFramework if green. Keep our patches as a series of commits on a `mlllm/main` branch in our fork so rebases are manageable.

### 2.3 C interop best practices

1. **Wrap every C handle in an `actor`**. Upstream `LibLlama.swift` already shows the pattern: one actor owns `OpaquePointer` for `llama_model`, `llama_context`, plus the `llama_batch` value type, and the actor's serialized executor gives us thread safety for free against the C API which is **not thread-safe per context**.
2. **Ownership rule**: Swift owns the actor; the actor owns the C pointers; deinit runs `llama_sampler_free`, `llama_batch_free`, `llama_free`, `llama_model_free`, `llama_backend_free()` exactly once. Tokens are passed across the boundary as `[llama_token]` (just `Int32` arrays); no copying needed because we hand `withUnsafeBufferPointer` directly to `llama_batch_add`/`llama_decode`.
3. **Async pattern**: expose `AsyncThrowingStream<Token, Error>` from `LlamaSession.generate(prompt:)`. Inside, the actor runs a single `Task` that loops `llama_decode` → `llama_sampler_sample` → `yield`. Cancellation via `Task.isCancelled` checked between decode steps; on cancel, call `llama_kv_self_seq_rm` to free draft tokens.
4. **Zero-copy KV cache**: never read the KV cache to Swift. Only addressable via opaque `llama_seq_id` values; speculative verify/reject is done by `llama_kv_self_seq_rm(ctx, seq_id, p0, p1)` from Swift, no buffers cross the boundary.
5. **Logits**: the only large data we cross is the logits array. Use `llama_get_logits_ith(ctx, i)` returning `UnsafePointer<Float>` and consume in-place with `UnsafeBufferPointer` — never `Array(...)` copy (vocab is ~256k for Gemma 4, so a copy is 1 MB per token).
6. **Strings**: use `llama_token_to_piece` into a stack buffer (`CChar` array of 32 bytes is enough for ~all tokens), then `String(decoding:as: UTF8.self)`. Carry forward the upstream `temporary_invalid_cchars` trick for split UTF-8 across token boundaries.
7. **C++ interop**: `common/` is C++. Either compile `common/` into the XCFramework and call its C wrapper helpers (preferred — avoids enabling Swift/C++ interop, which is recursive across the dep graph and brittle), or expose a thin `c_api.h` we author ourselves over `common_speculative_*`.

---

## 3. Build Flags and Configuration

Critical CMake flags for our iOS XCFramework, derived from upstream `build-xcframework.sh` plus our deltas:

```bash
# Common (both device and simulator)
-DBUILD_SHARED_LIBS=OFF
-DLLAMA_BUILD_EXAMPLES=OFF
-DLLAMA_BUILD_TOOLS=OFF
-DLLAMA_BUILD_TESTS=OFF
-DLLAMA_BUILD_SERVER=OFF
-DGGML_NATIVE=OFF              # cross-compile, do not auto-detect host
-DGGML_OPENMP=OFF              # iOS lacks libomp; serial OK on Apple GPU
-DGGML_OPENSSL=OFF

# Apple GPU
-DGGML_METAL=ON
-DGGML_METAL_EMBED_LIBRARY=ON  # bake .metallib into .a so we don't need a Bundle resource
-DGGML_METAL_USE_BF16=ON       # bf16 inputs on A19 Pro (iPhone 17 Pro) — significant win

# Apple CPU support
-DGGML_BLAS=ON
-DGGML_BLAS_VENDOR=Apple       # Accelerate framework
-DGGML_ACCELERATE=ON

# Disable everything else (binary size, no dead backends linked in)
-DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_SYCL=OFF -DGGML_VULKAN=OFF
-DGGML_KOMPUTE=OFF -DGGML_RPC=OFF -DGGML_CANN=OFF -DGGML_MUSA=OFF
-DGGML_CPU_AARCH64=ON          # ARM64 CPU kernels for fallback
-DGGML_LLAMAFILE=OFF

# iOS toolchain
-DCMAKE_SYSTEM_NAME=iOS
-DCMAKE_OSX_DEPLOYMENT_TARGET=18.0   # we want residency sets
-DCMAKE_OSX_ARCHITECTURES=arm64
-DCMAKE_C_FLAGS="-fembed-bitcode=marker -Wno-macro-redefined -Wno-shorten-64-to-32"
-DCMAKE_CXX_FLAGS="${CMAKE_C_FLAGS}"
```

Notes:

- Upstream defaults to `-DCMAKE_OSX_DEPLOYMENT_TARGET=16.4`. We bump to **18.0** because iOS 18's Metal residency-set API lets us wire model weights resident, which is the single biggest perf win available on iPhone 17 Pro for sustained decode (no eviction races during long generations).
- `GGML_METAL_EMBED_LIBRARY=ON` is mandatory on iOS — App Store validation rejects loose `.metallib` resources for some signing configurations (see whisper.cpp issue #3046).
- Indirect command buffers (ICBs) are not yet wired into `ggml-metal` — we may add this as part of our perf push (see Risks §6).
- Keep a separate **simulator slice** with `-DGGML_METAL=OFF` (Metal in iOS Simulator on Intel hosts is a footgun; on Apple Silicon hosts it works but the path is rarely exercised). Build it for arm64-simulator only.

---

## 4. Speculative Decoding Integration Plan

### 4.1 What we get from upstream

- **Draft-model speculative**: `common_speculative_init/draft/free` in `common/speculative.{h,cpp}`. Works with any "smaller model + same vocab" pair. Off-the-shelf usable from C++ but **not exported in the public C ABI** (`include/llama.h`). We must add a thin C wrapper in our patch set.
- **N-gram prompt lookup**: `--spec-type ngram-*`, `--spec-ngram-size-n/m`, `--spec-ngram-min-hits`. Same situation — exposed via CLI, not via stable C API. Easy to wrap.
- **EAGLE-3**: PR #18039 (merged late 2025/early 2026) adds `LLM_ARCH_EAGLE3`, an EAGLE-3 encoder/decoder graph, feature extraction at layers `[2, n//2, n-3]`, `g_embeddings` plumbing for the decoder, and `GGML_TENSOR_FLAG_SYNC` for GPU sync. Currently wired only into the `llama-speculative-simple` example, not into the public sampler chain.
- **`llama_sampler` chain**: `llama_sampler_chain_*` lets you compose samplers (top-k, top-p, temperature, mirostat, …). It does **not** currently include a speculative sampler op — speculative is one level up, in `common/speculative.cpp`, which orchestrates draft generation outside the sampler chain.

### 4.2 What we will fork / extend

We'll keep all extensions on a long-lived `mlllm/main` branch in a fork of `ggml-org/llama.cpp` so we can rebase onto upstream weekly. Patch set:

1. **Public C ABI for `common_speculative_*`** — add `llama_speculative.h` next to `llama.h`, exporting init / draft / accept / free as `extern "C"`. ~80 LOC. Submit upstream once stable.
2. **External-drafter callback** — add a `llama_drafter_callback` typedef so Swift can supply its own `(ctx, last_token, n_draft) -> [token]` function. Lets us host EAGLE-3 hidden-state extraction, MTP head logits, or pure prompt-lookup in Swift while reusing llama.cpp's verify/accept loop.
3. **Hidden-state extraction hook** — extend the existing EAGLE-3 graph patch to expose hidden states at layers `[2, n//2, n-3]` via a new `llama_get_hidden_states_ith()` C entry point. This is what we need to drive our own EAGLE-3 head from Swift if we want to swap drafters at runtime.
4. **Tree-attention sampling** — PR #18039 explicitly defers tree sampling. We have two options: (a) wait for upstream tree-attention support; (b) implement a simple 2-wide tree first (verify two drafts in parallel via two `llama_seq_id`s, accept the longer match). We start with (b) as it's ~150 LOC and doesn't require ggml graph changes.
5. **MTP integration** — the Gemma 4 MTP heads we already converted emit token logits directly. We register them as an external drafter via the callback in (2). No fork changes beyond (1)+(2).
6. **Prompt-lookup** — wrap upstream `--spec-type ngram-*` via the new C ABI, no further changes.

Order of work: ship (1)+(2)+(6) first (gets us prompt-lookup speculative end-to-end with no forking pain), then (3)+(5) (MTP — should be the highest-acceptance-rate drafter for short replies), then (4) (tree), then (3) again to drive EAGLE-3 as an external drafter.

`llama_sampler` chain stays untouched — speculative orchestration sits **outside** the sampler chain, calling it once per accepted token. We do not need to inject a "speculative sampler" custom op.

---

## 5. Week-1 Onboarding Procedure

```bash
# Day 1 — fork and pin
gh repo fork ggml-org/llama.cpp --clone=false
git submodule add -b mlllm/main \
  git@github.com:<us>/llama.cpp.git External/llama.cpp
cd External/llama.cpp && git checkout <upstream-good-sha> -b mlllm/main && cd ../..

# Day 1 — first XCFramework build (uses upstream script with our flag overrides)
cat > Scripts/build-xcframework.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export COMMON_CMAKE_ARGS="\
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_METAL_USE_BF16=ON -DGGML_BLAS=ON -DGGML_ACCELERATE=ON \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=18.0"
cd External/llama.cpp && ./build-xcframework.sh
cp -R build-apple/llama.xcframework ../../Frameworks/
SH
chmod +x Scripts/build-xcframework.sh
./Scripts/build-xcframework.sh   # ~5 min on M-series

# Day 2 — wire SwiftPM
# Package.swift adds:
#   .binaryTarget(name: "llama", path: "Frameworks/llama.xcframework"),
#   .target(name: "LlamaKernel", dependencies: ["llama"]),

# Day 2 — copy upstream Swift wrapper as our starting point
cp External/llama.cpp/examples/llama.swiftui/llama.cpp.swift/LibLlama.swift \
   Sources/LlamaKernel/LlamaContext.swift
# Rename `LlamaContext` → keep, and add LlamaSession.swift wrapping it.

# Day 3 — pull a Gemma 4 E2B GGUF
huggingface-cli download ggml-org/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-Q4_K_M.gguf --local-dir Models/

# Day 3 — first run on device
# Add Models/gemma-4-E2B-it-Q4_K_M.gguf to the app bundle (or download at first launch).
# In the app: try LlamaSession(modelPath:).generate(prompt: "Hello").

# Day 4 — baseline benchmark on iPhone 17 Pro
#   - record decode tok/s, prompt tok/s, peak RSS, peak GPU memory, thermalState
#   - target floor: ≥ llama.swiftui demo numbers; we expect ~80 tok/s pre-spec on E2B.

# Day 5 — first patch: expose common_speculative_* in public C ABI.
#   - implement, build XCFramework, add LlamaSpeculative.swift wrapper.
#   - smoke test with prompt-lookup (--spec-type ngram-simple).

# Day 5 — commit, push, open draft PR upstream for the C ABI export
#   so it is reviewed in parallel with our app work.
```

Acceptance criteria for end of Week 1: (a) Swift app launches Gemma 4 E2B Q4_K_M on device, (b) streams tokens at ≥ baseline upstream `llama.swiftui` rate, (c) the build is reproducible from a clean clone + `./Scripts/build-xcframework.sh`, (d) the CI runs the benchmark and posts tok/s back to the PR.

---

## 6. Known Risks and Workarounds

| # | Risk | Likelihood | Workaround |
|---|---|---|---|
| R1 | EAGLE-3 PR #18039 only wired into `speculative-simple` example, not into stable API. | High | Ship prompt-lookup + MTP first; treat EAGLE-3 as Phase 2. Track upstream issue #18301. |
| R2 | Gemma 4 perf regression on Apple Metal observed on master (issue #21655 — 3.8x slower than `b8680`). | High | **Pin to a known-good upstream SHA (e.g. `b8680` or later validated)** rather than tracking master. Re-bench every bump. |
| R3 | iOS 18 Metal residency sets not yet adopted in `ggml-metal`. Wired only into custom backends. | Medium | Add `MTLResidencySet` for model weight buffers in our own Metal device wrapper if upstream doesn't land it. ~200 LOC. |
| R4 | iOS Simulator on Intel host crashes Metal kernels. | Low | Build sim slice with `GGML_METAL=OFF`. Only test Metal on device or Apple-Silicon sim. |
| R5 | `mmap` of large GGUF on iOS causes paging thrash (jackyoustra.com saw 1 GB/s paging on iPad). | Medium | Pass `mlock=true` AND `mmap=true` to `llama_model_params`; fall back to `mlock=true, mmap=false` if iOS rejects mlock above an entitlement-gated limit (need `com.apple.developer.kernel.increased-memory-limit` for E2B's ~2.5 GB Q4_K_M weights on a 6 GB / 8 GB device). |
| R6 | iOS suspension while decoding kills the Metal command queue. | Medium | Hook `UIApplication.didEnterBackgroundNotification`; flush in-flight `llama_decode`, save KV cache state via `llama_state_save_file`, free the Metal command queue. Restore on `willEnterForeground`. |
| R7 | Memory pressure (`UIApplication.didReceiveMemoryWarningNotification`). | Medium | Drop the KV cache (`llama_kv_self_clear`), keep model weights resident, surface a "context reset" event to the UI. Do **not** unload weights — reloading is multi-second. |
| R8 | Thermal throttling on sustained decode. | Medium-High on iPhone 17 Pro for long sessions | Poll `ProcessInfo.processInfo.thermalState` every N tokens; on `.serious`/`.critical`, downshift batch size and pause draft model (cut FLOPs in half). Surface a "cooling" UI hint. |
| R9 | Swift/C++ interop is recursive across the SwiftPM dep graph and brittle. | Medium | Author our own C ABI shim (`llama_speculative.h`) to keep the Swift surface pure C. Never enable `interoperabilityMode(.Cxx)` on the consumer target. |
| R10 | Rebasing our patches onto fast-moving upstream. | Medium | Keep patch set tiny (six commits per §4.2). Submit (1) and (6) upstream so they leave the patch set entirely. |
| R11 | App Store binary validation issues with bundled `.metallib`. | Low | Already mitigated by `GGML_METAL_EMBED_LIBRARY=ON`. If signing complains, also strip the iphonesimulator slice before validation (whisper.cpp #3046 workaround). |
| R12 | No public llama.cpp benchmark for Gemma 4 E2B on iPhone 17 Pro yet. | Certain | Establish our own number Day 4, publish it. Closest references: A19 Pro TinyLlama-1B-Q4_0 = 87 tok/s, Phi-2-3B-Q4_0 = 36 tok/s (`discussion/4508`). Gemma 4 E2B (~5 B raw / 2.3 B effective) should sit near Phi-2-3B; speculative needed to hit 80–130 tok/s. |
| R13 | Tree-attention sampling not in upstream EAGLE-3 patch. | High for hitting top-end of perf target | Ship 2-wide chain spec first (no tree), measure acceptance rate; only invest in tree once we've squeezed chain. |

---

## 7. References

- llama.cpp main repo: https://github.com/ggml-org/llama.cpp
- iOS xcframework script: https://github.com/ggml-org/llama.cpp/blob/master/build-xcframework.sh
- `examples/llama.swiftui` (source template for our Swift wrapper): https://github.com/ggml-org/llama.cpp/tree/master/examples/llama.swiftui
- iOS deployment discussion #4423: https://github.com/ggml-org/llama.cpp/discussions/4423
- A-series performance discussion #4508 (A17/A18/A19 numbers): https://github.com/ggml-org/llama.cpp/discussions/4508
- Speculative decoding docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- DeepWiki, Speculative Decoding (`common_speculative_*` API map): https://deepwiki.com/ggml-org/llama.cpp/8.3-speculative-decoding
- EAGLE-3 implementation PR #18039: https://github.com/ggml-org/llama.cpp/pull/18039
- EAGLE-3 discussion #15902: https://github.com/ggml-org/llama.cpp/discussions/15902
- EAGLE-3 follow-up issue #18301: https://github.com/ggml-org/llama.cpp/issues/18301
- Gemma 4 model card (E2B GGUF): https://huggingface.co/blog/gemma4 and https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF
- Gemma 4 Apple Metal perf regression #21655: https://github.com/ggml-org/llama.cpp/issues/21655
- Build docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- Stanford SpeziLLM XCFramework reference: https://github.com/StanfordBDHG/llama.cpp
- SwiftLlama wrapper (rejected): https://github.com/ShenghaiWang/SwiftLlama
- LocalLLMClient wrapper (rejected): https://github.com/tattn/LocalLLMClient
- llama-cpp-swift wrapper (rejected): https://github.com/srgtuszy/llama-cpp-swift
- mattt/llama.swift (rejected): https://github.com/mattt/llama.swift
- mlx-swift LLM harness pattern reference: https://github.com/ml-explore/mlx-swift-examples
- jackyoustra.com llama on iOS gotchas: https://www.jackyoustra.com/blog/llama-ios
- Apple residency sets / Metal 4 (WWDC25 #205): https://developer.apple.com/videos/play/wwdc2025/205/
- SwiftPM binaryTarget docs: https://developer.apple.com/documentation/xcode/distributing-binary-frameworks-as-swift-packages
- whisper.cpp App Store validation issue #3046 (metallib): https://github.com/ggml-org/whisper.cpp/issues/3046
- App Store metal embed validation issue #14954: https://github.com/ggml-org/llama.cpp/issues/14954
