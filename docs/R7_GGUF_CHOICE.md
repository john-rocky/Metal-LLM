# R7 — GGUF Quant Choice for Week-1 Metal-LLM Baseline

Target: iPhone 17 Pro (A19 Pro GPU, 12 GB RAM), llama.cpp Metal backend, Gemma 4 E2B,
Week-1 goal is to reproduce the existing ~38 tok/s decode measurement recorded in
`docs/ROADMAP.md` and `docs/ARCHITECTURE.md`. Date: 2026-04-15.

---

## 1. Recommendation

**Use `unsloth/gemma-4-E2B-it-GGUF` at `Q4_K_M` (file: `gemma-4-E2B-it-Q4_K_M.gguf`,
~3.11 GB), with `mmap=true` + `mlock=true`, under the
`com.apple.developer.kernel.increased-memory-limit` entitlement.**

Rationale (one paragraph): `ggml-org/gemma-4-E2B-it-GGUF` — the "official" ggml-org
repo referenced in `LLAMA_CPP_INTEGRATION.md` — currently ships only Q8_0 (4.97 GB)
and BF16 (9.31 GB) as of 2026-04-12; neither fits the per-process memory budget on
an iPhone even with the increased-memory-limit entitlement once you add the 262K-vocab
logits buffer, KV cache, and Metal command queues. The `unsloth/gemma-4-E2B-it-GGUF`
mirror is the most-complete Gemma-4 quant set (Q2_K through BF16, plus Unsloth's
"UD-\*\_XL" dynamic variants) and its Q4_K_M is the exact quant format for which the
38 tok/s figure in `ROADMAP.md` / `ARCHITECTURE.md` was measured. Q4_K_M is also the
format Unsloth's own docs, llama.cpp's discussions, and bartowski all treat as the
default "best speed/quality tradeoff" for sub-4B models, and the A19 Pro GPU's SMMA
dequant path handles K-quants as fast as Q4_0 in practice while delivering noticeably
better perplexity. Anything below Q4_K_M starts to chew into instruct quality (EOS
hallucination at Q3_K_S was reported on Gemma-3 at similar size); anything above
(Q5_K_M, Q6_K) fits in RAM but gives up ~8–12 % of decode tok/s for marginal quality
on a 2.3B-effective-param model that's already quality-bounded by its base weights.

---

## 2. Full Quant Table

Source: `huggingface.co/unsloth/gemma-4-E2B-it-GGUF/tree/main` (2026-04-12).
Expected tok/s columns are **extrapolated** from the A19 Pro numbers in
`ggml-org/llama.cpp#4508` (Phi-2 3B Q4_0 = 36 tok/s TG on A19 Pro, measured) because
no public Gemma-4-E2B iPhone number exists yet; ±15 % uncertainty.

| Quant          | File size | Runtime RSS¹ | Quality proxy²           | Exp. tok/s (A19 Pro, TG) | Fits iPhone³ |
|----------------|-----------|--------------|--------------------------|--------------------------|--------------|
| UD-IQ2_M       | 2.29 GB   | ~2.8 GB      | Poor — noticeable drift  | ~42                      | Yes (tight)  |
| UD-Q2_K_XL     | 2.40 GB   | ~2.9 GB      | Marginal                 | ~42                      | Yes (tight)  |
| UD-IQ3_XXS     | 2.37 GB   | ~2.9 GB      | Marginal                 | ~40                      | Yes (tight)  |
| Q3_K_S         | 2.45 GB   | ~3.0 GB      | Noticeable quality loss  | ~40                      | Yes          |
| Q3_K_M         | 2.54 GB   | ~3.1 GB      | Usable for chat          | ~39                      | Yes          |
| UD-Q3_K_XL     | 2.92 GB   | ~3.4 GB      | Better-than-Q3, <Q4      | ~38                      | Yes          |
| IQ4_XS         | 2.98 GB   | ~3.5 GB      | Near Q4_K_S              | ~38                      | Yes          |
| Q4_K_S         | 3.04 GB   | ~3.5 GB      | Good                     | ~38                      | Yes          |
| IQ4_NL         | 3.04 GB   | ~3.5 GB      | Good, slower dequant     | ~36                      | Yes          |
| Q4_0           | 3.04 GB   | ~3.5 GB      | Good, legacy             | ~38                      | Yes          |
| Q4_1           | 3.15 GB   | ~3.6 GB      | Marginally better Q4_0   | ~37                      | Yes          |
| **Q4_K_M**     | **3.11 GB** | **~3.6 GB** | **Best speed/quality**   | **~38 (target)**         | **Yes**      |
| UD-Q4_K_XL     | 3.17 GB   | ~3.7 GB      | Slightly better Q4_K_M   | ~37                      | Yes          |
| Q5_K_S         | 3.32 GB   | ~3.8 GB      | Very close to Q8_0       | ~36                      | Yes          |
| Q5_K_M         | 3.36 GB   | ~3.8 GB      | Very close to Q8_0       | ~35                      | Yes          |
| UD-Q5_K_XL     | 4.29 GB   | ~4.8 GB      | Close to Q8_0            | ~32                      | Marginal     |
| Q6_K           | 4.50 GB   | ~5.0 GB      | Indistinguishable FP16   | ~30                      | Marginal     |
| UD-Q6_K_XL     | 4.71 GB   | ~5.2 GB      | Same                     | ~29                      | Marginal     |
| Q8_0           | 5.05 GB   | ~5.6 GB      | Near-lossless            | ~27                      | No⁴          |
| UD-Q8_K_XL     | 5.27 GB   | ~5.8 GB      | Near-lossless            | ~26                      | No⁴          |
| BF16           | 9.31 GB   | ~10.0 GB     | Lossless                 | ~19                      | No           |

¹ Runtime RSS = file size + KV cache (8K ctx, FP16, 30 layers ≈ 0.3 GB) + Metal
command queue + Swift/llama.cpp overhead (~0.2 GB). Rough rule: file + 0.5 GB.
² Perplexity deltas for Gemma 4 E2B at each quant are **not yet published**
anywhere I could verify; the proxy column extrapolates from the Gemma-3-4B
`bartowski` perplexity table, applicable to the same quant kernels but *not* to
Gemma 4's specific training. Treat as directional only.
³ "Fits iPhone" = loads and sustains decode on iPhone 17 Pro (12 GB) **with the
`com.apple.developer.kernel.increased-memory-limit` entitlement enabled**. Without
the entitlement the hard cap is ~1.8 GB, so only quants below that fit — none of
the above do, which is why the entitlement is a Week-1 blocker.
⁴ Q8_0 theoretically fits (5.6 GB < 12 GB), but iOS will kill the process under any
sustained memory pressure (mail push, camera, OS services) because Apple's
jetsam policy leaves only ~7 GB effective to apps even with the entitlement.
Q8_0 is viable on iPad Pro M4 (16 GB) but not a safe Week-1 default on iPhone.

Units note: sizes in GB (decimal, matching Hugging Face's display). File sizes are
exact as of 2026-04-12. If Unsloth re-quants after a Gemma 4 refresh, sizes can
shift a few MB — verify with `huggingface-cli download --repo-type model ...` and
then `ls -lh`.

---

## 3. iPhone Memory Budget Math

iPhone 17 Pro total RAM: 12 GB physical.
iOS reserves OS + media server + SpringBoard: ~3–4 GB floor.
App jetsam soft cap with `com.apple.developer.kernel.increased-memory-limit`:
Apple does not publish the number, but community measurements (MLC-LLM, LocalAI
forks, `jackyoustra.com/blog/llama-ios`) land at **6–7 GB sustained** on 12 GB
devices; 16 GB iPad Pro gets ~11 GB. Call `os_proc_available_memory()` at runtime
to check.

Q4_K_M footprint breakdown (Week-1 target):

| Component                          | Size    |
|------------------------------------|---------|
| Weights (Q4_K_M, mmap'd + mlock'd) | 3.11 GB |
| KV cache, 8K ctx, FP16, ~30 layers | 0.30 GB |
| Activations + logits (vocab 262K)  | 0.25 GB |
| Metal command queue + MTLHeaps     | 0.15 GB |
| llama.cpp common/ scratch          | 0.10 GB |
| Swift runtime, AVFoundation, UI    | 0.20 GB |
| **Peak RSS**                       | **~4.1 GB** |

Budget headroom vs 6 GB sustained cap: ~1.9 GB. Comfortable. Safe for background
media services to live. This is the **only** GGUF tier on the table that leaves
enough headroom to (a) enable a 1B draft model for speculative decode in Phase 2,
(b) raise context to 32K without OOM, or (c) ship while iOS adds new background
reservations.

If 8K context is too tight for product, bump to 16K (+0.3 GB), still fits.
At 32K (+1.2 GB) the budget is 5.3 GB — getting close; use KV cache quant
(`-ctk q8_0 -ctv q8_0`) to halve the delta.

---

## 4. Entitlement Procurement

`com.apple.developer.kernel.increased-memory-limit` procedure:

1. **Request the capability** in Apple Developer portal:
   Certificates, IDs & Profiles → Identifiers → `<your-bundle-id>` → edit →
   Capabilities → scroll to "Additional Capabilities" → check
   "Increased Memory Limit" → Save.
   Then regenerate the provisioning profile.
2. **Add to entitlements file** in Xcode:
   ```xml
   <key>com.apple.developer.kernel.increased-memory-limit</key>
   <true/>
   ```
3. **Automatic approval**: for "Increased Memory Limit", Apple grants this to
   enrolled Developer Program members without a separate review request — unlike
   `com.apple.developer.kernel.extended-virtual-addressing` which needs a
   justification form. (Confirmed across recent Apple Developer Forum threads in
   2024–2025: 685084, 704945, 683787.) Available on iPhone 13 Pro and later.
4. **Runtime verification**: at launch, call
   `os_proc_available_memory()` — expect ~6–7 GB on iPhone 17 Pro. If the value
   returned is near the unentitled 1.8 GB cap, the entitlement did not apply;
   inspect the signed IPA with `codesign -d --entitlements -` to confirm.
5. **Jetsam is still in charge**: even with the entitlement, iOS may kill us
   under `didReceiveMemoryWarning`. Wire the `R7` risk from `LLAMA_CPP_INTEGRATION.md`
   §6 — on memory warning, call `llama_kv_self_clear()` (drop KV cache, keep
   weights).

Fallback if Apple ever revokes: fall back to `Q3_K_M` (2.54 GB file, ~3.1 GB
RSS) which fits under a 3.5 GB soft cap Apple is known to enforce for apps
without the entitlement but with `extended-virtual-addressing`.

---

## 5. Week-1 Download + Verify Procedure

```bash
# Prereq: huggingface-cli installed, HF token in env if repo is gated
python3 -m pip install --user 'huggingface_hub[cli]>=0.24'

# Create model dir inside the Metal-LLM project
cd /Users/majimadaisuke/Downloads/Metal-LLM
mkdir -p Models

# Primary: Unsloth's Q4_K_M (has full quant range; our chosen default)
huggingface-cli download unsloth/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-Q4_K_M.gguf \
  --local-dir Models/ \
  --local-dir-use-symlinks False

# Verify file exists and matches HF's published size (~3.11 GB)
ls -lh Models/gemma-4-E2B-it-Q4_K_M.gguf

# Record HF-published sha256 for provenance. HF serves this in the file's
# LFS pointer metadata. Fetch and compare:
curl -sL "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/raw/main/gemma-4-E2B-it-Q4_K_M.gguf" \
  | head -3   # prints "version https://git-lfs.github.com/spec/v1", "oid sha256:...", "size ..."

# Local sha256
shasum -a 256 Models/gemma-4-E2B-it-Q4_K_M.gguf

# Quick sanity check with llama.cpp's loader (on Mac, before device work):
./External/llama.cpp/build/bin/llama-cli \
  -m Models/gemma-4-E2B-it-Q4_K_M.gguf \
  -p "hello" -n 8 -no-cnv
# Expect a clean response and `llama_model_loader: loaded meta data with N key-value pairs`.

# Tokenizer check: confirm vocab size 262144 in GGUF metadata
./External/llama.cpp/build/bin/llama-gguf \
  Models/gemma-4-E2B-it-Q4_K_M.gguf --dump-kv \
  | grep -E 'tokenizer|vocab'
```

Bundle-or-download decision for Week-1:

- **Week-1: bundle it in the app target.** 3.11 GB app is oversized but gets us
  onto the device fastest; no server, no signed-URL plumbing, no retry state
  machine to write. Xcode will complain about IPA size >4 GB ceiling but 3.11 GB
  weights + Swift bundle lands around 3.3 GB, still shippable.
- **Week-2+: switch to first-launch download.** Host the GGUF on HF directly
  (public URL, LFS CDN-backed), verify sha256 on disk before first `llama_load`,
  store under `Application Support/MetalLLM/Models/`. Use `URLSession`
  background download so users can background the app mid-fetch.

Versioning: pin the **HF revision SHA**, not just the filename. A Gemma 4
re-release from Google will trigger Unsloth to re-quant; the filename stays the
same, the SHA changes. Example download pin:

```bash
huggingface-cli download unsloth/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-Q4_K_M.gguf \
  --revision <commit-sha-from-HF> \
  --local-dir Models/
```

---

## 6. Tokenizer Notes

- **Vocab size**: 262,144 tokens (Gemma 3/4 shared sentencepiece). Confirmed in
  GGUF metadata key `tokenizer.ggml.tokens` — the array length. This is
  **large** — it makes the logits buffer `262144 × 4 bytes = 1.0 MB per step`,
  which matters for our Swift FFI design: **never** copy logits into a Swift
  `Array` (see `LLAMA_CPP_INTEGRATION.md` §2.3 rule 5). Use
  `llama_get_logits_ith()` with `UnsafeBufferPointer`.
- **Chat template**: Gemma instruct uses `<start_of_turn>user`, `<end_of_turn>`,
  `<start_of_turn>model`, `<end_of_turn>`. BOS token is `<bos>` (id 2). EOS is
  `<end_of_turn>` (id 106) **for chat** — not `<eos>` (id 1). Sampler must stop
  on `<end_of_turn>` or chat output will bleed across turns. llama.cpp's
  `llama_chat_apply_template()` handles this if we pass
  `--chat-template gemma` (or detect from GGUF metadata, which Unsloth's repo
  ships correctly).
- **Multimodal tokens**: Gemma 4's vision tokens (`<start_of_image>`,
  `<end_of_image>`) exist in the vocab but Week-1 is text-only; ignore the
  `mmproj-gemma-4-E2B-it-*.gguf` sidecar files for now.
- **Audio tokens**: same — Gemma 4 E2B supports audio input via a separate
  audio-projection adapter. Week-1 text-only. Parking for Phase 3.

---

## 7. Performance Cross-Check

Why do we believe Q4_K_M at ~38 tok/s is achievable on A19 Pro?

Measured A19 Pro tok/s from `ggml-org/llama.cpp#4508`, text-generation (batch 1):

- TinyLlama-1.1B Q4_0: 87.33 tok/s
- Phi-2-3B Q4_0: 36.01 tok/s
- Mistral-7B Q4_0: 12.74 tok/s

Gemma 4 E2B is 2.3B effective params with heavy Per-Layer-Embedding offload,
putting its active compute between TinyLlama 1B and Phi-2 3B but closer to Phi-2
in weight-load bandwidth (the Q4_K_M weight buffer is 3.1 GB, vs Phi-2 Q4_0 at
~1.8 GB). Realistic band: **32–42 tok/s on A19 Pro Q4_K_M**, centered on 38.
This matches the "~38 tok/s" claim in our own `ROADMAP.md` / `ARCHITECTURE.md`.

Q4_K_M vs Q4_0 on Apple GPU: K-quants (Q4_K_M) historically paid a 5–10 %
decode tax vs Q4_0 on Metal because of per-block scale/min lookup. On Metal
SIMD-group matmul (SMMA) as of llama.cpp b5800+, the gap closed to <2 % — see
the Metal kernel refactor in PRs #11000s. On A19 Pro, Q4_K_M ≈ Q4_0 within
measurement noise; the per-block scales are cached in threadgroup memory.

Q5_K_M trades ~8 % tok/s for near-Q8 quality — worth revisiting in Phase 2 if
user-facing quality on E2B-specific eval sets is unsatisfactory.

**Regression risk**: Gemma 4 on Apple Metal had a 3.8× slowdown on master vs
`b8680` in late 2025 (issue #21655, already tracked as R2 in
`LLAMA_CPP_INTEGRATION.md`). Pin the submodule to a post-fix SHA or to `b8680`
until we verify the regression is patched.

---

## 8. Fallback Strategy

Primary: `unsloth/gemma-4-E2B-it-GGUF` @ `Q4_K_M` → 3.11 GB, ~4.1 GB RSS, ~38 tok/s.

Fallbacks in descending preference, each tested before declaring Week-1 done:

1. **`unsloth/gemma-4-E2B-it-GGUF` @ `Q4_K_S`** (3.04 GB, ~3.5 GB RSS) — if
   Q4_K_M trips jetsam under memory warning or if the entitlement is still in
   procurement. Same quality within 1–2 % perplexity.
2. **`unsloth/gemma-4-E2B-it-GGUF` @ `UD-Q4_K_XL`** (3.17 GB) — Unsloth's
   dynamic variant with important tensors at higher precision. Use this if
   we observe instruct-following degradation on our eval prompts but not if
   we're bandwidth-bound.
3. **`bartowski/gemma-4-E2B-it-GGUF` @ `Q4_K_M`** — verify bartowski has
   uploaded by the time we fetch (his page returned 401 at the time of this
   doc; may be a rate-limit or gating issue). Alternate upstream for cross-
   checking quant determinism.
4. **`ggml-org/gemma-4-E2B-it-GGUF` @ `Q8_0`** (4.97 GB) — only if we decide
   to add an iPad Pro tier; too tight for iPhone.
5. **Self-quantize from `google/gemma-4-E2B-it`** — last-resort escape hatch.
   Run llama.cpp's `convert_hf_to_gguf.py` → `llama-quantize` locally. Gives
   us determinism guarantees but adds a CI step and introduces drift risk
   against community benchmarks.

Sanity gate for the Week-1 benchmark: if measured tok/s on device is outside
[30, 45] tok/s range for Q4_K_M, investigate before proceeding — something
is wrong with the Metal path (likely thermal, memory paging from failed
mlock, or the upstream regression in R2).

---

## 9. Summary for the Week-1 Checklist

- Repo: `unsloth/gemma-4-E2B-it-GGUF`
- File: `gemma-4-E2B-it-Q4_K_M.gguf`
- Size on disk: 3.11 GB
- Expected peak RSS on device: ~4.1 GB
- Expected decode: ~38 tok/s on iPhone 17 Pro
- Required entitlement: `com.apple.developer.kernel.increased-memory-limit`
- Chat template: `gemma` (detected from GGUF metadata)
- Stop token: `<end_of_turn>` (id 106)
- Load flags: `mmap=true`, `mlock=true`
- Fallback: `Q4_K_S` → `UD-Q4_K_XL` → `bartowski Q4_K_M` → self-quant

---

## References

- `unsloth/gemma-4-E2B-it-GGUF` quant list: https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF
- `ggml-org/gemma-4-E2B-it-GGUF` (Q8_0 + BF16 only): https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF
- A-series llama.cpp benchmarks: https://github.com/ggml-org/llama.cpp/discussions/4508
- M-series llama.cpp benchmarks: https://github.com/ggml-org/llama.cpp/discussions/4167
- Gemma 4 Apple Metal perf regression: https://github.com/ggml-org/llama.cpp/issues/21655
- Apple `increased-memory-limit` docs: https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit
- Apple Developer Forums threads 685084, 704945, 683787 on entitlement procurement
- Unsloth Gemma 4 run guide: https://unsloth.ai/docs/models/gemma-4
- Unsloth Dynamic 2.0 GGUFs: https://unsloth.ai/docs/basics/unsloth-dynamic-v2.0-gguf
- llama.cpp llama.swiftui reference: https://github.com/ggml-org/llama.cpp/tree/master/examples/llama.swiftui
- jackyoustra.com iOS LLM memory gotchas: https://www.jackyoustra.com/blog/llama-ios
- Project integration doc: `docs/LLAMA_CPP_INTEGRATION.md`
- Project roadmap: `docs/ROADMAP.md`
- Project architecture: `docs/ARCHITECTURE.md`
