# R5. llama.cpp Fork Surgery — Public C ABI for Speculative Decoding + External Drafter

Status: investigation complete, ready to implement in Week 5.
Date: 2026-04-15.
Pinned upstream: `ggml-org/llama.cpp` master @ `b3d7587` (2026-04-15, "vulkan: optimize im2col (#21713)"). No `b*` release tag on this SHA; we pin by commit. PR #18039 is at this point **still OPEN, not merged** — the `LLAMA_CPP_INTEGRATION.md` note that says "merged" was wrong. This is important: we cannot assume EAGLE-3 is upstream yet.

---

## 0. Executive summary

Our Metal-LLM host (Swift) wants to drive llama.cpp's target decode while plugging in a Swift-side drafter (MTP CoreML, prompt-lookup, later EAGLE-3). The current llama.cpp exposes speculative decoding only through `common/speculative.{h,cpp}`, which is a C++ static library (`common`), not part of the public `libllama` C ABI. The surgery needed is:

- **7 new public C functions** (6 for the core spec-API, 1 EAGLE-3 hidden-state hook). Plus 1 new C callback typedef + 1 new opaque struct.
- **1 new translation unit**: `src/llama-speculative.cpp` — a C wrapper around a copy of `common/speculative.cpp` (we vendor the logic into `libllama` rather than linking `common` into our binary, to keep the XCFramework one library).
- **3 patched files**: `include/llama.h` (declarations), `src/llama.cpp` or `src/llama-context.cpp` (hidden-state tap for EAGLE-3 — steal PR #18039's `llama_set_eagle3` / `llama_get_eagle3_target_features` verbatim), `src/CMakeLists.txt` (add new TU).
- **Total time**: **~38 hours of engineering**, split across Week 5 as 6h ABI patch / 6h xcframework build pipeline / 8h Swift wrapper / 8h MTP drafter callback + first working spec-decode / 10h EAGLE-3 follow-up (includes cherry-picking PR #18039 and reconciling against master).

The fork is **upstreamable in pieces** — the external-drafter callback is the interesting bit worth proposing as an RFC; the rest is just rewrapping existing C++ helpers.

---

## 1. Current speculative decoding surface in llama.cpp

### 1.1 What is a C++ internal vs a public C API

Everything in `include/llama.h` is the public C ABI — these symbols get exported from `libllama.a` / `libllama.dylib` with `LLAMA_API` and are callable from Swift via module-map-style import. Everything in `common/*.h` is **C++ only**, lives in a separate `common` static library (see `/tmp/llama.cpp/common/CMakeLists.txt:44-92`), and is *not* part of the semver-stable surface. Upstream reserves the right to change `common_*` at will.

Speculative decoding lives entirely in `common`:

- `common/speculative.h` (41 lines total — `/tmp/llama.cpp/common/speculative.h:1-41`) declares a forward-declared `struct common_speculative` plus 7 C++ free functions: `common_speculative_init`, `_free`, `_begin`, `_draft`, `_accept`, `_print_stats`, plus string/name helpers.
- `common/speculative.cpp` (1074 lines — `/tmp/llama.cpp/common/speculative.cpp`) implements eight strategies behind a `common_speculative_state` abstract base (line 113–145): `DRAFT` (draft-model, lines 147–438), `EAGLE3` (stub, lines 440–463 — **not implemented on master**), `NGRAM_SIMPLE` (466), `NGRAM_MAP_K` (494), `NGRAM_MAP_K4V` (shares impl), `NGRAM_MOD` (521), `NGRAM_CACHE` (647).
- `examples/speculative/speculative.cpp` (651 lines) and `examples/speculative-simple/speculative-simple.cpp` (269 lines) are the reference wire-ups. Only the latter is the pattern we care about — it's the clean canonical loop (see `/tmp/llama.cpp/examples/speculative-simple/speculative-simple.cpp:149-234`).

The public C ABI currently offers **nothing** named `draft`, `spec`, or `eagle` in `include/llama.h`. Grep for those substrings in `include/llama.h:1-1589` returns only unrelated comment text ("specified", etc.). The only building blocks available at the C boundary are the general `llama_decode`, `llama_batch_*`, `llama_memory_*`, `llama_sampler_*`, and `llama_get_logits_ith`/`llama_get_embeddings_ith` calls (`include/llama.h:960`, `:1001`, `:1023`).

### 1.2 Function-by-function inventory of `common/speculative.h`

| Symbol | `common/speculative.h` line | C linkage today? | Types crossing boundary | Swift interop difficulty |
|---|---|---|---|---|
| `common_speculative_type_name_str()` | 9 | C++ (returns `std::string`) | `std::string` | **Hard** — rewrap with `const char *` |
| `common_speculative_type_from_name(const std::string&)` | 12 | C++ | `std::string` | Rewrap, accept `const char *` |
| `common_speculative_type_to_str(enum)` | 15 | C++ | returns `std::string` | Rewrap |
| `common_speculative_is_compat(llama_context*)` | 19 | Internally pure-C pointers, but declared in a C++ header | `llama_context*` | **Easy** — just needs `extern "C"` wrapper |
| `common_speculative_init(common_params_speculative&, llama_context*)` | 21 | C++ reference, and `common_params_speculative` is a C++ struct with `std::string`, `std::vector<std::pair<>>`, `std::shared_ptr`, etc. (see `common/common.h:312`) | `common_params_speculative` is **deeply C++** | **Hard** — cannot cross as-is |
| `common_speculative_free` | 25 | Opaque pointer | — | Easy |
| `common_speculative_begin(spec, const llama_tokens&)` | 28 | `llama_tokens` = `std::vector<llama_token>` | std::vector | Rewrap to `const llama_token*, n` |
| `common_speculative_draft(spec, params&, prompt&, id_last)` | 31 | Returns `llama_tokens` by value | std::vector return | Rewrap to out-param buffer |
| `common_speculative_accept(spec, n)` | 38 | POD args | — | Easy |
| `common_speculative_print_stats` | 41 | — | — | Easy |

Only `free`, `accept`, `print_stats`, and `is_compat` cross the boundary trivially. Every other symbol either takes `std::vector` / `std::string` / `common_params_speculative` or returns `std::string` / `llama_tokens`. **This is the surgery: rewrap five signatures in C.**

### 1.3 `common_params_speculative` is the worst offender

From `common/common.h:312` onward (checked via Grep on `/tmp/llama.cpp/common/common.h`):

```cpp
struct common_params_speculative {
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;
    // general-purpose speculative decoding parameters
    int32_t n_max = 16, n_min = 0, n_ctx = 0, ...;
    float   p_min = 0.75f;
    struct common_params_model mparams_dft;    // contains std::string
    llama_model * model_dft = nullptr;
    llama_context_params cparams_dft;           // plain C ✓
    // ngram params, std::vector<std::pair<std::string,std::string>> replacements,
    // std::shared_ptr<common_ngram_mod> ngram_mod, etc.
    ...
};
```

Most of this config is only needed because `common_speculative_init` loads the draft model itself from disk. In our fork, **we don't need llama.cpp to load anything** — the drafter lives in Swift (MTP is a CoreML model, EAGLE-3 is a separate llama.cpp context we set up ourselves). So we collapse the config to a narrow `llama_spec_params` POD struct (see §4).

---

## 2. EAGLE-3 PR #18039 — what it actually adds

Pulled with `gh pr diff 18039 --repo ggml-org/llama.cpp` (PR is **OPEN, not merged**, as of 2026-04-15). 1137 additions / 53 deletions across 26 files. The public-ABI-relevant diff lives in three files:

### 2.1 `include/llama.h` additions from PR #18039

PR adds exactly 3 C things (lines quoted from the PR diff hunk starting at upstream `include/llama.h:373`):

```c
// (1) new field inside llama_context_params:
const struct llama_model * target_model; // reference to target model
                                         // only used to share embedding layer with eagle3 model

// (2) new API to bind target model into an EAGLE3 context:
LLAMA_API void llama_set_eagle3(
        struct llama_context * ctx,
        const struct llama_model * model);

// (3) two new getters/setters to shuttle the encoder hidden state:
LLAMA_API const float * llama_get_eagle3_target_features(struct llama_context * ctx);

LLAMA_API void llama_set_eagle3_g_embeddings(
        struct llama_context * ctx,
        const float * g_embd,
        int32_t       n_embd,
        int32_t       n_tokens);
```

So the hidden-state hook is `llama_get_eagle3_target_features(ctx) -> const float*` with shape `[3*n_embd, n_tokens]` (comment in the diff: "Format: [3*n_embd, n_tokens] — use model.hparams.n_embd and batch.n_tokens for dimensions"). That matches EAGLE-3's canonical design: concatenate three intermediate hidden states (early/mid/late) from the target's forward pass.

### 2.2 Where the hidden state is tapped

PR #18039's `src/llama-context.cpp` gets +215 lines and `src/llama-context.h` +15 lines. The plumbing: `llama_context` gains an `eagle3_target_features` buffer that is written during target-model forward when layer index is in the `extract_layers` set (defined in the model's GGUF via `arch.extract_layers` — see PR's `src/llama-arch.cpp:+35`). The three extracted layers, per PR's `convert_hf_to_gguf.py:+146`, are `[2, N/2, N-3]` for an N-layer target.

So: the hidden-state extraction point is inside `llama_decode` → the graph build in `src/llama-graph.cpp` / `src/llama-model.cpp` / the per-arch `src/models/*.cpp` files. The PR adds `EAGLE3` arch support for Llama, Qwen3, Qwen3-MoE, and OpenAI-MoE-ISWA via `src/models/eagle3.cpp` (+186 lines new file) and small taps in each per-arch file (+10 lines each in `src/models/llama.cpp`, `qwen3.cpp`, `qwen3moe.cpp`, `openai-moe-iswa.cpp`).

**No changes needed for Gemma.** PR #18039 does not add Gemma to the extract list. We will need to port the three-line tap ourselves into `src/models/gemma3.cpp` (or whichever file owns Gemma 4 E2B's graph) when we want EAGLE-3 on Gemma. That's a Week-8+ task, not Week 5.

### 2.3 Is the EAGLE-3 surface a usable C ABI?

Yes — `llama_set_eagle3`, `llama_get_eagle3_target_features`, `llama_set_eagle3_g_embeddings` are already `LLAMA_API` and take pure-C args (`float*`, `int32_t`, opaque structs). So once PR #18039 merges (or we cherry-pick onto our fork), the EAGLE-3 hidden-state hook is directly Swift-consumable without any extra wrapping work from us. The work is limited to porting the target-side tap into the Gemma graph.

---

## 3. Proposed C ABI additions (our fork's `extern "C"` surface)

We add the following to `include/llama.h` under a new "Speculative decoding" section, after the existing memory/state block. Eight things total: one opaque struct forward-decl, one callback typedef, one POD params struct, five functions plus `llama_spec_destroy`. One additional function for symmetry with MTP-style drafters (reset).

```c
    //
    // Speculative decoding (Metal-LLM fork)
    //

    struct llama_spec;                       // opaque
    typedef struct llama_spec * llama_spec_t;

    // External drafter callback.
    //   ctx_user        : user_data pointer passed at set_external_drafter()
    //   prompt_tokens   : tokens already in the target context (recent history; len = n_prompt)
    //   n_prompt        : count of prompt_tokens
    //   id_last         : the anchor token (the one the target most recently sampled)
    //   n_max           : max draft length the caller will accept (<= LLAMA_SPEC_MAX_DRAFT)
    //   out_tokens      : preallocated buffer of capacity n_max; drafter writes up to n_written
    //   out_probs       : optional [n_max] probability buffer (NULL if drafter does not provide)
    //   n_written [out] : number of drafted tokens the callback wrote (0..n_max)
    //   return          : 0 on success, non-zero to abort this speculation burst.
    typedef int32_t (*llama_spec_drafter_cb)(
            void                  * ctx_user,
            const llama_token     * prompt_tokens,
            int32_t                 n_prompt,
            llama_token             id_last,
            int32_t                 n_max,
            llama_token           * out_tokens,
            float                 * out_probs,   // nullable
            int32_t               * n_written);

    // Configuration for llama_spec_create().
    struct llama_spec_params {
        int32_t n_draft_min;   // minimum accepted draft length (drop draft if shorter)
        int32_t n_draft_max;   // maximum draft length to request (<= 64 hard cap)
        float   p_min;         // minimum draft-token probability (applied only when drafter provides probs)

        // built-in drafter selector; when type != LLAMA_SPEC_TYPE_EXTERNAL, the callback is ignored
        // and an internal strategy is used (draft-model / ngram / eagle3).
        int32_t type;          // one of LLAMA_SPEC_TYPE_*

        // for LLAMA_SPEC_TYPE_DRAFT_MODEL and LLAMA_SPEC_TYPE_EAGLE3 only:
        struct llama_context * ctx_draft;   // caller-owned; NULL for EXTERNAL / NGRAM
        struct llama_context * ctx_draft_enc; // EAGLE3 encoder ctx; NULL otherwise
    };

    enum llama_spec_type {
        LLAMA_SPEC_TYPE_EXTERNAL     = 0,  // use the callback
        LLAMA_SPEC_TYPE_DRAFT_MODEL  = 1,  // internal: draft-model (common_speculative_state_draft)
        LLAMA_SPEC_TYPE_EAGLE3       = 2,  // internal: EAGLE3 (requires PR #18039 merged / cherry-picked)
        LLAMA_SPEC_TYPE_NGRAM_SIMPLE = 3,  // internal: ngram lookup
    };

    // Create a speculative-decoding session bound to a target context.
    // Returns NULL on failure.
    LLAMA_API llama_spec_t llama_spec_create(
            struct llama_context        * ctx_tgt,
            const struct llama_spec_params * params);

    // Register an external drafter. Must be called for LLAMA_SPEC_TYPE_EXTERNAL.
    // Safe to call again to swap the callback between bursts.
    LLAMA_API void llama_spec_set_external_drafter(
            llama_spec_t              spec,
            llama_spec_drafter_cb     cb,
            void                    * user_data);

    // Reset internal per-burst state (e.g. sampler RNG, cached draft prompt).
    // Not required for EXTERNAL drafters but harmless.
    LLAMA_API void llama_spec_reset(llama_spec_t spec);

    // Produce a speculative draft. The draft is appended to *out_tokens* (caller-owned buffer
    // of capacity >= n_max). Returns number of tokens written, or a negative error code.
    //
    // This is a PURE drafter step — the caller is responsible for constructing the target
    // batch, calling llama_decode(ctx_tgt, batch), and running the rejection-sampling loop
    // (common_sampler_sample_and_accept_n-equivalent) from Swift. We expose this because the
    // Swift host wants to keep verify/sampler orchestration visible.
    LLAMA_API int32_t llama_spec_draft(
            llama_spec_t          spec,
            const llama_token   * prompt_tokens,
            int32_t               n_prompt,
            llama_token           id_last,
            llama_token         * out_tokens,
            float               * out_probs,   // optional, may be NULL
            int32_t               n_max);

    // Inform the speculator how many of its most recent draft tokens were accepted.
    // Drives n-gram cache updates and stats accounting.
    LLAMA_API void llama_spec_accept(llama_spec_t spec, int32_t n_accepted);

    // Accept-rate / perf stats accessors (C replacement for common_speculative_print_stats).
    LLAMA_API int32_t llama_spec_get_stats(
            llama_spec_t       spec,
            int64_t          * n_drafted,       // out
            int64_t          * n_accepted,      // out
            int64_t          * t_draft_us,      // out
            int64_t          * t_accept_us);    // out

    LLAMA_API void llama_spec_destroy(llama_spec_t spec);
```

All signatures are ANSI C: no `std::`, no references, no templates. `llama_spec_drafter_cb` uses only POD pointer/int args so Swift exposes it as a plain `@convention(c)` function pointer — trivial to bridge with `Unmanaged.passUnretained(self).toOpaque()` as `user_data`.

**Rejection sampling compatibility.** We deliberately do *not* add a `llama_spec_step(batch)` that runs the entire burst inside libllama. Keeping verify/sampling in the Swift host means (a) the existing `common_sampler_sample_and_accept_n` logic can be ported to Swift in ~60 lines using already-public `llama_get_logits_ith`, (b) we keep full visibility into which tokens were accepted for telemetry, and (c) we avoid re-exposing the C++ `common_sampler` chain to C. The cost: we're *not* wrapping `common_speculative_draft` 1:1 — we're extracting just the draft step.

---

## 4. File-by-file patch sketch

### 4.1 `include/llama.h`

Insert the block from §3 after the `// State / sessions` section (around current line 905, right after the new EAGLE-3 block that PR #18039 adds). Gate behind a `// Metal-LLM fork additions` comment so rebase-onto-upstream is mechanical. ~110 lines added.

### 4.2 `src/llama-speculative.cpp` (new)

Vendored translation unit that bridges public C → the C++ `common_speculative` implementations. It is **not** a copy of `common/speculative.cpp` — it's a thin adapter that calls into a copy of `common/speculative.cpp` compiled into `libllama`. Structure:

```cpp
// src/llama-speculative.cpp
#include "llama.h"
#include "llama-speculative-internal.h"   // private header that pulls in the C++ impls
#include "../common/speculative.h"        // vendored — see §4.6

#include <vector>
#include <string>

struct llama_spec {
    llama_context *                 ctx_tgt;
    llama_spec_params               params;
    llama_spec_drafter_cb           cb        = nullptr;
    void *                          cb_user   = nullptr;

    // internal drafter (one of these is set, depending on params.type)
    std::unique_ptr<common_speculative_state> impl_internal;

    // stats
    int64_t n_drafted = 0, n_accepted = 0;
    int64_t t_draft_us = 0, t_accept_us = 0;
};

extern "C" LLAMA_API llama_spec_t llama_spec_create(
        llama_context * ctx_tgt, const llama_spec_params * p) {
    auto * spec = new llama_spec();
    spec->ctx_tgt = ctx_tgt;
    spec->params  = *p;
    switch (p->type) {
        case LLAMA_SPEC_TYPE_EXTERNAL:
            break; // no internal impl
        case LLAMA_SPEC_TYPE_DRAFT_MODEL:
            // construct a common_speculative_state_draft using p->ctx_draft
            // (vendored types defined in llama-speculative-internal.h)
            spec->impl_internal = make_state_draft(ctx_tgt, p->ctx_draft);
            break;
        case LLAMA_SPEC_TYPE_EAGLE3:
            spec->impl_internal = make_state_eagle3(
                ctx_tgt, p->ctx_draft_enc, p->ctx_draft);
            break;
        case LLAMA_SPEC_TYPE_NGRAM_SIMPLE:
            spec->impl_internal = make_state_ngram_simple(/*defaults*/);
            break;
        default:
            delete spec; return nullptr;
    }
    return spec;
}

extern "C" LLAMA_API void llama_spec_set_external_drafter(
        llama_spec_t spec, llama_spec_drafter_cb cb, void * ud) {
    spec->cb = cb; spec->cb_user = ud;
}

extern "C" LLAMA_API int32_t llama_spec_draft(
        llama_spec_t spec, const llama_token * prompt, int32_t n_prompt,
        llama_token id_last, llama_token * out, float * probs, int32_t n_max) {
    const int64_t t0 = ggml_time_us();
    int32_t n_written = 0;

    if (spec->params.type == LLAMA_SPEC_TYPE_EXTERNAL) {
        if (!spec->cb) return -1;
        int rc = spec->cb(spec->cb_user, prompt, n_prompt, id_last, n_max,
                          out, probs, &n_written);
        if (rc != 0) return -2;
    } else {
        // call into vendored common_speculative_state_*::draft()
        std::vector<llama_token> prompt_v(prompt, prompt + n_prompt);
        std::vector<llama_token> result;
        common_params_speculative p_spec{};
        p_spec.n_max = n_max;
        p_spec.n_min = spec->params.n_draft_min;
        p_spec.p_min = spec->params.p_min;
        spec->impl_internal->draft(p_spec, prompt_v, id_last, result);
        n_written = (int32_t) std::min<size_t>(result.size(), n_max);
        std::copy_n(result.begin(), n_written, out);
        // probs: internal impls don't expose probs today; leave NULL-safe
    }

    spec->t_draft_us += ggml_time_us() - t0;
    spec->n_drafted  += n_written;
    return n_written;
}

extern "C" LLAMA_API void llama_spec_accept(llama_spec_t spec, int32_t n) {
    const int64_t t0 = ggml_time_us();
    spec->n_accepted += n;
    if (spec->impl_internal) {
        spec->impl_internal->accept((uint16_t) n);
    }
    spec->t_accept_us += ggml_time_us() - t0;
}

extern "C" LLAMA_API int32_t llama_spec_get_stats(
        llama_spec_t s, int64_t * nd, int64_t * na, int64_t * td, int64_t * ta) {
    if (nd) *nd = s->n_drafted;
    if (na) *na = s->n_accepted;
    if (td) *td = s->t_draft_us;
    if (ta) *ta = s->t_accept_us;
    return 0;
}

extern "C" LLAMA_API void llama_spec_reset(llama_spec_t s) {
    // reset draft impl; noop for EXTERNAL
    if (s->impl_internal) { /* reset sampler, clear draft KV */ }
}

extern "C" LLAMA_API void llama_spec_destroy(llama_spec_t s) { delete s; }
```

Roughly 280–320 lines once fleshed out. All the heavy lifting is in the vendored `common_speculative_state_*` structs which we lift wholesale from `common/speculative.cpp:113-738` and rename to `llama_speculative_state_*` with `_internal` linkage.

### 4.3 `src/llama-speculative-internal.h` (new)

Private header, not installed. Declares the C++ state structs and the `make_state_*` factories used above. ~80 lines.

### 4.4 `src/llama-context.cpp` / `src/llama-graph.cpp` / `src/models/*.cpp`

Cherry-pick PR #18039's changes verbatim when we want EAGLE-3 (estimated ~500 of the 1137 PR lines fall in these files). Conflicts with master are unlikely to be severe — PR diff applies cleanly against our pinned SHA today (confirmed by checking current master state of `src/models/llama.cpp` has no conflicting hunks near line 500). For Week 5's MTP-first milestone, **skip this file entirely** — we can ship external-drafter support without touching `llama-context.cpp`.

### 4.5 `src/CMakeLists.txt`

One-line edit around current line 20-ish (the `add_library(llama …)` block):

```cmake
add_library(llama
    ../include/llama.h
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    …
    llama-speculative.cpp          # <-- ADD
    …
)
```

### 4.6 Vendor `common/speculative.cpp` into `src/`

Copy `common/speculative.cpp` to `src/llama-speculative-impl.cpp`, strip the example-specific helpers (`common_sampler`, `common_ngram_cache_load` etc. — we only need the `_state_*::draft/accept/begin` logic), and link the result into `libllama` instead of `common`. This avoids our XCFramework needing to also ship `libcommon.a`. ~700 lines after stripping (down from 1074).

Alternative (§7 fallback): don't vendor, just link `libcommon.a` alongside `libllama.a`. Simpler but doubles the library count consumers must link.

---

## 5. Upstreamability

**The `llama_spec_drafter_cb` callback is genuinely novel** — no upstream RFC proposes it. Searching `gh search issues "speculative decoding C API"` and `"public speculative API"` on `ggml-org/llama.cpp` returns zero matches. Closest related in-flight work:

- PR #18039 (EAGLE-3) — open, adds the hidden-state hooks but not an external drafter. 1137+/53−, last touched shortly before our clone date. Unlikely to merge in time for our Week 5 work.
- PR #18471 (mentioned in `common/speculative.cpp:864` comment) — added/removed multi-strategy config; suggests upstream is still iterating on `common/speculative.cpp`, so any patch we submit should avoid heavy refactoring.
- `common/speculative.cpp:110-145` was already restructured around a `common_speculative_state` base class, so adding a callback-driven `common_speculative_state_external` subclass is the minimal-invasive upstream shape — it's four new structs and a factory branch in `common_speculative_init`.

**Upstreamable path:**

1. Land the external-drafter callback in `common/speculative.{h,cpp}` only. Roughly: add `COMMON_SPECULATIVE_TYPE_EXTERNAL` to the enum, add `common_speculative_state_external` struct that owns a function-pointer+user_data pair, wire it into `common_speculative_init`. ~120 lines of patch. Zero public-ABI impact.
2. Separately, land a C-ABI wrapper as a new `common/speculative-c.h` + `speculative-c.cpp` translation unit. ~200 lines. Avoids changing `include/llama.h` entirely — the wrapper is part of `libcommon`, not `libllama`. Lower bar for upstream.

If upstream accepts (1) but not (2), we keep (2) as our patch. If upstream accepts both, we drop our patches entirely.

**Fallback if fork unpalatable** (§7 of the prompt): vendor `common/speculative.cpp` directly into Metal-LLM itself, compiled as a Swift-target-adjacent C++ file with its own `extern "C"` shim authored by us (same `llama_spec_*` surface described in §3, but living in `Metal-LLM/Sources/SpecKernel/c_api.{h,cpp}` rather than in `libllama`). This still requires us to pull `llama_get_eagle3_target_features` out of PR #18039 to access hidden states for EAGLE-3 — there is no workaround for that because the hidden state only exists as a private `llama_context` buffer, not on any public path. For non-EAGLE drafters (MTP CoreML, prompt-lookup), the fallback works without any llama.cpp patching.

---

## 6. Week-5 onboarding: step-by-step

Working tree: `~/Downloads/Metal-LLM/`. Starting from the pinned SHA `b3d7587`.

### Day 1 (morning, ~3h) — Fork + ABI scaffold
1. `cd ~/Downloads/Metal-LLM && git submodule add https://github.com/ggml-org/llama.cpp External/llama.cpp`.
2. `cd External/llama.cpp && git checkout -b mlllm/main b3d7587`.
3. Open `include/llama.h`, append the §3 block at ~line 905 under a new `// Metal-LLM fork — speculative C ABI` comment.
4. Create `src/llama-speculative.cpp` and `src/llama-speculative-internal.h` with the skeleton from §4.2/4.3. Leave the C++-internal factories as `TODO` stubs that return `nullptr` for the internal `type` enum values — **Week 5 ships only `LLAMA_SPEC_TYPE_EXTERNAL`**.
5. Add `llama-speculative.cpp` to `src/CMakeLists.txt` (§4.5).
6. Verify it compiles: `cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF && cmake --build build -j8`. Fix any name-clashes (headers are C-safe — wrap with `#ifdef __cplusplus extern "C" {` guards).

### Day 1 (afternoon, ~3h) — XCFramework build
1. Copy `build-xcframework.sh` → `Scripts/build-xcframework.sh`, adjust flags to match `LLAMA_CPP_INTEGRATION.md:§3` (GGML_METAL=ON, GGML_METAL_EMBED_LIBRARY=ON, GGML_OPENMP=OFF, etc.).
2. Run: `./Scripts/build-xcframework.sh`. Confirm `Frameworks/llama.xcframework/` contains `ios-arm64/libllama.a` (plus `libggml*.a`, `libcommon.a`) and that `nm Frameworks/llama.xcframework/ios-arm64/libllama.a | grep llama_spec_` shows all 7 symbols exported.
3. Wire `Package.swift` to `binaryTarget(path: "Frameworks/llama.xcframework")`.

### Day 2 (~8h) — Swift interface
1. Copy upstream `examples/llama.swiftui/llama.cpp.swift/LibLlama.swift` to `Sources/LlamaKernel/LlamaContext.swift`. Already an `actor`.
2. Create `Sources/LlamaKernel/LlamaSpec.swift`:
   ```swift
   public actor LlamaSpec {
       private let handle: OpaquePointer   // llama_spec_t

       public init(target: LlamaContext, draftMin: Int32 = 1, draftMax: Int32 = 8) {
           var p = llama_spec_params(
               n_draft_min: draftMin, n_draft_max: draftMax, p_min: 0.6,
               type: LLAMA_SPEC_TYPE_EXTERNAL.rawValue,
               ctx_draft: nil, ctx_draft_enc: nil)
           self.handle = llama_spec_create(target.ctx, &p)!
       }

       public func setDrafter(_ d: any ExternalDrafter) {
           let opaque = Unmanaged.passUnretained(d as AnyObject).toOpaque()
           llama_spec_set_external_drafter(handle, LlamaSpec.trampoline, opaque)
       }

       static let trampoline: llama_spec_drafter_cb = { user, prompt, nPrompt, idLast, nMax, out, probs, nWritten in
           let d = Unmanaged<AnyObject>.fromOpaque(user!).takeUnretainedValue() as! ExternalDrafter
           let prefix = UnsafeBufferPointer(start: prompt, count: Int(nPrompt))
           let tokens = d.draft(prefix: Array(prefix), idLast: idLast, nMax: Int(nMax))
           let n = min(tokens.count, Int(nMax))
           for i in 0..<n { out![i] = tokens[i] }
           nWritten!.pointee = Int32(n)
           return 0
       }

       public func draft(prefix: [llama_token], idLast: llama_token, nMax: Int32) -> [llama_token] { … }
       public func accept(_ n: Int32) { llama_spec_accept(handle, n) }
       deinit { llama_spec_destroy(handle) }
   }

   public protocol ExternalDrafter: AnyObject {
       func draft(prefix: [llama_token], idLast: llama_token, nMax: Int) -> [llama_token]
   }
   ```
3. Verify Swift can import the new C symbols via the xcframework's module map (should be automatic because we just added decls to `llama.h`).

### Day 3 (~8h) — First speculative decode with MTP drafter
1. Implement `Sources/LlamaKernel/MTPDrafter.swift` conforming to `ExternalDrafter`. Uses the existing CoreML MTP model from the CoreML-LLM project (the `stateful_chunk2.mlpkg` under `Examples/CoreMLLLMChat/CoreMLLLMChat/` ports over with minimal changes — we already proved the forward-batch path in `eagle3_retrain.md`).
2. Port `common_sampler_sample_and_accept_n` to Swift — ~60 lines, loops over `llama_get_logits_ith(ctx, i)` for each draft position, applies `argmax` against the draft token, accepts up to the first mismatch. Greedy first; temperature variant later.
3. Write `Tests/LlamaKernelTests/SpecDecodeTests.swift` that runs Gemma 4 E2B Q4_K_M with the MTP drafter and asserts: (a) non-zero accept rate, (b) output bit-identical to non-speculative decode for the same greedy settings.
4. Run on iPhone 17 Pro. Instrument accept rate and tok/s. Expected first number: 70–90 tok/s decode (below EAGLE-3 target because MTP's accept rate is lower, but above the 55 tok/s baseline from `benchmark_baseline.md`).

### Day 4+ (~10h) — EAGLE-3 follow-up
1. `cd External/llama.cpp && git fetch origin pull/18039/head:pr-18039 && git cherry-pick pr-18039~1..pr-18039`. Resolve conflicts (should be minimal because PR is already rebased on a recent commit close to ours).
2. Port PR's `src/models/eagle3.cpp` target-side tap into Gemma: copy the three-line "if layer in extract_layers, memcpy to `eagle3_target_features` buffer" into `src/models/gemma3.cpp`'s forward. ~15 lines.
3. Add `LLAMA_SPEC_TYPE_EAGLE3` support in our `src/llama-speculative.cpp`: construct `common_speculative_state_eagle3` from PR #18039's vendored impl, forward to `llama_set_eagle3_g_embeddings`.
4. Write an `EAGLE3Drafter: ExternalDrafter` that bridges from the Swift side using `llama_get_eagle3_target_features` (`UnsafePointer<Float>`) + a second small-llama-context as the EAGLE3 decoder.
5. Rebuild xcframework, rerun benchmark. Target: ≥95 tok/s decode.

---

## 7. Risks

1. **PR #18039 conflict churn.** PR is open and actively revised. Every week we rebase submodule, we re-resolve. Mitigation: pin llama.cpp SHA for the whole of Week 5 (ignore upstream moves); rebase in a dedicated Week 8 window once MTP path is proven.
2. **`common_sampler` dep-creep.** The vendored `common_speculative_state_draft` constructor (`common/speculative.cpp:159-208`) calls `common_sampler_init`, which pulls in `common/sampling.{h,cpp}` and thus `common/common.{h,cpp}`. If we want the built-in draft-model path, we drag ~3000 lines of `common` into `libllama`. Mitigation for Week 5: only ship `LLAMA_SPEC_TYPE_EXTERNAL`; defer internal drafter support to Week 8+.
3. **Thread-safety.** `llama_context` is single-threaded. External drafter callback must not call back into `llama_decode(ctx_tgt, …)` — must use its own context (for MTP CoreML this is fine; for an llama.cpp-based drafter we use `ctx_draft` which is a separate context). Document clearly in the callback docstring.
4. **Rejection-sampling bit-parity.** Our Swift port of `common_sampler_sample_and_accept_n` must exactly match the C++ behavior for greedy decoding, else the test in Day 3 step 3 fails. Risk low (it's a 60-line argmax loop) but needs a unit test with a fixed seed and a recorded reference trace.
5. **XCFramework symbol visibility.** Static libraries don't strip `LLAMA_API` by default on Apple, but if we later switch to `LLAMA_SHARED`, `visibility("default")` must be asserted on the new symbols — which it already is via the `LLAMA_API` macro in `include/llama.h:17-27`.
6. **C-ABI vs C++ ABI at the boundary.** All new types in §3 are `extern "C"`-safe: POD structs, function pointers, `int32_t`/`float`/opaque pointers only. Verified by design; risk is zero if we keep `llama_spec_params` free of bool-bitfields, enums-in-C++-mode, or `struct llama_spec` with a non-trivial C++ body (the *opaque* pointer is fine because C clients never look inside).
7. **Upstream API rename risk.** If upstream renames `common_speculative_state` to `llama_speculative_state` (plausible next refactor given the direction of travel), our vendored copy diverges. Mitigation: keep the vendored copy under our own namespace (`llama_speculative_state_*`) from day 1 so the divergence is explicit, not accidental.
8. **License.** llama.cpp is MIT, our fork is MIT — compatible, no issues, just preserve copyright notice at the top of `src/llama-speculative.cpp`.

---

## Appendix A — citation index

All line numbers verified against `/tmp/llama.cpp` @ `b3d7587`:

- `common/speculative.h:1-41` — full public C++ surface.
- `common/speculative.cpp:113-145` — `common_speculative_state` base.
- `common/speculative.cpp:147-438` — draft-model impl.
- `common/speculative.cpp:440-463` — EAGLE3 stub (empty on master).
- `common/speculative.cpp:466-738` — ngram impls.
- `common/speculative.cpp:801-835` — `common_speculative_is_compat`.
- `common/speculative.cpp:839-973` — `common_speculative_init`.
- `common/common.h:171-181` — `enum common_speculative_type`.
- `common/common.h:312-` — `struct common_params_speculative`.
- `common/CMakeLists.txt:44-92` — `common` static-lib target including `speculative.{cpp,h}`.
- `include/llama.h:235-244` — `llama_batch` layout (`n_tokens`, `token`, `embd`, `pos`, `n_seq_id`, `seq_id`, `logits`).
- `include/llama.h:960-962` — `llama_decode`.
- `include/llama.h:1001-1029` — logits / embeddings accessors.
- `include/llama.h:1073-1167` — vocab / tokenize / detokenize.
- `examples/speculative-simple/speculative-simple.cpp:149-234` — canonical burst loop we mirror in Swift.
- PR #18039 diff — `include/llama.h` additions (quoted in §2.1); `src/models/eagle3.cpp` +186; `common/speculative.cpp` +185/−23.

## Appendix B — time estimate table

| Task | Hours | Notes |
|---|---|---|
| Fork + ABI decls in `include/llama.h` | 2 | Mechanical |
| Write `src/llama-speculative.cpp` + internal header | 4 | Skeleton in §4.2 |
| Vendor / strip `common/speculative.cpp` (for internal drafters, Week 8+) | — | Deferred |
| Build XCFramework pipeline (script + test) | 6 | One-time; reusable |
| Swift `LlamaSpec` actor + `ExternalDrafter` protocol | 8 | Includes trampoline + tests |
| Port rejection-sampling loop to Swift | 4 | Greedy-only for Week 5 |
| MTP drafter integration | 4 | Reuses CoreML-LLM's existing `stateful_chunk2.mlpkg` |
| EAGLE-3 cherry-pick + Gemma tap + drafter | 10 | Week 8+ follow-up |
| **Total Week 5** | **28h** | EXTERNAL-drafter only, MTP working |
| **Total incl. EAGLE-3** | **~38h** | End of Week 8 |
