# Custom Metal Runtime — Gemma 4 E2B Decode Engine

**Goal:** World's fastest single-token decode on iPhone 17 Pro. Not "use framework X" — build a new engine that cherry-picks every proven technique from llama.cpp, MLX, LiteRT-LM, and adds Gemma 4-specific optimizations impossible in any general framework.

## Why custom beats general

| Technique | llama.cpp | MLX | LiteRT-LM | **Ours** |
|---|---|---|---|---|
| NAX 16×16 tensor core matmul | no | yes | no | **yes** |
| Indirect Command Buffer (ICB) replay | no | no | yes | **yes** |
| JIT function-constant specialization | no | yes | no | **yes** |
| In-register INT4 dequant | yes | yes | no | **yes** |
| FP16 pre-dequant option (trade RAM) | no | no | yes | **yes** |
| Single encoder, no endEncoding | no | no | yes | **yes** |
| Fused RMSNorm+Gate+Up | no | partial | partial | **yes (Gemma 4 specific)** |
| KV share L15-34 exploit | no | no | no | **yes** |
| SWA constant-time attention | partial | partial | yes | **yes** |
| IOSurface ANE↔GPU zero-copy | no | no | no | **yes** |
| Double-buffer async pipeline | no | yes | yes | **yes** |
| Residency set pinning | yes | yes | no | **yes** |
| Untracked hazard mode | no | yes | no | **yes** |

**No single framework has all of these. Combined = fastest possible.**

---

## Architecture

### One token decode = one ICB replay

```
MTLIndirectCommandBuffer (pre-recorded at model load):
  ├─ dispatch: embed_lookup          [trivial, 1 dispatch]
  ├─ dispatch: ple_layer_embed       [per-layer embedding injection]
  │
  ├─ LAYER 0 (SWA, own KV):
  │   ├─ dispatch: fused_rmsnorm_qkv_proj_rope   [1 kernel]
  │   ├─ dispatch: sdpa_vector_swa                [1 kernel, window=512]
  │   ├─ dispatch: o_proj_residual                [1 kernel]
  │   ├─ dispatch: fused_rmsnorm_geglu_down       [1 kernel]
  │   └─ dispatch: residual_add                   [1 kernel]
  │
  ├─ LAYERS 1-3 (SWA, same pattern)
  ├─ LAYER 4 (full-attn, own KV):
  │   └─ ... sdpa_vector_full instead of swa
  ├─ LAYERS 5-14 (mixed SWA/full, 15 layers with own KV)
  │
  ├─ LAYERS 15-34 (Q-only, shared kv13/kv14):
  │   ├─ dispatch: fused_rmsnorm_q_proj_rope      [no K/V proj!]
  │   ├─ dispatch: sdpa_vector_shared_kv           [reads kv13 or kv14]
  │   ├─ dispatch: o_proj_residual
  │   ├─ dispatch: fused_rmsnorm_geglu_down_wide   [12288 intermediate for L15-34]
  │   └─ dispatch: residual_add
  │
  ├─ dispatch: final_rmsnorm
  └─ dispatch: lm_head_proj          [262K vocab, softcap=30]

Total dispatches: ~175 (vs llama.cpp ~350, LiteRT-LM ~270)
All in ONE MTLComputeCommandEncoder, ONE commit.
ICB recorded ONCE at load, replayed every token — zero CPU encode cost.
```

### Kernel inventory (7 custom Metal kernels)

| Kernel | What it fuses | Input | Output | NAX? |
|---|---|---|---|---|
| `fused_rmsnorm_qkv_proj_rope` | RMSNorm + Q/K/V matmul + RoPE | hidden(1536) + weights | Q,K,V + KV write | **yes** |
| `fused_rmsnorm_q_proj_rope` | Same but Q-only (L15-34) | hidden(1536) + Q weight | Q only | **yes** |
| `sdpa_vector_swa` | Q·K^T + mask + softmax + P·V (window) | Q, KV slab, pos | attn_out | no (register) |
| `sdpa_vector_full` | Same but full context | Q, KV slab, pos | attn_out | no (register) |
| `o_proj_residual` | O projection + residual add | attn_out + weight + residual | hidden | **yes** |
| `fused_rmsnorm_geglu_down` | RMSNorm + gate·up + SwiGLU + down + residual | hidden + 3 weights | hidden | **yes** |
| `lm_head_softcap` | matmul(262K) + tanh softcap(30) + argmax | hidden + lm_weight | token_id | **yes** |

**7 kernels × pre-compiled with function constants = zero runtime compilation.**
Each kernel is shape-specialized for Gemma 4 (hidden=1536, heads=8, kv_heads=1, head_dim=256/512).

### Memory layout

```
Weights:     MTLBuffer, storageModeShared, residencySet pinned
             INT4 group-quantized (3.1 GB) OR FP16 pre-dequant (6.2 GB, optional)
             
KV cache:    IOSurface-backed, single buffer per layer-group
             Layout: (heads, seq, head_dim), FP16 or INT8
             Zero-copy aliasable as MLMultiArray (for ANE prefill/drafter)

Activations: MTLBuffer × 2 (ping-pong), storageModeShared
             hidden: (1, 1536) FP16
             Pre-allocated, never freed during session

ICB:         MTLIndirectCommandBuffer, pre-recorded at load
             175 dispatches, replayed per token
             Only mutable args: embed_input, kv_position, attention_mask
```

### Per-token data flow

```
CPU:  write token_id to embed_input buffer    [~0 μs, pointer write]
      write position to kv_pos buffer          [~0 μs]  
      update mask buffer                        [~0 μs]
      [icb executeCommandsInBuffer:]            [single dispatch, ~5 μs]
      [commandBuffer commit]                    [~1 μs via doorbell]
      → CPU FREE, start preparing token N+1

GPU:  175 dispatches execute back-to-back
      All weights read from pinned resident buffers (no page faults)
      KV written in-place via position index
      Final argmax yields token_id

Total GPU time: ~7-10 ms per token (estimated for A19 Pro)
→ 100-143 tok/s WITHOUT speculative decoding
```

---

## Why this is faster than llama.cpp's ~38 tok/s on iPhone

| Bottleneck | llama.cpp | Custom runtime | Savings |
|---|---|---|---|
| Dispatch overhead | 350 dispatches, CPU encode per token | **175 dispatches, ICB replay** | ~50% fewer dispatches, ~0 CPU cost |
| Weight dequant | In-register but per-dispatch | **NAX matmul with INT4 native** or FP16 pre-dequant | 10-30% |
| Attention | Generic FlashAttn | **Gemma 4 specialized: SWA=constant, full=seq_len** | 15-20% for SWA layers |
| KV write | 35 layers × write K,V | **15 layers only (L15-34 share kv13/14)** | 57% fewer KV writes |
| Kernel launch | New encoder per op | **Single encoder, untracked hazards** | ~5% |
| Residency | Per-launch validation | **Residency set pinned** | ~3% |

Compound: 1.5 × 1.2 × 1.15 × 1.57 × 1.05 × 1.03 ≈ **3.5×**
38 tok/s × 3.5 = **~133 tok/s** (without speculative decoding)

---

## Implementation phases

### Phase A: Skeleton + matmul kernel (1 week)
- MTLDevice, MTLCommandQueue, MTLResidencySet setup
- GGUF weight loader (reuse llama.cpp's parser, output to MTLBuffer)
- First kernel: `matmul_q4_nax` — INT4 quantized matmul using NAX 16×16 frags
- Benchmark: single matmul vs llama.cpp's mul_mv_q4_K

### Phase B: Forward pass (1 week)
- All 7 kernels implemented
- Full 35-layer forward pass in one MTLCommandBuffer
- Verify output matches llama.cpp bit-for-bit on same GGUF
- Benchmark: full token decode vs llama.cpp

### Phase C: ICB + pipeline (3 days)
- Pre-record ICB at model load
- Double-buffer command submission
- Measure: dispatch overhead → ~0

### Phase D: KV optimization (3 days)
- IOSurface-backed KV with single-buffer in-place update
- INT8 KV option
- KV share for L15-34

### Phase E: Integration + bench (2 days)
- Wire into MetalLLMEngine
- iPhone 17 Pro benchmark
- Document results in BENCHMARKS.md
