//
//  gemma4_kernels.metal
//  MetalLLMCore
//
//  Gemma 4 E2B specialized Metal kernels.
//  SIMDgroup-cooperative matvec for Q4_0 quantized weights.
//

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Q4_0 block: 32 elements, 2 bytes scale + 16 bytes quants = 18 bytes
// ============================================================

struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

// ============================================================
// Kernel: RMSNorm
// ============================================================

kernel void kernel_rms_norm(
    device const half   * input   [[buffer(0)]],
    device       half   * output  [[buffer(1)]],
    device const half   * weight  [[buffer(2)]],
    constant     uint   & n_dim   [[buffer(3)]],
    constant     float  & eps     [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    float sum_sq = 0.0f;
    for (uint i = tid; i < n_dim; i += tpg) {
        float v = float(input[i]);
        sum_sq += v * v;
    }

    float warp_sum = simd_sum(sum_sq);
    threadgroup float shared[32];
    uint lane = tid % 32;
    uint warp = tid / 32;
    if (lane == 0) shared[warp] = warp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0;
        uint n_warps = (tpg + 31) / 32;
        for (uint i = 0; i < n_warps; i++) total += shared[i];
        shared[0] = rsqrt(total / float(n_dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = shared[0];

    for (uint i = tid; i < n_dim; i += tpg) {
        output[i] = half(float(input[i]) * scale * float(weight[i]));
    }
}

// ============================================================
// Kernel: SIMDgroup-cooperative Q4_0 MatVec
//
// One simdgroup (32 threads) computes one output row.
// Multiple simdgroups per threadgroup compute multiple rows.
// Each thread processes a subset of blocks, then simd_sum reduces.
//
// This matches the structure of llama.cpp's kernel_mul_mv_q4_K.
// ============================================================

kernel void kernel_matvec_q4_0(
    device const half          * input   [[buffer(0)]],
    device const block_q4_0   * weight  [[buffer(1)]],
    device       half          * output  [[buffer(2)]],
    constant     uint          & K       [[buffer(3)]],
    constant     uint          & N       [[buffer(4)]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint sglane [[thread_index_in_simdgroup]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint nsg   [[simdgroups_per_threadgroup]]
) {
    // Each simdgroup computes one output row
    uint row = tgid * nsg + sgid;
    if (row >= N) return;

    uint n_blocks = K / 32;
    float acc = 0.0f;

    device const block_q4_0 * rowW = weight + row * n_blocks;

    // Each lane processes blocks strided by 32 (simd width)
    for (uint b = sglane; b < n_blocks; b += 32) {
        float d = float(rowW[b].d);
        float block_sum = 0.0f;

        // Unrolled inner loop over 16 byte-pairs (32 elements)
        for (uint j = 0; j < 16; j++) {
            uint8_t packed = rowW[b].qs[j];
            int8_t q0 = int8_t(packed & 0x0F) - 8;
            int8_t q1 = int8_t(packed >> 4) - 8;
            uint idx = b * 32 + j * 2;
            block_sum += float(q0) * float(input[idx]) + float(q1) * float(input[idx + 1]);
        }
        acc += d * block_sum;
    }

    // Reduce across simdgroup
    float sum = simd_sum(acc);
    if (sglane == 0) {
        output[row] = half(sum);
    }
}

// ============================================================
// Kernel: Fused RMSNorm + Q4_0 MatVec
//
// Phase 1: All threads cooperate on RMSNorm (write to scratch)
// Phase 2: Each simdgroup computes one output row from normed input
// ============================================================

kernel void kernel_fused_rmsnorm_matvec_q4_0(
    device const half          * input      [[buffer(0)]],
    device const half          * norm_w     [[buffer(1)]],
    device const block_q4_0   * weight     [[buffer(2)]],
    device       half          * output     [[buffer(3)]],
    device       half          * normed     [[buffer(4)]],
    constant     uint          & K          [[buffer(5)]],
    constant     uint          & N          [[buffer(6)]],
    constant     float         & eps        [[buffer(7)]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tpg   [[threads_per_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint sglane [[thread_index_in_simdgroup]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint nsg   [[simdgroups_per_threadgroup]]
) {
    // Phase 1: RMSNorm (all threads cooperate)
    float sum_sq = 0.0f;
    for (uint i = tid; i < K; i += tpg) {
        float v = float(input[i]);
        sum_sq += v * v;
    }

    float warp_sum = simd_sum(sum_sq);
    threadgroup float shared[32];
    uint lane = tid % 32;
    uint warp = tid / 32;
    if (lane == 0) shared[warp] = warp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0;
        uint n_warps = (tpg + 31) / 32;
        for (uint i = 0; i < n_warps; i++) total += shared[i];
        shared[0] = rsqrt(total / float(K) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_scale = shared[0];

    for (uint i = tid; i < K; i += tpg) {
        normed[i] = half(float(input[i]) * rms_scale * float(norm_w[i]));
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 2: SIMDgroup-cooperative MatVec
    uint row = tgid * nsg + sgid;
    if (row >= N) return;

    uint n_blocks = K / 32;
    float acc = 0.0f;
    device const block_q4_0 * rowW = weight + row * n_blocks;

    for (uint b = sglane; b < n_blocks; b += 32) {
        float d = float(rowW[b].d);
        float block_sum = 0.0f;
        for (uint j = 0; j < 16; j++) {
            uint8_t packed = rowW[b].qs[j];
            int8_t q0 = int8_t(packed & 0x0F) - 8;
            int8_t q1 = int8_t(packed >> 4) - 8;
            uint idx = b * 32 + j * 2;
            block_sum += float(q0) * float(normed[idx]) + float(q1) * float(normed[idx + 1]);
        }
        acc += d * block_sum;
    }

    float sum = simd_sum(acc);
    if (sglane == 0) {
        output[row] = half(sum);
    }
}

// ============================================================
// Kernel: SwiGLU
// ============================================================

kernel void kernel_swiglu(
    device const half * gate   [[buffer(0)]],
    device const half * up     [[buffer(1)]],
    device       half * output [[buffer(2)]],
    constant     uint & N      [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    float g = float(gate[gid]);
    float u = float(up[gid]);
    float silu_g = g / (1.0f + exp(-g));
    output[gid] = half(silu_g * u);
}

// ============================================================
// Kernel: Residual add
// ============================================================

kernel void kernel_residual_add(
    device const half * a      [[buffer(0)]],
    device       half * b      [[buffer(1)]],
    constant     uint & N      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    b[gid] = half(float(a[gid]) + float(b[gid]));
}
