//
//  gemma4_kernels.metal
//  MetalLLMCore
//
//  Gemma 4 E2B specialized Metal kernels.
//  Q4_K and Q6_K quantized SIMDgroup-cooperative matvec.
//

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Quantization block structures (matching GGML)
// ============================================================

#define QK_K 256
#define K_SCALE_SIZE 12

struct block_q4_K {
    half d;              // super-block scale
    half dmin;           // super-block min scale
    uint8_t scales[12];  // 6-bit quantized scales
    uint8_t qs[128];     // 4-bit quants (256/2)
};
// sizeof = 144 bytes per 256 elements

struct block_q6_K {
    uint8_t ql[128];     // lower 4 bits (256/2)
    uint8_t qh[64];      // upper 2 bits (256/4)
    int8_t  scales[16];  // 8-bit scales (256/16)
    half d;              // super-block scale
};
// sizeof = 210 bytes per 256 elements

// Also keep Q4_0 for testing
struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

// ============================================================
// Scale extraction helper (from llama.cpp)
// ============================================================

inline uchar2 get_scale_min_k4_just2(int j, int k, device const uint8_t * s) {
    uchar2 r;
    if (j < 4) {
        r[0] = s[j  ] & 63;
        r[1] = s[j+4] & 63;
    } else {
        r[0] = (s[j+4] & 0xF) | ((s[j-4] >> 6) << 4);
        r[1] = (s[j+4] >>  4) | ((s[j-0] >> 6) << 4);
    }
    return r;
}

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
    if (tid % 32 == 0) shared[tid / 32] = warp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0;
        for (uint i = 0; i < (tpg + 31) / 32; i++) total += shared[i];
        shared[0] = rsqrt(total / float(n_dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = shared[0];

    for (uint i = tid; i < n_dim; i += tpg) {
        output[i] = half(float(input[i]) * scale * float(weight[i]));
    }
}

// ============================================================
// Kernel: Q4_K MatVec (SIMDgroup cooperative, matching llama.cpp)
//
// Each simdgroup computes one output row.
// 32 threads process blocks in stride, simd_sum to reduce.
// ============================================================

kernel void kernel_matvec_q4_K(
    device const half          * input   [[buffer(0)]],  // (K,) fp16
    device const block_q4_K   * weight  [[buffer(1)]],  // (N * K/256) blocks
    device       half          * output  [[buffer(2)]],  // (N,) fp16
    constant     uint          & K       [[buffer(3)]],
    constant     uint          & N       [[buffer(4)]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint sglane [[thread_index_in_simdgroup]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint nsg    [[simdgroups_per_threadgroup]]
) {
    uint row = tgid * nsg + sgid;
    if (row >= N) return;

    uint nb = K / QK_K;  // number of super-blocks per row
    float acc = 0.0f;

    device const block_q4_K * rowW = weight + row * nb;

    // Each lane handles blocks in stride
    for (uint b = sglane; b < nb; b += 32) {
        float d    = float(rowW[b].d);
        float dmin = float(rowW[b].dmin);
        device const uint8_t * sc = rowW[b].scales;
        device const uint8_t * q  = rowW[b].qs;

        float block_acc = 0.0f;

        // 8 sub-blocks of 32 elements each
        for (uint sb = 0; sb < 8; sb++) {
            // Extract 6-bit scale and min for this sub-block
            uint8_t sc_val, mn_val;
            if (sb < 4) {
                sc_val = sc[sb] & 63;
                mn_val = sc[sb + 4] & 63;
            } else {
                sc_val = (sc[sb + 4] & 0xF) | ((sc[sb - 4] >> 6) << 4);
                mn_val = (sc[sb + 4] >> 4)  | ((sc[sb    ] >> 6) << 4);
            }

            float dl = d * float(sc_val);
            float ml = dmin * float(mn_val);

            uint base_idx = b * QK_K + sb * 32;
            uint q_off = sb * 16;  // 32 elements = 16 bytes of packed nibbles

            // Process 32 elements (16 byte-pairs)
            for (uint j = 0; j < 16; j++) {
                uint8_t packed = q[q_off + j];
                float q0 = float(packed & 0x0F);
                float q1 = float(packed >> 4);
                float x0 = float(input[base_idx + j * 2]);
                float x1 = float(input[base_idx + j * 2 + 1]);
                block_acc += (dl * q0 - ml) * x0 + (dl * q1 - ml) * x1;
            }
        }
        acc += block_acc;
    }

    float sum = simd_sum(acc);
    if (sglane == 0) {
        output[row] = half(sum);
    }
}

// ============================================================
// Kernel: Q6_K MatVec (SIMDgroup cooperative)
// ============================================================

kernel void kernel_matvec_q6_K(
    device const half          * input   [[buffer(0)]],
    device const block_q6_K   * weight  [[buffer(1)]],
    device       half          * output  [[buffer(2)]],
    constant     uint          & K       [[buffer(3)]],
    constant     uint          & N       [[buffer(4)]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint sglane [[thread_index_in_simdgroup]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint nsg    [[simdgroups_per_threadgroup]]
) {
    uint row = tgid * nsg + sgid;
    if (row >= N) return;

    uint nb = K / QK_K;
    float acc = 0.0f;

    device const block_q6_K * rowW = weight + row * nb;

    for (uint b = sglane; b < nb; b += 32) {
        float d = float(rowW[b].d);
        device const uint8_t * ql = rowW[b].ql;
        device const uint8_t * qh = rowW[b].qh;
        device const int8_t  * sc = rowW[b].scales;

        float block_acc = 0.0f;

        // 16 sub-blocks of 16 elements
        for (uint sb = 0; sb < 16; sb++) {
            float dl = d * float(sc[sb]);
            uint base_idx = b * QK_K + sb * 16;

            for (uint j = 0; j < 8; j++) {
                uint qi = sb * 8 + j;
                uint8_t lo0 = ql[qi * 2]     & 0x0F;
                uint8_t lo1 = ql[qi * 2 + 1] & 0x0F;
                uint8_t hi0 = (qh[qi] & 0x03) << 4;
                uint8_t hi1 = (qh[qi] & 0x0C) << 2;

                int8_t v0 = int8_t(lo0 | hi0) - 32;
                int8_t v1 = int8_t(lo1 | hi1) - 32;

                block_acc += dl * float(v0) * float(input[base_idx + j * 2]);
                block_acc += dl * float(v1) * float(input[base_idx + j * 2 + 1]);
            }
        }
        acc += block_acc;
    }

    float sum = simd_sum(acc);
    if (sglane == 0) {
        output[row] = half(sum);
    }
}

// ============================================================
// Kernel: Fused RMSNorm + Q4_K MatVec
// ============================================================

kernel void kernel_fused_rmsnorm_matvec_q4_K(
    device const half          * input      [[buffer(0)]],
    device const half          * norm_w     [[buffer(1)]],
    device const block_q4_K   * weight     [[buffer(2)]],
    device       half          * output     [[buffer(3)]],
    device       half          * normed     [[buffer(4)]],
    constant     uint          & K          [[buffer(5)]],
    constant     uint          & N          [[buffer(6)]],
    constant     float         & eps        [[buffer(7)]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tpg    [[threads_per_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint sglane [[thread_index_in_simdgroup]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint nsg    [[simdgroups_per_threadgroup]]
) {
    // Phase 1: RMSNorm (cooperative)
    float sum_sq = 0.0f;
    for (uint i = tid; i < K; i += tpg) {
        float v = float(input[i]);
        sum_sq += v * v;
    }
    float warp_sum = simd_sum(sum_sq);
    threadgroup float shared[32];
    if (tid % 32 == 0) shared[tid / 32] = warp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0;
        for (uint i = 0; i < (tpg + 31) / 32; i++) total += shared[i];
        shared[0] = rsqrt(total / float(K) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_scale = shared[0];

    for (uint i = tid; i < K; i += tpg) {
        normed[i] = half(float(input[i]) * rms_scale * float(norm_w[i]));
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 2: Q4_K MatVec (same as kernel_matvec_q4_K)
    uint row = tgid * nsg + sgid;
    if (row >= N) return;

    uint nb = K / QK_K;
    float acc = 0.0f;
    device const block_q4_K * rowW = weight + row * nb;

    for (uint b = sglane; b < nb; b += 32) {
        float d    = float(rowW[b].d);
        float dmin = float(rowW[b].dmin);
        device const uint8_t * sc = rowW[b].scales;
        device const uint8_t * q  = rowW[b].qs;
        float block_acc = 0.0f;

        for (uint sb = 0; sb < 8; sb++) {
            uint8_t sc_val, mn_val;
            if (sb < 4) {
                sc_val = sc[sb] & 63;
                mn_val = sc[sb + 4] & 63;
            } else {
                sc_val = (sc[sb + 4] & 0xF) | ((sc[sb - 4] >> 6) << 4);
                mn_val = (sc[sb + 4] >> 4)  | ((sc[sb    ] >> 6) << 4);
            }
            float dl = d * float(sc_val);
            float ml = dmin * float(mn_val);
            uint base_idx = b * QK_K + sb * 32;
            uint q_off = sb * 16;

            for (uint j = 0; j < 16; j++) {
                uint8_t packed = q[q_off + j];
                float q0 = float(packed & 0x0F);
                float q1 = float(packed >> 4);
                block_acc += (dl * q0 - ml) * float(normed[base_idx + j * 2])
                           + (dl * q1 - ml) * float(normed[base_idx + j * 2 + 1]);
            }
        }
        acc += block_acc;
    }

    float sum = simd_sum(acc);
    if (sglane == 0) output[row] = half(sum);
}

// ============================================================
// Keep simple kernels
// ============================================================

// Q4_0 matvec (for testing)
kernel void kernel_matvec_q4_0(
    device const half          * input   [[buffer(0)]],
    device const block_q4_0   * weight  [[buffer(1)]],
    device       half          * output  [[buffer(2)]],
    constant     uint          & K       [[buffer(3)]],
    constant     uint          & N       [[buffer(4)]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint sglane [[thread_index_in_simdgroup]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint nsg    [[simdgroups_per_threadgroup]]
) {
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
            block_sum += float(q0) * float(input[idx]) + float(q1) * float(input[idx + 1]);
        }
        acc += d * block_sum;
    }
    float sum = simd_sum(acc);
    if (sglane == 0) output[row] = half(sum);
}

// Fused RMSNorm + Q4_0 matvec (kept for compat)
kernel void kernel_fused_rmsnorm_matvec_q4_0(
    device const half          * input      [[buffer(0)]],
    device const half          * norm_w     [[buffer(1)]],
    device const block_q4_0   * weight     [[buffer(2)]],
    device       half          * output     [[buffer(3)]],
    device       half          * normed     [[buffer(4)]],
    constant     uint          & K          [[buffer(5)]],
    constant     uint          & N          [[buffer(6)]],
    constant     float         & eps        [[buffer(7)]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tpg    [[threads_per_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint sglane [[thread_index_in_simdgroup]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint nsg    [[simdgroups_per_threadgroup]]
) {
    float sum_sq = 0.0f;
    for (uint i = tid; i < K; i += tpg) { float v = float(input[i]); sum_sq += v * v; }
    float ws = simd_sum(sum_sq);
    threadgroup float sh[32];
    if (tid % 32 == 0) sh[tid/32] = ws;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float t=0; for(uint i=0;i<(tpg+31)/32;i++) t+=sh[i]; sh[0]=rsqrt(t/float(K)+eps); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rs = sh[0];
    for (uint i = tid; i < K; i += tpg) normed[i] = half(float(input[i])*rs*float(norm_w[i]));
    threadgroup_barrier(mem_flags::mem_device);

    uint row = tgid * nsg + sgid;
    if (row >= N) return;
    uint n_blocks = K / 32;
    float acc = 0.0f;
    device const block_q4_0 * rowW = weight + row * n_blocks;
    for (uint b = sglane; b < n_blocks; b += 32) {
        float d = float(rowW[b].d);
        float bs2 = 0.0f;
        for (uint j = 0; j < 16; j++) {
            uint8_t p = rowW[b].qs[j];
            bs2 += float(int8_t(p&0xF)-8)*float(normed[b*32+j*2]) + float(int8_t(p>>4)-8)*float(normed[b*32+j*2+1]);
        }
        acc += d * bs2;
    }
    float sum = simd_sum(acc);
    if (sglane == 0) output[row] = half(sum);
}

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
    output[gid] = half(g / (1.0f + exp(-g)) * u);
}

kernel void kernel_residual_add(
    device const half * a      [[buffer(0)]],
    device       half * b      [[buffer(1)]],
    constant     uint & N      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    b[gid] = half(float(a[gid]) + float(b[gid]));
}
