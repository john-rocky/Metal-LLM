//
//  gemma4_kernels.metal
//  MetalLLMCore
//
//  Gemma 4 E2B specialized Metal kernels.
//  Fused operations to minimize dispatch count.
//
//  Q4_K dequantization format (from GGML):
//    Block of 256 elements, stored as:
//    - 2 fp16 scales (d, dmin)
//    - 12 bytes of scale factors (k_scale, 6-bit each)
//    - 4×32 = 128 bytes of 4-bit quantized values
//    Total: 144 bytes per 256 elements
//

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Q4_K block structure (matches GGML)
// ============================================================

// Simplified Q4_0 for initial prototype (simpler than Q4_K_M)
// Q4_0: 32 elements per block, 2 bytes scale + 16 bytes quants = 18 bytes
struct block_q4_0 {
    half d;             // delta (scale)
    uint8_t qs[16];     // 32 x 4-bit quants (packed in pairs)
};

// ============================================================
// Kernel 1: RMSNorm
// ============================================================

kernel void kernel_rms_norm(
    device const half   * input   [[buffer(0)]],
    device       half   * output  [[buffer(1)]],
    device const half   * weight  [[buffer(2)]],
    constant     uint   & n_dim   [[buffer(3)]],
    constant     float  & eps     [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]]
) {
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (uint i = tid; i < n_dim; i += tpg) {
        float v = float(input[i]);
        sum_sq += v * v;
    }

    // Reduce across threads (simplified — production uses simdgroup_sum)
    threadgroup float shared[32];
    uint lane = tid % 32;
    uint warp = tid / 32;

    float warp_sum = simd_sum(sum_sq);
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

    // Apply: output = input * scale * weight
    for (uint i = tid; i < n_dim; i += tpg) {
        output[i] = half(float(input[i]) * scale * float(weight[i]));
    }
}

// ============================================================
// Kernel 2: Fused Q4_0 MatVec (for decode: M=1, output = input @ W^T)
//   input:  (1, K)  fp16
//   weight: (N, K)  Q4_0 quantized
//   output: (1, N)  fp16
//
//  Each thread computes one output element (dot product of input with one weight row).
//  Weight dequantization is done in-register — no intermediate FP16 buffer.
// ============================================================

kernel void kernel_matvec_q4_0(
    device const half          * input   [[buffer(0)]],  // (K,) fp16
    device const block_q4_0   * weight  [[buffer(1)]],  // (N * K/32) blocks
    device       half          * output  [[buffer(2)]],  // (N,) fp16
    constant     uint          & K       [[buffer(3)]],  // input dim
    constant     uint          & N       [[buffer(4)]],  // output dim
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;

    uint n_blocks = K / 32;
    float acc = 0.0f;

    // Each row of the weight matrix is n_blocks consecutive block_q4_0
    device const block_q4_0 * row = weight + gid * n_blocks;

    for (uint b = 0; b < n_blocks; b++) {
        float d = float(row[b].d);

        // Process 32 elements (16 bytes of packed 4-bit values)
        for (uint j = 0; j < 16; j++) {
            uint8_t packed = row[b].qs[j];
            // Low nibble (element 2*j)
            int8_t q0 = int8_t(packed & 0x0F) - 8;
            // High nibble (element 2*j + 1)
            int8_t q1 = int8_t(packed >> 4) - 8;

            uint idx = b * 32 + j * 2;
            acc += d * float(q0) * float(input[idx]);
            acc += d * float(q1) * float(input[idx + 1]);
        }
    }

    output[gid] = half(acc);
}

// ============================================================
// Kernel 3: Fused RMSNorm + MatVec (combines kernel 1 + 2)
//   Saves one full dispatch + one activation read/write round-trip
// ============================================================

kernel void kernel_fused_rmsnorm_matvec_q4_0(
    device const half          * input      [[buffer(0)]],  // (K,) fp16
    device const half          * norm_w     [[buffer(1)]],  // (K,) fp16 RMSNorm weight
    device const block_q4_0   * weight      [[buffer(2)]],  // (N * K/32) blocks
    device       half          * output     [[buffer(3)]],  // (N,) fp16
    device       half          * normed     [[buffer(4)]],  // (K,) fp16 scratch for normed input
    constant     uint          & K          [[buffer(5)]],
    constant     uint          & N          [[buffer(6)]],
    constant     float         & eps        [[buffer(7)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]]
) {
    // Phase 1: RMSNorm (all threads cooperate)
    float sum_sq = 0.0f;
    for (uint i = tid; i < K; i += tpg) {
        float v = float(input[i]);
        sum_sq += v * v;
    }

    threadgroup float shared[32];
    uint lane = tid % 32;
    uint warp = tid / 32;
    float warp_sum = simd_sum(sum_sq);
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

    // Write normed input to scratch
    for (uint i = tid; i < K; i += tpg) {
        normed[i] = half(float(input[i]) * rms_scale * float(norm_w[i]));
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 2: MatVec Q4_0 (each thread computes one output)
    if (tid >= N) return;

    uint n_blocks = K / 32;
    float acc = 0.0f;
    device const block_q4_0 * row = weight + tid * n_blocks;

    for (uint b = 0; b < n_blocks; b++) {
        float d = float(row[b].d);
        for (uint j = 0; j < 16; j++) {
            uint8_t packed = row[b].qs[j];
            int8_t q0 = int8_t(packed & 0x0F) - 8;
            int8_t q1 = int8_t(packed >> 4) - 8;
            uint idx = b * 32 + j * 2;
            acc += d * float(q0) * float(normed[idx]);
            acc += d * float(q1) * float(normed[idx + 1]);
        }
    }
    output[tid] = half(acc);
}

// ============================================================
// Kernel 4: SwiGLU activation (gate * silu(up))
// ============================================================

kernel void kernel_swiglu(
    device const half * gate   [[buffer(0)]],  // (N,) fp16
    device const half * up     [[buffer(1)]],  // (N,) fp16
    device       half * output [[buffer(2)]],  // (N,) fp16
    constant     uint & N      [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    float g = float(gate[gid]);
    float u = float(up[gid]);
    // SiLU(x) = x * sigmoid(x)
    float silu_g = g / (1.0f + exp(-g));
    output[gid] = half(silu_g * u);
}

// ============================================================
// Kernel 5: Residual add
// ============================================================

kernel void kernel_residual_add(
    device const half * a      [[buffer(0)]],
    device       half * b      [[buffer(1)]],  // in-place: b += a
    constant     uint & N      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    b[gid] = half(float(a[gid]) + float(b[gid]));
}
