//
//  Gemma4Layer.swift
//  MetalLLMCore
//
//  One Gemma 4 transformer layer via custom Metal kernels.
//  SIMDgroup-cooperative matvec for all projections.
//

import Foundation
import Metal

public final class Gemma4Layer {
    private let runtime: MetalRuntime
    private let layerIdx: Int
    private let nEmbd: Int
    private let nHead: Int
    private let nHeadKV: Int
    private let headDim: Int
    private let nFF: Int
    private let isSWA: Bool
    private let eps: Float = 1e-6

    // Weights (buffer + type for kernel selection)
    private var attnNormW: MTLBuffer?
    private var qWeight: MTLBuffer?;  private var qType: Int32 = 0
    private var kWeight: MTLBuffer?;  private var kType: Int32 = 0
    private var vWeight: MTLBuffer?;  private var vType: Int32 = 0
    private var oWeight: MTLBuffer?;  private var oType: Int32 = 0
    private var ffnNormW: MTLBuffer?
    private var gateWeight: MTLBuffer?; private var gateType: Int32 = 0
    private var upWeight: MTLBuffer?;   private var upType: Int32 = 0
    private var downWeight: MTLBuffer?; private var downType: Int32 = 0

    // Scratch
    private var normedBuf: MTLBuffer?
    private var qBuf: MTLBuffer?
    private var kBuf: MTLBuffer?
    private var vBuf: MTLBuffer?
    private var oBuf: MTLBuffer?
    private var gateBuf: MTLBuffer?
    private var upBuf: MTLBuffer?
    private var swigluBuf: MTLBuffer?
    private var downBuf: MTLBuffer?

    // Dispatch config: N output rows, each computed by 1 simdgroup (32 threads)
    // simdgroups_per_threadgroup = 4 (128 threads per TG)
    private let nsgPerTG = 4
    private var simdWidth: Int { 32 }

    public init(runtime: MetalRuntime, layerIdx: Int, config: Gemma4Config) {
        self.runtime = runtime
        self.layerIdx = layerIdx
        self.nEmbd = config.nEmbd
        self.nHead = config.nHead
        self.nHeadKV = config.nHeadKV
        self.nFF = config.nFF[layerIdx]
        self.isSWA = config.swaPattern[layerIdx]
        self.headDim = isSWA ? 256 : 512

        let d = runtime.device
        normedBuf = d.makeBuffer(length: nEmbd * 2, options: .storageModeShared)
        qBuf = d.makeBuffer(length: nHead * headDim * 2, options: .storageModeShared)
        kBuf = d.makeBuffer(length: nHeadKV * headDim * 2, options: .storageModeShared)
        vBuf = d.makeBuffer(length: nHeadKV * headDim * 2, options: .storageModeShared)
        oBuf = d.makeBuffer(length: nEmbd * 2, options: .storageModeShared)
        gateBuf = d.makeBuffer(length: nFF * 2, options: .storageModeShared)
        upBuf = d.makeBuffer(length: nFF * 2, options: .storageModeShared)
        swigluBuf = d.makeBuffer(length: nFF * 2, options: .storageModeShared)
        downBuf = d.makeBuffer(length: nEmbd * 2, options: .storageModeShared)
    }

    public func loadWeights(from loader: GGUFLoader) {
        let l = layerIdx
        func load(_ comp: String) -> (MTLBuffer?, Int32) {
            if let t = loader.layerTensor(layer: l, component: comp) {
                return (t.buffer, Int32(t.type.rawValue))
            }
            return (nil, 0)
        }
        attnNormW = loader.layerTensor(layer: l, component: "attn_norm")?.buffer
        (qWeight, qType) = load("attn_q")
        (kWeight, kType) = load("attn_k")
        (vWeight, vType) = load("attn_v")
        (oWeight, oType) = load("attn_output")
        ffnNormW = loader.layerTensor(layer: l, component: "ffn_norm")?.buffer
        (gateWeight, gateType) = load("ffn_gate")
        (upWeight, upType) = load("ffn_up")
        (downWeight, downType) = load("ffn_down")
        print("[Layer \(l)] \(isSWA ? "SWA" : "Full") hd=\(headDim) ff=\(nFF) qType=\(qType) downType=\(downType)")
    }

    // MARK: - Forward

    public func forward(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        kvCache: MTLBuffer?,
        position: Int
    ) {
        // --- Attention block ---
        dispatchFusedNormMatvec(encoder: encoder, input: input, normW: attnNormW,
                                weight: qWeight, output: qBuf, normed: normedBuf,
                                inDim: nEmbd, outDim: nHead * headDim, weightType: qType)

        dispatchMatvec(encoder: encoder, input: normedBuf, weight: kWeight,
                       output: kBuf, inDim: nEmbd, outDim: nHeadKV * headDim, weightType: kType)

        dispatchMatvec(encoder: encoder, input: normedBuf, weight: vWeight,
                       output: vBuf, inDim: nEmbd, outDim: nHeadKV * headDim, weightType: vType)

        // TODO: RoPE, SDPA, O projection, attention residual add

        // --- FFN block ---
        dispatchFusedNormMatvec(encoder: encoder, input: input, normW: ffnNormW,
                                weight: gateWeight, output: gateBuf, normed: normedBuf,
                                inDim: nEmbd, outDim: nFF, weightType: gateType)

        dispatchMatvec(encoder: encoder, input: normedBuf, weight: upWeight,
                       output: upBuf, inDim: nEmbd, outDim: nFF, weightType: upType)

        // SwiGLU
        if let pipeline = runtime.pipeline("kernel_swiglu"),
           let gate = gateBuf, let up = upBuf, let swiglu = swigluBuf {
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(gate, offset: 0, index: 0)
            encoder.setBuffer(up, offset: 0, index: 1)
            encoder.setBuffer(swiglu, offset: 0, index: 2)
            var n = UInt32(nFF)
            encoder.setBytes(&n, length: 4, index: 3)
            encoder.dispatchThreads(
                MTLSize(width: nFF, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(nFF, 1024), height: 1, depth: 1)
            )
        }

        // Down projection
        dispatchMatvec(encoder: encoder, input: swigluBuf, weight: downWeight,
                       output: downBuf, inDim: nFF, outDim: nEmbd, weightType: downType)

        // Residual add
        if let pipeline = runtime.pipeline("kernel_residual_add"), let down = downBuf {
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(down, offset: 0, index: 0)
            encoder.setBuffer(input, offset: 0, index: 1)
            var n = UInt32(nEmbd)
            encoder.setBytes(&n, length: 4, index: 2)
            encoder.dispatchThreads(
                MTLSize(width: nEmbd, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(nEmbd, 1024), height: 1, depth: 1)
            )
        }
    }

    // MARK: - Dispatch helpers

    /// Select kernel name based on weight quantization type.
    private func matvecKernel(type: Int32) -> String {
        switch type {
        case 12: return "kernel_matvec_q4_K"   // GGML_TYPE_Q4_K
        case 14: return "kernel_matvec_q6_K"   // GGML_TYPE_Q6_K
        default: return "kernel_matvec_q4_0"   // fallback
        }
    }

    private func fusedNormMatvecKernel(type: Int32) -> String {
        switch type {
        case 12: return "kernel_fused_rmsnorm_matvec_q4_K"
        default: return "kernel_fused_rmsnorm_matvec_q4_0"
        }
    }

    /// Dispatch SIMDgroup-cooperative matvec: N output rows, each by 1 simdgroup.
    private func dispatchMatvec(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer?, weight: MTLBuffer?, output: MTLBuffer?,
        inDim: Int, outDim: Int, weightType: Int32 = 0
    ) {
        let kernelName = matvecKernel(type: weightType)
        guard let pipeline = runtime.pipeline(kernelName),
              let inp = input, let w = weight, let out = output else { return }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inp, offset: 0, index: 0)
        encoder.setBuffer(w, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        var k = UInt32(inDim), n = UInt32(outDim)
        encoder.setBytes(&k, length: 4, index: 3)
        encoder.setBytes(&n, length: 4, index: 4)

        // N rows, each needs 1 simdgroup. Pack nsgPerTG simdgroups per threadgroup.
        let nThreadgroups = (outDim + nsgPerTG - 1) / nsgPerTG
        encoder.dispatchThreadgroups(
            MTLSize(width: nThreadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: nsgPerTG * simdWidth, height: 1, depth: 1)
        )
    }

    /// Dispatch fused RMSNorm + matvec.
    private func dispatchFusedNormMatvec(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer?, normW: MTLBuffer?, weight: MTLBuffer?,
        output: MTLBuffer?, normed: MTLBuffer?,
        inDim: Int, outDim: Int, weightType: Int32 = 0
    ) {
        let kernelName = fusedNormMatvecKernel(type: weightType)
        guard let pipeline = runtime.pipeline(kernelName),
              let inp = input, let nw = normW, let w = weight,
              let out = output, let norm = normed else { return }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inp, offset: 0, index: 0)
        encoder.setBuffer(nw, offset: 0, index: 1)
        encoder.setBuffer(w, offset: 0, index: 2)
        encoder.setBuffer(out, offset: 0, index: 3)
        encoder.setBuffer(norm, offset: 0, index: 4)
        var k = UInt32(inDim), n = UInt32(outDim), e = eps
        encoder.setBytes(&k, length: 4, index: 5)
        encoder.setBytes(&n, length: 4, index: 6)
        encoder.setBytes(&e, length: 4, index: 7)

        let nThreadgroups = (outDim + nsgPerTG - 1) / nsgPerTG
        encoder.dispatchThreadgroups(
            MTLSize(width: nThreadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: nsgPerTG * simdWidth, height: 1, depth: 1)
        )
    }
}

// MARK: - Config

public struct Gemma4Config: Sendable {
    public let nEmbd: Int
    public let nHead: Int
    public let nHeadKV: Int
    public let nLayers: Int
    public let nFF: [Int]
    public let swaPattern: [Bool]
    public let swaWindow: Int
    public let vocabSize: Int

    public static let gemma4E2B = Gemma4Config(
        nEmbd: 1536, nHead: 8, nHeadKV: 1, nLayers: 35,
        nFF: (0..<35).map { $0 < 15 ? 6144 : 12288 },
        swaPattern: [
            true, true, true, true, false,
            true, true, true, true, false,
            true, true, true, true, false,
            true, true, true, true, true,
            true, true, true, true, true,
            true, true, true, true, true,
            true, true, true, true, true,
        ],
        swaWindow: 512, vocabSize: 262144
    )
}
