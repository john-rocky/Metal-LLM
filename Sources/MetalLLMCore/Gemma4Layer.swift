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

    // Weights
    private var attnNormW: MTLBuffer?
    private var qWeight: MTLBuffer?
    private var kWeight: MTLBuffer?
    private var vWeight: MTLBuffer?
    private var oWeight: MTLBuffer?
    private var ffnNormW: MTLBuffer?
    private var gateWeight: MTLBuffer?
    private var upWeight: MTLBuffer?
    private var downWeight: MTLBuffer?

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
        attnNormW = loader.layerTensor(layer: l, component: "attn_norm")?.buffer
        qWeight = loader.layerTensor(layer: l, component: "attn_q")?.buffer
        kWeight = loader.layerTensor(layer: l, component: "attn_k")?.buffer
        vWeight = loader.layerTensor(layer: l, component: "attn_v")?.buffer
        oWeight = loader.layerTensor(layer: l, component: "attn_output")?.buffer
        ffnNormW = loader.layerTensor(layer: l, component: "ffn_norm")?.buffer
        gateWeight = loader.layerTensor(layer: l, component: "ffn_gate")?.buffer
        upWeight = loader.layerTensor(layer: l, component: "ffn_up")?.buffer
        downWeight = loader.layerTensor(layer: l, component: "ffn_down")?.buffer
        print("[Layer \(l)] \(isSWA ? "SWA" : "Full") hd=\(headDim) ff=\(nFF) hasKV=\(kWeight != nil)")
    }

    // MARK: - Forward

    public func forward(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        kvCache: MTLBuffer?,
        position: Int
    ) {
        // --- Attention block ---
        // Fused RMSNorm + Q projection
        dispatchFusedNormMatvec(encoder: encoder, input: input, normW: attnNormW,
                                weight: qWeight, output: qBuf, normed: normedBuf,
                                inDim: nEmbd, outDim: nHead * headDim)

        // K projection (from normed)
        dispatchMatvec(encoder: encoder, input: normedBuf, weight: kWeight,
                       output: kBuf, inDim: nEmbd, outDim: nHeadKV * headDim)

        // V projection (from normed)
        dispatchMatvec(encoder: encoder, input: normedBuf, weight: vWeight,
                       output: vBuf, inDim: nEmbd, outDim: nHeadKV * headDim)

        // TODO: RoPE, SDPA, O projection, attention residual add
        // For now, skip attention output and go straight to FFN

        // --- FFN block ---
        // Fused RMSNorm + gate projection
        dispatchFusedNormMatvec(encoder: encoder, input: input, normW: ffnNormW,
                                weight: gateWeight, output: gateBuf, normed: normedBuf,
                                inDim: nEmbd, outDim: nFF)

        // Up projection (from normed)
        dispatchMatvec(encoder: encoder, input: normedBuf, weight: upWeight,
                       output: upBuf, inDim: nEmbd, outDim: nFF)

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
                       output: downBuf, inDim: nFF, outDim: nEmbd)

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

    /// Dispatch SIMDgroup-cooperative matvec: N output rows, each by 1 simdgroup.
    private func dispatchMatvec(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer?, weight: MTLBuffer?, output: MTLBuffer?,
        inDim: Int, outDim: Int
    ) {
        guard let pipeline = runtime.pipeline("kernel_matvec_q4_0"),
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
        inDim: Int, outDim: Int
    ) {
        guard let pipeline = runtime.pipeline("kernel_fused_rmsnorm_matvec_q4_0"),
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
