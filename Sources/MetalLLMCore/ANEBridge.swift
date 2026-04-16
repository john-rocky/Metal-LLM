//
//  ANEBridge.swift
//  MetalLLMCore
//
//  Bridge to the Apple Neural Engine via Core ML. Two use cases:
//
//    1. Prefill offload — reuse the chunk models from the sibling CoreML-LLM repo
//       to burn through long prompts on the ANE while the GPU is idle, then hand
//       the KV cache to the Metal decode path.
//
//    2. ANE-hosted drafter — run an EAGLE/MTP drafter on the ANE while the Metal
//       verifier runs on the GPU. Mirror-SD (Week 11).
//
//  Shared state between ANE and Metal goes through IOSurface when layouts permit
//  zero-copy; otherwise we fall back to a single CPU-resident staging buffer.
//

import Foundation
#if canImport(CoreML)
import CoreML
#endif

/// Configuration for the ANE bridge.
public struct ANEBridgeConfig {
    /// Location of the CoreML-LLM chunk models (the sibling repo's converted output).
    public var chunkModelsDirectory: URL
    /// Whether to use the ANE drafter path (Week 11).
    public var enableDrafter: Bool

    public init(chunkModelsDirectory: URL, enableDrafter: Bool = false) {
        self.chunkModelsDirectory = chunkModelsDirectory
        self.enableDrafter = enableDrafter
    }
}

/// ANE bridge. Skeleton — actual Core ML integration arrives Week 7.
public final class ANEBridge {

    private let config: ANEBridgeConfig

    public init(config: ANEBridgeConfig) {
        self.config = config
    }

    // MARK: - Prefill

    /// Prefill the KV cache on the ANE for a tokenized prompt. Returns a KV blob
    /// that `LlamaBridge` can ingest (same layout as llama.cpp's internal cache).
    public func prefill(tokens: [Int32]) async throws -> Data {
        // TODO(Week 7): load chunk_0..chunk_N .mlpackage via MLModel with
        //               .cpuAndNeuralEngine compute units.
        // TODO(Week 7): stream through chunks, reshape output KV into llama.cpp layout.
        // TODO(Week 8): IOSurface zero-copy handoff where Metal KV layout matches.
        fatalError("ANEBridge.prefill not implemented — Week 7 milestone")
    }

    // MARK: - Drafter (Mirror-SD)

    /// Run the ANE-hosted drafter for one step. Called in parallel with Metal decode.
    public func draftStep(context: [Int32], maxTokens: Int) async throws -> [Int32] {
        // TODO(Week 11): Mirror-SD — dispatch ANE drafter while Metal verifier runs.
        fatalError("ANEBridge.draftStep not implemented — Week 11 milestone")
    }
}
