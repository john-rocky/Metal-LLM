//
//  ANEBridge.swift
//  MetalLLMCore
//
//  Bridge to the Apple Neural Engine via Core ML. Two use cases:
//
//    1. Prefill offload — reuse the chunk models from the sibling CoreML-LLM repo
//       to burn through long prompts on the ANE while the GPU is idle, then hand
//       the KV cache to the Metal decode path via IOSurface zero-copy.
//
//    2. ANE-hosted MTP drafter — run the MTP drafter on the ANE while the Metal
//       verifier runs on the GPU. Mirror-SD pattern.
//
//  Phase 3 implementation. Interface defined now for Engine integration.
//

import Foundation
#if canImport(CoreML)
import CoreML
#endif

// MARK: - Configuration

public struct ANEBridgeConfig: Sendable {
    public var chunkModelsDirectory: URL
    public var drafterModelURL: URL?
    public var enableDrafter: Bool
    public var enablePrefill: Bool

    public init(
        chunkModelsDirectory: URL,
        drafterModelURL: URL? = nil,
        enableDrafter: Bool = false,
        enablePrefill: Bool = true
    ) {
        self.chunkModelsDirectory = chunkModelsDirectory
        self.drafterModelURL = drafterModelURL
        self.enableDrafter = enableDrafter
        self.enablePrefill = enablePrefill
    }
}

// MARK: - ANE Bridge

public actor ANEBridge {
    private let config: ANEBridgeConfig
    private var isReady = false

    public init(config: ANEBridgeConfig) {
        self.config = config
    }

    // MARK: - Lifecycle

    public func warmup() async throws {
        // Phase 3: Load and compile Core ML models
        // - Load chunk1..4 for prefill
        // - Load MTP drafter for Mirror SD
        // - Prewarm compute plans
        isReady = true
    }

    // MARK: - Prefill (Phase 3)

    /// Prefill the KV cache on ANE for a tokenized prompt.
    /// Returns the number of tokens processed.
    public func prefill(tokens: [Int32], kvSlab: SharedKVSlab?) async throws -> Int {
        guard isReady else {
            throw ANEBridgeError.notReady
        }
        // Phase 3 implementation:
        // 1. Load chunk models
        // 2. Run in 128-token batches through ANE
        // 3. Write KV into shared IOSurface slab
        // 4. Return token count
        fatalError("ANEBridge.prefill — Phase 3 milestone")
    }

    // MARK: - MTP Drafter (Phase 3 — Mirror SD)

    /// Run one MTP draft step on ANE. Returns up to K draft tokens.
    public func draftStep(
        lastToken: Int32,
        activations: [Float]?,
        maxTokens: Int
    ) async throws -> [Int32] {
        guard isReady, config.enableDrafter else {
            throw ANEBridgeError.drafterNotAvailable
        }
        // Phase 3 implementation:
        // 1. Embed token
        // 2. Concat with activations
        // 3. Run drafter model on ANE
        // 4. Argmax to get draft tokens
        fatalError("ANEBridge.draftStep — Phase 3 milestone")
    }
}

// MARK: - SharedKVSlab placeholder (Phase 3)

/// IOSurface-backed KV cache shared between ANE and Metal.
/// Full implementation in Phase 3.
public final class SharedKVSlab: @unchecked Sendable {
    // Phase 3: CVPixelBuffer + IOSurface + MLMultiArray + MTLBuffer views
    public init() {}
}

// MARK: - Errors

public enum ANEBridgeError: Error {
    case notReady
    case drafterNotAvailable
    case prefillFailed(String)
}
