//
//  Engine.swift
//  MetalLLMCore
//
//  Top-level decode engine. Owns the llama.cpp context, the spec-decode orchestrator,
//  and the ANE prefill bridge. The public surface is intentionally narrow: load a
//  model, stream tokens, report metrics.
//

import Foundation

/// Sampling configuration passed per-generation.
public struct SamplingConfig {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var maxTokens: Int

    public init(temperature: Float = 0.7, topK: Int = 40, topP: Float = 0.95, maxTokens: Int = 512) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.maxTokens = maxTokens
    }
}

/// A single streamed token event.
public struct TokenEvent {
    public let token: String
    public let tokenId: Int32
    public let isFinal: Bool
}

/// Public decode engine API.
public protocol DecodeEngine: AnyObject {
    /// Load a GGUF model from disk. Resolves once the Metal context is ready.
    func load(modelPath: URL) async throws

    /// Stream tokens for a prompt. The spec-decode orchestrator is consulted internally.
    func generate(prompt: String, config: SamplingConfig) -> AsyncThrowingStream<TokenEvent, Error>

    /// Latest metrics snapshot (tok/s, power, thermal).
    func metricsSnapshot() -> MetricsSnapshot
}

/// Default engine implementation. Skeleton only — no runtime yet.
public final class MetalLLMEngine: DecodeEngine {
    private let llama: LlamaBridge
    private let specDecode: SpecDecodeOrchestrator
    private let aneBridge: ANEBridge?
    private let metrics: MetricsCollector

    public init(specDecode: SpecDecodeOrchestrator = .init(),
                aneBridge: ANEBridge? = nil) {
        self.llama = LlamaBridge()
        self.specDecode = specDecode
        self.aneBridge = aneBridge
        self.metrics = MetricsCollector()
    }

    public func load(modelPath: URL) async throws {
        // TODO(Week 1): initialise llama.cpp context with Metal backend.
        // TODO(Week 7): if aneBridge != nil, warm ANE prefill chunks here.
        fatalError("MetalLLMEngine.load not implemented — Week 1 milestone")
    }

    public func generate(prompt: String, config: SamplingConfig) -> AsyncThrowingStream<TokenEvent, Error> {
        // TODO(Week 1): basic decode loop via LlamaBridge.
        // TODO(Week 5): route through specDecode.drafter + llama verifier.
        // TODO(Week 7): prefill via aneBridge, decode via llama.
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: EngineError.notImplemented)
        }
    }

    public func metricsSnapshot() -> MetricsSnapshot {
        metrics.snapshot()
    }
}

public enum EngineError: Error {
    case notImplemented
    case modelLoadFailed(String)
    case decodeFailed(String)
}
