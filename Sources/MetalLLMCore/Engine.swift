//
//  Engine.swift
//  MetalLLMCore
//
//  Top-level decode engine. Owns the llama.cpp context, the spec-decode
//  orchestrator, and the ANE prefill bridge.
//

import Foundation
#if canImport(llama)
import llama
#endif

// MARK: - Configuration

public struct SamplingConfig: Sendable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var maxTokens: Int
    public var speculative: Bool

    public init(
        temperature: Float = 0.0,
        topK: Int = 40,
        topP: Float = 0.95,
        maxTokens: Int = 512,
        speculative: Bool = true
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.maxTokens = maxTokens
        self.speculative = speculative
    }
}

// MARK: - Token event

public struct TokenEvent: Sendable {
    public let token: String
    public let tokenId: Int32
    public let tokensPerSecond: Double
    public let isFinal: Bool
    public let acceptedInBurst: Int
}

// MARK: - Engine

public actor MetalLLMEngine {
    private var llamaCtx: LlamaContext?
    private let specDecode: SpecDecodeOrchestrator
    public let metrics: MetricsCollector
    private var cancelled = false

    public init() {
        self.specDecode = SpecDecodeOrchestrator()
        self.metrics = MetricsCollector()
    }

    public func load(modelPath: URL) async throws {
        let ctx = try LlamaContext.load(
            path: modelPath.path,
            contextLength: 8192
        )
        self.llamaCtx = ctx
        let info = await ctx.modelInfo()
        let size = await ctx.modelSize()
        print("[Engine] Loaded: \(info), \(String(format: "%.2f", Double(size) / 1e9)) GB")
    }

    public func cancel() { cancelled = true }

    public func metricsSnapshot() -> MetricsSnapshot {
        metrics.snapshot()
    }

    // MARK: - Direct generate (callback-based, no stream complexity)

    /// Generate tokens, calling `onToken` for each one. Returns total count.
    @discardableResult
    public func generate(
        prompt: String,
        config: SamplingConfig = SamplingConfig(),
        onToken: @Sendable (TokenEvent) -> Void
    ) async throws -> Int {
        guard let ctx = llamaCtx else { throw EngineError.notLoaded }

        cancelled = false
        await ctx.kvClear()
        await ctx.configureSampler(config: config)

        let tokens = await ctx.tokenize(prompt, addBOS: true)
        guard !tokens.isEmpty else { return 0 }

        // Prefill
        metrics.beginPrefill()
        try await ctx.prefill(tokens: tokens)
        metrics.endPrefill(tokenCount: tokens.count)

        // Decode
        metrics.beginDecode()
        var totalGenerated = 0

        while totalGenerated < config.maxTokens && !cancelled {
            if config.speculative && config.temperature == 0,
               let result = try await specDecode.step(ctx: ctx) {
                // Speculative path succeeded
                metrics.recordDecodedTokens(result.emitted.count)
                metrics.recordSpecBurst(accepted: result.accepted, total: result.drafted)
                totalGenerated += result.emitted.count

                for tokenId in result.emitted {
                    let (text, isFinal) = await ctx.detokenizeAndCheckEOG(tokenId)
                    let snap = metrics.snapshot()
                    onToken(TokenEvent(
                        token: text, tokenId: tokenId,
                        tokensPerSecond: snap.decodeTokensPerSecond,
                        isFinal: isFinal, acceptedInBurst: result.accepted
                    ))
                    if isFinal { return totalGenerated }
                }
            } else {
                let (tokenId, text, isFinal) = try await ctx.sampleDecodeAndEmit()
                metrics.recordDecodedTokens()
                totalGenerated += 1
                let snap = metrics.snapshot()
                onToken(TokenEvent(
                    token: text, tokenId: tokenId,
                    tokensPerSecond: snap.decodeTokensPerSecond,
                    isFinal: isFinal, acceptedInBurst: 0
                ))
                if isFinal { break }
            }
        }
        return totalGenerated
    }

    // MARK: - AsyncThrowingStream wrapper (for SwiftUI)

    public func generateStream(
        prompt: String,
        config: SamplingConfig = SamplingConfig()
    ) -> AsyncThrowingStream<TokenEvent, Error> {
        AsyncThrowingStream { continuation in
            let p = prompt
            let c = config
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try await self.generate(prompt: p, config: c) { event in
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

// MARK: - Errors

public enum EngineError: Error, LocalizedError {
    case notLoaded
    case notImplemented
    case modelLoadFailed(String)
    case decodeFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notLoaded: return "Model not loaded"
        case .notImplemented: return "Not implemented"
        case .modelLoadFailed(let s): return "Model load failed: \(s)"
        case .decodeFailed(let s): return "Decode failed: \(s)"
        }
    }
}
