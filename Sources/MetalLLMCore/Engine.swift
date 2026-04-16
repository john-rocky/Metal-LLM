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
    public let acceptedInBurst: Int  // 0 for non-speculative
}

// MARK: - Engine

public actor MetalLLMEngine {
    private var llamaCtx: LlamaContext?
    private let specDecode: SpecDecodeOrchestrator
    private let metrics: MetricsCollector
    private var generation: Int = 0

    public init() {
        self.specDecode = SpecDecodeOrchestrator()
        self.metrics = MetricsCollector()
    }

    // MARK: - Lifecycle

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

    // MARK: - Generation

    public func generate(
        prompt: String,
        config: SamplingConfig = SamplingConfig()
    ) -> AsyncThrowingStream<TokenEvent, Error> {
        let gen = self.generation
        self.generation += 1

        return AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                do {
                    try await self.runGeneration(
                        prompt: prompt,
                        config: config,
                        generation: gen,
                        continuation: continuation
                    )
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func cancel() {
        generation += 1
    }

    public func metricsSnapshot() -> MetricsSnapshot {
        metrics.snapshot()
    }

    // MARK: - Private

    private func runGeneration(
        prompt: String,
        config: SamplingConfig,
        generation gen: Int,
        continuation: AsyncThrowingStream<TokenEvent, Error>.Continuation
    ) async throws {
        guard let ctx = llamaCtx else {
            throw EngineError.notLoaded
        }

        // Clear KV cache from any previous generation
        await ctx.kvClear()

        // Configure sampler
        await ctx.configureSampler(config: config)

        // Tokenize
        let tokens = await ctx.tokenize(prompt, addBOS: true)
        guard !tokens.isEmpty else {
            continuation.finish()
            return
        }

        // Prefill
        metrics.beginPrefill()
        try await ctx.prefill(tokens: tokens)
        metrics.endPrefill(tokenCount: tokens.count)

        // Decode loop
        metrics.beginDecode()
        var totalGenerated = 0

        while totalGenerated < config.maxTokens && self.generation == gen {
            if config.speculative && config.temperature == 0 {
                // Speculative decoding path (greedy only for now)
                let result = try await specDecode.step(ctx: ctx)
                metrics.recordDecodedTokens(result.emitted.count)
                metrics.recordSpecBurst(accepted: result.accepted, total: result.drafted)
                totalGenerated += result.emitted.count

                for (i, tokenId) in result.emitted.enumerated() {
                    let text = await ctx.detokenize(tokenId)
                    let snap = metrics.snapshot()
                    let isFinal = await ctx.isEOG(tokenId)
                    continuation.yield(TokenEvent(
                        token: text,
                        tokenId: tokenId,
                        tokensPerSecond: snap.decodeTokensPerSecond,
                        isFinal: isFinal,
                        acceptedInBurst: result.accepted
                    ))
                    if isFinal {
                        continuation.finish()
                        return
                    }
                }
            } else {
                // Vanilla decode
                let tokenId = try await ctx.sampleAndDecode()
                metrics.recordDecodedTokens()
                totalGenerated += 1

                let text = await ctx.detokenize(tokenId)
                let snap = metrics.snapshot()
                let isFinal = await ctx.isEOG(tokenId)

                if totalGenerated <= 3 || isFinal {
                    print("[decode] token=\(tokenId) text='\(text)' eog=\(isFinal) total=\(totalGenerated)")
                }

                continuation.yield(TokenEvent(
                    token: text,
                    tokenId: tokenId,
                    tokensPerSecond: snap.decodeTokensPerSecond,
                    isFinal: isFinal,
                    acceptedInBurst: 0
                ))
                if isFinal { break }
            }
        }

        continuation.finish()
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
