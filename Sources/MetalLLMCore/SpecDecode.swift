//
//  SpecDecode.swift
//  MetalLLMCore
//
//  Speculative decoding orchestrator with n-gram prompt-lookup drafter.
//  Draft → verify → accept/reject → KV rollback pipeline.
//  K=3, greedy verification, rolling acceptance EMA with fallback.
//

import Foundation

// MARK: - Result

public struct SpecDecodeResult: Sendable {
    public let emitted: [Int32]
    public let accepted: Int
    public let drafted: Int
}

// MARK: - External drafter protocol

public protocol ExternalDrafter: AnyObject, Sendable {
    func draft(lastToken: Int32, maxTokens: Int) async throws -> [Int32]
}

// MARK: - N-gram drafter (built-in, zero-cost)

/// Thread-safe n-gram drafter. Uses a simple array + lock instead of actor
/// to avoid async overhead in the hot path.
public final class NgramDrafter: @unchecked Sendable {
    private var history: [Int32] = []
    private let ngramMin: Int
    private let ngramMax: Int
    private let lock = NSLock()

    public init(ngramMin: Int = 2, ngramMax: Int = 4) {
        self.ngramMin = ngramMin
        self.ngramMax = ngramMax
    }

    public func feedTokens(_ tokens: [Int32]) {
        lock.lock()
        history.append(contentsOf: tokens)
        if history.count > 8192 { history = Array(history.suffix(8192)) }
        lock.unlock()
    }

    public func feedPromptTokens(_ tokens: [Int32]) {
        lock.lock()
        history = tokens
        lock.unlock()
    }

    /// Synchronous draft — no async, no actor hop.
    public func draft(maxTokens: Int) -> [Int32] {
        lock.lock()
        defer { lock.unlock() }

        // Try decreasing n-gram sizes for best match
        for n in stride(from: ngramMax, through: ngramMin, by: -1) {
            guard history.count >= n else { continue }
            let tail = Array(history.suffix(n))
            let searchEnd = history.count - n

            for i in stride(from: searchEnd - 1, through: 0, by: -1) {
                var match = true
                for j in 0..<n {
                    if history[i + j] != tail[j] { match = false; break }
                }
                if match {
                    let start = i + n
                    let end = min(start + maxTokens, history.count)
                    if start < end {
                        return Array(history[start..<end])
                    }
                }
            }
        }
        return []
    }
}

// MARK: - Orchestrator

public final class SpecDecodeOrchestrator: @unchecked Sendable {
    public var K: Int = 3

    private var rollingAccept: Double = 0.5
    private let alpha: Double = 0.15
    public var fallbackThreshold: Double = 0.20
    private var consecutiveMisses: Int = 0
    private let maxConsecutiveMisses: Int = 3

    private var externalDrafter: ExternalDrafter?
    public let ngramDrafter = NgramDrafter()

    private var totalDrafted: Int = 0
    private var totalAccepted: Int = 0
    private var totalBursts: Int = 0

    public init() {}

    public func setExternalDrafter(_ d: ExternalDrafter?) {
        self.externalDrafter = d
    }

    /// Run one speculative burst. All LlamaContext calls batched
    /// into minimal actor hops via combined methods.
    public func step(ctx: LlamaContext) async throws -> SpecDecodeResult? {
        // Get draft proposals (synchronous, no actor hop)
        let proposals = ngramDrafter.draft(maxTokens: K)

        guard !proposals.isEmpty else {
            consecutiveMisses += 1
            return nil  // no proposals → caller uses greedy
        }

        // Reset miss counter — we have proposals to try
        consecutiveMisses = 0

        // Single actor hop: sample anchor + verify batch + argmax all + manage KV
        let result = try await ctx.speculativeVerify(proposals: proposals)

        // Feed accepted tokens to n-gram history
        ngramDrafter.feedTokens(result.emitted)

        // Stats
        totalDrafted += proposals.count
        totalAccepted += result.accepted
        totalBursts += 1

        // Update rolling acceptance
        let rate = Double(result.accepted) / Double(proposals.count)
        rollingAccept = alpha * rate + (1 - alpha) * rollingAccept

        return result
    }

    public var acceptanceRate: Double {
        totalDrafted > 0 ? Double(totalAccepted) / Double(totalDrafted) : 0
    }

    public var burstCount: Int { totalBursts }
}
