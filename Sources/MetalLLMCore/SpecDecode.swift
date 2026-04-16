//
//  SpecDecode.swift
//  MetalLLMCore
//
//  Speculative decoding orchestrator. Supports:
//  - Prompt-lookup (n-gram) drafter (built-in, zero-cost)
//  - External drafter protocol (MTP via ANE, EAGLE-3 — Phase 2+)
//  - Greedy verification with rollback
//
//  The orchestrator runs draft → verify → accept/reject in bursts.
//  K=3 fixed, matching Google LiteRT-LM's MTP configuration.
//

import Foundation

// MARK: - Spec decode result

public struct SpecDecodeResult {
    public let emitted: [Int32]   // accepted tokens + bonus
    public let accepted: Int       // how many draft tokens matched
    public let drafted: Int        // how many were drafted (K)
}

// MARK: - External drafter protocol

public protocol ExternalDrafter: AnyObject, Sendable {
    func draft(lastToken: Int32, maxTokens: Int) async throws -> [Int32]
}

// MARK: - Prompt-lookup drafter

public actor PromptLookupDrafter: ExternalDrafter {
    private var history: [Int32] = []
    private let ngramSize: Int

    public init(ngramSize: Int = 3) {
        self.ngramSize = ngramSize
    }

    public func updateHistory(_ tokens: [Int32]) {
        history.append(contentsOf: tokens)
        if history.count > 4096 {
            history = Array(history.suffix(4096))
        }
    }

    public func draft(lastToken: Int32, maxTokens: Int) async throws -> [Int32] {
        guard history.count >= ngramSize else { return [] }

        let tail = Array(history.suffix(ngramSize))
        let searchEnd = history.count - ngramSize

        for i in stride(from: searchEnd - 1, through: 0, by: -1) {
            var match = true
            for j in 0..<ngramSize {
                if history[i + j] != tail[j] {
                    match = false
                    break
                }
            }
            if match {
                let start = i + ngramSize
                let end = min(start + maxTokens, history.count)
                if start < end {
                    return Array(history[start..<end])
                }
            }
        }
        return []
    }
}

// MARK: - Orchestrator

public final class SpecDecodeOrchestrator: @unchecked Sendable {
    public let K: Int = 3

    private var rollingAccept: Double = 1.0
    private let alpha: Double = 0.05
    public var fallbackThreshold: Double = 0.30

    private var drafter: ExternalDrafter?
    private let promptLookup = PromptLookupDrafter()

    private var lastToken: Int32 = -1

    public init() {}

    public func setDrafter(_ d: ExternalDrafter?) {
        self.drafter = d
    }

    /// Run one speculative burst: draft K tokens, verify, accept prefix.
    /// Returns nil if drafter has nothing — caller should use greedy path.
    public func step(ctx: LlamaContext) async throws -> SpecDecodeResult? {
        let activeDrafter: ExternalDrafter? = drafter
        guard let activeDrafter, rollingAccept >= fallbackThreshold else {
            return nil  // signal caller to use greedy
        }

        // Sample the "anchor" token from the current context
        let anchor = await ctx.sampleAt(-1)

        // Draft K tokens
        let proposals: [Int32]
        do {
            proposals = try await activeDrafter.draft(lastToken: anchor, maxTokens: K)
        } catch {
            return nil
        }

        guard !proposals.isEmpty else {
            return nil
        }

        // Build verify batch: [anchor, draft_0, draft_1, ..., draft_{K-1}]
        let verifyTokens = [anchor] + proposals
        let pos = await ctx.nCur

        try await ctx.verifyBatch(tokens: verifyTokens, startPos: pos)

        // Greedy acceptance: compare verifier argmax against proposals
        var matched = 0
        for k in 0..<proposals.count {
            let verifierChoice = await ctx.argmaxAt(Int32(k))
            if verifierChoice == proposals[k] {
                matched += 1
            } else {
                break
            }
        }

        // Bonus token: verifier's choice at the first disagreement position
        let bonusToken = await ctx.argmaxAt(Int32(matched))

        // Build emitted sequence
        var emitted: [Int32] = [anchor]
        emitted.append(contentsOf: proposals.prefix(matched))
        emitted.append(bonusToken)

        let totalAdvance = Int32(matched + 1)
        // Truncate KV for rejected positions
        let keepPos = pos + totalAdvance
        await ctx.kvRemove(seqId: 0, p0: keepPos, p1: -1)

        // Now decode the bonus token to prepare KV for next step
        try await ctx.verifyBatch(tokens: [bonusToken], startPos: keepPos)
        await ctx.setPosition(keepPos + 1)

        lastToken = bonusToken
        await promptLookup.updateHistory(emitted)

        // Update rolling acceptance
        let rate = proposals.isEmpty ? 0 : Double(matched) / Double(proposals.count)
        rollingAccept = alpha * rate + (1 - alpha) * rollingAccept

        return SpecDecodeResult(
            emitted: emitted,
            accepted: matched,
            drafted: proposals.count
        )
    }
}
