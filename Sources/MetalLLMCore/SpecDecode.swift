//
//  SpecDecode.swift
//  MetalLLMCore
//
//  Orchestrator for speculative decoding. Multiple drafters (EAGLE-3, MTP,
//  prompt-lookup) share a single verifier (llama.cpp target model). The orchestrator
//  picks which drafter to use per-step based on acceptance-rate feedback and latency
//  budget, and handles rollback when drafts are rejected.
//

import Foundation

/// A speculative draft — a short run of candidate token ids plus the drafter's own
/// logits (used by rejection sampling when temperature > 0).
public struct Draft {
    public let tokens: [Int32]
    public let logits: [[Float]]?       // per-position drafter logits, optional
    public let source: DrafterKind
}

public enum DrafterKind: String {
    case eagle3
    case mtp
    case promptLookup
}

/// Common interface every drafter implements.
public protocol Drafter: AnyObject {
    var kind: DrafterKind { get }

    /// Produce up to `maxTokens` speculative continuation tokens given the accepted
    /// context so far.
    func draft(context: [Int32], maxTokens: Int) throws -> Draft

    /// Feedback hook so the drafter can update its own state (e.g. EAGLE-3 hidden
    /// states) based on the verifier's acceptance decision.
    func commitAcceptance(accepted: [Int32], rejectedSuffix: [Int32])
}

/// The orchestrator. Skeleton — no scheduling heuristic yet.
public final class SpecDecodeOrchestrator {

    private var drafters: [DrafterKind: Drafter] = [:]
    private var acceptanceRates: [DrafterKind: Double] = [:]

    public init() {}

    public func register(_ drafter: Drafter) {
        drafters[drafter.kind] = drafter
        acceptanceRates[drafter.kind] = 0
    }

    /// Pick a drafter and return a draft for the current step.
    public func proposeDraft(context: [Int32], maxTokens: Int) throws -> Draft? {
        // TODO(Week 5): route by rolling acceptance rate + measured draft latency.
        // TODO(Week 11): Mirror-SD path — run GPU draft and ANE verify in parallel.
        return nil
    }

    /// Called by the engine after the verifier decides which prefix of a draft to keep.
    public func commit(draft: Draft, acceptedCount: Int) {
        // TODO(Week 5): update acceptance stats, forward to the source drafter.
    }
}

/// Verifier interface — the "big" model that scores drafts. In Week 1+ this is
/// `LlamaBridge.decode(returnAllLogits: true)`.
public protocol Verifier: AnyObject {
    /// Score a draft: returns the number of leading tokens accepted plus the
    /// bonus token sampled from the verifier's own distribution at the first
    /// rejected position.
    func verify(context: [Int32], draft: Draft, temperature: Float) throws -> (accepted: Int, bonus: Int32)
}
