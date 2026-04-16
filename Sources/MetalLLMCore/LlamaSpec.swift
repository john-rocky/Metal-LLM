//
//  LlamaSpec.swift
//  MetalLLMCore
//
//  Swift wrapper for the llama_spec_* C ABI (Metal-LLM fork).
//  Bridges external drafters (MTP via CoreML, prompt-lookup) to
//  llama.cpp's speculative decoding infrastructure.
//

import Foundation
#if canImport(llama)
import llama

// MARK: - Swift drafter protocol

/// Protocol for external drafters that can be plugged into the spec-decode pipeline.
public protocol SwiftDrafter: AnyObject, Sendable {
    /// Produce up to `maxTokens` draft tokens given the last accepted token.
    func draft(lastToken: llama_token, maxTokens: Int) async throws -> [llama_token]
}

// MARK: - LlamaSpec actor

/// Actor wrapping a llama_spec_t handle.
/// Manages the lifecycle of external drafter registration and spec-decode stats.
public final class LlamaSpec: @unchecked Sendable {
    private let handle: llama_spec_t
    private var currentDrafter: (any SwiftDrafter)?

    // Stored reference to prevent the closure context from being freed
    private var trampolineBox: TrampolineBox?

    public init(context: OpaquePointer) {
        var params = llama_spec_default_params()
        params.n_draft_max = 3  // K=3, matching Google LiteRT-LM
        self.handle = llama_spec_create(context, params)
    }

    deinit {
        llama_spec_destroy(handle)
    }

    // MARK: - Drafter registration

    /// Set the active external drafter. Pass nil to disable spec decode.
    public func setDrafter(_ drafter: (any SwiftDrafter)?) {
        self.currentDrafter = drafter

        guard let drafter else {
            llama_spec_set_external_drafter(handle, nil, nil)
            trampolineBox = nil
            return
        }

        // Create a box to hold the drafter reference for the C callback
        let box = TrampolineBox(drafter: drafter)
        self.trampolineBox = box
        let opaque = Unmanaged.passUnretained(box).toOpaque()

        llama_spec_set_external_drafter(handle, LlamaSpec.trampoline, opaque)
    }

    // MARK: - Draft

    /// Request draft tokens from the registered drafter via the C ABI.
    public func draft(
        promptTokens: [llama_token],
        lastToken: llama_token,
        maxTokens: Int32 = 3
    ) -> [llama_token] {
        var outTokens = [llama_token](repeating: 0, count: Int(maxTokens))
        let n = promptTokens.withUnsafeBufferPointer { prompt in
            outTokens.withUnsafeMutableBufferPointer { out in
                llama_spec_draft(
                    handle,
                    prompt.baseAddress,
                    Int32(prompt.count),
                    lastToken,
                    out.baseAddress,
                    nil,    // no probability output
                    maxTokens
                )
            }
        }
        if n > 0 {
            return Array(outTokens.prefix(Int(n)))
        }
        return []
    }

    // MARK: - Stats

    public func accept(_ count: Int32) {
        llama_spec_accept(handle, count)
    }

    public func stats() -> (drafted: Int64, accepted: Int64) {
        var nd: Int64 = 0, na: Int64 = 0
        llama_spec_get_stats(handle, &nd, &na, nil, nil)
        return (nd, na)
    }

    public func reset() {
        llama_spec_reset(handle)
    }

    // MARK: - C trampoline

    /// C-compatible trampoline function that bridges to the Swift drafter.
    /// This runs synchronously — the drafter.draft() is called via a blocking
    /// semaphore since the C callback can't be async.
    private static let trampoline: llama_spec_drafter_cb = {
        (ctx_user, prompt_tokens, n_prompt, id_last, n_max, out_tokens, out_probs, n_written) -> Int32 in

        guard let ctx_user, let out_tokens, let n_written else { return -1 }

        let box = Unmanaged<TrampolineBox>.fromOpaque(ctx_user).takeUnretainedValue()
        let drafter = box.drafter

        // Bridge async to sync via semaphore (acceptable because draft is fast, ~3-5ms)
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var result: [llama_token] = []

        let d = drafter
        let maxTok = Int(n_max)
        let lastTok = id_last

        Task { @Sendable in
            do {
                result = try await d.draft(lastToken: lastTok, maxTokens: maxTok)
            } catch {
                // Drafter failed
            }
            sem.signal()
        }
        sem.wait()

        if !result.isEmpty {
            let count = min(result.count, Int(n_max))
            for i in 0..<count {
                out_tokens[i] = result[i]
            }
            n_written.pointee = Int32(count)
        } else {
            n_written.pointee = 0
        }
        return 0
    }
}

// MARK: - Trampoline box

/// Box to hold a SwiftDrafter reference for the C callback trampoline.
private final class TrampolineBox: @unchecked Sendable {
    let drafter: any SwiftDrafter
    init(drafter: any SwiftDrafter) {
        self.drafter = drafter
    }
}

#endif
