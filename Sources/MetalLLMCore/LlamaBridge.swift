//
//  LlamaBridge.swift
//  MetalLLMCore
//
//  Swift wrapper around the llama.cpp C API. Responsible for model loading, batch
//  construction, forward passes, and KV-cache manipulation. This is the narrow seam
//  between Swift land and the C/Metal kernels.
//

import Foundation

/// Opaque handle for a llama.cpp model + context. Real type arrives in Week 1.
public struct LlamaContextHandle {
    // TODO(Week 1): wrap `llama_context *` + `llama_model *`.
    let raw: OpaquePointer?
}

/// Thin wrapper over the llama.cpp C API.
public final class LlamaBridge {

    private var handle: LlamaContextHandle?

    public init() {}

    // MARK: - Lifecycle

    /// Load a GGUF model and create a Metal-backed context.
    public func loadModel(path: URL, contextLength: Int) throws {
        // TODO(Week 1): llama_model_load_from_file + llama_init_from_model.
        // TODO(Week 1): set n_gpu_layers = -1 to force all layers onto Metal.
        fatalError("LlamaBridge.loadModel not implemented")
    }

    /// Free the context and model.
    public func unload() {
        // TODO(Week 1): llama_free + llama_model_free.
    }

    // MARK: - Tokenization

    public func tokenize(_ text: String, addBOS: Bool) throws -> [Int32] {
        // TODO(Week 1): llama_tokenize.
        fatalError("LlamaBridge.tokenize not implemented")
    }

    public func detokenize(_ tokens: [Int32]) throws -> String {
        // TODO(Week 1): llama_detokenize / llama_token_to_piece.
        fatalError("LlamaBridge.detokenize not implemented")
    }

    // MARK: - Forward passes

    /// Decode a batch of tokens. Returns logits for the last position (or all positions
    /// when `returnAllLogits` is true — needed by the spec-decode verifier).
    public func decode(tokens: [Int32], positions: [Int32], returnAllLogits: Bool) throws -> [Float] {
        // TODO(Week 1): llama_batch + llama_decode.
        // TODO(Week 5): expose per-position logits for verifier use.
        fatalError("LlamaBridge.decode not implemented")
    }

    // MARK: - KV cache

    /// Rewind the KV cache to a given position (used when a speculative draft is rejected).
    public func kvTruncate(to position: Int32) {
        // TODO(Week 5): llama_kv_cache_seq_rm.
    }

    /// Export the current KV cache so the ANE bridge can ingest / produce compatible state.
    public func kvExport() throws -> Data {
        // TODO(Week 7): zero-copy export via IOSurface if layouts match.
        fatalError("LlamaBridge.kvExport not implemented")
    }
}
