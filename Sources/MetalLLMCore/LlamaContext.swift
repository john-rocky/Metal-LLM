//
//  LlamaContext.swift
//  MetalLLMCore
//
//  Actor wrapping a llama.cpp context. Adapted from upstream
//  examples/llama.swiftui/llama.cpp.swift/LibLlama.swift with extensions
//  for speculative decoding, multi-batch verify, and metrics.
//

import Foundation
import llama

// MARK: - Batch helpers

public func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

public func llama_batch_add(
    _ batch: inout llama_batch,
    _ id: llama_token,
    _ pos: llama_pos,
    _ seq_ids: [llama_seq_id],
    _ logits: Bool
) {
    batch.token   [Int(batch.n_tokens)] = id
    batch.pos     [Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
    for i in 0..<seq_ids.count {
        batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
    }
    batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0
    batch.n_tokens += 1
}

// MARK: - LlamaContext actor

public actor LlamaContext {
    // nonisolated(unsafe) needed for deinit access in Swift 6 strict concurrency.
    // These are only mutated within the actor's isolation domain.
    private nonisolated(unsafe) let model: OpaquePointer
    private nonisolated(unsafe) let context: OpaquePointer
    private nonisolated(unsafe) let vocab: OpaquePointer
    private nonisolated(unsafe) var sampling: UnsafeMutablePointer<llama_sampler>
    private nonisolated(unsafe) var batch: llama_batch

    private var temporaryInvalidCChars: [CChar] = []

    public let nVocab: Int32
    public let nCtx: UInt32
    public private(set) var nCur: Int32 = 0
    public private(set) var nDecode: Int32 = 0

    // MARK: Init

    init(model: OpaquePointer, context: OpaquePointer) {
        self.model = model
        self.context = context
        self.vocab = llama_model_get_vocab(model)
        self.nVocab = llama_vocab_n_tokens(self.vocab)
        self.nCtx = llama_n_ctx(context)
        self.batch = llama_batch_init(2048, 0, 1)

        // Default sampler chain
        let sparams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(sparams)
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.7))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_top_k(40))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_top_p(0.95, 1))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(UInt32.random(in: 0..<UInt32.max)))
    }

    deinit {
        llama_sampler_free(sampling)
        llama_batch_free(batch)
        llama_free(context)
        llama_model_free(model)
    }

    // MARK: - Factory

    public static func load(path: String, contextLength: UInt32 = 8192) throws -> LlamaContext {
        llama_backend_init()

        var modelParams = llama_model_default_params()
        #if targetEnvironment(simulator)
        modelParams.n_gpu_layers = 0
        #else
        modelParams.n_gpu_layers = -1   // all layers on GPU
        #endif

        guard let model = llama_model_load_from_file(path, modelParams) else {
            throw LlamaContextError.modelLoadFailed(path)
        }

        let nThreads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = contextLength
        ctxParams.n_batch = 512
        ctxParams.n_threads = Int32(nThreads)
        ctxParams.n_threads_batch = Int32(nThreads)
        // FlashAttention disabled for now - enable after verifying stability
        // ctxParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED

        guard let context = llama_init_from_model(model, ctxParams) else {
            llama_model_free(model)
            throw LlamaContextError.contextCreateFailed
        }

        return LlamaContext(model: model, context: context)
    }

    // MARK: - Sampler configuration

    public func configureSampler(config: SamplingConfig) {
        llama_sampler_free(sampling)
        let sparams = llama_sampler_chain_default_params()
        sampling = llama_sampler_chain_init(sparams)

        if config.temperature > 0 {
            llama_sampler_chain_add(sampling, llama_sampler_init_top_k(Int32(config.topK)))
            llama_sampler_chain_add(sampling, llama_sampler_init_top_p(config.topP, 1))
            llama_sampler_chain_add(sampling, llama_sampler_init_temp(config.temperature))
            llama_sampler_chain_add(sampling, llama_sampler_init_dist(UInt32.random(in: 0..<UInt32.max)))
        } else {
            // Greedy
            llama_sampler_chain_add(sampling, llama_sampler_init_greedy())
        }
    }

    // MARK: - Tokenization

    public func tokenize(_ text: String, addBOS: Bool) -> [llama_token] {
        let utf8Count = text.utf8.count
        let nTokens = utf8Count + (addBOS ? 1 : 0) + 1
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: nTokens)
        defer { tokens.deallocate() }

        let count = llama_tokenize(vocab, text, Int32(utf8Count), tokens, Int32(nTokens), addBOS, false)
        return (0..<Int(count)).map { tokens[$0] }
    }

    public func detokenize(_ token: llama_token) -> String {
        let cchars = tokenToPiece(token)
        temporaryInvalidCChars.append(contentsOf: cchars)

        if let string = String(validatingUTF8: temporaryInvalidCChars + [0]) {
            temporaryInvalidCChars.removeAll()
            return string
        } else if (0..<temporaryInvalidCChars.count).contains(where: {
            $0 != 0 && String(validatingUTF8: Array(temporaryInvalidCChars.suffix($0)) + [0]) != nil
        }) {
            let string = String(cString: temporaryInvalidCChars + [0])
            temporaryInvalidCChars.removeAll()
            return string
        }
        return ""
    }

    public func isEOG(_ token: llama_token) -> Bool {
        llama_vocab_is_eog(vocab, token)
    }

    public func eotToken() -> llama_token {
        llama_vocab_eot(vocab)
    }

    // MARK: - Prefill

    public func prefill(tokens: [llama_token]) throws {
        let batchSize = Int(llama_n_batch(context))
        let ubatchSize = Int(llama_n_ubatch(context))
        let msg = "[prefill] n_tokens=\(tokens.count) n_batch=\(batchSize) n_ubatch=\(ubatchSize) n_ctx=\(nCtx)"
        print(msg)
        fputs(msg + "\n", stderr)

        // Process in chunks of n_batch
        var pos: Int32 = 0
        var remaining = tokens
        while !remaining.isEmpty {
            let chunk = Array(remaining.prefix(batchSize))
            remaining = Array(remaining.dropFirst(batchSize))

            llama_batch_clear(&batch)
            for (i, token) in chunk.enumerated() {
                let isLast = remaining.isEmpty && i == chunk.count - 1
                llama_batch_add(&batch, token, pos + Int32(i), [0], isLast)
            }

            let rc = llama_decode(context, batch)
            guard rc == 0 else {
                print("[LlamaContext] llama_decode failed during prefill, rc=\(rc), pos=\(pos), chunk_size=\(chunk.count)")
                throw LlamaContextError.decodeFailed("prefill rc=\(rc) at pos=\(pos)")
            }
            pos += Int32(chunk.count)
        }

        nCur = pos
        nDecode = 0
    }

    // MARK: - Single-token decode

    public func sampleAndDecode() throws -> llama_token {
        let newToken = llama_sampler_sample(sampling, context, -1)

        llama_batch_clear(&batch)
        llama_batch_add(&batch, newToken, nCur, [0], true)

        guard llama_decode(context, batch) == 0 else {
            throw LlamaContextError.decodeFailed("decode step \(nCur)")
        }

        nCur += 1
        nDecode += 1
        return newToken
    }

    // MARK: - Multi-token verify (for speculative decoding)

    /// Decode K+1 tokens in a single batch (for spec-decode verify).
    /// Returns logits pointers for each position.
    public func verifyBatch(tokens: [llama_token], startPos: Int32) throws {
        llama_batch_clear(&batch)
        for (i, token) in tokens.enumerated() {
            llama_batch_add(&batch, token, startPos + Int32(i), [0], true)
        }

        guard llama_decode(context, batch) == 0 else {
            throw LlamaContextError.decodeFailed("verify batch")
        }
    }

    /// Get logits at position i within the current batch.
    /// Returns an UnsafeBufferPointer to avoid copying 262K floats.
    public func logitsAt(_ i: Int32) -> UnsafeBufferPointer<Float> {
        let ptr = llama_get_logits_ith(context, i)!
        return UnsafeBufferPointer(start: ptr, count: Int(nVocab))
    }

    /// Argmax over logits at batch position i.
    public func argmaxAt(_ i: Int32) -> llama_token {
        let logits = logitsAt(i)
        var maxVal: Float = -Float.infinity
        var maxIdx: Int32 = 0
        for j in 0..<logits.count {
            if logits[j] > maxVal {
                maxVal = logits[j]
                maxIdx = Int32(j)
            }
        }
        return maxIdx
    }

    /// Sample from the sampler chain at batch position i.
    public func sampleAt(_ i: Int32) -> llama_token {
        llama_sampler_sample(sampling, context, i)
    }

    // MARK: - KV cache management

    /// Remove KV entries from position p0 to p1 (exclusive). Pass p1 = -1 for "to end".
    public func kvRemove(seqId: llama_seq_id = 0, p0: llama_pos, p1: llama_pos) {
        let mem = llama_get_memory(context)
        llama_memory_seq_rm(mem, seqId, p0, p1)
    }

    /// Clear all KV cache entries.
    public func kvClear() {
        let mem = llama_get_memory(context)
        llama_memory_clear(mem, false)
    }

    /// Update current position tracker after speculative accept.
    public func advancePosition(by n: Int32) {
        nCur += n
        nDecode += n
    }

    public func setPosition(_ pos: Int32) {
        nCur = pos
    }

    // MARK: - Info

    public func modelInfo() -> String {
        let buf = UnsafeMutablePointer<CChar>.allocate(capacity: 256)
        defer { buf.deallocate() }
        buf.initialize(repeating: 0, count: 256)
        let n = llama_model_desc(model, buf, 256)
        return String(cString: buf)
    }

    public func modelSize() -> UInt64 {
        llama_model_size(model)
    }

    // MARK: - Private

    private func tokenToPiece(_ token: llama_token) -> [CChar] {
        let buf = UnsafeMutablePointer<CChar>.allocate(capacity: 64)
        defer { buf.deallocate() }
        buf.initialize(repeating: 0, count: 64)

        let n = llama_token_to_piece(vocab, token, buf, 64, 0, false)
        if n < 0 {
            let bigBuf = UnsafeMutablePointer<CChar>.allocate(capacity: Int(-n))
            defer { bigBuf.deallocate() }
            bigBuf.initialize(repeating: 0, count: Int(-n))
            let n2 = llama_token_to_piece(vocab, token, bigBuf, -n, 0, false)
            return Array(UnsafeBufferPointer(start: bigBuf, count: Int(n2)))
        }
        return Array(UnsafeBufferPointer(start: buf, count: Int(n)))
    }
}

// MARK: - Errors

public enum LlamaContextError: Error, LocalizedError {
    case modelLoadFailed(String)
    case contextCreateFailed
    case decodeFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let path): return "Failed to load model: \(path)"
        case .contextCreateFailed: return "Failed to create llama context"
        case .decodeFailed(let step): return "llama_decode failed at: \(step)"
        }
    }
}
