//
//  main.swift
//  MetalLLMBench
//
//  Raw decode bench — measure the actual gap vs llama-bench.
//

import Foundation
import MetalLLMCore
#if canImport(llama)
import llama
#endif

@main
struct BenchMain {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("Usage: MetalLLMBench <path-to-gguf> [max-tokens]")
            return
        }

        let modelPath = args[1]
        let maxTokens = args.count > 2 ? Int(args[2]) ?? 128 : 128

        print("=== Metal-LLM Bench ===")

        // --- Test A: Raw llama.cpp C API (no Swift overhead) ---
        print("\n--- Test A: Raw C API (like llama-bench) ---")
        try await rawBench(modelPath: modelPath, tg: maxTokens)

        // --- Test B: Via LlamaContext actor ---
        print("\n--- Test B: LlamaContext actor ---")
        try await actorBench(modelPath: modelPath, maxTokens: maxTokens)
    }

    static func rawBench(modelPath: String, tg: Int) async throws {
        llama_backend_init()

        var mparams = llama_model_default_params()
        mparams.n_gpu_layers = -1
        guard let model = llama_model_load_from_file(modelPath, mparams) else {
            print("Failed to load model")
            return
        }
        defer { llama_model_free(model) }

        let nThreads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        var cparams = llama_context_default_params()
        cparams.n_ctx = 8192
        cparams.n_batch = 512
        cparams.n_threads = Int32(nThreads)
        cparams.n_threads_batch = Int32(nThreads)

        guard let ctx = llama_init_from_model(model, cparams) else {
            print("Failed to create context")
            return
        }
        defer { llama_free(ctx) }

        guard let vocab = llama_model_get_vocab(model) else { return }
        let sparams = llama_sampler_chain_default_params()
        guard let smpl = llama_sampler_chain_init(sparams) else { return }
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy())
        defer { llama_sampler_free(smpl) }

        // Prefill with a real prompt
        let prompt = "<start_of_turn>user\nExplain quantum computing in detail.<end_of_turn>\n<start_of_turn>model\n"
        let promptTokens = tokenize(vocab: vocab, text: prompt)
        print("Prompt: \(promptTokens.count) tokens")

        var batch = llama_batch_init(512, 0, 1)
        defer { llama_batch_free(batch) }

        // Prefill
        llama_batch_clear(&batch)
        for (i, tok) in promptTokens.enumerated() {
            llama_batch_add(&batch, tok, Int32(i), [0], i == promptTokens.count - 1)
        }
        let ppStart = DispatchTime.now().uptimeNanoseconds
        llama_decode(ctx, batch)
        llama_synchronize(ctx)
        let ppEnd = DispatchTime.now().uptimeNanoseconds
        let ppTime = Double(ppEnd - ppStart) / 1_000_000_000
        print("Prefill: \(String(format: "%.0f", Double(promptTokens.count)/ppTime)) tok/s")

        // Decode
        var nCur = Int32(promptTokens.count)
        let tgStart = DispatchTime.now().uptimeNanoseconds

        for _ in 0..<tg {
            let tok = llama_sampler_sample(smpl, ctx, -1)

            if llama_vocab_is_eog(vocab, tok) { break }

            llama_batch_clear(&batch)
            llama_batch_add(&batch, tok, nCur, [0], true)
            llama_decode(ctx, batch)

            nCur += 1
        }
        llama_synchronize(ctx)

        let tgEnd = DispatchTime.now().uptimeNanoseconds
        let tgTime = Double(tgEnd - tgStart) / 1_000_000_000
        let decoded = Int(nCur) - promptTokens.count
        let tps = Double(decoded) / tgTime
        print("Decode: \(decoded) tokens in \(String(format: "%.2f", tgTime))s = \(String(format: "%.1f", tps)) tok/s")

        llama_backend_free()
    }

    static func actorBench(modelPath: String, maxTokens: Int) async throws {
        let ctx = try await LlamaContext.load(path: modelPath, contextLength: 8192)

        let config = SamplingConfig(temperature: 0.0, maxTokens: Int(maxTokens))
        await ctx.configureSampler(config: config)

        let prompt = "<start_of_turn>user\nExplain quantum computing in detail.<end_of_turn>\n<start_of_turn>model\n"
        let tokens = await ctx.tokenize(prompt, addBOS: true)
        print("Prompt: \(tokens.count) tokens")

        // Prefill
        let ppStart = CFAbsoluteTimeGetCurrent()
        try await ctx.prefill(tokens: tokens)
        let ppTime = CFAbsoluteTimeGetCurrent() - ppStart
        print("Prefill: \(String(format: "%.0f", Double(tokens.count)/ppTime)) tok/s")

        // Decode
        let tgStart = CFAbsoluteTimeGetCurrent()
        var decoded = 0

        for _ in 0..<maxTokens {
            let tok = try await ctx.sampleAndDecode()
            decoded += 1
            if await ctx.isEOG(tok) { break }
        }

        let tgTime = CFAbsoluteTimeGetCurrent() - tgStart
        let tps = Double(decoded) / tgTime
        print("Decode: \(decoded) tokens in \(String(format: "%.2f", tgTime))s = \(String(format: "%.1f", tps)) tok/s")
    }

    // Raw tokenize helper
    static func tokenize(vocab: OpaquePointer, text: String) -> [llama_token] {
        let n = text.utf8.count + 2
        let buf = UnsafeMutablePointer<llama_token>.allocate(capacity: n)
        defer { buf.deallocate() }
        let count = llama_tokenize(vocab, text, Int32(text.utf8.count), buf, Int32(n), true, false)
        return (0..<Int(count)).map { buf[$0] }
    }
}
