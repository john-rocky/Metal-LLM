//
//  main.swift
//  MetalLLMBench — comprehensive A/B/C benchmark
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
        let prompt = "<start_of_turn>user\nExplain quantum computing in detail, including qubits, superposition, and entanglement.<end_of_turn>\n<start_of_turn>model\n"

        print("=== Metal-LLM Bench ===")
        print("Model: \(modelPath)")
        print("Max tokens: \(maxTokens)\n")

        // Use a single Engine instance for all tests (avoid GPU resource contention)
        let engine = MetalLLMEngine()
        try await engine.load(modelPath: URL(fileURLWithPath: modelPath))

        // --- Test C: Engine speculative first ---
        print("--- C: Engine speculative ---")
        let specTps = try await engineBenchShared(engine: engine, prompt: prompt, maxTokens: maxTokens, speculative: true)

        // --- Test B: Engine greedy ---
        print("\n--- B: Engine greedy (no spec) ---")
        let greedyTps = try await engineBenchShared(engine: engine, prompt: prompt, maxTokens: maxTokens, speculative: false)

        // Raw C benchmark last
        print("\n--- A: Raw C API (reference) ---")
        let rawTps = try await rawBench(modelPath: modelPath, prompt: prompt, tg: maxTokens)

        // --- Summary ---
        print("\n=== Summary ===")
        print("A) Raw C:           \(String(format: "%6.1f", rawTps)) tok/s")
        print("B) Engine greedy:   \(String(format: "%6.1f", greedyTps)) tok/s")
        print("C) Engine spec:     \(String(format: "%6.1f", specTps)) tok/s")
        print("B/A overhead:       \(String(format: "%.1f%%", (1.0 - greedyTps/rawTps) * 100))")
        if specTps > greedyTps {
            print("C/B spec speedup:   \(String(format: "%.2fx", specTps/greedyTps))")
        }
    }

    // MARK: - Raw C bench

    static func rawBench(modelPath: String, prompt: String, tg: Int) async throws -> Double {
        llama_backend_init()
        defer { llama_backend_free() }

        var mparams = llama_model_default_params()
        mparams.n_gpu_layers = -1
        guard let model = llama_model_load_from_file(modelPath, mparams) else { return 0 }
        defer { llama_model_free(model) }

        var cparams = llama_context_default_params()
        cparams.n_ctx = 8192
        cparams.n_batch = 2048
        let nThreads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        cparams.n_threads = Int32(nThreads)
        cparams.n_threads_batch = Int32(nThreads)

        guard let ctx = llama_init_from_model(model, cparams) else { return 0 }
        defer { llama_free(ctx) }
        guard let vocab = llama_model_get_vocab(model) else { return 0 }

        let sparams = llama_sampler_chain_default_params()
        guard let smpl = llama_sampler_chain_init(sparams) else { return 0 }
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy())
        defer { llama_sampler_free(smpl) }

        let promptTokens = tokenize(vocab: vocab, text: prompt)
        print("  Prompt: \(promptTokens.count) tokens")

        var batch = llama_batch_init(2048, 0, 1)
        defer { llama_batch_free(batch) }

        // Prefill
        llama_batch_clear(&batch)
        for (i, tok) in promptTokens.enumerated() {
            llama_batch_add(&batch, tok, Int32(i), [0], i == promptTokens.count - 1)
        }
        llama_decode(ctx, batch)
        llama_synchronize(ctx)

        // Decode
        var nCur = Int32(promptTokens.count)
        var decoded = 0
        let t0 = DispatchTime.now().uptimeNanoseconds

        for _ in 0..<tg {
            let tok = llama_sampler_sample(smpl, ctx, -1)
            if llama_vocab_is_eog(vocab, tok) { break }
            llama_batch_clear(&batch)
            llama_batch_add(&batch, tok, nCur, [0], true)
            llama_decode(ctx, batch)
            nCur += 1
            decoded += 1
        }
        llama_synchronize(ctx)

        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        let tps = Double(decoded) / elapsed
        print("  Decode: \(decoded) tokens, \(String(format: "%.2f", elapsed))s, \(String(format: "%.1f", tps)) tok/s")
        return tps
    }

    // MARK: - Engine bench (shared instance)

    static func engineBenchShared(engine: MetalLLMEngine, prompt: String, maxTokens: Int, speculative: Bool) async throws -> Double {
        let config = SamplingConfig(
            temperature: 0.0,
            maxTokens: Int(maxTokens),
            speculative: speculative
        )

        nonisolated(unsafe) var output = ""
        nonisolated(unsafe) var tokenCount = 0
        let t0 = CFAbsoluteTimeGetCurrent()

        try await engine.generate(prompt: prompt, config: config) { event in
            output += event.token
            tokenCount += 1
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        let tps = elapsed > 0 ? Double(tokenCount) / elapsed : 0

        let snap = await engine.metricsSnapshot()
        print("  Output: \(output.prefix(120))...")
        print("  Decode: \(tokenCount) tokens, \(String(format: "%.2f", elapsed))s, \(String(format: "%.1f", tps)) tok/s")
        if speculative {
            print("  Acceptance: \(String(format: "%.1f%%", snap.acceptanceRate * 100)), bursts: \(snap.specBursts)")
        }
        return tps
    }

    static func tokenize(vocab: OpaquePointer, text: String) -> [llama_token] {
        let n = text.utf8.count + 2
        let buf = UnsafeMutablePointer<llama_token>.allocate(capacity: n)
        defer { buf.deallocate() }
        let count = llama_tokenize(vocab, text, Int32(text.utf8.count), buf, Int32(n), true, false)
        return (0..<Int(count)).map { buf[$0] }
    }
}
