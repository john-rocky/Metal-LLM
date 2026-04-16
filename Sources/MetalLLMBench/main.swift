//
//  main.swift
//  MetalLLMBench — optimization sweep
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
        let prompt = "<start_of_turn>user\nWhat is quantum computing?<end_of_turn>\n<start_of_turn>model\n"

        print("=== Metal-LLM Optimization Sweep ===")
        print("Model: \((modelPath as NSString).lastPathComponent)")
        print("Max tokens: \(maxTokens)\n")

        // Sweep configurations
        let configs: [(name: String, nCtx: UInt32, flashAttn: Bool, kvType: Int32)] = [
            ("baseline (8K, no FA)",        8192, false, -1),
            ("flash_attn ON (8K)",          8192, true,  -1),
            ("ctx=2K, no FA",               2048, false, -1),
            ("ctx=2K + flash_attn",         2048, true,  -1),
            ("ctx=4K + flash_attn",         4096, true,  -1),
        ]

        var results: [(String, Double)] = []

        for config in configs {
            print("--- \(config.name) ---")
            let tps = try benchConfig(
                modelPath: modelPath,
                prompt: prompt,
                maxTokens: maxTokens,
                nCtx: config.nCtx,
                flashAttn: config.flashAttn
            )
            results.append((config.name, tps))
            print("")
        }

        // Summary
        print("=== Summary ===")
        let baseline = results[0].1
        for (name, tps) in results {
            let delta = baseline > 0 ? (tps / baseline - 1) * 100 : 0
            let sign = delta >= 0 ? "+" : ""
            print("  \(String(format: "%-30s", (name as NSString).utf8String!))  \(String(format: "%6.1f", tps)) tok/s  (\(sign)\(String(format: "%.1f", delta))%)")
        }
    }

    static func benchConfig(
        modelPath: String,
        prompt: String,
        maxTokens: Int,
        nCtx: UInt32,
        flashAttn: Bool
    ) throws -> Double {
        llama_backend_init()
        defer { llama_backend_free() }

        var mparams = llama_model_default_params()
        mparams.n_gpu_layers = -1
        guard let model = llama_model_load_from_file(modelPath, mparams) else { return 0 }
        defer { llama_model_free(model) }

        var cparams = llama_context_default_params()
        cparams.n_ctx = nCtx
        cparams.n_batch = 2048
        let nThreads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        cparams.n_threads = Int32(nThreads)
        cparams.n_threads_batch = Int32(nThreads)
        if flashAttn {
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED
        }

        guard let ctx = llama_init_from_model(model, cparams) else {
            print("  context init failed (flash_attn=\(flashAttn) may not be supported)")
            return 0
        }
        defer { llama_free(ctx) }
        guard let vocab = llama_model_get_vocab(model) else { return 0 }

        let sparams = llama_sampler_chain_default_params()
        guard let smpl = llama_sampler_chain_init(sparams) else { return 0 }
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy())
        defer { llama_sampler_free(smpl) }

        // Tokenize
        let promptTokens = tokenize(vocab: vocab, text: prompt)

        var batch = llama_batch_init(2048, 0, 1)
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
        let ppTps = Double(promptTokens.count) / (Double(ppEnd - ppStart) / 1e9)

        // Decode
        var nCur = Int32(promptTokens.count)
        var decoded = 0
        let t0 = DispatchTime.now().uptimeNanoseconds

        for _ in 0..<maxTokens {
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
        print("  pp=\(String(format: "%.0f", ppTps)) tok/s, tg=\(decoded) tokens, \(String(format: "%.1f", tps)) tok/s")
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
