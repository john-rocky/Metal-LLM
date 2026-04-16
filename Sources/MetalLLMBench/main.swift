//
//  main.swift
//  MetalLLMBench
//
//  Smoke test + benchmark for Metal-LLM engine on Mac Studio.
//

import Foundation
import MetalLLMCore

@main
struct BenchMain {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("Usage: MetalLLMBench <path-to-gguf> [max-tokens]")
            return
        }

        let modelPath = args[1]
        let maxTokens = args.count > 2 ? Int(args[2]) ?? 64 : 64

        print("=== Metal-LLM Bench ===")
        print("Model: \(modelPath)")
        print("Max tokens: \(maxTokens)")

        // Load model directly via LlamaContext
        print("\nLoading model...")
        let ctx = try await LlamaContext.load(path: modelPath, contextLength: 8192)
        print("Loaded: \(await ctx.modelInfo())")
        print("n_vocab=\(await ctx.nVocab), n_ctx=\(await ctx.nCtx)")

        // Configure greedy sampler
        let config = SamplingConfig(temperature: 0.0, maxTokens: Int(maxTokens), speculative: false)
        await ctx.configureSampler(config: config)

        // Tokenize
        let prompt = "<start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\n"
        let tokens = await ctx.tokenize(prompt, addBOS: true)
        print("Prompt tokens: \(tokens.count)")

        // Prefill
        print("Prefilling...")
        let prefillStart = CFAbsoluteTimeGetCurrent()
        try await ctx.prefill(tokens: tokens)
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart
        print("Prefill: \(String(format: "%.2f", prefillTime))s (\(String(format: "%.0f", Double(tokens.count)/prefillTime)) tok/s)")

        // Decode loop
        print("\nDecoding...")
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var output = ""
        var tokenCount = 0

        for _ in 0..<maxTokens {
            let tokenId = try await ctx.sampleAndDecode()
            let text = await ctx.detokenize(tokenId)
            let isEOG = await ctx.isEOG(tokenId)

            output += text
            tokenCount += 1

            if tokenCount <= 5 || tokenCount % 20 == 0 {
                let elapsed = CFAbsoluteTimeGetCurrent() - decodeStart
                let tps = Double(tokenCount) / elapsed
                print("  [\(tokenCount)] id=\(tokenId) eog=\(isEOG) '\(text.prefix(20))' \(String(format: "%.1f", tps)) tok/s")
            }

            if isEOG { break }
        }

        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
        let tps = Double(tokenCount) / decodeTime

        print("\n=== Result ===")
        print("Output: \(output.prefix(300))")
        print("Tokens: \(tokenCount)")
        print("Decode: \(String(format: "%.2f", decodeTime))s")
        print("Speed: \(String(format: "%.1f", tps)) tok/s")
    }
}
