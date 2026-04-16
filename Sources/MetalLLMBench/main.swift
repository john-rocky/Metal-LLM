//
//  main.swift
//  MetalLLMBench — custom Metal kernel benchmark
//

import Foundation
import MetalLLMCore
import Metal
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

        print("=== Metal-LLM Benchmark Suite ===\n")

        // Test 1: Custom Metal kernel benchmark
        print("--- 1: Custom Metal Kernel Test ---")
        try benchCustomKernels()

        // Test 2: llama.cpp baseline
        print("\n--- 2: llama.cpp Greedy Decode ---")
        let llamaTps = try benchLlamaCpp(modelPath: modelPath, maxTokens: maxTokens)

        // Test 3: Engine path
        print("\n--- 3: Engine Greedy Decode ---")
        let engineTps = try await benchEngine(modelPath: modelPath, maxTokens: maxTokens)

        print("\n=== Summary ===")
        print("  Custom kernel matvec:  see above")
        print("  llama.cpp decode:      \(String(format: "%.1f", llamaTps)) tok/s")
        print("  Engine decode:         \(String(format: "%.1f", engineTps)) tok/s")
    }

    // MARK: - Custom kernel benchmark

    static func benchCustomKernels() throws {
        let runtime = try MetalRuntime()

        // Benchmark: matvec Q4_0 — simulate one projection layer
        // Gemma 4 Q projection: input=1536 → output=2048 (8 heads × 256 head_dim)
        let K: UInt32 = 1536
        let N: UInt32 = 2048

        // Create random FP16 input
        var inputData = [Float16](repeating: 0, count: Int(K))
        for i in 0..<Int(K) { inputData[i] = Float16.random(in: -1...1) }

        // Create random Q4_0 weight blocks
        // Q4_0: 18 bytes per block of 32 elements
        let nBlocks = Int(K) / 32
        let totalBlocks = Int(N) * nBlocks
        var weightData = [UInt8](repeating: 0, count: totalBlocks * 18)
        for i in 0..<weightData.count { weightData[i] = UInt8.random(in: 0...255) }

        guard let inputBuf = runtime.makeBuffer(data: inputData),
              let weightBuf = runtime.makeBuffer(data: weightData),
              let outputBuf = runtime.makeBuffer(length: Int(N) * 2) else {
            print("  Buffer allocation failed")
            return
        }

        guard let pipeline = runtime.pipeline("kernel_matvec_q4_0") else {
            print("  Pipeline not found")
            return
        }

        // Warmup
        for _ in 0..<5 {
            runtime.execute { enc in
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(inputBuf, offset: 0, index: 0)
                enc.setBuffer(weightBuf, offset: 0, index: 1)
                enc.setBuffer(outputBuf, offset: 0, index: 2)
                var k = K, n = N
                enc.setBytes(&k, length: 4, index: 3)
                enc.setBytes(&n, length: 4, index: 4)
                enc.dispatchThreads(
                    MTLSize(width: Int(N), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(Int(N), pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                )
            }
        }

        // Benchmark: 1000 iterations
        let iterations = 1000
        let t0 = DispatchTime.now().uptimeNanoseconds

        for _ in 0..<iterations {
            runtime.execute { enc in
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(inputBuf, offset: 0, index: 0)
                enc.setBuffer(weightBuf, offset: 0, index: 1)
                enc.setBuffer(outputBuf, offset: 0, index: 2)
                var k = K, n = N
                enc.setBytes(&k, length: 4, index: 3)
                enc.setBytes(&n, length: 4, index: 4)
                enc.dispatchThreads(
                    MTLSize(width: Int(N), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(Int(N), pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                )
            }
        }

        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        let perCall = elapsed / Double(iterations) * 1e6  // microseconds
        let gflops = Double(2 * K * N) / (elapsed / Double(iterations)) / 1e9

        print("  matvec Q4_0 [\(K) → \(N)]:")
        print("    \(String(format: "%.1f", perCall)) μs/call, \(String(format: "%.1f", gflops)) GFLOPS")
        print("    \(iterations) iterations in \(String(format: "%.2f", elapsed))s")

        // Also test fused RMSNorm + MatVec
        guard let normWBuf = runtime.makeBuffer(data: [Float16](repeating: Float16(1.0), count: Int(K))),
              let normedBuf = runtime.makeBuffer(length: Int(K) * 2),
              let fusedPipeline = runtime.pipeline("kernel_fused_rmsnorm_matvec_q4_0") else {
            print("  Fused pipeline not found")
            return
        }

        // Warmup
        for _ in 0..<5 {
            runtime.execute { enc in
                enc.setComputePipelineState(fusedPipeline)
                enc.setBuffer(inputBuf, offset: 0, index: 0)
                enc.setBuffer(normWBuf, offset: 0, index: 1)
                enc.setBuffer(weightBuf, offset: 0, index: 2)
                enc.setBuffer(outputBuf, offset: 0, index: 3)
                enc.setBuffer(normedBuf, offset: 0, index: 4)
                var k = K, n = N, eps: Float = 1e-6
                enc.setBytes(&k, length: 4, index: 5)
                enc.setBytes(&n, length: 4, index: 6)
                enc.setBytes(&eps, length: 4, index: 7)
                enc.dispatchThreads(
                    MTLSize(width: Int(N), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(Int(N), fusedPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                )
            }
        }

        let t1 = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<iterations {
            runtime.execute { enc in
                enc.setComputePipelineState(fusedPipeline)
                enc.setBuffer(inputBuf, offset: 0, index: 0)
                enc.setBuffer(normWBuf, offset: 0, index: 1)
                enc.setBuffer(weightBuf, offset: 0, index: 2)
                enc.setBuffer(outputBuf, offset: 0, index: 3)
                enc.setBuffer(normedBuf, offset: 0, index: 4)
                var k = K, n = N, eps: Float = 1e-6
                enc.setBytes(&k, length: 4, index: 5)
                enc.setBytes(&n, length: 4, index: 6)
                enc.setBytes(&eps, length: 4, index: 7)
                enc.dispatchThreads(
                    MTLSize(width: Int(N), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(Int(N), fusedPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                )
            }
        }

        let elapsed2 = Double(DispatchTime.now().uptimeNanoseconds - t1) / 1e9
        let perCall2 = elapsed2 / Double(iterations) * 1e6

        print("  fused RMSNorm+MatVec [\(K) → \(N)]:")
        print("    \(String(format: "%.1f", perCall2)) μs/call")
        print("    Fusion saves: \(String(format: "%.1f", perCall - perCall2)) μs/call (\(String(format: "%.0f", (1 - perCall2/perCall) * 100))%)")
    }

    // MARK: - llama.cpp bench

    static func benchLlamaCpp(modelPath: String, maxTokens: Int) throws -> Double {
        llama_backend_init()
        defer { llama_backend_free() }

        var mp = llama_model_default_params()
        mp.n_gpu_layers = -1
        guard let model = llama_model_load_from_file(modelPath, mp) else { return 0 }
        defer { llama_model_free(model) }

        var cp = llama_context_default_params()
        cp.n_ctx = 8192; cp.n_batch = 2048
        let nt = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        cp.n_threads = Int32(nt); cp.n_threads_batch = Int32(nt)

        guard let ctx = llama_init_from_model(model, cp) else { return 0 }
        defer { llama_free(ctx) }
        guard let vocab = llama_model_get_vocab(model) else { return 0 }
        guard let smpl = llama_sampler_chain_init(llama_sampler_chain_default_params()) else { return 0 }
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy())
        defer { llama_sampler_free(smpl) }

        let prompt = "<start_of_turn>user\nWhat is quantum computing?<end_of_turn>\n<start_of_turn>model\n"
        let tokens = tokenize(vocab: vocab, text: prompt)
        var batch = llama_batch_init(2048, 0, 1)
        defer { llama_batch_free(batch) }

        llama_batch_clear(&batch)
        for (i, t) in tokens.enumerated() {
            llama_batch_add(&batch, t, Int32(i), [0], i == tokens.count - 1)
        }
        llama_decode(ctx, batch); llama_synchronize(ctx)

        var nCur = Int32(tokens.count)
        var decoded = 0
        let t0 = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<maxTokens {
            let tok = llama_sampler_sample(smpl, ctx, -1)
            if llama_vocab_is_eog(vocab, tok) { break }
            llama_batch_clear(&batch)
            llama_batch_add(&batch, tok, nCur, [0], true)
            llama_decode(ctx, batch)
            nCur += 1; decoded += 1
        }
        llama_synchronize(ctx)
        let tps = Double(decoded) / (Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
        print("  \(decoded) tokens, \(String(format: "%.1f", tps)) tok/s")
        return tps
    }

    // MARK: - Engine bench

    static func benchEngine(modelPath: String, maxTokens: Int) async throws -> Double {
        let engine = MetalLLMEngine()
        try await engine.load(modelPath: URL(fileURLWithPath: modelPath))
        let config = SamplingConfig(temperature: 0, maxTokens: maxTokens, speculative: false)
        let prompt = "<start_of_turn>user\nWhat is quantum computing?<end_of_turn>\n<start_of_turn>model\n"

        nonisolated(unsafe) var count = 0
        let t0 = CFAbsoluteTimeGetCurrent()
        try await engine.generate(prompt: prompt, config: config) { _ in count += 1 }
        let tps = Double(count) / (CFAbsoluteTimeGetCurrent() - t0)
        print("  \(count) tokens, \(String(format: "%.1f", tps)) tok/s")
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
