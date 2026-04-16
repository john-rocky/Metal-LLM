//
//  MetalRuntime.swift
//  MetalLLMCore
//
//  Custom Metal compute runtime for Gemma 4 E2B decode.
//  Manages device, command queue, pipeline states, and provides
//  a low-level interface for running fused kernels.
//

import Foundation
import Metal

public final class MetalRuntime {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalRuntimeError.noDevice
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MetalRuntimeError.noCommandQueue
        }
        self.commandQueue = queue

        // Load our custom shader library
        let bundle = Bundle(for: MetalRuntime.self)
        if let libURL = bundle.url(forResource: "default", withExtension: "metallib") {
            self.library = try device.makeLibrary(URL: libURL)
        } else {
            // Fallback: compile from source at runtime (dev only)
            let srcURL = bundle.url(forResource: "gemma4_kernels", withExtension: "metal")
                ?? URL(fileURLWithPath: #filePath)
                    .deletingLastPathComponent()
                    .appendingPathComponent("Metal/gemma4_kernels.metal")
            let source = try String(contentsOf: srcURL)
            self.library = try device.makeLibrary(source: source, options: nil)
        }

        // Pre-compile all pipeline states
        try compilePipeline("kernel_rms_norm")
        try compilePipeline("kernel_matvec_q4_0")
        try compilePipeline("kernel_fused_rmsnorm_matvec_q4_0")
        try compilePipeline("kernel_swiglu")
        try compilePipeline("kernel_residual_add")

        print("[MetalRuntime] Initialized on \(device.name)")
        print("[MetalRuntime] Max threadgroup size: \(device.maxThreadsPerThreadgroup)")
        print("[MetalRuntime] Unified memory: \(device.hasUnifiedMemory)")
        print("[MetalRuntime] \(pipelines.count) pipelines compiled")
    }

    // MARK: - Pipeline access

    public func pipeline(_ name: String) -> MTLComputePipelineState? {
        pipelines[name]
    }

    // MARK: - Buffer creation

    public func makeBuffer(length: Int) -> MTLBuffer? {
        device.makeBuffer(length: length, options: .storageModeShared)
    }

    public func makeBuffer<T>(data: [T]) -> MTLBuffer? {
        data.withUnsafeBytes { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: ptr.count, options: .storageModeShared)
        }
    }

    // MARK: - Single command execution (for benchmarking individual kernels)

    public func execute(_ block: (MTLComputeCommandEncoder) -> Void) {
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else { return }
        block(encoder)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    public func executeAsync(_ block: (MTLComputeCommandEncoder) -> Void) -> MTLCommandBuffer? {
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else { return nil }
        block(encoder)
        encoder.endEncoding()
        cmdBuf.commit()
        return cmdBuf
    }

    // MARK: - Kernel dispatch helpers

    /// Dispatch a 1D compute kernel.
    public func dispatch1D(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        count: Int
    ) {
        encoder.setComputePipelineState(pipeline)
        let threadgroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, count)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
    }

    // MARK: - Private

    private func compilePipeline(_ name: String) throws {
        guard let function = library.makeFunction(name: name) else {
            throw MetalRuntimeError.functionNotFound(name)
        }
        let pso = try device.makeComputePipelineState(function: function)
        pipelines[name] = pso
    }
}

// MARK: - Errors

public enum MetalRuntimeError: Error, LocalizedError {
    case noDevice
    case noCommandQueue
    case functionNotFound(String)
    case bufferAllocationFailed

    public var errorDescription: String? {
        switch self {
        case .noDevice: return "No Metal device found"
        case .noCommandQueue: return "Failed to create command queue"
        case .functionNotFound(let n): return "Metal function not found: \(n)"
        case .bufferAllocationFailed: return "MTLBuffer allocation failed"
        }
    }
}
