//
//  GGUFLoader.swift
//  MetalLLMCore
//
//  Loads GGUF model weights into Metal buffers for the custom runtime.
//  Uses the gguf.h API from the llama.cpp framework to parse tensor
//  metadata, then memory-maps the file to create MTLBuffers directly.
//

import Foundation
import Metal
#if canImport(llama)
import llama

// MARK: - Tensor descriptor

public struct GGUFTensor {
    public let name: String
    public let type: ggml_type
    public let offset: Int      // offset within the GGUF data section
    public let size: Int        // byte size
    public let buffer: MTLBuffer
}

// MARK: - GGUF Loader

public final class GGUFLoader {
    private let device: MTLDevice
    private var filePath: String = ""
    private var fileHandle: FileHandle?
    private var mappedData: Data?

    public private(set) var tensors: [String: GGUFTensor] = [:]
    public private(set) var dataOffset: Int = 0
    public private(set) var tensorCount: Int = 0

    // Model metadata
    public private(set) var nLayers: Int = 0
    public private(set) var nEmbd: Int = 0
    public private(set) var nHead: Int = 0
    public private(set) var nHeadKV: Int = 0
    public private(set) var nFF: [Int] = []  // per-layer FF sizes (6144 or 12288)

    public init(device: MTLDevice) {
        self.device = device
    }

    deinit {
        fileHandle?.closeFile()
    }

    // MARK: - Load

    /// Parse GGUF metadata and load specified tensors into MTLBuffers.
    public func load(path: String, tensorNames: [String]? = nil) throws {
        self.filePath = path

        // Parse with gguf.h
        var params = gguf_init_params()
        params.no_alloc = true
        params.ctx = nil

        guard let ctx = gguf_init_from_file(path, params) else {
            throw GGUFLoaderError.parseFailed(path)
        }
        defer { gguf_free(ctx) }

        dataOffset = Int(gguf_get_data_offset(ctx))
        tensorCount = Int(gguf_get_n_tensors(ctx))

        // Read model metadata
        readMetadata(ctx)

        // Memory-map the file
        fileHandle = FileHandle(forReadingAtPath: path)
        guard let fh = fileHandle else {
            throw GGUFLoaderError.fileOpenFailed(path)
        }
        mappedData = try fh.availableData  // mmap'd by the OS for large files

        print("[GGUFLoader] \(tensorCount) tensors, data offset: \(dataOffset)")
        print("[GGUFLoader] n_layers=\(nLayers) n_embd=\(nEmbd) n_head=\(nHead) n_head_kv=\(nHeadKV)")

        // Load requested tensors (or all if nil)
        let namesToLoad: Set<String>?
        if let names = tensorNames {
            namesToLoad = Set(names)
        } else {
            namesToLoad = nil
        }

        for i in 0..<tensorCount {
            let name = String(cString: gguf_get_tensor_name(ctx, Int64(i)))
            if let filter = namesToLoad, !filter.contains(name) { continue }

            let type = gguf_get_tensor_type(ctx, Int64(i))
            let offset = Int(gguf_get_tensor_offset(ctx, Int64(i)))
            let size = Int(gguf_get_tensor_size(ctx, Int64(i)))

            // Create MTLBuffer from mmap'd data (zero-copy on unified memory)
            let fileOffset = dataOffset + offset
            guard let buffer = createBuffer(fileOffset: fileOffset, size: size) else {
                print("[GGUFLoader] Warning: failed to create buffer for \(name) (\(size) bytes)")
                continue
            }

            tensors[name] = GGUFTensor(
                name: name, type: type,
                offset: offset, size: size, buffer: buffer
            )
        }

        print("[GGUFLoader] Loaded \(tensors.count) tensors into MTLBuffers")
    }

    // MARK: - Tensor access

    public func tensor(_ name: String) -> GGUFTensor? {
        tensors[name]
    }

    /// Get tensor for a specific layer component.
    /// Format: "blk.{layer}.{component}" e.g. "blk.0.attn_q.weight"
    public func layerTensor(layer: Int, component: String) -> GGUFTensor? {
        tensors["blk.\(layer).\(component).weight"]
            ?? tensors["blk.\(layer).\(component)"]
    }

    // MARK: - Private

    private func readMetadata(_ ctx: OpaquePointer) {
        func intVal(_ key: String) -> Int {
            let idx = gguf_find_key(ctx, key)
            guard idx >= 0 else { return 0 }
            return Int(gguf_get_val_u32(ctx, idx))
        }

        nLayers = intVal("gemma4.block_count")
        if nLayers == 0 { nLayers = intVal("llama.block_count") }
        nEmbd = intVal("gemma4.embedding_length")
        if nEmbd == 0 { nEmbd = intVal("llama.embedding_length") }
        nHead = intVal("gemma4.attention.head_count")
        if nHead == 0 { nHead = intVal("llama.attention.head_count") }
        nHeadKV = intVal("gemma4.attention.head_count_kv")
        if nHeadKV == 0 { nHeadKV = intVal("llama.attention.head_count_kv") }

        // Per-layer FF sizes
        let ffKey = gguf_find_key(ctx, "gemma4.feed_forward_length")
        if ffKey >= 0 {
            // Array type — read from GGUF array
            // For now, assume uniform 6144 (L0-14) and 12288 (L15-34)
            nFF = (0..<nLayers).map { $0 < 15 ? 6144 : 12288 }
        }
    }

    private func createBuffer(fileOffset: Int, size: Int) -> MTLBuffer? {
        guard size > 0 else { return nil }

        // Read data from file at offset
        guard let fh = fileHandle else { return nil }
        fh.seek(toFileOffset: UInt64(fileOffset))
        let data = fh.readData(ofLength: size)

        guard data.count == size else { return nil }

        return data.withUnsafeBytes { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: size, options: .storageModeShared)
        }
    }
}

// MARK: - Errors

public enum GGUFLoaderError: Error {
    case parseFailed(String)
    case fileOpenFailed(String)
    case tensorNotFound(String)
}

#endif
