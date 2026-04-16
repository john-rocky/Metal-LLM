//
//  LlamaBridge.swift
//  MetalLLMCore
//
//  Convenience re-exports and helpers for llama.cpp interop.
//  The actual llama.cpp wrapper is in LlamaContext.swift.
//

import Foundation
#if canImport(llama)
import llama

/// Initialize the llama backend once at app launch.
public func llamaBackendInit() {
    llama_backend_init()
}

/// Free the llama backend at app termination.
public func llamaBackendFree() {
    llama_backend_free()
}
#endif
