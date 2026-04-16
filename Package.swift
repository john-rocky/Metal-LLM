// swift-tools-version: 5.9
//
// Metal-LLM — Swift package manifest.
//
// Two products:
//   - MetalLLMCore: decode engine, llama.cpp bridge, spec-decode orchestrator, ANE bridge.
//   - MetalLLMUI:   streaming chat UI built on top of the core.
//
// llama.cpp is intentionally NOT wired up at Week 0. It is added in Week 1 as either
// a git submodule under `External/llama.cpp` or a binary xcframework, depending on
// whether we need to patch the Metal kernels. The commented dependency below records
// the intent.

import PackageDescription

let package = Package(
    name: "Metal-LLM",
    platforms: [
        .iOS(.v18),
        .macOS(.v14)
    ],
    products: [
        .library(name: "MetalLLMCore", targets: ["MetalLLMCore"]),
        .library(name: "MetalLLMUI", targets: ["MetalLLMUI"]),
    ],
    dependencies: [
        // Week 1: integrate llama.cpp.
        // Option A — submodule + system target:
        // .package(path: "External/llama.cpp"),
        // Option B — upstream Swift package (if/when it ships a stable SPM target):
        // .package(url: "https://github.com/ggml-org/llama.cpp.git", branch: "master"),
    ],
    targets: [
        .target(
            name: "MetalLLMCore",
            dependencies: [
                // .product(name: "llama", package: "llama.cpp"),
            ],
            path: "Sources/MetalLLMCore"
        ),
        .target(
            name: "MetalLLMUI",
            dependencies: ["MetalLLMCore"],
            path: "Sources/MetalLLMUI"
        ),
        .testTarget(
            name: "MetalLLMCoreTests",
            dependencies: ["MetalLLMCore"],
            path: "Tests/MetalLLMCoreTests"
        ),
    ]
)
