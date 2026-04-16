// swift-tools-version: 6.0
//
// Metal-LLM — Swift package manifest.
//
// Two products:
//   - MetalLLMCore: decode engine, llama.cpp bridge, spec-decode orchestrator, ANE bridge.
//   - MetalLLMUI:   streaming chat UI built on top of the core.
//
// llama.cpp is consumed via a prebuilt XCFramework at Frameworks/llama.xcframework.
// Build it with: ./Scripts/build-xcframework.sh

import PackageDescription

let package = Package(
    name: "Metal-LLM",
    platforms: [
        .iOS(.v18),
        .macOS(.v15)
    ],
    products: [
        .library(name: "MetalLLMCore", targets: ["MetalLLMCore"]),
        .library(name: "MetalLLMUI", targets: ["MetalLLMUI"]),
    ],
    targets: [
        .binaryTarget(
            name: "llama",
            path: "Frameworks/llama.xcframework"
        ),
        .target(
            name: "MetalLLMCore",
            dependencies: ["llama"],
            path: "Sources/MetalLLMCore",
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("CoreML"),
                .linkedFramework("CoreVideo"),
            ]
        ),
        .target(
            name: "MetalLLMUI",
            dependencies: ["MetalLLMCore"],
            path: "Sources/MetalLLMUI"
        ),
        .executableTarget(
            name: "MetalLLMBench",
            dependencies: ["MetalLLMCore"],
            path: "Sources/MetalLLMBench"
        ),
        .testTarget(
            name: "MetalLLMCoreTests",
            dependencies: ["MetalLLMCore"],
            path: "Tests/MetalLLMCoreTests"
        ),
    ]
)
