// swift-tools-version: 6.0
//
// MetalLLMApp — Thin iOS app shell.
//
// This package depends on the parent Metal-LLM package for MetalLLMCore and MetalLLMUI.
//
// Usage:
//   1. Open this Package.swift in Xcode 16+.
//   2. Select the MetalLLMApp scheme and an iOS 18+ device/simulator.
//   3. Build & run.
//
// NOTE: SwiftPM executable targets don't natively support iOS entitlements.
// See README.md for the recommended Xcode project wrapper approach.

import PackageDescription

let package = Package(
    name: "MetalLLMApp",
    platforms: [
        .iOS(.v18),
        .macOS(.v15)
    ],
    dependencies: [
        .package(path: "../.."),  // Metal-LLM root
    ],
    targets: [
        .executableTarget(
            name: "MetalLLMApp",
            dependencies: [
                .product(name: "MetalLLMCore", package: "Metal-LLM"),
                .product(name: "MetalLLMUI", package: "Metal-LLM"),
            ],
            path: "Sources"
        ),
    ]
)
