//
//  EngineTests.swift
//  MetalLLMCoreTests
//
//  Placeholder test target. Real tests arrive alongside the Week 1 llama.cpp
//  integration (load-model smoke test, tokenizer round-trip) and Week 5 spec-decode
//  work (drafter/verifier agreement tests).
//

import XCTest
@testable import MetalLLMCore

final class EngineTests: XCTestCase {

    func testEngineInitialises() {
        let engine = MetalLLMEngine()
        _ = engine.metricsSnapshot()
        // TODO(Week 1): assert initial metrics are zeroed.
    }

    // TODO(Week 1): testLoadModel_smallGGUF
    // TODO(Week 1): testTokenizerRoundTrip
    // TODO(Week 5): testDrafterAgreesWithVerifierAtTemperatureZero
}
