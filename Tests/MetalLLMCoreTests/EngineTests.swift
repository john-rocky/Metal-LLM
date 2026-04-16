//
//  EngineTests.swift
//  MetalLLMCoreTests
//
//  Basic integration tests for the Metal-LLM engine.
//  Note: Full model tests require a GGUF on disk and run on-device only.
//

import XCTest
@testable import MetalLLMCore

final class EngineTests: XCTestCase {

    func testSamplingConfigDefaults() {
        let config = SamplingConfig()
        XCTAssertEqual(config.temperature, 0.0)
        XCTAssertEqual(config.topK, 40)
        XCTAssertEqual(config.maxTokens, 512)
        XCTAssertTrue(config.speculative)
    }

    func testMetricsCollectorBasic() {
        let collector = MetricsCollector()
        collector.beginDecode()
        collector.recordDecodedTokens(10)

        let snap = collector.snapshot()
        XCTAssertEqual(snap.tokensGenerated, 10)
        XCTAssertGreaterThan(snap.decodeTokensPerSecond, 0)
    }

    func testNgramDrafterMatch() {
        let drafter = NgramDrafter(ngramMin: 2, ngramMax: 4)
        drafter.feedTokens([1, 2, 3, 4, 5, 1, 2])

        let proposals = drafter.draft(maxTokens: 3)
        XCTAssertEqual(proposals, [3, 4, 5])
    }

    func testNgramDrafterNoMatch() {
        let drafter = NgramDrafter(ngramMin: 2, ngramMax: 4)
        drafter.feedTokens([1, 2, 3])

        let proposals = drafter.draft(maxTokens: 3)
        XCTAssertTrue(proposals.isEmpty)
    }

    func testNgramDrafterLongerMatch() {
        let drafter = NgramDrafter(ngramMin: 2, ngramMax: 4)
        // Pattern: [10, 20, 30, 40, 50, 10, 20, 30]
        // Should find [10, 20, 30] match and propose [40, 50]
        drafter.feedTokens([10, 20, 30, 40, 50, 10, 20, 30])
        let proposals = drafter.draft(maxTokens: 3)
        XCTAssertEqual(proposals, [40, 50, 10])
    }

    func testSpecDecodeOrchestratorInit() {
        let orch = SpecDecodeOrchestrator()
        XCTAssertEqual(orch.K, 3)
        XCTAssertEqual(orch.fallbackThreshold, 0.15)
    }

    func testEngineInitialises() async {
        let engine = MetalLLMEngine()
        let snap = await engine.metricsSnapshot()
        XCTAssertEqual(snap.tokensGenerated, 0)
    }
}
