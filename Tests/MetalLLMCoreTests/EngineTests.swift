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

    func testPromptLookupDrafter() async throws {
        let drafter = PromptLookupDrafter(ngramSize: 2)
        await drafter.updateHistory([1, 2, 3, 4, 5, 1, 2])

        let proposals = try await drafter.draft(lastToken: 2, maxTokens: 3)
        XCTAssertEqual(proposals, [3, 4, 5])
    }

    func testPromptLookupNoMatch() async throws {
        let drafter = PromptLookupDrafter(ngramSize: 2)
        await drafter.updateHistory([1, 2, 3])

        let proposals = try await drafter.draft(lastToken: 3, maxTokens: 3)
        XCTAssertTrue(proposals.isEmpty)
    }

    func testSpecDecodeOrchestratorInit() {
        let orch = SpecDecodeOrchestrator()
        XCTAssertEqual(orch.K, 3)
        XCTAssertEqual(orch.fallbackThreshold, 0.30)
    }

    func testEngineInitialises() async {
        let engine = MetalLLMEngine()
        let snap = await engine.metricsSnapshot()
        XCTAssertEqual(snap.tokensGenerated, 0)
    }
}
