//
//  Metrics.swift
//  MetalLLMCore
//
//  Throughput, power, and thermal instrumentation.
//  Headline: sustained tok/s over a 60s decode window.
//

import Foundation

public struct MetricsSnapshot: Sendable {
    public let decodeTokensPerSecond: Double
    public let prefillTokensPerSecond: Double
    public let tokensGenerated: Int
    public let thermalState: ProcessInfo.ThermalState
    public let elapsedSeconds: Double
    public let acceptanceRate: Double
    public let specBursts: Int
}

public final class MetricsCollector: @unchecked Sendable {
    private var decodeStartTime: UInt64 = 0
    private var decodeTokenCount: Int = 0
    private var prefillStartTime: UInt64 = 0
    private var prefillDuration: Double = 0
    private var prefillTokenCount: Int = 0
    private var specAccepted: Int = 0
    private var specTotal: Int = 0
    private var specBurstCount: Int = 0
    private let lock = NSLock()

    public init() {}

    public func beginPrefill() {
        lock.lock(); defer { lock.unlock() }
        prefillStartTime = mach_absolute_time()
    }

    public func endPrefill(tokenCount: Int) {
        lock.lock(); defer { lock.unlock() }
        let elapsed = machTimeToSeconds(mach_absolute_time() - prefillStartTime)
        prefillDuration += elapsed
        prefillTokenCount += tokenCount
    }

    public func beginDecode() {
        lock.lock(); defer { lock.unlock() }
        decodeStartTime = mach_absolute_time()
        decodeTokenCount = 0
    }

    public func recordDecodedTokens(_ count: Int = 1) {
        lock.lock(); defer { lock.unlock() }
        decodeTokenCount += count
    }

    public func recordSpecBurst(accepted: Int, total: Int) {
        lock.lock(); defer { lock.unlock() }
        specAccepted += accepted
        specTotal += total
        specBurstCount += 1
    }

    public func snapshot() -> MetricsSnapshot {
        lock.lock(); defer { lock.unlock() }
        let now = mach_absolute_time()
        let elapsed = decodeStartTime > 0 ? machTimeToSeconds(now - decodeStartTime) : 0
        let decodeTps = elapsed > 0 ? Double(decodeTokenCount) / elapsed : 0
        let prefillTps = prefillDuration > 0 ? Double(prefillTokenCount) / prefillDuration : 0
        let acceptance = specTotal > 0 ? Double(specAccepted) / Double(specTotal) : 0

        return MetricsSnapshot(
            decodeTokensPerSecond: decodeTps,
            prefillTokensPerSecond: prefillTps,
            tokensGenerated: decodeTokenCount,
            thermalState: ProcessInfo.processInfo.thermalState,
            elapsedSeconds: elapsed,
            acceptanceRate: acceptance,
            specBursts: specBurstCount
        )
    }

    public func reset() {
        lock.lock(); defer { lock.unlock() }
        decodeStartTime = 0
        decodeTokenCount = 0
        prefillStartTime = 0
        prefillDuration = 0
        prefillTokenCount = 0
        specAccepted = 0
        specTotal = 0
        specBurstCount = 0
    }

    // MARK: - Private

    private static let timebaseInfo: mach_timebase_info_data_t = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return info
    }()

    private func machTimeToSeconds(_ ticks: UInt64) -> Double {
        let info = MetricsCollector.timebaseInfo
        let nanos = Double(ticks) * Double(info.numer) / Double(info.denom)
        return nanos / 1_000_000_000
    }
}
