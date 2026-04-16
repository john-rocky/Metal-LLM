//
//  Metrics.swift
//  MetalLLMCore
//
//  Throughput, power, and thermal instrumentation. The headline number we care about
//  is sustained tok/s over a 60s decode window (LiteRT-LM reports burst numbers, we
//  want to claim sustained). Power comes from `os_proc_available_memory`-adjacent
//  APIs plus `ProcessInfo.thermalState`; precise joules/token need the Xcode power
//  profiler and are reported manually in docs/BENCHMARKS.md.
//

import Foundation

/// Immutable snapshot of the current run's metrics.
public struct MetricsSnapshot {
    public let decodeTokensPerSecond: Double
    public let prefillTokensPerSecond: Double
    public let tokensGenerated: Int
    public let thermalState: ProcessInfo.ThermalState
    public let elapsedSeconds: Double
}

/// Collects per-token timings and exposes rolling averages.
public final class MetricsCollector {

    private var decodeStartTime: CFAbsoluteTime?
    private var decodeTokenCount: Int = 0
    private var prefillDuration: Double = 0
    private var prefillTokenCount: Int = 0
    private let lock = NSLock()

    public init() {}

    public func beginPrefill() {
        // TODO: record start timestamp.
    }

    public func endPrefill(tokenCount: Int) {
        // TODO: accumulate prefillDuration + prefillTokenCount.
    }

    public func beginDecode() {
        lock.lock(); defer { lock.unlock() }
        decodeStartTime = CFAbsoluteTimeGetCurrent()
        decodeTokenCount = 0
    }

    public func recordDecodedToken() {
        lock.lock(); defer { lock.unlock() }
        decodeTokenCount += 1
    }

    public func snapshot() -> MetricsSnapshot {
        lock.lock(); defer { lock.unlock() }
        let elapsed = decodeStartTime.map { CFAbsoluteTimeGetCurrent() - $0 } ?? 0
        let tps = elapsed > 0 ? Double(decodeTokenCount) / elapsed : 0
        let prefillTps = prefillDuration > 0 ? Double(prefillTokenCount) / prefillDuration : 0
        return MetricsSnapshot(
            decodeTokensPerSecond: tps,
            prefillTokensPerSecond: prefillTps,
            tokensGenerated: decodeTokenCount,
            thermalState: ProcessInfo.processInfo.thermalState,
            elapsedSeconds: elapsed
        )
    }
}
