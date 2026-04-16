//
//  MTPDrafter.swift
//  MetalLLMCore
//
//  MTP (Multi-Token Prediction) drafter that runs on the Apple Neural Engine
//  via Core ML. Reuses the MTP drafter model from the sibling CoreML-LLM repo.
//
//  The drafter produces K=3 speculative tokens per burst. Each step:
//    1. Embed the last accepted token
//    2. Concatenate with projected activations from the verifier
//    3. Run the 4-layer mini-transformer on ANE
//    4. Argmax the output logits
//
//  The drafter model is ~44 MB (4 layers, matching Google's MTP design).
//

import Foundation
#if canImport(CoreML)
import CoreML

@available(iOS 18.0, macOS 15.0, *)
public final class MTPDrafter: @unchecked Sendable {
    private var model: MLModel?
    private let modelURL: URL
    private let computeUnits: MLComputeUnits
    private var isLoaded = false

    // Carry state: projected activations from last verify
    private var lastActivations: MLMultiArray?

    public init(modelURL: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) {
        self.modelURL = modelURL
        self.computeUnits = computeUnits
    }

    // MARK: - Lifecycle

    public func load() async throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        model = try MLModel(contentsOf: modelURL, configuration: config)
        isLoaded = true
        print("[MTPDrafter] Loaded from \(modelURL.lastPathComponent), compute: \(computeUnits)")
    }

    /// Warm up the ANE compute plan with a dummy prediction.
    public func warmup() throws {
        guard let model else { throw MTPDrafterError.notLoaded }

        // Create dummy input matching the drafter's expected shape
        // activations: [1, 1, 3072] (embed_dim + projected_dim = 1536 + 1536)
        let dummyActivations = try MLMultiArray(
            shape: [1, 1, 3072],
            dataType: .float16
        )
        let dummyPos = try MLMultiArray(shape: [1], dataType: .int32)
        dummyPos[0] = NSNumber(value: 0)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "activations": MLFeatureValue(multiArray: dummyActivations),
            "input_pos": MLFeatureValue(multiArray: dummyPos),
        ])

        _ = try model.prediction(from: input)
        print("[MTPDrafter] Warmup complete")
    }

    // MARK: - Draft

    /// Produce up to K draft tokens. Called from the spec-decode orchestrator.
    public func draft(
        lastToken: Int32,
        embeddings: MLMultiArray?,
        activations: MLMultiArray?,
        position: Int32,
        maxTokens: Int = 3
    ) throws -> (tokens: [Int32], finalActivations: MLMultiArray?) {
        guard let model, isLoaded else {
            throw MTPDrafterError.notLoaded
        }

        var draftTokens: [Int32] = []
        var currentActivations = activations ?? lastActivations

        for step in 0..<maxTokens {
            let pos = position + Int32(step)

            // Build input
            guard let acts = currentActivations else {
                break  // No activations to feed — can't draft
            }

            let posArray = try MLMultiArray(shape: [1], dataType: .int32)
            posArray[0] = NSNumber(value: pos)

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "activations": MLFeatureValue(multiArray: acts),
                "input_pos": MLFeatureValue(multiArray: posArray),
            ])

            let output = try model.prediction(from: input)

            // Extract logits and argmax
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
                break
            }

            let token = argmax(logits)
            draftTokens.append(token)

            // Extract projected_activations for next step
            currentActivations = output.featureValue(for: "projected_activations")?.multiArrayValue
        }

        lastActivations = currentActivations
        return (draftTokens, currentActivations)
    }

    /// Update carry state after verification.
    public func updateActivations(_ activations: MLMultiArray?) {
        lastActivations = activations
    }

    // MARK: - Private

    private func argmax(_ array: MLMultiArray) -> Int32 {
        let count = array.count
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)

        var maxVal: Float16 = -.infinity
        var maxIdx: Int32 = 0

        for i in 0..<count {
            if ptr[i] > maxVal {
                maxVal = ptr[i]
                maxIdx = Int32(i)
            }
        }
        return maxIdx
    }
}

// MARK: - SwiftDrafter conformance

#if canImport(llama)
import llama

@available(iOS 18.0, macOS 15.0, *)
extension MTPDrafter: SwiftDrafter {
    public func draft(lastToken: llama_token, maxTokens: Int) async throws -> [llama_token] {
        let result = try draft(
            lastToken: lastToken,
            embeddings: nil,
            activations: lastActivations,
            position: 0,  // Position managed by orchestrator
            maxTokens: maxTokens
        )
        return result.tokens
    }
}
#endif

// MARK: - Errors

public enum MTPDrafterError: Error {
    case notLoaded
    case predictionFailed(String)
}

#endif // canImport(CoreML)
