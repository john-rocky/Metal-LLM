//
//  ModelLoaderView.swift
//  MetalLLMApp
//
//  Resolves the GGUF model location before presenting ChatView.
//  Search order:
//    1. App bundle (for models dragged into the Xcode project)
//    2. Documents directory (for models copied via Files.app / iTunes)
//    3. Shows instructions if nothing is found
//

import SwiftUI
import MetalLLMUI

// Default model filename. Change this to match your model.
private let kDefaultModelFilename = "gemma-4-E2B-it-Q4_K_M.gguf"

struct ModelLoaderView: View {
    @State private var modelURL: URL?
    @State private var scanning = true
    @State private var errorMessage: String?

    var body: some View {
        Group {
            if scanning {
                ProgressView("Scanning for model...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let url = modelURL {
                ChatView(modelURL: url)
            } else {
                modelNotFoundView
            }
        }
        .task {
            await scanForModel()
        }
    }

    // MARK: - Model not found

    private var modelNotFoundView: some View {
        VStack(spacing: 24) {
            Image(systemName: "doc.badge.gearshape")
                .font(.system(size: 64))
                .foregroundStyle(.secondary)

            Text("Model Not Found")
                .font(.title.bold())

            Text("Place a GGUF model file in one of these locations:")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 12) {
                instructionRow(
                    number: "1",
                    title: "Documents Directory (recommended)",
                    detail: "Open the Files app on your iPhone, navigate to On My iPhone > MetalLLMApp, and copy your .gguf file there."
                )

                instructionRow(
                    number: "2",
                    title: "Finder / iTunes File Sharing",
                    detail: "Connect your iPhone to a Mac, open Finder, select the device, go to Files tab, and drag the .gguf into MetalLLMApp."
                )

                instructionRow(
                    number: "3",
                    title: "Xcode Bundle Resource",
                    detail: "Drag the .gguf file into the Xcode project navigator. Warning: this increases app size significantly."
                )
            }
            .padding()

            if let error = errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            Text("Expected filename: \(kDefaultModelFilename)")
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)

            Button("Retry") {
                scanning = true
                Task { await scanForModel() }
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(32)
    }

    private func instructionRow(number: String, title: String, detail: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text(number)
                .font(.headline)
                .frame(width: 28, height: 28)
                .background(Circle().fill(.blue))
                .foregroundStyle(.white)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline.bold())
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Scanning

    private func scanForModel() async {
        defer { scanning = false }

        // 1. Check app bundle
        if let bundleURL = Bundle.main.url(
            forResource: kDefaultModelFilename.replacingOccurrences(of: ".gguf", with: ""),
            withExtension: "gguf"
        ) {
            modelURL = bundleURL
            return
        }

        // 2. Check Documents directory
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let docsPath = docs.appendingPathComponent(kDefaultModelFilename)
        if FileManager.default.fileExists(atPath: docsPath.path) {
            modelURL = docsPath
            return
        }

        // 3. Check for any .gguf file in Documents
        do {
            let contents = try FileManager.default.contentsOfDirectory(
                at: docs, includingPropertiesForKeys: nil
            )
            if let firstGGUF = contents.first(where: { $0.pathExtension == "gguf" }) {
                modelURL = firstGGUF
                return
            }
        } catch {
            errorMessage = "Failed to scan Documents: \(error.localizedDescription)"
        }

        // Not found
        modelURL = nil
    }
}
