//
//  ChatView.swift
//  MetalLLMUI
//
//  Streaming chat UI on top of MetalLLMCore. Minimal by design — the point of this
//  target is to give the benchmark harness a real-app-shaped surface so we measure
//  tok/s under realistic conditions (text rendering, scroll, state updates) rather
//  than in a tight loop.
//

#if canImport(SwiftUI)
import SwiftUI
import MetalLLMCore

@available(iOS 18.0, macOS 14.0, *)
public struct ChatView: View {

    @State private var prompt: String = ""
    @State private var messages: [ChatMessage] = []
    @State private var isGenerating: Bool = false

    // TODO: inject a DecodeEngine instance from the host app.
    // private let engine: DecodeEngine

    public init() {}

    public var body: some View {
        // TODO(Week 1): message list, input field, send button, tok/s HUD.
        // TODO(Week 2): streaming token rendering via AsyncThrowingStream.
        // TODO(Week 6): per-message acceptance-rate badge when spec-decode is on.
        Text("MetalLLMUI.ChatView — scaffold")
            .padding()
    }
}

public struct ChatMessage: Identifiable {
    public let id = UUID()
    public let role: Role
    public let content: String

    public enum Role {
        case user
        case assistant
    }
}
#endif
