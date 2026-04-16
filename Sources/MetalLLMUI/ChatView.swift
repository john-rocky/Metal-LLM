//
//  ChatView.swift
//  MetalLLMUI
//
//  Streaming chat UI on top of MetalLLMCore.
//  Displays messages, streams tokens, and shows a tok/s HUD.
//

#if canImport(SwiftUI)
import SwiftUI
import MetalLLMCore

// MARK: - Chat message model

public struct ChatMessage: Identifiable, Sendable {
    public let id = UUID()
    public let role: Role
    public var content: String

    public enum Role: Sendable {
        case user
        case assistant
        case system
    }

    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }
}

// MARK: - ViewModel

@available(iOS 18.0, macOS 15.0, *)
@MainActor
@Observable
public final class ChatViewModel {
    public var messages: [ChatMessage] = []
    public var inputText: String = ""
    public var isGenerating: Bool = false
    public var tokensPerSecond: Double = 0
    public var tokensGenerated: Int = 0
    public var acceptanceRate: Double = 0
    public var thermalState: String = "nominal"
    public var modelLoaded: Bool = false
    public var loadingStatus: String = "Not loaded"

    private var engine: MetalLLMEngine?
    private var modelURL: URL?

    public init() {}

    public func loadModel(at url: URL) async {
        loadingStatus = "Loading model..."
        let eng = MetalLLMEngine()
        do {
            try await eng.load(modelPath: url)
            self.engine = eng
            self.modelURL = url
            self.modelLoaded = true
            self.loadingStatus = "Ready"
        } catch {
            self.loadingStatus = "Failed: \(error.localizedDescription)"
        }
    }

    public func send() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, let engine else { return }

        inputText = ""
        messages.append(ChatMessage(role: .user, content: text))
        messages.append(ChatMessage(role: .assistant, content: ""))
        isGenerating = true

        let prompt = buildPrompt()
        let config = SamplingConfig(
            temperature: 0.0,
            maxTokens: 1024,
            speculative: true
        )

        let stream = await engine.generateStream(prompt: prompt, config: config)
        do {
            for try await event in stream {
                if let idx = messages.indices.last {
                    messages[idx].content += event.token
                }
                tokensPerSecond = event.tokensPerSecond
                tokensGenerated += 1

                let snap = await engine.metricsSnapshot()
                acceptanceRate = snap.acceptanceRate
                updateThermalDisplay(snap.thermalState)
            }
        } catch {
            if let idx = messages.indices.last {
                messages[idx].content += "\n[Error: \(error.localizedDescription)]"
            }
        }

        isGenerating = false
    }

    public func stop() async {
        await engine?.cancel()
        isGenerating = false
    }

    private func buildPrompt() -> String {
        var prompt = ""
        for msg in messages {
            switch msg.role {
            case .user:
                prompt += "<start_of_turn>user\n\(msg.content)<end_of_turn>\n"
            case .assistant:
                if msg.content.isEmpty {
                    prompt += "<start_of_turn>model\n"
                } else {
                    prompt += "<start_of_turn>model\n\(msg.content)<end_of_turn>\n"
                }
            case .system:
                prompt += "<start_of_turn>system\n\(msg.content)<end_of_turn>\n"
            }
        }
        return prompt
    }

    private func updateThermalDisplay(_ state: ProcessInfo.ThermalState) {
        switch state {
        case .nominal:  thermalState = "nominal"
        case .fair:     thermalState = "fair"
        case .serious:  thermalState = "serious"
        case .critical: thermalState = "critical"
        @unknown default: thermalState = "unknown"
        }
    }
}

// MARK: - Chat View

@available(iOS 18.0, macOS 15.0, *)
public struct ChatView: View {
    @State private var viewModel = ChatViewModel()
    private let modelURL: URL?

    public init(modelURL: URL? = nil) {
        self.modelURL = modelURL
    }

    public var body: some View {
        VStack(spacing: 0) {
            // HUD
            hudBar

            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: viewModel.messages.count) { _, _ in
                    if let last = viewModel.messages.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }

            // Input
            inputBar
        }
        .task {
            if let url = modelURL {
                await viewModel.loadModel(at: url)
            }
        }
    }

    private var hudBar: some View {
        HStack(spacing: 16) {
            Text(viewModel.loadingStatus)
                .font(.caption)
                .foregroundStyle(.secondary)

            Spacer()

            if viewModel.isGenerating {
                HStack(spacing: 8) {
                    Text(String(format: "%.1f tok/s", viewModel.tokensPerSecond))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.green)

                    if viewModel.acceptanceRate > 0 {
                        Text(String(format: "acc %.0f%%", viewModel.acceptanceRate * 100))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.orange)
                    }

                    Text(viewModel.thermalState)
                        .font(.caption)
                        .foregroundStyle(thermalColor)
                }
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(.bar)
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Message", text: $viewModel.inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...5)
                .onSubmit {
                    Task { await viewModel.send() }
                }
                .disabled(!viewModel.modelLoaded || viewModel.isGenerating)

            if viewModel.isGenerating {
                Button(action: { Task { await viewModel.stop() } }) {
                    Image(systemName: "stop.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.red)
                }
            } else {
                Button(action: { Task { await viewModel.send() } }) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                }
                .disabled(viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !viewModel.modelLoaded)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.bar)
    }

    private var thermalColor: Color {
        switch viewModel.thermalState {
        case "nominal": return .green
        case "fair": return .yellow
        case "serious": return .orange
        case "critical": return .red
        default: return .gray
        }
    }
}

// MARK: - Message Bubble

@available(iOS 18.0, macOS 15.0, *)
struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }

            Text(message.content.isEmpty ? " " : message.content)
                .padding(12)
                .background(backgroundColor)
                .clipShape(RoundedRectangle(cornerRadius: 16))
                .foregroundStyle(message.role == .user ? .white : .primary)

            if message.role == .assistant { Spacer(minLength: 60) }
        }
    }

    private var backgroundColor: Color {
        switch message.role {
        case .user: return .blue
        case .assistant: return Color.gray.opacity(0.15)
        case .system: return Color.gray.opacity(0.1)
        }
    }
}
#endif
