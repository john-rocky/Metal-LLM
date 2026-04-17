//
//  MetalLLMApp.swift
//  MetalLLMApp
//
//  Minimal iOS app entry point for Metal-LLM.
//  Looks for a GGUF model in the app's Documents directory or bundle,
//  then passes its URL to ChatView from MetalLLMUI.
//

import SwiftUI
import MetalLLMCore
import MetalLLMUI

@main
struct MetalLLMApp: App {
    var body: some Scene {
        WindowGroup {
            ModelLoaderView()
        }
    }
}
