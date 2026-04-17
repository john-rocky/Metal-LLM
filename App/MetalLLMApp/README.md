# MetalLLMApp

Minimal iOS app shell for the Metal-LLM inference engine.

## Quick Setup (XcodeGen - recommended)

XcodeGen generates the .xcodeproj from `project.yml`, keeping it out of version control.

```bash
# Install XcodeGen (once)
brew install xcodegen

# Generate the Xcode project
cd App/MetalLLMApp
xcodegen generate

# Open in Xcode
open MetalLLMApp.xcodeproj
```

This gives you a proper iOS app target with:
- Entitlements (increased-memory-limit)
- Info.plist with UIFileSharingEnabled for model transfer
- Correct bundle identifier and signing settings

## Alternative: Manual Xcode Project

If you prefer not to use XcodeGen:

1. In Xcode: File > New > Project > iOS > App (SwiftUI)
2. Set bundle ID to `com.example.MetalLLMApp`, deployment target iOS 18.0
3. Delete the generated ContentView.swift and App.swift
4. Drag the `Sources/` folder into the project
5. File > Add Package Dependencies > Add Local > select the Metal-LLM root
6. Add MetalLLMCore and MetalLLMUI to the app target
7. Add `MetalLLMApp.entitlements` to the project and set it in Signing & Capabilities
8. Set `Sources/Info.plist` as the Info.plist file in Build Settings

## Alternative: SwiftPM Only (limited)

```bash
# Open the Package.swift directly
open App/MetalLLMApp/Package.swift
```

This works for building and running on-device, but note:
- Entitlements must be configured manually in the scheme
- Info.plist settings (file sharing) require manual scheme configuration
- The increased-memory-limit entitlement is critical for loading 2.9GB GGUF models

## Model Setup

The app searches for GGUF models in this order:

1. **App bundle** - drag the .gguf into Xcode's project navigator (increases IPA size)
2. **Documents directory** - copy via Files.app or Finder

### Copy via Files.app (easiest on-device)

1. Transfer the GGUF to your iPhone (AirDrop, iCloud Drive, etc.)
2. Open Files.app
3. Navigate to: On My iPhone > MetalLLMApp
4. Move/copy the .gguf file there
5. Launch the app

### Copy via Finder (Mac connected)

1. Connect iPhone to Mac
2. Open Finder, select the device in the sidebar
3. Go to the Files tab
4. Drag the .gguf into MetalLLMApp
5. Launch the app

### Expected model

Default filename: `gemma-4-E2B-it-Q4_K_M.gguf`

The app will also detect any other `.gguf` file in the Documents directory.
