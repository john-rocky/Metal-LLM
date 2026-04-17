# Metal-LLM iOS App

## Quick Start (iPhone 17 Pro)

### Step 1: Create Xcode Project
Since SwiftPM executable targets can't deploy to iOS directly:

1. Open Xcode → File → New → Project → iOS App
2. Product Name: `MetalLLMApp`
3. Interface: SwiftUI, Language: Swift
4. Save in `Metal-LLM/App/`

### Step 2: Add Metal-LLM Package
1. File → Add Package Dependencies → Add Local → select `Metal-LLM/` root
2. Add `MetalLLMCore` and `MetalLLMUI` to the app target

### Step 3: Replace App Files
Copy these files into the Xcode project, replacing the generated ones:
- `Sources/MetalLLMApp.swift` → app entry point
- `Sources/ModelLoaderView.swift` → model loading UI
- `Sources/Info.plist` → enables file sharing
- `Sources/MetalLLMApp.entitlements` → increased memory limit

### Step 4: Configure Entitlements
1. Select the app target → Signing & Capabilities
2. Add "Increased Memory Limit" capability
3. Verify the entitlements file is set in Build Settings

### Step 5: Deploy Model
Two options:

**A. iTunes/Finder File Sharing (recommended):**
1. Build and run the app on iPhone
2. Connect iPhone to Mac via USB
3. Open Finder → iPhone → Files tab
4. Drag `Models/gemma-4-E2B-it-Q4_K_M.gguf` into MetalLLMApp
5. Relaunch the app

**B. Bundle in App (slow build):**
1. Drag `Models/gemma-4-E2B-it-Q4_K_M.gguf` into Xcode project navigator
2. Check "Copy items if needed" and add to app target
3. Build & run (warning: 2.9GB model increases build time significantly)

### Step 6: Run
App will scan for the GGUF file and load it. First load takes ~2-5 seconds.
The HUD shows tok/s, acceptance rate, and thermal state.

## Expected Performance (iPhone 17 Pro)
- Load: ~2-5s
- Greedy decode: ~38-40 tok/s
- With speculative decode (future): ~80-100+ tok/s

## Requirements
- Xcode 16+
- iPhone with A17 Pro or later (iOS 18+)
- `com.apple.developer.kernel.increased-memory-limit` entitlement
- 2.9 GB free storage for the GGUF model
