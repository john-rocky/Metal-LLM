#!/usr/bin/env bash
#
# Build llama.cpp as an XCFramework for Metal-LLM.
# Wraps the upstream build-xcframework.sh with our flag overrides.
#
set -euo pipefail
cd "$(dirname "$0")/.."

LLAMA_DIR="External/llama.cpp"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "Error: llama.cpp not found at $LLAMA_DIR"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

echo "=== Building llama.cpp XCFramework ==="
echo "Source: $LLAMA_DIR"

cd "$LLAMA_DIR"

# Use upstream build script with our defaults
# Override iOS min version to 18.0 for Metal residency sets
export IOS_MIN_OS_VERSION=18.0

# Run upstream build
bash build-xcframework.sh

echo ""
echo "=== Copying XCFramework to Frameworks/ ==="
cd "$(dirname "$0")/.."
mkdir -p Frameworks

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -d "$PROJ_ROOT/$LLAMA_DIR/build-apple/llama.xcframework" ]; then
    rm -rf "$PROJ_ROOT/Frameworks/llama.xcframework"
    cp -R "$PROJ_ROOT/$LLAMA_DIR/build-apple/llama.xcframework" "$PROJ_ROOT/Frameworks/"
    echo "Copied to Frameworks/llama.xcframework"
    echo ""
    echo "Listing framework slices:"
    ls -d Frameworks/llama.xcframework/*/
    echo ""
    echo "=== Build complete ==="
else
    echo "Error: XCFramework not found at $PROJ_ROOT/$LLAMA_DIR/build-apple/llama.xcframework"
    echo "Check build output above for errors."
    exit 1
fi
