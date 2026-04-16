// IOSurfaceSharedBufferTest.swift
//
// Purpose: prove zero-copy FP16 IOSurface sharing between
//   - MLMultiArray(pixelBuffer:shape:)  (Core ML / ANE input)
//   - MTLDevice.makeBuffer(iosurface:plane:)  (Metal kernel input)
//
// This is a standalone verification for Open Question #8 of the Metal-LLM
// architecture. It does NOT load a Core ML model — the question is
// purely "do the two view APIs alias the same physical bytes?".
//
// Build and run on a Mac (arm64) or iOS simulator:
//
//   swiftc -O IOSurfaceSharedBufferTest.swift \
//       -framework CoreML -framework Metal -framework CoreVideo -framework IOSurface \
//       -o iosurface_test
//   ./iosurface_test
//
// Expected output on success:
//
//   [alloc] CVPixelBuffer OK  width=512  height=8192  format=L016 (fp16)
//   [alloc] IOSurface present: bytesPerRow=1024 allocSize=8388608
//   [MLMultiArray] shape=[16, 1, 512, 512]  dataType=Float16  count=4194304
//   [MTLBuffer]    length=8388608  contents=non-nil
//   [write] wrote 0x3C00 (1.0 fp16) to MLMultiArray[0]
//   [read]  MTLBuffer first fp16 = 0x3C00  MATCH
//   [write] wrote 0xC000 (-2.0 fp16) to MTLBuffer[1]
//   [read]  MLMultiArray[1] fp16 = 0xC000  MATCH
//   [row-stride] MLMultiArray stride[2] (row) = 512 elems * 2 B = 1024 B
//   [row-stride] CVPixelBufferGetBytesPerRow = 1024 B   (aligned to 64? YES)
//   [verdict] ZERO-COPY ALIAS CONFIRMED
//
// If the IOSurface path silently falls back to CPU-only (e.g. kCVPixelBufferMetalCompatibilityKey
// ignored), we expect MTLBuffer creation to fail — the test will print the failure
// and exit nonzero.

import Foundation
import CoreML
import CoreVideo
import Metal
import IOSurface

// ---------------------------------------------------------------------------
// Shape we want to share. Matches the sibling CoreML-LLM chunk layout for KV:
//   [slots, 1, seq_len, head_dim_padded]
// Concretely here: slots=16 (one quarter-chunk of KV), seq_len=512, head_dim=512.
// Total bytes = 16 * 512 * 512 * 2 = 8 MiB. Fits easily on device.
let slots   = 16
let seqLen  = 512
let headDim = 512

// For the CVPixelBuffer we flatten the 4-D tensor into (rows, cols):
//   rows = slots * 1 * seq_len
//   cols = head_dim
// MLMultiArray(pixelBuffer:shape:) is documented to reinterpret the row-major
// pixel bytes under the supplied shape, so MLMultiArray[s, 0, p, d] maps to
// pixel at row = s*seq_len + p, col = d  (row-major, fp16, zero stride gaps
// iff CVPixelBufferGetBytesPerRow == cols * 2).

let pbWidth  = headDim
let pbHeight = slots * seqLen     // 16 * 512 = 8192

// ---------------------------------------------------------------------------
// 1. Allocate a CVPixelBuffer with IOSurface backing + Metal compatibility.

let attrs: [String: Any] = [
    kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
    kCVPixelBufferMetalCompatibilityKey   as String: true,
]

var pbOpt: CVPixelBuffer?
let status = CVPixelBufferCreate(
    kCFAllocatorDefault,
    pbWidth, pbHeight,
    kCVPixelFormatType_OneComponent16Half,  // single-plane fp16
    attrs as CFDictionary,
    &pbOpt)

guard status == kCVReturnSuccess, let pb = pbOpt else {
    fputs("[FAIL] CVPixelBufferCreate returned \(status)\n", stderr)
    exit(1)
}
print("[alloc] CVPixelBuffer OK  width=\(pbWidth)  height=\(pbHeight)  format=L016 (fp16)")

// Zero the whole surface so stale bytes don't fool the alias check.
CVPixelBufferLockBaseAddress(pb, [])
memset(CVPixelBufferGetBaseAddress(pb)!, 0, CVPixelBufferGetDataSize(pb))
CVPixelBufferUnlockBaseAddress(pb, [])

guard let surface = CVPixelBufferGetIOSurface(pb)?.takeUnretainedValue() else {
    fputs("[FAIL] CVPixelBuffer has no backing IOSurface — not IOSurface-backed\n", stderr)
    exit(1)
}
let bytesPerRow = IOSurfaceGetBytesPerRow(surface)
let allocSize   = IOSurfaceGetAllocSize(surface)
print("[alloc] IOSurface present: bytesPerRow=\(bytesPerRow) allocSize=\(allocSize)")

// ---------------------------------------------------------------------------
// 2. Wrap as MLMultiArray with the desired 4-D shape.

let mlShape: [NSNumber] = [
    NSNumber(value: slots), 1,
    NSNumber(value: seqLen), NSNumber(value: headDim),
]

let mlArray: MLMultiArray
do {
    mlArray = try MLMultiArray(pixelBuffer: pb, shape: mlShape)
} catch {
    fputs("[FAIL] MLMultiArray(pixelBuffer:shape:) threw: \(error)\n", stderr)
    exit(1)
}
print("[MLMultiArray] shape=\(mlArray.shape)  dataType=\(mlArray.dataType.rawValue)  count=\(mlArray.count)")
precondition(mlArray.dataType == .float16, "expected Float16, got \(mlArray.dataType.rawValue)")

// Stride sanity: inner stride must be 1 element, row stride must be headDim
// elements == bytesPerRow / 2. If bytesPerRow > headDim*2 because IOSurface
// padded the row, the MLMultiArray is still valid but downstream Metal code
// must honour the stride.
let strides = mlArray.strides.map { $0.intValue }
print("[row-stride] MLMultiArray strides (elems) = \(strides)")
print("[row-stride] CVPixelBufferGetBytesPerRow   = \(bytesPerRow) B   " +
      "(aligned to 64? \(bytesPerRow % 64 == 0 ? "YES" : "NO"))")

// ---------------------------------------------------------------------------
// 3. Wrap same IOSurface as MTLBuffer.

guard let device = MTLCreateSystemDefaultDevice() else {
    fputs("[FAIL] no Metal device\n", stderr); exit(1)
}
guard let mtlBuf = device.makeBuffer(iosurface: surface, plane: 0) else {
    fputs("[FAIL] makeBuffer(iosurface:plane:) returned nil — " +
          "IOSurface not Metal-compatible?\n", stderr)
    exit(1)
}
print("[MTLBuffer]    length=\(mtlBuf.length)  contents=\(mtlBuf.contents() == UnsafeMutableRawPointer(bitPattern: 0) ? "nil" : "non-nil")")

// ---------------------------------------------------------------------------
// 4. Round-trip alias test.
//    (a) MLMultiArray write -> MTLBuffer read
//    (b) MTLBuffer write    -> MLMultiArray read

// fp16 bit patterns:
let fp16_one:   UInt16 = 0x3C00   // +1.0
let fp16_negTwo: UInt16 = 0xC000  // -2.0

// (a) Write via MLMultiArray.
do {
    // MLMultiArray gives us typed pointer to element [0,0,0,0].
    mlArray.dataPointer
        .bindMemory(to: UInt16.self, capacity: mlArray.count)[0] = fp16_one
    print("[write] wrote 0x\(String(fp16_one, radix: 16, uppercase: true)) (1.0 fp16) to MLMultiArray[0]")

    // Read via MTLBuffer raw pointer.
    let p = mtlBuf.contents().bindMemory(to: UInt16.self, capacity: mtlBuf.length / 2)
    let readBack = p[0]
    let ok = (readBack == fp16_one)
    print("[read]  MTLBuffer first fp16 = 0x\(String(readBack, radix: 16, uppercase: true))  " +
          (ok ? "MATCH" : "MISMATCH"))
    if !ok { exit(2) }
}

// (b) Write via MTLBuffer.
do {
    let p = mtlBuf.contents().bindMemory(to: UInt16.self, capacity: mtlBuf.length / 2)
    // Write to element index 1 under the pixel buffer's stride. Since we asked
    // for cols=512 and a fp16 row of 512 elems == 1024 B, element 1 of the
    // logical MLMultiArray layout is pixel buffer byte offset 2 — UNLESS
    // the IOSurface added padding bytes per row; at row 0 indexing is
    // identical either way so we're safe here.
    p[1] = fp16_negTwo
    print("[write] wrote 0x\(String(fp16_negTwo, radix: 16, uppercase: true)) (-2.0 fp16) to MTLBuffer[1]")

    let q = mlArray.dataPointer
        .bindMemory(to: UInt16.self, capacity: mlArray.count)
    let readBack = q[1]
    let ok = (readBack == fp16_negTwo)
    print("[read]  MLMultiArray[1] fp16 = 0x\(String(readBack, radix: 16, uppercase: true))  " +
          (ok ? "MATCH" : "MISMATCH"))
    if !ok { exit(3) }
}

// ---------------------------------------------------------------------------
// 5. Padding check: if CVPixelBufferGetBytesPerRow != headDim*2, the two
//    views disagree on row-stride and downstream Metal kernels must be told.
//    We don't fail the test here, just surface it.

let expectedRowBytes = headDim * 2
if bytesPerRow != expectedRowBytes {
    print("[note] IOSurface added \(bytesPerRow - expectedRowBytes) B of per-row padding. " +
          "Metal kernels must stride by \(bytesPerRow) B per row, not \(expectedRowBytes) B.")
} else {
    print("[note] IOSurface row pitch is exactly headDim*2 — no stride surprise.")
}

print("[verdict] ZERO-COPY ALIAS CONFIRMED")
