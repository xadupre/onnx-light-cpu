# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.16] – Unreleased

### New Features

- Added SIMD-accelerated ONNX CPU kernels with runtime dispatch for x86 and ARM.
- Added optimized `com.microsoft` kernels for `BiasGelu`, `CDist`, and
  `GroupQueryAttention`.
- Added an AVX2 accuracy-level-4 `com.microsoft::MatMulNBits` kernel using
  prepacked INT4 weights and quantized activations.
- Added `com.microsoft::SkipSimplifiedLayerNormalization` inference support
  for matching FLOAT/FLOAT16/BFLOAT16 inputs, optional bias, saved statistics,
  and residual-sum output, symbolic shape inference, native schema lookup,
  and zero CPU scratch memory.
- Added global and session-local kernel registration APIs with public kernel and
  SIMD inspection.
- Added reusable C++ and Python backend correctness and benchmark runners.
- Added processor-performance profiling and machine-readable reports.

### Improvements

- Improved unary and binary elementwise execution with vectorized tails,
  specialized broadcasting, and calibrated parallel thresholds.
- Improved GEMM and MatMul blocking, packing, scheduling, fused bias, and
  half-precision and integer execution, including fused FLOAT16 bias conversion.
- Improved Attention for decode, prefill, realistic FP16, BF16, and FP32 model
  shapes, AVX2 boolean-mask softmax, and non-aligned short queries.
- Added AVX2/F16C FLOAT16 Cast and Tanh paths.
- Shared SIMD normalization primitives and used the fused FLOAT16 affine path
  for BatchNormalization inference.
- Reduced `BiasGelu` and `CDist` latency and added focused ONNX Runtime parity
  benchmarks.
- Optimized TreeEnsemble execution.

### Fixes

- Fixed build and runtime consistency when linking against an onnx-light source
  checkout.
- Fixed release installation and cross-package runtime library lookup.
- Stabilized backend benchmark reporting and integer division validation.

### Documentation & CI

- Added generated operator and API catalogues, design documentation, development
  roadmaps, and benchmark galleries.
- Expanded cross-platform, sanitizer, coverage, formatting, and typing checks.
- Added tag-triggered wheel and standalone C++ release workflows that publish
  artifacts to the matching GitHub release.
