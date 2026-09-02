# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.16] – Unreleased

### New Features

- Added SIMD-accelerated ONNX CPU kernels with runtime dispatch for x86 and ARM.
- Added optimized `com.microsoft` kernels for `BiasGelu`, `CDist`, and
  `GroupQueryAttention`.
- Added global and session-local kernel registration APIs with public kernel and
  SIMD inspection.
- Added reusable C++ and Python backend correctness and benchmark runners.
- Added processor-performance profiling and machine-readable reports.

### Improvements

- Improved unary and binary elementwise execution with vectorized tails,
  specialized broadcasting, and calibrated parallel thresholds.
- Improved GEMM and MatMul blocking, packing, scheduling, fused bias, and
  half-precision and integer execution.
- Improved Attention for decode, prefill, and realistic FP16, BF16, and FP32
  model shapes.
- Shared SIMD normalization primitives and optimized normalization and
  TreeEnsemble execution.
- Reduced `BiasGelu` and `CDist` latency and added focused ONNX Runtime parity
  benchmarks.

### Fixes

- Fixed build and runtime consistency when linking against an onnx-light source
  checkout.
- Fixed release installation and cross-package runtime library lookup.
- Stabilized backend benchmark reporting and integer division validation.

### Documentation & CI

- Added generated operator and API catalogues, design documentation, development
  roadmaps, and benchmark galleries.
- Expanded cross-platform, sanitizer, coverage, formatting, and typing checks.
