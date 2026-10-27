# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.0] – Unreleased

### New Features

- Added `RotaryEmbedding` CPU kernels for ONNX and `com.microsoft` with FLOAT, FLOAT16, and BFLOAT16 inputs and split-half or interleaved layouts ([#883](https://github.com/xadupre/onnx-light-cpu/pull/883)).
- Added `com.microsoft::MoE` decoder inference for FLOAT, FLOAT16, and BFLOAT16 with softmax top-k routing and interleaved SwiGLU ([#913](https://github.com/xadupre/onnx-light-cpu/pull/913)).
- Added `com.microsoft::QMoE` inference with INT4/INT8 CPU expert weights, block-wise scales, fused SwiGLU or separate FC3 projections, and FLOAT, FLOAT16, and BFLOAT16 activations ([#941](https://github.com/xadupre/onnx-light-cpu/pull/941)).
- Added `com.microsoft::MultiHeadAttention` v1 CPU support for self- and cross-attention, projection bias, masks, and KV caches with FLOAT, FLOAT16, and BFLOAT16 inputs ([#935](https://github.com/xadupre/onnx-light-cpu/pull/935)).
- Added `com.microsoft::CausalConvWithState` CPU support ([#934](https://github.com/xadupre/onnx-light-cpu/pull/934)).
- Added `com.microsoft::GatherBlockQuantized` CPU support for packed 2-/4-/8-bit embeddings with FLOAT/FLOAT16 scales, optional packed zero points, and INT32/INT64 indices, dequantizing only selected rows ([#973](https://github.com/xadupre/onnx-light-cpu/pull/973)).

### Improvements

- Extracted numerical implementations into the independently buildable `ml_cpu_compute` C++20 library, without ONNX or Python dependencies; ONNX adapters retain session-owned scheduling ([#953](https://github.com/xadupre/onnx-light-cpu/pull/953)).
- Aligned the package and remaining integration jobs with onnx-light 0.2.0 and its standalone proto headers ([#952](https://github.com/xadupre/onnx-light-cpu/pull/952), [#967](https://github.com/xadupre/onnx-light-cpu/pull/967)).
- Adopted onnx-light's required declarative preparation API for immutable `MatMulNBits` inputs, with deferred execution and source, temporary, and prepared memory estimates ([#956](https://github.com/xadupre/onnx-light-cpu/pull/956)).
- Supported encoded INT2/INT4/INT8 `MatMulNBits` weights with embedded scales, without repacking them for VNNI execution ([#895](https://github.com/xadupre/onnx-light-cpu/pull/895)).
- Added encoded INT3 `MatMulNBits` weights with block size 32 ([#915](https://github.com/xadupre/onnx-light-cpu/pull/915)).
- Extended INT2/INT4/INT8 `MatMulNBits` to power-of-two block sizes of at least 16, explicit zero points, and partial groups, including block-128 ternary weights ([#947](https://github.com/xadupre/onnx-light-cpu/pull/947)).
- Added prepared INT8 `MatMulNBits` paths for Qwen prefill and decode ([#893](https://github.com/xadupre/onnx-light-cpu/pull/893), [#923](https://github.com/xadupre/onnx-light-cpu/pull/923)).
- Added low-level constant-B AVX2 FP32 LM-head panel preparation and bounded BFLOAT16 `MatMul` conversion; LM-head preparation is not yet connected to the runtime `MatMul` kernel ([#922](https://github.com/xadupre/onnx-light-cpu/pull/922)).
- Added interleaved rotary pairing for `GroupQueryAttention` query and key inputs ([#906](https://github.com/xadupre/onnx-light-cpu/pull/906)).
- Added per-head attention sinks and partial rotary embeddings to `GroupQueryAttention` ([#933](https://github.com/xadupre/onnx-light-cpu/pull/933), [#942](https://github.com/xadupre/onnx-light-cpu/pull/942)).
- Optimized `BatchNormalization` for contiguous channels and improved normalization moment stability with Chan aggregation ([#870](https://github.com/xadupre/onnx-light-cpu/pull/870), [#888](https://github.com/xadupre/onnx-light-cpu/pull/888)).
- Added allocation-free COMPLEX64 and COMPLEX128 input and output support to the CPU Cast kernel.
- Added `com.microsoft::SkipLayerNormalization` for FLOAT, FLOAT16, and BFLOAT16 decoder graphs, including optional beta, bias, saved statistics, and residual outputs ([#896](https://github.com/xadupre/onnx-light-cpu/issues/896)).
- Added an explicitly opt-in, naive-kernel-gated rewrite removing redundant casts around `com.microsoft::SkipSimplifiedLayerNormalization`, preserving FP32 accumulation and residual semantics and converting exactly representable weights directly to FLOAT16 initializers ([#1044](https://github.com/xadupre/onnx-light-cpu/pull/1044)).
- Reduced AVX2 `Attention` latency with direct tile products, short-query tiling, and eligible fused softmax and strided output writes; the follow-up does not improve the default AVX-512 path ([#1024](https://github.com/xadupre/onnx-light-cpu/pull/1024), [#1042](https://github.com/xadupre/onnx-light-cpu/pull/1042)).
- Added guarded 32-row AVX2 tiles for `GroupQueryAttention` ([#1033](https://github.com/xadupre/onnx-light-cpu/pull/1033)).
- Reduced AVX2 `MatMul` square and skinny-M latency through register-resident accumulation and panel scheduling, and reduced `Gemm` register spills; skinny-N retains its original implementation ([#1037](https://github.com/xadupre/onnx-light-cpu/pull/1037), [#1039](https://github.com/xadupre/onnx-light-cpu/pull/1039)).
- Added packed AVX2 FLOAT comparisons for `Greater`, `GreaterOrEqual`, `Less`, and `LessOrEqual` ([#1004](https://github.com/xadupre/onnx-light-cpu/pull/1004), [#1008](https://github.com/xadupre/onnx-light-cpu/pull/1008), [#1012](https://github.com/xadupre/onnx-light-cpu/pull/1012), [#1031](https://github.com/xadupre/onnx-light-cpu/pull/1031)).
- Accelerated boolean `And`, `Or`, and `Xor` with AVX2 bulk and broadcast kernels ([#1019](https://github.com/xadupre/onnx-light-cpu/pull/1019), [#1020](https://github.com/xadupre/onnx-light-cpu/pull/1020), [#1026](https://github.com/xadupre/onnx-light-cpu/pull/1026)).
- Reduced AVX2 `Sigmoid` and `Softmax` register pressure and accelerated FLOAT `Tanh` with a rational approximation ([#1005](https://github.com/xadupre/onnx-light-cpu/pull/1005), [#1007](https://github.com/xadupre/onnx-light-cpu/pull/1007), [#1023](https://github.com/xadupre/onnx-light-cpu/pull/1023)).
- Accelerated FLOAT/FLOAT16 split-half `RotaryEmbedding` with AVX and AVX2/F16C vector paths ([#1014](https://github.com/xadupre/onnx-light-cpu/pull/1014)).
- Reduced FLOAT16 `CausalConvWithState` conversion overhead with bounded tiles and FLOAT16 `SimplifiedLayerNormalization` row overhead with batched dispatch ([#1021](https://github.com/xadupre/onnx-light-cpu/pull/1021), [#1022](https://github.com/xadupre/onnx-light-cpu/pull/1022)).
- Reduced AVX2 FLOAT `CDist` latency with 4-by-16 output tiles and `TreeEnsemble` batch interleaving overhead ([#1006](https://github.com/xadupre/onnx-light-cpu/pull/1006), [#1013](https://github.com/xadupre/onnx-light-cpu/pull/1013)).
- Enabled streaming stores for eligible serial, out-of-place AVX2 FLOAT `Add` and `Div` tensors exceeding both 32 MiB and detected L3 capacity, preserving cached stores for other paths ([#1032](https://github.com/xadupre/onnx-light-cpu/pull/1032), [#1036](https://github.com/xadupre/onnx-light-cpu/pull/1036)).
- Enabled AVX FLOAT `Abs` streaming stores at the L3-capacity boundary rather than only above it ([#1009](https://github.com/xadupre/onnx-light-cpu/pull/1009)).

### Fixes

- Accepted dynamic structured `MatMulNBits` weight inputs during shape inference without requiring a concrete encoded value ([#905](https://github.com/xadupre/onnx-light-cpu/pull/905)).
- Avoided double-counting shared `MatMulNBits` preparation buffers against the runtime preparation budget ([#961](https://github.com/xadupre/onnx-light-cpu/pull/961)).
- Normalized `MultiHeadAttention` softmax weights before FP32 value accumulation to avoid overflow for representable results, while preserving zero output for fully masked rows ([#943](https://github.com/xadupre/onnx-light-cpu/pull/943)).
- Fixed source builds with symlinked onnx-light extensions and compilation against its standalone threading library ([#932](https://github.com/xadupre/onnx-light-cpu/pull/932), [#955](https://github.com/xadupre/onnx-light-cpu/pull/955)).
- Fixed Windows release-wheel repair with external onnx-light DLLs ([#924](https://github.com/xadupre/onnx-light-cpu/pull/924)).
- Preserved FLOAT16/BFLOAT16 `com.microsoft::LinearAttention` graph construction after CPU registration, with graph-only backend cases kept separate from executable CPU cases; native Microsoft CPU execution remains FLOAT-only ([#1046](https://github.com/xadupre/onnx-light-cpu/pull/1046)).

### Documentation & CI

- Corrected the published-wheel compatibility guide to reference onnx-light 0.1.32 and added a regression check that keeps it aligned with the release version ([#901](https://github.com/xadupre/onnx-light-cpu/pull/901)).
- Generated C++ kernel and custom-operator API pages from Doxygen declarations ([#887](https://github.com/xadupre/onnx-light-cpu/pull/887)).
- Enabled weekly Dependabot updates for GitHub Actions and Python dependencies ([#872](https://github.com/xadupre/onnx-light-cpu/pull/872)).
- Verified Linux compatibility with recent onnx-light `main` revisions using the existing C++ and Python suites ([#963](https://github.com/xadupre/onnx-light-cpu/pull/963), [#965](https://github.com/xadupre/onnx-light-cpu/pull/965)).
- Fixed Pixi CI setup when no lockfile is committed ([#925](https://github.com/xadupre/onnx-light-cpu/pull/925)).
- Corrected benchmarks to feed CPU backend tensors directly and report loaded runtime versions, native module paths, and SIMD/build provenance ([#1042](https://github.com/xadupre/onnx-light-cpu/pull/1042)).
- Documented the unreproduced AVX2 `Split` performance gap and added middle-axis scheduling regressions without changing production kernels ([#1025](https://github.com/xadupre/onnx-light-cpu/pull/1025)).
- Extended coverage runs across processor dispatch levels and included standalone computation-library sources ([#1035](https://github.com/xadupre/onnx-light-cpu/pull/1035)).
- Scheduled weekly documentation builds with failure issues, archived completed 2026 plans, and fixed heading underlines that broke documentation builds ([#1016](https://github.com/xadupre/onnx-light-cpu/pull/1016), [#1011](https://github.com/xadupre/onnx-light-cpu/pull/1011), [#1040](https://github.com/xadupre/onnx-light-cpu/pull/1040)).

## [0.1.20] – 2026-10-05

### Improvements

- Added an SSSE3 fallback for INT8 and INT16 `Abs` on x86 processors without AVX2 ([#853](https://github.com/xadupre/onnx-light-cpu/pull/853)).
- Added packed `[Q | K | V]` inputs to optimized and scalar-reference `com.microsoft::GroupQueryAttention` kernels for FLOAT, FLOAT16, and BFLOAT16, including rotary embeddings and KV caches ([#862](https://github.com/xadupre/onnx-light-cpu/pull/862)).
- Prepared and compacted immutable `TreeEnsemble` plans during kernel construction, and replaced the private tuning registry with shared runtime tuning for FLOAT, DOUBLE, and FLOAT16 ([#866](https://github.com/xadupre/onnx-light-cpu/pull/866)).
- Moved immutable `MatMulNBits` INT4 preparation into runtime-owned storage so packed buffers participate in session residency accounting and eviction; dynamic inputs retain invocation-local plans without kernel synchronization ([#865](https://github.com/xadupre/onnx-light-cpu/pull/865)).

### Fixes

- Matched the effective native nanobind ABI used by onnx-light 0.1.30 in release wheels on Linux, Windows, and macOS ([#859](https://github.com/xadupre/onnx-light-cpu/pull/859)).
- Fixed Windows wheel builds by locating MSVC tools through `vswhere`, and macOS wheel repair by excluding separately installed onnx-light libraries ([#860](https://github.com/xadupre/onnx-light-cpu/pull/860)).
- Fixed the Windows imported core library configuration and macOS runtime library lookup during wheel repair ([#865](https://github.com/xadupre/onnx-light-cpu/pull/865)).
- Validated packed `GroupQueryAttention` widths and included unpacking, rotary, KV concatenation, and scalar score buffers in scratch-memory estimates ([#862](https://github.com/xadupre/onnx-light-cpu/pull/862)).
- Restored shared tuning and preparation hooks for session-local native kernel registration through the runtime lifecycle ([#866](https://github.com/xadupre/onnx-light-cpu/pull/866)).

## [0.1.19] – 2026-10-05

### Fixes

- Fixed release-wheel compatibility with onnx-light by building against its published wheel and matching its effective native nanobind ABI ([#854](https://github.com/xadupre/onnx-light-cpu/pull/854)).

## [0.1.18] – 2026-10-05

### New Features

- Added a native INT64 `ReduceSum` kernel with support for input and attribute axes, empty reductions, and exact integer arithmetic ([#846](https://github.com/xadupre/onnx-light-cpu/pull/846)).

### Improvements

- Updated development and AVX2 parity workflows to use onnx-light 0.1.30.
- Added 256-bit AVX-VNNI dispatch for packed INT8 and INT4 GEMM on supported processors without requiring AVX-512 ([#839](https://github.com/xadupre/onnx-light-cpu/pull/839)).

### Documentation & CI

- Pinned release-wheel builds and smoke tests to the declared onnx-light release instead of building against its `main` branch ([#834](https://github.com/xadupre/onnx-light-cpu/pull/834)).
- Added CI coverage for stable and native Python ABI builds and session-local kernel registration ([#843](https://github.com/xadupre/onnx-light-cpu/pull/843)).
- Evaluated AVX-512CD conflict detection for ScatterND and retained the existing scalar path after the prototype regressed on measured workloads ([#840](https://github.com/xadupre/onnx-light-cpu/pull/840)).
- Evaluated AVX-512VL narrow-vector and masked-tail candidates and retained the existing dispatch after finding no consistent kernel-level benefit ([#842](https://github.com/xadupre/onnx-light-cpu/pull/842)).

## [0.1.17] – 2026-10-02

### Fixes

- Included the compiled `_cpuregister` extension in release wheels so native kernel registration is available after installation.
- Added a pre-upload wheel smoke test that registers CPU kernels and verifies native `Abs` dispatch on Linux, Windows, and macOS.

### Documentation & CI

- Updated development and AVX2 parity workflows to use onnx-light 0.1.29.

## [0.1.16] – 2026-10-01

### New Features

- Added SIMD-accelerated ONNX CPU kernels with runtime dispatch for x86 and ARM.
- Added optimized `com.microsoft` kernels for `BiasGelu`, `CDist`, and `GroupQueryAttention`.
- Added an AVX2 accuracy-level-4 `com.microsoft::MatMulNBits` kernel using prepacked INT4 weights and quantized activations.
- Added `com.microsoft::SkipSimplifiedLayerNormalization` inference support for matching FLOAT/FLOAT16/BFLOAT16 inputs, optional bias, saved statistics, and residual-sum output, symbolic shape inference, native schema lookup, and zero CPU scratch memory.
- Added global and session-local kernel registration APIs with public kernel and SIMD inspection.
- Added reusable C++ and Python backend correctness and benchmark runners.
- Added processor-performance profiling and machine-readable reports.

### Improvements

- Improved unary and binary elementwise execution with vectorized tails, specialized broadcasting, and calibrated parallel thresholds.
- Improved GEMM and MatMul blocking, packing, scheduling, fused bias, and half-precision and integer execution, including fused FLOAT16 bias conversion.
- Improved Attention for decode, prefill, realistic FP16, BF16, and FP32 model shapes, AVX2 boolean-mask softmax, and non-aligned short queries.
- Added AVX2/F16C FLOAT16 Cast and Tanh paths.
- Shared SIMD normalization primitives and used the fused FLOAT16 affine path for BatchNormalization inference.
- Reduced `BiasGelu` and `CDist` latency and added focused ONNX Runtime parity benchmarks.
- Optimized TreeEnsemble execution.

### Fixes

- Fixed build and runtime consistency when linking against an onnx-light source checkout.
- Fixed release installation and cross-package runtime library lookup.
- Stabilized backend benchmark reporting and integer division validation.

### Documentation & CI

- Added generated operator and API catalogues, design documentation, development roadmaps, and benchmark galleries.
- Expanded cross-platform, sanitizer, coverage, formatting, and typing checks.
- Added tag-triggered wheel and standalone C++ release workflows that publish artifacts to the matching GitHub release.
