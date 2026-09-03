---
name: add-optimized-kernel
description: Add or optimize an onnx-light-cpu kernel with runtime parallelism, AVX/AVX2, and AVX-512 dispatch.
---

# Add an optimized CPU kernel

Use this skill when adding a low-level CPU kernel or optimizing an existing one. Deliver a
portable scalar implementation, runtime-owned parallel scheduling, safe SIMD dispatch, focused
correctness tests, and reproducible benchmarks in one change.

## 1. Establish the contract

1. Locate the public entry point, runtime registration, shape/type validation, existing scalar
   implementation, tests, and benchmarks. Reuse the repository's conversions, math
   approximations, packing helpers, and tuning structures instead of duplicating them.
2. Record supported types, layouts, broadcasting, aliasing, empty tensors, tails, NaN/infinity,
   overflow, and numerical tolerance. Treat the scalar implementation as the semantic reference.
3. Benchmark representative small, medium, large, aligned, and tail shapes before editing. Do not
   retain an optimization that only moves work or allocation overhead outside the measured region.

## 2. Implement the portable path first

1. Keep architecture-neutral orchestration in the normal `impl/` translation unit. Validate
   inputs once, return cleanly for empty work, use checked size arithmetic, and avoid temporary
   full-tensor materialization.
2. Make the scalar kernel operate on an explicit contiguous range or independent output unit.
   SIMD functions should implement the same range contract so dispatch does not duplicate
   orchestration.
3. Preserve scalar fallbacks byte-for-byte where practical when adding a fast path. Isolate
   specialized algorithms in a helper and use a small eligibility gate.

## 3. Parallelize through the runtime executor

1. Partition independent output rows, tiles, batches, heads, or contiguous element ranges with
   `ExecuteRanges` or `ExecuteCostedRanges` from `onnx_light_cpu/impl/execution.h`. Never create a
   private thread pool or hard-code a thread count.
2. Use `ExecutionSchedule` when shape thresholds and participant limits are known. Use
   `ExecutionWorkCost` when bytes read/written and compute cycles can describe the work. Let the
   executor select the final participant count.
3. Choose a block multiple that preserves natural row, tile, or SIMD boundaries. Each worker must
   write a disjoint output range; avoid false sharing and synchronization inside the hot loop.
4. Parallelize at one level only. `ExecutionInParallelRegion()` already suppresses nested runtime
   parallelism; explicitly keep inner GEMMs or sub-kernels serial when an outer batch/head schedule
   owns the parallel decision.
5. Use bounded worker-local or `thread_local` scratch only when reuse is profitable. Resize it
   outside inner loops and include its peak memory in tests when material.
6. Add a fake-executor scheduling test that verifies useful block distribution, serial behavior
   for small inputs, and bounded nesting without asserting a machine-specific thread count.

## 4. Add AVX and AVX2/FMA kernels safely

1. Guard x86 intrinsics with the repository x86 platform checks. Pure AVX code may live beside the
   scalar dispatcher when that translation unit already follows the established pattern, as in
   `impl/math/abs_kernel.cc`.
2. Put AVX2/FMA code in a file ending in `_avx2_fma.cc`; CMake discovers that suffix, compiles it
   with `-mavx2 -mfma` (or `/arch:AVX2`), and defines
   `ONNX_LIGHT_CPU_HAVE_AVX2_FMA`. Guard declarations and calls with that definition.
3. Do not infer FMA or F16C from `SimdLevel::kAVX2`. Require `CpuSupportsFma()` or
   `CpuSupportsF16C()` when the selected function executes those instructions.
4. Use unaligned loads unless alignment is established by the API. Handle every remainder without
   out-of-bounds access, using masked loads/stores when profitable or the scalar reference for
   short tails.
5. Preserve all accumulation, bias, broadcast, alpha/beta, aliasing, and special-value modes in
   both full vectors and tails. Test vector widths minus one, exact width, width plus one, and all
   masked tail widths.

## 5. Add AVX-512 kernels in isolated translation units

1. Put AVX-512F implementations under an `avx512/` directory. CMake assigns `-mavx512f` only to
   those files and excludes them when the compiler lacks support. Never place AVX-512 intrinsics
   in the baseline dispatcher translation unit.
2. Guard declarations and dispatch with `ONNX_LIGHT_CPU_HAVE_AVX512`. Require the matching feature
   helper for extensions beyond AVX-512F: `CpuSupportsAvx512BW()`, `CpuSupportsAvx512Fp16()`,
   `CpuSupportsAvx512Bf16()`, or `CpuSupportsAvx512Vnni()`.
3. Prefer AVX-512 mask registers for tails. Construct masks without undefined shifts at zero or at
   the full lane count, and verify no masked pointer crosses an invalid allocation.
4. Consider frequency throttling and setup overhead. Dispatch AVX-512 only when measurements show
   it wins for the eligible shape; retain AVX2 for smaller workloads when appropriate.

## 6. Wire runtime dispatch

1. Select once per call or cache a shape-independent dispatch table. Use `DetectSimdLevel()` and
   compile-time guards together: compilation support does not prove runtime CPU/OS support.
2. Order dispatch from the most specialized eligible implementation to AVX2/AVX, then scalar.
   Use `level >= ...` when a lower-ISA kernel is safe on newer hardware. Use exact equality only
   when intentionally preserving a separate higher-ISA path.
3. Keep layout, stride, mask, alignment, and type restrictions in an explicit eligibility
   predicate. Unsupported cases must execute the existing fallback, not a partial fast path.
4. Follow CMake's recognized naming conventions. If a new ISA family does not match
   `*_avx2_fma.cc`, `avx512/`, or another existing source group, update the source properties and
   compiler feature check rather than relying on global flags.

## 7. Prove correctness and performance

1. Add differential tests against an independent scalar/reference result for every dispatch mode,
   type, layout, tail, empty input, special value, and fallback eligibility boundary.
2. Keep ISA-specific tests portable: skip direct intrinsic tests unless the required runtime
   features exist. Generic dispatch tests must still pass on x86 without AVX-512, ARM, macOS, and
   Windows.
3. Build an AVX2 ceiling configuration when validating AVX2 on an AVX-512 host:

   ```bash
   cmake -S . -B build-avx2 -DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2 \
         -DONNX_LIGHT_CPU_BUILD_TESTS=ON -DONNX_LIGHT_CPU_BUILD_PYTHON=OFF
   cmake --build build-avx2 -j4
   ```

4. Benchmark scalar/baseline, AVX2, and AVX-512 with identical inputs, affinity, executor, warmup,
   and alternating run order. Report median latency in seconds, shape, dtype, ISA, thread count,
   and speedup. Include aligned and tail-heavy cases and reject noisy or regressing changes.
5. Run the smallest affected C++ tests, cross-ISA tests when available, `clang-format` with the
   repository configuration, `clang-format --dry-run --Werror`, and `git diff --check`.

Do not claim an ISA speedup unless the benchmark proves that the intended ISA-specific function
ran. Do not trade numerical semantics, memory safety, portable builds, or executor ownership for
throughput.
