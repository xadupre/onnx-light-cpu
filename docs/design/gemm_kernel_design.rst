Gemm Kernel Design
==================

This page documents the internal design of the ``Gemm`` (general matrix
multiplication) CPU kernel implemented in
``onnx_light_cpu/impl/math/gemm_kernel.cc`` and
``onnx_light_cpu/impl/math/gemm/avx512/gemm_kernel_avx512.cc``: how it picks a
micro-kernel at runtime, and which further optimizations were considered but
not (yet) implemented, with their expected gain and risk.

Kernel selection decision tree
-------------------------------

The kernel is selected **once per (element type, process)** by
``SelectGemmKernelKind<T>()`` and reused for every ``Gemm`` call; the decision
combines a compile-time capability (was the code built with a given
instruction set enabled?) with a runtime capability (does the executing CPU
actually support it, via ``DetectSimdLevel()``, which is a one-time CPUID
probe)::

    SelectGemmKernelKind<T>()
    │
    ├─ Is this a non-x86 build (ONNX_LIGHT_CPU_X86 == 0)?
    │  └─ yes → GemmKernelKind::kScalar
    │           (portable C++ fallback; also the tail handler on x86)
    │
    └─ x86 build: read the *runtime* SIMD level once (DetectSimdLevel(), cached)
       │
       ├─ Was gemm_kernel_avx512.cc compiled in?
       │  (ONNX_LIGHT_CPU_HAVE_AVX512 defined, i.e. the compiler accepted
       │  -mavx512f -- see the CheckCXXCompilerFlag probe in CMakeLists.txt)
       │  │
       │  ├─ yes AND runtime level >= SimdLevel::kAVX512
       │  │  └─ GemmKernelKind::kAVX512
       │  │     (512-bit vectors, NR=2: 32 float / 16 double lanes per step,
       │  │      gemm_kernel_avx512.cc, its own translation unit so only this
       │  │      file needs -mavx512f)
       │  │
       │  └─ no, or runtime level < kAVX512 → fall through
       │
       ├─ runtime level >= SimdLevel::kAVX
       │  └─ GemmKernelKind::kAVX
       │     (256-bit vectors, NR=2: 16 float / 8 double lanes per step;
       │      always compiled in at the baseline ONNX_LIGHT_CPU_SIMD_FLAGS,
       │      default -mavx2, so it is present in every build)
       │
       ├─ runtime level >= SimdLevel::kSSE2
       │  └─ GemmKernelKind::kSSE2
       │     (128-bit vectors, NR=2: 8 float / 4 double lanes per step;
       │      the safe baseline for any x86-64 CPU, which guarantees SSE2)
       │
       └─ otherwise → GemmKernelKind::kScalar

Two independent axes are worth calling out:

* **Compile-time gate** (``ONNX_LIGHT_CPU_HAVE_AVX512``): decided once, at
  build time, by whether the toolchain accepts ``-mavx512f`` /
  ``/arch:AVX512``. If it does not (e.g. an older compiler, or a
  cross-compilation target), ``gemm_kernel_avx512.cc`` is excluded from the
  source list entirely (see ``CMakeLists.txt``) and the macro is never
  defined, so the whole ``kAVX512`` branch above compiles out.
* **Runtime gate** (``DetectSimdLevel()``): decided once per process, by
  CPUID, regardless of what the compiler supports. This is what makes a
  single binary built on an AVX-512-capable machine still run correctly (via
  automatic fallback to AVX2/SSE2/scalar) on a CPU that lacks it.

Independently of the branch selected above, every micro-kernel call also goes
through two cache-blocking / packing steps in ``GemmImpl`` before the
micro-kernel is invoked (see the file-level comment in ``gemm_kernel.cc`` for
the full rationale):

1. ``K`` is split into ``kGemmTileK``-sized chunks, and ``A`` and ``B`` are
   each packed into small contiguous buffers per (task, k-chunk) --
   ``PackARowBlock`` and ``PackBPanel`` -- so the hot inner loop only ever
   touches L1/L2-resident, unit-stride memory regardless of ``trans_a`` /
   ``trans_b`` or the caller's strides.
2. The output grid is flattened into ``(row block, column panel)`` tasks and
   spread across ``ParallelFor`` on **both** the ``M`` and ``N`` axes, so
   "skinny" shapes (e.g. ``M == 1``, a single-example matvec) still get
   parallelism instead of running on a single thread.

Prepared execution interfaces
-----------------------------

``onnx_light_cpu/impl/math/gemm/gemm_plan.h`` defines reusable typed plans for
``float32`` and ``float64``. ``GemmPlan`` records dimensions, transpose and
scaling attributes, current cache/register blocking, useful parallelism, and
the typed kernel entry point. It can also own a constant B matrix so its
lifetime is independent of the caller. ``MatMulPlan`` implements ONNX rank-1
promotion, batched matrix multiplication, multidirectional batch broadcasting,
transpose-aware matrix dimensions, empty dimensions, and plan-owned constant B
tensors. ``StridedBatchedGemm`` and ``GroupedGemm`` expose uniform and
heterogeneous batches respectively. Plans derive ``MC``, ``NC``, and ``KC``
from deterministic CPUID cache descriptors on x86, align them to register
tiles, and retain conservative defaults when cache discovery is unavailable.
The selected values are passed to the five-loop engine rather than being
descriptive metadata.

The plan selects the general five-loop engine or a direct, skinny-M, skinny-N,
or split-K path once from the prepared shape. Persistent packed constant panels
are introduced by a later roadmap step.

Platform support (x86_64)
-------------------------

Every vectorized micro-kernel (AVX-512/AVX/SSE2) is x86-specific: it is
written directly against ``<immintrin.h>`` intrinsics and gated behind
``#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||
defined(_M_IX86)`` (``ONNX_LIGHT_CPU_X86``). ``DetectSimdLevel()``
(``onnx_light_cpu/impl/simd_level.cc``) is likewise x86-only: it uses CPUID
and XGETBV. On any x86_64 platform (Intel or AMD, Linux, Windows, or macOS)
this gate evaluates true and the kernel gets full SIMD acceleration; see
"Non-x86 platforms" below for what happens where it does not.

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Platform
     - Status
     - Notes
   * - Linux x86_64 (Intel)
     - Full support
     - Reference platform for this work; AVX2 always compiled in
       (``ONNX_LIGHT_CPU_SIMD_FLAGS`` default ``-mavx2``), AVX-512 compiled in
       when the toolchain accepts ``-mavx512f`` and used automatically when
       ``DetectSimdLevel()`` reports it on the running CPU.
   * - Linux / Windows x86_64 (AMD)
     - Full support, no changes needed
     - AMD is x86_64 like Intel: the same CPUID-based ``DetectSimdLevel()``
       and the same AVX/AVX2/AVX-512 micro-kernels apply unchanged. AVX2+FMA3
       has been available since Zen 1 (2017), so the ``kAVX`` path is active
       on any reasonably recent AMD CPU. AVX-512 is only available starting
       with Zen 4 (2022, Ryzen 7000 / EPYC Genoa); on Zen 1-3
       ``DetectSimdLevel()`` reports at most ``kAVX2``/``kAVX``, so the kernel
       automatically falls back to the AVX2 micro-kernel there -- no crash,
       just no AVX-512 codegen.
   * - Windows x86_64 (Intel/AMD)
     - Full support
     - Same code paths as Linux x86_64; the MSVC-specific flag spellings
       (``/arch:AVX2``, ``/arch:AVX512``) and intrinsics (``__cpuidex``,
       ``_xgetbv``) are already handled in ``CMakeLists.txt`` /
       ``simd_level.cc``.
   * - macOS, Intel (x86_64)
     - Full support
     - Same code paths as Linux/Windows x86_64. AVX-512 hardware support is
       rare on Intel Macs (Apple never shipped an AVX-512-capable Mac), so in
       practice most Intel Macs run the ``kAVX``/``kAVX2`` path.

Non-x86 platforms (ARM / Apple Silicon)
----------------------------------------

There is currently **no NEON, SVE, or other non-x86 SIMD implementation**.
This does not make the kernel fail on non-x86 platforms --
``GemmMicroKernel_ScalarImpl`` is a portable, architecture-agnostic C++
fallback that ``SelectGemmKernelKind<T>()`` returns whenever
``ONNX_LIGHT_CPU_X86 == 0`` -- but it does mean those platforms get **no SIMD
acceleration at all**, only the scalar path, so Gemm calls are correct but
much slower there than on x86.

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Platform
     - Status
     - Notes
   * - macOS, Apple Silicon (M1/M2/M3/M4, ARM64/AArch64)
     - Compiles and runs correctly, **no SIMD acceleration**
     - ``ONNX_LIGHT_CPU_X86`` evaluates to ``0``, so ``SelectGemmKernelKind<T>()``
       always returns ``GemmKernelKind::kScalar``: Gemm calls are correct but
       run through the un-vectorized scalar micro-kernel only. Adding NEON
       (and, if targeted, SVE/SVE2) micro-kernels -- see
       "Remaining optimizations" below -- would close this gap; this was
       flagged but not implemented in this pass.
   * - Other ARM64/AArch64 (e.g. Linux on ARM servers, Android)
     - Compiles and runs correctly, **no SIMD acceleration**
     - Same reasoning as Apple Silicon above: falls back to the scalar
       micro-kernel.

Implemented optimizations (for reference)
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Optimization
     - Status
   * - K-blocking (``kGemmTileK``)
     - Implemented -- bounds the packed ``B`` panel to an L2-resident size.
   * - Hierarchical L1/L2/L3 cache blocking
     - Implemented -- deterministic x86 CPUID cache descriptors select aligned
       ``MC``, ``NC``, and ``KC`` values stored in each plan; conservative
       defaults cover unavailable cache discovery.
   * - 2D M x N task parallelism
     - Implemented -- flattened task list, fixes the ``M == 1`` starvation case.
   * - A-panel packing (``PackARowBlock``)
     - Implemented -- resolves ``trans_a`` once per (row block, k-chunk).
   * - B-panel packing (``PackBPanel``)
     - Implemented -- reused across every A row block of a task/k-chunk.
   * - NR=2 wide micro-kernel tiles
     - Implemented -- AVX2/SSE2/AVX-512, all element types.
   * - AVX-512 micro-kernel
     - Implemented -- separate TU, compile+runtime gated (see tree above).
   * - Software prefetch (``B`` rows, T0 hint)
     - Implemented -- AVX2/AVX-512 NR=2 loops, ``kGemmPrefetchDistanceK = 4``.

Remaining optimizations (not implemented)
--------------------------------------------

Assuming a machine **dedicated** to this workload (no contention from other
tenants), so the estimates below reflect achievable, not just theoretical,
gains.

.. list-table::
   :header-rows: 1
   :widths: 22 30 24 24

   * - Optimization
     - Description
     - Likely gain
     - Risk
   * - Explicit k-loop unrolling
     - Manually unroll the inner ``k`` reduction (e.g. by 4) in the NR=2
       AVX2/SSE2/AVX-512 kernels to expose more independent FMA chains to the
       out-of-order scheduler, on top of what ``-O2``/``-O3`` may already do.
     - Small, ~5-10% on AVX2; a bit more on AVX-512 (more registers headroom).
       Real bottleneck is more likely memory bandwidth/loop overhead than FMA
       latency, which caps the upside.
     - **Register spilling**: AVX2 has only 16 YMM registers; the kernel
       already holds 8 live accumulators (``acc0``/``acc1`` x ``kGemmMR``).
       Unrolling by 4 without care can need 32+ live registers, forcing the
       compiler to spill to the stack and erase the gain (or regress). Needs a
       remainder loop for ``K`` not a multiple of the unroll factor. Lower risk
       on AVX-512 (32 ZMM registers).
   * - Hand-written assembly micro-kernels
     - Replace the C++/intrinsics micro-kernels with hand-scheduled assembly
       (wider tiles, e.g. 8x8 or 24x8, explicit instruction interleaving),
       mirroring OpenBLAS/BLIS.
     - Potentially large (BLAS-competitive), but only with substantial tuning
       per microarchitecture.
     - High engineering/maintenance cost; fragile across compiler versions and
       CPU generations; loses the portability of intrinsics-based code; large
       testing surface (correctness + performance regressions per target).
   * - Per-microarchitecture dispatch (DYNAMIC_ARCH style)
     - Detect the specific microarchitecture (Zen vs. Skylake vs. Haswell,
       etc.) instead of only the instruction-set level, and pick a
       microarchitecture-tuned kernel/tile size.
     - Moderate on mixed fleets; ~0 on a single, known, dedicated machine
       (the point of this row's premise) since there is only one
       microarchitecture to tune for.
     - Combinatorial growth of kernel variants to maintain and test; detection
       logic (CPUID family/model parsing) is itself a source of bugs.
   * - Native SIMD float16 / bfloat16 Gemm
     - **Large effort**, the remaining half of the work started in this pass.
       ``Gemm`` now has a **scalar/correctness layer** for
       ``float16``/``bfloat16``: ``GemmKernel`` widens ``A``/``B``/the bias
       ``C`` to ``float32`` (via ``onnx-light``'s ``Float16BitsToFloat`` /
       ``Bfloat16BitsToFloat``), calls the existing SIMD-accelerated
       ``GemmFloat32`` for the reduction, and rounds ``Y`` back down (see
       ``onnx_light_cpu/kernels/math/gemm_kernel.cc``) -- mirroring the
       approach ``Abs``/``Exp``/``Log`` already use for these types (see
       ``onnx_light_cpu/impl/math/exp_log_kernel.cc``). This is correct and
       reuses the full float32 SIMD path for the matmul itself, but the
       widen/round-trip is pure per-element overhead with no vectorization of
       its own (see ``plot_gemm_dtype_benchmark.py`` in the benchmark gallery
       for measured overhead across representative shapes). What remains is
       the **SIMD acceleration layer**: real half-precision vector
       micro-kernels (AVX2 ``F16C``/``VCVTPH2PS`` for convert-then-FMA-in-
       float32, or native ``AVX-512FP16``/``BF16`` instructions where present)
       integrated into the existing packing/blocking/dispatch machinery
       (``PackARowBlock``/``PackBPanel``, ``GemmKernelKind`` dispatch,
       per-file compile flags analogous to the AVX-512 ``.cc`` split), which
       would also fold the widen/round-trip into the vectorized loop instead
       of a separate full-buffer pass.
     - Potentially large throughput gain (up to 2x the lanes per vector versus
       float32, more with native BF16/FP16 dot-product instructions, plus
       removing the separate widen/round-trip pass) for models that use half
       precision on hardware that supports it, but ``0`` for the existing
       float32/float64 workloads this repository's callers use today.
     - Mixed-precision accumulation correctness (accumulate in float32 to
       avoid catastrophic cancellation, only round to half precision on
       output -- gets the summation-order subtleties already seen with
       float32 K-chunking, amplified by lower mantissa precision); detecting
       ``F16C``/``AVX-512FP16``/``AVX-512BF16`` support at both compile time
       and runtime (more branches in ``SelectGemmKernelKind``); a
       significantly larger test matrix (2 more element types x every
       existing Gemm test: transpose variants, bias/alpha/beta, K-chunking,
       multi-panel, etc.); out of scope unless a caller actually needs
       SIMD-accelerated half precision beyond the widen/round-trip path
       already shipped.
   * - ARM NEON / SVE micro-kernels
     - Port the AVX2-style NR=2 micro-kernel design to ``float32x4_t`` /
       ``float64x2_t`` NEON intrinsics (128-bit, all ARM64), and optionally to
       SVE/SVE2 (variable-width vectors) where available server-side, with a
       ``DetectSimdLevel()``-equivalent ARM feature probe (``getauxval(AT_HWCAP)``
       or ``/proc/cpuinfo``) instead of CPUID.
     - Large on ARM64 targets (macOS Apple Silicon, ARM Linux servers), which
       currently get **no** SIMD acceleration at all (scalar fallback only,
       see "Platform support" above); ``0`` on x86 targets.
     - New translation unit(s), new feature-detection code path, and a full
       new test/benchmark matrix on ARM hardware; SVE's variable vector width
       is a different programming model from NEON's fixed 128-bit registers,
       so it is effectively a second port rather than a small extension.

The most favorable next step, if any, is likely explicit k-loop unrolling on
the AVX-512 kernel specifically (more registers, less spilling risk) -- but
given the modest expected gain versus the completed optimizations above, it
was not implemented in this pass.
