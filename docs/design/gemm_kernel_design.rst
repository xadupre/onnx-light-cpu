Gemm Kernel Design
==================

This page documents the internal design of the ``Gemm`` (general matrix
multiplication) CPU kernel implemented in
``onnx_light_cpu/impl/math/gemm_kernel.cc`` and
``onnx_light_cpu/impl/math/gemm/avx512/gemm_kernel_avx512.cc``: how it picks a
micro-kernel at runtime, and which further optimizations were considered but
not (yet) implemented, with their expected gain and risk. Delivery order,
performance gates, and remaining work are tracked in the
:doc:`Gemm, MatMul, and Attention roadmap
<../next_steps/2026_08_gemm_matmul>`.

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
    │  └─ yes → ARM (or portable) path:
    │           ├─ NEON compiled in (ONNX_LIGHT_CPU_HAVE_NEON) and, once probed
    │           │  by DetectArmGemmProfile():
    │           │  ├─ profile == kSve  (ONNX_LIGHT_CPU_HAVE_SVE, scalable
    │           │  │                    vectors) → GemmKernelKind::kSve
    │           │  └─ profile == kNeon (128-bit Advanced SIMD)
    │           │                       → GemmKernelKind::kNeon
    │           └─ otherwise → GemmKernelKind::kScalar
    │                          (portable C++ fallback; also the tail handler on x86)
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
       ├─ Was gemm_kernel_avx2_fma.cc compiled in
       │  (ONNX_LIGHT_CPU_HAVE_AVX2_FMA) AND runtime level >= SimdLevel::kAVX2
       │  AND CpuSupportsFma()?
       │  └─ yes → GemmKernelKind::kAVX2FMA
       │           (fused multiply-add AVX2 kernel: 256-bit vectors with a
       │            single rounding per multiply-add, faster and more accurate
       │            than the plain-AVX path below)
       │
       ├─ runtime level >= SimdLevel::kAVX
       │  └─ GemmKernelKind::kAVX
       │     (256-bit vectors without FMA, NR=2: 16 float / 8 double lanes per
       │      step; always compiled in at the baseline ONNX_LIGHT_CPU_SIMD_FLAGS,
       │      default -mavx2, so it is present in every build)
       │
       ├─ runtime level >= SimdLevel::kSSE2
       │  └─ GemmKernelKind::kSSE2
       │     (128-bit vectors, NR=2: 8 float / 4 double lanes per step;
       │      the safe baseline for any x86-64 CPU, which guarantees SSE2)
       │
       └─ otherwise → GemmKernelKind::kScalar

The same decision tree as an interactive SVG (hover a box for the rationale
behind each branch):

.. raw:: html
   :file: _static/gemm/kernel_tree.svg

The green leaves are the ``GemmKernelKind`` values, annotated with their SIMD
width and rough relative gain; ``Scalar`` is the portable C++ kernel that
multiplies one element at a time (no SIMD), used both as the correctness
fallback and as the sub-vector tail of every vector kernel. This tree only
picks the *micro-kernel*: every kind then runs through the same
cache-blocking and A/B packing (below), and the *algorithm* (Direct,
skinny-M, skinny-N, Split-K, or the general five-loop path) is selected
separately from the shape -- see :ref:`the strategy-zones figure
<gemm-strategy-zones>` further down.

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
   ``PackAPanel`` and ``PackBPanel`` -- so the hot inner loop only ever
   touches L1/L2-resident, unit-stride memory regardless of ``trans_a`` /
   ``trans_b`` or the caller's strides.
2. The general path schedules the Cartesian product of row and column panels.
   Column panels are grouped into bounded waves, so every active B panel is
   packed once and shared without allocating an unbounded ``K x N`` workspace.

Task scheduler decomposition
----------------------------

The five-loop scheduler maps panels of ``A`` and ``B`` to disjoint rectangular
zones of ``Y``:

.. code-block:: text

                 B (K x N)
            +--------+--------+--------+
            | B0     | B1     | B2     |  NC-wide column panels
            +--------+--------+--------+

    A (M x K)                 Y (M x N)
    +--------+            +------+------+------+
    | A0     |----------->| T00  | T01  | T02  |
    +--------+            +------+------+------+
    | A1     |----------->| T10  | T11  | T12  |
    +--------+            +------+------+------+
    | A2     |----------->| T20  | T21  | T22  |
    +--------+            +------+------+------+
      MC-high                 independent output zones
      row panels

``T(i,j)`` computes ``Ai @ Bj`` and writes only output zone ``Y(i,j)``.
Consequently, tasks in the grid need no output locks. With six threads and
three row panels, two column panels form a full wave:

.. code-block:: text

   wave 1: B0, B1 -> T00 T10 T20 T01 T11 T21
   wave 2: B2     -> T02 T12 T22

For every K chunk, ``B0`` and ``B1`` are each packed once before the first
wave. Their packed buffers remain read-only while the six tasks reuse them.
Each task packs the required A row panel and accumulates into its own zone of
``Y``. Shape-aware constraints reduce cache-derived ``MC`` and ``NC`` only
when the original values would expose fewer useful tasks than available
threads.

The scheduler chooses the outermost useful dimension in this order:

1. independent small batch items;
2. the M x N panel grid of one GEMM;
3. split-K when neither batch nor M/N provides enough tasks.

Split-K divides the reduction dimension into independent packed SIMD products
and combines their partial outputs:

.. code-block:: text

   K = [K0 | K1 | K2]
         |    |    |
         v    v    v
        P0   P1   P2  ->  Y = alpha * (P0 + P1 + P2) + beta * C

When a product is already running inside a parallel batch region, nested
split-K is disabled: the product executes its M x N grid directly, avoiding
serial split-K partitions, temporary partial buffers, and a redundant
reduction.

Thread runtime and affinity
---------------------------

``onnx-light-cpu`` does not own worker threads. Registered GEMM kernels receive
the session ``CpuExecutor`` from ``onnx-light``; that executor owns processor
discovery, affinity, spin/park behavior, nesting, and diagnostics. Standalone
GEMM calls execute on the calling thread, allowing embedding applications to
partition work with their existing scheduler without hidden workers.

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
heterogeneous batches respectively. Small independent products can be
scheduled across the current session ``CpuExecutor``; products with useful
internal M/N parallelism keep the batch loop serial. Plans derive ``MC``,
``NC``, and ``KC`` from deterministic CPUID cache descriptors on x86, align
them to
register tiles, then reduce ``MC``/``NC`` when necessary to expose enough work
for the available threads. The selected values and ``useful_threads`` estimate
drive execution rather than being descriptive metadata.

A shape is *skinny* when one output dimension is smaller than the SIMD tile,
so the ordinary 2D micro-kernel would leave most of its vector lanes idle:
**skinny-M** means few rows (``m <= register_rows``, e.g. an ``M == 1`` matvec)
and **skinny-N** means few columns (``n <= vector_lanes``, e.g. a single output
column). In those cases the kernel vectorizes the large dimension (or the
``K`` reduction) instead of the tiny one.

The plan selects the general five-loop engine or a direct, skinny-M, skinny-N,
or split-K path once from the prepared shape. It may own constant B in its
original representation; persistent B prepacking is explicitly excluded from
the roadmap. The skinny-N path reduces each output element through several
partial accumulators over a unit-stride ``K`` body with an exact scalar tail,
so single-column and small-N shapes vectorize the reduction instead of walking
it with a serial dependency chain. A single output column (``N == 1``) is kept
on this path rather than split-K, whose partition and partial-reduction
overhead would dominate the tiny output. The skinny-M path is the dual GEMV
kernel for the few-row shapes (``M == 1`` matvec or a short batch): it streams
each ``B`` row once per ``k``, reuses it across the output rows with a
broadcast axpy over the output columns, and keeps one column panel's
accumulators in a small ``M x nb`` buffer. The axpy is unit-stride for
non-transposed ``B``, so it vectorizes over ``N`` -- the useful dimension when
``M`` is tiny -- while work is parallelized over ``N`` column panels.

``detail::SelectGemmAlgorithm()`` (``gemm_plan.cc``) picks one of these five
strategies from ``m``, ``n``, ``k``, ``trans_a``/``trans_b``, and the
target's ``vector_lanes``/``register_rows``. Split-K is checked first (it only
applies to a narrow, tiny-output/long-``K`` corner); the direct path follows
for small ``k`` with natural (non-transposed) layouts; everything else is
decided by comparing ``m`` to ``register_rows`` and ``n`` to ``vector_lanes``.
The interactive SVG below maps these thresholds onto the ``M`` x ``N`` output
plane, with the ``K``/transpose-gated Direct and Split-K overlays called out
separately (hover a zone for the exact condition):

.. _gemm-strategy-zones:

.. raw:: html
   :file: _static/gemm/strategy_zones.svg

Platform support
----------------

x86_64
~~~~~~

Every vectorized micro-kernel (AVX-512/AVX/SSE2) is x86-specific: it is
written directly against ``<immintrin.h>`` intrinsics and gated behind
``#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||
defined(_M_IX86)`` (``ONNX_LIGHT_CPU_X86``). ``DetectSimdLevel()``
(``onnx_light_cpu/impl/simd_level.cc``) is likewise x86-only: it uses CPUID
and XGETBV. On any x86_64 platform (Intel or AMD, Linux, Windows, or macOS)
this gate evaluates true and the kernel gets full SIMD acceleration.

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

ARM64 / Apple Silicon
~~~~~~~~~~~~~~~~~~~~~

ARM64 has dedicated FP32 and FP64 micro-kernels. The fixed-width NEON kernel
uses six output rows by two 128-bit vectors (``float32x4_t`` or
``float64x2_t``), reuses each loaded B vector across the six rows, and uses the
portable scalar kernel only for its final sub-vector tail. The SVE kernel uses
four rows by two scalable vectors and predicated loads/stores for every tail,
so its lane count follows ``svcntw()``/``svcntd()`` rather than a build-time
width.

``DetectArmSimdLevel()`` reads Linux ``AT_HWCAP``/``AT_HWCAP2`` for Advanced
SIMD, SVE, and SVE2. On other AArch64 systems, including Apple Silicon, NEON is
part of the architecture baseline. SVE and SVE2 share the same floating-point
kernel because SVE2 does not replace the SVE FP FMA instructions used here.
The SVE translation unit is compiled separately with ``-march=armv8-a+sve``;
the baseline library never executes it unless runtime detection succeeds.

SVE vector length is part of the runtime profile. Widths of 256 bits and above
select the predicated SVE kernel. A 128-bit SVE implementation retains the
six-row NEON kernel, which has the same lane width and more B reuse. If a
toolchain cannot compile SVE, the same binary keeps its NEON fallback.

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Platform
     - Status
     - Notes
   * - macOS, Apple Silicon (M1/M2/M3/M4, ARM64/AArch64)
     - NEON FP32/FP64
     - Uses the architecture-baseline NEON kernel; current Apple processors do
       not expose SVE.
   * - Linux ARM64/AArch64 (including Grace and Neoverse)
     - NEON plus runtime SVE/SVE2 dispatch
     - Linux HWCAP gates SVE instructions. Runtime vector lengths below 256
       bits and builds without SVE support fall back to NEON.
   * - Other architectures
     - Portable scalar fallback
     - ``GemmMicroKernel_ScalarImpl`` remains available when neither an x86 nor
       ARM64 vector profile is usable.

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
   * - M/N task parallelism
     - Implemented -- the general five-loop path schedules the row-panel x
       column-panel grid in bounded column waves. Each B panel is packed once
       per K chunk, while task-aware MC/NC constraints expose enough row and
       column work without an unbounded packed workspace.
   * - Batch scheduling
     - Implemented -- ``MatMulPlan``, ``StridedBatchedGemm``, and
       ``GroupedGemm`` parallelize collections of small products after
       validating every input. A product with useful internal parallelism runs
       its batches serially.
   * - Split-K
     - Implemented -- used for long-K products whose M/N grid is insufficient.
       Partitions use the same packed SIMD micro-kernels as the general path,
       then a parallel tolerance-preserving reduction. Active batch regions
       take priority over nested split-K.
   * - A-panel packing (``PackARowBlock``)
     - Implemented -- resolves ``trans_a`` once per (row block, k-chunk).
   * - B-panel packing (``PackBPanel``)
     - Implemented -- each bounded-wave panel is packed once per K chunk and
       reused across every A row panel.
   * - NR=2 wide micro-kernel tiles
     - Implemented -- AVX2/SSE2/AVX-512, NEON, and SVE, all FP32/FP64
       element types.
   * - ARM64 NEON micro-kernels
     - Implemented -- FP32/FP64 MR=1 through 6 variants use two 128-bit
       vectors, FMA, scalar sub-vector tails, and the shared packed panels.
   * - ARM64 SVE/SVE2 micro-kernels
     - Implemented -- FP32/FP64 MR=1 through 4 variants use two scalable
       vectors and predicated tails. Linux HWCAP and runtime vector length
       select SVE at 256 bits or wider, with a NEON fallback otherwise.
   * - AVX-512 micro-kernel
     - Implemented -- separate TU, compile+runtime gated (see tree above).
   * - Software prefetch (``B`` rows, T0 hint)
     - Implemented -- AVX2/AVX-512 NR=2 loops, ``kGemmPrefetchDistanceK = 4``.
   * - Dedicated AVX2+FMA micro-kernels
     - Implemented -- FP32/FP64 kernels are compiled with ``-mavx2 -mfma`` in
       a separate translation unit and selected only when CPUID reports FMA.
   * - AVX-512 K-loop unrolling
     - Implemented -- FP32/FP64 NR=1 and NR=2 loops reduce four K rows per
       iteration, with a remainder loop and no additional accumulators.
   * - AVX2+FMA K-loop unrolling
     - Implemented -- FP32/FP64 NR=1 and NR=2 loops reduce four K rows per
       iteration, with a remainder loop and no additional accumulators.
   * - AVX2+FMA MR variants
     - Implemented -- compile-time FP32/FP64 MR=1 through 6 variants cover
       both NR=1 and NR=2, selected from the actual row-block size.
   * - AVX-512 MR variants
     - Implemented -- compile-time FP32/FP64 MR=1 through 8 variants cover
       both NR=1 and NR=2.
   * - ISA-specific register blocking
     - Implemented -- candidate AVX2 MR=4/5/6 and AVX-512 MR=6/7/8 kernels
       are compiled. CPUID family/model dispatch uses the measured MR=5
       profile on modern Intel Core AVX2 processors, the measured MR=6 profile
       on AMD Zen AVX2 processors (whose two FMA pipelines need the wider
       six-row tile), and the conservative MR=4 AVX2 / MR=6 AVX-512 profiles
       elsewhere. The selected value drives algorithm selection, MC alignment,
       row packing, and execution.
   * - Aligned packed panels
     - Implemented -- A and B workspaces are 64-byte aligned. B row strides
       are padded to the active vector width, keeping AVX2 and AVX-512 loads
       on aligned addresses. The kernels use branch-free unaligned-load
       instructions, which have the same throughput on these aligned
       addresses and also keep direct kernels safe.
   * - Measured software prefetch
     - Implemented -- disabling the four-row lookahead increased local
       single-thread square-matrix medians by 1.5x to 2.1x, so the measured
       four-row T0 lookahead is retained.
   * - Generated instruction ordering
     - Implemented -- generated AVX2 assembly was inspected for every candidate.
       Removing the explicit four-row K unroll to reduce compiler-generated
       stack traffic made the measured kernels 2x to 3x slower, so the unrolled
       intrinsic ordering is retained rather than replaced by unproven
       handwritten assembly.
   * - Unit-scale and no-bias epilogues
     - Implemented -- scalar, SSE2, AVX, AVX2+FMA, and AVX-512 kernels skip
       redundant ``alpha``/``beta`` multiplies when the scale is one; zero
       ``beta`` and absent bias avoid reading ``C``.
   * - Typed broadcast and fused epilogues
     - Implemented -- scalar, row, column, and matrix bias layouts are consumed
       without an expanded M x N temporary. The typed epilogue can combine
       bias, broadcast residual, ReLU, and FP16/BF16 narrowing in one output
       pass; the matrix-bias-only case remains fused directly in the SIMD
       micro-kernel.

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
   * - Explicit SSE2 k-loop unrolling
     - Manually unroll the inner ``k`` reduction in the NR=2 SSE2 kernels.
       AVX2+FMA and AVX-512 are already unrolled by four.
     - Small on CPUs without AVX/FMA; zero on the priority AVX2/AVX-512 fleet.
     - Additional complexity in a fallback path, with limited performance
       relevance and a remainder loop required for non-multiple K values.
   * - Compile-time ``GemmAccumMode`` specialization
     - Dispatch once per micro-kernel call to ``kInitZero``, ``kInitBias``, or
       ``kAccumulate`` template variants so ``if constexpr`` removes the
       mode tests from each output-vector finalization. The current tests are
       outside the K/FMA loop and the mode is stable for the whole call, so
       branch prediction makes this optional rather than a correctness or
       parity prerequisite.
     - Small for large K; potentially measurable for tiny and small-K direct
       kernels where output finalization is a larger fraction of total work.
     - Multiplies x86, NEON, and SVE kernel variants and code size. Keep only
       if a pinned small-K benchmark demonstrates a repeatable gain.
   * - Hand-written assembly micro-kernels
     - Replace selected intrinsic kernels only if a pinned parity benchmark
       demonstrates a persistent arithmetic-kernel gap after thread-runtime
       work. The current x86 tuning pass found no priority regression versus
       its ``main`` baseline that justified an assembly variant.
     - Potentially large, but only after isolating a compiler-code-generation
       bottleneck from packing and scheduling costs.
     - High engineering and maintenance cost, with a separate implementation
       required for each microarchitecture.
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

The next x86 tuning step is to extend measured profiles beyond the currently
available modern Intel Core AVX2 measurements as benchmark hosts become
available, before adding additional family/model
dispatch.
