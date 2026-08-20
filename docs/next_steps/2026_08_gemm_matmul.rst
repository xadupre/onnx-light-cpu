Gemm and MatMul Performance Roadmap
===================================

:Date: 2026-08

**in progress**

Objective
---------

The objective is performance parity with the ONNX Runtime CPU execution
provider for the important ``Gemm`` and ``MatMul`` workloads, for every
supported data type, without sacrificing ONNX correctness.
For GEMM, parity means a corpus median speed-up of at least ``1.0x`` versus
ONNX Runtime and no priority shape below ``0.9x``. This is a catch-up effort
with standard ONNX tensors and semantics.

The current implementation is a correctness-first, register-blocked kernel
with AVX2/AVX-512 paths, K blocking, A/B packing, a task-aware M x N scheduler,
batch scheduling, packed SIMD split-K, and typed broadcast/fused epilogues; see
:doc:`the current design <../design/gemm_kernel_design>`. Roadmap PR01 closed
the scheduler under-utilization identified on multi-panel shapes, and Roadmap
PR02 removed expanded bias temporaries. The FP32 investigation in
`onnx-light-cpu #162
<https://github.com/xadupre/onnx-light-cpu/pull/162>`_ shows that scalar
skinny-N, weak GEMV/skinny-M, an operator path that bypasses ``GemmPlan``, and
untuned Zen/generic-x86 blocking still prevent parity. The remaining roadmap
work first closes those measured FP32/FP64 gaps, then covers low-precision
kernels and the final ONNX Runtime parity gates.

Related roadmap
---------------

Persistent state, decode, paged storage, and cache quantization are covered by
the separate :doc:`Persistent KV Cache and Decode roadmap <2026_08_kv_cache>`.
Tensor Attention and its streaming implementation are covered by the separate
:doc:`Attention Performance Roadmap <2026_08_attention>`.

Scope and type matrix
---------------------

``Gemm`` and ``MatMul`` should share one matrix-multiplication engine.
The operator adapters retain distinct ONNX semantics:

* ``Gemm`` handles rank-2 inputs, ``alpha``, ``beta``, optional broadcast bias,
  and ``transA``/``transB``.
* ``MatMul`` handles vectors, matrices, arbitrary leading batch dimensions,
  NumPy-style batch broadcasting, and output-rank squeezing.

The implementation must follow the type constraints of the selected ONNX
opset. Integer and quantized multiplication may be exposed through ``MatMul``,
``MatMulInteger`` or ``QLinearMatMul`` rather than forcing unsupported types
through ``Gemm``.

.. list-table::
   :header-rows: 1
   :widths: 16 22 34 28

   * - Type
     - Accumulation
     - Preferred kernel
     - Fallback
   * - ``float32``
     - ``float32``
     - AVX2+FMA, AVX-512F, NEON/SVE
     - Portable blocked scalar kernel
   * - ``float64``
     - ``float64``
     - AVX2+FMA, AVX-512F, NEON/SVE
     - Portable blocked scalar kernel
   * - ``float16``
     - Normally ``float32``
     - AVX-512FP16 or convert-and-FMA during packing
     - F16C/NEON conversion into packed ``float32`` panels
   * - ``bfloat16``
     - Normally ``float32``
     - AVX-512BF16, AMX-BF16, or ARM BF16
     - Convert during packing into ``float32`` panels
   * - ``float8``
     - ``float16`` or ``float32``
     - Hardware-specific tensor/dot-product path
     - Vectorized decode during packing
   * - ``int8``/``uint8``
     - ``int32``
     - AVX-VNNI/AVX-512VNNI, AMX-INT8, NEON dot product
     - Widening integer micro-kernel
   * - ``int32``/``int64``
     - Schema-defined integer result
     - Vectorized integer multiply/add where profitable
     - Portable exact-arithmetic path
   * - Packed ``int4``/``uint4``
     - ``int32``
     - Unpack-and-dot kernel or AMX/vendor extension
     - Vectorized unpack into an ``int8`` packed panel

Benchmark contract
------------------

Optimization must begin with a reproducible benchmark. End-to-end runtime
measurements and isolated kernel measurements answer different questions and
must both be retained.

* Use identical tensors, transposition flags, thread counts, CPU affinity, and
  correctness tolerances for MLAS and ``onnx-light-cpu``.
* Warm up every candidate, alternate candidate order, and report median and
  dispersion rather than the best observation.
* Run on an otherwise idle, pinned machine with a fixed power policy. Record
  CPU model, cache sizes, ISA features, compiler, and build flags.
* Measure packing, the blocked multiplication, and low-precision conversion
  separately. The end-to-end number must still include every cost visible to a
  caller.
* Cover tiny matrices, square matrices, skinny M, skinny N, large K, batched
  MatMul, broadcast batches, every transpose combination, and transformer
  projection shapes.
* Separate dynamic-B from constant-B cases. Constant weights must be packed
  once, not once per invocation.
* Compare single-thread throughput and scaling at 2, 4, physical-core, and
  logical-core thread counts. Hybrid P/E-core machines need their own results.

Four committed instruments implement this contract. ``tools/benchmark_gemm_parity.py``
is the end-to-end floating-point parity runner (PR06.0/PR10.3): it alternates
the registered operator against ONNX Runtime and reports GFLOP/s and speed-up
per priority shape. ``tools/gemm_throughput.cc`` is its isolated ``GemmPlan``
counterpart, built with ``-DONNX_LIGHT_CPU_BUILD_BENCHMARKS=ON``.
``tools/benchmark_integer_gemm_parity.py`` applies the same alternating,
raw-sample contract to UINT8 x INT8 ``MatMulInteger``. The opt-in
``tools/compact_gemm_throughput.cc`` driver publishes isolated INT8, packed
INT4, E4M3, and E5M2 throughput for the same tiny, direct, square, skinny,
large-K, and transformer shape families. Together the end-to-end and isolated
numbers distinguish operator overhead from packing and micro-kernel limits.

Target computation algorithm
----------------------------

Choosing an ISA-specific function once instead of once per call is useful, but
it cannot explain or close a 10x throughput gap. The central change must be the
matrix-multiplication algorithm: the order in which panels move through the
cache hierarchy and are reused by the arithmetic micro-kernel.

The general dense path should follow the five-loop GotoBLAS/BLIS decomposition.
For ``C = A @ B``, with ``MC``, ``NC`` and ``KC`` sized for the cache hierarchy
and ``MR x NR`` sized for the vector registers:

.. code-block:: text

   for jc in range(0, N, NC):          # L3-sized columns of C and B
     for pc in range(0, K, KC):        # reduction panel
       Bc = pack(B[pc:pc+KC, jc:jc+NC])       # packed once
       for ic in range(0, M, MC):      # L2-sized rows of A and C
         Ac = pack(A[ic:ic+MC, pc:pc+KC])     # reused across all NR panels
         for jr in range(0, NC, NR):
           for ir in range(0, MC, MR):
             microkernel(Ac[ir:], Bc[:, jr:], C[ic+ir:, jc+jr:])

This order is important:

* one packed B ``KC x NC`` panel is reused by every ``MC`` row block;
* one packed A ``MC x KC`` panel is reused by every ``NR`` column micro-panel;
* the ``MR x NR`` C tile remains in registers for the complete ``KC``
  reduction;
* the working sets deliberately move from L3 (``NC``) to L2 (``MC/KC``), then
  L1 and registers, instead of relying on one fixed 64 x 256 tile;
* transposition is resolved while packing, so the arithmetic loop sees only
  contiguous canonical panels.

The current five-loop engine packs one B panel for a column/K block and shares
it across its row panels. It then parallelizes either column panels or row
panels, not their Cartesian product. Its task granularity is therefore tied to
MC/NC: a large cache-derived NC can leave only one column task, while a large
MC can leave only a few row tasks.

One algorithm is not optimal for every shape. The plan must choose among
distinct computational algorithms:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Shape
     - Algorithm
   * - General M x N x K
     - Five-loop packed GEMM with hierarchical MC/NC/KC blocking.
   * - Tiny matrices
     - Direct, un-packed micro-kernel; packing costs more than the arithmetic.
   * - ``M == 1`` or very small M
     - GEMV/skinny-M kernel that streams B once and vectorizes across N.
   * - ``N == 1`` or very small N
     - Dot/GEMV kernel that vectorizes the K reduction and avoids B-panel
       packing.
   * - Small K
     - Outer-product or direct kernel with wide N tiles and no KC loop.
   * - Batched MatMul
     - Batch outer loop for small independent products; merge batch with M/N
       task dimensions when one product cannot occupy all cores.
   * - Constant B
     - Plan-owned B prepacked once when shape and layout are stable, with the
       original tensor representation retained only when required by a
       guarded fallback.
   * - Extremely large K with small M/N
     - Split K only when M/N/batch parallelism is insufficient, then combine
       partial accumulators in a controlled reduction.

Planning once, computing many times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kernel selection belongs in session preparation. A ``GemmPlan``/``MatMulPlan``
should be created after shapes, types, constant inputs, CPU features, and thread
limits are known. It should contain:

* the computational algorithm (general, direct, GEMV, batched, or split-K);
* MC/NC/KC and MR/NR;
* typed function pointers for packing, micro-kernel, and epilogue;
* the parallel decomposition and useful thread count;
* plan-owned constant B storage when B is an initializer.

The execution path then invokes the plan directly. This removes repeated
selection, but its main benefit is enabling the correct algorithm and data
layout to be prepared ahead of time; the branch removal itself is negligible
relative to the multiplication.

Phase 1: implement the blocked float32 algorithm
------------------------------------------------

``float32`` is the first parity target because it exposes the quality of the
core algorithm without conversion overhead.

1. Implement the five-loop algorithm as a separate, testable engine.
2. Correct the loop ownership so each B panel is packed once and shared across
   its row blocks.
3. Add direct, GEMV, skinny-M, skinny-N, and small-K algorithms.
4. Derive MC/NC/KC from measured L1/L2/L3 capacity and associativity, with
   conservative defaults when cache discovery is unavailable. This is
   implemented for x86 deterministic CPUID cache descriptors; other platforms
   currently use the conservative defaults.
5. Build the immutable execution plan once and benchmark each algorithm both
   single-threaded and multi-threaded. The benchmark corpus includes a
   shape-forced case for every algorithm and selects participants through the
   ``onnx-light`` session execution policy.

Phase 2: saturate the floating-point units
------------------------------------------

* Compile FMA micro-kernels in dedicated translation units and use them only
  after checking both AVX/AVX2 and FMA. The current default ``-mavx2`` build
  does not define ``__FMA__``. Dedicated FP32/FP64 AVX2+FMA micro-kernels are
  now compiled with ``-mavx2 -mfma`` and selected only when CPUID reports FMA;
  the baseline AVX path remains available on CPUs without FMA.
* Keep distinct SSE2, AVX2+FMA, AVX-512F, and microarchitecture-specific
  variants. Zen, Skylake, Ice Lake, and hybrid Intel CPUs can require different
  MR/NR and cache blocks even when they expose the same ISA.
* Generate several MR x NR micro-kernels instead of fixing ``MR == 4``.
  AVX2+FMA emits compile-time ``MR=1..6`` variants and AVX-512 emits
  ``MR=1..8``, both for NR=1 and NR=2. The detected microarchitecture selects
  MR=4 for generic AVX2/SSE, MR=5 for modern Intel Core AVX2, MR=6 for AMD Zen
  AVX2, and MR=6 for AVX-512, and the choice is propagated through cache
  blocking, packing, algorithm selection, and execution. Per-model tuning
  within an ISA remains.
* Unroll K enough to maintain independent FMA chains without spilling
  accumulators. The AVX2+FMA and AVX-512 FP32/FP64 kernels now reduce four K
  rows per loop iteration and use a scalar-count remainder loop without adding
  accumulator registers.
* Use aligned panel loads and software prefetch only where hardware-counter
  measurements show reduced stalls.
* Specialize the arithmetic epilogue for ``alpha == 1``, ``beta == 0``, scalar
  bias, row/column bias, and no bias. Unit ``alpha``/``beta``, zero ``beta``,
  and no-bias cases now avoid redundant vector/scalar multiplication and bias
  reads in every x86 micro-kernel and scalar tail; scalar and row/column
  broadcast bias interfaces remain.
* ARM64 NEON FP32/FP64 kernels are implemented with six-row, two-vector
  register tiles and scalar sub-vector tails. SVE/SVE2 use four-row,
  two-scalable-vector tiles and predicated tails; runtime vector lengths below
  256 bits deliberately retain NEON. Linux HWCAP detection and separate SVE
  compilation keep unsupported processors on the safe fallback.

Parallel execution
~~~~~~~~~~~~~~~~~~

Roadmap PR01 is implemented by `onnx-light-cpu #155
<https://github.com/xadupre/onnx-light-cpu/pull/155>`_ and Roadmap PR02 by
`onnx-light-cpu #156
<https://github.com/xadupre/onnx-light-cpu/pull/156>`_, and Roadmap PR03 by
`onnx-light-cpu #157
<https://github.com/xadupre/onnx-light-cpu/pull/157>`_, and Roadmap PR04 by
`onnx-light-cpu #158
<https://github.com/xadupre/onnx-light-cpu/pull/158>`_, and Roadmap PR05 by
`onnx-light-cpu #159
<https://github.com/xadupre/onnx-light-cpu/pull/159>`_. Roadmap PR06.0
implemented the parity runner in `onnx-light-cpu #160
<https://github.com/xadupre/onnx-light-cpu/pull/160>`_, but its measured gate
does not pass. Roadmap PR06.1 diagnoses the gap in `onnx-light-cpu #162
<https://github.com/xadupre/onnx-light-cpu/pull/162>`_. The remaining P4
implementation work is Roadmap PR06.2 through PR06.6 in the final table; the
blocking dedicated-machine parity gate is deliberately last as PR10.5.
Constant-B prepacking is now included in PR06.4 because #162 identifies its
absence from the operator hot path as part of the measured gap. No performance
work demonstrated necessary by the parity corpus may be deferred while the
gate remains unmet.

PR06 does not restart P4 or invalidate PR01 through PR05: their correctness,
dispatch, scheduling, and architecture tests remain required foundations.
The measured fixes proceed in this order:

#. Vectorize the skinny-N K reduction, including ``N == 1``, and keep split-K
   disabled when its partition and reduction costs dominate the tiny output.
#. Add a dedicated GEMV/skinny-M path that streams each B row once and reuses
   it across output columns.
#. Route ONNX ``Gemm`` and ``MatMul`` through immutable ``GemmPlan`` and
   ``MatMulPlan`` instances so shape selection and plan-owned constant-B state
   are prepared once, including persistent packed-B panels for initializer
   weights.
#. Tune Zen and generic-x86 MR/NR and cache blocking against the complete
   corpus so 1024³ and larger shapes sustain, rather than lose, throughput.
#. Rerun the complete FP32/FP64 Gemm, MatMul, and batched corpus on dedicated,
   frequency-stabilized machines. Publish raw results only when both dtype
   medians reach ``1.0x`` and every priority case reaches ``0.9x``.

The scheduler decomposes ``Y = A @ B`` into a Cartesian grid of row and column
panels:

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

Task ``T(i,j)`` multiplies row panel ``Ai`` by column panel ``Bj`` and writes
only the corresponding, disjoint zone of ``Y``. Column panels are processed in
bounded waves large enough to occupy the available threads. For example, with
six threads, three row panels, and three column panels:

.. code-block:: text

   wave 1: B0, B1 -> T00 T10 T20 T01 T11 T21
   wave 2: B2     -> T02 T12 T22

For each K chunk, every B panel in the active wave is packed once and shared by
all its row-panel tasks. The tasks accumulate into their zones of ``Y`` before
the scheduler advances to the next K chunk. If the complete M x N task grid
still cannot occupy the pool, split-K partitions the reduction dimension:

.. code-block:: text

   K = [K0 | K1 | K2]
         |    |    |
         v    v    v
        P0   P1   P2  ->  Y = alpha * (P0 + P1 + P2) + beta * C

Each partial ``Pi`` uses the same packed SIMD micro-kernels. Independent
batches take priority over split-K: when a GEMM already runs inside a parallel
batch region, it executes its M x N grid directly instead of creating nested
K partitions.

Phase 3: native low-precision kernels
--------------------------------------

The existing FP16/BF16 path widens complete tensors to ``float32``, calls
``GemmFloat32``, then narrows the complete result. It is correct but performs
extra full-matrix memory passes.

* For AVX2/F16C, load FP16 panels, convert vectors to FP32 while packing, and
  accumulate with FMA. Narrow only the final output. *(Landed: every FP16/BF16
  execution path -- the general algorithm plus the skinny-M, skinny-N, direct
  small-K, and split-K paths -- converts each element to float32 during packing
  or reduction, accumulates in float32, and narrows only in the epilogue with no
  full-tensor widening. ``GemmHalfPlan`` caches the selected algorithm and
  blocking. Vectorized F16C and AVX2 conversion for contiguous panels is isolated
  in Roadmap PR07.1; removing the remaining widening paths is Roadmap PR07.2.)*
* For AVX-512BF16 and AVX-512FP16, add native dot-product/multiply-accumulate
  kernels with FP32 accumulation where required by the ONNX numerical contract.
  *(First landed: Roadmap PR07.3 adds a native AVX-512FP16 general kernel that
  keeps both operands in FLOAT16 to the register file, widens each 16-lane
  ``B`` vector with ``vcvtph2psx``, and accumulates in float32. It is dispatched
  by ``CpuSupportsAvx512Fp16()`` for non-transposed ``B`` and otherwise keeps
  the converting float32 path. Roadmap PR07.4 adds the sibling native
  AVX-512BF16 general kernel that reduces pairs of ``k`` iterations with the
  ``vdpbf16ps`` dot-product, accumulating in float32; it is dispatched by
  ``CpuSupportsAvx512Bf16()`` for non-transposed ``B`` and otherwise keeps the
  converting float32 path.)*
* Add AMX tile kernels behind OS-enabled tile-state detection. AMX must remain
  optional because enabling the ISA and configuring tiles have non-trivial
  per-thread costs. *(First landed: Roadmap PR07.5 adds the AMX tile-state
  lifecycle -- ``CpuSupportsAmxTile``/``AmxBf16``/``AmxInt8`` detection, the
  one-time Linux ``XTILEDATA`` permission request behind
  ``AmxTileStateAvailable``, the validating ``AmxTileConfig`` builder, and the
  per-worker ``AmxTileScope`` (``LDTILECFG``/``TILERELEASE``) with a safe
  no-op fallback -- with no GEMM kernel yet; the AMX-BF16 kernel is PR07.6 and
  AMX-INT8 is PR09.4. Roadmap PR07.6 then adds the native AMX-BF16 GEMM kernel:
  a ``tdpbf16ps`` (``_tile_dpbf16ps``) tile micro-kernel that reuses the PR07.5
  lifecycle, keeps both operands in BFLOAT16 with a VNNI-packed ``B`` tile, and
  is dispatched ahead of AVX-512BF16 for non-transposed ``B`` when
  ``CpuSupportsAmxBf16()`` and ``AmxTileStateAvailable()`` both report the ISA;
  it falls back to AVX-512BF16 or the converting float32 path otherwise.)*
* Implement equivalent ARM FP16/BF16 and dot-product paths. *(First landed:
  Roadmap PR08.1 vectorizes the FP16/BF16 convert-while-packing panels on ARM
  with NEON -- a ``vmovl_u16`` zero-extend plus 16-bit shift for BFLOAT16 and the
  ``vcvt_f32_f16`` (``FCVTL``) instruction for FLOAT16, both with an exact scalar
  tail matching the bit decode and a scalar fallback when the FP16 intrinsics are
  unavailable. Roadmap PR08.2 then adds the native NEON arithmetic kernels
  (``GemmMicroKernel_NEON_BF16`` always, ``GemmMicroKernel_NEON_FP16`` when the
  FP16 intrinsics compile): both keep the operands half-precision to the register
  file, widen each ``B`` vector on the fly (zero-extend/shift for BFLOAT16,
  ``FCVTL`` for FLOAT16) and accumulate in float32, and are dispatched from
  ``GemmHalfPlanned<kGeneral>`` for non-transposed ``B`` ahead of the converting
  float32 path, which stays the fallback. Roadmap PR08.3 then adds the native SVE
  arithmetic kernels (``GemmMicroKernel_SVE_BF16`` / ``GemmMicroKernel_SVE_FP16``):
  they reuse the same drivers, keep the operands half-precision to the register
  file, widen each ``B`` vector on the fly (``svld1uh_u32`` zero-extend/shift for
  BFLOAT16, the SVE ``FCVT`` ``svcvt_f32_f16`` for FLOAT16), accumulate in
  float32, drive the lane count from the runtime vector length and cover the
  column remainder with an ``svwhilelt`` predicated tail, and are dispatched ahead
  of NEON when the runtime profile selects SVE (a vector length of at least 256
  bits); shorter vectors keep the better-unrolled NEON kernel.)*
* For INT8, fuse zero-point correction and requantization into packing and the
  epilogue. Accumulate in INT32 and define overflow behavior through the ONNX
  operator contract. *(First landed: Roadmap PR09.2 adds the x86 VNNI INT8 kernel
  (native ``vpdpbusd`` path with a portable scalar sibling) behind the shared
  ``IntegerMatMul2D`` driver for the contiguous rank >= 2 ``MatMulInteger`` case.
  Roadmap PR09.3 then adds the native ARM NEON dot-product INT8 kernel
  ``GemmMatMulIntegerNeonDotProd``, dispatched from that same ``IntegerMatMul2D``
  entry point. A single unsigned ``UDOT`` reduction serves every signedness
  combination by folding a signed operand's ``+128`` bias into its effective zero
  point and recovering the raw products with per-row / per-column byte-sum
  corrections, so the INT32 accumulation matches the portable scalar fallback bit
  for bit modulo 2^32; it is gated on the ``+dotprod`` build flag and the runtime
  ``CpuSupportsNeonDotProd`` capability, keeping the scalar reduction as the
  fallback.)*
* Treat Float8 and packed 4-bit types as separate packing formats, not as
  branches in the FP32 inner loop. *(First landed: Roadmap PR09.5 adds the four
  ONNX Float8 formats (``E4M3FN``, ``E4M3FNUZ``, ``E5M2``, ``E5M2FNUZ``) as
  separate packing formats: exact per-format decoders decode each one-byte
  pattern to float32 while packing -- the contiguous copies gather from an exact
  256-entry per-format table through an AVX2 ``vgatherdps`` helper with a scalar
  tail and fallback -- reusing the tuned FP32 algorithms with float32
  accumulation. Packed 4-bit types are Roadmap PR09.6.)*

Phase 4: complete MatMul
------------------------

The MatMul adapter should lower every ONNX shape case into a sequence of core
GEMM calls without materializing broadcast copies:

1. Normalize rank-1 inputs to temporary logical dimensions.
2. Compute broadcasted batch strides, using zero strides for broadcast axes.
3. Collapse contiguous batch dimensions where possible.
4. Dispatch independent batches through the same scheduler used by GEMM.
5. Select dedicated GEMV and dot-product kernels for ``M == 1`` or ``N == 1``.
6. Restore the exact ONNX output rank without copying data.

The adapter and engine require tests for empty dimensions, scalar-like vectors,
non-contiguous batch strides, asymmetric broadcasting, transposed packed
weights, and every supported type.

How to exceed ONNX Runtime
--------------------------

Matching MLAS with a generic dynamic GEMM is difficult. Beating it is more
realistic when ``onnx-light`` exploits model-level information. The estimates
below are relative to a tuned ONNX Runtime/MLAS run with the same thread count,
not relative to the current ``onnx-light-cpu`` implementation. They are targets
to verify, not guarantees.

.. list-table::
   :header-rows: 1
   :widths: 20 22 36 22

   * - Optimization
     - Expected gain over MLAS
     - Conditions and quantitative bound
     - Estimated effort
   * - Full shape specialization
     - **2-10%** normally; **10-20%** on stable skinny or tail-heavy shapes.
     - Instantiate loop order, MC/NC/KC, MR/NR, packing format, micro-kernel,
       and thread decomposition for one exact shape. Removing the dispatch
       branch alone is expected to save **less than 1%**; the gain comes from a
       better algorithm and eliminating generic tail work.
     - 5-10 days for a bounded shape family.
   * - Fused epilogues
     - **5-25%** for a compute-heavy GEMM; **1.2-1.8x** for small or
       bandwidth-bound GEMM chains.
     - Fusing each following FP32 elementwise operator avoids approximately
       **8 x M x N bytes** of traffic (one read and one write). The upper range
       requires an epilogue not already fused by ONNX Runtime, such as a
       project-specific bias + residual + activation + narrowing combination.
     - 3-7 days per epilogue family.
   * - Batch fusion
     - **1.1-1.5x** for ordinary batches of small matrices; **up to 2-3x** for
       hundreds of tiny or irregular products.
     - Useful when one product takes only a few microseconds and dispatch or
       thread synchronization is a significant fraction of its time. The gain
       tends to **0%** once each individual GEMM already saturates the cores.
     - 5-10 days.
   * - Sparse weights
     - **1.3-2x** around 70-80% sparsity; **2-4x** around 90% structured
       sparsity.
     - With nonzero density ``d``, the arithmetic upper bound is ``1 / d``.
       A realistic kernel commonly reaches **30-70% of that ideal** after index,
       packing, and load imbalance costs. Unstructured sparsity below roughly
       70% is unlikely to beat dense MLAS; block or N:M structure lowers that
       threshold.
     - 10-20 days per sparse format.
   * - Low-rank weights
     - **1.5-5x** when an exact or accepted approximation has rank
       ``r << min(K, N)``.
     - Replacing an ``M x K`` by ``K x N`` product with two factors costs
       roughly ``2 M r (K + N)`` operations instead of ``2 M N K``. The
       theoretical speed-up is therefore ``K N / (r (K + N))``, capped by the
       extra intermediate and model-accuracy constraints.
     - 5-10 days for exact factors; model work is additional for approximation.
   * - Model-specific autotuning
     - **3-15%** normally; **up to 20-30%** across heterogeneous CPUs or
       unusual shapes.
     - Benchmark 5-20 safe candidates during session preparation and cache the
       winner by CPU, type, shape, transpose flags, and thread count. Limit
       tuning to roughly **1-50 ms per unique shape** or load a persistent
       tuning cache; never tune in the inference path.
     - 5-10 days plus dedicated benchmark infrastructure.

These gains are not additive. Shape specialization and autotuning often choose
the same improvement. A credible target sequence is:

1. reach at least **1.0x ONNX Runtime** median performance with the generic
   blocked algorithm and tuned scheduler;
2. reach **1.05-1.15x ONNX Runtime** through shape specialization and tuning;
3. target **1.2-1.8x** on fused or tiny-batch workloads;
4. reserve gains above **2x** for workloads with exploitable sparsity,
   low-rank structure, or very large collections of tiny matrices.

External libraries
------------------

OpenBLAS, BLIS, oneDNN, and vendor BLAS libraries are useful as performance
oracles and optional large-matrix fallbacks. They do not remove the need for
internal kernels: small inference shapes, FP16/BF16/quantized types, constant
weights, and fused epilogues are precisely where a model-aware runtime can win.
Any optional dependency must have a deterministic internal fallback and must
not create a second competing thread pool.

Acceptance criteria
-------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Area
     - Exit criterion
   * - Correctness
     - ONNX backend and differential tests pass for every type, shape,
       transpose, broadcast, alpha/beta, bias, empty-dimension, and tail case.
   * - FP32/FP64 parity
     - Median speed-up is at least 1.0x versus ONNX Runtime on the priority
       shape corpus, with no priority shape below 0.9x.
   * - Low precision parity
     - FP16, BF16, and INT8 meet the same target on hardware with native
       support; fallback paths remain correct and avoid full-matrix conversion
       where panel conversion is possible.
   * - Scaling
     - Throughput improves through the physical-core count without severe
       regressions on tiny or skinny shapes.
   * - Data movement
     - Every dynamic A panel and constant B panel is packed no more often than
       required by the selected loop nest; low-precision paths avoid
       full-matrix conversion where panel conversion is possible.
   * - Exceeding MLAS
     - Constant-weight or fused workloads demonstrate at least a repeatable
       1.10x improvement on dedicated benchmark machines.

Performance gates should run on dedicated, pinned hardware and store the raw
samples and environment metadata. Shared CI machines can enforce correctness
and detect catastrophic slowdowns, but they should not decide a 5-10%
performance regression.

Implementation order and dependencies
-------------------------------------

This roadmap and its dependency ordering were consolidated in
`onnx-light-cpu #137
<https://github.com/xadupre/onnx-light-cpu/pull/137>`_. The status below
distinguishes implemented code from performance exit criteria that still
require measurements on dedicated hardware.

.. list-table::
   :header-rows: 1
   :widths: 7 21 30 14 13 15

   * - Step
     - Deliverable
     - Exit criterion
     - Dependency
     - Status
     - Pull requests
   * - P0
     - Reproducible MLAS cases in ``onnx-light``'s C++ ``TestMode::BENCHMARK``
       framework.
     - Stable medians and dispersion for the agreed shape/type corpus on pinned
       hardware.
     - None.
     - Corpus implemented; dedicated-hardware measurements pending.
     - `onnx-light-cpu #134
       <https://github.com/xadupre/onnx-light-cpu/pull/134>`_,
       `onnx-light #4412
       <https://github.com/xadupre/onnx-light/pull/4412>`_
   * - P1
     - ``GemmPlan``, ``MatMulPlan``, ``StridedBatchedGemm``, and
       ``GroupedGemm`` interfaces in
       ``onnx_light_cpu/impl/math/gemm/gemm_plan.h``.
     - Existing Gemm results remain correct with no material performance
       regression.
     - P0.
     - Implemented.
     - `onnx-light-cpu #135
       <https://github.com/xadupre/onnx-light-cpu/pull/135>`_
   * - P2
     - Complete MatMul shape/broadcast adapter.
     - Differential tests pass for rank-1, batched, broadcast, transpose, and
       empty-dimension cases.
     - P1.
     - Implemented.
     - `onnx-light-cpu #138
       <https://github.com/xadupre/onnx-light-cpu/pull/138>`_
   * - P3
     - Five-loop FP32/FP64 engine and shape-specific algorithms.
     - Generic dense path reaches at least 0.8x MLAS before assembly-level
       tuning.
     - P1.
     - Engine, algorithms, cache-derived blocking, and benchmark corpus
       implemented; 0.8x MLAS gate pending.
     - `onnx-light-cpu #136
       <https://github.com/xadupre/onnx-light-cpu/pull/136>`_,
       `onnx-light-cpu #139
       <https://github.com/xadupre/onnx-light-cpu/pull/139>`_,
       `onnx-light-cpu #140
       <https://github.com/xadupre/onnx-light-cpu/pull/140>`_
   * - P4
     - FMA/AVX2/AVX-512/ARM micro-kernels and tuned scheduler.
     - Priority FP32/FP64 corpus reaches at least 1.0x ONNX Runtime median
       performance with no priority shape below 0.9x.
     - P3.
     - Scheduler PR01, epilogue PR02, x86 tuning PR03, thread runtime PR04,
       ARM kernels PR05, parity runner PR06.0, diagnosis PR06.1, vectorized
       skinny-N selection PR06.2, the dedicated GEMV/skinny-M kernel PR06.3,
       immutable operator plans PR06.4, and the measured Zen/Intel register
       tiles PR06.5 and shared-runner diagnostics PR06.6 are implemented. The
       blocking dedicated-machine ONNX Runtime gate is deferred to PR10.5,
       after every GEMM implementation and type-specific gate.
     - `onnx-light-cpu #133
       <https://github.com/xadupre/onnx-light-cpu/pull/133>`_,
       `onnx-light-cpu #141
       <https://github.com/xadupre/onnx-light-cpu/pull/141>`_,
       `onnx-light-cpu #142
       <https://github.com/xadupre/onnx-light-cpu/pull/142>`_,
       `onnx-light-cpu #143
       <https://github.com/xadupre/onnx-light-cpu/pull/143>`_,
       `onnx-light-cpu #145
       <https://github.com/xadupre/onnx-light-cpu/pull/145>`_,
       `onnx-light-cpu #146
       <https://github.com/xadupre/onnx-light-cpu/pull/146>`_,
       `onnx-light-cpu #147
       <https://github.com/xadupre/onnx-light-cpu/pull/147>`_,
       `onnx-light-cpu #149
       <https://github.com/xadupre/onnx-light-cpu/pull/149>`_,
       `onnx-light-cpu #155
       <https://github.com/xadupre/onnx-light-cpu/pull/155>`_,
       `onnx-light-cpu #156
       <https://github.com/xadupre/onnx-light-cpu/pull/156>`_,
       `onnx-light-cpu #157
       <https://github.com/xadupre/onnx-light-cpu/pull/157>`_,
       `onnx-light-cpu #158
       <https://github.com/xadupre/onnx-light-cpu/pull/158>`_,
       `onnx-light-cpu #159
       <https://github.com/xadupre/onnx-light-cpu/pull/159>`_,
       `onnx-light-cpu #160
       <https://github.com/xadupre/onnx-light-cpu/pull/160>`_,
       `onnx-light-cpu #162
       <https://github.com/xadupre/onnx-light-cpu/pull/162>`_,
       `onnx-light-cpu #167
       <https://github.com/xadupre/onnx-light-cpu/pull/167>`_,
       `onnx-light-cpu #176
       <https://github.com/xadupre/onnx-light-cpu/pull/176>`_
   * - P5
     - Native/panel-converted FP16, BF16, and integer paths.
     - Low-precision corpus reaches at least 1.0x ONNX Runtime median
       performance with no priority shape below 0.9x where the type is
       supported.
     - P3-P4.
     - PR07.0, PR07.1, PR07.2, PR07.3, PR07.4, PR07.5, PR07.6, PR08.1, PR08.2,
       and PR08.3 are implemented. The remaining work is split by execution path,
       ISA, and type below; hardware-specific lanes may proceed in parallel
       after their shared semantic dependency.
     - Roadmap PR07.0 through PR10.5 below.

Remaining pull-request sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table is the single source of truth for the sequence after
``#149``. Large phases use decimal sub-PRs so that each change has one type or
ISA, one measurable merge criterion, and no unrelated performance gate.
Hardware-specific lanes may run in parallel; only their shared semantics and
fallbacks are ordered. Completed rows remain visible so scope is not lost.

.. list-table::
   :header-rows: 1
   :widths: 9 29 42 12 8

   * - PR
     - Scope
     - Merge criterion
     - Depends on
     - Status
   * - Roadmap PR01
     - Scheduler, blocking, batch, and split-K.
     - The five-loop engine schedules the full row-panel x column-panel grid,
       packs each B panel once, consumes ``useful_threads()``, and constrains
       MC/NC to expose enough tasks. ``MatMulPlan``, ``StridedBatchedGemm``, and
       ``GroupedGemm`` schedule small products without nested pools. Split-K is
       used only when M/N/batch work is insufficient and uses packed SIMD
       kernels with a tolerance-preserving reduction.
     - ``#149``
     - `Implemented in #155
       <https://github.com/xadupre/onnx-light-cpu/pull/155>`_
   * - Roadmap PR02
     - Broadcast and fused epilogues.
     - None, scalar, row, column, and full-matrix C layouts are consumed
       directly for every alpha/beta case; the expanded M x N bias temporary
       disappears. Priority bias, residual, activation, and output-conversion
       combinations use typed epilogues without intermediate tensors.
     - PR01
     - `Implemented in #156
       <https://github.com/xadupre/onnx-light-cpu/pull/156>`_
   * - Roadmap PR03
     - Complete x86 kernel tuning.
     - AVX2 and AVX-512 candidate MR/NR profiles, aligned panels/loads,
       instruction ordering, and measured prefetch choices are benchmarked.
       CPUID family/model dispatch selects the winners; remaining gaps receive
       assembly kernels, with no priority-shape regression.
     - PR02
     - `Implemented in #157
       <https://github.com/xadupre/onnx-light-cpu/pull/157>`_
   * - Roadmap PR04
     - Complete thread runtime.
     - The scheduler detects physical cores, SMT siblings, P-cores, and E-cores
       and applies tested Linux/Windows affinity. Bounded spin-before-park is
       configurable, and caller-owned pools run without nested workers or
       oversubscription.
     - PR01
     - `Implemented in #158
       <https://github.com/xadupre/onnx-light-cpu/pull/158>`_
   * - Roadmap PR05
     - ARM FP32/FP64 kernels.
     - NEON packing, kernels, tails, and dispatch pass all GEMM/MatMul cases.
       Runtime vector-length-aware SVE/SVE2 profiles pass the ARM correctness
       and performance corpus with NEON fallback.
     - PR02
     - `Implemented in #159
       <https://github.com/xadupre/onnx-light-cpu/pull/159>`_
   * - Roadmap PR06.0
     - FP32/FP64 parity runner.
     - The reproducible runner records raw alternating samples, dispersion,
       CPU affinity, SIMD level, and effective thread count for every priority
       Gemm shape.
     - PR01 through PR05
     - `Implemented in #160
       <https://github.com/xadupre/onnx-light-cpu/pull/160>`_. An initial
       six-core diagnostic run on an i7-13800H under WSL reaches 0.317x FP32
       and 0.347x FP64 median, with 0.064x and 0.036x minima. These diagnostic
       numbers are not final dedicated-machine evidence.
   * - Roadmap PR06.1
     - Isolate and explain the FP32 performance gaps.
     - Isolated C++ driver measurements plus traced operator and plan paths
       identify the responsible algorithm, blocking, planning, and conversion
       costs before kernel changes are proposed.
     - PR06.0
     - `In progress in #162
       <https://github.com/xadupre/onnx-light-cpu/pull/162>`_. The analysis
       identifies scalar skinny-N, weak GEMV/skinny-M, unused operator plans,
       and Zen/generic-x86 blocking as the next measured priorities.
   * - Roadmap PR06.2
     - Vectorized skinny-N and tiny-output selection.
     - ``N == 1`` and small-N reductions vectorize over K with exact tails.
       The selector avoids split-K when partition and reduction overhead
       dominates, and every skinny-N priority case improves without regressing
       general GEMM.
     - PR06.1
     - `Implemented in #167
       <https://github.com/xadupre/onnx-light-cpu/pull/167>`_. ``GemmSkinnyN``
       now reduces each output through several partial accumulators over a
       unit-stride K body with an exact scalar tail, and
       ``SelectGemmAlgorithm`` keeps single-column outputs on the skinny-N
       path instead of split-K. Dedicated-hardware speed-up measurements are
       still pending.
   * - Roadmap PR06.3
     - Dedicated GEMV/skinny-M kernel.
     - ``M == 1`` and small-M cases stream each B row once, reuse it across
       output columns, vectorize the useful dimension, and improve every
       priority GEMV case without regressing general GEMM.
     - PR06.2
     - `Implemented in #170
       <https://github.com/xadupre/onnx-light-cpu/pull/170>`_. ``GemmSkinnyM``
       now streams each B row once per K, reuses it across the few output
       rows through a broadcast axpy that vectorizes over N for non-transposed
       B, and applies the alpha/beta epilogue from a small per-panel
       accumulator instead of packing an almost-empty A panel through the
       five-loop tile kernel. Dedicated-hardware speed-up measurements are
       still pending.
   * - Roadmap PR06.4
     - Use immutable plans on operator paths.
     - Registered ONNX ``Gemm`` and ``MatMul`` construct guarded
       ``GemmPlan``/``MatMulPlan`` instances during preparation and execute
       them directly. Algorithm, blocking, threads, and packed constant-B
       panels are no longer re-derived on every run; dynamic shapes rebuild or
       retrieve a correctly keyed plan, and dynamic B retains the ordinary
       per-call packing path.
     - PR06.3
     - `Implemented in #176
       <https://github.com/xadupre/onnx-light-cpu/pull/176>`_. The registered
       ``Gemm`` kernel now caches a keyed immutable ``GemmPlan`` per node for
       FP32/FP64 and executes it through a new epilogue-aware
       ``GemmPlan::Execute`` overload, so the algorithm, blocking, and thread
       count are prepared once and only rebuilt when the dtype, shape, or
       attributes change. Dynamic B keeps the ordinary per-call packing path.
       FP16/BF16 now cache a keyed immutable ``GemmHalfPlan`` and execute from
       typed inputs with FP32 accumulation (Roadmap PR07.2). ``MatMul`` has no
       registered operator yet, so its plan wiring arrives with that kernel.
   * - Roadmap PR06.5
     - Sustain large-matrix throughput on Zen and generic x86.
     - Measured MR/NR candidates and shape-constrained MC/NC/KC choices keep
       enough independent FMA chains and parallel panels active. The 1024³ and
       2048³ priority cases no longer regress from 512³, and the complete
       corpus shows no priority-shape regression.
     - PR06.4
     - `Implemented in #180
       <https://github.com/xadupre/onnx-light-cpu/pull/180>`_. AMD Zen AVX2+FMA
       now selects the measured six-row register tile
       (``SelectGemmRegisterRowsForMicroarchitecture`` returns
       ``kGemmZenAVX2MR``) so Zen's two FMA pipelines keep enough independent
       FP32/FP64 accumulator chains in flight; modern Intel Core keeps the
       five-row tile and generic x86 keeps the conservative four-row tile. The
       tuned register rows propagate through MC alignment, row packing,
       algorithm selection, and the parallel task grid, so the large-matrix
       blocking is both microarchitecture- and shape-aware. Dedicated-hardware
       speed-up measurements for the 1024³/2048³ priority cases are still
       pending.
   * - Roadmap PR06.6
     - Shared-runner FP32/FP64 diagnostics.
     - Isolated measurements identify and verify fixes for skinny-N and
       large-matrix regressions without claiming ONNX Runtime parity.
     - PR06.2 through PR06.5
     - Implemented. The new isolated ``tools/gemm_throughput.cc`` driver adds
       kernel-level evidence between merges: on a shared CI runner (Intel Xeon
       Platinum 8370C, Ice Lake-SP, AVX-512, ``intel-core`` profile, six
       register rows, four vCPU) the single-thread square-matrix throughput is
       104 GFLOP/s FP32 and 48 GFLOP/s FP64 at 512³, and stays at 99-100
       GFLOP/s FP32 through 1024³ and 2048³, confirming that PR06.5 removed the
       FP32 large-matrix regression on Intel AVX-512. The single-thread FP64
       path still drops from 48 to ~32 GFLOP/s (~33%) at 1024³ and 2048³ while
       the multi-thread FP64 path recovers (48 to 58 GFLOP/s). The remaining
       skinny-N weakness is now fixed: the ``GemmSkinnyN`` unit-stride dot
       product carried only four partial sums, which the SLP vectorizer packed
       into a single 128-bit SSE accumulator instead of full-width AVX, so the
       ``skinny_n`` (M=1024, N=1, K=1024) column-vector shape stalled at a
       scalar-class rate. Carrying sixteen independent partial sums lets the
       vectorizer fill two full-width AVX accumulators; on a shared AMD EPYC
       7763 (Zen 3, AVX2, ``amd-zen`` profile, six register rows, four vCPU)
       runner ``skinny_n`` rises from 3.3 to 22 GFLOP/s FP32 (about 6.7x) and
       8.7 to 12.4 GFLOP/s FP64, with no regression on the square, skinny-M,
       large-K, transpose, or transformer-projection priority shapes. Against
       the plan's predictions this matches the measured-fix step that vectorizes
       the skinny-N K reduction, including ``N == 1``. The plan makes no
       absolute-GFLOP/s prediction: every quantitative target is a ratio versus
       ONNX Runtime, which this isolated driver cannot measure. These are
       diagnostic shared-runner numbers, not the dedicated-machine ONNX Runtime
       evidence the gate requires.
       The large-matrix weakness that followed is now fixed as well: the five
       loops packed one ``KC x NC`` B panel but walked it with the row tiles
       outside and the columns inside, so every row tile re-streamed the whole
       L3-sized panel, and each ``k`` step of the micro-kernel jumped by the
       panel width -- a large power of two for the priority shapes, which maps a
       micro-panel onto very few L1 sets and defeats the hardware prefetcher.
       ``PackBPanel`` now stores the panel as contiguous ``KC x column_block``
       column micro-panels (``SelectGemmColumnBlock``, four cache lines per
       ``k`` row) and the tile loops iterate ``jr`` outside and ``ir`` inside, as
       the five-loop decomposition above prescribes, so the micro-kernel reads B
       sequentially and each micro-panel is reused from L2 by every row tile of
       the packed A panel. On the shared AMD EPYC 7763 runner the single-thread
       square shapes rise from 74.9 to 85.1 GFLOP/s FP32 and 25.5 to 44.4 FP64 at
       512³, from 49.2 to 86.0 and 25.5 to 44.9 at 1024³, and from 48.3 to 85.9
       and 24.8 to 44.7 at 2048³ (about 1.8x for both types), with
       ``transformer_proj`` rising from 45.6 to 81.1 GFLOP/s FP32 and 23.4 to
       40.8 FP64; the two-thread runs improve in the same proportion (1024³ from
       69.5 to 120.5 GFLOP/s FP32 and 42.8 to 72.2 FP64, 2048³ from 83.1 to 144.9
       and 47.1 to 82.7). Throughput now sustains from 512³ to 2048³ instead of
       decaying, and no priority shape regresses: ``skinny_m_gemv``,
       ``skinny_n``, ``large_k``, ``trans_a_128``, and ``trans_b_128`` are equal
       or better. Re-measuring an Intel AVX-512 runner (Xeon 6900-series,
       AVX-512, generic profile, six register rows, four vCPU) confirms the
       column micro-panel layout removed the FP64 large-matrix drop there too:
       the single-thread FP64 square shapes now sustain about 73-84 GFLOP/s
       across 512³ (73-82), 1024³ (77-85), and 2048³ (73-81) instead of falling
       from 48 to ~32 GFLOP/s, and the single-thread FP32 square shapes sustain
       about 155-181 GFLOP/s over the same range rather than decaying; the
       multi-thread square shapes reach 225 GFLOP/s FP32 and 108 FP64 at 1024³
       and 284 and 130 at 2048³, ``transformer_proj`` holds 135-175 GFLOP/s
       FP32 and 62-87 FP64, and no priority shape regresses. With both the Zen
       and Intel AVX-512 large-matrix regressions now measured as fixed. The
       dedicated-machine ONNX Runtime comparison is tracked by PR10.5.
   * - Roadmap PR07.0
     - Panel-converted FP16/BF16 general path.
     - The five-loop engine converts FP16/BF16 while packing, accumulates in
       FP32, and narrows in the epilogue without full A/B tensor widening.
     - PR06.5
     - `Implemented in #194
       <https://github.com/xadupre/onnx-light-cpu/pull/194>`_
   * - Roadmap PR07.1
     - Vectorized contiguous x86 conversion.
     - Contiguous FP16 panels use F16C and contiguous BF16 panels use AVX2,
       with exact tails and runtime ISA fallback. No native dot-product kernel
       or unrelated execution path is included.
     - PR07.0
     - `Implemented in #199
       <https://github.com/xadupre/onnx-light-cpu/pull/199>`_
   * - Roadmap PR07.2
     - Remove remaining low-precision full-tensor widening.
     - Skinny-M, skinny-N, direct, small-K, and split-K execute from typed
       inputs and FP32 accumulators. Immutable plans cover FP16/BF16, and tests
       prove that no priority algorithm allocates expanded A or B tensors.
     - PR07.0
     - `Implemented in #203
       <https://github.com/xadupre/onnx-light-cpu/pull/203>`_
   * - Roadmap PR07.3
     - Native AVX-512FP16 kernel.
     - One FP16 micro-kernel family, its CPUID dispatch, tails, and differential
       tests land without BF16 or AMX changes.
     - PR07.2
     - Implemented. A new ``CpuSupportsAvx512Fp16()`` CPUID gate
       (leaf 7 ``EDX[23]`` plus OS AVX-512 state) selects the native path at
       runtime. The ``gemm/avx512fp16`` translation unit is compiled with
       ``-mavx512fp16`` and carries ``GemmMicroKernel_AVX512FP16``, which keeps
       both operands in FLOAT16 -- a packed FLOAT16 ``A`` panel and the
       non-transposed FLOAT16 ``B`` matrix -- widens each 16-lane ``B`` vector
       with ``vcvtph2psx`` (``_mm512_cvtxph_ps``), broadcasts each FLOAT16 ``A``
       element, and accumulates in float32, so the result matches the
       widen-then-float32 reference. Column counts that are not a multiple of
       sixteen finish through the shared scalar member
       ``GemmMicroKernel_ScalarFp16`` (also the portable, always-built family
       member). ``GemmHalfPlanned`` dispatches the general FLOAT16 algorithm to
       the native ``GemmFp16NativeGeneral`` driver when the CPU reports
       AVX-512FP16 and ``B`` is not transposed, halving the ``B`` traffic; every
       other shape and ISA keeps the converting float32 path from PR07.0-07.2.
       New ``GemmFp16Native`` C++ tests run the driver with the scalar member
       everywhere (square, column-tail, ``trans_a``, empty-K, non-unit alpha)
       and repeat the general FLOAT16 differential shapes through the public
       path on AVX-512FP16 hardware. BF16 and AMX are untouched. The
       dedicated-machine ONNX Runtime comparison is tracked by PR10.5.
   * - Roadmap PR07.4
     - Native AVX-512BF16 kernel.
     - One BF16 dot-product kernel family, its CPUID dispatch, tails, and
       differential tests land with the existing converted-panel fallback.
     - PR07.2
     - Implemented. A new ``CpuSupportsAvx512Bf16()`` CPUID gate
       (leaf 7 subleaf 1 ``EAX[5]`` plus OS AVX-512 state) selects the native
       path at runtime. The ``gemm/avx512bf16`` translation unit is compiled
       with ``-mavx512bf16`` and carries ``GemmMicroKernel_AVX512BF16``, which
       keeps both operands in BFLOAT16 -- a packed BFLOAT16 ``A`` panel and the
       non-transposed BFLOAT16 ``B`` matrix -- and reduces two ``k`` iterations
       at a time with the ``vdpbf16ps`` (``_mm512_dpbf16_ps``) dot-product,
       accumulating in float32, so the result matches the widen-then-float32
       reference while halving the ``B`` traffic. Each 16-lane pair vector is
       assembled by zero-extending two consecutive BFLOAT16 ``B`` rows and
       broadcasting the matching BFLOAT16 ``A`` pair; an odd ``K`` finishes with
       one pair whose second element is zeroed, and column counts that are not a
       multiple of sixteen finish through the shared scalar member
       ``GemmMicroKernel_ScalarBf16`` (also the portable, always-built family
       member). ``GemmHalfPlanned`` dispatches the general BFLOAT16 algorithm to
       the native ``GemmBf16NativeGeneral`` driver when the CPU reports
       AVX-512BF16 and ``B`` is not transposed; every other shape and ISA keeps
       the converting float32 path from PR07.0-07.2. New ``GemmBf16Native`` C++
       tests run the driver with the scalar member everywhere (square,
       column-tail, even/odd ``K``, ``trans_a``, empty-K, non-unit alpha) and
       repeat the general BFLOAT16 differential shapes through the public path on
       AVX-512BF16 hardware. FP16, AMX, and INT8 are untouched. The
       dedicated-machine ONNX Runtime comparison is tracked by PR10.5.
   * - Roadmap PR07.5
     - AMX tile-state lifecycle.
     - OS-enabled tile-state detection, per-worker tile configuration, and
       safe fallback pass focused lifecycle tests. No GEMM kernel is included.
     - PR07.2
     - Implemented. New ``CpuSupportsAmxTile()``, ``CpuSupportsAmxBf16()``, and
       ``CpuSupportsAmxInt8()`` CPUID gates (leaf 7 subleaf 0 ``EDX`` bits 24,
       22, and 25) check the AMX feature bits and, via a new
       ``OsSupportsAmxTileState()`` helper, that the OS has enabled the
       ``XTILECFG`` and ``XTILEDATA`` components in ``XCR0`` (bits 17 and 18).
       A dedicated ``gemm/amx`` translation unit compiled with ``-mamx-tile``
       owns the lifecycle: ``AmxTileStateAvailable()`` requests the
       ``XTILEDATA`` permission once per process on Linux
       (``ARCH_REQ_XCOMP_PERM`` through ``arch_prctl``, harmless when AMX is
       absent), then caches whether tile state is usable. The hardware-defined
       64-byte ``AmxTileConfig`` (``TILECFG``) is populated with the validating
       ``AmxTileConfigSetTile`` builder, and the per-worker RAII
       ``AmxTileScope`` runs ``LDTILECFG`` on construction and ``TILERELEASE``
       on destruction, degrading to a no-op that reports ``configured() ==
       false`` whenever AMX tile state is unavailable. No AMX GEMM kernel is
       added: the AMX-BF16 kernel is PR07.6 and AMX-INT8 is PR09.4. New
       ``GemmAmxTile`` C++ lifecycle tests cover the ``TILECFG`` layout, the
       builder's range validation, detection consistency, availability
       idempotency, and the safe fallback of the scope (including from a worker
       thread); when the toolchain lacks ``-mamx-tile`` the module keeps the
       fallback and still links. The dedicated-machine ONNX Runtime comparison
       is tracked by PR10.5.
   * - Roadmap PR07.6
     - AMX-BF16 kernel.
     - The AMX-BF16 kernel reuses PR07.5, passes differential tests on native
       hardware, and falls back to AVX-512BF16. No INT8 work is included.
     - PR07.4, PR07.5
     - Implemented. A dedicated ``gemm/amx/gemm_amx_bf16`` translation unit
       compiled with ``-mamx-tile -mamx-bf16`` (guarded by
       ``ONNX_LIGHT_CPU_HAVE_AMX_BF16``) adds ``GemmMicroKernel_AMXBF16``, a
       native member of the BFLOAT16 micro-kernel family sharing the
       ``GemmBf16MicroKernel`` signature so the PR07.4
       ``GemmBf16NativeGeneral`` driver, BFLOAT16 ``A`` packing (including
       ``trans_a``), and column-tail handling are reused. The kernel configures
       three AMX tiles through the PR07.5 lifecycle (``AmxTileConfig`` /
       ``AmxTileScope``) and reduces ``K`` with the ``tdpbf16ps``
       (``_tile_dpbf16ps``) dot-product: an ``mr x 32`` BFLOAT16 ``A`` tile times
       a 16-pair by 16-column VNNI-packed BFLOAT16 ``B`` tile accumulating a
       16x16 float32 ``C`` tile, with every partial ``mr``/``K``/column block
       zero-padded into the fixed 16x16 tiles so a single tile configuration
       handles all shapes. ``GemmHalfPlanned<kGeneral>`` dispatches here ahead of
       AVX-512BF16 for non-transposed ``B`` when ``CpuSupportsAmxBf16()`` and
       ``AmxTileStateAvailable()`` both report the ISA, and the kernel degrades
       to the shared scalar BFLOAT16 member when tile state is unavailable. New
       ``GemmBf16Native.AmxBf16KernelMatchesReferenceWhenSupported`` differential
       tests cover the full 16x16 tile, the zero-padded row/column/K tails, and a
       transposed ``A`` on capable hardware. The dedicated-machine ONNX Runtime
       comparison is tracked by PR10.5.
   * - Roadmap PR08.1
     - ARM FP16/BF16 panel conversion.
     - NEON vectorizes conversion while packing with exact tails and scalar
       fallback; native arithmetic and SVE are excluded.
     - PR07.2, PR05
     - Implemented. The NEON translation unit
       (``gemm/arm/gemm_kernel_neon``) adds ``GemmConvertBFloat16ToFloat32_NEON``
       -- a baseline zero-extend (``vmovl_u16``) and 16-bit left shift
       (``vshlq_n_u32``) -- and, when the ``vld1q_f16`` / ``vcvt_f32_f16``
       (``FCVTL``) intrinsics compile (probed as
       ``ONNX_LIGHT_CPU_HAVE_NEON_FP16``), ``GemmConvertFloat16ToFloat32_NEON``.
       Both widen eight contiguous half-precision patterns per iteration with an
       exact scalar tail that reuses ``Bfloat16BitsToFloat`` /
       ``Float16BitsToFloat`` so the vectorized result is bit-identical to the
       scalar decode. ``PackConvertContiguous`` dispatches to them on ARM the way
       it already dispatches to AVX2/F16C on x86, so the FP16/BF16
       convert-while-packing paths from Roadmap PR07.1-07.2 vectorize on ARM
       without a full-tensor widening pass; BF16 always uses NEON and FP16 falls
       back to the scalar bit decode when the FP16 intrinsics are unavailable.
       No native ARM FP16/BF16 arithmetic (PR08.2) or SVE conversion (PR08.3) is
       added. The existing FP16/BF16 differential tests plus a new
       ``GemmHalf.HalfVectorizedPackingTailRemainders`` case (every 1..7 element
       tail across the eight-lane width) run through the NEON path under the
       cross-compiled aarch64 QEMU CI and match the widen-then-float32 reference.
   * - Roadmap PR08.2
     - Native ARM FP16/BF16 arithmetic.
     - NEON FP16 and available BF16 dot-product kernels pass the complete
       differential corpus while retaining PR08.1 as fallback.
     - PR08.1
     - Implemented. The NEON translation unit (``gemm/arm/gemm_kernel_neon``)
       adds ``GemmMicroKernel_NEON_BF16`` (always, baseline NEON) and, when the
       FP16 vector load/convert intrinsics compile
       (``ONNX_LIGHT_CPU_HAVE_NEON_FP16``), ``GemmMicroKernel_NEON_FP16`` -- new
       members of the BFLOAT16 / FLOAT16 micro-kernel families that share the
       ``GemmBf16MicroKernel`` / ``GemmFp16MicroKernel`` signatures so the PR07.4
       / PR07.3 ``GemmBf16NativeGeneral`` / ``GemmFp16NativeGeneral`` drivers,
       half-precision ``A`` packing (including ``trans_a``), and column-tail
       handling are reused. Unlike the PR08.1 convert-while-packing path, both
       operands stay half-precision to the register file: each eight-column
       ``B`` row is widened on the fly (BFLOAT16 with the baseline NEON
       zero-extend / 16-bit shift, FLOAT16 with the ``vcvt_f32_f16`` /
       ``FCVTL`` instruction) into ``float32x4`` vectors register-blocked over
       up to ``kGemmNeonMR`` rows, the dot products accumulate in float32, and
       ``alpha`` is applied in the epilogue; the eight-, four-, and scalar
       (``GemmMicroKernel_ScalarBf16`` / ``ScalarFp16``) column paths keep the
       result identical to the widen-then-float32 reference.
       ``GemmHalfPlanned<kGeneral>`` dispatches here for non-transposed ``B``
       (BFLOAT16 always on NEON, FLOAT16 only when the FP16 intrinsics are
       present) ahead of the converting float32 path; every other shape, a
       transposed ``B``, or a toolchain without the FP16 intrinsics keeps the
       PR08.1 fallback. New ``GemmHalf.Float16NeonNativeGeneralColumnTails`` and
       ``GemmHalf.BFloat16NeonNativeGeneralColumnTails`` differential shapes
       (exact eight-lane ``N``, the four-lane and scalar column tails, even/odd
       ``K``, and a transposed ``A``) run through the NEON path under the native
       ARM64 and cross-compiled aarch64 QEMU CI and match the reference. No SVE
       conversion (PR08.3) is added. The dedicated-machine ONNX Runtime
       comparison is tracked by PR10.5.
   * - Roadmap PR08.3
     - SVE/SVE2 FP16/BF16 kernels.
     - Runtime-vector-length-aware kernels and predicated tails pass under
       native hardware or QEMU, with NEON selected for short vector lengths.
     - PR08.2
     - Implemented. The SVE translation unit (``gemm/arm/gemm_kernel_sve``) adds
       ``GemmMicroKernel_SVE_BF16`` and ``GemmMicroKernel_SVE_FP16`` -- new
       members of the BFLOAT16 / FLOAT16 micro-kernel families that share the
       ``GemmBf16MicroKernel`` / ``GemmFp16MicroKernel`` signatures, so the
       PR07.4 / PR07.3 ``GemmBf16NativeGeneral`` / ``GemmFp16NativeGeneral``
       drivers, half-precision ``A`` packing (including ``trans_a``), and the
       ``GemmMicroKernel_ScalarBf16`` / ``ScalarFp16`` row fallback are reused.
       Like the PR08.2 NEON kernels both operands stay half-precision to the
       register file: each ``B`` row is widened on the fly (BFLOAT16 with a
       zero-extending halfword load ``svld1uh_u32`` plus a 16-bit shift, FLOAT16
       with the same load reinterpreted so the SVE ``FCVT`` ``svcvt_f32_f16``
       reads the pattern from the low 16 bits of each word), the dot products
       accumulate in float32, and ``alpha`` is applied in the epilogue. The
       runtime vector length (``svcntw``) drives the lane count, the leading loop
       consumes two vectors per column step, and the ``svwhilelt`` predicated
       tail covers the column remainder without reading inactive columns, so the
       result is identical to the widen-then-float32 reference. Half precision is
       part of baseline SVE, so unlike the NEON FLOAT16 kernel the SVE FLOAT16
       kernel needs no separate FP16 feature gate. ``GemmHalfPlanned<kGeneral>``
       dispatches here for a non-transposed ``B`` when ``DetectArmGemmProfile``
       selects SVE (a vector length of at least 256 bits) ahead of the NEON path;
       a 128-bit vector length keeps the better-unrolled six-row NEON kernel, a
       transposed ``B`` keeps the converting float32 path, and a build without SVE
       is unaffected. New ``GemmHalf.Float16SveNativeGeneralColumnTails`` and
       ``GemmHalf.BFloat16SveNativeGeneralColumnTails`` differential shapes (an
       ``N`` spanning several SVE vectors, the ``svwhilelt`` predicated column
       tail, even/odd ``K``, and a transposed ``A``) run through the SVE path
       under the cross-compiled aarch64 QEMU CI at 512-bit and 384-bit vector
       lengths (and the NEON fallback at 128 bits) and match the reference. The
       dedicated-machine ONNX Runtime comparison is tracked by PR10.5.
   * - Roadmap PR09.1
     - Portable integer semantics.
     - INT8/UINT8/INT32/INT64 implement schema-defined zero points, overflow,
       accumulation, and requantization with exact scalar differential tests.
       No ISA-specific code is included.
     - PR07.2
     - Implemented. ``MatMulInteger`` accepts mixed INT8/UINT8 operands,
       optional scalar or per-row/per-column zero points, ONNX MatMul batch
       broadcasting and rank-1 promotion, and defines INT32 accumulation as
       modulo-2^32 arithmetic without signed-overflow undefined behaviour.
       ``QLinearMatMul`` accumulates in INT64, applies scalar FLOAT/FLOAT16
       scales, round-to-nearest-even requantization, output zero points, and
       exact INT8/UINT8 saturation. Both portable kernels are registered with
       onnx-light and covered by scalar differential, broadcast, overflow,
       validation, rounding, and saturation tests; no ISA-specific code is
       included.
   * - Roadmap PR09.2
     - x86 VNNI INT8 kernel.
     - Signed and unsigned VNNI paths have exact tails, runtime dispatch, and
       differential tests over the PR09.1 fallback. ARM and AMX are excluded.
     - PR09.1
     - Implemented. ``MatMulInteger`` routes the plain matrix product (both
       operands rank >= 2) through ``IntegerMatMul2D``
       (``impl/math/gemm/vnni/integer_gemm_vnni.cc``), which packs ``A`` into a
       UINT8 panel and ``B`` into a transposed INT8 panel, applying the
       ``vpdpbusd`` signedness offsets (``+128`` for a signed ``A``, ``-128``
       for an unsigned ``B``). The raw ``uint8 x int8`` dot-product is then
       corrected with the true row/column sums so both INT8 and UINT8 operands,
       scalar or per-row/per-column zero points, and INT32 modulo-2^32
       accumulation are reconstructed exactly. The native
       ``IntegerDotU8S8Avx512Vnni`` (``integer_gemm_avx512vnni.cc``, compiled
       with ``-mavx512f -mavx512vnni``) reduces four ``uint8 x int8`` products
       per lane with ``_mm512_dpbusd_epi32`` and finishes non-multiples of the
       64-byte vector through the same scalar tail; it is dispatched ahead of
       the portable ``IntegerDotU8S8Scalar`` sibling only when
       ``CpuSupportsAvx512Vnni()`` reports the ISA at runtime, so a single
       binary keeps working on CPUs without VNNI. Vector / rank-1 promotions and
       ``QLinearMatMul`` keep the PR09.1 scalar fallback. New
       ``IntegerVnniKernel`` differential tests exercise both dot-product paths
       against a naive reference across signedness, scalar and per-axis zero
       points, K tails, and the modulo-2^32 wrap. ARM and AMX are excluded
       (PR09.3 and PR09.4).
   * - Roadmap PR09.3
     - ARM dot-product INT8 kernel.
     - Signed and unsigned NEON dot-product paths have exact tails, runtime
       dispatch, and differential tests over the PR09.1 fallback.
     - PR09.1
     - Implemented. The contiguous rank >= 2 ``MatMulInteger`` product is routed
       through the shared ``IntegerMatMul2D`` driver
       (``impl/math/gemm/vnni/integer_gemm_vnni.cc``), which dispatches to the
       native NEON dot-product kernel ``GemmMatMulIntegerNeonDotProd``
       (``gemm/arm/gemm_kernel_neon_dotprod.cc``, built with
       ``-march=armv8.2-a+dotprod``) when the build and CPU (runtime
       ``CpuSupportsNeonDotProd``, ``HWCAP_ASIMDDP``) support it, ahead of the
       x86 VNNI path (PR09.2) and otherwise the shared portable scalar reduction.
       A single unsigned ``UDOT`` path
       serves every signedness combination: a signed operand maps into the
       unsigned byte domain by flipping its sign bit with the ``+128`` bias
       folded into that operand's effective zero point, and the raw products are
       recovered with per-row / per-column byte-sum corrections, so the result
       -- including scalar and per-axis zero points -- reproduces the scalar
       accumulator bit for bit modulo 2^32. ``A`` stays contiguous along ``K``
       while ``B`` is packed once into a column-major ``N x K`` buffer, the
       16-byte ``UDOT`` body carries an exact scalar ``K`` tail, and new
       ``IntegerMatMul2D`` differential shapes in ``test_gemm_kernel`` (every
       signedness combination, scalar and per-axis zero points, and aligned and
       tail ``K``) run through the shared dispatch under the cross-compiled
       aarch64 QEMU CI and match the reference. AMX is excluded (PR09.4).
   * - Roadmap PR09.4
     - AMX-INT8 kernel.
     - The PR07.5 tile-state lifecycle is reused for signed and unsigned INT8,
       with exact tails and VNNI fallback.
     - PR07.5, PR09.2
     - Implemented. The ``gemm/amx/gemm_amx_int8`` translation unit is compiled
       with ``-mamx-tile -mamx-int8`` and dispatches ahead of AVX-512 VNNI when
       ``CpuSupportsAmxInt8()`` and ``AmxTileStateAvailable()`` confirm the ISA
       and OS tile state. It maps signed operands into the unsigned domain,
       packs a 16x64-byte ``A`` tile and a four-byte VNNI-packed 16x16 ``B``
       tile, and uses ``tdpbuud`` to accumulate INT32 output tiles. Per-row and
       per-column byte sums and zero-point corrections preserve the exact
       modulo-2^32 ``MatMulInteger`` result for every signedness combination;
       partial row, column, and ``K`` tiles are zero-padded. If tile
       configuration is unavailable, the shared scalar path is used, and CPUs
       without AMX keep the PR09.2 VNNI then scalar dispatch. The
       ``IntegerMatMul2D`` differential corpus includes a 17x19x67 per-axis
       case to exercise all AMX tile tails.
   * - Roadmap PR09.5
     - Float8 packing formats.
     - Each supported Float8 format has an explicit vectorized decode/packing
       path, exact tail handling, and differential tests. Integer and INT4
       kernels are unchanged.
     - PR07.2
     - Implemented. Each of the four ONNX Float8 formats (``E4M3FN``,
       ``E4M3FNUZ``, ``E5M2``, ``E5M2FNUZ``) is handled as a separate packing
       format rather than a branch in the FP32 inner loop: new
       ``impl/math/gemm/float8/float8_conversion.h`` provides exact per-format
       scalar decoders (validated bit for bit against the ``ml_dtypes`` / ONNX
       reference for all 256 byte patterns, including the format-specific NaN
       encodings) and builds an exact 256-entry decode table. A ``Float8Source``
       view decodes each one-byte pattern to float32 while the operands are
       packed (general/direct/skinny-M) or reduced (skinny-N), so no full-tensor
       conversion buffer is allocated; the contiguous packing copies gather from
       the per-format decode table through the AVX2 ``vgatherdps`` helper
       ``GemmDecodeFloat8ToFloat32_AVX2`` (exact scalar tail, scalar fallback off
       x86) while the transposed strided gathers keep the per-element decode.
       The reduction accumulates in float32, reusing every tuned FP32 algorithm
       through ``GemmFloat8Planned`` / ``GemmFloat8ToFloat`` and the new public
       ``GemmFloat8WithEpilogue`` entry point. New ``GemmFloat8`` differential
       tests in ``test_gemm_kernel`` cover the decode tables, all four formats,
       transpose combinations, column and K tails, and the skinny-N / skinny-M /
       split-K / direct / empty-K algorithm variants against a reference computed
       on the decoded float32 values. Integer and INT4 kernels are unchanged.
   * - Roadmap PR09.6
     - Packed INT4/UINT4 formats.
     - Nibbles unpack into typed panels or a native dot-product path with exact
       odd-length tails and differential tests. Float8 is unchanged.
     - PR09.1
     - Implemented. ``IntegerMatMul4Bit2D`` consumes ONNX low-nibble-first
       row-major packed operands without materializing unpacked source
       matrices. INT4 two's-complement and UINT4 values are expanded directly
       into the shared UINT8 x INT8 panels with zero-point corrections and
       modulo-2^32 INT32 accumulation. The panels dispatch through AVX-512 VNNI
       or ARM NEON dot product when available and otherwise use the exact scalar
       sibling; AMX remains an INT8-only path. Differential tests cover all four
       operand signedness combinations, scalar and per-axis zero points,
       aligned and odd logical dimensions (including rows that cross nibble
       boundaries), and ignored final high nibbles. Float8 is unchanged.
   * - Roadmap PR10.1
     - FP16/BF16 correctness gate.
     - The complete FP16/BF16 corpus passes on x86, ARM, and every fallback.
       This PR contains tests and fixes only, not performance tuning.
     - PR07.2 through PR08.3
     - Implemented. ``test_gemm_kernel`` now includes a shared
       ``GemmHalf.HalfCorrectnessGateCorpus`` that runs the same FLOAT16 and
       BFLOAT16 matrix set across native and fallback paths (transpose
       combinations, non-trivial tails, and empty-``K`` bias handling), while
       retaining the PR08.2/PR08.3 ARM-native differential tests for NEON/SVE.
   * - Roadmap PR10.2
     - Integer and compact-format correctness gate.
     - The complete integer, Float8, and packed-4-bit corpus passes on every
       available ISA and fallback. This PR contains tests and fixes only.
     - PR09.1 through PR09.6
     - Implemented. ``GemmFloat8.Float8CorrectnessGateCorpus`` runs every Float8
       format through the same transpose, tail, general, direct, skinny-M,
       skinny-N, split-K, and empty-K cases on each CI architecture. The
       integer differential target now compiles the native-path checks that
       match the kernels present in the library and compares byte GEMM through
       the dispatcher, scalar, AVX-512 VNNI, ARM NEON dot-product, and AMX-INT8
       paths when available. Its packed INT4/UINT4 corpus checks the dispatcher,
       scalar, VNNI, and NEON paths across every signedness pairing, zero-point
       layout, nibble boundary, vector tail, empty-K, and modulo-INT32 case.
       Native ARM and emulated ARM CI build and run this target explicitly.
       This gate adds no performance tuning.
   * - Roadmap PR10.3
     - FP16/BF16 performance gate.
     - Dedicated-machine x86 and ARM results reach at least 1.0x ONNX Runtime
       median with no priority case below 0.9x where the type is supported.
       The PR contains measurement-driven tuning only.
     - PR10.1
     - In progress. The `first tuning pass in #267
       <https://github.com/xadupre/onnx-light-cpu/pull/267>`_ raises the 18-case
       median from 0.341x to 0.479x ONNX Runtime and the minimum from 0.138x to
       0.237x on a pinned AVX2/FMA/F16C i7-13800H thread. The `second tuning
       pass in #273 <https://github.com/xadupre/onnx-light-cpu/pull/273>`_
       vectorizes FP16/BF16 output narrowing, adds an AVX2 BF16 direct
       micro-kernel, and routes the operator workspace through the runtime
       arena. The focused FLOAT16 direct/tiny corpus reaches a 1.178x median
       and 1.046x minimum. The complete dedicated-machine gate remains open.
   * - Roadmap PR10.4
     - Integer and compact-format performance gate.
     - Dedicated-machine results are published per type and ISA. Types
       supported by ONNX Runtime reach at least 1.0x median with no priority
       case below 0.9x; unsupported types publish correctness and throughput.
     - PR10.2
     - In progress. The `first tuning pass in #274
       <https://github.com/xadupre/onnx-light-cpu/pull/274>`_ adds the integer
       parity and compact throughput instruments plus an exact AVX2 UINT8 x
       INT8 dot product used by both byte and packed-4-bit GEMM. On the
       diagnostic AVX2 host, isolated square-512 throughput rises from 20.66 to
       58.54 GOPS for INT8 and from 18.64 to 49.97 GOPS for INT4. The full
       dedicated-machine gate remains open. The second pass replaces the
       per-output AVX2 reduction with a 2x2 blocked output micro-kernel that
       reuses loaded A and packed B vectors across outputs while preserving the
       exact scalar output and reduction tails.
   * - Roadmap PR10.5
     - Final blocking GEMM parity gate.
     - Raw dedicated-machine results cover Gemm, shared MatMul, batched paths,
       every priority platform, and every supported type. FP32, FP64, and each
       type supported by ONNX Runtime reach at least 1.0x median performance
       with no priority case below 0.9x. The PR remains open while any target
       fails. The separate :doc:`Attention roadmap <2026_08_attention>` starts
       after this gate closes.
     - PR10.3, PR10.4
     - Pending

Roadmap PR10.5 is the final Gemm and MatMul roadmap PR.

The reproducible gate command is ``tools/benchmark_gemm_parity.py
--operator all --dtype all --threads 1 --output gemm_matmul_parity_results.json``.
It alternates the registered CPU kernel with ONNX Runtime, records every raw
sample and environment field, and includes dynamic and constant ``Gemm``,
shared ``MatMul``, batched/broadcast, vector, transpose, bias, skinny,
large-K, split-K, and transformer cases. Repeat the command with the
physical-core affinity and thread count for each priority machine; pass
``--enforce`` only when publishing a completed dedicated-machine result.

Roadmap PR10.3 tuning record
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first PR10.3 pass adds alternating FLOAT16 measurements to
``benchmark_gemm_parity.py`` and tunes the AVX2/F16C path without allocating
expanded FP32 operands. Contiguous A/B packing now widens eight values per
``vcvtph2ps`` instruction. Dedicated skinny-M and skinny-N kernels reuse each
widened operand across vector FMAs. Small-K direct GEMM widens B in registers,
while larger general GEMM deliberately retains shared FP32 B panels: measuring
the native kernel on all general shapes showed that reconverting B for every
row tile regressed ``square_1024`` and transformer projection. Tiny ``2 x 2``
split-K instead bypasses thread-pool partials and uses a two-column,
K-vectorized kernel; this changes its pinned one-thread result from 0.148x to
2.174x and its ten-thread result from 0.056x to 2.215x.

The `second pass in #273
<https://github.com/xadupre/onnx-light-cpu/pull/273>`_ isolates the
``32 x 128 x 16`` direct shape and shows that its prepared kernel reaches only
3.30 GFLOP/s FP16 and 1.97 GFLOP/s BF16 before operator dispatch is included,
disproving the initial assumption that plan construction is the primary
bottleneck. The hot epilogue narrows every float32 accumulator through a scalar
software conversion. AVX2/F16C conversion now narrows eight FP16 outputs at
once, AVX2 integer rounding narrows eight BF16 outputs at once, and both retain
the scalar contract for NaNs and tails. The direct BF16 path also keeps its
compact inputs to the register file through an AVX2 micro-kernel instead of
using converted FP32 panels. Isolated direct throughput rises to 20.60 GFLOP/s
FP16 and 23.36 GFLOP/s BF16 on the same host. At the operator level, FLOAT16
direct falls from 26.54 to 7.45 microseconds and rises from 0.330x to 1.178x
ONNX Runtime; ``tiny_dynamic`` and ``tiny_constant`` remain above parity at
1.252x and 1.046x. The float32 workspace now comes from the reusable runtime
execution arena instead of a new ``std::vector`` allocation on every invocation.

The third pass replaces scalar transposed FP16/BF16 gathers on AVX2 with blocked
``8 x 8`` 16-bit register transposes followed by eight-lane F16C or AVX2 BF16
widening. Both transposed A and B packing use the same kernel, while arbitrary
row and column tails retain the scalar conversion contract, including NaNs.
The compact throughput driver now reports isolated FP16 and BF16 throughput
alongside the other compact formats. Parity reports retain every timing sample
and dispersion and additionally record affinity policy, compiler, and NumPy,
onnx-light, onnx-light-cpu, and ONNX Runtime versions.

The published numbers above are diagnostic WSL measurements, not the
dedicated-machine evidence required to close PR10.3. ONNX Runtime 1.28.0 does
not implement CPU BFLOAT16 Gemm on this host, so BFLOAT16 remains an isolated
throughput measurement rather than a parity ratio. ARM results are also still
required.

The remaining PR10.3 tuning order is:

#. Reuse compact B panels across row tasks. For medium and large non-transposed
   GEMM, benchmark a cache-sized FP16 packed-B layout whose conversion happens
   inside the micro-kernel, but only dispatch it when B reuse beats the current
   once-per-panel FP32 widening.
#. Vectorize transposed packing. Add blocked FP16 transpose/convert kernels for
   A and B, then tune square and rectangular block sizes against ``trans_a``,
   ``trans_b``, and ``trans_ab`` independently.
#. Fix physical-core scaling. A ten-thread diagnostic corpus reaches only
   0.309x median after the tiny split-K correction, with high variance and a
   0.040x minimum; partition packed B panels across reusable physical-core
   tasks and eliminate per-invocation pool and allocation costs before
   retuning MC/NC/KC.
#. Run dedicated x86 and ARM sweeps. Publish one-thread and physical-core raw
   samples, dispersion, FP16 parity, and isolated BF16 throughput per ISA. Keep
   PR10.3 open until every supported platform reaches the stated 1.0x median
   and 0.9x minimum gates.

Roadmap PR10.4 tuning record
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `first PR10.4 pass in #274
<https://github.com/xadupre/onnx-light-cpu/pull/274>`_ establishes two
reproducible instruments. The Python runner alternates registered UINT8 x INT8
``MatMulInteger`` against ONNX Runtime, verifies exact INT32 output, and stores
every timing sample plus CPU, ISA, affinity, and thread metadata. The isolated
C++ driver reports INT8, packed INT4, E4M3, and E5M2 throughput across the
priority shape families.

The initial AVX2 measurement exposed a scalar fallback below AVX-512 VNNI:
isolated INT8 reached only 20.66 GOPS for square-512, 14.39 GOPS for large-K,
and 19.48 GOPS for transformer projection. The new AVX2 dot product splits
each UINT8 byte into low-seven-bit and high-bit terms before
``vpmaddubsw``. Each pair therefore remains inside the exact INT16 range,
including 255 x 127 and 255 x -128 adversarial inputs, before ``vpmaddwd``
accumulates modulo 2^32. The shared dispatcher uses it for both INT8 and
expanded packed-4-bit panels when VNNI, AMX, or NEON dot product is unavailable.
The same isolated shapes reach 58.54, 40.53, and 46.58 GOPS respectively;
packed INT4 square-512 rises from 18.64 to 49.97 GOPS.

The one-thread end-to-end MatMulInteger median improves from 0.109x to 0.193x
ONNX Runtime and its minimum from 0.063x to 0.097x on the diagnostic host. This
does not close PR10.4: the current driver still packs complete matrices for
every invocation.

The second pass replaces the per-output AVX2 reduction with a register-budgeted
2x2 micro-kernel for products with at least two rows, two columns, and one
32-byte reduction vector. On the same host with VNNI and AMX disabled, the
isolated square-512, large-K, and transformer cases improve from 102.15, 57.46,
and 80.33 GOPS to 105.12, 69.90, and 91.81 GOPS. Tiny and small-K products keep
the single-output reduction. These are diagnostic results, not the dedicated
machine evidence required to close PR10.4.

The current pass keeps the shared integer panels reusable across row work by
partitioning contiguous matrix batches through ``ExecuteRanges`` under the
session executor. ``QLinearMatMul`` now routes contiguous rank-2+ products
through ``IntegerMatMul2D`` and applies a vectorized AVX2 requantization
epilogue for INT8/UINT8 outputs while retaining the scalar fallback and exact
tail behavior for non-AVX2 and rank-1 promotion paths.

The final PR10.4 closure step is the dedicated machine evidence: run pinned
x86 and ARM one-thread and physical-core sweeps, publish raw samples and
dispersion with complete environment metadata, and keep the implementation PR
open until ORT-supported integer/compact formats satisfy the 1.0x median and
0.9x minimum parity gates.
