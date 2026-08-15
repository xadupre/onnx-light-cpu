Gemm, MatMul, and Attention Performance Roadmap
================================================

:Date: 2026-08

**in progress**

Objective
---------

The objective is performance parity with the ONNX Runtime CPU execution
provider for the important ``Gemm``, ``MatMul``, and tensor-based ``Attention``
workloads, for every supported data type, without sacrificing ONNX correctness.
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
kernels, Attention, and the final ONNX Runtime parity gates.

Related roadmap
---------------

Persistent state, decode, paged storage, and cache quantization are covered by
the separate :doc:`Persistent KV Cache and Decode roadmap <2026_08_kv_cache>`.

Scope and type matrix
---------------------

``Gemm`` and ``MatMul`` should share one matrix-multiplication engine.
``Attention`` should reuse its packing, dot-product micro-kernels, type
conversion, and scheduler, but it must not be implemented as two ordinary
materialized MatMul calls. The operator adapters retain distinct ONNX semantics:

* ``Gemm`` handles rank-2 inputs, ``alpha``, ``beta``, optional broadcast bias,
  and ``transA``/``transB``.
* ``MatMul`` handles vectors, matrices, arbitrary leading batch dimensions,
  NumPy-style batch broadcasting, and output-rank squeezing.
* ``Attention`` handles Q/K/V head geometry, scaling, masks, causal behavior,
  grouped-query or multi-query head mapping, softmax, and optional past/present
  KV state according to its selected ONNX opset.

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
* Cover attention prefill and decode separately: query lengths 1, 2-16, and
  long prefill; KV lengths from 1 to the target context limit; MHA, GQA, and
  MQA; causal, padding, additive, and boolean masks.
* Separate dynamic-B from constant-B cases. Constant weights must be packed
  once, not once per invocation.
* Compare single-thread throughput and scaling at 2, 4, physical-core, and
  logical-core thread counts. Hybrid P/E-core machines need their own results.
* For attention, report time to first token, per-token decode latency,
  tokens/second, peak temporary memory, and effective KV-cache bandwidth.

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
   shape-forced case for every algorithm and supports process-level thread
   selection through ``ONNX_LIGHT_CPU_NUM_THREADS``.

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
  AVX2+FMA emits compile-time ``MR=1..4`` variants and AVX-512 emits
  ``MR=1..6``, both for NR=1 and NR=2. The detected ISA selects MR=4 for
  AVX2/SSE and MR=6 for AVX-512, and the choice is propagated through cache
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
<https://github.com/xadupre/onnx-light-cpu/pull/162>`_. The remaining P4 work
and merge criteria are Roadmap PR06.2 through PR06.6 in the final table.
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
  accumulate with FMA. Narrow only the final output.
* For AVX-512BF16 and AVX-512FP16, add native dot-product/multiply-accumulate
  kernels with FP32 accumulation where required by the ONNX numerical contract.
* Add AMX tile kernels behind OS-enabled tile-state detection. AMX must remain
  optional because enabling the ISA and configuring tiles have non-trivial
  per-thread costs.
* Implement equivalent ARM FP16/BF16 and dot-product paths.
* For INT8, fuse zero-point correction and requantization into packing and the
  epilogue. Accumulate in INT32 and define overflow behavior through the ONNX
  operator contract.
* Treat Float8 and packed 4-bit types as separate packing formats, not as
  branches in the FP32 inner loop.

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

Phase 5: Attention plan and correctness path
--------------------------------------------

Start with a clear internal contract before optimizing. An ``AttentionPlan``
is built once from the model, static dimensions, CPU, and runtime options. It
records:

* batch size, query-head and KV-head counts, head dimensions, and GQA ratio;
* input/output layouts and strides;
* scale, causal mode, mask kind, and whether standard tensor ``past``/``present``
  inputs are enabled;
* prefill, short-query, or single-token decode algorithm;
* query-row and KV-column block sizes;
* dot-product/packing functions, accumulation type, and useful thread count.

The first implementation should be a simple materialized correctness path:

.. code-block:: text

   S = scale * Q @ transpose(K)
   S = apply_mask_and_causality(S)
   P = softmax(S)
   O = P @ V

This path is not the final performance target. It provides differential tests
against ONNX Runtime and a fallback for uncommon combinations while validating
all shape, mask, head-mapping, precision, and tensor ``past``/``present``
semantics. In this roadmap, appending past and present may allocate and copy;
this compatibility cost must be reported separately. Optimized persistent
state is deferred to the dedicated cache roadmap.

The Attention adapter should lower MHA, GQA, and MQA to one internal descriptor.
For GQA/MQA, several query heads reference the same K/V head through a zero-copy
head mapping; K and V must not be physically repeated.

Phase 6: streaming/online Attention
-----------------------------------

The optimized prefill algorithm should fuse ``Q @ K^T``, masking, softmax, and
``P @ V`` by blocks. It must not materialize the full
``[batch, heads, query_length, kv_length]`` score or probability tensors.

For a query block, process KV blocks from left to right while maintaining, per
query row, the running maximum ``m``, softmax denominator ``l``, and unnormalized
output accumulator ``o``. For a score block ``S``:

.. code-block:: text

   m_new = max(m, row_max(S))
   correction = exp(m - m_new)
   p = exp(S - m_new)
   l = correction * l + row_sum(p)
   o = correction * o + p @ V_block
   m = m_new

   output = o / l

This is the online-softmax recurrence used by FlashAttention-style algorithms.
It changes the computation from two materialized GEMMs plus softmax into a
blocked streaming algorithm. Arithmetic remains
``O(B * Hq * Lq * Lkv * D)``, but temporary score memory falls from
``O(B * Hq * Lq * Lkv)`` to ``O(Br * Bc)`` per worker, and each probability
block is consumed while hot.

The CPU implementation needs:

* a ``Q x K`` score micro-kernel whose epilogue applies scale, mask, causal
  bounds, and row maximum;
* a vectorized exponential and row reduction with FP32 accumulation;
* a ``probability x V`` update micro-kernel that accumulates directly into the
  query-block output;
* block sizes ``Br`` and ``Bc`` chosen jointly with head dimension and cache
  capacity;
* causal tile skipping: do not compute blocks wholly above the diagonal, and
  mask only the one intersecting diagonal block;
* direct handling of sliding-window or sparse masks by skipping absent KV
  blocks rather than filling them with negative infinity.

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
   * - Streaming Attention
     - **0-20%** when ONNX Runtime already selects an optimized fused path;
       **1.2-2x** when it materializes scores or probabilities.
     - The arithmetic count is similar, so the gain comes from avoiding
       ``O(B * Hq * Lq * Lkv)`` temporary traffic. It grows with context length
       and disappears for tiny sequences. Peak score memory drops by roughly
       ``(Lq * Lkv) / (Br * Bc)`` per head.
     - 10-20 days after the GEMM primitives and correctness path exist.

These gains are not additive. Shape specialization and autotuning often choose
the same improvement. A credible target sequence is:

1. reach at least **1.0x ONNX Runtime** median performance with the generic
   blocked algorithm and tuned scheduler;
2. reach **1.05-1.15x ONNX Runtime** through shape specialization and tuning;
3. target **1.2-1.8x** on fused or tiny-batch workloads;
4. reserve gains above **2x** for workloads with exploitable sparsity,
   low-rank structure, very large collections of tiny matrices, or an
   unfused/materialized Attention baseline.

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
   * - Attention correctness
     - Differential tests cover stateless Attention and tensor-based
       past/present compatibility, MHA/GQA/MQA, all mask forms, causal
       boundaries, empty sequences, and every supported type.
   * - Attention memory
     - The optimized path never materializes the complete score or probability
       tensor; temporary memory is bounded by worker count and Br x Bc blocks.
   * - Attention parity
     - Median speed-up is at least 1.0x versus ONNX Runtime on the priority
       model/context corpus, with no priority case below 0.9x.
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
     - Reproducible MLAS and Attention cases in ``onnx-light``'s C++
       ``TestMode::BENCHMARK`` framework.
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
       ARM kernels PR05, parity runner PR06.0, and diagnosis PR06.1 are
       implemented or in review; measured fixes PR06.2 through PR06.5 and
       final gate PR06.6 remain.
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
       <https://github.com/xadupre/onnx-light-cpu/pull/162>`_
   * - P5
     - Native/panel-converted FP16, BF16, and integer paths.
     - Low-precision corpus reaches at least 1.0x ONNX Runtime median
       performance with no priority shape below 0.9x where the type is
       supported.
     - P3-P4.
     - Four-PR sequence fixed below; all pending.
     - Roadmap PR07 through PR10 below.
   * - P6
     - ``AttentionPlan`` and materialized tensor correctness implementation.
     - MHA/GQA/MQA, masks, causal behavior, and tensor past/present
       differential tests pass.
     - P1-P2.
     - Two-PR sequence fixed below; all pending.
     - Roadmap PR11 through PR12 below.
   * - P7
     - Online-softmax prefill Attention.
     - No full score/probability tensor; median performance is at least 1.0x
       ONNX Runtime with no priority case below 0.9x and bounded temporary
       memory.
     - P4 and P6.
     - Three-PR sequence fixed below; all pending.
     - Roadmap PR13 through PR15 below.

Remaining pull-request sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table is the single source of truth for the sequence after
``#149``. PR06 is split into independently reviewable measured steps without
renumbering PR07 through PR15. Completed rows remain visible so scope is not
lost; each row contains its complete implementation scope, merge criterion,
dependency, and current status.

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
     - Pending
   * - Roadmap PR06.3
     - Dedicated GEMV/skinny-M kernel.
     - ``M == 1`` and small-M cases stream each B row once, reuse it across
       output columns, vectorize the useful dimension, and improve every
       priority GEMV case without regressing general GEMM.
     - PR06.2
     - Pending
   * - Roadmap PR06.4
     - Use immutable plans on operator paths.
     - Registered ONNX ``Gemm`` and ``MatMul`` construct guarded
       ``GemmPlan``/``MatMulPlan`` instances during preparation and execute
       them directly. Algorithm, blocking, threads, and packed constant-B
       panels are no longer re-derived on every run; dynamic shapes rebuild or
       retrieve a correctly keyed plan, and dynamic B retains the ordinary
       per-call packing path.
     - PR06.3
     - Pending
   * - Roadmap PR06.5
     - Sustain large-matrix throughput on Zen and generic x86.
     - Measured MR/NR candidates and shape-constrained MC/NC/KC choices keep
       enough independent FMA chains and parallel panels active. The 1024³ and
       2048³ priority cases no longer regress from 512³, and the complete
       corpus shows no priority-shape regression.
     - PR06.4
     - Pending
   * - Roadmap PR06.6
     - Final FP32/FP64 parity gate.
     - Raw dedicated-machine results cover Gemm, shared MatMul, batched paths,
       every priority platform, and both types; median speed-up is at least
       1.0x ONNX Runtime and no priority case is below 0.9x. The PR remains
       open while any target fails.
     - PR06.2 through PR06.5
     - Pending
   * - Roadmap PR07
     - x86 FP16/BF16 kernel family.
     - Immutable plans describe typed panels, FP32 accumulation, conversion
       epilogues, and ISA gates without full-tensor conversion. F16C converts
       while packing; AVX-512FP16/BF16 use native kernels; CPUID and OS tile
       state safely gate AMX with AVX-512 fallbacks.
     - PR06.6
     - Pending
   * - Roadmap PR08
     - ARM FP16/BF16 kernel family.
     - NEON and available SVE/SVE2 kernels convert or compute natively during
       packing, accumulate in FP32, narrow only final output, and pass the
       complete low-precision GEMM/MatMul corpus.
     - PR05, PR07
     - Pending
   * - Roadmap PR09
     - Integer, Float8, and packed 4-bit kernels.
     - x86 VNNI/AMX and ARM dot-product paths fuse zero-point correction,
       INT32 accumulation, requantization, and schema overflow. INT32/INT64
       retain exact fallback arithmetic. Float8 formats have explicit
       decode/packing, and packed INT4/UINT4 unpack or feed native dot/tile
       instructions with exact tails.
     - PR07, PR08
     - Pending
   * - Roadmap PR10
     - Low-precision parity gate.
     - Every supported low-precision type reaches at least 1.0x ONNX Runtime
       median performance with no priority case below 0.9x where ONNX Runtime
       supports the type; all other targets publish correctness and throughput.
       The PR remains open while any target fails.
     - PR07 through PR09
     - Pending
   * - Roadmap PR11
     - Materialized Attention implementation.
     - ``AttentionPlan`` validates layouts, head geometry, scale, masks, types,
       blocks, and threads. The materialized QK-softmax-PV path supports
       boolean/additive/padding/causal masks, zero-copy GQA/MQA head mapping,
       tensor past/present, FP32/FP16/BF16, and batch/head/query scheduling.
     - PR10
     - Pending
   * - Roadmap PR12
     - Materialized Attention correctness gate.
     - The complete MHA/GQA/MQA, mask, causal, past/present, layout, empty
       sequence, and type corpus matches ONNX Runtime; the path is registered
       as fallback for combinations not handled by streaming Attention.
     - PR11
     - Pending
   * - Roadmap PR13
     - Online Attention compute engine.
     - The online recurrence matches the materialized path. SIMD Q x K kernels
       fuse scale, masks, causal bounds, and row maximum; vector exponential
       and reductions are accurate; probability x V updates output directly.
       Cache-aware Br/Bc bounds memory, while causal, window, and sparse masks
       skip absent tiles.
     - PR12
     - Pending
   * - Roadmap PR14
     - Streaming Attention scheduling and types.
     - Batch/head/query-block prefill scheduling occupies useful threads
       without nested pools. Query lengths 1 and 2-16 use dedicated decode
       algorithms for MHA/GQA/MQA with past/present. FP16/BF16 score and
       V-update kernels match the materialized fallback.
     - PR10, PR13
     - Pending
   * - Roadmap PR15
     - Final roadmap parity and memory gate.
     - Every priority prefill/decode platform/type case has bounded temporary
       memory, reaches at least 1.0x ONNX Runtime median performance, and has no
       priority case below 0.9x. The PR remains open while any target fails.
     - PR14
     - Pending

Roadmap PR15 is the final roadmap PR.
