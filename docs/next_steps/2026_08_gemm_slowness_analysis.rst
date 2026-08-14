Why Gemm Is Slower Than ONNX Runtime Today
==========================================

:Date: 2026-08

**analysis**

This note answers a direct question from `onnx-light-cpu #161
<https://github.com/xadupre/onnx-light-cpu/issues/161>`_: *why is the current*
``Gemm`` *slow compared to ONNX Runtime, and can the* ``GemmPlan`` *be
trusted?* It measures the shipped kernel on a fixed machine, attributes the
gap to specific code paths, and maps each cause to the existing
:doc:`Gemm, MatMul, and Attention roadmap <2026_08_gemm_matmul>`. It changes
no kernel code; it locates the overhead first, as required by roadmap step
PR06.1 ("measure ... before changing kernel code").

Short answer
------------

Three findings explain most of the gap:

#. **The** ``GemmPlan`` **you should not trust is not on the hot path.** The
   registered ONNX ``Gemm`` kernel never constructs a ``GemmPlan``. It calls
   ``GemmFloat32WithEpilogue`` → ``GemmFloat32``, which re-runs
   ``SelectGemmAlgorithm`` and re-derives the cache blocking **on every call**.
   ``GemmPlan`` / ``MatMulPlan`` are compiled and unit-tested but are dead code
   for the operator, so the "plan once, run many times" benefit described in
   the roadmap is not realized yet.
#. **Two priority shapes fall onto weak kernels.** ``N == 1`` (skinny-N) uses a
   fully **scalar** triple loop, and ``M == 1`` (GEMV / skinny-M) reuses the
   register-blocked tile with a single active row. Both run at a small fraction
   of the vectorized kernels.
#. **The general kernel does not sustain throughput as the matrix grows.**
   Single-thread FP32 throughput peaks around a 512³ shape and then *drops* for
   1024³ and 2048³, i.e. blocking/scheduling is not keeping panels resident the
   way MLAS does.

Measurement setup
-----------------

The impl-level driver (``onnx_light_cpu/impl/math/gemm_kernel.cc`` plus the
``-mavx2 -mfma`` micro-kernel) was compiled at ``-O3`` and called directly
through ``GemmFloat32`` (``alpha = 1``, ``beta = 0``, no transpose, no bias),
so the numbers isolate the multiplication from the ONNX adapter and
``ReferenceEvaluator``. Each shape is warmed up three times; the best of many
timed repetitions is reported.

* CPU: AMD EPYC 7763 (Zen 3), ``avx2`` + ``fma``, no ``avx512``.
* 4 logical CPUs available; ~3.24 GHz observed.
* Selected FP32 configuration (from ``SelectGemmBlocking`` /
  ``SelectGemmRegisterRows``): ``MR = 4``, ``NR = 2`` vectors (16 lanes),
  ``MC = 212``, ``NC = 1024``, ``KC = 304``.

The single-core FMA peak is ``2 FMA units x 8 lanes x 2 flop x 3.24 GHz`` ≈
**104 GFLOP/s** for FP32. Read the table against that ceiling.

Measured FP32 throughput
------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 14 18 18 24

   * - Shape (M×N×K)
     - Algorithm
     - 1 thread (GFLOP/s)
     - 4 threads (GFLOP/s)
     - Note
   * - 128×128×128
     - General
     - 43.7
     - 47.5
     - small; does not scale
   * - 512×512×512
     - General
     - 62.7
     - 135.4
     - best single-thread (~60% peak)
   * - 1024×1024×1024
     - General
     - 43.0
     - 82.1
     - **regresses** vs 512³
   * - 2048×2048×2048
     - General
     - 43.2
     - 88.0
     - stuck at ~41% of peak
   * - 1×1024×1024
     - SkinnyM (GEMV)
     - 1.28
     - 7.10
     - bandwidth-starved
   * - 1024×1×1024
     - SkinnyN
     - 2.18
     - 2.17
     - **scalar**; no SIMD, no scaling
   * - 32×32×4096
     - General
     - 41.8
     - 53.1
     - large-K, tiny output
   * - 128×3072×768
     - General
     - 40.5
     - 86.6
     - transformer projection

Root causes, tied to the code
------------------------------

Scalar skinny-N (``N == 1``)
    ``SelectGemmAlgorithm`` routes ``n <= vector_lanes`` to
    ``GemmAlgorithm::kSkinnyN``, and ``GemmSkinnyN`` in
    ``gemm_kernel.cc`` is a plain ``for m / for n / for k`` scalar reduction
    with no SIMD and no K-vectorization. At 2.18 GFLOP/s it is ~20x below the
    general kernel, and it does not improve with more threads. This is exactly
    the "replace the scalar skinny-N K reduction with a vectorized path" item
    in roadmap PR06.2.

Weak GEMV / skinny-M (``M == 1``)
    ``M <= MR`` selects ``GemmAlgorithm::kSkinnyM``, which packs A and calls the
    register-blocked tile with ``mr = M = 1``. With a single active row the tile
    keeps only one row of accumulators and streams all of B once, so the shape
    is bandwidth-bound: 4 MiB of B read in ~1.6 ms is ~2.5 GB/s, far below DRAM
    bandwidth. There is no dedicated GEMV kernel that vectorizes the K reduction
    and reuses each loaded B row across several output columns (roadmap
    "``M == 1`` ... GEMV/skinny-M kernel").

General kernel does not sustain throughput
    Single-thread FP32 goes 62.7 → 43.0 → 43.2 GFLOP/s across 512³/1024³/2048³.
    A cache-blocked GEMM should hold roughly constant GFLOP/s across these
    sizes. Two structural reasons stand out:

    * ``MR = 4`` with ``NR = 2`` vectors keeps only **8** of the 16 YMM
      registers live as accumulators. That is not enough independent FMA chains
      to hide the ~4-cycle FMA latency on Zen 3, capping the inner loop near
      ~60% of peak even in the best case. MLAS uses wider register tiles
      (e.g. 6×2) here. ``SelectGemmRegisterRowsForMicroarchitecture`` only
      widens ``MR`` beyond 4 for ``kIntelCore``; Zen and generic x86 stay at 4.
    * ``NC = 1024`` means any ``N <= 1024`` forms a **single** column panel, so
      wide-N parallelism collapses to row panels only, and the packed
      ``KC×NC`` B panel (≈1.2 MiB) is far larger than the 512 KiB per-core L2 —
      it is streamed from L3 for every row block. The fixed cache-derived
      blocking is not re-tuned per shape.

Per-call selection instead of a plan
    Because the operator path re-selects the algorithm and re-computes
    ``ConstrainGemmBlockingForTasks`` on every ``Run``, none of the plan's
    prepared state (algorithm, blocking, thread count, prepacked constant B) is
    reused. The branch cost itself is negligible (< 1%, as the roadmap notes),
    but the *missed* opportunities — prepacking a constant B initializer once
    and choosing a shape-specialized algorithm ahead of time — are real and are
    the entire reason ``GemmPlan`` exists. Wiring the operator through
    ``GemmPlan`` / ``MatMulPlan`` is prerequisite roadmap work, not an
    afterthought.

FP16 / BF16 widen the whole tensor
    ``GemmCompute`` widens A, B and C to fresh ``float32`` buffers, runs
    ``GemmFloat32``, then narrows the whole result. It is correct but adds full
    O(M·K + K·N + M·N) memory passes that MLAS avoids by converting during
    packing (roadmap Phase 3).

What ONNX Runtime / MLAS does differently
-----------------------------------------

* Dedicated GEMV and dot-product kernels for ``M == 1`` and ``N == 1`` instead
  of degenerate register tiles or scalar loops.
* Wider register microtiles sized to the target microarchitecture, keeping the
  FPU busy across the whole K reduction.
* Prepacked constant weights and a prepared plan, so packing and selection are
  paid once, not per call.
* Native low-precision paths that convert while packing.

Recommended order of fixes
--------------------------

These are already captured by the roadmap; this analysis fixes their priority
from measured impact:

#. Vectorize ``GemmSkinnyN`` (largest relative gap, isolated change).
#. Add a real GEMV path for ``M == 1`` that reuses each B row across columns.
#. Route the ONNX ``Gemm`` / ``MatMul`` operators through ``GemmPlan`` /
   ``MatMulPlan`` so constant B is prepacked and selection is amortized.
#. Revisit ``MR`` and cache blocking for Zen/generic x86 so large square shapes
   stop regressing; verify against the parity corpus before adopting.

Reproducing
-----------

Run the end-to-end parity gate against ONNX Runtime with
``tools/benchmark_gemm_parity.py`` (see :doc:`../benchmarks`). The isolated
per-shape throughput above comes from calling ``GemmFloat32`` directly on the
priority corpus shapes, which removes the ONNX adapter and evaluator from the
measurement.
