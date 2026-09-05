AVX2 FP32/FP64 GEMM and MatMul Gap Closure
===========================================

:Date: 2026-09
:Updated: 2026-09-05

**in progress**

Objective
---------

Issue `#633 <https://github.com/xadupre/onnx-light-cpu/issues/633>`_ asks for
the dominant AVX2 FP32/FP64 ``Gemm``/``MatMul`` gaps against ONNX Runtime to be
measured and closed under an
``-DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2`` ceiling build, covering square,
transformer-projection, batched, skinny-M, skinny-N, large-K, transpose,
bias, dynamic-B, and ``N=1..7`` tail cases, without touching AVX-512
dispatch/heuristics or compact/Attention code.

Baseline methodology and its sandbox limitation
------------------------------------------------

``tools/benchmark_gemm_parity.py`` is the canonical parity gate, but it
imports ``onnx_light`` (the sibling ``onnx-light`` checkout) to build the
reference graphs and run the ``ReferenceEvaluator``/kernel-usage instrumentation.
That checkout is not available in this sandbox, so the canonical gate could
not be executed here. To still gather evidence, this pass used two
independent, narrower substitutes on an AVX2+AVX-512-capable 4-core host:

* An AVX2-ceiling Release build
  (``-DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2 -DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=OFF``)
  linked into a standalone timing harness that calls ``GemmFloat32``/
  ``GemmFloat64`` directly (no onnx-light dependency), covering the priority
  shape corpus from ``tools/benchmark_gemm_parity.py`` at one thread and with
  a naive per-call thread-spawning executor standing in for the "available
  physical-core policy".
* A plain ``onnx`` + ``onnxruntime`` (pip) Python script building single-node
  ``Gemm`` graphs and timing ``InferenceSession.run`` with
  ``intra_op_num_threads=1`` for the same shapes.

The important caveat: upstream ONNX Runtime has no supported way to cap its
own CPU execution provider to AVX2 at runtime (confirmed - this would require
rebuilding ONNX Runtime with AVX-512 disabled at compile time). This host
supports AVX-512, so the ORT numbers below reflect ORT's AVX-512 MLAS
kernels, not an AVX2-vs-AVX2 comparison. Any gap on square/wide-projection
shapes below is therefore expected to include a genuine ISA-width component
(8 vs 16 lanes) that this issue explicitly excludes from scope ("do not
change AVX-512 dispatch"). Closing that portion would require either a
genuinely AVX2-only host or an ONNX Runtime build with AVX-512 compiled out,
neither of which this sandbox provides.

Baseline findings (AVX2 ceiling, single thread, this host)
------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 16 16 16

   * - Case
     - onnx-light-cpu f32
     - ONNX Runtime (AVX-512) f32
     - Ratio
   * - ``direct`` (32x128x16)
     - 34.4 GFLOPS
     - 16.6 GFLOPS
     - 2.07x
   * - ``tiny_dynamic`` (1x64x64)
     - 2.8 GFLOPS
     - 2.3 GFLOPS
     - 1.20x
   * - ``square_128``
     - 91.4 GFLOPS
     - 176.2 GFLOPS
     - 0.52x
   * - ``square_512``
     - 100.7 GFLOPS
     - 195.1 GFLOPS
     - 0.52x
   * - ``square_1024``
     - 101.9 GFLOPS
     - 171.3 GFLOPS
     - 0.59x
   * - ``skinny_m`` (1x1024x1024)
     - 10.2 GFLOPS
     - 14.1 GFLOPS
     - 0.72x
   * - ``skinny_n`` (1024x1x1024)
     - 14.7 GFLOPS
     - 14.1 GFLOPS
     - 1.05x
   * - ``large_k`` (32x32x4096)
     - 82.7 GFLOPS
     - 138.2 GFLOPS
     - 0.60x
   * - ``split_k`` (2x2x4096)
     - 5.0 GFLOPS
     - 2.6 GFLOPS
     - 1.96x
   * - ``trans_a`` (128x128x128)
     - 84.5 GFLOPS
     - 155.5 GFLOPS
     - 0.54x
   * - ``trans_b`` (128x128x128)
     - 94.5 GFLOPS
     - 150.5 GFLOPS
     - 0.63x
   * - ``transformer_projection`` (128x3072x768)
     - 98.5 GFLOPS
     - 163.0 GFLOPS
     - 0.60x
   * - ``transformer_down_projection`` (128x768x3072)
     - 97.9 GFLOPS
     - 178.2 GFLOPS
     - 0.55x

The single-core AVX2 microkernel (``GemmMicroKernel_AVX2FMA_F32Impl``/
``...F64Impl`` in
``onnx_light_cpu/impl/math/gemm/avx2/gemm_kernel_avx2_fma.cc``) already
reaches roughly 78-80% of this host's theoretical AVX2 FMA peak
(2 FMA ports x 8 lanes x 2 flops/FMA) on square/projection/transpose shapes,
and the float64 path shows the same efficiency against its own peak. GEMV-like
``skinny_m``/``skinny_n`` shapes are memory-bandwidth bound (streaming the
full ``B``/``A`` operand once) and already reach or exceed the measured ORT
number; ``split_k`` and ``direct``/``tiny_dynamic`` (call-overhead-dominated
tiny shapes) are already ahead of ORT here. The remaining gap is concentrated
in square/projection/transpose shapes and tracks the AVX2-vs-AVX-512 lane
width, not an under-tuned AVX2 code path: register blocking
(``kGemmAVX2MR`` = 6 rows x 16 columns), K-unrolling by 4, prefetching, and
per-microarchitecture register-row selection (Intel Core vs AMD Zen, see
``SelectGemmRegisterRowsForMicroarchitecture`` in
``onnx_light_cpu/impl/math/gemm/gemm_blocking.cc``) were already in place
before this pass.

A naive multi-thread scaling probe (thread-per-call executor, 4 physical
cores) showed close to 2x scaling on ``square_1024``/``transformer_projection``
(consistent with a partially memory-bandwidth-bound workload on this host)
and a regression for very small shapes such as ``square_128`` whose serial
runtime (about 47 microseconds) is smaller than the thread-creation/join cost
of the naive harness executor. That regression is judged to be a benchmark-
harness artifact (the real embedding runtime supplies its own persistent
thread pool, and ``SelectGemmParticipantCount`` already caps ``square_128``
at 2 participants, not 4) rather than evidence of a fixable admission-control
bug, so no participant-limit change was made without a persistent-pool
measurement to confirm it.

Status
------

No AVX2-specific microkernel, packing, blocking, or scheduling change is
included in this pass: the measured evidence available in this sandbox does
not demonstrate a fixable AVX2 regression separate from the AVX2-vs-AVX-512
lane-width difference, and the issue's own instruction is to "optimize only
bottlenecks demonstrated by the baseline." ``test_gemm_kernel`` and
``test_gemm_plan`` continue to pass unchanged under the AVX2 ceiling build.

Remaining work requires the dedicated-machine setup used by prior roadmap
items (see :doc:`the Gemm/MatMul roadmap <2026_08_gemm_matmul>`): an
``onnx-light`` checkout to run ``tools/benchmark_gemm_parity.py`` directly,
and either a genuinely AVX2-only host or an ONNX Runtime build with AVX-512
compiled out, so the reported ratios reflect an AVX2-vs-AVX2 comparison
instead of AVX2-vs-AVX-512.
