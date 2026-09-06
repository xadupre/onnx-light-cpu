AVX2 Isolated-Runtime Diagnostic Baseline
=========================================

:Date: 2026-09-06

**Diagnostic measurements; parity gate still pending**

This local follow-up to :doc:`2026_09_avx2_performance` identifies matrix and
Attention workloads that remain behind ONNX Runtime after the first AVX2
passes. It also exposes a measurement bias in the shared-process benchmark.
No production kernel or benchmark implementation changes accompany this report.

Environment and build provenance
---------------------------------

* Intel Core i7-13800H under WSL2, with process affinity
  ``0,2,4,6,8,10,12,14,16,18`` and explicit one-thread and ten-thread policies.
  The guest topology does not establish which cores are native P-cores.
* AVX2, FMA, F16C and AVX-VNNI are available; AVX-512 is absent.
  ORT's use of AVX-VNNI was not controlled. In particular, the INT8 results
  must not be presented as a strict AVX2-only instruction-set comparison.
* ONNX Runtime 1.28.0, Python 3.12.3, GCC 14.2.0, Release build with
  ``-O3 -DNDEBUG`` and ``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2``.
* The original broad sweep used CPU binaries built from ``f2d9c6c``.
  Main advanced after that build; its generated report incorrectly records
  ``292bfb8`` because the script captures HEAD at report-generation time,
  not the source revision used to compile the binaries.
* All isolated measurements and spinning-control experiments below used a
  fresh build from ``292bfb8``, including the FP16 RMSNormalization change
  in `#641 <https://github.com/xadupre/onnx-light-cpu/pull/641>`_.
* The linked onnx-light source checkout was
  ``8021ad3b3a6496046cac3cf5b7c4cb9a30ff6c1c``. Missing Python distribution
  metadata in the original report did not mean that the runtime was absent.
  Source-built CPU extension paths and AVX2 dispatch were checked explicitly.

This is a shared-environment diagnostic, not a dedicated-runner certification.
All times are end-to-end backend-model timings, not isolated microkernel
throughput measurements.

Why the original sweep cannot establish parity
----------------------------------------------

The initial ``benchmark_avx2_parity.py`` sweep produced 1,140 comparable
case/thread-policy combinations and 260 unsupported comparisons. The same
cases were measured with two thread policies; these are not 1,140 distinct
models.

The unsupported comparisons comprise:

* 224 Pow implementation rejections by ORT;
* 22 Attention cache/output-contract rejections;
* 12 opset-27 rejections;
* two Attention precision/type failures.

The runner saved its reports and returned status 1 because the comparison
corpus was incomplete. Unsupported comparisons are not evidence of parity
and must remain visible.

More importantly, ``_measure_case`` keeps both runtimes' thread pools alive
at the same time. Separate timing phases and alternating the first runtime
do not isolate those pools. With the initial 100 ms sampling budget, idle
ORT spinning substantially affected the CPU measurements.

The control experiment retained ten threads, five warmups, at most 100
samples and a 100 ms sampling budget:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Case
     - CPU latency with ORT spinning, two runs
     - CPU latency without ORT spinning, two runs
   * - Abs, 4,194,304 FP32 values
     - 4.21 / 6.41 ms
     - 2.46 / 2.32 ms
   * - Gemm, 512 square FP32
     - 5.01 / 6.14 ms
     - 2.90 / 2.00 ms

Disabling spinning also changes ORT's own timings, so that is not the proposed
fairness fix. The subsequent measurements instead ran the two runtimes in
separate, sequential processes, preserving their normal scheduling policies.
For comparison, isolated Abs CPU latency was approximately 0.44--0.48 ms
with ten threads.

The original multithread family medians and rankings must therefore not be
used as final parity evidence.

Isolated measurement protocol
------------------------------

The diagnostic driver collected existing backend cases with
``TestMode.BENCHMARK`` and ``generate_benchmark_expected_outputs=False``.
Both runtimes received the same backend models and their generated inputs:

* the CPU process used the existing ``_measure_case`` with
  ``with_onnxruntime=False``, including its assertion that the registered CPU
  kernel executed;
* the ORT process used ``CPUExecutionProvider``, ``ORT_SEQUENTIAL``,
  ``inter_op_num_threads=1`` and the matching explicit intra-op thread count;
* no two measurement processes ran concurrently, and no kernel arithmetic
  was reimplemented.

The first isolated pass covered 15 cases under both thread policies, requesting
20 warmups and at most 1,000 samples or one second per case. The CPU helper
also caps its warmup phase at the per-case time budget; ORT ran all 20 warmups.
The confirmation pass covered nine cases with a two-second budget and
reversed runtime order:
ORT then CPU for one thread, CPU then ORT for ten threads. Sampling stopped
when either the count or time limit was reached.

These timings are not an independent correctness sweep. Raw samples,
unsupported-case diagnostics, build logs and the diagnostic scripts were
retained locally; generated artifacts are not committed, as required by the
roadmap. The tables below are a curated report rather than a replacement
for those raw samples.

Confirmed matrix and Attention gaps
-----------------------------------

The table reports the confirmation pass. **Ratio = ORT median / CPU median**:
a ratio below 1 means onnx-light-cpu is slower.

.. list-table::
   :header-rows: 1
   :widths: 36 22 10 22 10

   * - Workload
     - One thread, CPU / ORT ms
     - Ratio
     - Ten threads, CPU / ORT ms
     - Ratio
   * - MatMul FP16, 512 square
     - 8.264 / 2.974
     - 0.360
     - 4.653 / 1.091
     - 0.234
   * - MatMul FP32, M=1, K=N=4096
     - 11.367 / 2.748
     - 0.242
     - 3.132 / 1.680
     - 0.536
   * - Qwen gate/up FP16, M=1, K=4096, N=12288
     - 59.753 / 25.831
     - 0.432
     - 85.366 / 11.140
     - 0.131
   * - Gemm FP32, 512 square
     - 3.023 / 2.812
     - 0.930
     - 1.679 / 1.045
     - 0.623
   * - Gemm FP64, 512 square
     - 6.449 / 6.012
     - 0.932
     - 5.727 / 2.044
     - 0.357
   * - MatMulInteger UINT8 x INT8, 512 square
     - 4.250 / 0.732
     - 0.172
     - 2.504 / 0.246
     - 0.098
   * - MatMulInteger UINT8 x UINT8, 512 square
     - 4.266 / 2.083
     - 0.488
     - 2.665 / 0.868
     - 0.326
   * - Attention FP32, Q128/KV8192/H64, 12 heads
     - 103.951 / 45.422
     - 0.437
     - 20.001 / 14.409
     - 0.720
   * - Abs FP32, 4,194,304 values
     - 1.928 / 1.773
     - 0.919
     - 0.480 / 0.409
     - 0.851

Some multithread results remain noisy: CPU p90/p10 reaches approximately 2.9,
and the Attention ten-thread ratio varied from 0.465 to 0.720 between passes.
Qwen gate/up was added only in the confirmation pass; its negative scaling
needs a dedicated repeatability study. The large LM-head cases from the
initial short-budget sweep must not be extrapolated from single samples.

The exact confirmation fixtures are:

.. code-block:: text

    test_cpu_abs_n4194304_float32_benchmark
    test_cpu_matmul_square_512_float16_benchmark
    test_cpu_matmul_skinny_m_float32_benchmark
    test_cpu_matmul_llm_qwen3_8b_gate_up_m1_k4096_n12288_float16_benchmark
    test_cpu_gemm_square_512_float32_transA_0_transB_0_bias_none_benchmark
    test_cpu_gemm_square_512_float64_transA_0_transB_0_bias_none_benchmark
    test_cpu_matmulinteger_square_512_uint8xint8_benchmark
    test_cpu_matmulinteger_square_512_uint8xuint8_benchmark
    test_cpu_attention_opset23_rank4_mha_q128_kv8192_hd64_none_stateless_float32_benchmark

Supplemental coverage
---------------------

The fixed baseline omits RMSNormalization, BiasGelu, SwiGLU, FP64 matrices
and compact integer matrix multiplication. Selected instances were added
to the isolated pass rather than declaring those families covered.

.. list-table::
   :header-rows: 1
   :widths: 60 20 20

   * - Workload
     - One-thread ratio
     - Ten-thread ratio
   * - RMSNormalization FP32, 128x4096
     - 3.47
     - 3.08
   * - RMSNormalization FP16, Qwen hidden4096, S1
     - 2.44
     - 2.09
   * - RMSNormalization FP16, Qwen hidden4096, S128
     - 10.96
     - 4.33
   * - BiasGelu FP32, 256x4096
     - 2.36
     - 1.30
   * - BiasGelu FP16, 256x4096
     - 2.42
     - 1.45

The SwiGLU FP32/FP16 fixtures use opset 28, which ORT 1.28 rejects.
No schema was silently downgraded. Integer/bool binaries, INT4 and full-model
decode remain outside the isolated follow-up scope.

Implementation priorities
-------------------------

1. Repair the shared-process benchmark protocol before using it as a parity
   gate. Isolate runtime worker pools without changing normal execution
   policies, and record the actual compiled revision rather than only HEAD.
2. Investigate AVX2 FP16 matrix paths, especially M=1/Qwen, including packing,
   conversion and scaling costs. Repeat the Qwen case before selecting a fix.
3. Investigate MatMulInteger UINT8 x INT8 and UINT8 x UINT8. Separate VNNI
   capability differences from packing, correction and scheduling overhead.
4. Optimize FP32 M=1 and multithread FP64 matrix paths from isolated timings.
5. Continue long-context Attention work; the one-thread prefill gap is clear.
6. Do not prioritize RMSNormalization/BiasGelu over these gaps based on the
   selected cases, but do not infer that every normalization or activation
   workload is finished.

The full isolated corpus, differential correctness runs and dedicated native
AVX2 acceptance gate remain pending.
