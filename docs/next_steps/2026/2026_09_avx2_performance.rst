AVX2 Kernel Performance Follow-up
=================================

:Date: 2026-09

**in progress**

Objective
---------

The AVX-512 optimization phase is complete. The next performance work focuses
on AVX2, where several kernels still use narrower register tiles, conversion
paths, scalar tails, or scheduling defaults inherited from the portable
implementation.

The objective is to bring the priority AVX2 corpus to the same implementation
quality as the completed AVX-512 paths without regressing AVX-512, SSE2, or
portable execution. Correctness and ONNX semantics remain identical across
the runtime-selected implementations.

Measurement contract
--------------------

`#614 <https://github.com/xadupre/onnx-light-cpu/pull/614>`_ added
``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2``. Every optimization can therefore be
measured both on native AVX2 hardware and, for controlled A/B diagnosis, on
the same AVX-512 host with AVX-512 and AMX dispatch disabled.

Published decisions use:

* Release builds with the same compiler and runtime settings for both sides;
* pinned physical cores and separate process phases for onnx-light-cpu and
  ONNX Runtime;
* raw samples, medians, dispersion, CPU model, affinity, thread count, and
  detected SIMD level;
* one-thread and physical-core runs so compute, packing, memory bandwidth, and
  scheduling gaps are not conflated;
* the existing backend benchmark CLI and operator-specific parity tools.

An AVX2 change must improve a measured priority case, preserve all differential
tests and vector-tail cases, and leave the automatic AVX-512 build unchanged.
The final corpus target is at least ``1.0x`` ONNX Runtime median performance
for each priority family, with no priority case below ``0.9x``.

Current foundation
------------------

The first AVX2-specific passes are already merged:

* `#604 <https://github.com/xadupre/onnx-light-cpu/pull/604>`_ adds fused
  AVX2/FMA ``Sigmoid`` and ``Softmax`` kernels and shortens the ``BiasGelu``
  dependency chain;
* `#605 <https://github.com/xadupre/onnx-light-cpu/pull/605>`_ improves
  medium GEMM, batched MatMul, and Attention decode scheduling;
* `#608 <https://github.com/xadupre/onnx-light-cpu/pull/608>`_ adds the
  dedicated AVX2/FMA single-query Attention path and removes scalar FP32 GEMM
  tails for one through seven columns.

These changes establish the AVX2 implementations and benchmark cases, but do
not constitute a complete AVX2 parity sweep. The explicit SIMD ceiling now
makes that sweep reproducible and prevents an AVX-512-capable development
machine from hiding an AVX2 fallback.

Work sequence
-------------

.. list-table::
   :header-rows: 1
   :widths: 10 27 45 10 8

   * - Step
     - Scope
     - Exit criterion
     - Depends on
     - Status
   * - AVX2 PR01
     - Baseline and gap inventory.
     - The priority backend corpus runs with the AVX2 ceiling and publishes
       operator, type, shape, thread, and loop-family results. Native AVX2 and
       AVX2-capped measurements agree closely enough to separate ISA gaps from
       host-specific scheduling effects.
     - #614
     - Pending
   * - AVX2 PR02
     - GEMM, MatMul, and compact matrix paths.
     - FP32/FP64 register tiles, masked tails, packing, prefetch, and
       participant selection are tuned from measured gaps. FP16/BF16
       conversion and integer/packed paths avoid scalar or full-tensor
       conversion bottlenecks on the priority shapes.
     - PR01
     - Started through #605 and #608
   * - AVX2 PR03
     - Attention.
     - Decode, short-query, and prefill cases use AVX2 score and value kernels
       with productive head/query scheduling. Masks, causal bounds, GQA/MQA,
       FP16/BF16 conversion, and vector tails retain differential parity.
     - PR01, PR02
     - Started through #605 and #608
   * - AVX2 PR04
     - Activations, normalization, unary, and binary elementwise kernels.
     - Priority contiguous and broadcast cases avoid scalar tails and
       unnecessary widening, while expensive arithmetic and conversion paths
       retain their numerical contracts.
     - PR01
     - Started through #604
   * - AVX2 PR05
     - Final parity and regression gate.
     - Every priority family reaches the median and minimum targets on native
       AVX2 hardware. The same commit passes the AVX2-ceiling and automatic
       AVX-512 correctness suites without an AVX-512 performance regression.
     - PR02--PR04
     - Pending

AVX2 PR05 completes this follow-up. Architecture-specific work for AVX-512FP16,
AVX-512BF16, VNNI, AMX, NEON, or SVE remains independent and must not be used
to hide a missing AVX2 implementation.
