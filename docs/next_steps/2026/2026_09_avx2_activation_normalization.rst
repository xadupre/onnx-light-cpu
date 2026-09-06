AVX2 Activation and Normalization Gap Closure
==============================================

:Date: 2026-09
:Updated: 2026-09-05

**in progress**

Objective
---------

Measure and close remaining AVX2 gaps (``-DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2``)
for the transformer activation and normalization kernels used by Qwen-shaped
workloads: ``Sigmoid``, ``Softmax``, ``BiasGelu``, and ``RMSNormalization``.
This follows the completed AVX2 activation work in `#604
<https://github.com/xadupre/onnx-light-cpu/pull/604>`_ (fused AVX2/FMA
``Sigmoid``/``Softmax``, independent Horner chains for ``BiasGelu``) and
extends it only where a fresh measurement shows a remaining bottleneck.

Measured baseline
------------------

A direct AVX2-ceiling Release build (no onnx-light integration required) was
used to microbenchmark every targeted kernel entry point at small-row,
transformer-hidden (896/1536/2048/3584/4096), and large-contiguous widths, and
those numbers were cross-checked against ONNX Runtime 1.29 single-node CPU
latency (``intra_op_num_threads=1``) for the same shapes:

* ``Sigmoid``/``Softmax``/``BiasGelu`` FP32 AVX2/FMA kernels already run
  compute-bound near 1.5-2.2 elements/ns and comfortably beat the measured
  ONNX Runtime single-node latency at every sampled width; no regression or
  remaining gap below ``0.9x`` was found for these paths on the development
  host.
* ``RMSNormalization`` FP32 (``normalization_kernel_avx2_fma.cc``) and
  BFloat16 (``rms_normalization_bfloat16_avx2_fma.cc``) already accumulate
  their mean-square reduction across four independent FMA/multiply-add
  vectors, shortening the reduction's dependency chain.
* ``RMSNormalization`` Float16 (F16C, ``rms_normalization_kernel_avx2_f16c.cc``)
  was the one remaining outlier: its reduction used a single accumulator, so
  the multiply-add chain serialized across the whole row instead of letting
  the out-of-order engine overlap independent accumulators. Isolating the
  reduction phase showed roughly a 3x throughput gap versus the same
  four-accumulator pattern already used by the FP32 and BFloat16 paths.

Change
------

``RmsNormalizationFloat16_F16C`` now accumulates its mean-square reduction
across four independent vectors (32 half-precision elements per outer
iteration, matching the FP32/BFloat16 stride), keeping the existing 8-wide
tail loop, F16C narrow/widen conversions, epsilon handling, and per-lane NaN
fallback in the affine pass unchanged. This translation unit is compiled with
``-mavx -mf16c`` only (no FMA), so the accumulation uses a separate multiply
and add rather than ``_mm256_fmadd_ps``.

Before/after (development host, single call, AVX2-ceiling build)
------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - Width
     - Before (elements/ns)
     - After (elements/ns)
     - Speedup
   * - 64
     - 1.71
     - 1.71
     - 1.00x
   * - 256
     - 2.45
     - 3.33
     - 1.36x
   * - 896
     - 2.83
     - 4.17
     - 1.47x
   * - 1536
     - 2.93
     - 4.42
     - 1.51x
   * - 2048
     - 2.95
     - 4.50
     - 1.53x
   * - 3584
     - 3.02
     - 4.64
     - 1.54x
   * - 4096
     - 3.02
     - 4.51
     - 1.49x

The width-64 case (below the 32-wide unroll threshold) is unaffected, showing
no priority regression outside measurement dispersion. Every wider Qwen-shaped
case measured improves by 36% to 54%.

Validation
----------

* New ``unittests/cc/math/test_rms_normalization_kernel.cc`` differentially
  checks ``RmsNormalizationFloat32``, ``RmsNormalizationFloat16``,
  ``RmsNormalizationBFloat16``, and the direct
  ``RmsNormalizationFloat16_F16C`` entry point against a reference
  implementation across widths that straddle the 8/32-wide unroll boundaries
  (1, 7, 8, 31, 32, 33, 64, 896), multiple independent rows, and an infinite
  input (verifying the NaN/epsilon contract is preserved).
* The full C++ test suite (447 cases) passes with
  ``-DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2``.
* A second configuration build with automatic dispatch (no forced SIMD
  ceiling, native compiler flags) confirms the change compiles and passes
  without the AVX2/FMA compile flags available to the AVX2-ceiling build,
  since this translation unit only requests ``-mavx -mf16c``.

Remaining priority cases
-------------------------

No registered ``Sigmoid``, ``Softmax``, ``BiasGelu``, or ``RMSNormalization``
FP32 case was found below ``0.9x`` ONNX Runtime on the development host using
the direct-kernel/ONNX-Runtime-single-node comparison described above. A full
backend-corpus run through ``onnx-light-cpu benchmark --onnxruntime`` (which
requires the onnx-light Python integration) is needed to confirm this holds
through the complete registered-kernel dispatch, scheduling, and tensor
allocation path, and to check the remaining BFloat16 and Float16 ``Softmax``
low-precision loop-family cases end to end.
