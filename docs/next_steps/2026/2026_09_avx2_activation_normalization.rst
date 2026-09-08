AVX2 Activation and Normalization Gap Closure
==============================================

:Date: 2026-09
:Updated: 2026-09-08

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

.. warning::

    This September 5 direct-kernel comparison did not establish registered
    runtime parity. The later :doc:`2026_09_avx2_performance` follow-up found
    unreachable fused activation dispatch and end-to-end gaps, including
    substantial multithread overhead. Use isolated registered-runtime
    measurements for parity decisions.

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

September 7 registered-runtime follow-up
----------------------------------------

After the dispatch correction in
`#653 <https://github.com/xadupre/onnx-light-cpu/pull/653>`_, a focused pass
improves the existing AVX2/FMA activation implementation:

* Sigmoid uses vector division instead of reciprocal approximation,
  Newton refinement, and a separate zero-exponential correction. Two
  independent eight-lane groups share the same helper with masked tails.
  Positive-infinity saturation remains exactly one.
* The negative exponential helper is explicitly inlined and omits its
  redundant positive-input comparison: its only callers pass negative
  absolute values or max-subtracted Softmax values. The polynomial and
  subnormal/NaN/infinity fallback are unchanged.
* AVX2 FP32 Softmax keeps inputs below 1 MiB serial instead of dispatching
  workers from 128 KiB. Larger inputs retain runtime-owned row parallelism.
  The ISA-dependent threshold is cached, avoiding CPUID on every call.
  Other ISA/type schedules and Sigmoid scheduling are unchanged.

The baseline was rebuilt from ``66bf6ed`` on a native AVX2 Core i7-13800H,
Windows/MSVC Release, Python 3.13 and ORT 1.29. Baseline and modified native
modules were archived separately; worker imports pin and verify their paths
and hashes. Each existing backend fixture was serialized once and reused
unchanged. No expanded input or output computation was moved outside timing.

For each case, separate processes ran baseline CPU, modified CPU, ORT,
modified CPU, then baseline CPU. Each process exited before the next started.
The table uses the median of the two per-process CPU medians, with 30 warmups,
up to 1,000 samples and a 0.25 s sampling budget per phase. Affinity masks
were ``0x10`` for one thread and ``0x555`` for six threads.
All latencies are milliseconds; **ORT/CPU below one means CPU remains slower**.

.. list-table::
   :header-rows: 1

   * - FP32 case
     - Threads
     - Before
     - After
     - Before/after
     - ORT/CPU
   * - Sigmoid, 65,536
     - 1
     - 0.08065
     - 0.06195
     - 1.30x
     - 0.84x
   * - Sigmoid, 1,048,576
     - 1
     - 1.12288
     - 1.03160
     - 1.09x
     - 0.63x
   * - Sigmoid, 1,048,576
     - 6
     - 0.27802
     - 0.20438
     - 1.36x
     - 0.46x
   * - Sigmoid, 4,194,304
     - 6
     - 2.07120
     - 1.25253
     - 1.65x
     - 1.62x
   * - Softmax, 32 x 1,024
     - 6
     - 0.05205
     - 0.01845
     - 2.82x
     - 0.81x
   * - Softmax, 1,024 x 1,024
     - 1
     - 0.81262
     - 0.50930
     - 1.60x
     - 0.97x
   * - Softmax, 1,024 x 1,024
     - 6
     - 0.34985
     - 0.34123
     - 1.03x
     - 0.46x

These are shared-host diagnostics, not a completed parity gate. Earlier
whole-corpus-before/after phases showed substantial frequency/load variation;
the paired protocol reduces but does not eliminate it. In particular,
the 1.03x large multithread Softmax change is not an established improvement.
A longer tiny-shape confirmation (1,000 warmups, up to 20,000 samples or 1 s)
measured Sigmoid/1,024 at 0.00860 ms for both variants and Softmax/1x1,024
at 0.00880 versus 0.00860 ms, resolving an apparent short-run regression.
Alternating direct-kernel measurements also support a roughly 1.4x Sigmoid
compute improvement, but do not replace the registered-runtime results.

Regression coverage includes exact dispatch, aliasing, every short Sigmoid
tail, subnormal-range inputs, mixed NaN/infinite Softmax rows, and serial
versus reverse-order executor equivalence on a large tail-bearing tensor.
It also checks the new serial threshold, bounded participant counts, and
nested-dispatch suppression.

Remaining priority cases
-------------------------

The direct-kernel comparison above must not be interpreted as registered
``Sigmoid`` or ``Softmax`` parity. The full dispatch, scheduling, and tensor
allocation path must be measured with isolated runtime processes. Large
multithread activations and the remaining BFloat16 and Float16 ``Softmax``
loop families still require end-to-end parity coverage.

InstanceNormalization affine follow-up
-------------------------------------

After the SIMD moments improvement in `#667
<https://github.com/xadupre/onnx-light-cpu/pull/667>`_, FP32
``InstanceNormalization`` still applied its channel multiplier and offset
through the baseline inline loop. The affine pass now uses cached AVX2/FMA
or AVX-512 dispatch for tensors containing at least 16,384 elements and
spatial slices containing at least 32 elements. Smaller tensors retain the
inline loop; selection happens once per tensor rather than once per slice.
Other normalization operators, floating-point types, moments calculations,
and executor scheduling are unchanged.

The new scalar-parameter affine helper preserves a separately rounded
multiply and add. FMA contraction is disabled for this helper, not for the
existing moments or RMS kernels: contraction can turn an exactly cancelling
constant-input result into a nonzero value. Coverage includes all short
vector tails, unaligned and in-place buffers, NaN/infinity, large constants,
and serial, executor, and nested execution.

The following registered-runtime measurements used an Intel Xeon Platinum
8480C and 96 configured runtime threads, comparing against the normalization
implementation in main ``4de4b9c`` (including #667). Each value is the median
of two run medians, with 200 warmups, up to 1,000 samples, and a one-second
budget per phase. Baseline/candidate order was reversed in the second pair.
Latencies are seconds and speedups compare against main, not ONNX Runtime.

.. list-table::
   :header-rows: 1

   * - FP32 shape
     - ISA ceiling
     - Before
     - After
     - Before/after
   * - [8, 16, 128]
     - AVX-512
     - 0.000008405
     - 0.000006273
     - 1.34x
   * - [4, 32, 16, 16]
     - AVX-512
     - 0.000012202
     - 0.000009424
     - 1.29x
   * - [8, 16, 128]
     - AVX2
     - 0.000006756
     - 0.000006186
     - 1.09x
   * - [4, 32, 16, 16]
     - AVX2
     - 0.000011432
     - 0.000008970
     - 1.27x

The two 8,192-element benchmark shapes retain the inline path. Their
AVX-512 before/after ratios were 0.94x and 0.98x, and their AVX2 ratios
were 1.00x and 1.02x; no speedup is claimed for those shapes. Single-thread
comparisons also improved the two target shapes, with the small shapes
remaining within approximately 4% of baseline. A small-worker-team
experiment was rejected because it did not improve consistently over the
SIMD affine change alone.

Reproduce with identically configured builds on both revisions:

.. code-block:: bash

   python -m onnx_light_cpu benchmark --dtype float32 --onnxruntime \
       --test '^test_cpu_instancenormalization_' --threads 96 \
       -r 1000 -w 200 -t 1 -o instancenormalization.xlsx

Repeat with an AVX2-ceiling build, and with ``--threads 1`` for the serial
comparison. ONNX Runtime on this AVX-512 host is not itself capped to AVX2.
