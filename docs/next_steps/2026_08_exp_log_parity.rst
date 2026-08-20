Exp and Log ONNX Runtime Parity Roadmap
=======================================

:Date: 2026-08

**in progress**

Objective
---------

The objective is float32 performance parity with the ONNX Runtime CPU
execution provider for the ``Exp`` and ``Log`` kernels, without weakening
their numerical contract. Parity means a median speed-up of at least ``1.0x``
over the priority corpus, no priority case below ``0.9x``, and at least
``0.95x`` on the 4,194,304-element benchmark models.

This roadmap is the focused implementation plan for the existing kernels. The
broader :doc:`Unary Elementwise Performance Roadmap
<2026_08_unary_elementwise>` covers the common unary engine and additional
operators. The :doc:`Processor-Aware Tuning roadmap
<2026_08_elementwise_kernel_tuning>` covers persistent processor-specific
scheduling profiles after the serial kernels and benchmark evidence are
sound.

Measured baseline
-----------------

The `2026-08-19 dashboard snapshot
<https://xadupre.github.io/dashboard/onnx-light-cpu/examples-benchmark.html>`_
was recorded on an AVX2 runner with onnx-light-cpu 0.1.16 and ONNX Runtime
1.29.0. It shows two different situations:

.. list-table::
   :header-rows: 1
   :widths: 17 18 18 18 29

   * - Operator
     - Elements
     - onnx-light-cpu
     - ONNX Runtime
     - Speed-up
   * - ``Exp``
     - 1,000,000
     - 0.570 ms
     - 0.225 ms
     - ``0.40x``
   * - ``Exp``
     - 4,194,304
     - 2.234 ms
     - 0.858 ms
     - ``0.38x``
   * - ``Exp``
     - 10,000,000
     - 8.323 ms
     - 2.257 ms
     - ``0.27x``
   * - ``Log``
     - 1,000,000
     - 0.716 ms
     - 0.937 ms
     - ``1.31x``
   * - ``Log``
     - 4,194,304
     - 4.058 ms
     - 3.877 ms
     - ``0.96x``
   * - ``Log``
     - 10,000,000
     - 12.856 ms
     - 9.055 ms
     - ``0.70x``

Small tensors are already faster than ONNX Runtime. ``Exp`` therefore needs a
new compute kernel rather than runtime-overhead work. ``Log`` is much closer
to parity and first needs stable scheduling and instruction-level tuning.

Confirmed technical gaps
------------------------

AVX2 without FMA
~~~~~~~~~~~~~~~~

``onnx_light_cpu/impl/math/exp_log_kernel.cc`` is compiled with the baseline
``-mavx2`` option. Its AVX2 Horner chains emit separate multiply and add
instructions. In contrast, ONNX Runtime dispatches float32 ``Exp`` to an MLAS
FMA3 kernel on capable x86 processors. That kernel combines range reduction
and polynomial evaluation with fused multiply-add instructions and uses two
scale factors to reconstruct the complete float32 exponent range.

The first optimized implementation must therefore be a separate AVX2+FMA
translation unit, selected only when both features are available. Enabling
``-mfma`` for the complete baseline library is not acceptable because runtime
dispatch must keep binaries usable on AVX2 processors without FMA.

Incorrect SIMD subnormal behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current SIMD ``Exp`` clamps its lower range near ``-88.38`` and constructs
one exponent scale from the biased IEEE exponent field. Values such as
``exp(-90)`` and ``exp(-100)`` incorrectly become zero even though the
float32 result is subnormal. The true non-zero range extends to approximately
``-103.97``.

The SIMD ``Log`` path clamps every positive subnormal to the smallest positive
normal value before exponent extraction. A full SIMD vector therefore returns
the same result for distinct subnormal inputs. Scalar tail elements call
``std::log`` and return the correct values, so results currently depend on
array length and lane position.

Existing tests cover ordinary ranges and special values but do not exercise
these vectorized subnormal cases. Correctness must be fixed before comparing
alternative approximations.

Uncalibrated shared scheduling cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both operators use ``kExpLogCostPerElement = 20``. This makes
``ParallelFor`` request all available participants from relatively small
tensors even though ``Exp`` and ``Log`` have different instruction costs and
crossovers. Local isolated measurements also show that increasing from two to
all available participants can regress ``Exp`` through several million
elements.

ONNX Runtime uses independent cost estimates for these operators. The new
benchmark must determine separate minimum block sizes and useful participant
counts. Persistent processor-specific calibration remains follow-up work in
the processor-aware tuning roadmap.

Benchmark and correctness contract
----------------------------------

Optimization starts with an isolated C++ throughput driver for
``ExpFloat32`` and ``LogFloat32``. It writes into preallocated output and
reports cycles per element, median time, dispersion, selected ISA, and
participant count. A separate end-to-end runner retains allocation,
``ReferenceEvaluator``, and ONNX Runtime session costs.

Both layers must:

* use identical tensors and alternating candidate order;
* cover sizes ``10^2`` through ``10^8`` and the 4,194,304-element benchmark
  model;
* run with 1, 2, 4, physical-core, and configured session thread counts;
* record CPU topology, affinity, compiler flags, ISA, raw samples, and median;
* compare equal-thread configurations diagnostically while preserving normal
  ONNX Runtime settings for the published headline result;
* distinguish compute, worker dispatch, output allocation, and runtime
  bookkeeping.

The differential corpus must cover:

* dense ordinary ranges used by the dashboard;
* every SIMD tail length and deliberately unaligned buffers;
* positive and negative zero, infinities, and NaNs;
* the complete normal/subnormal boundaries for ``Exp`` and positive
  subnormals for ``Log``;
* overflow and underflow transition neighborhoods;
* random bit patterns plus monotonic sweeps around range-reduction boundaries;
* float64, float16, and bfloat16 regression coverage even though initial
  performance parity targets float32.

Accuracy is measured in ULPs for finite float32 results, with explicit
classification checks for zero, subnormal, normal, infinity, and NaN.
Vectorized and scalar-tail results must obey the same contract.

Target implementation
---------------------

Exp
~~~

The AVX2+FMA kernel should use:

* magic-bias rounding for ``x / log(2)``;
* split high/low ``log(2)`` constants and FMA range reduction;
* a documented minimax polynomial evaluated with FMA;
* two exponent scale factors covering exponents from ``-150`` through
  ``128`` without flushing valid subnormals;
* two or more independent vectors per loop when measurements show that
  unrolling hides the Horner dependency chain;
* vector mask loads/stores or one common correct scalar tail;
* explicit NaN, infinity, overflow, and underflow handling outside the common
  finite-data dependency chain where profitable.

AVX-512 receives the same numerical reconstruction and polynomial after the
AVX2 design is proven. SSE2 retains a portable non-FMA implementation with the
same edge semantics.

Log
~~~

The first ``Log`` change should preserve the current approximation while:

* normalizing positive subnormals and adjusting their extracted exponent;
* evaluating the polynomial with FMA in the new feature-specific unit;
* unrolling independent vectors to reduce dependency-chain stalls;
* retaining rare special-value correction through masks.

Horner and Estrin evaluation are benchmark candidates only after the
correctness corpus passes. A shorter or different polynomial is accepted only
when it meets the documented ULP bound across the complete input domain.

Scheduling
~~~~~~~~~~

``Exp`` and ``Log`` receive independent scheduling parameters. Candidate
thresholds are chosen from isolated measurements, not inferred from polynomial
degree. Each block must contain enough vectors to amortize dispatch, and the
participant count must stop increasing after throughput saturates.

Registered runtime execution uses the session-owned executor described by the
runtime-controls roadmap. Standalone calls remain serial and cannot introduce
nested workers.

Remaining pull-request sequence
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 25 43 12 10

   * - PR
     - Scope
     - Merge criterion
     - Depends on
     - Status
   * - ExpLog PR01
     - Reproducible benchmark and numerical gate.
     - Isolated and end-to-end runners record raw samples and environment
       metadata. Differential tests expose SIMD subnormal failures, tails, and
       transition boundaries without changing production behavior.
     - None
     - Pending
   * - ExpLog PR02
     - Correct full-domain SIMD semantics.
     - ``Exp`` preserves valid subnormal outputs; ``Log`` normalizes positive
       subnormals; scalar and every SIMD width/tail satisfy the same
       classification and ULP contract.
     - PR01
     - Pending
   * - ExpLog PR03
     - AVX2+FMA float32 ``Exp`` kernel.
     - Runtime feature dispatch selects a dedicated FMA unit. Isolated
       single-thread throughput improves by at least ``2x`` on the reference
       AVX2 machine, with no correctness or small-tensor regression.
     - PR02
     - Pending
   * - ExpLog PR04
     - AVX2+FMA ``Log`` and AVX-512 alignment.
     - FMA and unrolling improve or retain every priority ``Log`` case;
       AVX-512 uses the corrected shared numerical design; all ISA fallbacks
       pass the same differential corpus.
     - PR02, PR03
     - Pending
   * - ExpLog PR05
     - Operator-specific scheduling.
     - Independent thresholds and participant caps improve or retain every
       priority size and thread configuration. No nested-pool or
       oversubscription regression is observed.
     - PR03, PR04
     - Pending
   * - ExpLog PR06
     - Final ONNX Runtime parity gate.
     - The priority-corpus median is at least ``1.0x``, every priority case is
       at least ``0.9x``, and both 4,194,304-element models reach at least
       ``0.95x`` under the published benchmark contract.
     - PR01 through PR05
     - Pending

ExpLog PR06 remains open until both operators satisfy the final gate.
