Processor-Aware Tuning for Abs, Exp, Log, and Not
==================================================

:Date: 2026-08

**discussion**

Objective
---------

The objective is to integrate the ``Abs``, ``Exp``, ``Log``, and ``Not``
kernels with onnx-light's processor-aware tuning registry. Each kernel should
resolve an immutable configuration for its implementation, element type,
processor, and effective session thread count. Calibration must improve the
serial-to-parallel crossover and the useful worker count without changing ONNX
semantics.

The target is median performance of at least ``1.0x`` ONNX Runtime on the
priority corpus, with no large-tensor case below ``0.9x``. ONNX Runtime remains
in its normal multithreaded configuration in published comparisons.

Available onnx-light infrastructure
-----------------------------------

The onnx-light tuning implementation completed by
`onnx-light #4428 <https://github.com/xadupre/onnx-light/pull/4428>`_ provides:

* exact keys containing library, kernel, implementation, element type, device,
  and tuning ABI;
* processor and effective-thread-count profile resolution;
* validated portable defaults and immutable per-session configuration;
* deterministic calibration callbacks with correctness checks and resource
  limits;
* an atomic persistent cache and Python/CLI inspection and calibration APIs.

The built-in onnx-light ``Abs`` and ``Not`` kernels already register calibration
callbacks. Its ``Exp`` kernel registers a parallel schema without a callback,
and its ``Log`` kernel is not tunable. These registrations cannot be reused
directly: onnx-light-cpu replaces the implementations and therefore needs keys,
defaults, validation, and calibration owned by ``library="onnx_light_cpu"``.

Required execution contract
---------------------------

The session executor, affinity, and spin API is tracked by the
:doc:`Runtime Execution Controls Roadmap <2026_08_runtime_execution_controls>`.
Registered adapters execute SIMD ranges through the session-owned executor, so
the tuning descriptor and actual participant count are identical. Standalone
C++ entry points are serial and own no competing scheduler.

The session parallel API must support a kernel-selected maximum participant
count. This is necessary for memory-bound ``Abs`` and ``Not``: their best worker
count can be lower than the session limit after memory bandwidth saturates.

Tuning schema
-------------

Version 1 uses two parameters:

``parallel.minimum_elements``
    Minimum tensor size that justifies worker dispatch.

``parallel.maximum_threads``
    Maximum participants useful to this kernel. It must be positive and no
    greater than the effective session thread count.

Each operator and element type has an independent exact key:

.. code-block:: text

    library        = onnx_light_cpu
    kernel         = Abs | Exp | Log | Not
    implementation = simd_dispatch
    element_type   = exact ONNX tensor element type
    device         = CPU
    tuning_abi     = 1

``simd_dispatch`` is stable because the processor descriptor and feature set
already distinguish SSE2, AVX2, AVX-512, NEON, and SVE machines. No ISA branch,
cache lookup, string lookup, allocation, or lock is allowed in the execution
path after session preparation.

Calibration contract
--------------------

Calibration writes into caller-preallocated output tensors so allocation and
runtime lifetime costs do not distort kernel thresholds. End-to-end benchmarks
remain separately required.

* ``Abs`` covers every supported floating-point and signed integer type,
  negative zero, NaN, infinities, integer minima, SIMD tails, and in-place
  execution.
* ``Not`` covers canonical and non-canonical byte values, tails, and in-place
  execution.
* ``Exp`` covers ordinary values, underflow, overflow, infinities, NaN, and the
  FP16/BF16 conversion paths.
* ``Log`` uses positive calibration inputs plus zero, negative values,
  infinities, NaN, subnormals, and conversion paths.
* The forced serial implementation is the reference. Every candidate output
  must pass the operator/type-specific exact or tolerance check before its
  timing can be considered.
* Searches use warmups, median timings, consecutive wins, and explicit duration
  and memory limits. Threshold and worker-count searches are not treated as
  independent when their interaction changes the winner.

Benchmark contract
------------------

The performance corpus measures two layers:

#. the SIMD compute path with preallocated input and output;
#. the complete ``ReferenceEvaluator`` path, including output allocation and
   runtime bookkeeping.

It covers sizes from ``10^2`` through ``10^8``, every supported element type,
the portable scalar fallback, available SIMD ISAs, and session thread counts
``1``, ``2``, ``4``, physical-core count, and logical-core count. Results record
the selected parameters, CPU descriptor, affinity, effective threads, raw
samples, median, and dispersion.

Published ONNX Runtime measurements use its normal multithreaded CPU execution
provider. Additional equal-thread experiments are diagnostic only and must not
replace the published baseline.

Remaining pull-request sequence
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 9 27 43 13 8

   * - PR
     - Scope
     - Merge criterion
     - Depends on
     - Status
   * - Tuning PR01
     - Truthful session parallel execution.
     - onnx-light exposes the session pool and effective thread count to
       kernels, supports a maximum participant count, and proves that
       ``RuntimeParameters::num_threads`` controls the workers actually used.
       onnx-light-cpu adapters execute SIMD ranges without another scheduler.
     - None
     - Pending
   * - Tuning PR02
     - Common onnx-light-cpu tuning schema.
     - Exact keys, portable defaults, validation, typed immutable
       configuration, and ``TuningKey``/``Configure`` integration cover all
       four kernels and supported element types. Registry access counters stay
       unchanged during repeated execution.
     - PR01
     - Pending
   * - Tuning PR03
     - ``Abs`` and ``Not`` calibration.
     - Deterministic callbacks calibrate the crossover and useful worker count,
       reject incorrect candidates, obey resource limits, and improve or retain
       every priority case.
     - PR02
     - Pending
   * - Tuning PR04
     - ``Exp`` and ``Log`` calibration.
     - Type-specific domains and tolerances cover special values and
       FP16/BF16 conversion. Independent profiles improve or retain every
       priority ``Exp`` and ``Log`` case.
     - PR02
     - Pending
   * - Tuning PR05
     - Cache, registration, and Python lifecycle.
     - ``register_kernels()`` registers schemas and callbacks before session
       creation, loads matching ``onnx_light_cpu`` profiles even when
       onnx-light loaded its cache earlier, and exposes discovery, override,
       proposal, calibration, and inspection through the existing APIs.
       Existing sessions retain their captured generation.
     - PR03, PR04
     - Pending
   * - Tuning PR06
     - Correctness and performance gate.
     - Every supported type passes differential and cache-lifecycle tests.
       Median priority performance is at least ``1.0x`` ONNX Runtime and no
       large-tensor case is below ``0.9x``. Raw preallocated and end-to-end
       results identify any remaining allocator/runtime overhead.
     - PR05
     - Pending

Tuning PR06 is the final roadmap PR.
