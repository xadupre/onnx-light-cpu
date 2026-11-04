MatMul Kernel Design
====================

The ``MatMul`` adapter implements ONNX vector, matrix, and batched
matrix-multiplication semantics on top of the same GEMM plans and
micro-kernels described in :doc:`gemm_kernel_design`.

Shape and plan model
--------------------

``MatMulPlan<T>`` resolves the complete output shape before execution:

* rank-one inputs are promoted for the product and squeezed from the output;
* leading batch dimensions use multidirectional broadcasting;
* each input receives precomputed batch strides, including zero strides for
  broadcast dimensions;
* empty batches and zero-sized outputs return without reading input data.

.. code-block:: text

   A shape + B shape
          |
          v
      MatMulPlan<T>
      - promoted M, N, K
      - broadcast batch shape
      - batch strides
      - embedded GemmPlan<T>
          |
          v
   one GEMM per broadcast batch item

The reusable ``MatMulPlan`` library API can own a constant ``B`` tensor, giving
callers a stable lifetime independent of the source buffer. The registered
ONNX adapter caches plans by dtype and concrete input shapes. Eligible immutable
FP32 projection weights can be prepared separately in the runtime value store.

Types and algorithm selection
-----------------------------

The registered adapter accepts equal-typed ``FLOAT``, ``DOUBLE``, ``FLOAT16``,
and ``BFLOAT16`` inputs. Dynamic negative dimensions, mixed types, unsupported
types, and incompatible matrix or batch dimensions fail before compute.

FLOAT and DOUBLE execute directly through ``MatMulPlan<T>``. FLOAT16 and
BFLOAT16 use ``MatMulPlan<float>`` together with ``GemmHalfPlan`` for the
native half-capable algorithms, with FP32 accumulation and final narrowing.
The adapter owns one FP32 output workspace across all broadcast batches.
Native half routines pack compact panels; fallback routines widen only while
packing or reducing, not by materializing complete FP32 copies of A and B.

Peak-memory accounting
---------------------

The CPU ``MatMul`` peak-memory hook uses the same shape plans and checked
packing-size formulas as execution. It excludes declared output ``Y``, inputs,
and preparation-owned weights, and includes the FP32 half-output workspace,
shared B panel waves, per-worker A/micro-B panels, skinny accumulators, and
split-K partial outputs. Concurrent batch products are accounted for separately
from inner-GEMM parallelism; nested split-K packing is serial per partition.

The shape-only registration API carries neither dtype nor executor policy.
Consequently the hook reports only a conservative worst case across FLOAT,
DOUBLE, FLOAT16, BFLOAT16 and all plan-admitted executor sizes. It does not
assume that profiling's current (often serial) executor is execution's executor.
For a known participant limit, the plan ``ScratchBytes(limit)`` and
``HalfScratchBytes(is_bfloat16, limit)`` APIs give a tighter bound. Unrestricted
executor bounds can exceed typical session usage, especially for large batches
or projections; this is packing concurrency, not full-source promotion.

The embedded ``GemmPlan`` chooses the current shape-specific algorithm once:

* direct execution for small natural-layout reductions;
* skinny-M or skinny-N kernels when one output dimension underfills the
  ordinary register tile;
* split-K for the guarded tiny-output, long-reduction corner;
* the general blocked and packed panel engine otherwise.

Scheduling and invariants
-------------------------

Batch items are independent. They run in parallel only when the inner GEMM is
effectively serial; if one product already exposes useful M/N or split-K
parallelism, the batch loop remains serial to avoid nested executor regions.
All work uses the current session ``CpuExecutor``.

The constant-B storage must contain exactly the planned matrix, and the
constant-only ``Execute`` overload is rejected when the plan does not own B.
The output buffer must cover every broadcast batch item and its complete
``M x N`` matrix.

``MatMulInteger`` and ``QLinearMatMul`` are separate registered adapters with
their own integer GEMM paths; they do not execute through this ``MatMul``
adapter.
