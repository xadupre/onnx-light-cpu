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
ONNX adapter does not currently cache a plan or constant B: it constructs a
fresh plan from both concrete input shapes on every invocation.

Types and algorithm selection
-----------------------------

The registered adapter accepts equal-typed ``FLOAT``, ``DOUBLE``, ``FLOAT16``,
and ``BFLOAT16`` inputs. Dynamic negative dimensions, mixed types, unsupported
types, and incompatible matrix or batch dimensions fail before compute.

FLOAT and DOUBLE execute directly through ``MatMulPlan<T>``. FLOAT16 and
BFLOAT16 use ``MatMulPlan<float>`` together with ``GemmHalfPlan`` for the
native half-capable algorithms, with FP32 accumulation and final narrowing.
For BF16 general and direct algorithms, the current fallback materializes
complete FP32 copies of A and B plus the FP32 output before converting back to
BF16; this path is not bounded to one panel.

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
