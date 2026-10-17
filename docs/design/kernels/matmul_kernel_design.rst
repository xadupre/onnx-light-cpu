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
ONNX adapter caches its shape-derived plan and rebuilds it when the concrete
input shapes or dtype change.

Immutable weight preparation
----------------------------

During the first ``RuntimeSession::Run``, the adapter's preparation hook packs
an immutable rank-two ``B`` once into the session hierarchy's ``ValueStore``.
FLOAT weights are rearranged into contiguous column panels. FLOAT16 and
BFLOAT16 weights are converted directly into the same FLOAT panel format,
without an intermediate full-size widened tensor. Later inference calls read
the prepared tensor through a stable integer slot; they neither hash nor
repack the complete weight.

The key includes the initializer name and content digest, original dtype,
``K``/``N`` dimensions, effective ISA, panel width, and layout version. It does
not depend on ``M``: prefill, decode, vector A, and batched A with a shared
rank-two B reuse the same representation. Matching nodes and sessions sharing
the same ``ValueStore`` share one allocation. A shape change rebuilds only the
local execution plan, not B. Packed execution uses the existing FP32 SIMD
micro-kernels and the session CPU executor; half inputs require only bounded
A conversion panels and the output-sized FP32 workspace.

Initializers that are also graph inputs, or replaced by caller values before
session initialization, remain dynamic. DOUBLE, vector B, batched B, and empty
B retain the existing execution path. The original model and its initializers
must remain immutable during the session lifetime.

``RuntimeSession::prepared_bytes`` includes this storage, which occupies
``K * N * sizeof(float)`` bytes for each distinct prepared contract.
``ValueStore`` releases it with the owning store. This synchronous path does
not use ``PreparedExecutionState`` and does not implement eviction,
rematerialization, or disk persistence.

Types and algorithm selection
-----------------------------

The registered adapter accepts equal-typed ``FLOAT``, ``DOUBLE``, ``FLOAT16``,
and ``BFLOAT16`` inputs. Dynamic negative dimensions, mixed types, unsupported
types, and incompatible matrix or batch dimensions fail before compute.

FLOAT and DOUBLE execute directly through ``MatMulPlan<T>``. FLOAT16 and
BFLOAT16 use ``MatMulPlan<float>`` together with ``GemmHalfPlan`` for the
native half-capable algorithms, with FP32 accumulation and final narrowing.
The dynamic half paths convert bounded panels as needed, rather than widening
the complete B tensor. Prepared immutable weights instead use the persistent
FLOAT panels described above.

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

Focused measurement
-------------------

Build with ``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON`` and
``ONNX_LIGHT_CPU_BUILD_BENCHMARKS=ON``, then run
``matmul_prepack_throughput 4``. This kernel-level driver compares dynamic and
prepared weights for FP32, FP16, and BF16, using the Qwen projection shapes
``512x32000``, ``512x1376``, ``1376x512``, ``512x1024``, and ``512x512``, each
with ``M=1`` and ``M=128``. It reports explicit preparation-hook latency
separately from steady-state median/p90 latency and prepared bytes. It does
not measure model loading or full-model inference. Timings are diagnostic,
not correctness gates. FLOAT preparation adds a full-size weight allocation;
FLOAT16 and BFLOAT16 preparation uses twice the original weight bytes.
Removing repeated conversion does not guarantee a speedup for memory-bound
decode: the wider persistent representation can increase memory traffic.
