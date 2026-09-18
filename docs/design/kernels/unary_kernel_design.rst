Unary Kernel Design
===================

The unary runtime adapters include ``Abs``, ``Exp``, ``Log``, ``Not``, and ``Tanh``.
Their node-specific classes live under ``onnx_light_cpu/kernels/math`` and
``onnx_light_cpu/kernels/logical``; typed compute functions and tuning helpers
live under ``onnx_light_cpu/impl``.

Execution architecture
----------------------

Each registered adapter follows the same path:

.. code-block:: text

   NodeProto + RuntimeContext
              |
              v
        KernelBase::Run
              |
              v
      validate type and shape
              |
              v
      resolve immutable tuning
              |
              v
   scalar/SIMD range function
              |
              v
      session CpuExecutor

The output retains the input shape and type. ``Run`` allocates it through the
runtime context, while the direct ``operator()`` entry points require matching
preallocated input and output tensors.

Operators and types
-------------------

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Operator
     - Registered types
     - Implementation
   * - ``Abs``
     - FLOAT, DOUBLE, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64
     - Typed scalar fallbacks and ISA-selected vector loops, including
       low-precision and integer paths.
   * - ``Exp`` / ``Log``
     - FLOAT, DOUBLE, FLOAT16, BFLOAT16
     - Shared scheduling with operator-specific approximation, conversion,
       exceptional-value, and tail handling.
   * - ``Not``
     - BOOL
     - Byte-valued boolean inversion; ONNX BOOL tensors are not bit-packed.
   * - ``Tanh``
     - FLOAT, FLOAT16, BFLOAT16
     - Portable ``std::tanh`` and runtime-selected AVX2/FMA; low-precision
       conversion uses bounded worker-local storage.

Tuning and scheduling
---------------------

``unary_execution_tuning.h`` contains the shared range schedulers.
``ExecuteUnaryRanges`` uses a byte threshold for inexpensive operations, while
``ExecuteCostedUnaryRanges`` also accounts for the operation cost. The resolved
tuning snapshot contains the bulk threshold, target block size, and participant
limit; ``Abs`` can additionally select preferred participants and a
streaming-store threshold.

Small tensors execute on the calling thread. Setting the parallel threshold to
zero disables executor dispatch completely. Otherwise, larger tensors are
divided into independent contiguous ranges and submitted to onnx-light's
current ``CpuExecutor``. A participant limit of zero means that the session
executor may use every participant it admits.

Dispatch and invariants
-----------------------

ISA selection is cached and gated by both compiled translation units and
runtime CPU capabilities. Unsupported instructions are never entered on a
weaker host, and every vector implementation has an exact scalar tail and a
portable fallback.

The adapters reject unsupported types and mismatched buffers before compute.
Floating-point paths preserve their documented NaN, infinity, signed-zero, and
domain behavior; integer absolute value avoids undefined signed overflow.

Tanh and logit softcapping
-------------------------

``ai.onnx::Tanh`` preserves the input shape and element type, including scalar
and empty tensors. It supports in-place compute, signed zero, subnormals, NaNs,
and infinities (mapped to signed one). FP16 and BF16 compute in FP32 and round
once on output. Each worker uses at most a 1,024-element FP32 conversion block;
there is no full-tensor intermediate beyond the output.

The AVX2/FMA range evaluates a small-argument polynomial to avoid cancellation
near zero and reuses the exponential approximation for larger arguments.
CPU feature detection is cached; machines without AVX2/FMA use the portable
path. Scheduling uses the session-owned executor rather than a private pool.

The backend benchmark registry includes small and tail-heavy vectors plus
Muse-Glimmer logits with shapes ``[1, 1, 202048]``, ``[1, 16, 202048]``, and
``[1, 128, 202048]`` for all three types. These measure the Tanh step in
``20 * tanh((0.19611613513818404 * logits) / 20)``; the complete expression is
also checked against ONNX Runtime in the integration tests.

.. code-block:: bash

   python -m onnx_light_cpu benchmark --tests "^test_cpu_tanh_" \
       --dtypes float32 float16 --threads 1 --repeat 100 --warmup 10 \
       --onnxruntime --output tanh.xlsx

ONNX Runtime's CPU provider does not supply a BF16 Tanh kernel. BF16 parity is
therefore checked with BF16-rounded inputs through its FP32 kernel, followed
by BF16 output rounding. Some ORT CPU versions flush FP32 subnormal results;
the tests allow that underflow-only difference while independently requiring
this kernel to preserve subnormals and the sign of zero exactly.
