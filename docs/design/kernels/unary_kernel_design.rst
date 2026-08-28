Unary Kernel Design
===================

The unary runtime adapters implement ``Abs``, ``Exp``, ``Log``, and ``Not``.
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
