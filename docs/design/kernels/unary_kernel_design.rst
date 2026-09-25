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
     - Portable ``std::tanh`` and runtime-selected AVX2/FMA or AVX-512;
       low-precision conversion uses bounded worker-local storage.

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

FP32 ``Exp`` starts executor dispatch at 128 KiB and limits cache-sized work
to three participants. FP32 ``Sigmoid`` uses up to eight AVX-512 participants
for vectors between 96K and 256K elements. Its AVX-512 denominator remains
in ``[1, 2]``, so one Newton-refined reciprocal replaces vector division.

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
--------------------------

``ai.onnx::Tanh`` preserves the input shape and element type, including scalar
and empty tensors. It supports in-place compute, signed zero, subnormals, NaNs,
and infinities (mapped to signed one). FP16 and BF16 compute in FP32 and round
once on output. Each worker uses at most a 1,024-element FP32 conversion block;
there is no full-tensor intermediate beyond the output.

The AVX2/FMA range evaluates a small-argument polynomial to avoid cancellation
near zero and reuses the exponential approximation for larger arguments.
The AVX-512 range instead reduces ``-2 * abs(x)`` and computes
``expm1(reduced)`` with the existing exponential coefficients. Reconstructing
the numerator as ``(1 - scale) - scale * expm1(reduced)`` avoids cancellation
near zero without a second polynomial. Clamping the magnitude to 10 keeps the
exponential normal, so general overflow/underflow handling is unnecessary.
The denominator lies in ``[1, 2]``; a reciprocal estimate with one Newton
refinement replaces vector division while preserving the FP32 error tolerance.
CPU feature detection is cached, preferring AVX-512, then AVX2/FMA,
then the portable path. The AVX-512 range processes 16 lanes per vector, with
two-vector unrolling and masked tails; FP16 and BF16 share it through the
existing worker-local FP32 conversion blocks.
Scheduling uses the session-owned executor rather than a private pool.
On AVX-512, FP32 inputs below 32,768 elements stay serial. At and above that
threshold, ranges target at least 16,384 elements per participant, so a large
one-token vocabulary can also use the executor. Other dispatch levels retain
the 65,536-element threshold. Nested calls do not submit another parallel region.

The backend benchmark registry includes small and tail-heavy vectors plus
logit tensors with shapes ``[1, 1, 202048]``, ``[1, 16, 202048]``, and
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

Measured AVX2 path
^^^^^^^^^^^^^^^^^^

On an AMD EPYC 7763 (GCC Release build, AVX2/FMA, one thread pinned to CPU 3),
the existing throughput driver compares the scalar and AVX2 range functions
directly. Both use the same preallocated FP32 input/output buffers, three
warmups, and 15 samples with alternating measurement order. The table reports
median seconds for aligned inputs; the driver also includes unaligned and
tail-heavy cases. This comparison excludes allocation and session overhead.

.. code-block:: bash

   cmake -S . -B build-tanh -DONNX_LIGHT_CPU_BUILD_PYTHON=OFF \
       -DONNX_LIGHT_CPU_BUILD_BENCHMARKS=ON -DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2
   cmake --build build-tanh --target exp_log_throughput -j4
   taskset -c 3 build-tanh/exp_log_throughput 15 tanh

.. list-table::
   :header-rows: 1

   * - FP32 shape
     - Scalar seconds
     - AVX2/FMA seconds
     - Speedup
   * - 1 x 1 x 202048
     - 0.003291949
     - 0.000150932
     - 21.81x
   * - 1 x 16 x 202048
     - 0.052805285
     - 0.002448955
     - 21.56x
   * - 1 x 128 x 202048
     - 0.422916743
     - 0.019602640
     - 21.57x

These are speedups over this kernel's portable implementation, not over ONNX
Runtime. For example, the one-thread backend benchmark (30 repeats, five
warmups, ORT 1.30) measured FP32 one-token medians of 0.000205579 seconds here
and 0.000107706 seconds in ORT. Small eight-element sessions measured
0.000006568 and 0.000014382 seconds respectively. Absolute session timings on
this shared runner are indicative rather than performance guarantees.
These earlier AVX2 measurements do not establish an AVX-512 speedup: the
EPYC 7763 cannot execute AVX-512.

To compare the AVX-512 range on compatible hardware, build the same driver
with ``-DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX512`` instead. The ``tanh`` mode adds
``TanhAVX512`` rows only when that implementation is compiled and the CPU/OS
supports it. It rotates measurement order across scalar, AVX2/FMA, and AVX-512
using identical inputs and one thread, including sizes 15, 16, 17, 31, 32, and
33 to expose vector and masked-tail overhead. The reported speedup is relative
to the scalar range, not ONNX Runtime; use the backend command above for an
end-to-end ORT comparison.

Measured AVX-512 logits optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On an AMD EPYC 9V74 with AVX-512 (GCC Release, one thread pinned to CPU 0),
the normal-range exponential reconstruction and refined reciprocal reduced
the direct FP32 range latency as follows. These measurements use the driver
above with 15 samples, three warmups, identical preallocated inputs and
rotating scalar/AVX2/AVX-512 measurement order. Both columns execute AVX-512;
the baseline is the original general-exponential, small-polynomial and
division implementation.

.. list-table::
   :header-rows: 1

   * - Elements (aligned)
     - Before (seconds)
     - After (seconds)
     - Speedup
   * - 202048
     - 0.000113447
     - 0.000072267
     - 1.57x
   * - 202055 (masked tail)
     - 0.000113758
     - 0.000072277
     - 1.57x
   * - 3232768
     - 0.001826498
     - 0.001169998
     - 1.56x

For ``test_cpu_tanh_logits_1x1x202048_float32_benchmark``, the end-to-end
one-thread median was 0.000076353 seconds versus 0.000059608 seconds for
ONNX Runtime 1.30.0 (100 repeats, 10 warmups, pinned to CPU 0).
The arithmetic optimization narrows the gap but does not yet match ORT on
this case. Direct range timings exclude session overhead and must not be
reported as end-to-end ORT speedups.
