Unary Kernel Design
===================

The unary runtime adapters include ``Abs``, ``Exp``, ``Log``, ``Not``, and ``Tanh``.
Their node-specific classes live under ``onnx_light_cpu/kernels/math`` and
``onnx_light_cpu/kernels/logical``; typed compute functions and tuning helpers
live under ``ml_cpu_compute/src``.

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

The AVX2/FMA FP32 range evaluates the 13-level continued fraction
``x / (1 + x*x / (3 + ... + x*x / 25))`` as two polynomials in ``x*x``.
It uses a reciprocal estimate with one Newton refinement, clamps the magnitude
and result, and explicitly preserves exact saturation, tiny inputs, and NaNs.
This avoids exponential range reduction and cancellation near zero.
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

Measured AVX2 logits optimization (issue #980)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On October 24, 2026, the three named FP32 logits cases reproduced a repeatable
ORT advantage on an AMD EPYC 9V45 virtual runner (four logical CPUs, two
physical cores). This is not the dashboard's EPYC 7763. The build used GCC
13.3.0, Release, Python 3.13.16, NumPy 2.5.4, and an AVX2/FMA ceiling
(``simd_level=3``). The baseline kernel source was ``eb4a699``; the optimized
source was ``05335f9``. ONNX Runtime was the 1.31.0 Release wheel, reporting
commit ``402979a``. The AVX2 ceiling applies to onnx-light-cpu; ORT retains
its normal hardware dispatch on this AVX-512-capable host.
The onnx-light 0.2.0 cp313 manylinux x86-64 release wheel supplied matching
headers and runtime libraries through ``PYTHONPATH``, with SHA-256
``8e47ccce3f3a70bbe31cd99c66c144bd3feb1e85b2dae1d1f4c5e849344595ae``.
Its source commit was not exposed by the wheel metadata.

Code inspection and disassembly identified the exponential, separate
small-argument polynomial, and vector division in the AVX2 hot loop. Hardware
performance counters were unavailable (``perf_event_paranoid=4``). Porting the
AVX-512 reconstruction and reciprocal approach did not improve direct AVX2
throughput and was rejected. The retained rational approximation removes
exponential reduction and uses one eight-lane vector per iteration.
Disassembly confirms ``vrcpps`` with FMA refinement and no ``vdivps`` in the
FP32 Tanh range. Dispatch, executor tuning, FP16 arithmetic, benchmark inputs,
and accuracy tolerances are unchanged.

**Steady-state end-to-end:** the existing backend benchmark runner measured
``test_cpu_tanh_logits_1x{1,16,128}x202048_float32_benchmark`` with the original
seed ``987654321`` and native tensor feeds. Each setting used three campaigns,
10 warmups, 100 measured calls, and a 10-second phase cap; all 100 samples
were retained. These longer-than-default settings reduce shared-runner noise.
Requested and effective session threads were one or two, inter-op threads
one. Neither process nor workers were pinned (CPUs 0--3 available).
No spin overrides were applied: onnx-light uses its adaptive default and ORT
its default spinning policy, not an assertion of identical spin budgets.
The standard runner measures CPU first, then constructs and measures ORT;
the CPU session remains alive during the ORT phase.
Untimed kernel recording confirmed ``onnx_light_cpu::Tanh``.

The table reports the median of the three campaign medians, in milliseconds:

.. list-table:: End-to-end FP32 Tanh
   :header-rows: 1

   * - Tokens
     - Threads
     - CPU before
     - CPU after
     - ORT before / after
     - CPU speedup
   * - 1
     - 1
     - 0.093476
     - 0.078399
     - 0.042590 / 0.042855
     - 1.19x
   * - 16
     - 1
     - 1.520108
     - 1.241419
     - 0.665786 / 0.633624
     - 1.22x
   * - 128
     - 1
     - 12.380025
     - 9.983109
     - 5.665010 / 5.306403
     - 1.24x
   * - 1
     - 2
     - 0.049625
     - 0.040697
     - 0.045874 / 0.045970
     - 1.22x
   * - 16
     - 2
     - 0.810311
     - 0.641020
     - 0.343865 / 0.329748
     - 1.26x
   * - 128
     - 2
     - 6.699164
     - 5.390190
     - 4.621950 / 4.245648
     - 1.24x

For one thread, the envelopes of the campaigns' p10--p90 CPU intervals
were 0.091117--0.101544 to 0.077626--0.081274 ms (one token),
1.429328--1.586709 to 1.216600--1.273245 ms (16 tokens), and
11.788844--12.514251 to 9.551528--10.329715 ms (128 tokens).
Thus latency falls by 16--19% with one thread and 18--21% with two threads,
but ORT remains faster on the larger cases. These diagnostic measurements
do not establish parity or reconstruct the original campaign's settings.

Reproduce the end-to-end settings with the existing CLI, running it three
times per build and thread setting (replace ``--threads 1`` with ``2``):

.. code-block:: bash

   python -m onnx_light_cpu benchmark \
       --tests "^test_cpu_tanh_logits_.*_float32_benchmark$" --dtypes float32 \
       --threads 1 --repeat 100 --warmup 10 --max-repeat-time 10 \
       --onnxruntime --output /tmp/tanh.xlsx

**Kernel throughput:** ``taskset -c 0 build/exp_log_throughput 31 tanh``
used preallocated buffers, three warmups, and 31 samples, rotating scalar
and AVX2 measurement order with identical inputs. Aligned median seconds
(IQR) before and after were:

.. list-table::
   :header-rows: 1

   * - Elements
     - Before seconds (IQR)
     - After seconds (IQR)
   * - 202048
     - 0.000094264 (0.000005498)
     - 0.000076666 (0.000001352)
   * - 202055 (masked tail)
     - 0.000093852 (0.000003866)
     - 0.000077237 (0.000001161)
   * - 3232768
     - 0.001509332 (0.000016103)
     - 0.001274207 (0.000031757)
   * - 25862144
     - 0.012244703 (0.000199844)
     - 0.009744212 (0.000388759)

Unaligned and small/tail-heavy sizes also improved in this driver.
Full outputs for all three original logits tensors passed untimed comparison
against built-in onnx-light and ORT at the existing FP32 tolerance
(``rtol=2e-6``, ``atol=1e-7``), with maximum absolute differences of
2.39e-7 and 4.18e-7 respectively. Focused C++ tests retain the stricter
5e-7 relative check against double-precision ``std::tanh``, including random
bit patterns, logarithmic magnitudes, reduction neighbors, signed zeros,
subnormals, NaNs, exact saturation, in-place buffers, and every vector tail.
