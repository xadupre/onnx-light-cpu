Binary Kernel Design
====================

Nineteen binary ONNX operators share one descriptor, broadcast-plan, and
runtime-adapter architecture. The implementation is under
``ml_cpu_compute/src/math/binary`` and the registered adapter is
``onnx_light_cpu/kernels/elementwise/binary_kernel.cc``.

Prepared execution
------------------

.. code-block:: text

   NodeProto
      |
      v
   BinaryOperationDescriptor
   (operator, opset, attributes, typed adapters)
      |
      +---- concrete input shapes and types
      v
   BinaryBroadcastPlanCache (8-entry LRU)
      |
      v
   immutable BinaryBroadcastPlan
   (output shape, coalesced strides, loop family)
      |
      v
   scalar or bulk typed function

``BinaryOperationDescriptor`` is built once for the node. It validates
operator-specific attributes such as ``Mod.fmod`` and
``BitShift.direction``, resolves the output type, and binds scalar and optional
bulk functions for the exact input/output type triple.

Dynamic shapes use a bounded eight-entry LRU cache. Its key contains a
monotonic descriptor identity, input/output types, and both shapes; it never
stores raw descriptor pointers. A miss constructs and validates a fresh plan,
so stale strides cannot be reused.

Operators and traversal
-----------------------

The manifest covers arithmetic (``Add``, ``Sub``, ``Mul``, ``Div``, ``Mod``,
``Pow``), comparisons, logical operators, bitwise operators, ``BitShift``, and
``PRelu``. It is the source of truth for registered type signatures.

``Mod`` tracks schema version 28 while retaining minimum opset 10 compatibility.
Starting at opset 28, floating-point ``fmod=0`` uses floor-remainder semantics:
nonzero results and exact zeros take the divisor's sign. The implementation
adjusts ``std::fmod`` rather than forming a potentially overflowing quotient.
``fmod=1`` retains truncated-remainder semantics, including the dividend's
signed zero. Earlier opsets still require ``fmod=1`` for floating-point inputs.
FLOAT16 and BFLOAT16 use the same scalar semantics through the existing
SIMD-capable float conversion adapters.

After right-aligning shapes, the plan assigns zero strides to broadcast
dimensions, removes unit dimensions, coalesces compatible adjacent dimensions,
and selects one loop family:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Loop family
     - Purpose
   * - Contiguous
     - Both inputs and the output advance linearly.
   * - Left / right scalar
     - Broadcast one scalar while preserving operand order.
   * - Repeated contiguous block
     - Reuse one broadcast value or block across a contiguous suffix.
   * - Inner-vector / outer broadcast
     - Vectorize a contiguous inner region while advancing coalesced outer
       offsets.
   * - General strided
     - Use the validated coalesced strides as the correctness fallback.

Offsets advance incrementally across outer blocks; division and modulo are not
performed for every output element. Empty dimensions produce no work and read
neither input.

Compute, scheduling, and safety
-------------------------------

Same-type common signatures bind bulk contiguous and scalar-broadcast kernels.
Other legal signatures use typed scalar adapters. Validation occurs before
unchecked loops for integer division, modulo, shifts, and other operations with
invalid inputs. Comparison results and logical tensors use one byte per BOOL.

The tuning snapshot has separate thresholds for bulk, block-broadcast, and
scalar layouts plus a target block size and participant limit. Independent
flat ranges or outer blocks are submitted through the session executor; the
plan itself owns no scheduler and stores no thread count.

An input may alias the output only when its type and shape equal the output and
the selected traversal cannot overwrite a value before its final read. A
broadcast input is never expanded in place.

Float32 Greater on AVX2
-----------------------

``Greater`` binds its float32 contiguous bulk callback to an AVX2 implementation
when both compilation support and runtime CPU/OS support are present. The same
callback handles contiguous suffixes of row broadcasts. Four eight-element
ordered comparisons are packed into 32 bytes, reordered across the 128-bit
lanes, and masked to canonical BOOL values (0 or 1). Eight-element and scalar
tails never read or write beyond the range. NaNs compare false. Other types,
scalar broadcasts, general strided loops, and non-AVX2 builds retain their
existing implementations. Dispatch is resolved at descriptor construction, not
inside each row, and scheduling is unchanged.

Issue #998 measurements (2026-10-23)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Host: AMD EPYC 7763, Linux x86-64 VM exposing four logical CPUs (two cores,
two threads per core), process affinity CPUs 0-3. GCC 13.3.0 Release build,
``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2``; ``detect_simd_level()`` returned 3.
Baseline onnx-light-cpu 0.2.0 source:
``765a814e1d98e4192f45718f176fa5b6f8f07f38`` (the subsequent initial-plan
commit changed no source); optimized source:
``d680cdf``. Dependencies were the released Linux CPython 3.13 wheels for
onnx-light 0.2.0 and ONNX Runtime 1.31.0, with NumPy 2.5.4. Wheel build
commits were not available; these are not asserted to match the dashboard's
retained measurements.

Both runtimes used explicit thread limits (1 or 4), no worker affinity, and
disabled spin-wait (``park_immediately`` for onnx-light, both ORT
``allow_spinning`` settings zero). ORT used sequential graph execution and one
inter-op thread. For each case/thread pair, three campaigns used 200 warm-up
calls and 1,000 timed calls per backend, alternating backend order on each
iteration, with a five-second cumulative per-backend limit (not reached).
Native backend tensors fed onnx-light-cpu; NumPy views fed ORT. Models, input
generation, shapes, and operand order were the unmodified named backend cases.
Outputs were checked for exact equality, and kernel recording confirmed
``onnx_light_cpu::Greater`` before timing.

The table reports the median of the three campaign medians, in milliseconds.
``row`` is ``float32[4096,256] > float32[256]``; ``contiguous`` and
``swapped`` are the two ``float32[4194304]`` cases named in the issue.

========== ======= ========== ========== ========= =========
Case       Threads CPU before ORT before CPU after ORT after
========== ======= ========== ========== ========= =========
row        1       0.192028   0.223447   0.108493  0.219525
contiguous 1       1.224774   1.231417   1.026360  1.223788
swapped    1       1.238185   1.205388   0.998292  1.173505
row        4       0.143558   0.157213   0.101795  0.156933
contiguous 4       0.811078   0.801149   0.801099  0.810692
swapped    4       0.798940   0.791766   0.780185  0.793505
========== ======= ========== ========== ========= =========

The dashboard's row gap did not reproduce. The four-thread contiguous gap was
repeatable but smaller (1.2-1.6%); swapped was near parity. The optimized row
case improved by 1.41x with four threads and 1.77x with one. Single-thread
contiguous cases improved by 1.19-1.24x. Four-thread contiguous timings remain
near parity: the optimized ordinary case ranged from 0.788 to 0.809 ms and
even lost narrowly to ORT in one campaign. These small differences are not a
decisive win; neither historical latencies nor all historical gaps were reproduced.

Hardware-counter profiling was unavailable (``perf_event_paranoid=4``).
Assembly inspection showed baseline SSE comparisons followed by extensive
unpacking/shuffling to produce byte results. The selected AVX2 callback instead
uses four ``vcmpgt_oqps``, two ``vpackssdw``, one ``vpacksswb``, ``vpermd``,
and ``vpand`` per 32 results. The row and single-thread improvements support
reducing that packing overhead; the four-thread streaming cases remain limited
by effects this change does not resolve.

Validation passed all 41 targeted binary C++ tests and 154 Greater backend
correctness cases. The new lengths 0-257, unaligned buffers, row/swapped cases,
and IEEE special-value test also passed AddressSanitizer/UndefinedBehaviorSanitizer.
All 13 broadcast-plan tests passed with AVX2/F16C source groups excluded; that
forced configuration also exposed an unrelated float16 dispatch-pointer
assertion in the descriptor suite. The normal AVX2 configuration passes it.

To reproduce the timing loop after building the Python integration, select
``TestMode.BENCHMARK`` cases with
``generate_benchmark_expected_outputs=False`` and this exact regular expression::

   ^test_cpu_greater_v13_(row_float32xfloat32_to_bool_n1048576|contiguous_float32xfloat32_to_bool(_swapped)?_n4194304)_benchmark$

Use ``onnx_light_cpu._benchmark._prepare_feeds`` for separate native/NumPy
feeds, the session policies above, and
``tools.benchmark_binary_parity.measure_alternating((cpu_run, ort_run),
repeat=1000, warmup=200, max_repeat_time=5)`` three times per case/thread pair.
Run with ``taskset -c 0-3`` and keep kernel recording disabled during timing.
Do not use that parity tool's case selector: it excludes swapped cases and
float32 Greater is not part of its priority matrix.
