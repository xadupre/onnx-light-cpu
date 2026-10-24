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

Float32 Less on AVX2
--------------------

``Less`` uses the same packed float32 comparison implementation as ``Greater``,
with operands reversed, for contiguous, left-scalar, and right-scalar bulk
callbacks. Reversing operands rather than negating a comparison preserves
ordered IEEE semantics: NaNs compare false, including in scalar broadcasts and
tails. Descriptor construction checks compilation and runtime AVX2 support.
Other types, general strided loops, scheduling, and non-AVX2 fallbacks are
unchanged.

Issue #995 measurements (2026-10-23)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Host: AMD EPYC 9V45, Linux x86-64 VM exposing four logical CPUs (two cores,
two threads per core), process affinity CPUs 0-3. GCC 13.3.0 Release
(``-O3 -DNDEBUG``), ``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2``;
``detect_simd_level()`` returned 3. Baseline onnx-light-cpu 0.2.0 source:
``e60715618a465b4455674f5dcfdf4ecc0d27ea65``; optimized source:
``a5b0389611f374654274c6d9c38ab36f52edc168``.
Dependencies were the released Linux CPython 3.13 wheels for onnx-light 0.2.0
and ONNX Runtime 1.31.0, with NumPy 2.5.4. Wheel build commits were unavailable.
The local bindings used nanobind 3.1.0 and
``ONNX_LIGHT_CPU_PYTHON_STABLE_ABI=OFF`` to match the onnx-light wheel.
Only onnx-light-cpu was ISA-capped; ORT retained its stock wheel dispatch.
These builds and this processor are not asserted to reproduce the dashboard
environment.

Both runtimes used explicit thread limits (1 or 4), no worker affinity, and
disabled spinning (``park_immediately`` for onnx-light, both ORT
``allow_spinning`` settings zero). ORT used sequential graph execution and one
inter-op thread. Each case/thread pair used three campaigns, each with 200
warm-up calls and 1,000 timed calls per backend, alternating backend order on
each iteration. The five-second cumulative per-backend limit was not reached.
Inputs, models, shapes, and operand order were the unmodified named backend
cases. Native backend tensors fed onnx-light-cpu; NumPy views fed ORT.
Outputs matched ORT exactly, and recording confirmed ``onnx_light_cpu::Less``
before timing; recording was disabled during timing.

Median of the three campaign medians, in milliseconds:

================== ======= ========== ========== ========= =========
Case               Threads CPU before ORT before CPU after ORT after
================== ======= ========== ========== ========= =========
contiguous         1       0.840420   0.864576   0.687678  0.860010
contiguous swapped 1       0.850450   0.856989   0.689266  0.848642
scalar swapped     1       0.096707   0.098770   0.048754  0.100813
contiguous         4       0.578533   0.578313   0.554907  0.584622
contiguous swapped 4       0.584372   0.578662   0.558032  0.579890
scalar swapped     4       0.084283   0.077678   0.048544  0.073176
================== ======= ========== ========== ========= =========

``contiguous`` and ``contiguous swapped`` have two float32[4194304] inputs.
``scalar swapped`` is the issue's ``right_scalar`` swapped case: its actual
inputs are float32[] and float32[1048576].

The historical 21-23% contiguous gaps did not reproduce. With four threads,
ordinary contiguous was near parity and swapped varied from narrowly faster
to about 1% slower than ORT. Scalar swapped was consistently 6-9% slower.
After optimization, the CPU scalar-swapped case improved by 1.74x with four
threads and 1.98x with one. Single-thread contiguous cases improved by
1.22-1.23x; four-thread improvements were only 4-5%. Optimized four-thread
campaign medians ranged from 0.554822 to 0.556119 ms (contiguous),
0.556950 to 0.562809 ms (swapped), and 0.047903 to 0.049180 ms (scalar swapped).
These are repeated local observations, not confidence intervals or proof of
the same gains on EPYC 7763.

Hardware counters were unavailable (``perf_event_paranoid=4``). Assembly
inspection showed baseline SSE comparisons with extensive BOOL unpacking and
shuffling. The optimized contiguous wrapper jumps to the shared packed AVX2
implementation; scalar broadcasts hoist the scalar load. Dispatch-pointer
tests confirm selection of all three AVX2 callbacks. The 39 descriptor and
broadcast-plan tests passed, including Less IEEE values, unaligned buffers,
empty/null ranges, and every length 0-97. All 154 Less backend correctness
cases passed against the corpus's built-in-kernel oracle at both thread counts.
The comparison test and all 13 broadcast-plan tests also passed ASan/UBSan,
both with AVX2 enabled and with AVX2/F16C source groups excluded. Excluding
only AVX2 exposed unrelated normalization F16C link dependencies, so both
groups were excluded for fallback validation.

To reproduce, use the timing procedure above for issue #998, substituting this
exact backend-case regular expression::

   ^test_cpu_less_v13_(contiguous_float32xfloat32_to_bool(_swapped)?_n4194304|right_scalar_float32xfloat32_to_bool_swapped_n1048576)_benchmark$

Construct the CPU session with ``num_threads`` set to 1 or 4,
``affinity_policy="none"``, and ``spin_policy="park_immediately"``.
The binary parity tool's selector excludes swapped cases; select directly
through ``collect_test_cases_by_name`` instead.

Bool Or on AVX2
---------------

``Or`` resolves contiguous and left/right-scalar callbacks once when constructing
the descriptor. AVX2-capable builds and CPUs use unaligned 32-byte loads,
bitwise OR, a zero comparison, and masking to canonical BOOL bytes (0 or 1).
Noncanonical nonzero input bytes remain true. Scalar broadcasts load the scalar
once: true fills the output with ones without reading the vector; false
normalizes the vector. Short ranges and tails stay within the input/output
buffers, and exact vector/output aliasing is supported. Other CPUs retain the
existing portable bulk callbacks. Broadcast traversal, scheduling thresholds,
executor ownership, and input generation are unchanged.

Issue #974 measurements (2026-10-24)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Host: AMD EPYC 9V45 96-Core Processor, Linux x86-64 VM exposing four logical
CPUs (two cores, two threads per core), process affinity CPUs 0-3. GCC 13.3.0
Release build, ``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2``; runtime SIMD level 3.
Source before optimization: ``6818040``; optimized source: ``4e4660f``.
Dependencies: released Linux CPython 3.13 wheels for onnx-light 0.2.0 and
ONNX Runtime 1.31.0, NumPy 2.5.4. Dependency wheel build commits were not
available. These versions and this processor are not attributed to the older
dashboard campaign.

The eight historical names/shapes were restored as additional Or-only backend
benchmark cases. The table uses their shape-family tags; every case has the
prefix ``test_cpu_or_v7_`` and suffix
``_boolxbool_to_bool_n<output-elements>_benchmark``.

======================= ================== ============= ========
Case                    Left shape         Right shape   Elements
======================= ================== ============= ========
repeated_block_4d_large  8x32x256x64         32x1x1        4194304
repeated_block_4d        2x8x128x64          8x1x1         131072
inner_vector_4d_large    16x16x256x64        1x16x1x64     4194304
general_5d              4x1x16x1x64         1x8x1x128x1   4194304
inner_vector_4d          4x16x256x64         1x16x1x64     1048576
contiguous_3d            4x16x1024           4x16x1024     65536
contiguous_4d            3x5x17x257          3x5x17x257    65535
contiguous_2d            128x256             128x256       32768
======================= ================== ============= ========

Both runtimes used explicit thread limits of 1 or 4 and no worker affinity.
Spin-wait was disabled: ``spin_policy="park_immediately"`` for onnx-light,
both ORT ``allow_spinning`` settings zero. ORT used sequential graph execution
and one inter-op thread. Three campaigns each used 200 warm-ups and 1,000 timed
calls per backend, alternating backend order on every iteration, with a
five-second cumulative per-backend limit (not reached). CPU feeds were native
backend tensors; ORT feeds were NumPy views. Kernel recording verified
``onnx_light_cpu::Or`` before timing and was disabled during timing. Each
output matched ORT and an onnx-light built-in session warmed before registering
the accelerated kernels, with exact equality at both thread counts.

For the controlled before run, source ``4e4660f`` was built with AVX2/F16C
source groups excluded, selecting the unchanged pre-optimization Or callbacks.
Both groups were excluded because excluding only AVX2 exposes an unrelated
normalization F16C link dependency. The after run restored both groups. Or
uses neither F16C nor FMA. Separate three-campaign single-thread runs of the
original AVX2-ceiling build through the repository benchmark CLI also found
only the large repeated-block gap (1.26x), reduced by this change.

Median of the three campaign medians, milliseconds:

======================= ======= ========== ========== ========= =========
Case                    Threads CPU before ORT before CPU after ORT after
======================= ======= ========== ========== ========= =========
repeated_block_4d_large  1       0.078428   0.066701   0.060360  0.065468
repeated_block_4d        1       0.004797   0.005489   0.004056  0.005398
inner_vector_4d_large    1       0.250503   0.482887   0.254978  0.486891
general_5d              1       0.322691   0.465271   0.305884  0.476445
inner_vector_4d          1       0.064727   0.123266   0.064777  0.123405
contiguous_3d            1       0.003795   0.005358   0.003706  0.005248
contiguous_4d            1       0.003866   0.005318   0.003856  0.005238
contiguous_2d            1       0.003175   0.004376   0.003195  0.004411
repeated_block_4d_large  4       0.076526   0.054352   0.060085  0.054512
repeated_block_4d        4       0.005068   0.005819   0.004066  0.005479
inner_vector_4d_large    4       0.165474   0.284559   0.150807  0.280090
general_5d              4       0.193682   0.280603   0.171047  0.278568
inner_vector_4d          4       0.053360   0.088679   0.050621  0.087461
contiguous_3d            4       0.003866   0.005489   0.003906  0.005348
contiguous_4d            4       0.003856   0.005448   0.003936  0.005283
contiguous_2d            4       0.003175   0.004497   0.003215  0.004467
======================= ======= ========== ========== ========= =========

The historical 3.7-72.7x gaps did not reproduce. Only the large repeated-block
gap was repeatable here: before optimization, ORT led by 1.18x (one thread)
and 1.41x (four). CPU latency improved by 1.30x and 1.27x, respectively.
The four-thread residual ORT advantage is 1.10x; this is not a universal win.
All seven other cases remained faster than ORT. Some near-unchanged cases
regressed by up to 2%; no improvement is claimed for those cases.
The optimized large repeated-block campaign medians ranged from
0.060286 to 0.060501 ms (one thread) and 0.058884 to 0.060711 ms (four).
These local measurements are not confidence intervals or evidence about the
older campaign's unknown processor, input payloads, or build provenance.

Hardware-counter profiling was denied (``perf_event_paranoid=4``). Assembly
inspection showed baseline 16-byte SSE bulk loops, including repeated
scalar-broadcast comparisons. The optimized callbacks use 32-byte AVX2
``vpor``, ``vpcmpeqb``, and ``vpandn``; true broadcast scalars take the fill
path. Dispatch-pointer tests verify callback selection. This removes bulk-loop
work without adding broadcast expansion or moving allocations outside timing.
All 41 targeted broadcast/descriptor tests passed normally and under
ASan/UBSan, as did all seven generated Or backend tests. The 15 broadcast-plan
tests also passed ASan/UBSan with AVX2/F16C groups excluded.

To reproduce, use the issue #998 timing procedure above with this selector::

   ^test_cpu_or_v7_(repeated_block_4d|inner_vector_4d|general_5d|contiguous_[234]d)

Use the session policies and thread limits specified here, including disabled
spinning, and ``measure_alternating`` with the same repetition settings.
The existing benchmark CLI can also select these eight cases directly, but
uses separate backend phases and default spin policies rather than this
controlled alternating procedure.

Bool And on AVX2
----------------

``And`` also resolves contiguous and left/right-scalar AVX2 callbacks once at
descriptor construction. Each 32-byte vector compares both inputs against zero,
ORs the zero masks, then uses AND-NOT with ones to produce canonical BOOL bytes.
A bitwise AND of the input bytes would be incorrect for noncanonical true values
such as 2 and 128. A false broadcast scalar fills the output with zeros without
reading the vector; a true scalar normalizes the vector. Empty ranges return
before reading either input. Unaligned buffers, scalar tails, and exact
vector/output aliasing are supported. The portable callbacks, broadcast
traversal, executor scheduling, and tuning thresholds are unchanged.

Issue #975 measurements (2026-10-24)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Host: AMD EPYC 7763 64-Core Processor, Linux x86-64 VM exposing four logical
CPUs (two cores, two threads per core), process affinity CPUs 0-3. GCC 13.3.0
Release build with ``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2`` (runtime level 3).
Baseline source: ``766dc0c0f3ead8d8d29d9ed6021fa8724bfcab0c``; optimized
source: ``4c1aa6f``. Both builds used the same AVX2 ceiling and compiler flags.
The baseline additionally registered the eight historical And benchmark shapes,
without changing their input generation or any computation.
Dependencies: released Linux CPython 3.13 wheels for onnx-light 0.2.0 and
ONNX Runtime 1.31.0, NumPy 2.5.4. Dependency build commits were unavailable.
Neither these dependencies nor this processor are attributed to the older
dashboard measurements.

The cases have exactly the eight shapes and element counts in the issue #974
shape table above, with prefix ``test_cpu_and_v7_`` instead of
``test_cpu_or_v7_``. Both runtimes used explicit thread limits of 1 or 4, no
worker affinity, and disabled spinning (onnx-light ``park_immediately``, both
ORT ``allow_spinning`` settings zero). ORT used sequential graph execution and
one inter-op thread. Three campaigns each used 200 warm-up calls and 1,000
timed calls per backend, alternating backend order on every iteration, with a
five-second cumulative per-backend limit (not reached). Native backend tensors
fed onnx-light-cpu; NumPy views fed ORT. Kernel recording confirmed
``onnx_light_cpu::And`` and was disabled during timing. Outputs matched ORT
exactly before and after, and optimized outputs also matched built-in
onnx-light sessions run before registering the accelerated kernels.

Median of the three campaign medians, milliseconds:

======================= ======= ========== ========== ========= =========
Case                    Threads CPU before ORT before CPU after ORT after
======================= ======= ========== ========== ========= =========
repeated_block_4d_large  1       0.181157   0.110245   0.107039  0.106454
repeated_block_4d        1       0.015775   0.016606   0.013295  0.016090
inner_vector_4d_large    1       0.558665   0.980333   0.500377  0.990087
general_5d              1       0.683117   0.992628   0.584904  0.997326
inner_vector_4d          1       0.147024   0.256448   0.127973  0.254033
contiguous_3d            1       0.013159   0.015970   0.013836  0.016360
contiguous_4d            1       0.012904   0.015339   0.013546  0.016190
contiguous_2d            1       0.011266   0.013736   0.011401  0.014066
repeated_block_4d_large  4       0.159327   0.099761   0.109760  0.102686
repeated_block_4d        4       0.015749   0.016105   0.013185  0.016075
inner_vector_4d_large    4       0.340153   0.552188   0.298546  0.558806
general_5d              4       0.399198   0.563108   0.337168  0.572546
inner_vector_4d          4       0.114944   0.179800   0.104660  0.178728
contiguous_3d            4       0.013290   0.016200   0.012744  0.015599
contiguous_4d            4       0.012929   0.015619   0.012653  0.015504
contiguous_2d            4       0.011126   0.013886   0.011121  0.013846
======================= ======= ========== ========== ========= =========

Only the largest repeated-block gap reproduced: ORT led by 1.64x with one thread
and 1.60x with four. CPU latency improved by 1.69x and 1.45x, respectively.
The residual ORT advantage is approximately 1% and 7%, not a universal win.
Optimized campaign medians for this case ranged from 0.106198 to 0.107260 ms
(one thread) and 0.109424 to 0.111788 ms (four threads). The other seven
historical gaps did not reproduce, even before this change. The other broadcast
cases improved by 1.10-1.19x. Small contiguous cases changed by approximately
-4% to +5% in latency, including single-thread regressions; no decisive
improvement is claimed for them. All seven remain faster than ORT.
These results do not explain the older campaign's unknown processor or build
provenance. A later timing attempt overlapped lingering CodeQL extraction after
its tool timeout; it slowed both backends and was discarded, not included here.

Hardware-counter profiling was unavailable (``perf_event_paranoid=4``).
Assembly inspection showed baseline 16-byte SSE loops with repeated
scalar-broadcast comparisons. The optimized callbacks use 32-byte AVX2
``vpcmpeqb``, ``vpor``, and ``vpandn``; false scalar blocks take the zero-fill
path. Dispatch-pointer assertions confirm the intended callbacks ran.
No broadcast expansion or allocation was moved out of the timed workload.

To reproduce, use the issue #998 timing procedure with this selector::

   ^test_cpu_and_v7_(repeated_block_4d|inner_vector_4d|general_5d|contiguous_[234]d).*_benchmark$

Use the explicit no-spin policies above, thread limits 1 and 4, and
``measure_alternating(repeat=1000, warmup=200, max_repeat_time=5)`` three times.
The existing benchmark CLI also selects these cases, but uses separate backend
phases and default spin policies rather than the controlled alternating loop.
