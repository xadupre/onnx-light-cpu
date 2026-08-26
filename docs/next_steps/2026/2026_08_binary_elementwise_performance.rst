Binary Elementwise Performance Follow-up
=========================================

:Date: 2026-08

**in progress**

Objective
---------

The functional Binary roadmap is complete: all 19 registered operators share
one prepared broadcast engine, execute through the session-owned executor, and
have reproducible correctness and benchmark corpora. This follow-up addresses
the performance gaps exposed by the complete end-to-end benchmark rather than
adding more operator semantics.

The final gate is:

* at least ``1.0x`` ONNX Runtime median performance for every priority
  operator/type/loop-family group;
* no priority case below ``0.9x``;
* no small-tensor p90 regression greater than 2% against the current serial
  SIMD baseline;
* no private scheduler, inference-time tuning, or registry access in the hot
  path.

Measured baseline
-----------------

The first tuning pass removed the fixed four-participant ceiling, increased
executor granularity, and added typed bulk loops for integer and
half-precision ``Add``, ``Sub``, and ``Mul``. On the measured ``Sub`` corpus:

* median speed-up over ONNX Runtime increased from ``0.155x`` to ``1.533x``;
* median onnx-light-cpu execution time improved by ``3.0x``;
* the fraction of cases faster than ONNX Runtime increased from 6.0% to 51.8%;
* the median ``n=4096`` and ``n=65536`` cases reached ``2.36x`` and ``1.93x``;
* the median ``n=1048576`` and ``n=4194304`` cases remained at only ``0.39x``
  and ``0.56x``.

These numbers are diagnostic, not a final parity claim. They cover one
operator on one host, and ONNX Runtime timing dispersion was significant for
some large broadcast cases. The complete unfiltered corpus must be rerun with
raw samples, affinity, effective thread count, and selected tuning parameters
recorded before accepting any optimization.

Current execution and tuning contract
-------------------------------------

Every worker combines both levels of parallelism: ``ExecuteRanges`` assigns an
independent output range to a session worker, and that worker invokes the bulk
SIMD loop for its range. Small inputs remain serial SIMD to avoid executor
overhead.

Binary kernels now register tuning ABI 2 under exact operator and left-input
element-type keys with ``library="onnx_light_cpu"`` and
``implementation="broadcast_plan"``. The immutable per-session parameters are:

* ``parallel.bulk_threshold_bytes`` (portable default: 1 MiB);
* ``parallel.block_threshold_bytes`` (portable default: 1 MiB);
* ``parallel.scalar_threshold_bytes`` (portable default: 256 KiB);
* ``parallel.target_block_bytes`` (portable default: 1 MiB).
* ``parallel.max_participants`` (portable default: 0, meaning the complete
  effective session executor).

The registry validates and resolves these parameters before execution. The hot
path reads only the configured typed values; it performs no registry lookup,
string lookup, allocation, or lock. The participant ceiling is applied to both
flat and multidimensional schedules.

A targeted 1,176-case pass covering ``Add``, ``Sub``, ``Mul``, ``Div``,
``Mod``, ``Pow``, ``PRelu``, ``Equal``, ``Less``, ``And`` and
``BitwiseAnd`` improved the aggregate median from ``0.394x`` to ``0.669x``
ONNX Runtime. The fraction below parity fell from 50.8% to 37.5%.
Comparison and bitwise median gains in onnx-light-cpu time were ``3.0x`` to
``4.4x``. These figures remain diagnostic: repeated large arithmetic runs
show enough host dispersion that they do not satisfy the final gate.

Remaining bottlenecks
---------------------

True low-precision SIMD
~~~~~~~~~~~~~~~~~~~~~~~

FP16/BF16 ``Add``, ``Sub``, ``Mul``, ``Div`` and ``PRelu`` now widen and
narrow vectorized blocks through the shared half-conversion kernels for
contiguous and scalar-broadcast loops. ``Mod`` and ``Pow`` use the same
block-conversion approach for contiguous loops. Add dedicated
AVX-512FP16, AVX-512BF16 where applicable, F16C/AVX2 conversion-vector, NEON
FP16, and SVE/SVE2 implementations. Unsupported instruction sets must retain
the portable bulk loop.

Integer and predicate bulk coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Div``, ``Mod``, comparisons, logical operators, bitwise operators, shifts,
and integer ``PRelu`` now have typed contiguous and scalar-broadcast loops for
every same-type signature. Comparisons emit canonical byte ``BOOL``.
Integer ``Div``/``Mod`` and ``BitShift`` validate the complete input first and
then enter unchecked typed compute loops. The remaining work in this area is
ISA-specific predicate packing and validation bulk loops; validation still
uses the prepared strided traversal.

Expensive arithmetic
~~~~~~~~~~~~~~~~~~~~

FP32 ``Pow`` now stays in FP32 rather than promoting every operation to
``long double``, and float/integer mixed signatures have typed bulk loops.
On AVX-512, exponents 0 through 5 use exact multiplications, positive finite
bases with finite exponents use the shared vector ``Log``/``Exp`` primitives,
and exceptional values fall back to ``std::pow``. Contiguous and left-scalar
loops have dedicated vector paths; small repeated blocks avoid the large-loop
dispatch overhead. The approximation is checked against ``std::pow`` with a
``3e-5`` relative tolerance.

At 4,194,304 FP32 elements on the development host, all measured contiguous,
scalar, row, per-channel, outer, and general-strided orientations reached
parity with ONNX Runtime. The narrowest result was the row broadcast at
``1.03x``; the other broadcast families reached ``1.21x`` to ``2.85x``, scalar
cases ``3.81x`` to ``4.17x``, and contiguous ``2.56x``.

General broadcast traversal
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contiguous, scalar, repeated-block, and vector-inner families can call bulk
loops. Rank 2-4 prepared plans now use fixed-size traversal state. They seed offsets
once per worker range, avoid heap allocation in workers, and dispatch their
typed inner bulk loop without the arbitrary-rank counter. The generic
strided traversal remains the correctness fallback for higher ranks and for
adapters without a compatible inner bulk loop.

On the targeted 116-case ``general`` corpus, median onnx-light-cpu time for
``n=1048576`` fell from 2.28 ms to 0.31 ms and ``n=4194304`` from 2.43 ms to
0.36 ms. Small cases were retained (3.83 to 3.64 us at ``n=4096``). ONNX
Runtime timings varied substantially on the largest group, so these are
implementation-time comparisons rather than parity claims.

Do not add an unbounded template matrix for arbitrary ranks. Retain the
current prepared general loop as the correctness fallback and specialize only
patterns demonstrated by the backend corpus and real models.

Executor granularity
~~~~~~~~~~~~~~~~~~~~

The fixed four-worker cap hid dispatch overhead but prevented large tensors
from scaling. Removing it improved large inputs only after each submitted
range was increased to roughly 1 MiB of useful traffic. One portable block
size is still unlikely to fit every processor, data type, and operation.

Calibration must jointly search crossover and block size. It should measure
candidate thresholds ``0``, 64 KiB, 256 KiB, 1 MiB, 4 MiB and 16 MiB, while
the executor derives participant count from useful blocks and the session
limit. Record the actual participant count after SIMD/cache-line alignment;
the requested count alone is not sufficient evidence.

Runtime and allocation overhead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measure both a preallocated ``BinaryBroadcastPlan::Execute`` layer and the
complete ``ReferenceEvaluator`` path. If the preallocated kernel reaches
parity but end-to-end execution does not, profile output allocation, tensor
metadata construction, plan-cache access, and runtime dispatch separately.
Do not compensate for runtime allocation overhead by over-parallelizing the
compute loop.

Calibration and persistence
---------------------------

Deterministic calibration callbacks are registered for every Binary tuning
key. Each callback jointly evaluates seven complete profiles spanning serial,
coarse, balanced, fine-grained, compute-heavy and participant-limited
schedules in addition to the portable profile. Calibration uses equal-shape,
scalar-broadcast and prepared rank-4 cases and:

* creates deterministic inputs and caller-preallocated outputs;
* compares every candidate byte-for-byte against forced-serial output;
* records repeated samples and scores candidates relative to serial execution;
* jointly selects all four byte parameters and the participant ceiling;
* rejects profiles that regress small-tensor p90 by more than 2%;
* obeys explicit duration and cumulative live-memory limits;
* publishes atomically through onnx-light's execution-specific registry.

Portable defaults remain available when no exact processor profile exists.
Existing sessions retain their resolved immutable configuration when a later
calibration publishes a new generation. Python and CLI inspection must show
the exact key, source profile, four configured byte values, participant
ceiling, and effective session thread count.

Benchmark and acceptance matrix
-------------------------------

The benchmark matrix crosses:

* all 19 operators and every manifest signature;
* contiguous, left/right scalar, row, per-channel, outer, and general
  broadcasts, including swapped non-commutative operands;
* output sizes 4,096, 65,536, 1,048,576, and 4,194,304;
* session limits 1, 2, 4, physical cores, and logical cores;
* portable scalar, available SIMD levels, portable tuning defaults, and exact
  calibrated profiles.

Published comparisons use ONNX Runtime's normal CPU execution provider. The
tool publishes in-memory calibrated Binary profiles before creating benchmark
sessions and records those calibration reports with the backend case name,
operator, complete type signature, loop family, shapes, byte traffic model,
SIMD level, affinity, warmups, raw samples, median, dispersion, and correctness
tolerance. ONNX Runtime CPU signatures that are not implemented, currently the
priority BF16 arithmetic and PRelu cases on the measured build, are reported as
not comparable rather than silently dropped or substituted with FP32.

The final implementation pass also replaced callback-per-element bulk loops
for logical operators and FP32 PRelu with direct vectorizable/SIMD loops.
FP32 Pow scalar exponents 0 through 5 use multiplication/copy/fill kernels
instead of libm. At 65,536 elements on the development host this improved:

* logical And by 4.1x to 12.5x in onnx-light-cpu time, reaching 1.46x to 2.62x
  ONNX Runtime across the measured families;
* the previously slow PRelu families by about 5x, while the explicit FP32 SIMD
  contiguous and scalar paths reached 4.5x to 5.5x ONNX Runtime;
* Pow right-scalar by 33x to 35x and integer-valued per-channel exponents to
  about 9.75x ONNX Runtime.

A complete 1,008-cell development-host pass found all comparable matrix cells
and classified 280 BF16 cells as unsupported by ONNX Runtime CPU. Its aggregate
median was 1.41x and every operator/type/loop-family group except contiguous
FP32 Div (0.983x) reached a 1.0x median. The run did not pass the per-case or
small-p90 gates: the shared 96-core host showed severe large-case executor
dispersion. These results are diagnostic and do not replace the required
pinned dedicated-machine acceptance run.

Pull-request sequence
---------------------

.. list-table::
   :header-rows: 1
   :widths: 10 25 43 14 8

   * - PR
     - Scope
     - Merge criterion
     - Depends on
     - Status
   * - Binary Perf PR01
     - Reproducible baseline and diagnostics.
     - The unfiltered corpus records raw samples, selected tuning, actual
       participants, and separate preallocated/end-to-end timings. Results are
       grouped by operator, complete signature, loop family, and size.
     - Completed Binary roadmap
     - Completed
   * - Binary Perf PR02
     - Arithmetic and low-precision bulk kernels.
     - Dedicated FP16/BF16 SIMD and complete typed ``Add/Sub/Mul/Div/Mod``
       paths improve or retain every arithmetic priority group with exact
       invalid-input and overflow semantics.
     - PR01
     - Implemented; native half ISA paths remain
   * - Binary Perf PR03
     - Comparison, logical, bitwise, shift, and PRelu bulk kernels.
     - Every supported width has contiguous and scalar-broadcast bulk paths;
       comparisons emit canonical byte ``BOOL`` and no priority group regresses.
     - PR01
     - Implemented; predicate ISA paths remain
   * - Binary Perf PR04
     - Prepared broadcast specializations.
     - Priority rank 2-4 strided patterns avoid per-element type-erased calls
       and repeated index arithmetic while the arbitrary-rank fallback remains
       correct.
     - PR02, PR03
     - Implemented
   * - Binary Perf PR05
     - Processor calibration and persistence.
     - Correctness-gated callbacks jointly tune all five exposed parameters;
       cache lifecycle, overrides, inspection, and immutable-session behavior
       pass onnx-light integration tests.
     - PR01, PR04
     - Implemented
   * - Binary Perf PR06
     - Final parity and runtime-overhead gate.
     - Every priority group reaches median ``1.0x`` ONNX Runtime, no case is
       below ``0.9x``, small p90 stays within 2%, and any residual runtime cost
       is assigned to a measured component rather than hidden by kernel timing.
     - PR05
     - Implemented; dedicated-machine acceptance pending

Binary Perf PR06 implements the final gate; the follow-up closes only after a
pinned dedicated-machine run satisfies it.
