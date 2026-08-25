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

Binary kernels now register tuning ABI 1 under exact operator and left-input
element-type keys with ``library="onnx_light_cpu"`` and
``implementation="broadcast_plan"``. The immutable per-session parameters are:

* ``parallel.bulk_threshold_bytes`` (portable default: 1 MiB);
* ``parallel.block_threshold_bytes`` (portable default: 1 MiB);
* ``parallel.scalar_threshold_bytes`` (portable default: 256 KiB);
* ``parallel.target_block_bytes`` (portable default: 1 MiB).

The registry validates and resolves these parameters before execution. The hot
path reads only the configured typed values; it performs no registry lookup,
string lookup, allocation, or lock. All loop families may use the executor's
full effective thread count when enough target-sized blocks exist.

Remaining bottlenecks
---------------------

True low-precision SIMD
~~~~~~~~~~~~~~~~~~~~~~~

FP16/BF16 ``Add``, ``Sub``, and ``Mul`` currently use typed bulk loops, but
they still widen and narrow individual values. ``Sub`` FP16 remained the only
measured type with a median regression after the first pass. Add dedicated
AVX-512FP16, AVX-512BF16 where applicable, F16C/AVX2 conversion-vector, NEON
FP16, and SVE/SVE2 implementations. Unsupported instruction sets must retain
the portable bulk loop.

Integer and predicate bulk coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only ``Add``, ``Sub``, and ``Mul`` gained typed integer bulk loops.
``Div``, ``Mod``, comparisons, logical operators, bitwise operators, shifts,
and integer ``PRelu`` still rely on per-element type-erased callbacks in at
least one important loop family.

Add typed contiguous and scalar-broadcast loops for every supported signature.
Integer ``Div``/``Mod`` and ``BitShift`` must validate invalid inputs once
before entering an unchecked bulk loop rather than repeating validation in
each element callback. Comparisons must emit canonical byte ``BOOL`` without
materializing wider masks. Signed overflow behavior must remain bit-exact and
free of C++ undefined behavior.

Expensive arithmetic
~~~~~~~~~~~~~~~~~~~~

``Pow``, floating ``Mod``, and mixed-type signatures need separate treatment
from bandwidth-bound arithmetic. Their compute cost may justify parallelism
below the current byte thresholds, while integer exponent validation and
mixed input widths change profitable block sizes. Calibration must determine
whether tuning ABI 2 needs a right-input-type discriminator; ABI 1 deliberately
shares one profile across signatures with the same operator and left type.

General broadcast traversal
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contiguous, scalar, repeated-block, and vector-inner families can call bulk
loops. General strided traversal still advances offsets and invokes a
type-erased scalar callback in its inner loop. Add prepared specializations
for rank 2-4 column/channel broadcasts, collapse invariant dimensions, seed
offsets once per worker range, and keep division/modulo out of per-element
indexing.

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

Register deterministic calibration callbacks for every priority key. A
callback must:

#. create fixed inputs and caller-preallocated outputs for each relevant loop
   family;
#. compare every candidate against the forced-serial result before timing it;
#. warm up each candidate and retain raw samples, median, and dispersion;
#. jointly select thresholds and target block size;
#. reject a candidate that improves throughput but regresses small-tensor p90
   by more than 2%;
#. obey explicit duration and memory limits;
#. publish through onnx-light's atomic persistent tuning cache.

Portable defaults remain available when no exact processor profile exists.
Existing sessions retain their resolved immutable configuration when a later
calibration publishes a new generation. Python and CLI inspection must show
the exact key, source profile, four configured byte values, and effective
session thread count.

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

Published comparisons use ONNX Runtime's normal CPU execution provider. Each
row records the backend case name, operator, complete type signature, loop
family, shapes, byte traffic model, selected tuning values, actual
participants, SIMD level, affinity, warmups, raw samples, median, dispersion,
and correctness tolerance.

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
     - In progress
   * - Binary Perf PR02
     - Arithmetic and low-precision bulk kernels.
     - Dedicated FP16/BF16 SIMD and complete typed ``Add/Sub/Mul/Div/Mod``
       paths improve or retain every arithmetic priority group with exact
       invalid-input and overflow semantics.
     - PR01
     - Planned
   * - Binary Perf PR03
     - Comparison, logical, bitwise, shift, and PRelu bulk kernels.
     - Every supported width has contiguous and scalar-broadcast bulk paths;
       comparisons emit canonical byte ``BOOL`` and no priority group regresses.
     - PR01
     - Planned
   * - Binary Perf PR04
     - Prepared broadcast specializations.
     - Priority rank 2-4 strided patterns avoid per-element type-erased calls
       and repeated index arithmetic while the arbitrary-rank fallback remains
       correct.
     - PR02, PR03
     - Planned
   * - Binary Perf PR05
     - Processor calibration and persistence.
     - Correctness-gated callbacks jointly tune all four exposed parameters;
       cache lifecycle, overrides, inspection, and immutable-session behavior
       pass onnx-light integration tests.
     - PR01, PR04
     - Planned
   * - Binary Perf PR06
     - Final parity and runtime-overhead gate.
     - Every priority group reaches median ``1.0x`` ONNX Runtime, no case is
       below ``0.9x``, small p90 stays within 2%, and any residual runtime cost
       is assigned to a measured component rather than hidden by kernel timing.
     - PR05
     - Planned

Binary Perf PR06 completes this follow-up.
