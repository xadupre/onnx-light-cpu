.. _l-benchmark-methodology:

Benchmark Methodology
=====================

CPU inference benchmarks are easy to invalidate because kernel registration,
thread pools, allocator arenas, packed weights, and runtime sessions all retain
process-wide or session-wide state. This page lists the requirements for
benchmarks published by ``onnx-light-cpu`` and the pitfalls we have already hit.

Choose the measurement layer
----------------------------

State which layer answers the question before writing the benchmark.

Kernel throughput
    Call the typed kernel with preallocated inputs and outputs. This isolates
    arithmetic, packing, and conversion, but not model dispatch, output
    allocation, or runtime lifetime management.

End-to-end steady state
    Reuse a prepared ``ReferenceEvaluator`` or equivalent session and time one
    inference, including every allocation, arena lease, and dispatch cost.
    Exclude model parsing, kernel registration, session construction, and
    first-run preparation unless the benchmark is explicitly about startup.

Startup
    Report model serialization, registration, session construction, first run,
    and cache loading separately. Do not fold startup into steady-state samples.

At least one end-to-end measurement must accompany an isolated kernel result: a
fast micro-kernel does not prove the registered operator is fast.

Verify what is executing
------------------------

``register_kernels()`` changes a process-wide dispatch table, and a runtime
session resolves and caches a kernel on its first run. Consequently:

* resolve a built-in baseline before registering ``onnx-light-cpu``;
* create accelerated sessions after registration;
* enable kernel-usage recording for an untimed probe and assert the exact
  library-qualified kernel name, then disable it during timing;
* print the detected SIMD level, selected algorithm or tuning profile, and
  effective thread count;
* treat a missing integration extension as a skipped backend, not a successful
  accelerated measurement.

Correct output is necessary but does not prove the intended kernel ran. Verify
both dispatch identity and numerical results.

Control persistent executors
----------------------------

ONNX Runtime, ``onnx-light``, ``onnx-light-cpu``, NumPy, and the BLAS selected by
NumPy may each own a persistent worker pool whose idle workers spin, retain
affinity, and perturb a different backend without appearing in its timed
interval. On shared or uncontrolled machines, measure backends in isolated global
phases:

#. prepare identical models and inputs without constructing competing runtime
   sessions;
#. run every ``onnx-light-cpu`` and built-in ``onnx-light`` case;
#. release all of those sessions;
#. measure NumPy in its own phase when it may initialize a threaded BLAS;
#. only then construct and run every ONNX Runtime session;
#. perform cross-backend correctness checks after every timed phase.

Regenerate identical inputs from the same seed in each phase. For strict
isolation, run one backend per child process. Alternating candidates in one
process is appropriate only on a dedicated, pinned host after proving they leave
no competing spinning pools. Keep each runtime's documented default spin policy;
phase separation, rather than disabling spin, prevents cross-runtime pollution.
Never run independent CPU benchmarks concurrently on the same cores.

Threads, affinity, and spinning
-------------------------------

Published results use each backend's documented default unless the experiment
is explicitly about thread scaling. Do not reduce ONNX Runtime's thread count to
make a comparison look better. Record the requested and effective participant
counts (including the calling thread), the CPU topology (physical cores, logical
threads, SMT, hybrid P/E cores), the process CPU set and worker affinities, the
spin-before-park policy, and whether nested parallelism is disabled. A profile
calibrated for one effective thread count must not run on a pool using another.

Build and machine state
-----------------------

Performance extensions and benchmark binaries must be built in ``Release``.
Report the compiler, optimization flags, enabled ISA translation units, and the
runtime ISA selected by dispatch; a Debug extension can be correct yet several
times slower.

Ideally publishable evidence would also pin the CPU microarchitecture, cache
sizes, NUMA topology, OS and kernel, frequency governor and turbo policy, and the
exact package versions of ``onnx-light-cpu``, ``onnx-light``, ONNX Runtime,
NumPy, and BLAS. In practice this microarchitecture record is **not** captured
automatically here: reliable capture is machine-specific and the probing itself
steals time and can perturb the very measurement it documents. Note the machine
manually when it matters instead of trusting an automated snapshot.

Shared CI and development machines provide diagnostics, not parity gates.
Unexpected dispersion or changing NumPy times are evidence of contention, not an
invitation to select the best observation.

Warmup and sampling
-------------------

Warm up until lazy preparation, allocation growth, packing caches, and worker
creation have completed, and keep warmup outside every timed sample.

Bound each repeated gallery measurement by both a maximum repetition count and
two seconds of cumulative measured execution. Stop at whichever limit is
reached first, allowing a call already in progress to finish.

Retain raw samples and report at least the median and a dispersion measure
(interquartile range or percentiles). Use enough work per sample that timer and
scheduler resolution are negligible: batch nearby short calls and divide by the
batch size, reduce repetitions for large workloads without dropping below a
documented minimum, and synchronize asynchronous backends before stopping the
timer. Never report only the minimum or the fastest rerun. The batch must repeat
the same public operation; moving packing, conversion, or allocation outside the
batch changes the measured layer and requires a separate label.

Inputs and correctness
----------------------

Use identical logical inputs, shapes, attributes, and transposition flags.
Generate in a common high-precision type before rounding independently to each
tested dtype, preserving the same seed and generation order in every phase.

Validate after timing: exact results where the operator contract is bit-exact;
signed zero, infinities, NaNs, integer limits, empty tensors, and SIMD tails
where relevant; explicit dtype- and reduction-length-dependent tolerances for
floating-point reductions; and the complete output, not a prefix or checksum.
Different accumulation precision can require different tolerances without
invalidating the comparison, but document the difference rather than silently
widening tolerance until a failing backend passes.

Memory and lifetime
-------------------

Confirm whether inputs are borrowed or copied and whether outputs come from the
execution arena, I/O arena, or a backend allocator. For zero-copy claims, assert
the input address or allocator counters; for arena-reuse claims, release the
output, verify the lease count, rerun the same shape, and compare addresses.

Distinguish dynamic from constant weights: a constant-weight benchmark may
prepack once, while a dynamic-input benchmark must include packing on every call.
Do not compare one policy against the other without labeling it. Watch peak
memory when regenerating large inputs, and exclude garbage collection from timed
intervals.

Issues already faced
--------------------

The requirements above come from concrete failures. Watch for these:

* Small-size timing checks are flaky on shared CI runners: the smallest shapes
  are dominated by scheduler and timer noise, so a SIMD kernel does not reliably
  beat the built-in one at tiny sizes. Retry a few times rather than failing on a
  single observation.
* NumPy/BLAS invoked before the accelerated phase left a spinning worker pool
  that penalized the next backend measured on the same cores.
* A kernel resolved before ``register_kernels()`` kept dispatching to the
  built-in implementation, so a "fast" number was never produced by the
  accelerated kernel.
* Correct output was mistaken for correct dispatch; only kernel-usage recording
  revealed the wrong kernel had run.
* A Debug build produced correct but multiples-slower results that were reported
  as a regression.
* Emulated ISA runs (for example QEMU SVE at a non-power-of-two vector length)
  were far slower than native and skewed comparisons.
* Prepacked constant weights were compared against dynamic-input packing without
  labeling the difference, hiding per-call packing cost.

Review checklist
----------------

Before accepting benchmark evidence, verify all of the following:

* Release build and exact commits are recorded.
* The expected accelerated kernel and ISA are asserted outside timing.
* Setup, first run, and steady state are separated.
* Competing persistent pools do not coexist during a timed phase.
* NumPy/BLAS has not been invoked before the accelerated phase.
* Threads, affinity, spin policy, and CPU set are reported truthfully.
* Inputs and outputs are copied or borrowed as claimed.
* Every backend is validated with an explicit numerical contract.
* Raw samples, median, and dispersion are retained.
* Unsupported dtype/backend combinations are reported as ``not supported``.
* Shared-runner results are labeled diagnostic.
