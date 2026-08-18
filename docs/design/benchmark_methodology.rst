.. _l-benchmark-methodology:

Benchmark Methodology
=====================

Performance results are only useful when the measured implementation, execution
policy, and timed work are unambiguous. CPU inference benchmarks are especially
easy to invalidate because kernel registration, thread pools, allocator arenas,
packed weights, and runtime sessions all retain process-wide or session-wide
state.

This page defines the requirements for benchmarks published by
``onnx-light-cpu``. It applies to gallery examples, tuning experiments, pull
request evidence, and dedicated performance runners.

Choose the measurement layer
----------------------------

State which layer answers the question before writing the benchmark.

Kernel throughput
    Call the typed kernel with preallocated inputs and outputs. This isolates
    arithmetic, packing, conversion, and the kernel scheduler. It does not
    measure model dispatch, Python conversion, output allocation, or runtime
    lifetime management.

End-to-end steady state
    Reuse a prepared ``ReferenceEvaluator`` or equivalent session and time one
    inference. Include every allocation, arena lease, input adapter, output
    adapter, and dispatch cost visible to an application. Exclude model parsing,
    kernel registration, session construction, and first-run preparation unless
    the benchmark is explicitly about startup.

Startup
    Report model serialization, registration, session construction, first run,
    and cache loading separately. Do not add startup to every steady-state
    inference or silently omit lazy work from a startup claim.

At least one end-to-end measurement must accompany an isolated kernel result.
A fast micro-kernel does not prove that the registered operator is fast, and an
end-to-end regression does not identify the responsible kernel.

Verify what is executing
------------------------

``register_kernels()`` changes a process-wide dispatch table. A runtime session
resolves and caches a kernel on its first run. Consequently:

* resolve a built-in baseline before registering ``onnx-light-cpu``;
* create accelerated sessions after registration;
* enable kernel-usage recording for an untimed probe and assert the exact
  library-qualified kernel name;
* disable usage recording during timing because its lock and process-wide log
  are instrumentation overhead;
* print the detected SIMD level, selected algorithm or tuning profile, and
  effective thread count;
* treat a missing integration extension as a skipped backend, not a successful
  accelerated measurement.

Correct output is necessary but does not prove that the intended kernel ran.
Every benchmark must verify both dispatch identity and numerical results.

Control persistent executors
----------------------------

ONNX Runtime, ``onnx-light``, ``onnx-light-cpu``, NumPy, and the BLAS selected by
NumPy may each own a persistent worker pool. Idle workers can spin before
parking, retain affinity, consume scheduling quanta, and perturb a different
backend without appearing inside its timed interval.

On shared or otherwise uncontrolled machines, measure backends in isolated
phases:

#. run ``onnx-light-cpu`` before constructing ONNX Runtime or invoking a
   multithreaded NumPy/BLAS operation;
#. measure a cached built-in ``onnx-light`` session after the accelerated phase;
#. measure NumPy next;
#. construct and measure ONNX Runtime last;
#. perform cross-backend correctness checks after every timed phase.

Regenerate identical inputs from the same seed in each phase instead of
retaining every large tensor. For strict isolation, run one backend per child
process and return raw samples through a non-timed channel.

Alternating candidates in one process is appropriate only on a dedicated,
pinned host after proving that the candidates do not leave competing spinning
pools. Alternation does not make interference fair: a backend with aggressive
spinning or affinity can systematically penalize the next backend.

Never run independent CPU benchmarks concurrently on the same cores. Tool-level
parallelism is not benchmark parallelism.

Threads, affinity, and spinning
-------------------------------

Published results use each backend's documented default unless the experiment
is explicitly about thread scaling. Do not reduce ONNX Runtime's thread count
to make a comparison look better.

Record:

* requested and effective participant counts, including the calling thread;
* physical cores, logical threads, SMT, and hybrid P/E-core topology;
* the process CPU set and every worker affinity;
* spin-before-park policy and budget;
* whether nested parallelism is disabled;
* relevant environment variables and build-time ceilings.

Equal-thread results are useful diagnostics, but they are not a replacement for
the default-policy comparison. A profile calibrated for one effective thread
count must not be executed by a pool using another count.

Build and machine state
-----------------------

Performance extensions and benchmark binaries must be built in ``Release``.
Report the compiler, optimization flags, enabled ISA translation units, and the
runtime ISA selected by dispatch. A Debug extension can be functionally correct
while being several times slower.

For publishable evidence, also record:

* CPU model and microarchitecture, cache sizes, NUMA topology, and microcode;
* operating system, kernel, virtualization or container limits, and CPU set;
* frequency governor, turbo policy, thermal state, and power mode;
* package versions and exact commits of ``onnx-light-cpu``, ``onnx-light``,
  ONNX Runtime, NumPy, and BLAS;
* other CPU load before and during the run.

Shared CI and development machines provide diagnostics, not parity gates.
Unexpected dispersion or changing NumPy times are evidence of contention, not
an invitation to select the best observation.

Warmup and sampling
-------------------

Warm up until lazy preparation, allocation growth, packing caches, and worker
creation have completed. Keep warmup outside every timed sample.

Retain raw samples and report at least the median and a dispersion measure such
as interquartile range or percentiles. Use enough work per sample that timer and
scheduler resolution are negligible:

* batch nearby short calls and divide by the batch size;
* keep the output alive only when the application would do so;
* reduce repetitions for large workloads without dropping below a documented
  minimum;
* synchronize asynchronous backends before stopping the timer;
* never report only the minimum or the fastest rerun.

The batch must repeat the same public operation. Moving packing, conversion, or
allocation outside the batch changes the measured layer and requires a separate
label.

Inputs and correctness
----------------------

Use identical logical inputs, shapes, attributes, and transposition flags.
Generate in a common high-precision type before rounding independently to each
tested dtype. Preserve the same random seed and generation order in every
isolated phase.

Validate after timing:

* exact results where the operator contract is bit-exact;
* signed zero, infinities, NaNs and payloads, integer limits, empty tensors, and
  SIMD tails where relevant;
* explicit dtype- and reduction-length-dependent tolerances for floating-point
  reductions;
* the complete output, not a prefix or checksum alone.

Different accumulation precision can require different tolerances without
making the performance comparison invalid. Document the numerical difference;
do not silently widen tolerance until a failing backend passes.

Memory and lifetime
-------------------

Confirm whether inputs are borrowed or copied and whether outputs come from the
execution arena, I/O arena, or a backend allocator. For zero-copy claims, assert
the input address or allocator counters. For arena-reuse claims, release the
output, verify the lease count, rerun the same shape, and compare addresses.

Distinguish dynamic from constant weights. A constant-weight benchmark may
prepack once; a dynamic-input benchmark must include packing performed on every
call. Do not compare one policy against the other without labeling it.

Watch peak memory when regenerating large inputs. Avoid retaining expected
outputs or every size in the grid when deterministic regeneration is possible.
Exclude garbage collection from timed intervals and force it only when needed
to verify ownership or arena reuse.

Ratios and plots
----------------

Define speed-up as:

.. code-block:: text

    baseline_time / candidate_time

Values above one are faster than the named baseline. Print raw times beside the
ratio and label both numerator and denominator. The text report and plot must
use the same direction.

Do not substitute a different baseline for one dtype in the same speed-up
panel. If ONNX Runtime does not support ``bfloat16``, show raw NumPy and
``onnx-light-cpu`` times but omit the ONNX Runtime speed-up. An unsupported
backend is ``not supported``, never zero, NaN presented as a result, or a
success-shaped fallback.

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
* Speed-up direction and baseline are explicit and consistent.
* Unsupported dtype/backend combinations do not produce a ratio.
* Shared-runner results are labeled diagnostic.

