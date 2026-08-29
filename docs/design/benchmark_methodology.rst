.. _l-benchmark-methodology:

Benchmark Methodology
=====================

This is a reference for results published by ``onnx-light-cpu``. It records
the safeguards needed to interpret this project's kernels and runtimes; it is
not a general CPU benchmarking guide.

Choose the measurement
----------------------

Label the result with exactly one of these layers:

* **Kernel throughput:** call a typed kernel with preallocated inputs and
  outputs. It measures arithmetic, packing, and conversions, but not runtime
  dispatch or allocations.
* **Steady-state end-to-end:** reuse a prepared evaluator/session and time
  inference. Exclude parsing, registration, construction, and first-run setup.
* **Startup:** report serialization, registration, session construction,
  first-run preparation, and cache loading separately.

An isolated kernel result needs a steady-state end-to-end companion: it does
not establish the cost of the registered operator. Use the
:doc:`examples gallery <../examples>` or the parity drivers in
``tools/benchmark_*_parity.py`` as the starting point.

Prove the selected kernel ran
-----------------------------

``register_kernels()`` changes process-wide dispatch, while sessions resolve
and cache kernels on their first run. Therefore:

* Resolve a built-in ``onnx-light`` baseline before registration and construct
  accelerated sessions after it.
* For an untimed probe, enable kernel-usage recording and assert the expected
  library-qualified kernel name; disable it while timing because its mutex
  changes per-call cost.
* Report detected SIMD level, selected algorithm/tuning profile when relevant,
  and the effective thread count. A missing integration extension is ``not
  supported``, not an accelerated result.
* Check complete outputs after every backend phase, using the operator's
  exact contract or an explicit dtype- and reduction-size-appropriate
  tolerance. Correct output alone does not prove dispatch.

Keep executors separate
-----------------------

``onnx-light-cpu``, ``onnx-light``, ONNX Runtime, NumPy, and BLAS can retain
worker pools. Do not construct or run competing backends during a timed phase.
Run the ``onnx-light-cpu``/built-in phase, release its sessions, then measure
NumPy and ONNX Runtime in separate phases; regenerate inputs from the same seed
per phase. Use separate child processes when strict isolation is required.

Report the requested and effective participant counts (including the caller)
and relevant CPU configuration: CPU model/topology, process CPU set or
affinity, and any non-default spin or nested-parallelism setting. Use a
``Release`` build and identify the selected ISA. Shared-runner measurements
are diagnostic, not parity gates.

Use the script safeguards
-------------------------

Keep the benchmark scripts' default ``--warmup``, ``--repeat``, and
``--max-repeat-time`` settings unless the experiment documents a reason to
change them. Warmups and timed samples are separately bounded; do not time
first-run preparation. Retain the raw samples and report their median plus the
dispersion the driver emits (for example, IQR or percentiles), rather than a
best observation.

Use identical models, inputs, shapes, and attributes for every backend. If
constant weights are prepacked, label that result separately from dynamic
weights, whose packing belongs in every invocation.
