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

.. _l-cast-simd-measurements:

Cast SIMD measurements (September 16, 2026)
-------------------------------------------

The Cast fast paths convert contiguous float32/float16 buffers with F16C and
float32/bfloat16 buffers with AVX2. Runtime CPU/OS checks and compiled-source
availability gate dispatch; AVX2 alone does not establish F16C support.
The existing codec handles other pairs and scalar tails. Half-to-float vectors
containing NaNs also use the codec to preserve signaling NaN bits. The
float-to-half path canonicalizes NaNs in SIMD and saves/restores MXCSR,
including exception masks and flags: F16C ignores ``_MM_FROUND_NO_EXC``.
No parallel thresholds changed.

Environment and protocol
~~~~~~~~~~~~~~~~~~~~~~~~

* Shared Linux runner, AMD EPYC 9V74, two physical cores/four logical CPUs.
  Both runtimes were restricted to CPU set ``0,2``, with one- and two-thread
  policies; workers were otherwise unpinned.
* GCC 13.3.0, Release ``-O3 -DNDEBUG``,
  ``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2``, Python 3.13.15, NumPy 2.5.3,
  ONNX Runtime 1.30.0. ORT's ISA selection was not capped; the host also
  supports AVX-512.
* Baseline CPU source ``e0f43c0`` and the SIMD implementation accompanying
  this report were freshly built against the same onnx-light source,
  ``e03a1ba561451cf991788368397011d35010aa7a``. The release wheel lacked
  the runtime recording API required by this checkout.
* Existing backend benchmark cases, 20 warmups, 201 samples, two-second
  per-phase cap. Builds were idle during measurement. Two full runs reversed
  baseline/SIMD order; each case measured CPU before creating its ORT session.
  The table reports the median of the two per-run medians, in seconds.
* Untimed ``cpu_kernel_paths`` probes confirmed
  ``Cast.float32_to_float16.f16c``, ``Cast.float16_to_float32.f16c``,
  ``Cast.float32_to_bfloat16.avx2`` and ``Cast.bfloat16_to_float32.avx2``.

Reproduce each build's measurement with its source tree on ``PYTHONPATH``:

.. code-block:: bash

   taskset -c 0,2 python -m onnx_light_cpu benchmark \
       --test '^test_cpu_cast_.*_(float32_to_float16|float16_to_float32|float32_to_bfloat16|bfloat16_to_float32|float32_to_float32)_benchmark$' \
       --threads 1 --repeat 201 --warmup 20 --max-repeat-time 2 \
       --onnxruntime --output /tmp/cast-one-thread.xlsx

Repeat with ``--threads 2`` and a different output filename. Keep the raw
and aggregated workbook sheets, including path diagnostics and percentiles.

One-thread end-to-end results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Shape / conversion
     - Baseline (s)
     - SIMD (s)
     - ORT (s)
     - ORT / SIMD
   * - 65537, float32 → float16
     - 2.36236e-4
     - 6.685e-6
     - 8.238e-6
     - 1.232
   * - 65537, float16 → float32
     - 7.3977e-5
     - 7.176e-6
     - 8.087e-6
     - 1.127
   * - 1048576, float32 → float16
     - 4.676337e-3
     - 7.8058e-5
     - 7.9479e-5
     - 1.018
   * - 1048576, float16 → float32
     - 1.161491e-3
     - 8.6660e-5
     - 7.8894e-5
     - 0.910
   * - 256 × 4096, float32 → float16
     - 4.659887e-3
     - 7.8097e-5
     - 7.9389e-5
     - 1.017
   * - 256 × 4096, float16 → float32
     - 1.162013e-3
     - 8.6741e-5
     - 7.8769e-5
     - 0.908

The scalar, 16-element and 1024-element float16 cases took approximately
1.87–1.98 microseconds, at least 2.18 times faster than ORT. All twelve
one-thread float16 cases met 0.9x in this run, but the large reverse conversion
is close to the boundary: its second-run p10–p90 interval was
8.7091e-5–9.1728e-5 seconds.

For 1048576 elements, float32 → bfloat16 improved from 3.50051e-4 to
1.17772e-4 seconds; bfloat16 → float32 changed from 8.2494e-5 to
8.1191e-5 seconds. ORT's Python interface rejected bfloat16 input/output
conversion, so these rows have no ORT speedup claim. Additional integer SIMD
paths are deferred; their existing codec semantics remain unchanged.

Remaining bottleneck and variability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two-thread large-vector medians were 4.3390e-5 seconds (float32 → float16,
1.062x ORT) and 4.6545e-5 seconds (float16 → float32, 0.939x). One matrix run
was much slower despite identical element counts: the reverse path ranged
from 4.6610e-5 to 7.9379e-5 seconds across runs. A separate longer confirmation
(100 warmups, 1001 samples) measured 4.8593e-5 and 4.8744e-5 seconds for the
reverse vector/matrix cases, only 0.878x and 0.873x ORT. Thus these shared-runner
results do **not** establish a stable two-thread parity gate.

A separate one-core, preallocated ``CastConvert`` timing comparison used
201 samples, 20 warmups and alternating baseline/SIMD calls. At 1048576
elements, float16 → float32 fell from 1.159815e-3 to 8.6440e-5 seconds.
That direct-call time is already approximately the one-thread end-to-end
8.6660e-5 seconds: the remaining serial cost is in the conversion/memory loop,
not registration or tensor allocation. The loop reads/writes six MiB per call
and retains a NaN-preservation check for every eight elements. The direct
identity-copy control was essentially unchanged (9.7877e-5 versus
9.8257e-5 seconds). These ctypes timings include call/dispatch overhead and
use normally distributed finite inputs, unlike the backend corpus; they are
diagnostic, not a subtraction-based runtime-overhead estimate.

Hardware performance counters were unavailable (``perf_event_paranoid=4``),
so instruction-versus-memory stalls and the two-thread scheduling variation
remain unseparated. Further work should profile that loop and executor
handoff on a dedicated runner before changing parallel thresholds or claiming
stable multithreaded parity.
