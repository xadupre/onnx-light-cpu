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
* **Allocation-inclusive operator call:** call the public C++ operator API
  directly with preconstructed inputs. Include output and scratch allocations,
  parameter conversion, and coefficient/statistics preparation performed by
  each call. State which allocations are timed, the selected implementation,
  and the executor policy. This excludes graph/session dispatch and is not an
  end-to-end inference result.
* **Steady-state end-to-end:** reuse a prepared evaluator/session and time
  inference. Exclude parsing, registration, construction, and first-run setup.
* **Startup:** report serialization, registration, session construction,
  first-run preparation, and cache loading separately.

An isolated kernel result needs a steady-state end-to-end companion: it does
not establish the cost of the registered operator. Use the
:doc:`examples gallery <../examples>` or the parity drivers in
``tools/benchmark_*_parity.py`` as the starting point.

Allocation-inclusive operator-call results establish only the cost of that
operator API. A claim about graph/session speedup also requires a steady-state
end-to-end companion; do not extrapolate an operator-call ratio to a model.

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

Normalization moments (October 8, 2026)
---------------------------------------

Float32 normalization keeps the single-pass SIMD raw-moments path when its
variance is safely separated from the cancellation floor. Ill-conditioned
reductions below 8,192 elements retain the centered second pass. At or above
that threshold, 256-element blocks accumulate moments relative to their first
element and combine ``(count, mean, M2)`` states with Chan's formula in double
precision. RMS normalization is unchanged because it does not subtract a
mean.

Measurement layer: **kernel throughput**. The
``tools/normalization_moments_throughput.cc`` driver performs 20 warmups and
101 samples, batches enough calls to process at least 1,048,576 elements per
sample, and reports per-call median and interquartile bounds. The ``shifted``
distribution adds 65,536 to a repeating low-variance pattern and exercises the
cancellation path. The test host was a 13th Gen Intel Core i7-13800H pinned to
one CPU, GCC 14.2.0, Release mode, and AVX2/FMA dispatch
(``detected_simd_level=3``). Baseline source was ``0239e12``.

.. list-table:: Shifted float32 moments
   :header-rows: 1

   * - Elements
     - Baseline median (s)
     - Chan median (s)
     - Baseline mean / variance
     - Chan mean / variance
   * - 16,384
     - 1.19652813e-5
     - 1.24434531e-5
     - 65536.0078 / 0.374969423
     - 65536 / 0.374925584
   * - 65,536
     - 4.79544375e-5
     - 4.995775e-5
     - 65536 / 0.375009537
     - 65536 / 0.375009537

The double-precision references for 16,384 elements are
65,535.99980163574 and 0.37492557405494153. Chan aggregation rounds both to
the correct float32 results while adding 4.0% latency on this intentionally
ill-conditioned case. The unchanged well-conditioned SIMD path avoids this
cost. Reproduce the measurement with:

.. code-block:: bash

   cmake -S . -B build -DONNX_LIGHT_CPU_BUILD_BENCHMARKS=ON \
       -DONNX_LIGHT_CPU_BUILD_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
   cmake --build build --target normalization_moments_throughput -j8
   taskset -c 0 build/normalization_moments_throughput

BatchNormalization contiguous channels (October 6, 2026)
--------------------------------------------------------

For float32 and float64 inputs with a singleton spatial extent and at least
16 channels, BatchNormalization traverses contiguous channel ranges instead
of processing one-element spatial slices. Inference uses row tiles of up to
256 channels, including when a single wide row needs parallel scheduling.
Training uses 16-channel tiles with bounded worker-local stack storage.
Its four accumulation streams and centered variance calculation retain the
existing reduction order for each channel. Small channel counts, float16,
bfloat16, and larger spatial extents retain the generic algorithm.

Both layouts use the session executor. Layout-specific entry points remain
out of line: allowing the NC specialization to consume the generic path's
compiler inlining budget caused spatial-layout slowdowns during development.
The contiguous loops are portable C++; GCC emitted baseline SSE2 vector
arithmetic in the tested build. This is not a new AVX-specific dispatch path.

Measurement layer: **allocation-inclusive operator call**. The benchmark is
``tools/batch_normalization_throughput.cc``. It calls the public kernel API
directly, not a registered graph session. Each case includes output allocation,
coefficient preparation, and running-statistics computation. Tensor construction
is outside timing. The driver uses 20 warmups and 101 samples and prints the
median and interquartile bounds in seconds. An optional output filename saves
all output tensors and statistics as a binary stream, plus raw timings in
``<filename>.samples.csv``.

Measurements below used a shared Intel Xeon Platinum 8480C host, GCC 13.2.0,
Release ``-O3 -DNDEBUG``, ``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AUTO``, and one calling
thread pinned to CPU 0. Baseline source was ``24e13a10``. The same executable
used either the saved baseline operator library or the changed library, with
unchanged core and low-level dependencies. Two runs reversed backend order;
the table averages the two per-run medians.

.. list-table:: Shape [2000, 64], spatial extent 1
   :header-rows: 1

   * - Type / mode
     - Baseline (s)
     - Contiguous channels (s)
     - Speedup
   * - float32 inference
     - 3.384225e-4
     - 3.3218e-5
     - 10.19
   * - float32 training
     - 4.29681e-4
     - 5.2693e-5
     - 8.15
   * - float64 inference
     - 3.38404e-4
     - 5.07675e-5
     - 6.67
   * - float64 training
     - 7.15997e-4
     - 1.641005e-4
     - 4.36

All outputs and running statistics were bitwise identical across 48 benchmark
configurations, including threshold/tile tails and generic spatial layouts.
No final measured regression exceeded both 20% and one microsecond.
These shared-host measurements are diagnostic, not a guarantee for other CPUs,
thread counts, compilers, or graph-session workloads. No steady-state
end-to-end companion was measured, so the speedups above apply only to the
direct operator call, not to graph/session inference.

With an already configured onnx-light-integrated shared-library build, reproduce
the changed-library run with:

.. code-block:: bash

   cmake -S . -B build -DONNX_LIGHT_CPU_BUILD_BENCHMARKS=ON
   cmake --build build --target batch_normalization_throughput -j8
   taskset -c 0 build/batch_normalization_throughput /tmp/bn-patched.bin \
       > /tmp/bn-patched.csv

For a before/after comparison, save the baseline
``liblib_onnx_light_cpu_kernels.so`` before rebuilding. Select it using
``LD_PRELOAD=/path/to/baseline/liblib_onnx_light_cpu_kernels.so`` with the same
command and a different output filename, then compare the binary outputs
using ``cmp``. Verify the selected library with ``LD_TRACE_LOADED_OBJECTS=1``;
``LD_LIBRARY_PATH`` alone may not override the executable's RPATH.

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

AVX-512VL narrow vectors and tails (October 4, 2026)
----------------------------------------------------

**No production change.** The following is a direct-kernel feasibility probe,
not evidence of a steady-state end-to-end improvement. No AVX-512VL runtime
helper or dispatch tier was added.

Candidate inventory
~~~~~~~~~~~~~~~~~~~

* ``NotBool_AVX512`` processes 64 bytes at a time and has a scalar remainder
  of up to 63 bytes. A 256-bit AVX-512VL/BW compare with masked loads/stores
  can avoid that remainder. The existing AVX2 path processes 32 bytes at a
  time and also has a scalar remainder.
* ``AbsFloat32_AVX512`` processes 16 floats at a time and has a scalar
  remainder of up to 15. A 256-bit AVX-512VL/F masked tail is possible,
  but the existing 256-bit AVX kernel already covers eight floats at a time.
* ``CastInt64ToFloat32_AVX512DQ`` converts eight int64 values into eight
  floats; a 256-bit VL/DQ variant would convert four at a time. This host
  exposes DQ, but halving the input width does not itself establish a win.
  Split's 4x4 transpose instead processes batches of 16 rows by loading four
  512-bit vectors, each of which holds four 16-byte rows; narrowing it would
  require a different transpose, not just a masked tail.
  AVX-512VNNI is **not** exposed on this host, so no VL/VNNI result is
  claimed. VL is a separate CPUID feature, not implied by AVX-512F, BW, or
  DQ; a production implementation would need its own OS-gated capability
  check and per-source ``-mavx512vl`` (plus the relevant subset flags).

Protocol and raw measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared Linux runner, AMD EPYC 9V74 (two physical cores, four logical CPUs),
GCC 13.3.0, ``-O3 -DNDEBUG -std=c++20``. CPU set ``0`` (one pinned
thread), no other backend running; host flags include AVX2, AVX-512F/BW/DQ/VL
but not AVX-512VNNI. Source: ``8d27b4b`` (refreshed ``origin/main``).
Existing kernel source was compiled with ``-mavx2`` or
``-mavx512f -mavx512bw`` as appropriate. Experimental functions were
compiled only in a disposable probe with
``target("avx512f,avx512bw,avx512vl")``; the 256-bit Not candidate uses
32-byte vectors and a masked remainder of 1–31 bytes. The 512-bit masked
alternative uses only F/BW, 64-byte vectors and a masked remainder of
1–63 bytes. No library code or dispatch was changed.

The disposable probe source was not retained, so its exact input-generation
and timing code and its complete build/run commands are unavailable. The raw
measurements below are therefore a historical feasibility observation, not a
reproducible benchmark baseline, and must not justify a production change.
Any future evaluation must add its probe and commands to the repository before
using new measurements to alter dispatch.

Each function used the same preallocated input/output, 100 warmups and seven
rotated-order runs (30,000 calls per run below 65,537 bytes; 100 at 65,537).
Values are **raw nanoseconds per call**, in run order, not best times.
Before each timed phase, every output byte was checked against the scalar
logical-not contract (including noncanonical input bytes 2 and 255) and
two output guard bytes were checked for overwrite. Small shapes fit in cache.

.. code-block:: text

   bytes  existing AVX2 (seven samples)                     existing AVX-512BW (seven samples)
     15   7.37 6.77 7.15 6.77 7.24 6.77 6.77                 6.90 6.90 6.50 6.90 6.90 6.50 7.31
     31   7.58 7.58 7.58 7.58 7.58 8.04 7.58                12.54 12.07 12.48 12.47 12.01 11.78 12.07
     33   3.25 3.25 3.25 3.85 3.50 2.98 2.98                20.32 20.30 20.30 20.57 20.30 20.30 20.76
     63   7.58 8.06 7.58 8.00 7.58 8.02 7.58                12.11 11.64 11.64 11.64 12.18 11.64 11.64
    127   7.04 7.04 7.04 7.04 7.04 7.04 7.27                11.64 11.84 11.86 11.64 11.64 11.96 11.64
    255   8.12 8.54 8.12 8.12 8.35 8.12 8.35                12.72 12.72 12.94 12.75 12.72 12.96 12.78
  65537   1246.67 1247.98 1246.38 1249.57 1264.80 1245.87 1247.97
          1280.43 1346.43 1347.53 1279.22 1280.23 1287.43 1280.42 (AVX-512BW)

   bytes  candidate VL/BW 256 (seven samples)                 alternative masked 512 F/BW (seven samples)
     15   2.98 3.36 2.98 2.98 2.98 2.98 2.98                 7.85 7.85 7.87 7.87 7.85 8.28 7.87
     31   2.98 2.98 2.98 2.98 2.98 2.98 2.98                 7.96 9.38 7.87 7.87 8.33 7.85 7.92
     33   8.49 8.45 8.12 8.12 8.12 8.56 8.12                 7.87 7.93 8.29 8.19 8.33 7.85 7.85
     63   2.98 2.98 2.98 2.98 2.98 2.98 2.98                 2.98 2.98 2.98 2.98 2.98 2.98 2.98
    127   3.52 3.59 3.52 3.73 3.52 3.52 3.52                 2.71 2.71 2.71 2.71 2.71 2.71 2.71
    255   5.09 4.87 5.14 4.87 4.87 4.87 4.88                 3.25 3.25 3.25 3.25 3.25 3.25 3.25
  65537   1337.41 1337.41 1364.55 1339.41 1397.00 1337.31 1338.52
          1280.12 1280.73 1282.53 1280.32 1282.43 1288.23 1358.84 (masked 512)

For float32 Abs, a separate AVX-512VL/F 256-bit loop with a masked
1–7-float tail was compared against the existing AVX 256-bit and AVX-512F
512-bit functions with the same protocol (7 runs, 30,000 calls below
65,537 elements; 100 calls at 65,537). Bitwise parity was checked for
negative zero, infinity and NaN; output guards were checked too:

.. code-block:: text

   floats   existing AVX 256 (ns/call)                      existing AVX-512F (ns/call)                     candidate VL 256 (ns/call)
       15   3.52 3.94 3.52 3.52 3.52 3.52 3.52             3.79 3.79 3.79 3.79 4.12 3.79 3.79             2.95 2.71 2.71 2.71 2.71 2.71 2.71
       32   2.17 1.90 1.90 1.90 1.90 1.90 1.90             2.95 2.71 2.71 2.71 2.71 2.71 2.71             4.33 4.33 4.33 4.62 4.33 4.33 4.33
     1025   35.96 35.74 35.73 37.04 35.73 38.52 35.80   37.02 35.73 35.73 35.73 35.73 35.73 35.74   72.01 73.21 72.43 72.01 73.17 72.01 72.01
    65537   4857.50 4817.44 4802.72 4819.54 4811.94 4823.05 4808.83
            4958.66 4961.06 5022.85 4962.56 4974.88 4962.86 4953.74 (AVX-512F)
            4916.39 4961.56 4922.70 4926.61 5100.47 4959.96 4906.98 (VL)

The narrow masked candidate does win some tail-heavy **kernel-only** cases
(Not 31 bytes: median 2.98 ns versus 7.58 ns AVX2 and 12.07 ns AVX-512BW),
but loses at 33 bytes (8.12 ns versus 3.25 ns AVX2), and at 127/255 bytes
the existing-ISA masked 512 alternative is faster. The Abs candidate loses
substantially at 32 and 1,025 floats. A shape-specific dispatch could avoid
those regressions but would add feature probing, build flags and branches for
a few nanoseconds of isolated short-tensor work. These measurements neither
prove an end-to-end win nor control AVX-512 frequency effects on a dedicated
machine. Retain existing dispatch and scalar tails until a representative
end-to-end benchmark demonstrates a repeatable improvement.
