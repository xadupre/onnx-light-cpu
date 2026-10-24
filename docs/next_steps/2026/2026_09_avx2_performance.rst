AVX2 Kernel Performance Follow-up
=================================

:Date: 2026-09

**in progress**

Objective
---------

The AVX-512 optimization phase is complete. The next performance work focuses
on AVX2, where several kernels still use narrower register tiles, conversion
paths, scalar tails, or scheduling defaults inherited from the portable
implementation.

The objective is to bring the priority AVX2 corpus to the same implementation
quality as the completed AVX-512 paths without regressing AVX-512, SSE2, or
portable execution. Correctness and ONNX semantics remain identical across
the runtime-selected implementations.

Measurement contract
--------------------

`#614 <https://github.com/xadupre/onnx-light-cpu/pull/614>`_ added
``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2``. Every optimization can therefore be
measured both on native AVX2 hardware and, for controlled A/B diagnosis, on
the same AVX-512 host with AVX-512 and AMX dispatch disabled.

Published decisions use:

* Release builds with the same compiler and runtime settings for both sides;
* pinned physical cores and separate process phases for onnx-light-cpu and
  ONNX Runtime;
* raw samples, medians, dispersion, CPU model, affinity, thread count, and
  detected SIMD level;
* one-thread and physical-core runs so compute, packing, memory bandwidth, and
  scheduling gaps are not conflated;
* the existing backend benchmark CLI and operator-specific parity tools.

An AVX2 change must improve a measured priority case, preserve all differential
tests and vector-tail cases, and leave the automatic AVX-512 build unchanged.
The final corpus target is at least ``1.0x`` ONNX Runtime median performance
for each priority family, with no priority case below ``0.9x``.

Split middle-axis follow-up
--------------------------

Issue `#992 <https://github.com/xadupre/onnx-light-cpu/issues/992>`_ reports
0.301951 ms for onnx-light-cpu versus 0.217037 ms for ONNX Runtime on
``test_cpu_split_middle_axis_float32_benchmark``. The gap was **not
reproduced** on 2026-10-24. No production optimization was retained.

The unchanged backend case uses FP32[64,128,128], axis 1, and an INT64[3]
input containing [32,64,32]. It materializes three outputs, copying 4 MiB
across 64 input rows. The current generic implementation already copies all
outputs row by row and dynamically distributes large rows through the runtime
executor. The narrow four-output SIMD transposes do not apply to this case.

The sandbox processor was an AMD EPYC 7763 under virtualization, with four
visible logical CPUs, two physical cores, and 32 MiB L3. The working branch
matched its upstream after fetching remote refs. The baseline was
``4d886f27af0c3fa648c36c5da11e38f1344a0e3c``; the after run added only the
regression test and this report. Both used onnx-light-cpu 0.2.0, GCC 13.3.0,
Release ``-O3 -DNDEBUG``, and ``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2``
(runtime-reported SIMD level 3). Dependencies were the onnx-light 0.2.0
CPython 3.13 Linux release wheel, ONNX Runtime 1.31.0, and NumPy 2.4.2.
Dependency source commits were not available from the wheel metadata.
Bindings linked the wheel's exact runtime libraries with the matching native
Python ABI (``ONNX_LIGHT_CPU_PYTHON_STABLE_ABI=OFF``).

Each table entry lists three independent process medians in milliseconds.
Each process used 100 warmups and 1,001 timed session calls per backend,
with a 10-second cap that was not reached. Inputs, output allocation,
validation, and session execution followed the existing benchmark harness:
backend tensors feed onnx-light-cpu, NumPy arrays feed ORT, and session
construction and feed conversion are outside timing. Output materialization
remains inside timing.

.. list-table::
   :header-rows: 1
   :widths: 20 15 32 33

   * - Revision
     - Threads / affinity
     - onnx-light-cpu (ms)
     - ONNX Runtime (ms)
   * - Before
     - 1 / CPU 0
     - 0.117488, 0.117028, 0.115606
     - 0.121677, 0.124011, 0.127358
   * - After (tests only)
     - 1 / CPU 0
     - 0.115806, 0.117138, 0.115596
     - 0.124983, 0.124502, 0.123120
   * - Before
     - Default / CPUs 0-3
     - 0.071794, 0.073627, 0.073737
     - 0.080831, 0.079519, 0.081452
   * - After (tests only)
     - Default / CPUs 0-3
     - 0.073156, 0.072124, 0.072565
     - 0.080790, 0.080830, 0.080641

The primary comparison is one-thread execution: neither backend has workers
spinning. Additional default-thread runs retain each runtime's default
thread/spin policy (onnx-light adaptive spin, ORT default spinning), so they
are diagnostic rather than a spin-matched comparison. ORT uses sequential
execution, one inter-op thread, and the CPU execution provider. The harness
runs the CPU phase before constructing ORT, then measures ORT; it does not
alternate backend order. Both phases use the same process affinity, with
onnx-light worker affinity disabled.

Reproduce a one-thread run from the built source tree with dependencies on
``PYTHONPATH``::

    taskset -c 0 python -m onnx_light_cpu benchmark \
      --test '^test_cpu_split_middle_axis_float32_benchmark$' --dtype float32 \
      --threads 1 --repeat 1001 --warmup 100 -t 10 --onnxruntime \
      -o /tmp/split.xlsx

For the default-thread diagnostic use ``taskset -c 0-3`` and ``--threads 0``.
Hardware-counter profiling was denied (``perf_event_paranoid=4``), and no
counter-verified bottleneck or new performance improvement is claimed.
The original campaign lacks per-case commits and thread/spin settings, so
these results cannot establish why its older measurement differed.

The focused C++ regression compares the exact middle-axis shape and nearby
row counts against onnx-light's built-in Split, byte for byte. It also checks
the parallel threshold at 15/16/17 rows, empty output segments, concurrent
execution, single-thread execution, and suppression of nested scheduling.
All 13 Split tests pass. Since the reported gap is not repeatable here,
changing the copy algorithm or benchmark workload is not justified by this
evidence.

Abs cache-capacity boundary follow-up
-------------------------------------

Issue `#994 <https://github.com/xadupre/onnx-light-cpu/issues/994>`_ lists
``test_cpu_abs_n4194304_float32_benchmark``. On a processor with 32 MiB L3,
its 16 MiB input plus 16 MiB output reaches cache capacity. The AVX/AVX2
default previously required one additional input byte before enabling
streaming stores. It now includes equality: the cache has no headroom for
other runtime data at this boundary. Smaller inputs, unknown cache topology,
explicit tuning, and the AVX-512 default are unchanged.

The 2026-10-23 sandbox investigation used an AMD EPYC 7763 under virtualization
(four logical CPUs, two visible physical cores, 32 MiB L3), GCC 13.3.0,
Release ``-O3 -DNDEBUG``, and the AVX2 ceiling (reported SIMD level 3).
The baseline was repository commit ``5a9b0e7364e174409e2ce941eb6b91622a3add59``;
the after build used that revision plus this threshold change. Remote refs
were fetched before measuring and the working branch matched its upstream.

The existing ``abs_throughput`` tool now uses safe runtime dispatch instead
of calling AVX-512 directly, and accepts ``default``, ``cached``, or
``streaming`` as its third argument. It reports median, p10, and p90 latency.
Measurements used affinity ``taskset -c 0``, one thread, no worker pools or
spinning, ten warmups, and 1,001 single-call samples per process. Allocation,
initialization, and output validation are outside timing; both versions
reuse the same input/output buffers. Store policies ran in separate sequential
processes, including reversed cached/streaming order after the change.

For FP32[4194304], the three before-default medians were 0.74775, 0.70656,
and 0.76158 ms. Two after-default medians were 0.67431 and 0.67495 ms:
approximately 10% lower than the median before result. After p10/p90 ranges
were 0.66221--0.68653 and 0.66314--0.69239 ms; before ranges were
0.67881--0.81704, 0.66025--0.76823, and 0.71164--0.82140 ms.
The overlapping distributions and variable cached-store results should not
be interpreted as a universal speedup. Aligned-count neighbours 4194303 and
4194305, small/medium inputs, and 8388608 elements were also exercised.
The focused C++ suite checks the exact default threshold, special-value bits,
unaligned tails, in-place execution, and concurrent scheduling.

These are **single-thread kernel diagnostics, not end-to-end backend parity
results**. The original 1.339158x ORT advantage could not be reproduced or
disproved: ``onnx-light`` was absent and unavailable from the configured
package index, preventing the named backend case from running. ONNX Runtime
1.31.0 was available, but no comparable two-backend timing is claimed.
Hardware-counter profiling was denied (``perf_event_paranoid=4``); the store
policy A/B test identifies a memory-traffic sensitivity, not a counter-verified
cache-miss diagnosis. The original campaign's thread/spin settings and
per-case build provenance remain unknown.

Reproduce the diagnostic (absolute paths below assume this checkout location):

.. code-block:: bash

    cmake -S /home/runner/work/onnx-light-cpu/onnx-light-cpu -B /tmp/abs-build \
      -DCMAKE_BUILD_TYPE=Release -DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2 \
      -DONNX_LIGHT_CPU_BUILD_TESTS=ON -DONNX_LIGHT_CPU_BUILD_PYTHON=OFF \
      -DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=OFF -DONNX_LIGHT_CPU_BUILD_BENCHMARKS=ON
    cmake --build /tmp/abs-build --target test_abs_kernel abs_throughput \
      --parallel "$(nproc)"
    /tmp/abs-build/test_abs_kernel
    taskset -c 0 /tmp/abs-build/abs_throughput 4194304 1001 default
    taskset -c 0 /tmp/abs-build/abs_throughput 4194304 1001 cached
    taskset -c 0 /tmp/abs-build/abs_throughput 4194304 1001 streaming

Full acceptance still requires the existing backend benchmark with onnx-light
available, under identical explicit thread and spin policies for both runtimes.
The backend workload and its correctness requirements were not modified.

SimplifiedLayerNormalization follow-up
----------------------------------------

Issue `#724 <https://github.com/xadupre/onnx-light-cpu/issues/724>`_ fixes
``stash_type`` selection and removes the FP32-only, exact-suffix restriction
from the common optimized paths. FP16 reuses the F16C RMS reduction and affine
implementation, but rounds only after scaling. FP64 uses AVX with either FP32
or FP64 accumulation. Broadcast-contiguous suffixes need one scale index per
row or block rather than per element. Mixed types, BF16, unaligned buffers,
and other layouts retain portable fallbacks.

Measurements on 2026-09-17 used an AMD EPYC 9V74 runner (two visible physical
cores), GCC Release, the AVX2 ceiling, onnx-light 0.1.27, and ONNX Runtime
1.30.0 CPUExecutionProvider. The existing backend CLI measured all 72 FP16,
FP32, and FP64 cases (12 shapes, with and without inverse RMS) using
``stash_type=1``, 100 warm-ups, up to 2,000 repetitions, and a one-second
limit per phase. Shapes include decode, prefill, narrow/tail rows, suffix
axes, and outer broadcasting. Every case exceeded ``0.9x``; the minimum
single-thread speedup was ``1.45x``.

An additional cross-check used the existing
``tools._avx2_parity_worker.measure_isolated`` API on the medium, prefill4096,
and outer-broadcast fixtures, with identical serialized inputs and thread
counts. Runtimes ran in separate subprocesses, exited between phases, and
alternated first-runtime order by case. Affinity was ``taskset -c 0,2``
(one hardware thread per physical core). The 18 cases at one and two threads
all exceeded ``0.9x``; the minimum across these 36 combinations was ``1.32x``.
Representative isolated **median latencies in seconds**, with Y only:

.. list-table::
   :header-rows: 1
   :widths: 29 8 16 16 12 19

   * - X / Scale shape, dtype
     - Threads
     - onnx-light-cpu
     - ONNX Runtime
     - ORT / CPU
     - Recorded path
   * - [64,512] / [512], FP16
     - 1
     - 0.000008122
     - 0.000115275
     - 14.19x
     - f16c/row-scale
   * - [64,512] / [512], FP16
     - 2
     - 0.000008102
     - 0.000062875
     - 7.76x
     - f16c/row-scale
   * - [4,64,128] / [4,1,128], FP32
     - 1
     - 0.000009113
     - 0.000079320
     - 8.70x
     - float32/normalization/row-scale
   * - [4,64,128] / [4,1,128], FP32
     - 2
     - 0.000009194
     - 0.000042494
     - 4.62x
     - float32/normalization/row-scale
   * - [1,128,4096] / [4096], FP64
     - 1
     - 0.000193082
     - 0.000801040
     - 4.15x
     - avx/row-scale
   * - [1,128,4096] / [4096], FP64
     - 2
     - 0.000101874
     - 0.000403906
     - 3.96x
     - avx/row-scale

The workbook's ``cpu_kernel_paths`` column records the registered operator
and its selected implementation, not merely registration eligibility.
The small cases intentionally stay serial inside the CPU executor even when
two threads are available. The runner is virtualized; these measurements are
not universal hardware guarantees. Hardware sampling was unavailable
(``perf_event_paranoid=4``); no hardware-profile claim is made. No measured
case remained below the acceptance threshold. Performance parity is not
claimed for unmeasured mixed-type, BF16, or FP64-stash workloads.

Reproduce the full single-thread corpus from the repository checkout:

.. code-block:: bash

    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2" \
      python setup.py build_ext --inplace --onnx-light-source
    taskset -c 0,2 python -m onnx_light_cpu benchmark \
      --test '^test_cpu_simplified_layer_normalization_' \
      --dtype float16 float32 float64 --threads 1 \
      --repeat 2000 --warmup 100 --max-repeat-time 1 --onnxruntime \
      --output /tmp/simplified-layer-normalization.xlsx

SimplifiedLayerNormalization FP16 row-range follow-up
-----------------------------------------------------

Issue `#984 <https://github.com/xadupre/onnx-light-cpu/issues/984>`_ lists
six AVX2 FP16 cases. On 2026-10-24, none reproduced the reported ORT advantage
on this runner, even before this change. This does not disprove the original
EPYC 7763 measurements: their per-case commits and thread/spin settings are
unknown, and this runner is an AMD EPYC 9V74 under virtualization (four logical
CPUs, two physical cores).

The baseline is ``0d7606c2105791b2846007c6d860f129bd2663dc``, whose production
code matches ``ca018642cfb90f7569730deadf4c2d60daa1f3aa``. Remote refs were
fetched before measuring; the branch already contained remote main.
Both builds used GCC 13.3.0, Release, and
``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2`` (detected level 3). The after build adds
this row-range change. Dependencies were the onnx-light 0.2.0 CPython 3.13
Linux release wheel (native Python ABI, source commit not embedded in build
metadata), ONNX Runtime 1.31.0 (Release, commit ``402979a``), NumPy 2.5.4,
and onnx-light-cpu 0.2.0 from this checkout.

The unchanged backend fixtures use seeds 941/942, FP16 X and scale,
``axis=-1``, ``epsilon=1e-6``, and ``stash_type=1``. Each backend used one
thread, affinity ``taskset -c 0``, 100 warmups and 2,000 single-call samples
with a two-second phase limit (all 2,000 samples completed). ORT used
CPUExecutionProvider, sequential execution, and one inter-op thread.
With one participant neither backend has worker spinning; spin settings were
otherwise left at their defaults. The existing backend benchmark API times
CPU first, before creating the ORT session, then ORT. Three complete passes
were collected before rebuilding, then three after; build order was not
randomized. CPU feeds remained backend tensors, and only ORT received NumPy
views. Allocation and runtime overhead remain inside the timed session calls.

Median latency in **milliseconds** below is the median of the three per-pass
medians. ``inv`` denotes the optional FP32 inverse-RMS output.

.. list-table::
   :header-rows: 1

   * - X shape / output
     - CPU before
     - CPU after
     - ORT before
     - ORT after
     - CPU latency reduction
   * - [1,128,5120] / Y
     - 0.098576
     - 0.097024
     - 0.142872
     - 0.143112
     - 1.6%
   * - [1,128,4096] / Y
     - 0.080690
     - 0.079017
     - 0.115822
     - 0.115832
     - 2.1%
   * - [1,128,4096] / Y, inv
     - 0.082262
     - 0.079859
     - 0.116833
     - 0.116713
     - 2.9%
   * - [1,128,2048] / Y
     - 0.044395
     - 0.042352
     - 0.062202
     - 0.062653
     - 4.6%
   * - [1,128,2048] / Y, inv
     - 0.045908
     - 0.043345
     - 0.063213
     - 0.063495
     - 5.6%
   * - [1,4096,128] / Y, inv
     - 0.172907
     - 0.137294
     - 0.183182
     - 0.183272
     - 20.6%

The q_norm before medians ranged from 0.172736 to 0.177283 ms; after medians
ranged from 0.137074 to 0.137333 ms. Its three after p10/p90 intervals were
0.137023--0.145686, 0.137054--0.145678, and 0.136813--0.145308 ms.
The smaller prefill improvements should not be treated as universal wins:
sample distributions overlap, and the shared runner showed occasional
outliers, including one before 4096/inv median of 0.090164 ms.

Hardware-counter profiling was denied (``perf_event_paranoid=4``).
Assembly inspection identified per-row MXCSR save/mask/restore and separate
primitive calls. The FP16 exact-suffix, FP32-stash path now dispatches once
per executor range, saves/restores MXCSR once for that range, and writes
optional inverse statistics directly. It reuses the existing F16C reduction,
affine arithmetic, NaN handling, and final-only FP16 rounding; no scratch
allocation or private pool is added. Recording still reports
``SimplifiedLayerNormalization/f16c/row-scale``. Broadcast, mixed-type,
unaligned, and double-stash fallbacks are unchanged. Width-one inputs retain
their original adapter path.

Reproduce after building the matching onnx-light integration in Release with
the AVX2 ceiling and ``ONNX_LIGHT_CPU_PYTHON_STABLE_ABI`` matching the runtime:

.. code-block:: bash

    cd /home/runner/work/onnx-light-cpu/onnx-light-cpu
    taskset -c 0 python - <<'PY'
    from onnx_light_cpu._benchmark import run_backend_benchmark

    pattern = (
        r"^test_cpu_simplified_layer_normalization_qwen_("
        r"hidden(2048|4096)_prefill_x_float16_stash1_(y|with_inv)|"
        r"hidden5120_prefill_x_float16_stash1_y|"
        r"q_norm_x_float16_stash1_with_inv)_float16_benchmark$"
    )
    for _ in range(3):
        raw, rows = run_backend_benchmark(
            [pattern], ["float16"], 2000, 100, 2.0, 1, True
        )
        for row in rows:
            print(row["case"], row["median_s"], row["onnxruntime_median_s"],
                  row["p10_s"], row["p90_s"], row["cpu_kernel_paths"])
    PY

The focused tests compare all four issue shapes with the builtin-operator
reference, including Y-only and inverse outputs. Primitive tests also compare
the batched path bit-for-bit with the original per-row primitives across
vector boundaries, partial ranges, in-place output, special values, and
empty ranges, and check F16C floating-point-state restoration. Runtime-owned
scheduling and nested-parallel suppression remain covered. The original
EPYC 7763 and multithreaded campaign still require a separate reproduction.

Validation passed 92 affected C++ tests and seven Python integration tests.
All six original seeded benchmark fixtures also matched their generated
reference outputs at the existing FP16 tolerances.
Reconfiguring the same build with ``ML_CPU_COMPUTE_HAVE_F16C_FLAG=OFF`` also
passed 31 focused tests; the F16C-only floating-point-state test was skipped.
The build was then restored to AVX2/F16C. Secret scanning and formatting checks
passed; CodeQL timed out without returning an analysis result.

Reproducible baseline
---------------------

`Issue #631 <https://github.com/xadupre/onnx-light-cpu/issues/631>`_ establishes
the measurement baseline for this roadmap without changing kernel
implementations. Update the target branch before collecting results, then build
the Python package in Release mode with the AVX2 ceiling:

.. code-block:: bash

    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2" \
      python setup.py build_ext --inplace --onnx-light-source
    PYTHONPATH=. python -c \
      "from onnx_light_cpu import detect_simd_level; print(detect_simd_level().name)"
    python tools/benchmark_avx2_parity.py \
      --environment pinned --output avx2-parity.json

The detected level must be ``AVX2``. The fixed float32 and float16 corpus covers
GEMM/MatMul, Attention, activation and normalization, unary, and binary
elementwise cases. It runs one-thread and process-visible physical-core policies
with identical onnx-light-cpu and ONNX Runtime thread counts. Each runtime
runs in its own process, which exits before the next runtime starts. The first
runtime alternates between consecutive cases. Merely separating timing phases
inside one process is insufficient: idle ORT workers can keep spinning while
the CPU runtime is measured.

Each case's model and input bytes are serialized once, checksummed, and reused
by both runtimes and both thread policies. Per-case scratch files are removed
afterwards. The parent does not create a runtime execution pool, and normal ORT
thread-pool policies remain unchanged.

.. warning::

    The :doc:`isolated-runtime diagnostic baseline
    <2026_09_avx2_diagnostic_baseline>` demonstrates substantial interference
    from idle ORT spinning in the original shared-process runner.
    `#647 <https://github.com/xadupre/onnx-light-cpu/pull/647>`_ now isolates
    the runtimes. Results collected with the older runner must not be used for
    parity decisions and remain diagnostic even when ``--environment pinned``
    was set.
    Report the revision used to compile the binaries, not just HEAD at the
    end of the run.

The JSON records every raw sample, medians and dispersion, shapes, data types,
loop families, CPU and affinity, SIMD ceiling and detected level, compiler,
package versions, and timing order. Results are ranked by positive absolute
latency gap and speedup, with Qwen decode and prefill rows labelled explicitly.
Native Python extension paths and hashes are pinned and verified in CPU workers;
on Linux, the loaded onnx-light and CPU shared-library paths and hashes are also
checked. Shared-library discovery is explicitly unavailable on other platforms.
Checkout HEAD and compiler environment are labelled as such: the tool cannot
infer the compiled source revision from them. Capture the actual build revision
alongside the report, and do not rebuild the runtime during measurement.
The companion Markdown groups rows into ``<0.5x``, ``0.5x-0.9x``,
``0.9x-1.0x``, and ``>=1.0x`` ONNX Runtime.

Run the ``AVX2 parity baseline`` workflow to publish the JSON, Markdown, and
environment capture as one artifact. Generated results are not committed.
Results from shared runners are diagnostic, especially within 5--10% of parity;
only isolated-runtime results collected on pinned native AVX2 hardware and
labelled ``--environment pinned`` may make a final parity decision. Follow-up
issues should be opened only for measured bottlenecks confirmed without
cross-runtime thread-pool interference.

The :doc:`2026_09_avx2_diagnostic_baseline` records the September 6 isolated
follow-up. FP16 matrix paths, FP32 M=1, compact integer matrices, multithread
FP64 and long-context Attention remain priorities. Selected RMSNormalization
and BiasGelu cases are ahead of ORT, but the full isolated corpus and final
acceptance gate remain pending.

Current foundation
------------------

September 8 full activation-corpus audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The priority corpus alone was insufficient to establish Sigmoid/Softmax
parity: it selected only CPU-specific FP32 benchmarks, omitting inherited
``test_cc_*`` cases, FP16/FP64, and ordinary backend cases. The isolated runner
now offers a complete activation selection:

.. code-block:: console

    python tools/benchmark_avx2_parity.py --corpus activations \
      --physical-threads 6 --repeat 10000 --warmup 40 \
      --max-repeat-time 0.2 --output activations.json

This selects both benchmark and regular cases, including large cases, and
reads the actual model input type instead of guessing it from the test name.
All four floating-point types are included unless ``--dtype`` narrows the
selection. Unsupported ORT types remain explicit errors in the report, not
parity wins. ``--physical-threads`` only controls the requested participant
count; it does not pin the process. Pin the parent process to the intended
cores before invoking the runner, and use the same affinity for both runtimes.
The default priority corpus and its existing type selection are unchanged.

The baseline at ``aa405a9`` contains 44 benchmark and 22 regular activation
cases. On a native AVX2 i7-13800H, Windows/MSVC Release, Python 3.13 and
ORT 1.29, the audit used CPU 4 for one-thread runs and logical CPUs
0, 2, 4, 6, 8, 10 for six-thread runs. Windows topology identified these as
six distinct performance cores. Each runtime ran in a separate process,
with two alternating phases per case/thread policy and identical serialized
fixtures. The baseline native binaries were preserved before rebuilding.

There were 22 slower supported benchmark/thread combinations: six serial
and sixteen multithread. Every supported regular case was ahead of ORT.
Eleven benchmark and three regular BF16 cases were unsupported by ORT.
The largest residuals were not all exponential-throughput problems:

* FP64 Sigmoid and Softmax still evaluated exponentials with scalar
  ``std::exp`` despite AVX2 availability.
* FP16 Sigmoid converted through two stack buffers every 256 elements.
  The new AVX2/FMA/F16C path converts directly in registers, with independent
  runtime F16C detection and bounded copies for the final one to seven values.
* Sigmoid's 256 KiB minimum block left 65,535 FP32 values and 131,072
  FP16 values serial. FP64 at 65,535 values also remained serial, immediately
  before a two-participant discontinuity at 65,536.
* Simply assigning six participants to small activations regressed them:
  dispatch and coordination cost exceeded the work saved. Bounded small
  teams are required, not just a lower global parallel threshold.
* FP32 Sigmoid unnecessarily multiplied the negative exponential by a
  reciprocal after division. Selecting the numerator before one division
  removes that operation without changing the stable saturation formula.
* Normal FP32 Softmax rows repeated the same exponential range check for
  every vector. A row-level min/max eligibility check allows the normal
  exponential loop to omit those branches; exceptional and subnormal rows
  retain the full exponential path.

Near-zero rational Sigmoid approximations and a reordered exponential
polynomial were also measured and discarded. A narrow rational fast path
improved small inputs but regressed larger Gaussian inputs through
unpredictable per-vector branches. Neither small-case improvements nor
direct preallocated-kernel timings establish end-to-end parity.

The all-case parity gate remains open. In particular, large FP32 multithread
activations require confirmation through the registered runtime, not an
inference from serial SIMD throughput. These development-machine results
remain diagnostic: process affinity does not eliminate frequency variation
or other host activity.

The final two-phase run covered all 132 case/thread combinations. Of the
104 supported combinations, eight benchmark combinations remained below
``1.0x`` ORT, down from 22 in the baseline. All measured FP64 and supported
regular cases were ahead of ORT. Selected CPU latencies, in microseconds,
are the median of the two phase medians:

.. list-table::
   :header-rows: 1

   * - Case
     - Threads
     - Before
     - After
     - Before / after
   * - Sigmoid FP64 / 1,048,576
     - 1
     - 7916.05
     - 1394.20
     - 5.68x
   * - Sigmoid FP64 / 1,048,576
     - 6
     - 2741.52
     - 440.60
     - 6.22x
   * - Softmax FP64 / 1,024 x 1,024
     - 1
     - 7714.65
     - 1718.60
     - 4.49x
   * - Sigmoid FP16 / 131,072
     - 6
     - 81.95
     - 35.95
     - 2.28x
   * - Sigmoid FP32 / 65,535
     - 6
     - 30.90
     - 18.70
     - 1.65x

The remaining combinations are listed without excluding inherited cases.
``cc`` identifies the inherited benchmark rather than the CPU-specific
case of the same shape:

.. list-table::
   :header-rows: 1

   * - Case
     - Threads
     - CPU (us)
     - ORT (us)
     - ORT / CPU
   * - Softmax FP32 / 1,024 x 1,024
     - 6
     - 227.10
     - 103.80
     - 0.457x
   * - Sigmoid FP32 / 1,048,576
     - 6
     - 168.78
     - 83.05
     - 0.492x
   * - Softmax FP32 / 2,048 x 2,048 (cc)
     - 6
     - 2135.80
     - 1054.98
     - 0.494x
   * - Sigmoid FP32 / 4,194,304 (cc)
     - 6
     - 1202.37
     - 596.50
     - 0.496x
   * - Sigmoid FP16 / 1,048,576
     - 6
     - 195.02
     - 145.98
     - 0.749x
   * - Sigmoid FP32 / 131,072
     - 6
     - 29.60
     - 22.70
     - 0.767x
   * - Sigmoid FP32 / 1,048,576
     - 1
     - 407.98
     - 382.45
     - 0.937x
   * - Sigmoid FP32 / 65,536
     - 6
     - 19.90
     - 19.80
     - 0.995x

Repeat measurements with the preserved baseline confirmed substantial
large-case variability in both binaries. They do not establish a multithread
FP16 gain, nor a stable regression for the inherited FP32 cases. No final
parity claim should be inferred from these development-machine ratios.

The worker protocol also now passes requests through standard input and uses
unique result files. Reopening a command-line JSON request intermittently
failed with Windows file-sharing violations after the preceding worker had
exited. This transport change is outside the timed inference region and
retains the separate-process and fixture-identity guarantees.

September 7 dashboard follow-up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The registered ``Sigmoid`` and last-axis ``Softmax`` dispatchers could not
reach their existing fused AVX2/FMA implementations: the implementation
library's ``ML_CPU_COMPUTE_HAVE_AVX2_FMA`` definition was private and absent
from the separate registration library. The registration target now receives
the availability definition when those kernels are built, without applying
AVX2 compiler flags to its portable translation units. Runtime dispatch still
requires both AVX2 and FMA. The scalar activation fallback also explicitly
disables the exponential cost model so the outer activation scheduler owns
parallelism.

An integration regression compares registered FP32 output exactly with the
direct fused implementation, rather than accepting a numerically close scalar
fallback. It fails before the CMake correction. Reference, aliasing, tail,
special-value and executor cases cover the newly reachable path, including
exact sigmoid saturation at positive infinity.

On a native AVX2 Intel Core i7-13800H, Windows/MSVC Release, ORT 1.29,
one thread and process affinity fixed to logical CPU 4, the isolated-runtime
backend runner measured the following FP32 medians. Each runtime exits before
the other starts; each phase uses 50 warmups and up to 500 samples or 0.5 s.
The baseline is the registration path without the availability fix.

.. list-table::
   :header-rows: 1

   * - Operator / shape
     - Before (ms)
     - After (ms)
     - ORT (ms)
     - Before / after
   * - Sigmoid / 65,536
     - 0.1792
     - 0.0424
     - 0.0314
     - 4.23x
   * - Sigmoid / 131,072
     - 0.4572
     - 0.0800
     - 0.0730
     - 5.71x
   * - Sigmoid / 1,048,576
     - 3.0583
     - 1.2173
     - 0.4703
     - 2.51x
   * - Softmax / 32 x 1,024
     - 0.1026
     - 0.0189
     - 0.0220
     - 5.43x
   * - Softmax / 1,024 x 1,024
     - 3.2041
     - 0.8593
     - 0.5074
     - 3.73x

Attention additionally reuses its AVX2 online-softmax row kernel for short
query bursts and its vector mask/softmax helpers in the tiled FP32 path.
Alternating full ``ComputeAttentionFloat32`` measurements, with the matrix
implementation held constant, improve Q8/KV1024/D64 causal/nonpad from
7.240 to 1.347 ms and the Q128/KV128/D64 additive case from 1.824 to
0.911 ms (batch 1, 12 heads, one thread). Unsupported mask layouts,
softcap/window modes and other ISAs retain their existing paths.

For Gemm/MatMul, non-transposed FP32 skinny-N matrices now reduce strided
columns with bounded AVX2 gathers and independent FMA accumulators. FP64
register kernels use masked vectors for their final one to three columns.
At M1024/N7/K1024, alternating standalone runs improve FP32 from 4.820 to
1.339 ms (3.60x) and FP64 from 4.978 to 2.203 ms (2.26x). Four-worker
configurations also improve these cases, but the FP64 plan remains serial:
these results do not establish a scheduling or scaling improvement.
Aligned wide FP64 and FP32 M=1 GEMV are unchanged; experiments that regressed
those paths were discarded.

The existing throughput driver accepts a case substring and FP32/FP64
selection for reproducing focused matrix measurements:

.. code-block:: console

    gemm_throughput 1 dashboard_skinny_n7 fp32
    gemm_throughput 1 dashboard_skinny_n7 fp64
    gemm_throughput 4 dashboard fp32

These are diagnostic development-machine measurements, not completion of the
parity gate. The registered optimized attention cases still measure about
0.70--0.84x ORT, and the largest activation cases retain gaps. CPU affinity
does not eliminate frequency variation or other host activity. In particular,
the earlier direct-kernel activation measurements did not establish that the
registered runtime actually reached those kernels.
A six-participant run also leaves the selected attention and activation
cases below parity (0.32--0.59x ORT); closing multithread scheduling and
throughput gaps remains necessary.

The :doc:`2026_09_avx2_matrix_kernel_improvements` adds production AVX2
FP16/FP32 single-row kernels and bounded FP16 panel widening, with controlled
before/after measurements against ORT. Selected Qwen FP16 projections are
ahead of ORT; small and medium matrix gaps and the full parity gate remain.

The first AVX2-specific passes are already merged:

* `#604 <https://github.com/xadupre/onnx-light-cpu/pull/604>`_ adds fused
  AVX2/FMA ``Sigmoid`` and ``Softmax`` kernels and shortens the ``BiasGelu``
  dependency chain;
* `#605 <https://github.com/xadupre/onnx-light-cpu/pull/605>`_ improves
  medium GEMM, batched MatMul, and Attention decode scheduling;
* `#608 <https://github.com/xadupre/onnx-light-cpu/pull/608>`_ adds the
  dedicated AVX2/FMA single-query Attention path and removes scalar FP32 GEMM
  tails for one through seven columns.

These changes establish the AVX2 implementations and benchmark cases, but do
not constitute a complete AVX2 parity sweep. The explicit SIMD ceiling now
makes that sweep reproducible and prevents an AVX-512-capable development
machine from hiding an AVX2 fallback.

Skinny-M MatMulInteger profiling (#725)
-----------------------------------------

The September 17 checkout already bypassed packing for ``M=1``. Its AVX2
kernel widened eight B bytes to INT32, then traversed all of K before
advancing to the next eight columns. Thus the reported M=1 slowdown was
not caused by a full B pack in this revision: narrow, strided B reads and
the widening/multiply loop were the target. ``M=2`` still transposed B.

Before changing dispatch, ``integer_gemm_throughput`` measured the production
B packer, the existing dot kernel on an already packed B, the forced packed
driver (including allocations/corrections), and public dispatch separately.
On a Xeon Platinum 8573C, GCC 13.3 Release, CPU affinity 0, one participant,
AVX2 ceiling, UINT8 x INT8 with scalar zero points 128 and 0:

.. list-table::
   :header-rows: 1

   * - M / N / K
     - B packing (s)
     - Packed dots only (s)
     - Packed driver (s)
     - Original dispatch (s)
   * - 1 / 4096 / 4096
     - 0.067409
     - 0.000743
     - 0.070419
     - 0.008151
   * - 2 / 4096 / 4096
     - 0.067964
     - 0.001528
     - 0.068869
     - 0.068952

These are medians of 31 samples after five warmups. Isolated phase timings
are not additive (allocation and cache state differ). They nevertheless
show packing dominating M=2 and confirm that M=1 needs a better streaming
kernel. Hardware-counter profiling was unavailable on this runner
(``perf_event_paranoid=4``); these are wall-clock phase profiles, not PMU
attributions.

The new plan interleaves four K values **in registers**, reuses the exact
AVX2 split-byte dot arithmetic or AVX-512 VNNI, and sweeps columns within
bounded K bands. No B panel or retained packed weights are allocated.
All four signedness combinations and scalar/per-axis zero points are
supported. AVX2 specializes the normalized A zero points 0 and 128 to avoid
unnecessary column sums. M=1 retains its streaming selection; M=2 uses it
for N >= 32 and K >= 4 (the existing packed AVX2 microkernel has MR=2).
Larger M and unsupported ISAs retain the packed path. Runtime-owned,
cache-line-aligned column ranges provide parallelism without nesting.

The end-to-end parity tool now prints and saves ``packed``, ``no-pack-avx2``,
or ``no-pack-vnni`` from the production planner. Its ``skinny_m_4096`` case
omits both zero points, as does the public backend benchmark; the
``skinny_m_4096_shifted`` case separately checks A zero point 128.
With onnx-light 0.1.27 and ONNX Runtime 1.30.0, one thread, affinity 0,
31 alternating samples and five warmups, the omitted-zero-point AVX2 case
measured **0.001803 s versus ORT 0.005461 s (3.03x)**. The shifted case was
0.001928 s versus 0.005775 s (3.00x); M=2 was 1.50x and the
``2 x 1025`` by ``1025 x 257`` tail case was 1.34x. Results are checked
for exact equality before timing; ORT worker spinning is disabled.

This clears the selected same-host comparison, **not** the full integer
parity gate or the issue's published absolute latency. The unchanged
``direct`` packed case remained below parity. Background extraction work
on this shared runner affected timings, so dedicated-machine reproduction
is still needed. The original ORT 0.354 ms number was not reproduced.
For the omitted-zero-point case, a final isolated profile measured
0.075958 s packing, 0.000935 s prepacked dots, and 0.001815 s streaming.
Even prepacked exact AVX2 arithmetic exceeds the published ORT latency;
register interleaving, output-band updates, and memory traffic remain in
the streaming path. The phase comparison is a diagnostic floor, not a
precise attribution of that remaining absolute gap.

After background extraction finished, an isolated AUTO build on the same
host verified ``no-pack-vnni`` and measured the public-compatible case at
**0.000715 s versus ORT 0.004785 s (6.69x)**. The shifted case measured
0.000719 s versus 0.004808 s (6.69x), M=2 measured 3.29x, and the tail
case measured 1.78x. These used the same one-thread, CPU-0,
31-alternating-sample/five-warmup protocol. All 13 native differential tests
passed, as did all 13 AVX2 tests under AddressSanitizer/UndefinedBehaviorSanitizer;
a build without compiled AVX kernels passed its 10 applicable tests.

Reproduce without timing model construction or moving packing outside the
end-to-end measurement:

.. code-block:: console

    cmake -S . -B build-integer -DCMAKE_BUILD_TYPE=Release \
      -DONNX_LIGHT_CPU_BUILD_BENCHMARKS=ON -DONNX_LIGHT_CPU_BUILD_PYTHON=OFF \
      -DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=OFF -DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2
    cmake --build build-integer --target integer_gemm_throughput -j4
    taskset -c 0 build-integer/integer_gemm_throughput 1 4096 4096 0 0
    taskset -c 0 build-integer/integer_gemm_throughput 2 4096 4096 128 0
    python -m tools.benchmark_integer_gemm_parity --threads 1 --cpus 0 \
      --case skinny_m_4096 --case skinny_m_4096_shifted --case skinny_m_mr \
      --case skinny_m_tail --repeat 31 --warmup 5 --output /tmp/integer-parity.json

The Python command requires an in-place extension built with the same ISA
ceiling. Use ``AUTO`` instead of ``AVX2`` for native VNNI measurements.
The C++ phase benchmark deliberately has no executor and reports one
participant; use the Python tool's ``--threads`` for runtime parallelism.

Work sequence
-------------

.. list-table::
   :header-rows: 1
   :widths: 10 27 45 10 8

   * - Step
     - Scope
     - Exit criterion
     - Depends on
     - Status
   * - AVX2 PR01
     - Reproducible AVX2 ceiling and baseline mechanism.
     - The build accepts an AVX2 ceiling, excludes AVX-512 and AMX kernels,
       reports AVX2 runtime dispatch, and lets the existing backend corpus
       measure AVX2 paths on wider x86 hosts.
     - None
     - Implemented in #614
   * - AVX2 PR01.1
     - Baseline and gap inventory.
     - The priority backend corpus publishes operator, type, shape, thread,
       and loop-family results under the AVX2 ceiling. The report ranks gaps
       by absolute latency and ONNX Runtime ratio before further tuning.
     - PR01
     - Tooling from `#632
       <https://github.com/xadupre/onnx-light-cpu/pull/632>`_ now isolates
       runtime processes; the complete measured inventory remains pending
   * - AVX2 PR02a
     - FP32/FP64 GEMM and MatMul.
     - FP32/FP64 register tiles, masked tails, packing, prefetch, and
       participant selection are tuned from measured gaps.
     - PR01.1
     - Assigned in `#633
       <https://github.com/xadupre/onnx-light-cpu/issues/633>`_; foundations
       delivered through #605 and #608
   * - AVX2 PR02b
     - Compact matrix paths.
     - FP16/BF16 conversion and integer/packed paths avoid scalar or
       full-tensor conversion bottlenecks on the priority shapes.
     - PR01.1
     - Initial work in `#634
       <https://github.com/xadupre/onnx-light-cpu/issues/634>`_; FP16 follow-up
       in :doc:`2026_09_avx2_matrix_kernel_improvements`, full compact-type
       parity still pending
   * - AVX2 PR03
     - Attention.
     - Decode, short-query, and prefill cases use AVX2 score and value kernels
       with productive head/query scheduling. Masks, causal bounds, GQA/MQA,
       FP16/BF16 conversion, and vector tails retain differential parity.
     - PR01.1, PR02a
     - Assigned in `#635
       <https://github.com/xadupre/onnx-light-cpu/issues/635>`_; foundations
       delivered through #605 and #608
   * - AVX2 PR04a
     - Activations and normalization.
     - Priority transformer shapes avoid unnecessary memory passes,
       conversion, scalar tails, and unproductive scheduling while retaining
       their numerical contracts.
     - PR01.1
     - Delivered in `#638
       <https://github.com/xadupre/onnx-light-cpu/issues/638>`_ (see
       :doc:`2026_09_avx2_activation_normalization`); foundations delivered
       through #604
   * - AVX2 PR04b
     - Unary and binary elementwise kernels.
     - Priority contiguous and broadcast cases avoid scalar tails and
       unnecessary widening, while expensive arithmetic and conversion paths
       retain their numerical contracts.
     - PR01.1
     - Assigned in `#640
       <https://github.com/xadupre/onnx-light-cpu/issues/640>`_
   * - AVX2 PR05
     - Final parity and regression gate.
     - Every priority family reaches the median and minimum targets on native
       AVX2 hardware. The same commit passes the AVX2-ceiling and automatic
       AVX-512 correctness suites without an AVX-512 performance regression.
     - PR02a--PR04b
     - Pending

AVX2 PR05 completes this follow-up. Architecture-specific work for AVX-512FP16,
AVX-512BF16, VNNI, AMX, NEON, or SVE remains independent and must not be used
to hide a missing AVX2 implementation.

Softmax follow-up (issue #997)
------------------------------

On 2026-10-23, the exact
``test_cpu_softmax_1024x1024_float32_benchmark`` case reproduced an ORT
advantage on an AMD EPYC 9V74 virtualized runner (four logical CPUs, two
physical cores), not the issue's EPYC 7763. The input remains the registered
float32 ``[1024,1024]`` tensor, seed ``987654321``, axis ``-1``, opset 13.
No workload, tolerance, allocation, or executor policy was changed.

The baseline was commit ``38940c9fc353f8f4c9856575541c8e3f904ed372``,
which includes remote main ``b3fc52ef3c52f0378b56664e4c11c02837c07e2c``;
remote refs were refreshed before measurement. Builds used GCC 13.3,
Release, and ``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2``; detected SIMD level was
3. Backends were onnx-light-cpu 0.2.0 from this checkout, the published
onnx-light 0.2.0 CPython 3.13 Linux wheel, and ONNX Runtime 1.31.0
CPUExecutionProvider. The wheel's underlying commit was not available.

Generated-code inspection found register spills in the fused row loop.
The retained change uses compact 128-column exponential/sum blocks for
normal-range rows up to 2048 columns. Bounded partial sums avoid losing
small contributions after dominant logits. Wider rows and extreme-range
rows retain the original multi-accumulator loop. The exponential
approximation, SIMD dispatch, scalar fallback, and runtime scheduling
are unchanged.

The final baseline/final comparison used one thread, ``taskset -c 2``,
100 warm-ups and 2000 repetitions per backend, with a two-second limit per
phase. Each backend ran all 2000 samples in three fresh CLI processes.
CPU feeds were backend tensors; ORT feeds were NumPy arrays, as in the
existing runner. Backend phase order was CPU then ORT on both builds.
Spin settings were the unchanged backend defaults: onnx-light adaptive
bounded spinning and ORT's default spinning. These policies are not
identical, and the published campaign's spin/thread settings are unknown.

End-to-end latencies below are **milliseconds**, summarized as the median
of three per-process medians, with their full range:

.. list-table::
   :header-rows: 1

   * - Backend
     - Before
     - After
   * - onnx-light-cpu
     - 0.416699 (0.382647--0.426699)
     - 0.375611 (0.374304--0.427099)
   * - ONNX Runtime
     - 0.299411 (0.295515--0.299947)
     - 0.296547 (0.292271--0.301064)

ORT gain (CPU / ORT median) decreased from 1.392x to 1.267x, but the
process ranges overlap substantially: the apparent 9.9% session improvement
is indicative, not a stable hardware guarantee or a parity claim.
The direct AVX2/FMA driver, on the same CPU and affinity with preallocated
buffers, ten warm-ups and 301 samples, measured 0.370377 to 0.360834 ms
for aligned ``[1024,1024]`` (2.6% lower), and 0.421224 to 0.417670 ms
with a one-float alignment offset. This is a separate diagnostic input,
not a substitute for the registered workload.

Earlier one/two/default-thread runs also reproduced the ORT gap.
Reverse-order cross-checks became contaminated by validation extractors
saturating the runner, including CPU 0's SMT sibling; those results are
excluded. The final matched-core comparison used CPU 2 instead.
Hardware sampling was unavailable (``perf_event_paranoid=4``); the spill
diagnosis is from disassembly, not hardware counters. Native EPYC 7763
confirmation and a quiet multithreaded rerun remain outstanding.

Reproduce from the checkout with the selected onnx-light runtime on
``PYTHONPATH``:

.. code-block:: bash

    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2" \
      python setup.py build_ext --inplace --onnx-light-source
    taskset -c 2 python -m onnx_light_cpu benchmark \
      --test '^test_cpu_softmax_1024x1024_float32_benchmark$' \
      --threads 1 --repeat 2000 --warmup 100 --max-repeat-time 2 \
      --onnxruntime --output /tmp/softmax.xlsx
    # Configure ONNX_LIGHT_CPU_BUILD_BENCHMARKS=ON to build this diagnostic.
    taskset -c 2 ./build/exp_log_throughput 301 softmax

Focused validation passed 36 direct Exp/Log/activation tests and 18 runtime
Sigmoid/Softmax tests, including every vector tail, unaligned/in-place
buffers, the 2048-column boundary, extreme ranges, noncanonical NaNs, and
dominant logits with small remaining probabilities.
