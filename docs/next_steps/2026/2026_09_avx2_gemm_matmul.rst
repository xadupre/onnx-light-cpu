AVX2 FP32/FP64 GEMM and MatMul Gap Closure
===========================================

:Date: 2026-09
:Updated: 2026-10-26

**in progress**

This September report also includes an `October 2026 MatMul follow-up (#988)`_
with updated measurements and implementation decisions.

Objective
---------

Issue `#633 <https://github.com/xadupre/onnx-light-cpu/issues/633>`_ asks for
the dominant AVX2 FP32/FP64 ``Gemm``/``MatMul`` gaps against ONNX Runtime to be
measured and closed under an
``-DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2`` ceiling build, covering square,
transformer-projection, batched, skinny-M, skinny-N, large-K, transpose,
bias, dynamic-B, and ``N=1..7`` tail cases, without touching AVX-512
dispatch/heuristics or compact/Attention code.

Baseline methodology and its sandbox limitation
------------------------------------------------

``tools/benchmark_gemm_parity.py`` is the canonical parity gate, but it
imports ``onnx_light`` (the sibling ``onnx-light`` checkout) to build the
reference graphs and run the ``ReferenceEvaluator``/kernel-usage instrumentation.
That checkout is not available in this sandbox, so the canonical gate could
not be executed here. To still gather evidence, this pass used two
independent, narrower substitutes on an AVX2+AVX-512-capable 4-core host:

* An AVX2-ceiling Release build
  (``-DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2 -DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=OFF``)
  linked into a standalone timing harness that calls ``GemmFloat32``/
  ``GemmFloat64`` directly (no onnx-light dependency), covering the priority
  shape corpus from ``tools/benchmark_gemm_parity.py`` at one thread and with
  a naive per-call thread-spawning executor standing in for the "available
  physical-core policy".
* A plain ``onnx`` + ``onnxruntime`` (pip) Python script building single-node
  ``Gemm`` graphs and timing ``InferenceSession.run`` with
  ``intra_op_num_threads=1`` for the same shapes.

The important caveat: upstream ONNX Runtime has no supported way to cap its
own CPU execution provider to AVX2 at runtime (confirmed - this would require
rebuilding ONNX Runtime with AVX-512 disabled at compile time). This host
supports AVX-512, so the ORT numbers below reflect ORT's AVX-512 MLAS
kernels, not an AVX2-vs-AVX2 comparison. Any gap on square/wide-projection
shapes below is therefore expected to include a genuine ISA-width component
(8 vs 16 lanes) that this issue explicitly excludes from scope ("do not
change AVX-512 dispatch"). Closing that portion would require either a
genuinely AVX2-only host or an ONNX Runtime build with AVX-512 compiled out,
neither of which this sandbox provides.

Baseline findings (AVX2 ceiling, single thread, this host)
------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 16 16 16

   * - Case
     - onnx-light-cpu f32
     - ONNX Runtime (AVX-512) f32
     - Ratio
   * - ``direct`` (32x128x16)
     - 34.4 GFLOPS
     - 16.6 GFLOPS
     - 2.07x
   * - ``tiny_dynamic`` (1x64x64)
     - 2.8 GFLOPS
     - 2.3 GFLOPS
     - 1.20x
   * - ``square_128``
     - 91.4 GFLOPS
     - 176.2 GFLOPS
     - 0.52x
   * - ``square_512``
     - 100.7 GFLOPS
     - 195.1 GFLOPS
     - 0.52x
   * - ``square_1024``
     - 101.9 GFLOPS
     - 171.3 GFLOPS
     - 0.59x
   * - ``skinny_m`` (1x1024x1024)
     - 10.2 GFLOPS
     - 14.1 GFLOPS
     - 0.72x
   * - ``skinny_n`` (1024x1x1024)
     - 14.7 GFLOPS
     - 14.1 GFLOPS
     - 1.05x
   * - ``large_k`` (32x32x4096)
     - 82.7 GFLOPS
     - 138.2 GFLOPS
     - 0.60x
   * - ``split_k`` (2x2x4096)
     - 5.0 GFLOPS
     - 2.6 GFLOPS
     - 1.96x
   * - ``trans_a`` (128x128x128)
     - 84.5 GFLOPS
     - 155.5 GFLOPS
     - 0.54x
   * - ``trans_b`` (128x128x128)
     - 94.5 GFLOPS
     - 150.5 GFLOPS
     - 0.63x
   * - ``transformer_projection`` (128x3072x768)
     - 98.5 GFLOPS
     - 163.0 GFLOPS
     - 0.60x
   * - ``transformer_down_projection`` (128x768x3072)
     - 97.9 GFLOPS
     - 178.2 GFLOPS
     - 0.55x

The single-core AVX2 microkernel (``GemmMicroTile_AVX2FMA_F32Impl``/
``...F64Impl`` in
``ml_cpu_compute/src/math/gemm/avx2/gemm_avx2_fma.cc``) already
reaches roughly 78-80% of this host's theoretical AVX2 FMA peak
(2 FMA ports x 8 lanes x 2 flops/FMA) on square/projection/transpose shapes,
and the float64 path shows the same efficiency against its own peak. GEMV-like
``skinny_m``/``skinny_n`` shapes are memory-bandwidth bound (streaming the
full ``B``/``A`` operand once) and already reach or exceed the measured ORT
number; ``split_k`` and ``direct``/``tiny_dynamic`` (call-overhead-dominated
tiny shapes) are already ahead of ORT here. The remaining gap is concentrated
in square/projection/transpose shapes and tracks the AVX2-vs-AVX-512 lane
width, not an under-tuned AVX2 code path: register blocking
(``kGemmAVX2MR`` = 6 rows x 16 columns), K-unrolling by 4, prefetching, and
per-microarchitecture register-row selection (Intel Core vs AMD Zen, see
``SelectGemmRegisterRowsForMicroarchitecture`` in
``ml_cpu_compute/src/math/gemm/gemm_blocking.cc``) were already in place
before this pass.

A naive multi-thread scaling probe (thread-per-call executor, 4 physical
cores) showed close to 2x scaling on ``square_1024``/``transformer_projection``
(consistent with a partially memory-bandwidth-bound workload on this host)
and a regression for very small shapes such as ``square_128`` whose serial
runtime (about 47 microseconds) is smaller than the thread-creation/join cost
of the naive harness executor. That regression is judged to be a benchmark-
harness artifact (the real embedding runtime supplies its own persistent
thread pool, and ``SelectGemmParticipantCount`` already caps ``square_128``
at 2 participants, not 4) rather than evidence of a fixable admission-control
bug, so no participant-limit change was made without a persistent-pool
measurement to confirm it.

Status
------

No AVX2-specific microkernel, packing, blocking, or scheduling change is
included in this pass: the measured evidence available in this sandbox does
not demonstrate a fixable AVX2 regression separate from the AVX2-vs-AVX-512
lane-width difference, and the issue's own instruction is to "optimize only
bottlenecks demonstrated by the baseline." ``test_gemm_kernel`` and
``test_gemm_plan`` continue to pass unchanged under the AVX2 ceiling build.

Remaining work requires the dedicated-machine setup used by prior roadmap
items (see :doc:`the Gemm/MatMul roadmap <2026_08_gemm_matmul>`): an
``onnx-light`` checkout to run ``tools/benchmark_gemm_parity.py`` directly,
and either a genuinely AVX2-only host or an ONNX Runtime build with AVX-512
compiled out, so the reported ratios reflect an AVX2-vs-AVX2 comparison
instead of AVX2-vs-AVX-512.

October 2026 MatMul follow-up (#988)
------------------------------------

The investigation above describes the September pass, not the current
implementation. On 2026-10-25, the eight cases in
`issue #988 <https://github.com/xadupre/onnx-light-cpu/issues/988>`_ were
remeasured on an AMD EPYC 7763, which supports AVX2/FMA/F16C but not AVX-512.
All eight gaps reproduced in both pinned baseline campaigns below. An
unpinned two-thread run did not reproduce the FP32 128-square gap, illustrating
why the original dashboard numbers cannot be treated as a stable baseline.

Environment and protocol
~~~~~~~~~~~~~~~~~~~~~~~~

* Shared Linux VM: four logical processors, two physical cores; process affinity
  restricted to logical CPUs ``0,2`` (one sibling per core) for both backends.
  This is still a shared runner, not dedicated hardware.
* Original parity campaign: onnx-light-cpu 0.2.0 baseline production sources
  ``86c8762be5bb0c01459c1998f5daf8dc70e268fa``; optimized sources
  ``83952ac5da670ddeb7431565f3c5e87f5d967d90``. These Python/ORT timings
  predate the rejected paired-row implementation
  (``232b1ca5f1050b144d40129770d2b499d42b5c97``), so their skinny-N entries
  are historical results, not gains retained by this PR. Both builds used GCC
  13.3.0, CMake Release, and
  ``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2``. Runtime detection returned ``3``
  (AVX2); the throughput driver reported AMD Zen, FMA, and six register rows.
  The baseline was rebuilt in the same build tree after verifying its two
  production files against the baseline revision; no second runtime build was
  loaded alongside it.
* ONNX Runtime 1.31.0, official CPython 3.13 Linux wheel, Release build,
  reported commit ``402979a``; CPUExecutionProvider only.
* onnx-light 0.2.0, official release wheel
  ``onnx_light-0.2.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl``.
  Release tag commit: ``6400f0f812b13bc32469e405bd2f762ba3cba7f0``;
  the wheel does not report a compiled source commit, so the tag alone is
  not proof of binary provenance. Its loaded core library SHA-256 was
  ``181a18f807267481b5006060f6fbe6a05cafe3ce8dea75f90f5fb44c9d59741a``.
  The CPU extension linked to the wheel's exact core/proto/thread libraries
  with ``ONNX_LIGHT_CPU_PYTHON_STABLE_ABI=OFF``.
* Python 3.13.16, NumPy 2.5.4. Inputs came directly from the existing named
  backend cases (seeds 433/434), with both matrices dynamic. CPU feeds remained
  native backend tensors; only ORT feeds used NumPy.
* Separate runs at one and two explicitly configured intra-op threads.
  ORT used sequential execution and one inter-op thread. Neither backend
  pinned individual workers; both inherited the same process affinity.
  Spin settings were unchanged in every campaign: onnx-light's adaptive
  policy (default 10,000 spin iterations), ORT's default spinning enabled.
  These are backend defaults, not identical spin budgets; the one-thread
  measurements avoid worker-spin effects.
* Two fresh CLI processes per revision and thread count; each case used 20
  warm-ups and 101 measured calls, with a 10-second cap that did not truncate
  the measurements. Tables show the two independent per-process medians in
  milliseconds. The ORT column spans its medians across all four
  before/after processes, exposing reference-backend drift.
* The standard backend CLI measures CPU first, then constructs and measures
  ORT. That order was identical before/after. The isolated AVX2 parity worker
  was attempted but rejected by this onnx-light wheel before timing because
  its deserialized input wrappers were not accepted as tensors.

Reproduction
~~~~~~~~~~~~

Configure the existing source-integration build against the installed
onnx-light runtime, with Release, tests enabled, the AVX2 ceiling, and matching
Python ABI. Build ``_cpuregister``, ``_cpukernels``, ``gemm_throughput``,
``test_gemm_kernel``, ``test_gemm_plan``, and
``test_onnx_light_matmul_kernel`` using ``cmake --build`` with
``--parallel 48``. With the source checkout on ``PYTHONPATH``, run the
following twice for each revision, changing the output prefix each time:

.. code-block:: bash

   ROOT=/home/runner/work/onnx-light-cpu/onnx-light-cpu
   cd "$ROOT"
   for threads in 1 2; do
     PYTHONPATH="$ROOT" taskset -c 0,2 python -m onnx_light_cpu benchmark \
       --test '^test_cpu_matmul_(square_(128|512|1024)_float(16|32)|skinny_[mn]_float32)_benchmark$' \
       --dtype float32 float16 --threads "$threads" \
       --repeat 101 --warmup 20 --max-repeat-time 10 --onnxruntime \
       --output "/tmp/matmul-${threads}.xlsx"
   done

One-thread results (milliseconds)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Case suffix (``test_cpu_matmul_…_benchmark``)
     - Before, runs 1 / 2
     - After (83952ac5), runs 1 / 2
     - ORT median range
   * - ``square_128_float32``
     - 0.070552 / 0.064320
     - 0.054873 / 0.053981
     - 0.054732–0.055604
   * - ``square_128_float16``
     - 0.081582 / 0.080551
     - 0.060643 / 0.061104
     - 0.060473–0.061836
   * - ``square_512_float32``
     - 4.377377 / 4.295656
     - 3.071542 / 3.074969
     - 2.910933–2.952982
   * - ``square_512_float16``
     - 4.531956 / 4.410601
     - 3.231583 / 3.216244
     - 2.980252–3.046958
   * - ``square_1024_float32``
     - 37.000828 / 35.022375
     - 24.749660 / 25.231993
     - 23.581358–24.631252
   * - ``square_1024_float16``
     - 37.389930 / 36.676462
     - 25.406130 / 26.413833
     - 24.055971–24.766324
   * - ``skinny_m_float32``
     - 3.990410 / 3.776765
     - 1.718670 / 1.518633
     - 1.477728–1.991273
   * - ``skinny_n_float32``
     - 2.632572 / 2.518879
     - 1.410512 / 1.549080
     - 1.506652–1.817144

Two-thread results (milliseconds)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Case suffix
     - Before, runs 1 / 2
     - After (83952ac5), runs 1 / 2
     - ORT median range
   * - ``square_128_float32``
     - 0.038442 / 0.036689
     - 0.031299 / 0.030718
     - 0.032852–0.033964
   * - ``square_128_float16``
     - 0.080671 / 0.080070
     - 0.061074 / 0.061424
     - 0.042539–0.045996
   * - ``square_512_float32``
     - 2.343951 / 2.300823
     - 1.678233 / 1.694903
     - 1.488699–1.500391
   * - ``square_512_float16``
     - 2.347838 / 2.377476
     - 1.702249 / 1.689343
     - 1.592884–1.610268
   * - ``square_1024_float32``
     - 19.456243 / 19.936642
     - 13.611237 / 13.597008
     - 11.938359–12.267190
   * - ``square_1024_float16``
     - 19.121096 / 18.339178
     - 13.730320 / 13.295695
     - 12.260777–12.658571
   * - ``skinny_m_float32``
     - 2.498620 / 2.223136
     - 1.619132 / 1.577696
     - 1.197634–1.544633
   * - ``skinny_n_float32``
     - 1.508815 / 1.449786
     - 1.050207 / 1.046921
     - 1.027666–1.233049

Rejected paired-row skinny-N follow-up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original Python parity runner could not be repeated in this environment:
the onnx-light runtime is not installed or available from its configured
package index. The standalone C++ ``gemm_throughput`` driver was therefore
used to measure the paired-row kernel in isolation, not as an end-to-end or ORT
comparison. Baseline sources were ``86c8762be5bb0c01459c1998f5daf8dc70e268fa``;
the paired-row sources were ``232b1ca5f1050b144d40129770d2b499d42b5c97``.
Both were built with GCC 13.3.0, CMake Release, and the AVX2 ceiling. This
shared AMD EPYC 9V74 VM has four logical processors and two physical cores;
the processes were restricted to CPUs ``0,2``. The driver reported AVX2,
FMA, AMD Zen, and six register rows.

For each revision and thread count, two fresh processes ran
``gemm_throughput <threads> skinny_n fp32``. Each case uses three warm-ups and
51 measured calls; the table shows the two per-process median latencies for
``M=1024, N=1, K=1024``. The substring filter also runs the dashboard skinny-N
cases before the named ``skinny_n`` result. Speedup is the mean baseline
latency divided by the mean paired-row latency.

.. list-table::
   :header-rows: 1

   * - Threads
     - Baseline, runs 1 / 2 (µs)
     - Paired-row ``232b1ca5``, runs 1 / 2 (µs)
     - Speedup
   * - 1
     - 47.441 / 47.341
     - 51.187 / 51.107
     - 0.93x
   * - 2
     - 51.367 / 50.426
     - 54.772 / 54.281
     - 0.93x

The paired-row optimization was removed because it regressed this workload
at both thread counts. The retained skinny-N implementation is the original
single-row, four-chain reduction, unchanged from the baseline. No shape or
CPU eligibility gate is introduced without measurements demonstrating a
profitable regime, and this PR claims no skinny-N speedup.

Findings and validation
~~~~~~~~~~~~~~~~~~~~~~~

Hardware-counter profiling was unavailable (``perf_event_paranoid=4``).
Disassembly of GCC's six-row FP32 microkernel showed twelve accumulator-array
stores inside each four-step K iteration. Named accumulators remove that
repeated array writeback. The retained epilogue consumes those accumulators
directly through compile-time-unrolled row stores, without intermediate arrays.
The same microkernel serves converted FP16 square panels. No reduction-order,
packing, or FP16 conversion change was needed for these squares.

Skinny-M improved by limiting FP32's row slab to four rows and increasing
contiguous column slices up to 1024 columns while retaining enough tasks for
the executor. The FP16 slab remains unchanged. A wider-column-only experiment
was not sufficient; it was not retained without the shallow-slab improvement.
The original optimized skinny-N kernel shared each B vector across four A rows
with two reduction chains, but changed finite-input cancellation behavior.
The subsequent paired-row implementation preserved the original four-chain
reduction while sharing B across two rows, but regressed the measured workload.
Neither skinny-N optimization is retained.

Relative to the mean of the two baseline medians, square-case improvements
range from 1.21x to 1.45x, and skinny-M from 1.48x to 2.40x in the original
parity campaign. Its 1.41–1.74x skinny-N result applies only to revision
``83952ac5`` and does not apply to the retained implementation. The follow-up
isolated C++ measurements above show 0.93x at both thread counts on a different
shared VM; they are not comparable to the original Python/ORT campaign and do
not establish end-to-end parity. They motivated restoring the original skinny-N
kernel rather than retaining a measured regression. Remaining ORT advantages
in the original campaign include the larger squares, two-thread FP16
128-square, and two-thread skinny-M. The original historical latencies were
not reproduced exactly, and their thread/build provenance remains unknown.

After restoring the single-row kernel, the AVX2 build passed 100 GEMM kernel
tests, 49 plan tests, and seven MatMul integration tests on an Intel Xeon
Platinum 8480C with GCC 13.2.0. Unsupported AVX-512/AMX cases were skipped.
The original parity campaign passed 99 GEMM kernel tests, 49 plan tests, and
seven MatMul integration tests; unsupported AVX-512/AMX cases were skipped.
Added coverage includes all FP32 microtile row counts and accumulation modes, unaligned
column ranges, vector/K tails, skinny-N special values and cancellation, ignored
NaN bias, and skinny-M executor limits at panel boundaries. All eight exact
benchmark inputs also matched onnx-light's built-in kernels at one and two
threads, with reference outputs computed before CPU-kernel registration
(FP32 ``rtol=atol=1e-4``, FP16 ``rtol=atol=1e-2``).
