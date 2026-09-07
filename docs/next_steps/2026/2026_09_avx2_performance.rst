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

September 7 dashboard follow-up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The registered ``Sigmoid`` and last-axis ``Softmax`` dispatchers could not
reach their existing fused AVX2/FMA implementations: the implementation
library's ``ONNX_LIGHT_CPU_HAVE_AVX2_FMA`` definition was private and absent
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
