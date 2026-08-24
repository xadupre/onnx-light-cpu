.. _l-next-steps-processor-performance-profile:

Processor Memory and Compute Performance Profile
================================================

:Date: 2026-08

**complete** (Profile PR05 portability gate)

Objective
---------

The objective is to expose one reproducible, explicit benchmark that estimates
the effective L1, L2, L3, and RAM bandwidth and latency visible to
``onnx-light-cpu``, together with the processor's sustained arithmetic
throughput. The final Python API must return a structured profile and one
gallery example must visualize it.

This profile is the first input to a future optimal-transport model for GEMM
tile placement. It is also independently useful for kernel tuning and Roofline
analysis.

Interpretation
--------------

Software cannot command or directly observe every physical transfer between
RAM and the cache hierarchy. The reported values are therefore **effective
working-set measurements**, not theoretical bus specifications:

* an L1 measurement repeatedly accesses a working set that fits comfortably in
  the detected L1 data cache;
* an L2 measurement exceeds L1 while fitting comfortably in L2;
* an L3 measurement exceeds private caches while fitting in the shared
  last-level cache;
* a RAM measurement exceeds the last-level cache and streams enough data to
  prevent cache residency from dominating.

The result must record the working-set size and method used for every value.
Missing topology, insufficient memory budget, unsupported affinity, and
unavailable cycle counters are explicit diagnostics; they must not produce
plausible-looking fallback numbers.

Non-goals
---------

This roadmap does not:

* change GEMM blocking, scheduling, or arithmetic;
* run calibration during module import, session creation, or inference;
* claim a physical cache-link bandwidth from a working-set benchmark;
* infer processor quality from a single best timing;
* require hardware performance counters or privileged access;
* enforce ``L1 > L2 > L3 > RAM`` on noisy, virtualized, or heterogeneous
  machines.

Measurement contract
--------------------

All timed kernels must:

* allocate, align, initialize, and prefault memory before timing;
* pin participants when the operating system supports affinity;
* execute warmups separately from recorded samples;
* run long enough to exceed timer noise while respecting a configured duration
  and memory budget;
* retain every raw sample and report median and dispersion;
* consume a checksum outside the timed region so the compiler cannot remove
  work;
* record the resolved affinity, topology, SIMD path, compiler, operating
  system, and whether the process appears virtualized;
* use monotonic wall-clock time as the portable reference and report cycles
  only when a reliable platform counter is available.

The benchmark is intentionally explicit and expensive. It runs only through a
user call, command-line tool, or documentation example.

Memory profile
--------------

Topology
~~~~~~~~

The implementation reuses ``GetCpuTopology`` for process-visible logical
threads, physical cores, SMT relationships, core kinds, and affinities. Cache
detection currently private to ``gemm_blocking.cc`` must become a reusable
internal descriptor containing:

* level and kind;
* size and cache-line bytes;
* sharing count or sharing mask where available;
* whether the value is detected, inferred, or a portable fallback.

The benchmark must not alter the safe blocking defaults used by GEMM.

Working sets
~~~~~~~~~~~~

For each available level, choose a size with a safety margin rather than
testing exactly at a cache boundary:

``L1``
    At most half of the usable L1 data cache per participant.

``L2``
    Larger than L1 and at most half of the usable L2 per participant.

``L3``
    Larger than the aggregate private-cache footprint and at most half of the
    participant-visible shared cache.

``RAM``
    Larger than twice the participant-visible last-level cache. If the
    configured memory budget cannot satisfy this condition, mark RAM bandwidth
    unavailable instead of silently measuring cache.

Heterogeneous cache domains must be reported separately or restricted to a
homogeneous selected affinity set. The profile records the exact resolved
working-set bytes.

Bandwidth kernels
~~~~~~~~~~~~~~~~~

Measure at least:

``read``
    Sequential aligned loads with enough independent accumulators to avoid a
    reduction dependency becoming the bottleneck.

``write``
    Sequential cached stores. Non-temporal stores, when supported, are a
    separate mode and never replace the portable result.

``copy``
    One read and one write stream with traffic accounting stated explicitly.

``read_modify_write``
    A load, arithmetic update, and store to the same stream.

Report useful bytes per second and the traffic convention used by each kernel.
Run one participant and the selected physical-core participant count. Each
participant uses disjoint, aligned storage; shared-cache and RAM results report
aggregate throughput.

Latency kernels
~~~~~~~~~~~~~~~

Use a randomized dependent pointer chase so the next address cannot be issued
before the previous load completes. Build the permutation outside the timed
region and validate that every element is visited exactly once.

Report nanoseconds per dependent load and cycles per load where a reliable
cycle counter exists. Bandwidth and latency remain separate metrics.

Compute profile
---------------

Compute kernels keep operands and accumulators in registers and expose enough
independent accumulators to cover instruction latency. They must not read a
large working set or measure memory bandwidth accidentally.

The initial profile covers:

* FP32 and FP64 on every platform;
* FP16 and BF16 only when the compiled and detected ISA provides a meaningful
  native arithmetic path;
* INT8 dot-product throughput only when a native dot-product path exists.

One fused multiply-add counts as two floating-point operations. Integer results
use operations per second with the exact dot-product convention recorded.
Every result identifies the actual scalar, SSE, AVX2, AVX-512, NEON, SVE, AMX,
or other implementation selected.

Measure one participant and the selected physical-core participant count.
Participant creation, synchronization, and final checksum reduction remain
outside the timed arithmetic body.

Public result
-------------

The final immutable result is ``ProcessorPerformanceProfile`` (name fixed by
this roadmap). It contains:

``metadata``
    Schema version, timestamp, platform, compiler, SIMD implementation,
    measurement options, timer, and diagnostics.

``topology``
    Process-visible logical threads, physical cores, selected affinities, cache
    descriptors, and topology confidence.

``memory``
    One entry per measured level and participant policy. Each entry stores
    working-set bytes plus raw and summarized bandwidth and latency samples.

``compute``
    One entry per element type, implementation, and participant policy with raw
    and summarized operation rates.

``roofline``
    Derived compute ceilings and the arithmetic-intensity crossover for each
    memory level. Derived values retain references to their source
    measurements.

``warnings``
    Explicit unavailable, inferred, noisy, unpinned, virtualized, or
    memory-budget-limited conditions.

The result provides ``to_dict()`` for stable JSON serialization. Schema
evolution requires an explicit version.

Target Python API
-----------------

The expensive action is visible in the function name:

.. code-block:: python

    from onnx_light_cpu import benchmark_processor_performance

    profile = benchmark_processor_performance(
        thread_policies=("single", "physical"),
        repeats=7,
        minimum_duration_ms=50,
        memory_budget_bytes=512 * 1024 * 1024,
        include_latency=True,
    )

    print(profile.memory["L1"]["single"].read.median_gbps)
    print(profile.memory["RAM"]["physical"].copy.median_gbps)
    print(profile.compute["float32"]["physical"].median_gflops)

Invalid durations, repeat counts, memory budgets, thread policies, and explicit
affinities fail before allocating or timing. A missing level is absent from the
measurement map and explained in ``warnings``.

Python example
--------------

``docs/examples/processor/plot_processor_performance.py`` must:

* call the public function rather than duplicate measurement code;
* print topology, working-set sizes, warnings, and a compact result table;
* plot read/write/copy bandwidth by memory level;
* plot dependent-load latency by level;
* plot arithmetic throughput by element type and participant policy;
* render a Roofline chart from the same returned profile;
* use a bounded ``UNITTEST_GOING=1`` mode that still executes every public
  result path.

The example labels results as effective measurements. It must not use words
such as hardware maximum, physical link rate, or guaranteed peak.

Implementation sequence
-----------------------

Every row is exactly one issue and one pull request. A PR must close only its
own issue and must not absorb a later row.

.. list-table::
   :header-rows: 1
   :widths: 12 27 41 12 8

   * - PR
     - Scope
     - Merge criterion
     - Depends on
     - Status
   * - Profile PR01
     - Reusable processor and cache descriptors.
     - Process-visible topology and cache descriptors are deterministic,
       cross-platform, confidence-labelled, and covered by injected-topology
       tests. Existing GEMM blocking behavior is unchanged.
     - Completed runtime controls
     - Completed
   * - Profile PR02
     - Memory bandwidth and latency measurement engine.
     - L1/L2/L3/RAM working-set selection, read/write/copy/read-modify-write,
       pointer-chase latency, affinity, raw samples, budgets, and explicit
       unavailable states pass deterministic and bounded native tests.
     - PR01
     - Completed
   * - Profile PR03
     - Register-resident compute measurement engine.
     - FP32/FP64 and supported low-precision paths report correct operation
       counts, implementation identity, raw samples, and single/physical-core
       throughput without memory-sized timed operands.
     - PR01
     - Completed
   * - Profile PR04
     - Aggregate profile and public Python API.
     - ``benchmark_processor_performance`` returns the immutable, versioned,
       serializable profile; validation and warning semantics are tested and no
       benchmark runs during import, session creation, or inference.
     - PR02, PR03
     - Completed
   * - Profile PR05
     - Python example, Roofline derivation, and final portability gate.
     - The example exercises every result section in normal and bounded modes;
       Linux, Windows, and macOS are supported or explicitly diagnosed; focused
       C++/Python tests and a warnings-as-errors documentation build pass.
     - PR04
     - Completed

Profile PR05 completes this roadmap.

Correctness and acceptance
--------------------------

Tests must validate byte and operation accounting independently of elapsed
time. Timing tests enforce finite positive values and bounded duration, but do
not enforce a universal ordering between cache levels.

Final acceptance requires:

* deterministic topology and working-set selection for injected descriptors;
* exact useful-byte and operation counts for every kernel;
* no timed allocation, page fault setup, worker creation, or checksum
  consumption;
* explicit behavior when affinity, cache information, cycles, native low
  precision, or sufficient RAM is unavailable;
* raw samples, medians, and dispersion in every successful measurement;
* repeatable serialization with a schema version;
* no process-global tuning mutation and no inference-time execution;
* focused C++ and Python tests, repository formatting, and a clean
  warnings-as-errors documentation build.

Relationship to optimal-transport GEMM planning
-----------------------------------------------

A later roadmap may consume this profile as the cost model for tile movement:

.. math::

    T_{\mathrm{compute}} = \frac{\mathrm{operations}}{P_{\mathrm{compute}}},
    \qquad
    T_l = \frac{\mathrm{bytes\ moved\ at\ level}\ l}{B_l}.

The first conservative tile estimate is:

.. math::

    T_{\mathrm{tile}} =
    \max(T_{\mathrm{compute}}, T_{\mathrm{L1}}, T_{\mathrm{L2}},
         T_{\mathrm{L3}}, T_{\mathrm{RAM}}).

This roadmap only supplies measured inputs and their uncertainty. It does not
select GEMM tiles, infer overlap, or claim that the Roofline maximum is an exact
execution-time predictor.
