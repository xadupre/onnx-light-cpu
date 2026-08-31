com.microsoft Domain Support
============================

:Date: 2026-08

**complete** (`#479 <https://github.com/xadupre/onnx-light-cpu/pull/479>`_)

Objective
---------

The objective was to establish the first non-standard ONNX domain supported
end to end by ``onnx-light-cpu``. The implementation introduces
``com.microsoft::CDist`` and ``com.microsoft::BiasGelu`` through the same
schema, graph-building, runtime, differentiation, optimization, testing, and
documentation surfaces used by standard operators.

Operator contract
-----------------

``CDist``
    Computes pairwise distances between rows of two rank-two tensors. Version
    1 supports ``FLOAT`` and ``DOUBLE`` with ``sqeuclidean`` and ``euclidean``
    metrics and produces an ``[M, K]`` result from ``[M, N]`` and ``[K, N]``
    inputs.

``BiasGelu``
    Adds a rank-one last-dimension bias and applies exact GELU. Version 1
    supports ``FLOAT16``, ``FLOAT``, ``DOUBLE``, and ``BFLOAT16`` while
    preserving the input shape and type.

CDist latency parity
--------------------

``tools/benchmark_cdist_parity.py`` compares the optimized kernel directly
with ONNX Runtime's CPU execution provider for both metrics and both supported
types.  The matrix reproduces the ONNX Runtime FLOAT and DOUBLE vectors and
includes empty rows, singleton and rectangular inputs, feature counts around
SIMD boundaries, repeated and near-identical points, a large feature tail,
representative model shapes, and sizes immediately around each type's
parallel threshold.

Both runtimes reuse the same inputs and receive the same thread count,
affinity, warm-up, alternating sample order, output-allocation policy, and
sequential execution mode.  Every JSON row retains raw samples, median, p90,
interquartile range, candidate order, maximum numerical difference, and
median/tail speed-up.  Run the benchmark on an idle pinned host, once at the
base revision and once at the candidate revision:

.. code-block:: bash

   python tools/benchmark_cdist_parity.py --cpus 0 --threads 1 \
       --profile-runs 100 --output cdist_parity_results.json --enforce

Profiling calls are excluded from the timed samples.  ``--profile-runs`` adds
the onnx-light-cpu Python/runtime call breakdown and ONNX Runtime's native
per-node trace, allowing setup and dispatch overhead to be separated from node
computation.  The report records the revision, build flags, package versions,
affinity, allocator policy, and effective execution policy.

ONNX Runtime expands squared distance as
``-2 * A @ transpose(B) + sum(A**2) + sum(B**2)`` and applies ``abs`` before
returning it or taking ``sqrt``.  The optimized onnx-light-cpu kernel
deliberately retains direct squared-difference accumulation: it cannot produce
a small negative distance and is more accurate for near-identical,
large-magnitude points.  Ordinary cases use the command-line parity tolerance;
only repeated and near-identical cases use a reported cancellation bound
``8 * eps * N * scale**2`` (or its square root for ``euclidean``).  Non-finite
values follow IEEE propagation, and the generic kernel rejects a zero feature
dimension like ONNX Runtime while permitting empty row dimensions.

BiasGelu latency parity
-----------------------

``tools/benchmark_bias_gelu_parity.py`` compares the optimized FLOAT kernel
directly with ONNX Runtime's CPU execution provider.  Its matrix includes
empty and singleton tensors, sizes immediately around AVX2 and AVX-512 vector
boundaries, transformer dimensions, a large outer dimension, and sizes around
the 256 KiB parallel threshold.  Both runtimes receive the same input buffers,
thread count, process affinity, warm-up count, alternating sample order, and
default arena/memory-pattern allocation policy.  Each JSON row retains raw
samples, median, p90, interquartile range, and median/tail speed-up.

Use an idle pinned host and the same release build for published comparisons:

.. code-block:: bash

   python tools/benchmark_bias_gelu_parity.py --cpus 0 --threads 1 \
       --profile-runs 100 --output bias_gelu_parity_results.json --enforce

``--profile-runs`` adds a separate cProfile call breakdown for onnx-light-cpu
and ONNX Runtime's native per-node trace to the report; these diagnostic calls
are excluded from latency samples.  This distinguishes Python/runtime setup
and dispatch from node computation instead of attributing a wall-clock gap to
the kernel without evidence.  The report also records the revision, affinity,
compiler flags, package versions, allocator policy, seed, and effective
execution policy.  Shared CI validates the runner and direct numerical parity;
latency enforcement belongs on a dedicated machine.

FLOAT uses one runtime-selected scalar, AVX2/FMA, or AVX-512 implementation.
SIMD remainders stay in vector registers and use the same polynomial and
operation ordering as full vectors; dispatch selection is hoisted out of the
row loop.  FLOAT16 and BFLOAT16 are intentional extensions beyond ONNX
Runtime's FLOAT CPU registration and remain checked against the independent
naive kernel with dtype-specific tolerances.  DOUBLE is likewise validated
independently rather than included in the ONNX Runtime latency gate.

Delivered integration
---------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Surface
     - Contribution
   * - Schemas
     - Adds lightweight versioned schemas and a combined schema lookup so
       graph builders can resolve standard ONNX and ``com.microsoft``
       operators through one callback.
   * - Shape and memory
     - Registers symbolic shape inference and zero-scratch peak-memory
       functions for both operators.
   * - CPU runtime
     - Adds portable typed implementations, registered ``KernelBase`` adapters,
       tuning schemas, usage names, and structured inventory metadata.
   * - Fusion patterns
     - Fuses exact ``Add`` plus ``Gelu`` into ``BiasGelu`` and the
       squared-distance ``Unsqueeze``/``Sub``/``Mul``/``ReduceSum`` expansion
       into ``CDist(metric="sqeuclidean")``.
   * - Gradients
     - Registers standard-ONNX backward graphs for both operators. The C++
       documentation added in
       `#484 <https://github.com/xadupre/onnx-light-cpu/pull/484>`_ describes
       the complete gradient data flow.
   * - Python API
     - Exposes custom schemas, combined schema lookup, operator-support
       inventory, support registration, and gradient registration alongside
       the existing kernel registration entry point.
   * - Tests and examples
     - Adds low-level, runtime, registration, usage, gradient, fusion, backend
       TEST/BENCHMARK, gallery, and ONNX Runtime comparison coverage.

Validation and completion
-------------------------

The completion gate requires both operators to build through their custom
schemas, infer shapes, report peak memory, execute through the registered
onnx-light runtime, differentiate into standard ONNX nodes, and participate in
their guarded fusion patterns. Their registrations must also appear in the
public kernel inventory and backend corpus.

All layers and their focused C++ and Python tests were delivered in #479.
This establishes the domain-oriented directory and registration pattern to
follow when another custom domain or operator is added.
