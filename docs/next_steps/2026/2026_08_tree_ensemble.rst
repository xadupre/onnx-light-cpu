Tree Ensemble Classification and Regression Roadmap
===================================================

:Date: 2026-08

**complete**

Objective
---------

The objective is a prepared, processor-aware CPU engine for classification
and regression forests with performance parity against the ONNX Runtime CPU
execution provider. The implementation covers only version 5 of the
``ai.onnx.ml`` operator set. Versions 1 through 4, their conversion rules, and
their historical edge behavior are explicitly out of scope.

ONNX-ML opset 5 introduces the common ``TreeEnsemble`` operator and deprecates
``TreeEnsembleClassifier`` and ``TreeEnsembleRegressor``. The common operator
is the primary implementation target. The two deprecated version-5 schemas
remain in scope as adapters because existing exporters may still emit them,
but they must lower into the same prepared engine rather than retain separate
tree evaluators.

For the priority corpus, parity means median end-to-end performance of at
least ``1.0x`` ONNX Runtime, no priority case below ``0.9x``, and no
single-row latency regression greater than 10% after tuning. Model preparation
is measured separately from repeated inference.

Latest-opset scope
------------------

``TreeEnsemble`` version 5
~~~~~~~~~~~~~~~~~~~~~~~~~~

The preferred schema consumes a rank-2 tensor ``[N, F]`` and returns
``[N, n_targets]``. Its input, split, membership, leaf-weight, and output types
are ``float16``, ``float32``, or ``float64`` with matching types.

The first parity milestone covers float32 and float64 because those are the
types registered by ONNX Runtime's current version-5 CPU kernel. Float16
correctness is required from the first complete implementation, but its
optimized path is a later extension and is not used to claim an
apples-to-apples ONNX Runtime parity result.

The engine must implement:

* branch modes ``LEQ``, ``LT``, ``GTE``, ``GT``, ``EQ``, ``NEQ``, and
  ``MEMBER``;
* explicit true/false node and leaf references;
* multiple tree roots and multiple targets;
* missing-value routing through ``nodes_missing_value_tracks_true``;
* membership sets delimited by NaNs in ``membership_values``;
* aggregation ``AVERAGE``, ``SUM``, ``MIN``, and ``MAX``;
* post transforms ``NONE``, ``SOFTMAX``, ``LOGISTIC``, ``SOFTMAX_ZERO``, and
  ``PROBIT``.

Classification uses ``TreeEnsemble`` scores followed by ``ArgMax`` and, when
labels are not zero-based integers, ``LabelEncoder`` or ``GatherND``. The
roadmap must benchmark both the score kernel alone and this complete
classification graph.

Deprecated version-5 adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TreeEnsembleRegressor`` version 5 supports float, double, int32, and int64
inputs and produces float scores. ``TreeEnsembleClassifier`` version 5
supports the same inputs, float scores, and string or int64 labels. Their
legacy attribute tuples are parsed once and converted directly into the
canonical prepared representation.

No version dispatch is added to the hot path. Registration is exact for
``ai.onnx.ml`` opset 5, and models importing an older ML opset continue to use
another implementation or fail according to the normal dispatch contract.

Correctness contract
--------------------

Model preparation validates all structural invariants before execution:

* every required attribute has the expected type, rank, and length;
* feature, node, leaf, target, and tree-root indices are in range;
* every internal node is reachable from exactly one tree root;
* trees contain no cycles or shared internal nodes;
* every path terminates at a valid leaf;
* each membership node owns one non-empty, correctly delimited set;
* ``n_targets`` is positive and every leaf target is in range;
* split and leaf values use the type required by the selected v5 schema;
* deprecated classifier adapters define exactly one label representation.

Differential tests cover every comparison at below, equal, and above the
threshold; positive and negative zero; infinities; NaNs routed both ways;
float16 rounding boundaries; membership hit, miss, duplicate, empty-invalid,
and large-set cases; one-node trees; unbalanced and maximum-depth trees;
multiple targets; empty batches; and every aggregate/post-transform
combination.

``SUM`` and ``AVERAGE`` preserve deterministic tree order by default.
Parallel candidates may change floating-point reduction grouping only when
the result remains inside the documented tolerance and classification labels
do not change. ``MIN`` and ``MAX`` preserve NaN and tie behavior. Class
selection has an explicit stable tie rule matching the latest ONNX contract
and the scalar reference.

Prepared tree plan
------------------

Each node constructs an immutable ``TreeEnsemblePlan`` during session
preparation. Repeated execution performs no attribute parsing, tree
validation, metadata allocation, string lookup, tuning-cache lookup, or lock
acquisition. Output and bounded workspace allocation remain runtime-visible,
but traversal loops perform no allocation.

The plan records:

* canonical tree roots, node and leaf counts, maximum and average depth;
* input, split, accumulation, and output types;
* feature, target, and class-label metadata;
* aggregate and post-transform functions selected as typed function pointers;
* branch-mode and missing-value distributions;
* the prepared node, leaf, and membership layouts;
* a batch-size-dependent execution policy containing strategy crossovers,
  row/tree chunks, batch sizes, and participant caps;
* preallocated workspace requirements and alignment;
* an exact model signature.

Node layout
~~~~~~~~~~~

The first implementation includes ONNX Runtime's compact traversal layout as
the reference candidate rather than assuming pointer-free indices are faster.
Every layout owns stable storage in the plan. Candidates include:

``ort_compact_aos_pointer``
    One compact record stores the feature id, split or unique leaf weight, a
    tagged mode/missing byte, and either a direct pointer to the true child or
    leaf-weight metadata. Nodes are reordered so the false child is the next
    record. This mirrors ONNX Runtime's scalar-friendly layout and is the
    initial parity baseline.

``compact_aos_index``
    One compact record replaces child pointers with 32-bit indices. It reduces
    record size and permits relocation but is selected only when measured
    traversal wins offset the address calculation.

``split_soa``
    Separate aligned arrays for feature ids, splits, children, modes, and
    missing flags. This favors batched or interleaved traversal and avoids
    loading unused fields.

``preorder_hot``
    Trees are reordered into depth-first layout, with ``nodes_hitrates`` used
    only as an optional layout hint. The likely branch becomes fall-through
    where this improves locality; true/false semantics remain explicit. Since
    ONNX Runtime deliberately ignores hit rates during inference, this layout
    cannot become a portable default without tests against shifted input
    distributions and a demonstrated end-to-end win.

The initial portable default is selected from measured evidence, not from the
smallest node size alone. Layout conversion happens once. Large indices retain
a safe 64-bit fallback when a model cannot be represented by the compact
format.

Membership layout
~~~~~~~~~~~~~~~~~

Membership nodes select one immutable representation during preparation:

* linear scan for very small sets;
* sorted array plus binary search for medium sets;
* bounded integer bitset when the value range is compact;
* immutable hash set for large irregular sets.

The first baseline mirrors ONNX Runtime's split between a compact integer
bitmask for small sets and a prepared category-set structure for large sets.
Additional representations enter calibration only after this baseline passes.
The representation threshold is tunable, but the values and floating-point
equality semantics are not. NaN remains the set delimiter and is never a
member value.

Execution strategies
--------------------

No single traversal order is optimal for every combination of rows, trees,
depth, and targets. The plan therefore stores an ordered decision table
evaluated from the runtime batch size ``N``. It does not capture one strategy
for every dynamic shape.

The initial portable policy reproduces ONNX Runtime's current scheduling
structure and constants as a benchmark baseline: tree parallelism begins near
80 trees, tree-major batches contain 128 rows, and the small-row crossover is
50 rows. These values are not the final defaults; tuning must prove each
replacement.

The policy chooses among bounded strategies:

``row_parallel``
    Each task owns a contiguous range of input rows and evaluates all trees.
    It needs no cross-thread reduction and is the default for large batches.

``tree_parallel``
    Tasks own contiguous tree ranges for one or a few rows and write
    thread-local accumulators followed by a deterministic merge. It targets
    single-row or very small-batch inference with large forests.

``tree_major_batch``
    A cache-sized row batch is initialized, then each tree is evaluated over
    the complete batch. It reuses tree nodes and branch-predictor state while
    input rows remain cache-resident.

``interleaved_rows``
    Several rows traverse one tree together. It may use SIMD comparisons and
    gathers when paths remain coherent, but must fall back cheaply as lanes
    diverge. This is an extension after float32/float64 parity, not a portable
    default or an input to the first scheduling calibration.

Static traversal specialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model preparation emits separate typed traversal functions for the cases
already exploited by ONNX Runtime:

* one homogeneous comparison mode versus mixed node modes, so a homogeneous
  forest has no per-node mode switch;
* no missing-value routing versus explicit NaN routing, so ordinary forests
  do not test NaN at every node;
* one target versus multiple targets, with a scalar accumulator for the
  single-target path;
* binary classification versus multiclass classification;
* all-positive classifier weights versus general signed weights.

The hot loop selects one specialization when the plan is constructed. These
specializations are part of the parity baseline and precede SoA, prefetch, or
SIMD traversal experiments.

The plan detects degenerate one-node trees, stumps, and symmetric/oblivious
trees. Specialized branchless evaluators are permitted when detection proves
the required structure; arbitrary trees always retain the general evaluator.

Workspace
---------

All execution workspace is sized during plan creation and obtained from the
runtime allocator once per invocation. Per-row vectors and per-thread
accumulators must not allocate inside traversal loops.

Workspace is bounded by:

.. code-block:: text

    participants * active_rows * n_targets * sizeof(accumulator)

The plan reduces batch size or participants before exceeding its configured
memory limit. Sparse target accumulation is a separate candidate for models
whose leaves update few targets; dense accumulation remains the default for
small ``n_targets``.

Benchmark corpus
----------------

The corpus contains both generated v5 graphs and converted real-world model
families:

* random forests, extra trees, gradient-boosted trees, and isolation-style
  unbalanced forests;
* regression with 1, 2, 16, and 128 targets;
* binary and multiclass classification with 2, 10, 100, and 1,000 classes;
* 1 to 4,096 trees, depths 1 to 16, and 8 to 4,096 features;
* batches 1, 2, 8, 32, 128, 1,024, and 16,384;
* balanced, skewed, stump-heavy, and mixed-depth forests;
* dense numeric, missing-value-heavy, and membership-heavy inputs;
* float16, float32, float64, plus deprecated-adapter int32/int64 inputs.

Float32 and float64 form the performance-parity corpus. Float16 and deprecated
integer adapters form correctness and internal-regression corpora until an
equivalent ONNX Runtime version-5 kernel exists for a direct comparison.

Every case records preparation time, first-run latency, steady-state latency,
rows per second, traversed nodes per second, branch misses, cache misses,
workspace bytes, strategy, layout, tuning parameters, raw samples, and
dispersion. Hardware counters are diagnostic and optional; correctness and
wall-clock results are mandatory.

Published comparisons use identical models, inputs, affinity, and normal
multithreaded runtime settings. Equal-thread and single-thread measurements
diagnose kernel and scheduling differences. Construction and conversion time
must not be hidden in steady-state inference, but neither may it be charged on
every run.

PR04 scheduling baseline measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The isolated ``tree_ensemble_throughput`` driver records the selected strategy,
bounded workspace, latency, and throughput. The initial diagnostic run used an
AMD EPYC 7763 GitHub runner exposing two physical cores and four logical
processors, a persistent executor without affinity, Release mode, three warmups,
and the median of eleven samples. The synthetic forests contain one-target
stumps; these numbers validate scheduling and expose the cost of the portable
ONNX Runtime crossover table rather than claiming final tuning wins.

.. list-table:: Rows per second (2026-08-20)
   :header-rows: 1
   :widths: 28 10 12 12 12 14

   * - Scenario
     - Strategy at 4 threads
     - 1 thread
     - 2 threads
     - 4 threads
     - Physical (2)
   * - 1 row, 1,024 trees
     - ``tree_parallel``
     - 98,629
     - 25,966
     - 22,048
     - 29,724
   * - 4,096 rows, 81 trees
     - ``tree_parallel``
     - 1,312,423
     - 1,482,799
     - 1,381,058
     - 1,509,951
   * - 4,096 rows, 3 trees
     - ``row_parallel``
     - 30,739,443
     - 3,819,951
     - 40,458,317
     - 4,170,332

The large penalty for two-thread, three-tree execution is inherited from the
baseline rule (trees greater than or equal to workers choose tree parallelism)
and is evidence for PR05 tuning rather than a reason to alter the compatibility
table in PR04. Peak scratch space in this run was 8,192 bytes for the
four-thread, 128-row tree-parallel batch.

Tuning architecture
-------------------

Tuning has two levels:

Static structural selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model preparation derives facts that do not need timing:

* whether compact 32-bit indices are safe;
* whether a tree is a stump, symmetric, or general;
* whether mode, missing-value, target-count, and binary-classifier
  specializations apply;
* membership representation candidates;
* whether dense or sparse target accumulation is legal;
* workspace bounds for each strategy.

Invalid or dominated candidates are removed before calibration. Static
selection never depends on the input values used for one inference.

Measured scheduling selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibration times the remaining legal candidates using deterministic inputs
that exercise representative paths. It stores a bounded, ordered execution
policy rather than one global strategy. Each region has an inclusive maximum
row count, a strategy, a batch size, row/tree chunks, and a participant cap;
the final region has no maximum.

``execution.regions``
    One to four strictly ordered row-count regions selecting
    ``row_parallel``, ``tree_parallel``, or ``tree_major_batch``. The initial
    implementation excludes ``interleaved_rows`` until the parity baseline is
    complete.

``execution.regions[].maximum_rows``
    Inclusive upper batch-size boundary. The final region is unbounded.

``execution.regions[].strategy``
    Strategy selected for runtime batch sizes in the region.

``execution.regions[].batch_rows``
    Number of active rows retained by tree-major execution.

``execution.regions[].maximum_threads``
    Maximum useful participants in the region.

``execution.regions[].row_chunk``
    Contiguous rows assigned per row-parallel task.

``execution.regions[].tree_chunk``
    Contiguous trees assigned per tree-parallel task.

``membership.linear_limit``
    Largest membership set evaluated with a linear scan.

``membership.bitset_range_limit``
    Largest compact non-negative integer range represented as a bitset.

``traversal.prefetch_distance``
    Optional node-prefetch distance; zero disables prefetch.

Parameters are strongly typed and range-checked. Region boundaries are
strictly increasing, every strategy-specific field is validated, workspace
limits are checked per region, and the complete policy is captured immutably
by the prepared plan.

Tuning key
~~~~~~~~~~

An exact profile key contains:

.. code-block:: text

    library          = onnx_light_cpu
    kernel           = TreeEnsemble
    domain/opset     = ai.onnx.ml/5
    implementation   = prepared_tree_ensemble
    input_type       = exact tensor element type
    accumulator_type = exact accumulation type
    processor        = normalized CPU and feature descriptor
    threads          = effective session thread count
    model_signature  = canonical structural digest

The canonical digest covers tree topology, feature ids, modes, missing flags,
target ids, value types, and the structural buckets used by scheduling. Raw
split and leaf values may be excluded only if two models are guaranteed to
share all legal execution choices; otherwise they remain in the digest.

A portable default profile is indexed by structural buckets: tree-count
range, depth range, target-count range, branch-mode mix, and membership
density. Runtime row count is intentionally absent from the profile key
because ``execution.regions`` encodes all calibrated row-count crossovers.
Exact profiles override portable defaults. A profile for one forest must
never silently apply to an incompatible topology or workspace requirement.

Calibration inputs and correctness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibration builds a bounded input set from model metadata:

* values immediately below, equal to, and above sampled thresholds;
* NaNs for features with explicit missing routing;
* membership hits and misses for every representation family;
* deterministic random rows covering the observed feature range;
* batch sizes around every candidate crossover.

The serial prepared evaluator is the reference. Every candidate must pass
typed output comparison before its timing is accepted. Regression scores use
an aggregate-specific floating tolerance. Classification requires identical
labels and score tolerance. Candidate failure is explicit and stored with its
reason; it never becomes a success-shaped fallback.

Search procedure
~~~~~~~~~~~~~~~~

Calibration uses a hierarchical search to avoid a combinatorial sweep:

#. benchmark the ONNX Runtime-style AoS/pointer baseline against legal serial
   layouts and membership representations;
#. verify homogeneous-mode, no-missing, one-target, and classifier
   specializations independently;
#. compare traversal strategies at representative row-count points and derive
   an ordered decision table rather than one global winner;
#. sweep ``batch_rows`` over bounded powers of two that fit the cache and
   workspace budget;
#. locate decision-region boundaries by exponential search followed by
   refinement;
#. sweep participant caps, then row/tree chunk sizes within each region;
#. test prefetch only after the winning layout and strategy are fixed;
#. revalidate the winner on all calibration batches and edge inputs.

Candidates run in alternating order with warmups, median samples, dispersion
checks, and a duration budget. A new candidate must win by a configurable
noise margin and repeat the win before replacing the portable default.
Successive halving may discard clear losers early. Calibration records all
samples and the rejected-candidate reasons for inspection.

Lifecycle and cache
~~~~~~~~~~~~~~~~~~~

The schema and calibration callback are registered before session creation.
Profile resolution happens while constructing ``TreeEnsemblePlan``. Repeated
execution performs no registry or cache access. Existing sessions retain
their captured profile generation when a later calibration updates the
persistent cache.

Cache writes are atomic and include the library version, processor, thread
count, model signature, parameters, objective, samples, dispersion, and
correctness result. Users can inspect the selected profile, force the portable
default, override validated parameters, or disable calibration without
disabling the optimized kernel.

Runtime integration
-------------------

Registered kernels use the session-owned executor and effective thread count
from the :doc:`Runtime Execution Controls Roadmap
<2026_08_runtime_execution_controls>`. Standalone C++ entry points are serial,
so they cannot introduce a competing pool.

The runtime API must support a plan-selected participant cap and preallocated
per-participant workspace. Hybrid processors require topology-aware worker
selection; calibration results from one P/E-core policy are not reused under
another policy.

PR07 advanced candidates
------------------------

The prepared plan now exposes bounded candidates for compact 32-bit AoS, split
SoA, hit-rate-guided preorder, stump and symmetric traversal, four-row
interleaving, sparse targets, prefetch distances, and prepared float16 splits.
The ONNX Runtime-style evaluator remains the portable default. Advanced
policies are captured only after calibration proves an end-to-end win on every
required repeat and passes typed correctness, shifted-input correctness and
latency, declared workspace, prepared-storage, and observed peak-memory gates.
Later calibration stages compose with earlier winners instead of resetting
them.

Float16 traversal prepares exactly rounded binary16 split values and passes the
complete v5 corpus, including rounding boundaries. It remains an internal
correctness and regression result and is not included in ONNX Runtime parity
claims because ONNX Runtime has no equivalent version-5 CPU kernel.

Final parity gate
-----------------

``tools/benchmark_tree_ensemble_parity.py`` is the reproducible final-gate
driver. Its priority corpus covers regression and complete classification
graphs, float32 and float64, scalar and batched execution, shallow and deep
trees, small and large forests, homogeneous and mixed branches, membership,
multiple targets, score transforms, and integer and string labels. Run it on
each dedicated machine with pinned affinity for the single-thread and
physical-core policies::

    python tools/benchmark_tree_ensemble_parity.py \
        --threads 1 --cpus 2 --baseline trees_pr07_threads1.json \
        --output trees_pr08_threads1.json --enforce
    python tools/benchmark_tree_ensemble_parity.py \
        --threads 16 --cpus 2-17 --baseline trees_pr07_physical.json \
        --output trees_pr08_physical.json --enforce

The JSON report keeps environment metadata, session preparation samples,
repeated inference samples, and the implementation-derived workspace upper
bound in separate sections. The companion Markdown file contains the
side-by-side latency table. ``--enforce`` succeeds only when regression and
classification pass the per-type median and minimum parity thresholds, the
supplied PR07 single-row baseline stays within 10%, correctness passes, and
the configurable preparation and workspace budgets pass. Raw JSON and
Markdown outputs from every dedicated machine are the evidence attached to
the pull request; aggregate summaries alone are insufficient.

Remaining pull-request sequence
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 25 43 12 10

   * - PR
     - Scope
     - Merge criterion
     - Depends on
     - Status
   * - Trees PR01
     - Opset-5 corpus and scalar reference.
     - Latest ``TreeEnsemble``, classifier, and regressor schemas have
       differential generators covering all modes, aggregates, transforms,
       types, invalid structures, and classification composition. No older
       opset is registered.
     - None
     - `Implemented in #301
       <https://github.com/xadupre/onnx-light-cpu/pull/301>`_
   * - Trees PR02
     - Canonical parser and immutable plan.
     - All three v5 schemas lower into one validated representation. Repeated
       execution performs no parsing, validation, allocation, lock, or string
       dispatch in traversal.
     - PR01
     - `Implemented in #305
       <https://github.com/xadupre/onnx-light-cpu/pull/305>`_
   * - Trees PR03
     - ONNX Runtime-compatible serial baseline.
     - The compact AoS/pointer layout, false-child fall-through, homogeneous
       mode, no-missing, one/multi-target, and classifier specializations pass
       the scalar corpus. Prepared membership, aggregation, and transforms
       improve or retain single-thread latency against ONNX Runtime.
     - PR02
     - `Implemented in #309
       <https://github.com/xadupre/onnx-light-cpu/pull/309>`_
   * - Trees PR04
     - ONNX Runtime-compatible scheduling baseline.
     - Row-parallel, tree-parallel, and 128-row tree-major batching reproduce
       the reference scheduling structure and its 80-tree/50-row crossovers
       through one dynamic decision table. Bounded workspace and the session
       executor introduce no oversubscription or nondeterministic label result.
     - PR03; Runtime Controls PR02
     - `Implemented in #311
       <https://github.com/xadupre/onnx-light-cpu/pull/311>`_
   * - Trees PR05
     - Dynamic tuning policy and structural signatures.
     - Exact keys, portable buckets, typed ordered regions, model digests,
       per-region workspace validation, and cache lifecycle tests cover all
       row-count crossovers without placing dynamic ``N`` in the profile key
       or accessing the registry in the hot path.
     - PR03, PR04
     - `Implemented in #313
       <https://github.com/xadupre/onnx-light-cpu/pull/313>`_
   * - Trees PR06
     - Calibration and inspection APIs.
     - Hierarchical search rejects incorrect candidates, respects time and
       memory budgets, persists raw evidence atomically, and improves or
       retains every priority profile. Selection and overrides are inspectable
       through the existing tuning APIs.
     - PR05
     - `Implemented in #318
       <https://github.com/xadupre/onnx-light-cpu/pull/318>`_
   * - Trees PR07
     - Advanced layouts, traversal, and float16.
     - Index AoS, SoA, hit-rate layout, stump, symmetric-tree,
       interleaved-row, sparse-target, prefetch, and optimized float16
       candidates land only where measured wins satisfy correctness,
       distribution-shift, and memory gates. General trees retain the
       ONNX Runtime-style portable evaluator.
     - PR05, PR06
     - `Implemented in #319
       <https://github.com/xadupre/onnx-light-cpu/pull/319>`_
   * - Trees PR08
     - Final parity gate.
     - Float32/float64 median end-to-end performance is at least ``1.0x`` ONNX
       Runtime, no priority case is below ``0.9x``, single-row tuned latency
       does not regress by more than 10%, all v5 types pass correctness, and
       preparation/workspace budgets pass.
     - PR01 through PR07
     - `Implemented in #320
       <https://github.com/xadupre/onnx-light-cpu/pull/320>`_

The TreeEnsemble roadmap is complete.
