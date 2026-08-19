Tree Ensemble Classification and Regression Roadmap
===================================================

:Date: 2026-08

**discussion**

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
validation, allocation, string lookup, tuning-cache lookup, or lock
acquisition.

The plan records:

* canonical tree roots, node and leaf counts, maximum and average depth;
* input, split, accumulation, and output types;
* feature, target, and class-label metadata;
* aggregate and post-transform functions selected as typed function pointers;
* branch-mode and missing-value distributions;
* the prepared node, leaf, and membership layouts;
* the selected execution strategy, row/tree chunks, batch size, and maximum
  participants;
* preallocated workspace requirements and alignment;
* an exact model signature and tuning ABI.

Node layout
~~~~~~~~~~~

The baseline uses pointer-free indices and owns all storage in the plan.
Candidate layouts include:

``compact_aos``
    One compact node record containing a 32-bit feature id, 32-bit child or
    leaf indices, split value, mode, and missing-direction flag. This favors
    one-row pointer chasing.

``split_soa``
    Separate aligned arrays for feature ids, splits, children, modes, and
    missing flags. This favors batched or interleaved traversal and avoids
    loading unused fields.

``preorder_hot``
    Trees are reordered into depth-first layout, with ``nodes_hitrates`` used
    only as a layout hint. The likely branch becomes fall-through where this
    improves locality; true/false semantics remain explicit.

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

The representation threshold is tunable, but the values and floating-point
equality semantics are not. NaN remains the set delimiter and is never a
member value.

Execution strategies
--------------------

No single traversal order is optimal for every combination of rows, trees,
depth, and targets. The plan chooses among bounded strategies:

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
    diverge. This is a candidate, not the portable default.

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

Tuning architecture
-------------------

Tuning has two levels:

Static structural selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model preparation derives facts that do not need timing:

* whether compact 32-bit indices are safe;
* whether a tree is a stump, symmetric, or general;
* membership representation candidates;
* whether dense or sparse target accumulation is legal;
* workspace bounds for each strategy.

Invalid or dominated candidates are removed before calibration. Static
selection never depends on the input values used for one inference.

Measured scheduling selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibration times the remaining legal candidates using deterministic inputs
that exercise representative paths. Version 1 of the tuning schema contains:

``execution.strategy``
    ``row_parallel``, ``tree_parallel``, ``tree_major_batch``, or
    ``interleaved_rows`` when supported by the model.

``execution.batch_rows``
    Number of rows retained in a tree-major or interleaved batch.

``parallel.minimum_rows``
    Batch threshold below which worker dispatch is disabled.

``parallel.minimum_trees``
    Forest-size threshold below which tree-parallel execution is disabled.

``parallel.maximum_threads``
    Maximum useful participants for the model profile.

``parallel.row_chunk``
    Contiguous rows assigned per row-parallel task.

``parallel.tree_chunk``
    Contiguous trees assigned per tree-parallel task.

``membership.linear_limit``
    Largest membership set evaluated with a linear scan.

``membership.bitset_range_limit``
    Largest compact non-negative integer range represented as a bitset.

``traversal.prefetch_distance``
    Optional node-prefetch distance; zero disables prefetch.

The tuning ABI is versioned. Parameters are strongly typed, range-checked,
cross-validated against workspace limits, and captured immutably by the
prepared plan.

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
    tuning_abi       = 1

The canonical digest covers tree topology, feature ids, modes, missing flags,
target ids, value types, and the structural buckets used by scheduling. Raw
split and leaf values may be excluded only if two models are guaranteed to
share all legal execution choices; otherwise they remain in the digest.

A portable default profile is indexed by structural buckets: tree-count
range, depth range, row-count range, target-count range, branch-mode mix, and
membership density. Exact profiles override portable defaults. A profile for
one forest must never silently apply to an incompatible topology or workspace
requirement.

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

#. benchmark legal serial layouts and membership representations;
#. compare traversal strategies at representative ``(rows, trees, targets)``
   points;
#. sweep ``batch_rows`` over bounded powers of two that fit the cache and
   workspace budget;
#. locate row/tree parallel crossovers by exponential search followed by
   refinement;
#. sweep participant caps, then row/tree chunk sizes;
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

Cache writes are atomic and include library version, tuning ABI, processor,
thread count, model signature, parameters, objective, samples, dispersion,
and correctness result. Users can inspect the selected profile, force the
portable default, override validated parameters, or disable calibration
without disabling the optimized kernel.

Runtime integration
-------------------

Registered kernels use the session-owned executor and effective thread count
from the :doc:`Runtime Execution Controls Roadmap
<2026_08_runtime_execution_controls>`. Standalone C++ entry points may use the
onnx-light-cpu pool, but pools are never nested.

The runtime API must support a plan-selected participant cap and preallocated
per-participant workspace. Hybrid processors require topology-aware worker
selection; calibration results from one P/E-core policy are not reused under
another policy.

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
     - Pending
   * - Trees PR02
     - Canonical parser and immutable plan.
     - All three v5 schemas lower into one validated representation. Repeated
       execution performs no parsing, validation, allocation, lock, or string
       dispatch in traversal.
     - PR01
     - Pending
   * - Trees PR03
     - Compact layouts and serial evaluators.
     - AoS/SoA candidates, prepared membership structures, typed aggregation,
       and post transforms pass the scalar corpus. The portable default
       improves or retains single-thread latency against ONNX Runtime.
     - PR02
     - Pending
   * - Trees PR04
     - Parallel traversal strategies.
     - Row-parallel, tree-parallel, and tree-major batching use bounded
       workspace and the session executor. Every corpus shape has a safe
       default with no oversubscription or nondeterministic label result.
     - PR03; Runtime Controls PR02
     - Pending
   * - Trees PR05
     - Tuning schema and structural signatures.
     - Exact keys, portable buckets, typed validation, immutable configuration,
       model digests, workspace checks, and cache lifecycle tests cover every
       strategy and parameter without hot-path registry access.
     - PR03, PR04
     - Pending
   * - Trees PR06
     - Calibration and inspection APIs.
     - Hierarchical search rejects incorrect candidates, respects time and
       memory budgets, persists raw evidence atomically, and improves or
       retains every priority profile. Selection and overrides are inspectable
       through the existing tuning APIs.
     - PR05
     - Pending
   * - Trees PR07
     - Specialized traversal.
     - Stump, symmetric-tree, interleaved-row, sparse-target, and prefetch
       candidates land only where measured wins satisfy correctness and memory
       gates. General trees retain the portable evaluator.
     - PR05, PR06
     - Pending
   * - Trees PR08
     - Final parity gate.
     - Median end-to-end performance is at least ``1.0x`` ONNX Runtime, no
       priority case is below ``0.9x``, single-row tuned latency does not
       regress by more than 10%, and preparation/workspace budgets pass.
     - PR01 through PR07
     - Pending

Trees PR08 remains open until regression and classification both satisfy the
final gate.
