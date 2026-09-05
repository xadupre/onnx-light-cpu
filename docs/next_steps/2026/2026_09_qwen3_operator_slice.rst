Qwen3 Non-MatMulNBits Operator Slice
====================================

:Date: 2026-09
:Updated: 2026-09-05

**planned** (prerequisite compute kernels are partially implemented)

Objective
---------

The objective is to implement every non-projection operator required by the
audited Qwen3 INT4 artifact in
:doc:`Qwen3 CPU Inference Critical Path <2026_08_qwen3_inference>`. This plan
separates the small graph-execution and transformer-block slice from the much
larger ``com.microsoft::MatMulNBits`` project so both can progress and be
reviewed independently.

The :doc:`missing-kernel implementation plan <2026_09_qwen3_missing_kernels>`
breaks Operator PR03 and PR04 into smaller input, layout, and normalization
PRs and coordinates them with the independent INT4 projection work.

The exit condition is stronger than registering kernels by name: one complete
Qwen decoder block must execute through native ``onnx-light`` and
``onnx-light-cpu`` paths, match ONNX Runtime, preserve view and optional-output
semantics, and report no untracked fallback for any operator covered here.

Scope
-----

This plan covers the missing execution surfaces and model-level coverage for
the operators and shared primitives already registered:

.. list-table::
   :header-rows: 1
   :widths: 29 8 20 43

   * - Operator
     - Nodes
     - Primary owner
     - Required Qwen contract
   * - ``ai.onnx::Constant``
     - 6
     - ``onnx-light`` core
     - Materialize INT64 shape and axes values during preparation, not as a
       timed CPU kernel.
   * - ``ai.onnx::Shape``
     - 1
     - ``onnx-light`` core
     - Produce the INT64 shape of the rank-two attention mask without reading
       tensor payload bytes.
   * - ``ai.onnx::Reshape``
     - 112
     - ``onnx-light`` core
     - Implement opset-26 ``0`` and ``-1`` semantics and return a metadata view
       whenever the contiguous input can be safely aliased.
   * - ``ai.onnx::Cast``
     - 2
     - ``onnx-light-cpu``
     - Convert INT64 sequence lengths to INT32 with ONNX Cast semantics;
       validate legal sequence lengths at the ``GroupQueryAttention`` boundary.
   * - ``ai.onnx::Gather``
     - 2
     - ``onnx-light-cpu``
     - Support FP32 embedding lookup with INT64 token ids on axis 0 and the
       INT64 scalar gather used by the mask prelude.
   * - ``ai.onnx::ReduceSum``
     - 1
     - ``onnx-light-cpu``
     - Reduce INT64 mask data using an axes tensor input and ``keepdims=0``.
   * - ``ai.onnx::Split``
     - 28
     - both repositories
     - Split FP32 QKV projections on the last axis with the input split tensor
       ``[2048,1024,1024]``; use non-overlapping views when runtime ownership
       permits and a materialized fallback otherwise.
   * - ``ai.onnx::Sigmoid``
     - 28
     - ``onnx-light-cpu``
     - Reuse the registered FP32 activation for ``x * sigmoid(x)`` and retain
       model-level coverage without creating a private approximation.
   * - ``ai.onnx::SimplifiedLayerNormalization``
     - 57
     - ``onnx-light-cpu``
     - Validate ``axis=-1``, ``epsilon=1e-6`` and ``stash_type=1``, then lower
       to the shared RMS-normalization engine with proven accumulation and
       output semantics.
   * - ``com.microsoft::SkipSimplifiedLayerNormalization`` v1
     - 56
     - ``onnx-light-cpu``
     - Lower residual addition plus RMS-style normalization to shared plans,
       including sparse optional outputs; fuse the traversal only after
       measurements justify it.
   * - ``ai.onnx::Mul`` / ``ai.onnx::Sub``
     - 56 / 1
     - ``onnx-light-cpu``
     - Add opset-26 model coverage for FP32 multiplication and INT64
       subtraction; no new kernel is expected.

Explicit exclusions
-------------------

``com.microsoft::MatMulNBits`` is not part of this roadmap. Its packed-weight
validation, preparation, decode GEMV, small-M and prefill GEMM work remains
Qwen PR02--PR03 in the parent Qwen3 roadmap.

``com.microsoft::GroupQueryAttention`` is also not new work here: `#494
<https://github.com/xadupre/onnx-light-cpu/pull/494>`_ and `#507
<https://github.com/xadupre/onnx-light-cpu/pull/507>`_ delivered its CPU
adapter, Qwen contract and differential coverage. Likewise, `#498
<https://github.com/xadupre/onnx-light-cpu/pull/498>`_ and `#579
<https://github.com/xadupre/onnx-light-cpu/pull/579>`_ provide the shared
normalization kernels this plan must reuse. This roadmap does not introduce
another attention or normalization engine.

The audited native graph does not require separate ``RotaryEmbedding``,
``Softmax``, ``Concat`` or ``Slice`` kernels because GQA owns those operations.
The portable standard-ONNX graph retains them as differential coverage, but
their broad operator roadmaps do not block this plan.

Current implementation status
-----------------------------

The shared compute foundations are further along than the five-step sequence
alone suggests:

* ``GroupQueryAttention`` and its fused rotary path are delivered by #494 and
  #507;
* the shared ``RMSNormalization`` engine is delivered and optimized by #498
  and #579;
* ``Sigmoid`` is registered and subsequently optimized by #585, #600, and
  #604;
* ``Mul`` and ``Sub`` already use the registered binary engine and need
  Qwen-shaped model coverage rather than new kernels.

The operator slice itself is not complete. The deterministic block fixture,
prepared metadata/view ownership, the audited ``Cast``/``Gather``/
``ReduceSum``/``Split`` execution contracts, and both normalization
compatibility adapters remain pending. References to ``Gather``, ``Cast``,
``Expand``, ``Reshape``, ``Transpose`` or ``Where`` in graph construction and
gradient code do not establish native execution coverage for this roadmap.

Ownership and architecture
--------------------------

Metadata-only work belongs in ``onnx-light``. Audit existing core support
before adding implementations. Constants are prepared once; dynamic ``Shape``
values use the current invocation's dimensions without reading payload bytes.
``Reshape`` must modify tensor metadata rather than copy a contiguous payload.
Their plans must preserve
storage ownership, byte offsets, alignment, liveness and safe output release.
A view may never outlive its backing allocation.

Typed computation belongs in ``onnx-light-cpu``. ``Cast``, ``Gather``,
``ReduceSum`` and ``Sigmoid`` register ordinary CPU kernels with explicit
opset and type coverage. They use the session executor only when measured work
justifies parallel dispatch and do not create private thread pools.

``Split`` crosses the boundary: core runtime ownership determines whether
outputs may alias disjoint ranges, while the CPU implementation provides the
materialized fallback. The selected path must be observable through the
execution diagnostics. Last-axis splits across multiple rows require strided
views; use the materialized fallback if the runtime or consumers cannot
represent those views.

Persistent KV-cache lifecycle belongs to ``onnx-light`` and is outside this
kernel slice. The first complete block uses GQA's existing tensor-cache
interface; native CPU append and cache-aware Attention follow the upstream
persistent-state contract separately.

Experimental and Microsoft-domain normalization nodes are compatibility
adapters, not independent numerical implementations. They validate their
source schemas and construct the same internal RMS-normalization and Add plans
used by standard operators. Sparse optional outputs remain sparse; an omitted
output must not allocate a hidden tensor.

Implementation sequence
-----------------------

Operator PR01: executable fixture and ownership
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Freeze a deterministic, self-contained one-block fixture derived from the
audited graph's dimensions, types and attributes. It does not load the
external audited artifact or depend on its weights. The fixture uses small
deterministic payloads and ordinary FP32 ``MatMul`` projections so this
roadmap can complete without ``MatMulNBits``.

Add an inventory assertion that classifies every node as registered CPU,
prepared core, tested compatibility adapter, or the explicitly excluded
``MatMulNBits``. Unknown fallback and silent reference execution fail the
test. Record which repository owns each missing surface before adding kernels.

Operator PR02: metadata and views
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reuse or complete prepared ``Constant`` and invocation-aware ``Shape`` values,
followed by checked
``Reshape`` semantics. Cover scalar and empty shapes, one inferred ``-1``
dimension, ``allowzero`` behavior, invalid element counts and integer
overflow. Add runtime tests proving that legal contiguous reshapes share
storage and that release order cannot invalidate a live view.

Add the shared ownership contract required for view-producing ``Split``.
This PR need not implement every Split type, but it must make legal Qwen views
representable without special ownership outside the runtime.

Operator PR03: input and layout kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement the exact ``Cast``, ``Gather``, ``ReduceSum`` and ``Split`` contracts
listed above. Validate axes, negative indices, split totals, output shapes
and empty dimensions before entering hot
loops. ``Gather`` must not expand or duplicate the embedding initializer.
Integer conversion must follow ONNX Cast semantics rather than introducing
Qwen-specific rejection rules into the generic kernel.

Each kernel first receives direct tests and ONNX Runtime differential cases,
then executes in the one-block fixture. Optimized paths retain an independent
portable or scalar oracle.

Operator PR04: activation and normalization adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reuse the registered FP32 ``Sigmoid`` implementation and extend its coverage
to the audited Qwen shapes. Measure the separate ``Sigmoid`` and ``Mul``
traversals before considering a SiLU or gated MLP fusion. Reuse the existing
registered ``SwiGLU`` kernel when its exact two-input contract matches; do not
add a duplicate gated activation.

Implement ``SimplifiedLayerNormalization`` and
``SkipSimplifiedLayerNormalization`` as schema adapters over the shared
normalization and Add plans. Differential tests cover epsilon, accumulation,
residual output, omitted outputs, aliasing and malformed inputs. A fused skip
kernel is optional and must not replace the compositional correctness path.

Operator PR05: complete block and performance gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute the deterministic FP32-``MatMul`` block and compare every observable
intermediate and output with ONNX Runtime. Record per-node timings for the ten
surfaces in this roadmap so projection latency is not attributed to the
non-projection slice.

Executing the audited native graph is a parent-roadmap integration follow-up
after ``MatMulNBits`` becomes available. It must require no contract change in
the operators delivered here, but it is not a completion criterion for this
independent plan.

Profile node latency, allocations, copied bytes and peak memory. Optimize only
measured bottlenecks. Metadata operators must disappear from timed compute;
``Reshape`` and eligible ``Split`` nodes must report zero payload bytes copied.

Correctness matrix
------------------

Focused tests must cover:

* exact audited opsets, attributes, types, shapes and optional outputs;
* scalar, empty, singleton, dynamic and malformed shape cases;
* negative Gather indices and out-of-range rejection;
* ONNX INT64-to-INT32 Cast boundary semantics and GQA length validation;
* ReduceSum axes supplied as a tensor, with ``keepdims=0``;
* Reshape ``0``/``-1`` rules, byte-size overflow and view lifetime;
* equal and uneven Split sizes, last-axis views and materialized fallback;
* Sigmoid non-finite values, SIMD boundaries and serial/parallel execution;
* normalization epsilon, FP32 accumulation, residual output and sparse
  optional outputs;
* registered ``Mul`` and ``Sub`` behavior at opset 26;
* direct ONNX Runtime comparisons for each operator and for the complete
  deterministic block.

Performance and memory gates
----------------------------

Correctness comes before fusion. Shared CI runs deterministic differential
tests and records benchmark data; median latency acceptance is evaluated on an
idle pinned host with identical inputs, thread count, affinity, warm-up,
allocator policy and build configuration.

Priority compute cases should reach at least ``0.9x`` ONNX Runtime median
performance, with the complete non-projection block targeting ``1.0x``.
Any exception must identify the dominant measured cost. Tail latency and
allocation counts are reported alongside the median.

The following structural gates are mandatory:

* prepared ``Constant`` and dynamic ``Shape`` perform no payload computation;
* contiguous ``Reshape`` copies zero payload bytes;
* eligible ``Split`` outputs share storage without violating ownership;
* embedding ``Gather`` reads only selected rows;
* omitted normalization outputs allocate no buffers;
* no operator creates a private worker pool or materializes a full attention
  score, expanded embedding table, or normalization decomposition.

PR sequence
-----------

.. list-table::
   :header-rows: 1
   :widths: 12 16 34 25 13

   * - PR
     - Repository
     - Deliverable
     - Depends on
     - Status
   * - Operator PR01
     - ``onnx-light-cpu``
     - Deterministic block fixture, ownership map and no-fallback inventory
       gate.
     - Existing Qwen graph inventory
     - Pending
   * - Operator PR02
     - ``onnx-light``
     - Prepared Constant/Shape, zero-copy Reshape and shared view ownership.
     - Operator PR01
     - Pending
   * - Operator PR03
     - ``onnx-light-cpu``
     - Cast, Gather, ReduceSum and Split execution.
     - Operator PR02
     - Pending
   * - Operator PR04
     - ``onnx-light-cpu``
     - Qwen Sigmoid integration and both normalization compatibility adapters.
     - Operator PR03; completed ExpLog PR01--PR03, #498 and #579
     - Partially implemented: Sigmoid and the RMS engine are delivered; both
       compatibility adapters remain
   * - Operator PR05
     - ``onnx-light-cpu``
     - Differential complete-block and performance/memory gates.
     - Operator PR01--PR04
     - Pending

Completion criteria
-------------------

This roadmap is complete when the deterministic Qwen block and every
non-``MatMulNBits`` node in the audited graph have explicit execution
ownership, native registration or prepared-core handling, focused failure
tests and ONNX Runtime differential coverage. The block must satisfy the
structural memory gates above and execute without an untracked fallback.

The parent Qwen3 roadmap may then treat this operator slice as a completed
dependency. End-to-end INT4 inference remains blocked only by work explicitly
owned by the separate packed-projection and persistent-cache plans.
