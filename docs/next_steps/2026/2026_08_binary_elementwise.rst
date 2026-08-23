Binary Elementwise and Broadcasting Performance Roadmap
=======================================================

:Date: 2026-08

**planned**

Objective
---------

The objective is to implement one common CPU traversal and dispatch engine for
the numeric and boolean ONNX binary elementwise operators listed below. The
priority corpus must reach at least ``1.0x`` ONNX Runtime median performance,
with no priority case below ``0.9x``. The engine should exceed ONNX Runtime
where a prepared broadcast plan, a shape-specific loop, or fusion across
several elementwise operators removes indexing and memory traffic.

The correctness implementation in ``onnx-light`` already provides
``BroadcastShape``/``BroadcastInfo`` validation and equal-shape, scalar, and
generic rank-aware traversal through ``BinaryElementwise``. The roadmap builds
``BinaryBroadcastPlan`` and optimized loop selection on top of that baseline;
it does not recreate broadcast validation. The generic loop computes both
input offsets from every dimension for every output element. That remains an
appropriate fallback, but not an optimized implementation for large tensors or
repeated inference.

Scope
-----

This roadmap covers every operator whose output elements independently combine
the corresponding broadcasted elements of two inputs:

* arithmetic: ``Add``, ``Sub``, ``Mul``, ``Div``, ``Mod`` and ``Pow``;
* comparisons: ``Equal``, ``Greater``, ``GreaterOrEqual``, ``Less`` and
  ``LessOrEqual``;
* logical: ``And``, ``Or`` and ``Xor``;
* integer bits: ``BitwiseAnd``, ``BitwiseOr``, ``BitwiseXor`` and ``BitShift``;
* parameterized: ``PRelu``.

Schemas with two required inputs but cross-element or metadata behavior do not
use this engine. This explicitly excludes matrix multiplication, gather and
indexing, reshape/expand/tile, normalization, cumulative operations,
grid/geometry operators, sequence access, top-k, string concatenation, and
other operators where one output element is not a scalar function of two
broadcasted input elements.

``CastLike`` is a unary conversion selected by metadata from its second input;
it stays in the unary roadmap and never enters ``BinaryBroadcastPlan``.
``SwiGLU`` v28 has two equal-shape inputs but does not allow broadcasting. It
reuses the equal-shape traversal and the completed ``Exp`` primitive through a
specialized adapter, without weakening its schema to accept broadcasting.
Variadic ``Sum``, ``Mean``, ``Min`` and ``Max`` use a later
``VariadicElementwisePlan`` that iterates every input in one output traversal;
they are not implemented as a chain of binary plans and temporary tensors.

Schema and type contract
------------------------

The source of truth is the latest schema registered by ``onnx-light`` for each
operator plus its existing portable kernel dispatch. Binary PR01 generates a
checked operator/opset/input-type/output-type manifest from those registries;
``onnx-light-cpu`` does not advertise a wider type set. In particular:

* arithmetic inputs and outputs have the same type except for the schema's
  mixed base/exponent forms of ``Pow``;
* comparisons take equal-typed inputs and emit one-byte ``BOOL``;
* logical operations accept and emit one-byte ``BOOL``;
* bitwise and shift operations use the schema-supported signed/unsigned
  integer widths;
* string ``Equal`` and types without a useful CPU SIMD representation remain
  explicit portable fallbacks and are correctness-tested but not registered
  as optimized kernels;
* FP16/BF16 are optimized only where the schema and portable implementation
  already support them; this roadmap does not add blanket Float8 or packed
  sub-byte arithmetic.

Each adapter owns schema semantics that cannot live in a generic functor:
``Mod.fmod``, ``BitShift.direction``, ``Pow`` mixed types and output type,
``PRelu`` slope constraints, and the exact supported opset interval.
Integer ``Div`` truncates toward zero. Integer ``Mod`` implements Python
remainder for ``fmod=0`` and truncated C remainder for ``fmod=1``; floating
``Mod(fmod=1)`` follows ``std::fmod`` and its IEEE special cases. Integer zero
divisors, signed ``INT_MIN / -1`` and shift amounts at least the element width
are invalid corpus inputs and are rejected before the optimized loop.
``Add``/``Sub``/``Mul`` overflow is implemented with unsigned-width arithmetic
and bit-preserving conversion instead of signed C++ overflow. Integer ``Pow``
cases are restricted to representable results until a separately tested
overflow contract exists. Differential tests cover both accepted boundaries
and these validation failures; undefined behavior never reaches a native
instruction.

The common engine owns shape normalization, loop selection, iteration,
parallel scheduling, and ISA dispatch. Each ONNX adapter retains its own type
constraints and numerical semantics. In particular, integer ``Div`` and
``Mod``, mixed-type ``Pow``, comparisons returning ``BOOL``, NaN propagation,
and signed zero behavior must not be hidden behind an unsafe generic
approximation.

Broadcast plan
--------------

An immutable ``BinaryKernelDescriptor`` is created with the node. It contains
the operator, opset, attributes, validated type signature, and scalar/ISA
functions that the current CPU may use. It contains no invocation tensors,
shape-specific offsets, thread count, or mutable tuning state.

Each invocation validates both concrete shapes through the existing
``BroadcastShape``/``BroadcastInfo`` contract and obtains a
``BinaryBroadcastPlan`` keyed by descriptor identity, both shapes, input and
output types, and ISA profile. It contains:

* the right-aligned output shape;
* zero strides for broadcast dimensions and contiguous strides otherwise;
* coalesced dimensions;
* the selected loop family;
* contiguous inner-loop length and outer-loop count;
* the selected typed scalar and vector function pointers;
* safe output-alias candidates and the minimum task granularity;
* guards for dynamic input shapes.

Static shapes retain one plan. Dynamic nodes use a bounded eight-entry
least-recently-used cache per node; eviction never invalidates an executing
plan. The key never contains raw pointers. A miss builds and validates a new
plan, while a failed guard always falls back to rebuilding rather than using
stale strides.

The executor decision is separate and per invocation. It combines output
bytes, loop family, operation cost, outer-block count, processor profile,
session policy, and current nesting state to choose serial execution or a
number of submitted tasks. The plan exposes independent work but stores no
fixed thread count. The session executor alone controls admitted workers.

An input may back the output only when runtime last-use analysis permits it,
its shape and type equal the output, and the selected traversal cannot
overwrite a value before its final read. A broadcast input is never reused as
the larger output. Alias and non-alias execution use identical numerical
tests.

The plan should classify the operation once:

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * - Family
     - Example
     - Loop
   * - Contiguous
     - ``[N, C] op [N, C]``
     - One vector loop over both inputs and output.
   * - Left scalar
     - ``[] op [N, C]``
     - Broadcast one loaded scalar across SIMD vectors.
   * - Right scalar
     - ``[N, C] op []``
     - Separate from left scalar because ``Sub``, ``Div``, ``Pow`` and
       comparisons are not commutative.
   * - Repeated contiguous block
     - ``[N, C, H, W] op [C, 1, 1]``
     - Load one broadcast value per channel and process the contiguous HxW
       block.
   * - Inner vector broadcast
     - ``[N, C, H, W] op [1, C, 1, W]``
     - Reuse a contiguous vector from one input across outer dimensions.
   * - Outer broadcast
     - ``[N, 1, H, W] op [1, C, H, W]``
     - Iterate over coalesced outer blocks and vectorize the contiguous suffix.
   * - General strided
     - Alternating singleton dimensions
     - Odometer over coalesced dimensions, with offsets updated once per inner
       block rather than recomputed for every element.

Dimension coalescing
--------------------

Right-align both shapes and derive element strides, using zero for a broadcast
dimension. Adjacent output dimensions may be collapsed when each input is
either contiguous across their boundary or broadcast across the complete
collapsed region.

For example:

.. code-block:: text

   output shape: [2, 3, 4, 5]
   X strides:    [60, 20, 5, 1]
   Y strides:    [ 0,  0, 0, 1]

can become an outer count of 24 and a contiguous inner block of 5. The hot loop
then performs five-element vector/tail operations per outer position without a
four-dimensional index calculation for every element.

Coalescing should also remove dimensions of length one. Empty dimensions must
produce zero work without reading either input.

Loop architecture
-----------------

The engine should separate traversal from calculation:

.. code-block:: text

   BinaryBroadcastPlan
       -> loop family
       -> typed/ISA compute function
       -> operator functor or operation identifier

For the common contiguous and scalar cases, instantiate direct functions such
as ``AddFloatAvx2`` rather than calling an erased functor per element. The
general fallback may use templates to retain one shared traversal
implementation without introducing an indirect call inside the inner loop.

The optimized loop nest is:

.. code-block:: text

   for outer_block in assigned_blocks:
       x = X + x_offset(outer_block)
       y = Y + y_offset(outer_block)
       z = Z + output_offset(outer_block)
       vector_kernel(x, y, z, inner_count, x_inner_stride, y_inner_stride)

Offsets should be advanced incrementally with a coalesced-dimension odometer.
Division/modulo by output dimensions must not appear in the per-element hot
loop.

SIMD kernels
------------

Provide portable scalar, AVX2, AVX-512, and ARM NEON/SVE implementations where
supported by the project:

* vector loads for contiguous inputs and register broadcasts for zero strides;
* unrolled vector loops with independent accumulators only where operation
  latency requires it, particularly ``Div`` and ``Pow``;
* masked AVX-512 tails or one shared scalar tail for AVX2/NEON;
* vectorized FP16/BF16 conversion, calculation, and narrowing when native
  arithmetic is unavailable;
* native FP16/BF16 arithmetic when it is faster and preserves the operator
  contract;
* byte/word/dword/qword integer kernels where the ISA supports the exact
  arithmetic, comparison, logical, and bitwise semantics;
* byte-valued ONNX ``BOOL`` output without bit packing;
* scalar fallback for operations without an efficient SIMD instruction, such
  as general integer division and some ``Pow`` combinations.

Approximate reciprocal, reciprocal-square-root, or transcendental
implementations require a separately documented error contract. They must not
silently replace exact ONNX behavior in the default kernels.

Data types and semantics
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 32 46

   * - Type family
     - Preferred compute
     - Important constraint
   * - FP32/FP64
     - Native vector arithmetic
     - Preserve NaN, infinity, signed zero, and comparison semantics.
   * - FP16/BF16
     - Native SIMD or convert vectors to FP32
     - Do not decode and encode one scalar at a time when vector conversion is
       available.
   * - Signed/unsigned integers
     - Native vector arithmetic where available
     - Match ONNX overflow, division, modulo, and shift rules; avoid C++ signed
       overflow assumptions in optimized code.
   * - ``BOOL``
     - Byte vectors
     - Inputs and outputs use the runtime's byte representation.
   * - Mixed input/output
     - Typed vector adapters
     - Comparisons return ``BOOL``; ``Pow`` may accept different base and
       exponent types according to its opset.

Parallel scheduling
-------------------

Binary elementwise kernels are normally bandwidth-bound. Parallel execution
should therefore be conservative. Registered kernels use the existing
onnx-light session executor and processor-aware tuning foundation; this
roadmap adds plan-selected work partitioning and limits, not a private
scheduler:

* remain single-threaded below a measured byte-count threshold;
* split contiguous output ranges into cache-line-aligned chunks;
* split only at inner-block boundaries for broadcast patterns;
* cap threads when aggregate memory bandwidth is saturated;
* avoid assigning adjacent partial cache lines to different threads;
* keep scalar or small broadcast operands hot in shared cache.

Thresholds should depend on element size, operation cost, loop family, and
hardware. ``Pow`` may benefit from threads much earlier than ``Add`` because it
is compute-bound.

Binary PR03 measures byte thresholds from ``0``, ``4 KiB``, ``16 KiB``,
``64 KiB``, ``256 KiB`` and ``1 MiB`` and participant caps from ``1``, ``2``,
``4`` and physical cores for every priority operation/loop-family/type group.
It selects the smallest threshold and smallest cap whose median is within 2%
of that group's best result and whose small-tensor p90 does not regress the
serial baseline by more than 2%. The selected values and raw samples are
stored respectively in the existing tuning registry and benchmark artifact.
Untested processors use the portable conservative profile and are marked for
later tuning rather than silently inheriting reference-machine results.

Prepared plans and dynamic shapes
---------------------------------

Static shapes create ``BinaryBroadcastPlan`` once per session. Dynamic shapes
use the bounded cache defined above. Processor-specific thresholds belong to
the tuning registry and processor profile, not to the cache key; changing a
session policy changes the per-invocation executor decision without rebuilding
shape traversal metadata.

Planning avoids repeated shape alignment and loop classification, but its
expected gain is small for large tensors. The primary benefit is allowing the
hot path to call the selected loop directly. For tiny tensors, graph-level
fusion matters more than either planning or SIMD.

Fusion
------

Binary kernels frequently appear in longer elementwise expressions:

.. code-block:: text

   residual = Add(x, projection)
   gate = Mul(Sigmoid(a), b)
   normalized = Mul(Sub(x, mean), inv_std)
   scores = Add(Mul(qk, scale), mask)

A later ``ElementwisePlan`` should fuse compatible unary, binary, comparison,
and selection operations into one traversal. It should:

* infer one common broadcast iteration space;
* retain intermediate values in SIMD registers;
* generate or select a bounded expression kernel;
* preserve ONNX evaluation order and type conversions;
* reject fusion when an intermediate has another consumer or when aliasing,
  numerical, or dynamic-shape constraints are unsafe.

The first implementation is deliberately bounded to two templates:

``swiglu_gate``
    ``Mul(Mul(gate, Sigmoid(gate)), up)`` with equal FP32/BF16 shapes
    ``[1,S,H]`` for ``S`` in ``{1,32,512}`` and ``H`` in ``{3072,9728}``.

``scaled_masked_scores``
    ``Add(Mul(scores, scalar_scale), additive_mask)`` with FP32 scores shaped
    ``[1,H,Q,K]`` for ``H`` in ``{16,32}``, ``(Q,K)`` in
    ``{(1,128),(32,1024),(128,4096)}``, and a broadcast mask shaped
    ``[1,1,Q,K]``.

The fused kernel preserves the unfused operation order and uses the same
primitive approximations. FP32 results satisfy the unfused ULP/error contract;
BF16 results satisfy the frozen converted-reference tolerance. Integer and
BOOL fusion are outside the first implementation. Unary roadmap work may add
new templates later, but Binary PR07 depends only on the completed Exp
primitive needed by ``Sigmoid``.

Fusion is the principal route to gains substantially above ONNX Runtime because
an isolated ``Add`` already approaches the memory-bandwidth limit.

Benchmark contract
------------------

Compare with ONNX Runtime using identical inputs, shapes, types, threads, and
affinity. Binary PR01 pins the reference CPU, OS, compiler, ONNX Runtime,
power mode, NUMA placement, and compact-affinity policy. Warm up both
implementations, alternate their order, retain raw samples, and report median,
p10/p90, and dispersion.

The complete correctness matrix includes:

* element counts ``0``, ``1``, ``7``, ``31``, ``32``, ``33``, ``255``,
  ``256``, ``257``, ``4,096``, ``65,536``, ``1,048,576`` and ``4,194,304``;
* left and right scalar broadcasting;
* row ``[M,N] op [N]``, column ``[M,N] op [M,1]``, NCHW
  ``[2,64,32,32] op [64,1,1]``, NHWC
  ``[2,32,32,64] op [64]``, and attention
  ``[2,16,128,128] op [1,16,1,1]`` broadcasting;
* leading-rank expansion and alternating singleton shapes
  ``[2,1,4,1,8,1] op [1,3,1,5,1,7]``;
* ranks 0 through at least 8;
* empty dimensions and SIMD tails;
* every type emitted by the generated schema/kernel manifest;
* attributes, operand order, special values and domain boundaries for every
  operator;
* single-thread and physical-core configurations;
* alias-permitted and forced-distinct output storage.

The required performance matrix is smaller and fixed: ``Add``, ``Sub``,
``Mul``, ``Div`` and ``PRelu`` for FP32/BF16; ``Equal`` and ``Less`` for FP32
and INT32; ``And`` for BOOL; ``BitwiseAnd`` for INT32; and ``Pow``/``Mod`` for
FP32. It covers contiguous, both scalar directions, row, per-channel, outer,
and general-strided families at 4,096, 65,536, 1,048,576 and 4,194,304 output
elements where the shape family permits, under one-thread and physical-core
policies. Remaining operator/type pairs are reported but do not decide the
isolated-kernel parity gate.

Report latency, output elements/second, selected loop family, ISA, submitted
tasks, admitted workers, and operation identity. Report two byte models:
``unique tensor bytes`` counts each logical input and output element once,
while ``expanded operand bytes`` counts one value from each input per output
element. Neither is labelled as actual DRAM traffic; hardware counters may
report measured traffic separately.

Backend test corpus
-------------------

Binary PR01 adds lazy ``onnx-light-cpu`` backend cases in correctness and
``TestMode::BENCHMARK`` modes for all 19 binary operators. They are registered
through the standard CPU backend collector so the backend API, benchmark
runner, and dashboard consume the same models. Names encode operator, opset,
input/output types, loop family, shape, operand order where relevant, and
thread policy.

Unit tests run bounded representatives through each registered CPU adapter.
Metadata tests inspect the complete corpus without allocating large tensors
and enforce unique names, lazy construction, exact element counts, the
generated schema/type manifest, every loop family, left/right non-commutative
cases, and opt-in 4,194,304-element cases. Raw benchmark output retains the
backend case name and selected plan diagnostics.

Completed foundations
---------------------

``onnx-light`` already supplies ``BroadcastShape``/``BroadcastInfo`` validation,
the correctness-first ``BinaryElementwise`` traversal, portable kernels, and
backend cases. ``onnx-light-cpu`` currently registers no binary elementwise
kernel. Binary PR01 starts after that baseline and concentrates on the checked
manifest, CPU backend corpus, reusable descriptor/plan preparation, dimension
coalescing, bounded caching, and loop classification.

The :doc:`Runtime Execution Controls Roadmap
<2026_08_runtime_execution_controls>` is complete through
`onnx-light-cpu #271
<https://github.com/xadupre/onnx-light-cpu/pull/271>`_ and
`#314 <https://github.com/xadupre/onnx-light-cpu/pull/314>`_. Registered
kernels execute through the session-owned executor, while the onnx-light
processor-aware tuning registry completed in
`onnx-light #4428 <https://github.com/xadupre/onnx-light/pull/4428>`_
provides immutable processor and effective-thread profiles. The remaining
binary PRs consume these foundations and add no private scheduler.

Remaining pull-request sequence
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 9 27 44 12 8

   * - PR
     - Scope
     - Merge criterion
     - Depends on
     - Status
   * - Binary PR01
     - Manifest, backend corpus, descriptor, prepared plan, and scalar loops.
     - The generated operator/opset/type manifest and lazy correctness/benchmark
       cases cover all 19 operators and loop families.
       ``BinaryKernelDescriptor`` is immutable; shape-guarded
       ``BinaryBroadcastPlan`` coalesces dimensions, uses a bounded dynamic
       cache, selects scalar loops, advances offsets per inner block, and
       proves safe alias decisions.
     - Existing onnx-light broadcast validation and traversal
     - Pending
   * - Binary PR02
     - FP32/FP64 arithmetic SIMD.
     - ``Add``, ``Sub``, ``Mul``, and ``Div`` provide contiguous, left/right
       scalar, SSE2/AVX2/AVX-512, NEON, and SVE/SVE2 kernels with exact operand
       order, special values, and tails.
     - PR01
     - Pending
   * - Binary PR03
     - Vector broadcast families and executor decisions.
     - Repeated block, inner-vector, outer, and general strided patterns use
       vector inner loops. Per-invocation processor-aware decisions submit
       independent tasks to the session executor, scale expensive operations,
       cap bandwidth-bound operations, and do not store a thread count or
       introduce another scheduler.
     - PR01, PR02; completed runtime foundation
     - Pending
   * - Binary PR04
     - Comparison, logical, bitwise, shift, and integer arithmetic.
     - Comparisons emit byte ``BOOL``; logical and bitwise kernels cover every
       supported width. Integer arithmetic, division, modulo, shifts and
       overflow boundaries are implemented without C++ undefined behavior;
       unsupported string ``Equal`` remains an explicit portable fallback.
     - PR03
     - Pending
   * - Binary PR05
     - Specialized arithmetic and low precision.
     - ``Pow``, ``Mod`` and ``PRelu`` preserve attributes, mixed types and
       numerical contracts through the common plan. Equal-shape ``SwiGLU``
       reuses traversal plus the completed Exp primitive without enabling
       broadcasting. Applicable FP16/BF16 paths use native arithmetic or
       vector conversion without a full-tensor conversion.
     - PR03, PR04; completed ExpLog foundation
     - Pending
   * - Binary PR06
     - Variadic ``Sum``/``Mean``/``Min``/``Max`` adapter.
     - ``VariadicElementwisePlan`` validates all inputs, derives one common
       iteration space, and computes each output in one traversal without
       pairwise temporary tensors. One-input, empty, broadcasting, floating
       evaluation order, integer overflow, and Min/Max NaN semantics match the
       portable kernels.
     - PR03-PR05
     - Pending
   * - Binary PR07
     - Shared ``ElementwisePlan`` fusion.
     - The ``swiglu_gate`` and ``scaled_masked_scores`` templates retain
       intermediates in registers, preserve the declared shapes, broadcasting,
       evaluation order, aliasing and graph lifetimes, reuse prepared traversal
       and the session executor, and each reaches at least 1.20x ONNX Runtime
       median with no case below 1.10x.
     - PR01 through PR06; completed ExpLog foundation
     - Pending
   * - Binary PR08
     - Final correctness and parity gate.
     - Every in-scope operator/type/broadcast case passes differential tests;
       median priority performance is at least 1.0x ONNX Runtime with no
       priority case below 0.9x. This PR remains open while any target fails.
     - PR01 through PR07
     - Pending

Binary PR08 is the final binary roadmap PR.

Expected gains
--------------

The estimates compare with either the current generic ``onnx-light`` loop or a
tuned ONNX Runtime kernel, as stated explicitly.

.. list-table::
   :header-rows: 1
   :widths: 22 25 35 18

   * - Optimization
     - Expected gain
     - Conditions
     - Estimated effort
   * - Dimension coalescing
     - **2-10x over the current generic loop**; **0-15% over ONNX Runtime**
     - Non-scalar broadcasting with rank-dependent per-element indexing.
     - 3-5 days.
   * - Contiguous/scalar SIMD
     - **2-8x over scalar code**; **0-10% over ONNX Runtime**
     - Enough elements to amortize dispatch; the memory-bandwidth ceiling
       limits ``Add`` and ``Mul``.
     - 5-8 days per ISA/type family.
   * - Pattern-specific broadcasting
     - **1.5-5x over the generic loop**; **5-25% over a generic ONNX Runtime
       broadcast path**
     - Common per-channel, per-head, row, or suffix pattern with a long
       contiguous inner block.
     - 5-10 days.
   * - Vector FP16/BF16 conversion
     - **2-6x over scalar decode/encode**; **0-15% over ONNX Runtime**
     - Hardware has vector conversion or native low-precision arithmetic.
     - 5-10 days per ISA family.
   * - Parallel scheduling
     - **1.5-4x over one thread** for large tensors; **0-10% over ONNX Runtime**
     - Operation has not already saturated memory bandwidth and tensor size
       amortizes thread-pool overhead.
     - 3-7 days.
   * - Prepared broadcast plan
     - **1-5%** for medium tensors; **5-20%** for repeatedly executed tiny
       kernels
     - Static or recurring shapes; no graph fusion.
     - 2-4 days after plan infrastructure.
   * - Elementwise fusion
     - **1.3-3x over separate kernels**; **10-50% over ONNX Runtime** when its
       selected execution path does not fuse the same expression
     - At least one intermediate read and write is eliminated; expression has
       enough work and compatible broadcasting.
     - 10-20 days for the first bounded fusion engine.

For isolated bandwidth-bound arithmetic, a sustainable advantage above 10% is
unlikely on every shape. Large gains should be expected from pathological
generic broadcasting, low-precision conversion that is currently scalar, or
fusion that removes complete tensor passes.

Acceptance criteria
-------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Area
     - Exit criterion
   * - Correctness
     - Differential tests pass for the generated schema/type manifest, operand
       order, attributes, multidirectional broadcasting, ranks, empty
       dimensions, tails, NaN, infinity, signed zero, defined integer edge
       cases, mixed ``Pow`` types, and BOOL outputs.
   * - Common architecture
     - Arithmetic, comparison, logical, and bitwise adapters reuse
       ``BinaryBroadcastPlan`` and its traversal without duplicating broadcast
       loops.
   * - Performance parity
     - The fixed priority performance matrix reaches at least 1.0x ONNX Runtime
       median with no priority isolated-kernel case below 0.9x.
   * - Robustness
     - Every plan is shape-guarded and has a portable correctness fallback.
   * - Scaling
     - Large kernels scale to the bandwidth or compute limit without
       regressing small single-thread workloads.
   * - Exceeding ONNX Runtime
     - At least two representative fused model expressions demonstrate a
       repeatable 1.20x gain with identical semantics and thread count.

Shared CI should enforce correctness and detect only large performance
regressions. Tight performance gates require pinned, dedicated machines.
