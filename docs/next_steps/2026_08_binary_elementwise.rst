Binary Elementwise and Broadcasting Performance Roadmap
=======================================================

:Date: 2026-08

**discussion**

Objective
---------

The objective is to implement a common CPU engine for binary elementwise
operators with full ONNX multidirectional broadcasting. Isolated kernels
should reach within 10% of ONNX Runtime on the priority corpus. The engine
should then exceed ONNX Runtime where a prepared broadcast plan, a
shape-specific loop, or fusion across several elementwise operators removes
indexing and memory traffic.

The correctness implementation in ``onnx-light`` already centralizes broadcast
validation in ``BroadcastInfo`` and provides equal-shape, scalar, and generic
rank-aware loops through ``BinaryElementwise``. The generic loop computes both
input offsets from every dimension for every output element. That is an
appropriate fallback, but not an optimized implementation for large tensors or
repeated inference.

Scope
-----

The first CPU implementation should cover:

* arithmetic: ``Add``, ``Sub``, ``Mul`` and ``Div``;
* comparisons: ``Equal``, ``Greater``, ``GreaterOrEqual``, ``Less`` and
  ``LessOrEqual``;
* logical operators: ``And``, ``Or`` and ``Xor``;
* integer bitwise operators and shifts;
* ``Min``, ``Max``, ``Mod``, ``Pow``, ``PRelu`` and other operators that can
  use the same iteration engine with a specialized scalar or vector function.

The common engine owns shape normalization, loop selection, iteration,
parallel scheduling, and ISA dispatch. Each ONNX adapter retains its own type
constraints and numerical semantics. In particular, integer ``Div`` and
``Mod``, mixed-type ``Pow``, comparisons returning ``BOOL``, NaN propagation,
and signed zero behavior must not be hidden behind an unsafe generic
approximation.

Broadcast plan
--------------

A ``BinaryBroadcastPlan`` should be prepared from the two input shapes, element
sizes, output type, operator, CPU, and thread limit. It contains:

* the right-aligned output shape;
* zero strides for broadcast dimensions and contiguous strides otherwise;
* coalesced dimensions;
* the selected loop family;
* contiguous inner-loop length and outer-loop count;
* typed scalar and vector function pointers;
* task size and useful thread count;
* guards for dynamic input shapes.

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
* byte/word/dword integer kernels for supported arithmetic, comparison,
  logical, and bitwise operations;
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
should therefore be conservative:

* remain single-threaded below a measured byte-count threshold;
* split contiguous output ranges into cache-line-aligned chunks;
* split only at inner-block boundaries for broadcast patterns;
* cap threads when aggregate memory bandwidth is saturated;
* avoid assigning adjacent partial cache lines to different threads;
* keep scalar or small broadcast operands hot in shared cache.

Thresholds should depend on element size, operation cost, loop family, and
hardware. ``Pow`` may benefit from threads much earlier than ``Add`` because it
is compute-bound.

Prepared plans and dynamic shapes
---------------------------------

Static shapes should create ``BinaryBroadcastPlan`` once per session. Dynamic
shapes should use a small cache keyed by the two shapes, types, operator, and
thread count. The fallback always rebuilds a validated plan; it must not reuse
one whose shape guards fail.

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

Fusion is the principal route to gains substantially above ONNX Runtime because
an isolated ``Add`` already approaches the memory-bandwidth limit.

Benchmark contract
------------------

Compare with ONNX Runtime using identical inputs, shapes, types, threads, and
affinity. Warm up both implementations, alternate their order, and report the
median and dispersion.

The benchmark matrix should include:

* equal shapes from tens of elements through hundreds of millions;
* left and right scalar broadcasting;
* vectors broadcast over the first, middle, and last axis;
* NCHW and NHWC per-channel broadcasting;
* batch and head broadcasting used by MatMul and Attention epilogues;
* leading-rank expansion and alternating singleton dimensions;
* ranks 0 through at least 8;
* empty dimensions and SIMD tails;
* FP32, FP64, FP16, BF16, integer, and ``BOOL`` types;
* cheap operations, division, comparison, ``Mod`` and ``Pow``;
* single-thread and physical-core configurations;
* representative fused expression chains.

Report latency, effective output bandwidth, total bytes accessed, selected loop
family, selected ISA, and thread count. For broadcasting, a bandwidth metric
must count reused input values according to actual loads rather than output
elements alone.

Implementation order
--------------------

.. list-table::
   :header-rows: 1
   :widths: 8 30 42 20

   * - Step
     - Deliverable
     - Exit criterion
     - Dependency
   * - 0
     - Differential and performance corpus.
     - Every broadcast family, type, tail, empty shape, and operand order is
       represented against ONNX Runtime.
     - None.
   * - 1
     - ``BinaryBroadcastPlan`` and coalesced scalar fallback.
     - Full correctness with no O(rank) offset calculation per output element.
     - Runtime shape metadata.
   * - 2
     - FP32/FP64 contiguous and scalar SIMD kernels.
     - ``Add``, ``Sub``, ``Mul`` and ``Div`` reach within 1.10x ONNX Runtime on
       equal-shape and scalar cases.
     - Step 1.
   * - 3
     - Repeated-block and general broadcast SIMD loops.
     - NCHW/NHWC per-channel and alternating-singleton cases reach within
       1.10x ONNX Runtime.
     - Steps 1-2.
   * - 4
     - Integer, comparison, logical, bitwise, FP16 and BF16 kernels.
     - Priority type/operator corpus reaches parity without changing numerical
       semantics.
     - Vector conversion and ISA dispatch.
   * - 5
     - Cost-aware parallel scheduler.
     - Large kernels scale until measured memory-bandwidth saturation without
       regressing small tensors.
     - Runtime thread pool.
   * - 6
     - ``Mod``, ``Pow``, ``Min``, ``Max``, ``PRelu`` and mixed-type adapters.
     - All applicable binary operators use the common plan or a documented
       specialized path.
     - Steps 1-5.
   * - 7
     - Fused ``ElementwisePlan``.
     - At least two representative model expressions exceed ONNX Runtime by a
       repeatable 20%.
     - Graph fusion and lifetime analysis.

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
     - Differential tests pass for operand order, every supported type,
       multidirectional broadcasting, ranks, empty dimensions, tails, NaN,
       infinity, signed zero, integer edge cases, and mixed output types.
   * - Common architecture
     - Arithmetic, comparison, logical, and bitwise adapters reuse
       ``BinaryBroadcastPlan`` and its traversal without duplicating broadcast
       loops.
   * - Performance parity
     - Median latency is no worse than 1.10x ONNX Runtime across the priority
       isolated-kernel corpus.
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
