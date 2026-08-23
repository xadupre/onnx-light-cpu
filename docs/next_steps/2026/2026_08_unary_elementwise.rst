Unary Elementwise Performance Roadmap
=====================================

:Date: 2026-08

**discussion**

Objective
---------

The objective is to provide one prepared CPU engine for every ONNX unary
elementwise operator, with portable scalar semantics and tuned SIMD kernels for
x86 and ARM. The priority corpus must reach at least ``1.0x`` ONNX Runtime
median performance, with no priority case below ``0.9x``. Correctness includes
all supported data types, attributes, special values, empty tensors, tails,
aliasing rules, and opset-specific behavior.

``onnx-light-cpu`` already registers optimized kernels for ``Abs``, ``Exp``,
``Log``, and ``Not``. These operators exist today; the roadmap extends their
shared architecture and adds the remaining operators rather than replacing
their working entry points. The common unary plan should retain their kernels
while moving selection, traversal, type conversion, accuracy policy, and
scheduling policy into shared preparation.

Scope
-----

This roadmap covers operators whose output elements are independently derived
from one input element. Attributes and optional scalar parameters may affect
the calculation, but no output element depends on another input position.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Family
     - Operators
     - Current status
   * - Basic arithmetic
     - ``Abs``, ``Neg``, ``Reciprocal``, ``Sqrt``, ``Sign``, ``Ceil``,
       ``Floor``, ``Round``
     - ``Abs`` implemented; others pending
   * - Exponential and error
     - ``Exp``, ``Log``, ``Erf``
     - ``Exp``/``Log`` implemented; ``Erf`` pending
   * - Trigonometric
     - ``Sin``, ``Cos``, ``Tan``, ``Asin``, ``Acos``, ``Atan``
     - Pending
   * - Hyperbolic
     - ``Sinh``, ``Cosh``, ``Tanh``, ``Asinh``, ``Acosh``, ``Atanh``
     - Pending
   * - Activations
     - ``Relu``, ``LeakyRelu``, ``ThresholdedRelu``, ``Elu``, ``Celu``,
       ``Selu``, ``Sigmoid``, ``HardSigmoid``, ``HardSwish``, ``Softplus``,
       ``Softsign``, ``Mish``, ``Swish``, ``Gelu``, ``Shrink``
     - Pending
   * - Predicates and logical
     - ``IsInf``, ``IsNaN``, ``Not``
     - ``Not`` implemented; predicates pending
   * - Bit and type transforms
     - ``BitwiseNot``, ``Cast``, ``BitCast``, ``Identity``
     - Pending
   * - String predicates
     - ``RegexFullMatch``
     - Pending specialized path
   * - Parameterized unary
     - ``Clip`` with optional scalar bounds
     - Pending

Operators with one required input but cross-element behavior are not unary
elementwise kernels and do not use this engine. This explicitly excludes
reductions and index selection (``ArgMax``, ``ArgMin``, ``NonZero``,
``Unique``), normalization and axis reductions (``Softmax``, ``LogSoftmax``,
``Hardmax``, ``LpNormalization``), pooling, layout transforms, random
generation, sequence/control-flow operators, non-elementwise string
operators, and shape operators. They require separate roadmaps rather than
hidden special cases here.

Unary execution plan
--------------------

Static nodes should construct an immutable ``UnaryElementwisePlan`` from the
operator, opset, input/output types, attributes, tensor size, CPU features,
thread limit, and accuracy policy. Dynamic shapes may cache plans by those
properties and element count.

The plan records:

* the typed scalar fallback and selected ISA function;
* input and output element sizes and whether in-place execution is legal;
* normalized activation attributes or ``Clip`` bounds;
* exact, correctly-rounded, or bounded-approximation semantics;
* vector width, unroll factor, tail strategy, task size, and useful threads;
* conversion/compute type for FP16, BF16, Float8, and integer inputs;
* guards that reject incompatible shapes, types, attributes, or opsets.

Traversal and calculation remain separate. A hot loop calls a typed function
selected once by the plan; it must not branch on the operator or perform
type-erased calls for every element.

Kernel families
---------------

Native instruction kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~

``Abs``, ``Neg``, ``Relu``, ``Sign``, rounding, predicates, logical/bitwise
operations, and many casts map directly to SIMD arithmetic, masks, or
conversion instructions. These kernels should provide:

* scalar, SSE2/AVX2, AVX-512, NEON, and SVE/SVE2 implementations;
* vector tails through masks where profitable and one shared scalar fallback;
* byte-valued ONNX ``BOOL`` output without bit packing;
* defined integer behavior without relying on C++ signed-overflow assumptions;
* alias-safe in-place execution only where input and output representation
  permit it.

Transcendental kernels
~~~~~~~~~~~~~~~~~~~~~~

Trigonometric, hyperbolic, exponential, logarithmic, error, sigmoid, and GELU
families require range reduction and polynomial or rational approximations.
Share primitives instead of independently approximating each operator:

* ``Exp`` feeds sigmoid, softplus, swish, mish, and parts of hyperbolic
  functions;
* ``Log``/``Log1p`` feed softplus and inverse hyperbolic functions;
* shared sine/cosine range reduction computes ``Sin`` and ``Cos`` together;
* ``Erf`` feeds the exact GELU formulation;
* reciprocal and reciprocal-square-root helpers retain explicit refinement and
  error contracts.

Every approximation documents maximum ULP or relative error over normal,
subnormal, overflow, and underflow ranges. NaN payload behavior, infinities,
domain errors, and signed zero must match the ONNX/host reference contract.
Fast-math behavior is opt-in and never silently replaces the default kernel.

Composite activations
~~~~~~~~~~~~~~~~~~~~~

Activation kernels should compose shared vector primitives but remain fused in
one traversal. For example, ``Mish`` must not allocate outputs for softplus and
tanh, and ``HardSwish`` should keep its clamp and multiply in registers.
Attribute-bearing activations normalize constants in the plan so the hot loop
loads broadcast vectors rather than parsing attributes.

Types
-----

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Type family
     - Preferred implementation
     - Required behavior
   * - FP32/FP64
     - Native SIMD or documented vector approximation
     - Preserve special values, signed zero, domain, and accuracy contracts.
   * - FP16/BF16
     - Native arithmetic or vector conversion to FP32
     - Convert vectors, compute safely, and narrow once; avoid scalar
       decode/encode loops.
   * - Float8
     - Explicit vector decode, FP16/FP32 compute, explicit encode
     - Keep each ONNX Float8 encoding and saturation rule distinct.
   * - Integers
     - Native vector arithmetic/masks where defined
     - Match operator type constraints and exact overflow semantics.
   * - ``BOOL``
     - Byte-vector masks
     - Preserve the runtime byte representation.
   * - Cast output
     - Typed conversion kernels
     - Match truncation, saturation, string, and unsupported-conversion rules
       for the selected opset; non-numeric conversions may use a specialized
       fallback outside the SIMD numeric loop.
   * - String tensors
     - Exact specialized loops
     - ``RegexFullMatch`` preserves regex and invalid-pattern semantics and
       may parallelize independent strings without entering numeric SIMD code.

Parallel scheduling
-------------------

Unary arithmetic is usually memory-bandwidth-bound, while transcendental
operators are compute-bound. Registered kernels already execute through the
onnx-light session executor, with no private ``onnx-light-cpu`` scheduler.
Prepared kernels therefore select measured limits for the session-owned
executor by operator family, type, processor profile, and ISA:

* cheap kernels stay single-threaded until enough bytes amortize dispatch;
* expensive functions may parallelize at much smaller element counts;
* blocks begin on cache-line and SIMD boundaries;
* thread count is capped when memory bandwidth or available blocks saturate;
* caller-owned pools and the internal pool must not oversubscribe each other.

Fusion
------

Unary kernels expose typed functions and plan metadata to the shared
``ElementwisePlan`` described by the binary roadmap. Fusion may combine unary
and binary nodes only when evaluation order, broadcasting, types, aliasing,
and graph lifetimes permit it. Standalone unary parity does not depend on
fusion, but fusion is required to remove intermediate tensor traffic in common
activation and normalization expressions.

Benchmark corpus
----------------

The corpus compares isolated kernels and representative fused expressions
against ONNX Runtime. It covers:

* sizes from zero and scalar tensors through bandwidth-saturating tensors;
* every SIMD tail and deliberately misaligned input/output addresses;
* every supported type, attribute boundary, and opset behavior;
* NaN, infinities, signed zero, subnormals, domain boundaries, and saturation;
* single-thread, physical-core, and logical-core configurations;
* latency, throughput, effective bandwidth, selected ISA, and thread count;
* accuracy histograms and worst-case error for approximate functions.

Shared CI enforces correctness. Tight performance and numerical-search gates
run on pinned machines and preserve raw samples and environment metadata.

Completed foundations
---------------------

The dedicated :doc:`Exp and Log ONNX Runtime Parity Roadmap
<2026_08_exp_log_parity>` is complete through
`onnx-light-cpu #315
<https://github.com/xadupre/onnx-light-cpu/pull/315>`_. Its numerical gates,
AVX2+FMA and AVX-512 kernels, benchmark corpus, operator-specific scheduling,
and preserved evidence are inputs to this roadmap. Unary PR03 reuses those
``Exp``/``Log`` implementations and primitives; it does not reimplement their
parity work.

The :doc:`Runtime Execution Controls Roadmap
<2026_08_runtime_execution_controls>` is also complete through
`onnx-light-cpu #271
<https://github.com/xadupre/onnx-light-cpu/pull/271>`_ and
`#314 <https://github.com/xadupre/onnx-light-cpu/pull/314>`_. Registered
kernels use the session-owned executor, and the existing onnx-light
processor-aware tuning registry supplies the profile-resolution foundation.
The remaining unary PRs build on these completed foundations; they do not add
a private scheduler.

Remaining pull-request sequence
-------------------------------

The following table is the single source of truth for the unary roadmap.

.. list-table::
   :header-rows: 1
   :widths: 9 27 44 12 8

   * - PR
     - Scope
     - Merge criterion
     - Depends on
     - Status
   * - Unary PR01
     - Corpus, plan, registry, and scalar semantics.
     - ``UnaryElementwisePlan`` and common adapters cover every in-scope
       operator/type/opset with guarded scalar fallbacks. Differential cases
       include attributes, special values, empty tensors, tails, and aliasing,
       while existing ``Abs``/``Exp``/``Log``/``Not`` kernels remain usable.
     - Completed runtime foundation
     - Pending
   * - Unary PR02
     - Native arithmetic, predicates, bits, and numeric casts.
     - Basic arithmetic, rounding, sign, Relu-family native operations,
       ``IsInf``, ``IsNaN``, ``Not``, ``BitwiseNot``, and numeric
       ``Cast``/``BitCast``/``Identity`` use shared x86 and ARM SIMD traversal.
     - PR01
     - Pending
   * - Unary PR03
     - Reuse Exp/Log primitives for reciprocal, sqrt, and composite
       activations.
     - The merged ``Exp``/``Log`` kernels, numerical gates, corpus, and
       scheduling evidence remain authoritative. Reciprocal, sqrt, sigmoid,
       softplus, softsign, hard activations, swish, mish, GELU, Elu/Celu/Selu,
       shrink, and clip use fused vector kernels with documented special-value
       and error contracts.
     - PR01, PR02
     - Pending
   * - Unary PR04
     - Trigonometric, hyperbolic, inverse, and error functions.
     - Shared range reduction and approximation primitives implement
       sin/cos/tan, inverse trigonometric, hyperbolic, inverse hyperbolic, and
       erf on x86 and ARM within the documented numerical limits.
     - PR03
     - Pending
   * - Unary PR05
     - Low-precision and remaining conversion families.
     - FP16/BF16/Float8 vector conversion or native kernels cover all
       applicable operators; non-numeric casts and ``RegexFullMatch`` use exact
       specialized fallbacks. No full-tensor conversion pass is added around
       unary compute.
     - PR02 through PR04
     - Pending
   * - Unary PR06
     - Session-executor tuning and fusion integration.
     - Processor-aware limits submitted to the session executor scale
       compute-bound kernels and cap bandwidth-bound kernels without
       small-tensor regressions or a private scheduler. Unary functions
       integrate with ``ElementwisePlan`` without indirect calls in hot loops.
     - PR02 through PR05; Binary PR07
     - Pending
   * - Unary PR07
     - Final correctness and parity gate.
     - Every in-scope operator/type passes differential tests; median priority
       performance is at least 1.0x ONNX Runtime with no priority case below
       0.9x. This PR remains open while any target fails.
     - PR01 through PR06
     - Pending

Unary PR07 is the final unary roadmap PR.
