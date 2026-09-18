Qwen3 Missing-Kernel Implementation Plan
=========================================

:Date: 2026-09

**in progress**

Objective and boundary
----------------------

Make the audited dense Qwen3 INT4 graph executable by implementing its missing
CPU kernels and compatibility adapters. This is the kernel delivery plan, not
the complete generation, model-quality, or persistent-cache roadmap.

The exact graph inventory and projection shapes are recorded in
:doc:`Qwen3 CPU Inference Critical Path <2026_08_qwen3_inference>`.
The independent float-block integration contract is recorded in
:doc:`Qwen3 Non-MatMulNBits Operator Slice <2026_09_qwen3_operator_slice>`.
The steps below split those deliverables into independently reviewable PRs;
they do not introduce a second implementation of their shared engines.

Already available: ``GroupQueryAttention`` with fused RoPE and tensor cache
inputs/outputs, ``RMSNormalization``, ``Sigmoid``, ``Mul``, ``Sub``, and the
shared matrix engine. Since this plan was written, ``Gather``, ``Cast``,
``Split``, ``SimplifiedLayerNormalization``, and
``SkipSimplifiedLayerNormalization`` have also been delivered. Reuse them
rather than introducing new attention, normalization, layout, or activation
kernels.

Runtime prerequisites, not new CPU kernels
------------------------------------------

``onnx-light`` owns ``Constant``, ``Shape``, ``Reshape``, storage lifetime,
and view representation. First establish which contracts already work in
that repository; absence from the CPU registration inventory is not proof
that an upstream implementation is missing.

Constant values can be prepared once. Dynamic shape values must be resolved
from the current invocation without reading tensor payloads. Legal contiguous
reshapes should share storage. Last-axis ``Split`` outputs are not generally
contiguous across multiple rows: use views only if the runtime and consumers
support the required strides; otherwise use a correct materialized copy.

Persistent KV-cache ownership, capacity, reset, isolation, and lifetime across
invocations belong to ``onnx-light`` (Qwen PR06a). CPU append and cache-aware
Attention belong to Qwen PR06b after that contract exists. Neither is a
prerequisite for the initial kernel milestone, which uses the existing tensor
``past``/``present`` GQA interface and reports its copy cost.

Progress snapshot
-----------------

As of 2026-09-18, K04 and K07 are complete. K05, K06, and K08 have delivered
their principal kernels but retain the integration work listed below. K00 is
partially covered by their direct and backend fixtures. K01--K03 and K09 have
not started.

The delivered operator work is:

* ``Gather``: `#666 <https://github.com/xadupre/onnx-light-cpu/pull/666>`_;
* ``Split``: `#673 <https://github.com/xadupre/onnx-light-cpu/pull/673>`_;
* ``Cast``: `#674 <https://github.com/xadupre/onnx-light-cpu/pull/674>`_;
* ``SimplifiedLayerNormalization``:
  `#675 <https://github.com/xadupre/onnx-light-cpu/pull/675>`_, with subsequent
  precision and SIMD work in
  `#732 <https://github.com/xadupre/onnx-light-cpu/pull/732>`_;
* ``SkipSimplifiedLayerNormalization``:
  `#678 <https://github.com/xadupre/onnx-light-cpu/pull/678>`_.

The remaining critical path is the exact ``MatMulNBits`` contract, decode
GEMV, prefill GEMM, INT64 ``ReduceSum``, and complete-block integration.
Zero-copy ``Reshape`` and persistent KV-cache work remain owned by
``onnx-light``; they affect efficiency but do not block the initial
past/present-tensor correctness milestone.

PR sequence
-----------

K00 continues alongside the remaining implementation steps. Model downloads
and a complete generation benchmark are not prerequisites for isolated kernel
development.

.. list-table::
   :header-rows: 1
   :widths: 8 13 24 43 12

   * - Step
     - Status
     - Deliverable
     - Acceptance
     - Depends on
   * - K00
     - Partial
     - Audited contracts and C++ backend fixtures.
     - Deterministic small inputs reproduce the exact domains, opsets,
       attributes, types, optional outputs, and dynamic dimensions. Expected
       outputs come from ONNX Runtime or supported built-in onnx-light
       kernels, not reimplemented Python math. Record upstream execution
       ownership and missing contracts. Operator fixtures exist for K04--K08;
       the executable block, ``MatMulNBits`` contract, and complete ownership
       inventory remain.
     - Existing graph inventory
   * - K01
     - Not started
     - ``MatMulNBits`` schema, adapter, and prepared storage.
     - Validate the audited three-input FP32/UINT8/FP32 contract with
       ``bits=4``, ``block_size=32``, and ``accuracy_level=4``. Match packed
       layout, implicit zero points, scale indexing, and padding against ONNX
       Runtime. Reuse constant packed weights; reject unsupported variants
       explicitly without expanding the complete weight matrix.
     - K00
   * - K02
     - Not started
     - Packed INT4 decode GEMV.
     - The K01 plan executes ``M=1`` for every audited projection, including
       the vocabulary head. Scales and zero points are applied while
       consuming bounded panels. Exact packing/tail tests and numerical
       comparisons pass; repeated calls do not repack constant weights.
     - K01
   * - K03
     - Not started
     - INT4 small-M and prefill GEMM.
     - Reuse the same compressed weights for short prompts and prefill.
       Bound workspace, cover partial blocks and output tails, and publish
       latency and memory comparisons without regressing decode.
     - K02
   * - K04
     - Complete
     - ``Gather``.
     - FP32 embedding lookup on axis 0 with INT64 indices and the INT64
       scalar shape-gather case pass negative-index, bounds, empty-index,
       shape, and scalar-output cases. Read selected rows only; do not copy
       the entire embedding table.
     - K00
   * - K05
     - Partial
     - ``Cast`` and ``ReduceSum`` mask prelude.
     - INT64 mask reduction with tensor axes and ``keepdims=0`` followed by
       INT64-to-INT32 conversion matches the source schemas. Test empty
       reductions, axes, dtype boundaries, and malformed inputs. GQA length
       validation must not redefine generic ONNX Cast semantics. ``Cast`` and
       its audited Qwen conversions are delivered; INT64 ``ReduceSum`` remains.
     - K00
   * - K06
     - Partial
     - ``Split`` and layout integration.
     - Last-axis FP32 QKV splitting with ``[2048,1024,1024]`` works for
       multi-token as well as single-token inputs. Verify split totals,
       ownership, release order, output strides, and a materialized fallback.
       The registered materialized implementation covers decode, small-M,
       prefill, and batched shapes. Complete the composed release-order and
       upstream ``Reshape``-view validation.
     - K00; upstream layout contract
   * - K07
     - Complete
     - ``SimplifiedLayerNormalization`` adapter.
     - Resolve the experimental ``ai.onnx`` schema/loading contract and
       reuse RMSNormalization for ``axis=-1``, ``epsilon=1e-6``, and
       ``stash_type=1``. Prove output and accumulation equivalence, including
       optional outputs if supported; do not advertise unsupported forms.
     - K00; existing RMS engine
   * - K08
     - Partial
     - ``SkipSimplifiedLayerNormalization`` adapter.
     - Reuse residual addition and the shared RMS engine. Preserve sparse
       optional outputs for all audited output layouts, especially the
       residual result. Cover omitted outputs, aliasing, epsilon, shape
       errors, and accumulation precision. The registered kernel and sparse
       optional-output tests are delivered; complete the ORT differential and
       caller-output aliasing evidence. Fusion is a measured follow-up, not a
       prerequisite for correct execution.
     - K07
   * - K09
     - Not started
     - Complete block and native INT4 graph integration.
     - First execute a deterministic block with FP32 MatMul using K04-K08,
       then replace projections with K01-K03. Compare intermediates, logits,
       prefill and consecutive decode calls against ONNX Runtime. Every node
       has explicit execution ownership; report tensor-cache copies
       separately from kernel time.
     - K03--K08; upstream metadata contracts

K01-K03 implement the native part of parent Qwen PR02-PR03. Standard
``DequantizeLinear -> MatMul`` recognition and tied quantized embedding
support remain separate parent-roadmap work; neither is required to execute
this audited graph, whose embedding table is FP32. Existing packed integer
MatMul is a reusable primitive, not an implementation of weight-only
``MatMulNBits`` semantics.

K04-K06 refine Operator PR03; K07-K08 refine Operator PR04. K00 and K09
provide the fixtures and integration gates of Operator PR01 and PR05.
Input/layout work and normalization can proceed independently of INT4
projections after K00. Do not wait for the persistent-cache API to merge
these kernels.

Common implementation and performance gates
--------------------------------------------

Each kernel PR includes registration, shape/schema integration where needed,
direct C++ tests, lazy C++ backend correctness and benchmark cases, and the
affected documentation. Registration must match the supported contract;
unsupported types or optional variants must not silently execute reference
kernels in production.

Use the existing session executor and shared primitives. Start with a correct
portable path, then add or reuse AVX2 and AVX-512 dispatch where measurements
justify it. Do not require SIMD for metadata or tiny mask-prelude operations.

Record a Release baseline against ONNX Runtime before optimization, then
repeat the same cases with the AVX2 ceiling and automatic dispatch at one
thread and a fixed physical-core policy. Publish raw samples, dispersion,
CPU/build metadata, preparation time, steady-state latency, workspace, and
copied bytes. Shared-runner results are diagnostic, not a narrow parity gate.

Functional completion means K09 executes without an untracked fallback.
Performance completion is separate: target at least ``1.0x`` ONNX Runtime
median for each priority compute family and no priority case below ``0.9x``
on the reference machine. List unresolved gaps explicitly rather than
blocking unrelated correct kernels or claiming that graph execution proves
end-to-end Qwen performance parity.
