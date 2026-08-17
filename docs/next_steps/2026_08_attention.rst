Attention Performance Roadmap
=============================

:Date: 2026-08

**in progress**

Objective
---------

The objective is a correct, bounded-memory CPU implementation of tensor-based
ONNX ``Attention`` with performance parity against the ONNX Runtime CPU
execution provider. Parity means a corpus median speed-up of at least ``1.0x``
and no priority case below ``0.9x`` on dedicated hardware.

This roadmap starts after the final blocking GEMM and MatMul parity gate,
:doc:`Roadmap PR10.5 <2026_08_gemm_matmul>`. Attention reuses the shared
matrix-multiplication engine's packing, dot-product micro-kernels, type
conversion, and scheduler, but is not implemented as two ordinary materialized
MatMul calls on its optimized path.

Related roadmaps
----------------

* :doc:`Gemm and MatMul Performance Roadmap <2026_08_gemm_matmul>` provides
  the prerequisite matrix-multiplication engine and closes with PR10.5.
* :doc:`Persistent KV Cache and Decode roadmap <2026_08_kv_cache>` owns
  persistent state, paged storage, cache quantization, and optimized cache
  append. This roadmap covers standard tensor ``past``/``present``
  compatibility and reports its allocation and copy cost separately.

Scope
-----

The Attention adapter handles Q/K/V head geometry, scaling, masks, causal
behavior, grouped-query and multi-query head mapping, and optional tensor
``past``/``present`` state according to the selected ONNX opset.

The implementation supports:

* MHA, GQA, and MQA through one internal descriptor;
* boolean, additive, padding, causal, sliding-window, and sparse masks;
* prefill, short-query lengths 2-16, and single-token decode;
* FP32, FP16, and BF16 inputs with FP32 softmax accumulation;
* zero-copy GQA/MQA head mapping, without physically repeating K or V;
* exact ONNX layouts, output shapes, empty sequences, and validation errors.

Benchmark contract
------------------

Optimization begins with reproducible end-to-end measurements against ONNX
Runtime and isolated measurements for the score, softmax, and value-update
stages.

* Use identical tensors, masks, layouts, thread counts, CPU affinity, and
  correctness tolerances.
* Warm up every candidate, alternate candidate order, and report median and
  dispersion rather than the best observation.
* Run on an idle, pinned machine with a fixed power policy and record CPU,
  cache, ISA, compiler, and build metadata.
* Cover query lengths 1, 2-16, and long prefill; KV lengths from 1 to the
  target context limit; MHA, GQA, and MQA; every supported mask and type.
* Report time to first token, per-token decode latency, tokens/second, peak
  temporary memory, and effective KV-cache bandwidth.
* Measure materialized and streaming paths separately. End-to-end results must
  include every conversion, allocation, mask, and tensor past/present copy
  visible to the caller.

Phase 1: plan and materialized correctness path
-----------------------------------------------

An immutable ``AttentionPlan`` is built from model metadata, static
dimensions, CPU features, and runtime options. It records:

* batch size, query-head and KV-head counts, head dimensions, and GQA ratio;
* input/output layouts and strides;
* scale, causal mode, mask kind, and tensor past/present use;
* prefill, short-query, or single-token decode algorithm;
* query-row and KV-column block sizes;
* dot-product/packing functions, accumulation type, and useful thread count.

The first implementation is a materialized correctness path:

.. code-block:: text

   S = scale * Q @ transpose(K)
   S = apply_mask_and_causality(S)
   P = softmax(S)
   O = P @ V

It validates all shape, mask, head-mapping, precision, and tensor
past/present semantics against ONNX Runtime. It remains the deterministic
fallback for combinations not supported by streaming Attention.

Phase 2: streaming and online softmax
-------------------------------------

The optimized path fuses ``Q @ K^T``, masking, softmax, and ``P @ V`` by
blocks. It never materializes the complete
``[batch, heads, query_length, kv_length]`` score or probability tensors.

For each query block, KV blocks are processed left to right while maintaining
the running maximum ``m``, denominator ``l``, and unnormalized output ``o``:

.. code-block:: text

   m_new = max(m, row_max(S))
   correction = exp(m - m_new)
   p = exp(S - m_new)
   l = correction * l + row_sum(p)
   o = correction * o + p @ V_block
   m = m_new

   output = o / l

The CPU engine requires:

* a SIMD ``Q x K`` score kernel that fuses scale, mask, causal bounds, and row
  maximum;
* vector exponential and reductions with FP32 accumulation;
* a probability-by-V kernel that updates the output accumulator directly;
* cache-aware ``Br`` and ``Bc`` block sizes;
* causal, sliding-window, and sparse tile skipping;
* batch/head/query-block scheduling without nested thread pools;
* dedicated MHA/GQA/MQA algorithms for query lengths 1 and 2-16;
* FP16/BF16 score and V-update kernels with the materialized path as fallback.

Temporary score storage falls from
``O(B * Hq * Lq * Lkv)`` to ``O(Br * Bc)`` per worker. The expected gain is
``0-20%`` when ONNX Runtime already uses a fused path and ``1.2-2x`` when it
materializes scores or probabilities; these are targets to measure, not
guarantees.

Acceptance criteria
-------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Area
     - Exit criterion
   * - Correctness
     - Differential tests cover stateless Attention and tensor past/present,
       MHA/GQA/MQA, every mask, causal boundary, layout, empty sequence, and
       supported type.
   * - Fallback
     - The materialized path remains registered for every valid combination
       not handled by the streaming path.
   * - Memory
     - Streaming Attention never materializes the complete score or
       probability tensor; temporary memory is bounded by worker count and
       ``Br x Bc`` blocks.
   * - Scaling
     - Prefill scheduling scales through the physical-core count, while
       short-query and decode paths avoid harmful parallel overhead.
   * - Performance
     - Every priority platform/type case reaches at least ``1.0x`` ONNX Runtime
       median performance with no priority case below ``0.9x``.
   * - Reproducibility
     - Dedicated-machine results retain raw samples and environment metadata;
       shared CI enforces correctness but does not decide narrow performance
       regressions.

Pull-request sequence
---------------------

.. list-table::
   :header-rows: 1
   :widths: 12 25 43 12 8

   * - PR
     - Deliverable
     - Merge criterion
     - Dependency
     - Status
   * - Roadmap PR11
     - Materialized Attention implementation.
     - ``AttentionPlan`` validates layouts, head geometry, scale, masks, types,
       blocks, and threads. Materialized QK-softmax-PV supports all masks,
       zero-copy GQA/MQA mapping, tensor past/present, FP32/FP16/BF16, and
       batch/head/query scheduling.
     - GEMM/MatMul PR10.5
     - Pending
   * - Roadmap PR12
     - Materialized Attention correctness gate.
     - The complete MHA/GQA/MQA, mask, causal, past/present, layout, empty
       sequence, and type corpus matches ONNX Runtime; the path is registered
       as the streaming fallback.
     - PR11
     - Pending
   * - Roadmap PR13
     - Online Attention compute engine.
     - The online recurrence matches the materialized path. SIMD score kernels
       fuse scale, masks, causal bounds, and row maximum; vector exponential
       and reductions are accurate; probability-by-V updates output directly.
     - PR12
     - Pending
   * - Roadmap PR14
     - Streaming scheduling and types.
     - Prefill and dedicated short-query/decode scheduling cover MHA/GQA/MQA
       without nested pools. FP16/BF16 score and V-update kernels match the
       materialized fallback.
     - GEMM/MatMul PR10.3, PR13
     - Pending
   * - Roadmap PR15
     - Final parity and memory gate.
     - Every priority prefill/decode platform/type case has bounded temporary
       memory, reaches at least ``1.0x`` ONNX Runtime median performance, and
       has no priority case below ``0.9x``.
     - PR14
     - Pending

Roadmap PR15 is the final Attention PR.
