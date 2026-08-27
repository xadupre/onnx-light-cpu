Attention Performance Roadmap
=============================

:Date: 2026-08

**complete** (Attention PR15 parity, memory, and reproducibility gate)

``onnx-light-cpu`` provides materialized and bounded-memory streaming Attention
for the roadmap types and cache modes. The registered backend corpus and
``tools/benchmark_attention_parity.py`` provide the final reproducible
default-policy gate against ONNX Runtime. Dedicated-machine reports, rather
than shared CI timings, decide the narrow performance thresholds.

Objective
---------

The objective is a correct, bounded-memory CPU implementation of tensor-based
ONNX ``Attention`` with performance parity against the ONNX Runtime CPU
execution provider. Parity means a corpus median speed-up of at least ``1.0x``
and no priority case below ``0.9x`` on dedicated hardware.

This roadmap starts from the shared matrix engine delivered through
:doc:`Roadmap PR10.5 <2026_08_gemm_matmul>`. Attention reuses its packing,
dot-product micro-kernels, type conversion, and scheduler. The materialized
correctness path may compose those primitives, but the optimized path is not
implemented as two ordinary materialized MatMul calls.

Related roadmaps
----------------

* :doc:`Gemm and MatMul Performance Roadmap <2026_08_gemm_matmul>` provides
  the prerequisite matrix-multiplication engine. Its remaining default-policy
  parity work is independent of Attention.
* :doc:`Persistent KV Cache and Decode roadmap <2026_08_kv_cache>` owns
  persistent state, paged storage, cache quantization, and optimized cache
  append. This roadmap covers standard tensor ``past``/``present``
  compatibility and reports its allocation and copy cost separately.

Scope
-----

The adapter implements ``ai.onnx::Attention`` v23 and v24. It handles Q/K/V
head geometry, scaling, masks, causal behavior, grouped-query and multi-query
head mapping, and optional tensor cache state according to the selected opset.

The implementation supports:

* rank-4 ``[B, H, L, D]`` and rank-3 packed ``[B, L, H * D]`` layouts;
* MHA, GQA, and MQA through one internal descriptor, including a value head
  dimension different from the Q/K head dimension;
* broadcastable rank-2 through rank-4 boolean and additive ``attn_mask``
  tensors, causal masking, their intersection, and zero output for a fully
  masked query row;
* v24 ``nonpad_kv_seqlen`` external-cache masking and v23/v24 internal
  ``past_key``/``past_value`` concatenation with optional ``present`` outputs;
* ``scale``, ``softcap``, ``qk_matmul_output_mode`` 0 through 3, and the
  optional observable ``qk_matmul_output``;
* prefill, short-query lengths 2-16, and single-token decode;
* equal-type FP32, FP16, and BF16 Q/K/V with default or explicit FP32 softmax
  accumulation;
* zero-copy GQA/MQA head mapping, without physically repeating K or V;
* exact ONNX layouts, output shapes, empty sequences, and validation errors.

The ONNX schema also permits DOUBLE, a V type different from Q/K, and softmax
precisions other than FP32. Those combinations are outside this roadmap and
are not currently supported by the portable kernel either. CPU registration
must not advertise unsupported Q/K types; combinations that can reach the CPU
adapter through a supported Q type fail with an explicit unsupported error
rather than silently changing precision or returning an approximation. A
requested ``qk_matmul_output`` necessarily materializes that observable tensor;
the dispatcher therefore selects the materialized path for it. Sliding-window
or sparse behavior is in scope only when represented by a valid
``attn_mask``. It does not introduce a separate non-standard mask contract.

Benchmark contract
------------------

Optimization begins with reproducible end-to-end measurements against ONNX
Runtime and isolated measurements for the score, softmax, and value-update
stages.

* Use identical tensors, masks, layouts, CPU affinity, and correctness
  tolerances.
* The primary comparison leaves thread selection unset so ONNX Runtime and
  onnx-light choose their own default execution policies. Explicit equal-thread
  runs are separate scaling diagnostics and never replace the primary result.
* Warm up every candidate, alternate candidate order, and report median and
  dispersion rather than the best observation.
* Run on an idle, pinned machine with a fixed power policy and record CPU,
  cache, ISA, compiler, and build metadata.
* Cover query lengths 1, 2, 8, 16, 128, and 512; KV lengths 1, 128, 1024,
  4096, and 8192 when machine memory permits; MHA ``12/12``, GQA ``16/4``,
  and MQA ``16/1`` query/KV-head ratios; head dimensions 64 and 128; rank-3
  and rank-4 layouts; and FP32, FP16, and BF16.
* Pair the full attribute and mask matrix with small correctness cases. The
  timed priority corpus covers no mask, causal, boolean, additive, v24
  ``nonpad_kv_seqlen``, and internal tensor-cache cases without taking their
  impractical Cartesian product.
* Report time to first token, per-token decode latency, tokens/second, peak
  temporary memory, bytes copied for tensor ``past``/``present``, and effective
  KV-cache bandwidth.
* Measure materialized and streaming paths separately. End-to-end results must
  include every conversion, allocation, mask, and tensor past/present copy
  visible to the caller.
* The priority platforms are x86-64 AVX2, x86-64 AVX-512 when dedicated
  hardware is available, and Arm64 NEON. SVE measurements are reported
  separately until a dedicated SVE machine is part of the regular gate.

Backend test corpus
-------------------

Roadmap PR11 adds ``onnx-light-cpu`` backend test cases for ``Attention`` in
both correctness and ``TestMode::BENCHMARK`` modes. The benchmark cases:

* are registered through ``RegisterCpuAttentionCases`` and the CPU backend
  collector, so the standard onnx-light backend API, the onnx-light-cpu
  benchmark runner, and the dashboard consume the same models;
* produce inputs and expected outputs lazily so large context cases are not
  allocated while the registry is collected;
* use globally unique ``test_cpu_attention_*_benchmark`` names encoding opset,
  rank-3/rank-4 layout, MHA/GQA/MQA geometry, query and KV lengths, cache and
  mask mode, and element type;
* include the complete timed priority corpus defined above, including KV length
  8192 only in the opt-in large corpus when machine memory permits.

Unit tests execute bounded representative benchmark cases through the
registered CPU kernel. Metadata tests cover the complete corpus without
materializing every workload and enforce unique names, lazy construction,
declared input/output element counts, every priority type and geometry, and
the opt-in 8192 cases. The benchmark runner records the backend case name in
raw output so published dashboard rows remain traceable to the registered
model.

Final parity gate
-----------------

The default-policy corpus is run on a pinned dedicated machine with:

.. code-block:: bash

   python tools/benchmark_attention_parity.py --cpus 0-15 \
       --output attention_parity_results.json --enforce

The default excludes the memory-intensive KV-length 8192 cases; add ``--large``
when machine memory permits. ``--threads N`` produces a separate equal-thread
diagnostic and cannot be combined with ``--enforce``.

Every JSON row retains the globally unique backend case name, raw alternating
samples and candidate order, median and interquartile dispersion, effective
worker count, conservative peak streaming scratch and score-tile bytes from the
kernel's allocation model, tensor-cache bytes copied, and effective KV
bandwidth. Report metadata includes affinity, CPU and cache topology, ISA flags,
compiler and flags, package versions, platform, and the exact git revision. The
summary applies the ``1.0x`` median and ``0.9x`` minimum thresholds
independently to FP32, FP16, and BF16 and also requires the streaming-memory
bound. Shared CI checks the corpus structure, globally unique names, gate
arithmetic, dispatch correctness, and bounded streaming implementation; it does
not run ``--enforce``.

Phase 1: plan and materialized correctness path
-----------------------------------------------

An immutable ``AttentionDescriptor`` is built when the node is initialized. It
contains only information known without runtime inputs:

* opset, attributes, optional input/output positions, and observable outputs;
* statically known rank, type, head counts, and head dimensions when present;
* candidate dot-product, conversion, and packing functions allowed by the CPU.

Every invocation validates the actual Q/K/V, mask, and cache tensors and builds
a lightweight ``AttentionPlan`` from their concrete shapes and strides. It
records:

* batch size, query-head and KV-head counts, head dimensions, and GQA ratio;
* input/output layouts and strides, query and KV lengths, and cache mode;
* resolved scale, causal bounds, mask representation, and output obligations;
* materialized, prefill-streaming, short-query, or decode algorithm;
* query-row and KV-column block sizes and the available outer task count.

The plan does not prepack Q, K, or V at model load because those tensors are
unknown then. It emits work items to the onnx-light executor; it does not store
or impose a fixed thread count. The runtime policy decides how many workers are
admitted. Shape-specialized decisions may be cached only by a complete key and
must not retain invocation tensors.

The first implementation is a materialized correctness path:

.. code-block:: text

   S = scale * Q @ transpose(K)
   S = apply_mask_and_causality(S)
   P = softmax(S)
   O = P @ V

It validates all shape, mask, head-mapping, precision, optional-output, and
tensor cache semantics against ONNX Runtime. It remains the deterministic CPU
fallback for supported types. DOUBLE, mixed Q/K and V types, and unsupported
``softmax_precision`` combinations remain outside the registered support
boundary and produce an explicit error if encountered.

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

Here sliding-window and sparse tile skipping are optimizations inferred from a
valid boolean or additive ``attn_mask``. Arbitrary masks remain correct even
when no skip structure can be inferred. Internal ``past`` tensors and external
v24 caches may be consumed block by block, but requested ``present`` or
``qk_matmul_output`` tensors still incur the allocations required by ONNX.

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
     - Differential tests against ONNX Runtime and the portable kernel cover
       v23/v24, stateless and both cache modes, MHA/GQA/MQA, every attribute
       and optional output, mask/causal composition, fully masked rows, rank-3
       and rank-4 layouts, empty sequences, and every dispatched type.
   * - Fallback
     - The materialized path handles every optimized-type combination not
       handled by streaming. Types and precision modes outside CPU scope are
       not advertised and fail explicitly if dispatch still reaches the CPU
       adapter.
   * - Memory
     - Streaming Attention never materializes the complete score or
       probability tensor; temporary memory is bounded by worker count and
       ``Br x Bc`` blocks.
   * - Scaling
     - Prefill exposes enough independent work to scale through the physical
       cores when useful, while short-query and decode avoid harmful parallel
       overhead. The runtime, not the kernel, controls admitted workers.
   * - Performance
     - On each priority platform/type corpus, median speed-up is at least
       ``1.0x`` ONNX Runtime and no priority case is below ``0.9x``.
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
   * - Roadmap PR11 (`#387
       <https://github.com/xadupre/onnx-light-cpu/issues/387>`_)
     - Adapter, planning contract, backend corpus, and FP32 materialized
       baseline.
     - ``AttentionDescriptor`` parses v23/v24 attributes and optional IO.
       Per-invocation planning validates concrete tensors without prepacking
       unknown inputs or fixing threads. Rank-3/rank-4 stateless FP32
       MHA/GQA/MQA, scale, causal and boolean/additive masks pass differential
       tests. Lazy correctness and ``TestMode::BENCHMARK`` backend cases are
       available through the shared collector and enforce the priority corpus.
       Features not implemented yet delegate to the portable kernel only when
       that kernel already supports them; otherwise they fail explicitly.
     - Shared GEMM/MatMul engine
     - Complete
   * - Roadmap PR12 (`#389
       <https://github.com/xadupre/onnx-light-cpu/issues/389>`_)
     - Complete materialized semantics and type gate.
     - FP16/BF16, softcap, mask/causal intersection, fully masked rows, all
       ``qk_matmul_output_mode`` values, optional outputs, internal tensor
       past/present, and v24 external-cache ``nonpad_kv_seqlen`` match ONNX
       Runtime. Explicit FP32 ``softmax_precision`` matches the default
       accumulation. DOUBLE, mixed-type, and other softmax-precision cases
       prove the declared unsupported boundary.
     - PR11 / #387
     - Complete
   * - Roadmap PR13 (`#388
       <https://github.com/xadupre/onnx-light-cpu/issues/388>`_)
     - FP32 online Attention compute engine.
     - The online recurrence matches the materialized path for stateless
       prefill, short-query, and decode. SIMD score kernels fuse scale, masks,
       causal bounds, softcap, and row maximum; vector exponential and
       reductions are accurate; probability-by-V updates output directly.
     - PR12 / #389
     - Complete
   * - Roadmap PR14 (`#391
       <https://github.com/xadupre/onnx-light-cpu/issues/391>`_)
     - Streaming scheduling, cache modes, and low-precision types.
     - Runtime-owned scheduling covers MHA/GQA/MQA without nested pools or a
       fixed worker count. Internal/external cache blocks, inferable mask tile
       skipping, and FP16/BF16 score/V-update kernels match the materialized
       fallback. Observable full-tensor outputs select materialized execution.
     - PR13 / #388
     - Complete
   * - Roadmap PR15 (`#390
       <https://github.com/xadupre/onnx-light-cpu/issues/390>`_)
     - Final parity and memory gate.
     - Every priority platform/type corpus has bounded temporary memory,
       reaches at least ``1.0x`` median speed-up over ONNX Runtime, and has no
       priority case below ``0.9x``. Raw default-policy samples and environment
       metadata are published; controlled-thread runs remain diagnostic.
     - PR14 / #391
     - Complete

Roadmap PR15 (`#390
<https://github.com/xadupre/onnx-light-cpu/issues/390>`_) completed the
Attention roadmap.
