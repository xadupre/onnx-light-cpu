Persistent KV Cache and Decode Performance Roadmap
===================================================

:Date: 2026-08

**discussion**

Objective
---------

The objective is to add first-class persistent caches to ``onnx-light`` and
use them to optimize autoregressive Attention without copying the complete
past K/V tensors at every token.

This roadmap starts after the tensor-based Attention correctness path from the
:doc:`Gemm, MatMul, and Attention roadmap <2026_08_gemm_matmul>` exists. It
extends the runtime, not the standard ONNX ``Attention`` schema: standard
models continue to accept and return ``past``/``present`` tensors, while an
optimized internal graph may replace loop-carried tensors with a mutable cache
value.

Why a dedicated runtime value is needed
---------------------------------------

Representing a growing KV cache as an ordinary ``Tensor`` forces one of two
bad contracts: reallocate and copy ``O(Lkv)`` elements on every append, or
expose unused capacity and mutable aliases through a tensor whose ONNX shape
claims to contain only valid elements. A ``Sequence<Tensor>`` avoids one large
copy but adds per-token objects, pointer chasing, and a layout the Attention
kernel cannot tune. Neither representation naturally carries a page table,
capacity, valid length, quantization scales, or beam-sharing metadata.

Introduce a moveable, reference-counted ``KVCache`` runtime value in
``onnx-light``. It is not a new ONNX tensor element type. It owns or references:

* K and V storage, element/accumulation types, device, and allocator;
* batch size, KV-head count, head dimensions, valid length, and capacity;
* logical layout plus contiguous or paged physical layout descriptors;
* block size, page table, per-sequence lengths, and optional sliding-window
  start positions;
* optional per-head/per-block quantization scales and zero points;
* sharing/copy-on-write metadata required by beam search.

The standard ONNX boundary remains tensor-based. A standard ``Attention`` node
can import ``past_key``/``past_value`` tensors into a cache and export
``present_key``/``present_value`` tensors when those outputs are observable.
An optimization pass may remove both conversions only when it proves that the
present values are consumed exclusively as cache state by the next invocation.
Internal cache-aware nodes may use
``opaque(ai.onnx-light.runtime,KVCache)`` in an optimized model or execution
plan, but serialized standard ONNX graphs must not be rewritten to claim that
the standard ``Attention`` inputs have an opaque type.

Required changes in ``onnx-light``
----------------------------------

The cache cannot be implemented only in ``onnx-light-cpu`` because ownership,
graph value typing, release scheduling, subgraphs, and repeated invocations
belong to the runtime. The following changes are required:

* Add ``onnx_core/runtime/kv_cache.h`` with the backend-neutral ``KVCache``
  handle and layout metadata. Storage operations are supplied by the selected
  backend so the core runtime does not depend on CPU-specific packing.
* Add ``CacheMap`` and ``HasCache``/``GetCache``/``PutCache``/``RemoveCache``
  to ``RuntimeContext``, parallel to its tensor, sequence, map, and shape
  stores. Subgraph and function contexts must propagate cache handles without
  deep copies and document when mutation is visible to the parent.
* Split the current all-or-nothing ``RuntimeContext::Clear()`` contract:
  ``ClearValues()`` starts a new invocation while preserving declared
  persistent state, and ``ResetState()`` releases caches as well. Repeated
  ``RuntimeSession::Run`` calls must not accidentally discard or retain cache
  state.
* Extend ``ExecuteActionKind`` and ``ExecutionPlan`` with cache create/delete
  actions and type-aware input/output lookup. Cache values must participate in
  last-use analysis, graph outputs, control-flow captures, and event/profiling
  records without being treated as allocator-backed tensors.
* Add a persistent-state allocator or arena contract to ``RuntimeSession``.
  Cache pages outlive one invocation, must not use temporary-buffer lifetime,
  and must expose allocated/committed bytes to memory accounting.
* Extend runtime bindings so C++ and Python callers can create, inspect, reset,
  clone/fork, truncate, and reorder a cache. APIs must expose logical metadata,
  not raw mutable pointers by default.
* Add graph analysis that recognizes safe loop-carried
  ``past -> concat/scatter -> present`` patterns and builds a cache-aware
  execution plan. If aliasing, external observation, or unsupported control
  flow prevents the rewrite, retain the standard tensor path.
* Add internal ``CacheCreate``, ``CacheAppend``, ``CacheView``,
  ``CacheExport``, ``CacheReorder`` and ``CacheTruncate`` operations, or
  equivalent execution-plan actions. Beam search requires reorder/fork with
  copy-on-write pages; speculative decoding requires checkpoint and truncate.
* Make state ownership explicit: a session is not concurrently mutable unless
  each request has its own cache handle. Cloning a session must not silently
  share writable cache pages.

Required changes in ``onnx-light-cpu``
--------------------------------------

The CPU backend supplies the physical implementation behind ``KVCache``:

* contiguous capacity-growing storage first, followed by fixed-size paged
  blocks and a page-table iterator consumed directly by Attention;
* separate K and V layouts when measurements justify them: K optimized for
  query dot products and V for weighted accumulation;
* append/conversion kernels for FP32, FP16, and BF16 that never transpose or
  copy the complete past;
* zero-copy GQA/MQA head mapping and grouped processing of query heads sharing
  one KV head;
* cache-aware streaming Attention whose block iterator accepts contiguous,
  paged, sliding-window, and eventually quantized blocks;
* explicit export kernels for observable standard ONNX ``present`` outputs;
* INT8, then INT4, old-block quantization with scales stored in cache metadata
  and fused decode in the streaming kernel.

Phase 1: decode with contiguous KV cache
----------------------------------------

Single-token decode is a different algorithmic regime. With ``Lq == 1``, it is
primarily a KV-cache bandwidth problem, not a GEMM problem. Use a dedicated
streaming kernel:

1. compute query/K dot products for one KV block;
2. update online-softmax state;
3. immediately accumulate the corresponding V block;
4. continue without storing the score vector.

The first implementation must:

* append present K/V without transposing or copying the complete past cache;
* store K in a layout friendly to dot products and V in a layout friendly to
  weighted accumulation, or maintain a justified shared blocked layout;
* map GQA/MQA query heads onto shared KV blocks without duplicate cache reads
  when query heads can be processed together;
* establish the block-iterator contract used by later paged and quantized
  storage.

For FP16/BF16, convert Q/K/V vectors inside the block load/packing step and
keep score, softmax, and output accumulation in FP32. Native AVX-512BF16,
AVX-512FP16, AMX, or ARM dot-product variants can replace conversion paths
when available.

Parallelism should be selected in this order: batch, KV head/query-head group,
query block, then KV block. Splitting one query row across KV workers requires
a numerically correct merge of ``(m, l, o)`` states and should only be used
when the outer dimensions cannot occupy the cores.

Phase 2: paged cache and generation operations
----------------------------------------------

Replace capacity-growing contiguous buffers with fixed-size pages once the
contiguous implementation is correct. Attention consumes pages directly
through the cache block iterator. Add beam reorder/fork, copy-on-write,
sliding-window eviction, speculative checkpoints, truncate, and deterministic
out-of-memory behavior. No operation may gather the complete logical cache.

Phase 3: cache quantization and fusion
--------------------------------------

Quantize cold pages to INT8 before considering INT4. Keep recent tokens in the
model's native type when that improves quality or avoids short-context decode
overhead. Quantization policy is a runtime option and part of the cache
descriptor; it is never inferred only from the tensor type. Fuse projection
layout conversion, rotary embedding, cache append, and Attention only after
the unfused cache-aware graph is differentially correct.

Benchmark contract
------------------

Cache measurements must distinguish runtime-state gains from kernel gains:

* compare standard tensor ``past``/``present`` compatibility, contiguous cache,
  and paged cache as separate configurations;
* do not request or time ``CacheExport`` when a real generation graph does not
  observe ``present`` tensors;
* report append bytes copied, bytes read by Attention, committed/resident cache
  bytes, page-table overhead, fragmentation, and peak temporary memory;
* cover context growth, fixed long contexts, sliding windows, batch changes,
  beam fork/reorder, speculative checkpoint/truncate, GQA, and MQA;
* compare ONNX Runtime with an equivalent cache representation and execution
  contract; a tensor-exporting baseline is not a valid comparator for an
  internal opaque-cache path;
* verify that append work is proportional to new tokens, not existing context
  length, and that no hidden full-cache gather occurs.

Expected gains
--------------

The following estimates are targets to verify against ONNX Runtime with the
same thread count and an equivalent cache contract, not guarantees.

.. list-table::
   :header-rows: 1
   :widths: 20 22 36 22

   * - Optimization
     - Expected gain
     - Conditions and quantitative bound
     - Estimated effort
   * - Decode-specific KV layout
     - **5-30%** for ordinary MHA/GQA decode; **1.2-1.8x** when it avoids
       cache transposes, gathers, or duplicated GQA reads.
     - Decode reads approximately ``2 * Lkv * Dkv`` K/V elements per KV head
       and token. The upper bound is the ratio between the old and new bytes
       transferred; compute optimization cannot exceed the memory-bandwidth
       ceiling.
     - 7-15 days for contiguous cache, additional work for paged cache.
   * - Quantized KV cache
     - **1.2-1.8x decode throughput** at long context if decode is
       bandwidth-bound; approximately **2x less cache traffic** for FP16 to
       INT8 and **4x less** for FP16 to INT4.
     - Actual speed-up is below the compression ratio because scales must be
       loaded and values decoded. Requires model-quality validation and an
       explicit runtime contract for the quantized cache representation.
     - 10-20 days per quantized format.
   * - Fused projection and Attention
     - **5-20%** end-to-end for prefill or decode blocks; potentially
       **1.2-1.5x** for small-token decode.
     - Fuse Q/K/V projection layout conversion, rotary embedding, cache append,
       Attention, and output projection boundaries only where the graph proves
       equivalent semantics.
     - 10-20 days per supported fusion pattern.

Acceptance criteria
-------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Area
     - Exit criterion
   * - ONNX compatibility
     - Standard tensor ``past``/``present`` models produce the same observable
       outputs with and without the internal cache rewrite.
   * - Cache lifetime
     - Reset, reuse, subgraph capture, session destruction, request isolation,
       beam fork/reorder, and speculative truncate have deterministic ownership
       and sanitizer-covered tests.
   * - Append complexity
     - Appending new tokens never copies or transposes the complete existing
       cache; measured copied bytes are proportional to appended tokens.
   * - Paged execution
     - Attention consumes contiguous and paged blocks directly without a
       full-cache gather, including GQA/MQA and sliding-window cases.
   * - Decode parity
     - Median single-token decode latency is no worse than 1.10x ONNX Runtime
       across the target context range with equivalent cache type and threads.
   * - Memory
     - Committed bytes, live bytes, fragmentation, page-table overhead, and
       quantization metadata are observable and stay within defined limits.
   * - Quantization
     - INT8/INT4 cache modes meet agreed model-quality tolerances and improve
       long-context bandwidth or are disabled for that workload.

Performance gates should run on dedicated, pinned hardware and store raw
samples and environment metadata. Shared CI machines can enforce correctness
and detect catastrophic slowdowns, but they should not decide a 5-10%
performance regression.

Implementation order and dependencies
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 8 24 36 17 15

   * - Step
     - Deliverable
     - Exit criterion
     - Dependency
     - Pull requests
   * - C0
     - Backend-neutral ``KVCache`` value, ``CacheMap``, persistent allocator,
       lifetime actions, reset semantics, events, and C++/Python bindings in
       ``onnx-light``.
     - Cache values survive invocation cleanup, are explicitly reset, and pass
       ownership, subgraph, release-plan, and request-isolation tests.
     - Attention step P6.
     - TBD.
   * - C1
     - Tensor import/export adapters, safe graph rewrite, contiguous CPU cache,
       and cache append/view operations.
     - Standard ONNX outputs remain identical; append copies only new tokens;
       rewrite falls back whenever cache state is externally observable.
     - C0 and Attention step P6.
     - TBD.
   * - C2
     - Cache-aware streaming single-token decode.
     - Decode is within 1.1x of ONNX Runtime over the target context range and
       no complete score, present tensor, or cache gather is materialized.
     - C1 and Attention step P7.
     - TBD.
   * - C3
     - Paged storage, sliding-window eviction, beam fork/reorder, copy-on-write,
       and speculative checkpoint/truncate.
     - Generation operations avoid full-cache copies and paged decode matches
       contiguous-cache results.
     - C2.
     - TBD.
   * - C4
     - INT8/INT4 cold-page quantization and proven-safe projection/rotary/cache
       fusion.
     - At least one representative long-context workload exceeds ONNX Runtime
       by a repeatable 10% within agreed model-quality tolerances.
     - Attention step P5 and C3.
     - TBD.
