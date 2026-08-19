Qwen3 CPU Inference Critical Path
=================================

:Date: 2026-08

**discussion**

Objective
---------

The objective is the shortest path from the current kernel set to useful
batch-1 CPU inference for a dense, text-only Qwen3 model, with particular
priority given to weight-only INT4. This is not a request to complete every
generic operator roadmap first. Work is ordered by its effect on:

* model load time and resident weight memory;
* time to first token for prompt prefill;
* steady-state single-token decode latency;
* copied KV-cache bytes per generated token.

The first functional target is Qwen3-0.6B because it keeps iteration and
correctness tests short. Qwen3-4B is the first performance and memory target.
The initial scope excludes Qwen3 MoE, VL, hybrid recurrent/attention variants,
continuous batching, beam search, and speculative decoding. Those extensions
must not delay the dense decoder.

The critical order is:

#. benchmark one frozen Qwen3 graph end to end;
#. implement constant weight-only INT4 MatMul, decode GEMV first;
#. complete the small set of Qwen block operators;
#. implement batch-1 causal GQA and rotary embedding;
#. eliminate full KV-cache copies during decode;
#. tune prefill, threads, and fusions only after decode is structurally sound.

The first useful milestone ends after step 5. Paged caches, INT4 KV caches,
sampling inside the graph, generic attention masks, and broad operator parity
come later.

Plans to execute first
----------------------

Do not execute the existing roadmaps from top to bottom. Use only their
Qwen-critical slices in this order:

.. list-table::
   :header-rows: 1
   :widths: 10 31 41 18

   * - Order
     - Existing/new plan
     - Work to execute
     - Work deliberately deferred
   * - 1
     - This roadmap, Qwen PR01
     - Frozen graph, operator inventory, generation correctness, TTFT/decode
       benchmark, memory and per-node profile.
     - All kernel changes until the baseline is reproducible.
   * - 2
     - Gemm/MatMul PR09.6 plus Qwen PR02-PR03
     - Standard QDQ INT4 recognition, packed decode GEMV, small-M, then prefill
       GEMM for exact Qwen shapes.
     - Float8, full integer parity, float64, and generic PR10.5.
   * - 3
     - ExpLog PR01-PR03 plus Qwen PR04
     - Correct/fast Exp, RMSNorm, RoPE, Gather, SiLU gate, Softmax, and layout
       operations needed by one block.
     - Log tuning and the full unary/binary operator matrices.
   * - 4
     - Qwen PR05, narrow slice of Attention PR11-PR14
     - Batch-1 causal GQA, online softmax, prefill, and single-token decode.
     - General masks, layouts, types, and the generic Attention PR15 gate.
   * - 5
     - KV-cache C0-C2 plus Qwen PR06
     - Request-owned contiguous cache and cache-aware streaming decode.
     - Paging, beam/speculative operations, and KV quantization.
   * - Parallel prerequisite
     - Runtime PR01, PR02, and PR04
     - Session executor and no nested pools, completed before final Qwen
       scheduling/tuning.
     - Runtime inspection, standalone compatibility, and broad policy gates.
   * - 6
     - Qwen PR07-PR08
     - End-to-end scheduling, measured fusions, and parity gate.
     - Any optimization absent from the profile.

Export contract
---------------

Two equivalent standard-ONNX graphs are retained:

``qwen3-float``
    A correctness graph with ordinary constant float32, float16, or bfloat16
    weights and ONNX opset 23 or later.

``qwen3-int4-qdq``
    A weight-only graph whose constant INT4/UINT4 weights, block scales, and
    optional zero points are represented by ``DequantizeLinear`` feeding
    ``MatMul``. The runtime must recognize the constant QDQ pattern and pass
    packed weights directly to an internal weight-only plan. It must never
    materialize a complete float weight tensor.

Opset 23 is preferred because it provides the standard
``RMSNormalization`` and ``RotaryEmbedding`` schemas. An exporter that emits
their primitive decompositions remains supported through graph matching, but
those decompositions are not the preferred optimized representation.

Every benchmark artifact records the exact model revision, exporter revision,
opsets, graph digest, quantization block size, symmetric/asymmetric encoding,
weight packing order, tokenizer, prompt tokens, generated token ids, and
runtime options. No result from one INT4 encoding is attributed to another.

Operator priority
-----------------

Functional graph coverage and performance priority are different. Shape-only
operators may be required to run the graph but do not deserve a dedicated
optimization project before MatMul.

.. list-table::
   :header-rows: 1
   :widths: 13 29 37 21

   * - Priority
     - Standard ONNX operators
     - Role
     - Required optimization
   * - P0
     - ``DequantizeLinear`` + ``MatMul``
     - Q/K/V/O projections, gate/up/down MLP projections, and LM head.
     - Recognize constant blocked INT4 QDQ; packed weight-only GEMV/GEMM with
       fused scales and zero points.
   * - P0
     - ``MatMul``, ``Gemm``
     - Float fallback, attention products, and projection reference.
     - Constant-B packing, ``M == 1``/small-M kernels, then prefill GEMM.
   * - P0
     - ``Gather``
     - Token embedding lookup.
     - Correct typed rows and contiguous copy; optimize only if profiling makes
       it visible.
   * - P1
     - ``RMSNormalization`` v23
     - Pre-attention, pre-MLP, final, and optional Q/K normalization.
     - One-pass or stable two-pass SIMD reduction with fused scale.
   * - P1
     - ``RotaryEmbedding`` v23
     - RoPE on Q and K.
     - SIMD interleaved/half-split layouts without temporary tensors.
   * - P1
     - ``Add``, ``Mul``, ``Sigmoid``
     - Residuals and the SiLU-gated MLP.
     - SIMD; fuse ``Sigmoid(x) * x`` and the following gate multiply when the
       graph proves single use.
   * - P1
     - ``Softmax``
     - Materialized attention reference and sampling.
     - Stable row softmax; online softmax belongs to the Attention plan.
   * - P1
     - ``Reshape``, ``Transpose``, ``Slice``, ``Concat``, ``Split``
     - Head layouts and tensor ``past``/``present`` compatibility.
     - Metadata views where legal; avoid copies or fuse layout conversion into
       projection, rotary, and cache append.
   * - P2
     - ``Pow``, ``ReduceMean``, ``Sqrt``, ``Div``
     - Decomposed RMSNorm fallback.
     - Recognize and lower to the same RMSNorm plan; do not optimize four
       independent materialized passes first.
   * - P2
     - ``Sin``, ``Cos``, ``Gather``, ``Range``
     - Decomposed or dynamically generated RoPE tables.
     - Prefer constant tables or the standard RotaryEmbedding plan.
   * - P3
     - ``TopK``, ``ArgMax``, ``Multinomial``, sampling ``Softmax``
     - Token selection.
     - Keep sampling outside the model initially; optimize after decoder
       latency is competitive.
   * - Later
     - ``DynamicQuantizeLinear``, ``MatMulInteger``, ``QLinearMatMul``
     - Activation-aware INT8 alternatives.
     - Benchmark only after the weight-only INT4 path works.

The graph inventory may add small correctness adapters such as ``Cast``,
``Shape``, ``Unsqueeze``, ``Squeeze``, ``Expand`` and ``Where``. They are
implemented or delegated to ``onnx-light`` fallback as required, but they do
not change the priority order unless measured in the end-to-end profile.

INT4 MatMul is first
--------------------

Model weights dominate Qwen3 memory traffic. During batch-1 decode, each
projection has one or a few activation rows, so a generic square GEMM result
does not predict token latency. The first optimized kernel is a float-activation
by block-quantized INT4 constant-weight GEMV.

Prepared weight-only plan
~~~~~~~~~~~~~~~~~~~~~~~~~

Session preparation validates and captures:

* logical ``K`` and ``N``, nibble order, signedness, block size, and tails;
* per-block or per-channel scales and optional zero points;
* transposition and exporter packing convention;
* CPU ISA and decode/prefill micro-kernel functions;
* packed constant storage aligned for the selected kernel;
* bounded workspace and useful thread count.

The plan keeps weights compressed. Decode kernels unpack into vector registers
or small cache-resident panels, apply zero-point correction and scales, and
accumulate into float32. They do not expand the full matrix to INT8 or float.
Odd ``K``/``N`` tails, partial quantization blocks, and asymmetric zero points
have scalar differential tests.

Decode and prefill are separate algorithms:

``int4_gemv``
    Optimized first for ``M == 1`` and short speculative blocks. Partition
    output columns across workers so each weight byte is streamed once.

``int4_small_m``
    Reuses unpacked weight blocks across a few activation rows.

``int4_gemm``
    Packs activation panels and reuses weight blocks for prompt prefill.
    It lands only after decode correctness and throughput are established.

The kernel corpus uses the exact Qwen3 dimensions for every Q, K, V, O,
gate, up, down, and LM-head matrix, not only synthetic powers of two. It
reports weight bytes read, effective memory bandwidth, unpack/scale time,
first-use packing time, and steady-state throughput.

Phase Q0: freeze the executable baseline
----------------------------------------

Add an end-to-end generation driver before another generic kernel:

* Qwen3-0.6B correctness and Qwen3-4B performance models;
* float and standard QDQ INT4 graphs with identical public inputs/outputs;
* prompt lengths 1, 32, 128, 512, and 2,048;
* generated lengths 1, 32, and 128 at batch 1;
* context checkpoints at 128, 1,024, 4,096, and the model's supported limit;
* greedy decoding outside the graph for the first milestone;
* per-node and per-phase profiling with warmup and raw samples.

Report model-load latency, peak and steady resident memory, time to first
token, prefill tokens/second, median and tail decode latency, generated
tokens/second, and bytes allocated/copied per token. Compare identical graphs
where possible; when comparing standard QDQ with ONNX Runtime
``MatMulNBits``, label the graph-contract difference explicitly.

The output gate is token identity for greedy decoding plus bounded logit error
at every checked step. INT4 quality is compared with the same quantized
weights, not with the unquantized model alone.

Phase Q1: INT4 projections and LM head
--------------------------------------

Extend the pending packed-INT4 work in the
:doc:`Gemm and MatMul roadmap <2026_08_gemm_matmul>` into a complete standard
QDQ weight-only contract:

#. constant ``DequantizeLinear -> MatMul`` recognition;
#. ``int4_gemv`` for Qwen decode shapes;
#. plan-owned packed weights reused by every token;
#. ``int4_small_m`` for short prompts/speculative blocks;
#. ``int4_gemm`` for prefill;
#. LM-head and vocabulary-tail specialization.

Do not wait for Float8, every integer operator, float64 parity, or the final
generic MatMul PR10.5 gate. The Qwen path depends on existing float
correctness, constant-B planning, and the new weight-only kernels only.

Q1 exits when the full INT4 graph loads without expanded float weights, every
projection is dispatched to the packed plan, and batch-1 decode projection
time is competitive with ONNX Runtime ``MatMulNBits`` on the reference CPU.

Phase Q2: one complete Qwen block
---------------------------------

Implement the minimum standard-operator slice needed to run one decoder block:

* ``RMSNormalization`` v23 plus recognition of its decomposed pattern;
* ``RotaryEmbedding`` v23 plus a decomposed RoPE matcher;
* ``Gather`` embedding;
* SIMD Add/Mul/Sigmoid and fused SiLU-gate traversal;
* stable Softmax for the materialized reference;
* zero-copy Reshape/Slice views and measured Transpose/Concat copies.

This phase reuses only the Qwen-relevant portions of the unary and binary
elementwise roadmaps. It does not wait for their complete operator matrices.
ExpLog PR01 through PR03 are pulled forward because corrected, fast ``Exp`` is
needed by sigmoid and online softmax; ``Log`` tuning is not on the Qwen
critical path.

Q2 exits when one float and one INT4 decoder block match ONNX Runtime and no
primitive RMSNorm or RoPE decomposition materializes avoidable intermediates.

Phase Q3: narrow causal GQA
---------------------------

Pull a Qwen-specific slice ahead of the full
:doc:`Attention roadmap <2026_08_attention>`. The initial descriptor supports
only:

* batch 1;
* Qwen3 query-head/KV-head geometry;
* causal attention without arbitrary masks;
* float32 first, then the activation/cache type used by the frozen INT4 model;
* prompt prefill and ``Lq == 1`` decode;
* standard tensor ``past_key``/``past_value`` and
  ``present_key``/``present_value`` for the correctness fallback.

The graph optimizer recognizes the standard
``MatMul -> scale/mask -> Softmax -> MatMul`` attention pattern and lowers it
to an internal GQA plan. It does not expose a ``com.microsoft`` operator from
this repository. GQA maps query-head groups onto shared K/V heads without
physically repeating K or V.

The optimized path uses blocked online softmax and never materializes the
complete attention-score matrix. Prefill uses query blocks; decode uses a
single-query streaming kernel. General masks, arbitrary layouts, and broad
type parity remain in the generic Attention roadmap.

Q3 exits when the narrow fused path and the standard materialized fallback
produce equivalent outputs for the frozen models and improve end-to-end
prefill or decode without increasing peak memory.

Phase Q4: persistent contiguous KV cache
----------------------------------------

Tensor ``Concat`` of all past K/V is acceptable only as a correctness
baseline. It copies work proportional to context length at every generated
token and prevents competitive long-context decode.

Execute the minimum C0-C2 slice of the
:doc:`Persistent KV Cache roadmap <2026_08_kv_cache>`:

#. one request-owned persistent runtime cache value;
#. contiguous K/V storage with capacity growth;
#. append of only the new rotated K/V token;
#. direct block iteration by the Q3 decode kernel;
#. explicit reset and session/request isolation;
#. import/export adapters only when standard tensor cache outputs are visible.

Do not block this milestone on paged storage, beam reorder, speculative
truncate, sliding windows, INT8/INT4 KV compression, or observable cache
format APIs. Append bytes must be proportional to new tokens, and a generated
token must not allocate or copy the complete past.

Q4 is the first fast-Qwen3 milestone. It exits when decode latency is measured
over growing context, no hidden full-cache gather occurs, and token outputs
remain identical to the tensor-cache reference.

Phase Q5: scheduling and fusion
-------------------------------

Execute Runtime PR01, PR02, and PR04 from the
:doc:`Runtime Execution Controls roadmap
<2026_08_runtime_execution_controls>` before final tuning. Registered Qwen
kernels share the session executor; standalone entry points are serial.

Tune distinct policies for:

* INT4 GEMV decode, usually partitioned over output channels;
* INT4 GEMM prefill, partitioned over row and output panels;
* RMSNorm and elementwise passes, kept serial below measured sizes;
* prefill attention, partitioned by head and query block;
* decode attention, partitioned by KV/query-head group only when outer work is
  insufficient.

Only then measure graph-level fusions:

* RMSNorm into packed projection input;
* Q/K projection layout, RoPE, and cache append;
* gate projection, SiLU, and gate/up multiplication;
* residual Add with the following RMSNorm;
* final RMSNorm, LM-head GEMV, and greedy ArgMax.

Each unfused standard graph remains a correctness fallback. A fusion lands
only when it removes measured memory traffic and preserves graph-visible
values.

Deferred work
-------------

The following work is explicitly not required for the first fast dense
Qwen3 result:

* SVM, tree, convolution, and broad unary/binary parity roadmaps;
* Float8, float64, and generic integer MatMul completion;
* generic Attention masks and every MHA layout;
* paged/quantized KV cache, continuous batching, beam search, and speculative
  decoding;
* in-graph sampling and full generation control flow;
* Qwen3 MoE, VL, multimodal rotary, and later hybrid recurrent layers.

These items return to priority only after Q4 profiling shows that their
corresponding functionality is needed.

``com.microsoft`` operator inventory
------------------------------------

These operators are implemented in the separate repository requested for
Microsoft-domain compatibility. They are listed here so the export and
integration contract is complete; none is registered by
``onnx-light-cpu``.

.. list-table::
   :header-rows: 1
   :widths: 31 12 39 18

   * - Operator
     - Priority
     - Qwen role
     - Standard/internal counterpart here
   * - ``com.microsoft.MatMulNBits`` v1
     - C0
     - Primary weight-only 2/4/8-bit projection format used by ONNX Runtime.
     - QDQ INT4 matcher plus internal packed MatMul plan.
   * - ``com.microsoft.MatMulBnb4`` v1
     - C0 if present
     - Alternative bitsandbytes-style 4-bit matrix format.
     - Format adapter into the same internal weight-only plan.
   * - ``com.microsoft.GroupQueryAttention`` v1
     - C1
     - Fused GQA, RoPE options, and tensor KV-cache path used by Qwen exports.
     - Standard attention-pattern matcher plus internal GQA/cache plan.
   * - ``com.microsoft.RotaryEmbedding`` v1
     - C1
     - Legacy/contrib RoPE representation.
     - Standard ``RotaryEmbedding`` v23.
   * - ``com.microsoft.SkipSimplifiedLayerNormalization`` v1
     - C2
     - Residual Add fused with RMS-style normalization.
     - Standard Add plus RMSNormalization matcher.
   * - ``com.microsoft.Attention`` v1
     - C2
     - Older fused self-attention representation.
     - Standard attention matcher.
   * - ``com.microsoft.MultiHeadAttention`` v1
     - C2
     - Fused dense MHA alternative.
     - Generic Attention roadmap.
   * - ``com.microsoft.QAttention`` v1
     - C3
     - Legacy activation-quantized attention alternative; not the preferred
       Qwen3 INT4 path.
     - No first-milestone counterpart.
   * - ``com.microsoft.Sampling`` v1
     - C3
     - In-graph generation and sampling control.
     - External sampler first; standard TopK/ArgMax later.
   * - ``com.microsoft.PagedAttention`` v1
     - Conditional
     - Continuous batching and paged KV storage. Current ONNX Runtime sources
       register CUDA/WebGPU implementations, not a CPU compute kernel.
     - Later paged KV-cache phase.
   * - ``com.microsoft.MoE`` and ``com.microsoft.QMoE`` v1
     - Conditional
     - Dense and quantized expert routing for Qwen3 MoE variants.
     - Separate MoE roadmap after dense Qwen3.
   * - ``com.microsoft.MRotaryEmbedding`` v1
     - Conditional
     - Multimodal rotary positions used by VL variants.
     - Separate Qwen3-VL scope.
   * - ``com.microsoft.SparseAttention`` v1
     - Conditional
     - Sparse/sliding attention variants.
     - Generic sparse Attention work.
   * - ``com.microsoft.GatedRMSNorm`` and
       ``com.microsoft.CausalConvWithState`` v1
     - Conditional
     - Later hybrid Qwen-family recurrent/linear-attention blocks.
     - Out of the dense Qwen3 scope.

ONNX Runtime also carries experimental ``SimplifiedLayerNormalization`` in the
default ONNX domain at version 1. It is not the standard ONNX
``RMSNormalization`` v23 schema and should be treated as another compatibility
adapter by the separate repository.

Benchmark gates
---------------

.. list-table::
   :header-rows: 1
   :widths: 23 77

   * - Gate
     - Exit criterion
   * - Graph
     - Every executed node is reported with its backend; no silent reference
       fallback occurs in a timed priority region.
   * - INT4 memory
     - Full float weights are never materialized. Persistent packed weights
       plus scales/metadata stay within a documented bound over serialized
       INT4 bytes.
   * - Correctness
     - Float and INT4 logits satisfy explicit tolerances; greedy token ids
       match at every checked generation step.
   * - Decode structure
     - One token performs no full-weight conversion, complete score-matrix
       allocation, or full-KV copy.
   * - First milestone
     - Qwen3-0.6B and Qwen3-4B complete prefill and 128-token greedy decode at
       batch 1 with bounded memory and stable repeated latency.
   * - Performance
     - Q4 publishes TTFT and tokens/second against ONNX Runtime on the same
       CPU, graph contract, threads, affinity, and prompt/context matrix.
       Initial acceptance requires no priority case below ``0.9x``; final
       tuning targets at least ``1.0x`` median.

Pull-request sequence
---------------------

.. list-table::
   :header-rows: 1
   :widths: 10 24 43 13 10

   * - PR
     - Deliverable
     - Merge criterion
     - Depends on
     - Status
   * - Qwen PR01
     - Frozen graphs and generation benchmark.
     - Float/QDQ graph digests, operator inventories, correctness tokens, TTFT,
       decode, memory, and per-node profiles are reproducible.
     - None
     - Pending
   * - Qwen PR02
     - Standard QDQ INT4 plan and decode GEMV.
     - Constant QDQ is captured once; every decode projection keeps weights
       compressed and passes exact packing/tail plus model-logit tests.
     - Qwen PR01; Gemm PR09.6
     - Pending
   * - Qwen PR03
     - INT4 small-M and prefill GEMM.
     - Qwen prompt shapes reuse packed weights and improve TTFT without
       regressing decode or exceeding the memory bound.
     - Qwen PR02
     - Pending
   * - Qwen PR04
     - Qwen block operator slice.
     - Standard RMSNorm, RoPE, Gather, SiLU gate, Softmax, and layout paths run
       one complete float and INT4 block without avoidable intermediates.
     - Qwen PR02; ExpLog PR01-PR03
     - Pending
   * - Qwen PR05
     - Narrow standard-pattern GQA.
     - Batch-1 causal prefill and decode use online softmax and zero-copy
       query/KV-head grouping, with a tensor-cache fallback.
     - Qwen PR04
     - Pending
   * - Qwen PR06
     - Persistent contiguous KV cache.
     - Append work is proportional to new tokens; decode reads the cache
       directly and produces no full present-cache copy or gather.
     - Qwen PR05; KV C0-C2
     - Pending
   * - Qwen PR07
     - Session executor and measured fusion.
     - Runtime PR01/02/04 integration prevents nested pools; only measured
       Qwen fusions land, with inspectable scheduling decisions.
     - Qwen PR06
     - Pending
   * - Qwen PR08
     - First dense-Qwen3 performance gate.
     - Both reference models pass correctness/memory gates, every priority case
       is at least ``0.9x`` ONNX Runtime, and median performance reaches
       ``1.0x`` or remains open with published bottleneck evidence.
     - Qwen PR01-PR07
     - Pending

Qwen PR08 closes the first fast dense-Qwen3 milestone. Later work starts from
its end-to-end profile rather than from an operator checklist.

Reference evidence
------------------

The ordering follows the current ONNX and ONNX Runtime contracts:

* ONNX opset 23 defines
  `RMSNormalization and RotaryEmbedding
  <https://github.com/onnx/onnx/blob/main/onnx/defs/operator_sets.h>`_;
* ONNX Runtime's CPU
  `MatMulNBits
  <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cpu/quantization/matmul_nbits.cc>`_
  is the direct weight-only comparison;
* its CPU
  `GroupQueryAttention
  <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cpu/bert/group_query_attention.cc>`_
  confirms the GQA/KV-cache execution target;
* the ONNX Runtime GenAI
  `Qwen builder
  <https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builders/qwen.py>`_
  distinguishes full-attention layers and their KV-cache handling.
