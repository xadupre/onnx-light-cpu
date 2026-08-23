Qwen3 CPU Inference Critical Path
=================================

:Date: 2026-08

**planned**

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

Frozen model contract
---------------------

Qwen PR01 pins immutable Hugging Face model and tokenizer commit hashes; model
names alone are not reproducible inputs. The initial revisions must retain the
official dense configurations:

.. list-table::
   :header-rows: 1
   :widths: 18 10 12 12 10 10 10 18

   * - Model
     - Layers
     - Hidden
     - Intermediate
     - Q heads
     - KV heads
     - Head size
     - Vocabulary
   * - ``Qwen/Qwen3-0.6B``
     - 28
     - 1,024
     - 3,072
     - 16
     - 8
     - 128
     - 151,936
   * - ``Qwen/Qwen3-4B``
     - 36
     - 2,560
     - 9,728
     - 32
     - 8
     - 128
     - 151,936

Both models use BF16 activations and weights in the unquantized artifact,
Q/K RMS normalization, SwiGLU, tied token-embedding/LM-head weights, full
half-split RoPE with theta 1,000,000, no rope scaling or sliding window, and a
40,960-token configured context. A revision that changes any of these values
is a new benchmark artifact, not an update to an existing result.

The large context is an opt-in structure and memory check. Shared CI uses a
deterministic synthetic two-layer fixture with the same head size, GQA ratio,
Q/K normalization, tied-weight aliasing, cache boundary, and operator forms.
The existing four-layer Qwen3-like fixture in ``onnx-light`` remains useful
for shape and memory planning, but it is not an executable or numerical Qwen
baseline and does not satisfy Qwen PR01.

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
     - Completed Gemm/MatMul PR09.6 plus Qwen PR02-PR03
     - Standard QDQ INT4 recognition, packed decode GEMV, small-M, then prefill
       GEMM for exact Qwen shapes.
     - Float8, full integer parity, float64, and generic PR10.5.
   * - 3
     - Completed ExpLog PR01-PR03 plus Qwen PR04
     - Correct/fast Exp, RMSNorm, RoPE, Gather, SiLU gate, Softmax, and layout
       operations needed by one block.
     - Non-Qwen unary/binary operator matrices and graph fusions.
   * - 4
     - Attention PR11-PR14 plus Qwen PR05 integration
     - Reuse the shared materialized and online Attention engines for batch-1
       causal GQA, prefill, and single-token decode.
     - The generic Attention PR15 performance gate.
   * - 5
     - Corrected KV-cache C0-C2 plus Qwen PR06a-PR06b
     - Request-owned contiguous cache and cache-aware streaming decode.
     - Paging, beam/speculative operations, and KV quantization.
   * - Completed prerequisite
     - Runtime PR01, PR02, and PR04
     - Reuse the delivered session executor and no-nested-pools contract.
     - No new runtime-executor integration in the Qwen sequence.
   * - 6
     - Qwen PR07-PR08
     - End-to-end scheduling, measured fusions, and parity gate.
     - Any optimization absent from the profile.

Export contract
---------------

Two equivalent standard-ONNX graphs are retained:

``qwen3-float``
    The primary correctness graph with ordinary constant BF16 weights and
    activations at ONNX opset 23. An FP32 diagnostic export is retained for
    numerical localization, but it is not a separate performance target.

``qwen3-int4-qdq``
    A weight-only BF16 graph using one canonical encoding. Each non-embedding
    projection is a logical ``UINT4[K,N]`` initializer in ONNX row-major,
    low-nibble-first storage. ``DequantizeLinear`` uses ``axis=0``,
    ``block_size=32``, BF16 scales shaped ``[ceil(K/32),N]``, explicit UINT4
    zero points of the same shape, and BF16 output feeding ``MatMul``.

    The tied embedding/LM-head weight is stored once as logical
    ``UINT4[vocabulary,hidden]``. Its ``DequantizeLinear`` uses ``axis=1`` and
    scale/zero-point shape ``[vocabulary,ceil(hidden/32)]``. The standard graph
    feeds that dequantized value to ``Gather`` for embeddings and through
    ``Transpose`` to the LM-head ``MatMul``. The optimized plan dequantizes only
    selected embedding rows and streams the same packed initializer in
    transposed access order for the LM-head; it does not clone the serialized
    initializer.

    Qwen dimensions have no partial quantization block. Partial K blocks, odd
    N, and packed-byte tails remain mandatory synthetic tests. A transposed or
    differently blocked exporter output is a distinct artifact and must be
    normalized before matching.

    The runtime recognizes only this exact constant pattern initially and
    passes packed weights, scales, and zero points directly to an internal
    BF16-activation weight-only plan. It never materializes a complete BF16 or
    float weight tensor. The portable ``DequantizeLinear`` path must accept
    BF16 scales/output before Qwen PR02 exits. Qwen PR01 may record that
    current gap but makes no production-kernel change.

Opset 23 is preferred because it provides the standard
``RMSNormalization`` and ``RotaryEmbedding`` schemas. An exporter that emits
their primitive decompositions remains supported through graph matching, but
those decompositions are not the preferred optimized representation.

Every benchmark artifact records the exact model revision, exporter revision,
opsets, graph digest, quantization block size, symmetric/asymmetric encoding,
weight packing order, tokenizer, prompt tokens, generated token ids, and
runtime options. It also records initializer aliasing: the tied embedding and
LM-head storage may have separate access plans but must not duplicate the
serialized weight or an unbounded expanded representation. No result from one
INT4 encoding is attributed to another.

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
     - Recognize the tied quantized initializer and dequantize only selected
       rows; retain correct typed contiguous copy for float weights.
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

The implemented PR09.6 integer kernel supplies tested nibble decoding, panel
packing, tail handling, and dot-product primitives. It does not implement
BF16/FP32 activations multiplied by scaled INT4 weights and is therefore prior
art, not the Qwen weight-only plan.

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

The kernel corpus uses the exact Qwen3 dimensions for embedding, every Q, K,
V, O, gate, up, down, and LM-head matrix, not only synthetic powers of two.
It reports weight bytes read, effective memory bandwidth, unpack/scale time,
first-use packing time, and steady-state throughput. Tests prove that the
embedding row path and transposed LM-head path reference one packed
initializer.

Phase Q0: freeze the executable baseline
----------------------------------------

Add an end-to-end generation driver before another generic kernel:

* Qwen3-0.6B correctness and Qwen3-4B performance models;
* float and standard QDQ INT4 graphs with identical public inputs/outputs;
* prompt lengths 1, 32, 128, 512, and 2,048;
* generated lengths 1, 32, and 128 at batch 1;
* context checkpoints at 128, 1,024, and 4,096, with 40,960 opt-in;
* greedy decoding outside the graph for the first milestone;
* per-node and per-phase profiling with warmup and raw samples;
* a backend manifest for every executed node, including explicit portable
  fallback outside timed priority regions.

Qwen PR01 also registers lazy backend cases through the standard
``onnx-light-cpu`` collector. Case names encode model, float/QDQ contract,
prefill/decode, prompt/context/generated lengths, and thread policy. Shared CI
executes the synthetic fixture in correctness and ``TestMode::BENCHMARK``
modes. Real Qwen3-0.6B and Qwen3-4B cases are opt-in, download pinned
revisions into an external cache, and publish through the same benchmark
runner and dashboard; model weights are not committed to this repository.
Metadata tests verify unique names, lazy construction, graph digests, tensor
types and sizes, exact projection shapes, and the opt-in large-context cases.

Report model-load latency, first-use preparation latency, peak and steady
resident memory, time to first
token, prefill tokens/second, median and tail decode latency, generated
tokens/second, and bytes allocated/copied per token. Compare identical graphs
under two separately labelled contracts:

``standard-contract``
    The identical standard ONNX float or QDQ graph runs in both runtimes. This
    is the correctness and graph-coverage comparison. Qwen PR01 establishes
    the graph and reference outputs even if the current onnx-light runtime
    reports unsupported BF16 blocked dequantization; Qwen PR02 must make the
    QDQ graph executable before publishing standard-contract timings.

``native-performance``
    The same quantized values, scales, zero points, prompts, and cache
    semantics are converted to ONNX Runtime ``MatMulNBits`` and
    ``GroupQueryAttention`` where required. This is the primary product-level
    performance comparison; graph conversion time and persistent converted
    bytes are reported. Results are never presented as identical-graph speed.

Qwen PR01 pins the reference CPU, OS, compiler, ONNX Runtime version, power
mode, NUMA placement, and compact affinity. It publishes one-thread and
physical-core policies; both runtimes receive the same admitted thread count,
affinity, warmup, and alternating sample order. The priority performance
matrix is Qwen3-4B with prompt lengths 32 and 512, context checkpoints 128,
1,024, and 4,096, and 128 generated tokens.

Correctness uses teacher-forced prefixes so one unstable greedy choice does
not hide later numerical errors. PR01 freezes explicit per-type absolute and
relative logit tolerances from the reference implementation. The selected
token must be identical when the reference top-1 margin exceeds twice the
measured logit error bound; near ties may select any token within that bound.
Stable reference prompts additionally retain exact greedy-token regression
sequences. INT4 quality is compared with the same quantized weights, scales,
and zero points, not with the unquantized model alone.

Phase Q1: INT4 projections and LM head
--------------------------------------

Extend the completed packed-integer INT4 work in the
:doc:`Gemm and MatMul roadmap <2026_08_gemm_matmul>` into a complete standard
QDQ weight-only contract:

#. constant ``DequantizeLinear -> MatMul`` recognition;
#. ``int4_gemv`` for Qwen decode shapes;
#. plan-owned packed weights reused by every token;
#. tied quantized ``Gather`` and transposed LM-head access without duplication;
#. ``int4_small_m`` for short prompts/speculative blocks;
#. ``int4_gemm`` for prefill;
#. LM-head and vocabulary-tail specialization.

Do not wait for Float8, every integer operator, float64 parity, or the final
generic MatMul PR10.5 gate. The Qwen path depends on existing float
correctness, constant-B planning, and the new weight-only kernels only.

Q1 exits when the full INT4 graph loads without expanded BF16/float weights, every
projection is dispatched to the packed plan, and batch-1 decode projection
time is competitive with ONNX Runtime ``MatMulNBits`` on the frozen reference
CPU. Persistent packed storage, including tied-weight plans, stays within
``1.25x`` the serialized UINT4 weights plus serialized scales and zero points.

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
ExpLog PR01 through PR03 are complete and provide the corrected, fast ``Exp``
needed by sigmoid and online softmax. Qwen PR04 adds only missing
Qwen-specific adapters and optimized traversals. Existing portable
``RMSNormalization``, ``RotaryEmbedding``, and ``DequantizeLinear`` kernels in
``onnx-light`` remain the differential fallback; their existence does not
count as optimized CPU dispatch.

Q2 exits when one float and one INT4 decoder block match ONNX Runtime and no
primitive RMSNorm or RoPE decomposition materializes avoidable intermediates.
The standard RoPE nodes use rank-4 ``[batch,heads,sequence,128]`` inputs,
``interleaved=0``, ``rotary_embedding_dim=128``, and explicit position ids;
Q and K use their respective frozen head counts.

Phase Q3: narrow causal GQA
---------------------------

Qwen PR05 does not create a second attention descriptor, planner, graph
matcher, or compute engine. It integrates the frozen graph with the shared
``AttentionDescriptor``, per-invocation ``AttentionPlan``, materialized
fallback, and online engine delivered by Attention PR11 through PR14. The
Qwen priority subset is:

* batch 1;
* Qwen3 query-head/KV-head geometry;
* causal attention without arbitrary masks;
* BF16 for the frozen models, with FP32 as the diagnostic fallback;
* prompt prefill and ``Lq == 1`` decode;
* standard tensor ``past_key``/``past_value`` and
  ``present_key``/``present_value`` for the correctness fallback.

The existing ``onnx-light`` graph optimizer recognizes the standard
``MatMul -> scale/mask -> Softmax -> MatMul`` attention pattern and emits
standard ``ai.onnx::Attention``. ``onnx-light-cpu`` dispatches that node to the
shared engine; Qwen code does not lower it to a private GQA operator. It does
not expose a ``com.microsoft`` operator from this repository. GQA maps
query-head groups onto shared K/V heads without physically repeating K or V.

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

Execute C0 through C2 from the
:doc:`Persistent KV Cache roadmap <2026_08_kv_cache>`:

#. one request-owned persistent runtime cache value;
#. contiguous K/V storage with capacity growth;
#. append of only the new rotated K/V token;
#. direct block iteration by the Q3 decode kernel;
#. explicit reset and session/request isolation;
#. import/export adapters only when standard tensor cache outputs are visible.

This cross-repository work is split at the ownership boundary. Qwen PR06a in
``onnx-light`` delivers the request-owned cache handle, persistent lifetime,
reset/isolation semantics, execution-plan actions, safe tensor-cache rewrite,
and backend-neutral import/export interfaces. Qwen PR06b in
``onnx-light-cpu`` delivers contiguous CPU allocation and growth, append/view/
export kernels, the block iterator, and direct consumption by shared streaming
Attention. PR06b is the implementation vehicle for the Qwen-critical C0-C2
slice; the later paged and quantized parts of the broader KV roadmap remain
discussed.

Do not block this milestone on paged storage, beam reorder, speculative
truncate, sliding windows, INT8/INT4 KV compression, or observable cache
format APIs. Append bytes must be proportional to new tokens, and a generated
token must not allocate or copy the complete past.

Q4 is the first fast-Qwen3 milestone. It exits when decode latency is measured
over growing context, no hidden full-cache gather occurs, and token outputs
remain identical to the tensor-cache reference.

Phase Q5: scheduling and fusion
-------------------------------

Runtime PR01, PR02, and PR04 from the
:doc:`Runtime Execution Controls roadmap
<2026_08_runtime_execution_controls>` are complete. Registered Qwen kernels
must reuse the delivered session executor and nesting guard; standalone entry
points remain serial. Qwen PR07 tunes participants and fusions but adds no
executor or private pool.

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
     - Full BF16/float weights are never materialized. Persistent packed
       weights plus scales/metadata stay within ``1.25x`` serialized UINT4
       weights plus serialized scales/zero points, with tied storage counted
       once.
   * - Correctness
     - Float and INT4 teacher-forced logits satisfy the frozen tolerances.
       Greedy ids match for stable top-1 margins; documented near ties satisfy
       the bounded candidate rule. Stable prompts match exact token sequences.
   * - Decode structure
     - One token performs no full-weight conversion, complete score-matrix
       allocation, or full-KV copy.
   * - First milestone
     - Qwen3-0.6B and Qwen3-4B complete prefill and 128-token greedy decode at
       batch 1 with bounded memory and stable repeated latency.
   * - Performance
     - Q4 publishes both ``standard-contract`` and ``native-performance`` TTFT
       and tokens/second on the frozen CPU, threads, affinity, and
       prompt/context matrix. Against the native-performance baseline, initial
       acceptance requires no priority case below ``0.9x`` and final tuning
       reaches at least ``1.0x`` median.

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
     - Pinned model/exporter/tokenizer revisions, the canonical QDQ contract,
       graph digests, lazy backend cases, correctness rules, both comparator
       contracts, TTFT, decode, memory, and per-node profiles are reproducible.
     - None
     - Pending
   * - Qwen PR02
     - Standard QDQ INT4 plan and decode GEMV.
     - Constant QDQ is captured once; every decode projection keeps weights
       compressed and passes exact packing/tail plus model-logit tests.
     - Qwen PR01; reuses completed Gemm PR09.6 primitives
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
     - Qwen PR02; completed ExpLog PR01-PR03
     - Pending
   * - Qwen PR05
     - Frozen-graph integration with shared Attention.
     - Standard ``Attention`` dispatches to the shared descriptor, planner,
       materialized fallback, and online engine. Batch-1 causal prefill/decode
       use zero-copy query/KV-head grouping; no Qwen-only engine is introduced.
     - Qwen PR04; Attention PR14 / #391
     - Pending
   * - Qwen PR06a
     - Backend-neutral persistent cache in ``onnx-light``.
     - Request-owned state survives invocation cleanup, reset and isolation are
       explicit, safe tensor-cache rewrites preserve observable outputs, and
       execution planning accounts for persistent lifetime and bytes.
     - Qwen PR05; KV C0
     - Pending
   * - Qwen PR06b
     - Contiguous CPU cache and cache-aware decode in ``onnx-light-cpu``.
     - Append work is proportional to new tokens; decode reads the cache
       directly through the shared Attention block iterator and produces no
       full present-cache copy or gather.
     - Qwen PR06a; KV C1-C2
     - Pending
   * - Qwen PR07
     - Participant tuning and measured fusion.
     - The completed Runtime PR01/02/04 executor is reused without a private
       pool; only measured Qwen fusions land, with inspectable scheduling
       decisions.
     - Qwen PR06b
     - Pending
   * - Qwen PR08
     - First dense-Qwen3 performance gate.
     - Both reference models pass correctness/memory gates, every priority case
       is at least ``0.9x`` ONNX Runtime, and median performance reaches
       ``1.0x``. If the target is missed, bottleneck evidence is published and
       PR08 remains open.
     - Qwen PR01-PR07, including PR06a and PR06b
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
* the pinned model contracts originate from the official
  `Qwen3-0.6B configuration
  <https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json>`_ and
  `Qwen3-4B configuration
  <https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json>`_;
* the ONNX Runtime GenAI
  `Qwen builder
  <https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builders/qwen.py>`_
  distinguishes full-attention layers and their KV-cache handling.
