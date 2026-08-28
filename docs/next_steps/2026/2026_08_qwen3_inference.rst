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

The first functional target is the concrete 28-layer INT4 artifact inventoried
below because it keeps the graph contract fixed and exposes every immediate
execution blocker. Qwen3-0.6B remains the short correctness reference and
Qwen3-4B remains the first performance and memory target.
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

Audited ORT INT4 artifact
-------------------------

The first executable target is the local artifact
``/home/xadupre/examples/models/qwen3-8b-cpu-int4/model.onnx``. The absolute
path is a development input, not a portable model identity. Qwen PR01 must
copy the graph and external data into the external model cache under immutable
artifact metadata before CI or published benchmarks use it.

The audited graph has SHA-256
``f6745c77935bc5640751b3a961246e77a9bc515055dc8cba08d0f2ca5ba183a3``.
It uses IR version 10, ``ai.onnx`` opset 26 and ``com.microsoft`` opset 1,
contains 519 nodes and 398 initializers, and references
``model.onnx.data`` with 1,016,070,144 physical bytes and SHA-256
``43ea4a553800251cd2f74505a7a4140f4e19be78bad27e343425ab9dd420608c``.
Its graph dimensions identify 28 decoder layers, hidden size 1,024,
intermediate size 3,072, 16 query heads, 8 KV heads, head size 128, and
vocabulary size 151,936. These dimensions do not describe an 8B model despite
the development directory name; model metadata and tokenizer revision must
determine the final published artifact name.

The public contract is FP32: ``input_ids`` and ``attention_mask`` are INT64,
56 past K/V inputs and 56 present K/V outputs use FP32, and ``logits`` uses
FP32. Rotary caches have shape ``[40960,64]``. The embedding initializer is an
unquantized FP32 ``[151936,1024]`` matrix; only the 141 projection nodes use
INT4 ``MatMulNBits``. This means the audited artifact is not equivalent to the
fully quantized tied-embedding QDQ contract described below.

Exact kernel inventory
~~~~~~~~~~~~~~~~~~~~~~

The table is exhaustive for this graph. ``Registered`` means a kernel is present
in the current ``onnx-light-cpu`` registration inventory, not merely that an
ONNX reference implementation may exist. All twelve rows marked ``Missing``
must receive either a CPU kernel or an explicit, tested ``onnx-light`` core
execution path before the artifact can run without an untracked fallback.

.. list-table::
   :header-rows: 1
   :widths: 26 8 15 51

   * - Operator
     - Nodes
     - Current status
     - Exact required contract
   * - ``ai.onnx::Mul``
     - 56
     - Registered
     - FP32 tensor multiplication for SiLU and gated MLP products.
   * - ``ai.onnx::Sub``
     - 1
     - Registered
     - INT64 scalar/vector subtraction in the attention-mask prelude; verify
       opset-26 dispatch against the registered v14+ kernel.
   * - ``ai.onnx::Constant``
     - 6
     - Missing CPU registration
     - Materialize the six INT64 shape/axis constants. This should normally be
       a prepared graph value in ``onnx-light`` core rather than a timed CPU
       compute kernel, but its execution ownership must be explicit.
   * - ``ai.onnx::Cast``
     - 2
     - Missing
     - INT64 to INT32 conversion for ``seqlens_k`` and
       ``total_sequence_length`` consumed by GQA.
   * - ``ai.onnx::Gather``
     - 2
     - Missing
     - FP32 embedding lookup with INT64 token ids on axis 0, plus one INT64
       scalar gather from the attention-mask shape.
   * - ``ai.onnx::ReduceSum``
     - 1
     - Missing
     - INT64 reduction over an axes tensor input with ``keepdims=0`` for the
       attention-mask sequence length.
   * - ``ai.onnx::Reshape``
     - 112
     - Missing
     - FP32 reshape from an INT64 shape tensor, including ``0`` and ``-1``
       semantics; use a metadata view whenever the input is contiguous.
   * - ``ai.onnx::Shape``
     - 1
     - Missing
     - Produce the INT64 shape of the rank-2 ``attention_mask`` input.
   * - ``ai.onnx::Sigmoid``
     - 28
     - Missing
     - FP32 sigmoid for ``x * sigmoid(x)``. Reuse the existing Exp primitives
       and retain an unfused kernel before adding a SiLU/gate fusion.
   * - ``ai.onnx::SimplifiedLayerNormalization``
     - 57
     - Missing compatibility adapter
     - FP32 RMS-style normalization with ``axis=-1``, ``epsilon=1e-6`` and
       ``stash_type=1``. Lower to the existing ``RMSNormalization`` engine
       only after proving identical output and accumulation semantics.
   * - ``ai.onnx::Split``
     - 28
     - Missing
     - Split FP32 QKV projections on ``axis=-1`` using the INT64 input
       ``[2048,1024,1024]`` and produce three views where legal.
   * - ``com.microsoft::MatMulNBits`` v1
     - 141
     - Missing and blocking
     - FP32 activation, packed UINT8 storage containing 4-bit weights, FP32
       scales, ``block_size=32`` and ``accuracy_level=4``. Required shapes are
       ``1024x3072`` (56), ``1024x4096`` (28), ``2048x1024`` (28),
       ``3072x1024`` (28), and ``1024x151936`` (1). The implementation must
       keep weights compressed and cover GEMV decode plus GEMM prefill.
   * - ``com.microsoft::GroupQueryAttention`` v1
     - 28
     - Missing and blocking
     - FP32 causal GQA with 16 query heads, 8 KV heads, head size 128,
       ``scale=1/sqrt(128)``, integrated half-split RoPE, external FP32
       cos/sin caches, tensor past/present K/V, and three outputs. Adapt this
       contract to the shared Attention planner instead of creating a second
       attention engine.
   * - ``com.microsoft::SkipSimplifiedLayerNormalization`` v1
     - 56
     - Missing and blocking
     - Fused FP32 residual addition and RMS-style normalization with
       ``epsilon=1e-6``. Fifty-five nodes expose the normalized value and the
       residual through sparse optional outputs; the final node exposes only
       the normalized value. Lower to Add plus the shared normalization engine
       first, then add a measured fused traversal.

The minimum functional implementation order for this exact artifact is:
prepared ``Constant``/``Shape`` plus ``Gather``/``Cast``/``ReduceSum`` for
inputs; ``MatMulNBits``; ``Split``/``Reshape``;
``SimplifiedLayerNormalization``; ``GroupQueryAttention``;
``Sigmoid``; and ``SkipSimplifiedLayerNormalization``. ``Mul`` and ``Sub``
need model-level coverage but no new kernel. ``GroupQueryAttention`` already
owns rotary embedding, softmax, mask handling, and tensor KV concatenation in
this graph, so separate ``RotaryEmbedding``, ``Softmax``, ``Concat`` and
``Slice`` kernels are not execution prerequisites for this artifact.

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
     - Direct ``MatMulNBits`` v1 adapter for the audited artifact, shared
       packed decode GEMV, small-M, then prefill GEMM; retain standard QDQ as
       the portable serialization.
     - Float8, full integer parity, float64, and generic PR10.5.
   * - 3
     - Completed ExpLog PR01-PR03 plus Qwen PR04
     - The exact standard-operator inventory above: Gather, Cast, ReduceSum,
       Shape, Reshape, Split, Sigmoid, SimplifiedLayerNormalization, and the
       SkipSimplifiedLayerNormalization adapter.
     - Non-Qwen unary/binary operator matrices and graph fusions.
   * - 4
     - Attention PR11-PR14 plus Qwen PR05 integration
     - Adapt the audited ``GroupQueryAttention`` contract to the shared
       materialized and online Attention engines for batch-1 causal GQA,
       integrated RoPE, tensor-cache prefill, and single-token decode.
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

Three graph contracts are retained. The audited native graph is executable
first; the two standard-ONNX graphs provide portability and differential
correctness:

``qwen3-ort-int4``
    The audited opset-26 graph described above. It is the immediate execution
    target and uses ``MatMulNBits``, ``GroupQueryAttention`` and
    ``SkipSimplifiedLayerNormalization`` from ``com.microsoft`` plus
    experimental ``ai.onnx::SimplifiedLayerNormalization``. Each adapter must
    lower to the same internal compute plans as the standard contracts so
    native and portable support do not fork into separate engines.

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
QDQ weight-only contract and a direct adapter for the audited graph:

#. ``com.microsoft::MatMulNBits`` v1 validation and lowering for the exact
   three-input, four-bit, block-32 contract;
#. constant ``DequantizeLinear -> MatMul`` recognition into the same plan;
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

* ``Gather`` for FP32 embeddings and the INT64 shape scalar;
* INT64 ``Shape``, ``ReduceSum``, ``Sub`` and INT64-to-INT32 ``Cast`` for the
  attention-mask prelude;
* ``SimplifiedLayerNormalization`` lowered to the RMSNormalization engine;
* zero-copy FP32 ``Reshape`` and ``Split`` views where legal;
* SIMD FP32 Mul/Sigmoid and fused SiLU-gate traversal;
* ``SkipSimplifiedLayerNormalization`` lowered to Add plus the shared
  normalization engine, including sparse optional outputs.

Standard ``RMSNormalization`` v23, ``RotaryEmbedding`` v23, stable Softmax,
Slice, Transpose and Concat remain required by the portable graph contract,
but they are not blockers for the audited native graph because GQA contains
RoPE, attention softmax, and tensor-cache concatenation.

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
matcher, or compute engine. It first adapts the audited
``com.microsoft::GroupQueryAttention`` node, then integrates the portable
standard graph with the shared
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
shared engine. The native compatibility layer validates
``GroupQueryAttention`` attributes and inputs and builds the same descriptor;
it is not a private GQA compute engine. GQA maps query-head groups onto shared
K/V heads without physically repeating K or V.

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

No Microsoft-domain operator in this table is currently registered by
``onnx-light-cpu``. The audited artifact makes ``MatMulNBits``,
``GroupQueryAttention`` and ``SkipSimplifiedLayerNormalization`` immediate
execution dependencies rather than optional comparator formats. Their
compatibility registrations may remain in the dedicated Microsoft-domain
repository, but they must lower to the shared packed MatMul, Attention, Add,
and RMSNormalization plans owned here. The remaining rows stay conditional
until a frozen graph contains them.

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
adapter that lowers to the shared normalization engine.

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
     - The audited native artifact and portable graphs have pinned
       model/exporter/tokenizer revisions, external-data digests, exhaustive
       kernel inventories, lazy backend cases, correctness rules, comparator
       contracts, TTFT, decode, memory, and per-node profiles.
     - None
     - Pending
   * - Qwen PR02
     - Native MatMulNBits and standard QDQ INT4 plan plus decode GEMV.
     - ``MatMulNBits`` and constant QDQ lower to one prepared plan; every
       decode projection keeps weights compressed and passes exact
       packing/tail plus model-logit tests.
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
     - Every missing standard-domain row in the audited inventory plus the
       Simplified/SkipSimplified normalization adapters runs one complete
       native INT4 block; the portable RMSNorm, RoPE, Gather, SiLU, Softmax,
       and layout paths remain differential coverage.
     - Qwen PR02; completed ExpLog PR01-PR03
     - Pending
   * - Qwen PR05
     - Frozen-graph integration with shared Attention.
     - Native ``GroupQueryAttention`` and standard ``Attention`` dispatch to
       the shared descriptor, planner, materialized fallback, and online
       engine. Batch-1 causal prefill/decode use zero-copy query/KV-head
       grouping; no Qwen-only compute engine is introduced.
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
