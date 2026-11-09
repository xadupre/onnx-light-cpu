Generation-only logits
======================

Full-sequence logits remain the default. Neither MatMul kernels nor session
registration truncate outputs. Generation exporters can explicitly request one
token's logits per batch item, selecting hidden states **before** the LM head.
This changes the model output contract from ``[B, S, V]`` to ``[B, 1, V]``;
it is not a transparent optimization of an all-token benchmark.

Opt-in preparation of already-selected graphs
---------------------------------------------

.. py:function:: onnx_light_cpu.generation.optimize_generation_logits(model)

   Return a prepared ``ModelProto`` copy, leaving unsupported graphs unchanged.

The supported graph is standard-domain ONNX (opset 12 or later)::

    hidden: [B, S, H]   weight: [H, V]   positions: INT64[B, 1, 1]
    full = MatMul(hidden, weight)
    logits = GatherND(full, positions, batch_dims=1)

Preparation produces::

    selected_hidden = GatherND(hidden, positions, batch_dims=1)
    logits = MatMul(selected_hidden, weight)

Call it explicitly before creating sessions, passing the same prepared model to
both runtimes::

    from onnx_light_cpu.generation import optimize_generation_logits

    generation_model = optimize_generation_logits(exported_generation_model)
    session = ReferenceEvaluator(generation_model)
    register_kernels_for_session(session)
    ort_session = onnxruntime.InferenceSession(generation_model.SerializeToString())

The input model is copied, not mutated. Initializers are not consumed; sessions
may be reused or constructed repeatedly from that same model. Do not enable
initializer-consuming ownership when recreating sessions from a model.

Why the rewrite preserves semantics
-----------------------------------

For batch item ``b`` and position ``p = positions[b, 0, 0]``, each result is
``hidden[b, p, :] @ weight``. Selection therefore commutes with this row-wise
projection. The weight must have rank two (no batch broadcasting); hidden states
must have rank three with a known, positive contraction dimension matching the
weight. Positions must have shape ``[B, 1, 1]`` and INT64 dtype. Selection keeps
the singleton token axis and the original element type. Floating-point GEMM
implementations can use different accumulation orders, so comparisons use
dtype-appropriate tolerances, not bitwise equality.
For overridable initializers, the proof uses the declared graph input shape,
not the shape of the default initializer.

The full MatMul result must have exactly one consumer, that GatherND, and must
not be a graph output. Otherwise preparation leaves the graph unchanged. It
also conservatively skips missing/unsupported shapes, nonstandard operators,
and graphs containing subgraphs (which could capture full logits). The nodes
are inserted at the original selection site, after any computed positions.
This is not a general MatMul/Slice/Cast/quantized-head rewrite.

Token positions, padding, and empty prompts
------------------------------------------

Positions are physical indices into the current hidden-state sequence, **not**
lengths in the accumulated KV cache. The caller must compute them from its
actual padding/position convention. For right-padded contiguous prompts of
positive lengths ``[3, 5]`` in a five-token buffer, supply ``[[[2]], [[4]]]``.
For left-padded prompts whose last physical token is valid, supply
``[[[4]], [[4]]]``. For masks with gaps, find the last valid physical index;
``sum(mask) - 1`` is not generally correct. No mask convention is inferred.

Negative indices retain ONNX GatherND semantics (``-1`` means the last
physical row, not necessarily the last valid token). During single-token
decode use index zero, regardless of cache length. Sequence lengths may change
between calls to the same session.

A nonempty batch with zero sequence length has no selectable token: selection
fails with an out-of-bounds error before or after preparation. Similarly, an
all-padding batch item has no valid last token; the generation caller must
reject it rather than supply a made-up position. Empty batches have no
generation token to produce and should be handled by the caller.

Exporter and ASV integration
----------------------------

TinyLLM is built in
`onnx-asv-benchmark/benchmarks/models/llm/tiny_llm.py
<https://github.com/xadupre/onnx-asv-benchmark/blob/main/benchmarks/models/llm/tiny_llm.py>`_
using ``modelbuilder.builder.create_model``. The export option belongs at that
builder boundary, before either backend constructs its session. An exporter
can emit the selected-hidden graph directly, or emit the supported selected
logits graph with shape metadata and apply this preparation.

The existing ``TinyLLM.time_prefill`` validates and returns all token logits.
Leave it unchanged. An upstream generation-only scenario should have a
different name (for example ``TinyLLMLastToken``), validate ``[B, 1, V]``,
pass padding-aware positions, and use the exact same reduced-output model for
ORT and onnx-light-cpu. A Python-side slice after ``session.run`` alone does
not reduce the projection. No upstream repository is modified here.

Separate measurements
----------------------

From the repository root::

    python -m tools.benchmark_generation_logits --threads 1 --repeat 20

This separately named scenario measures both runtimes, reusing sessions. It
reports complete prefill and isolated projection median latency, plus logits
output bytes (not peak runtime memory), for all-token, selected-after-projection,
and selected-before-projection models. CPU timings finish before ORT sessions
are constructed, and use native tensor feeds.

Without inputs, the tool uses a **synthetic** Relu prefix and an FP32
``[B, 128, 512] @ [512, 32000]`` head. This is a projection/prefill proxy,
not an end-to-end TinyLLM latency claim. For an actual exported selected-output
model, provide ``--model model.onnx --feeds feeds.npz`` and optionally
``--logits output_name``. The terminal selection must have the supported form,
weight must be an initializer, and hidden/full/positions shape metadata must
be present. Feeds must include explicit positions. Other outputs, such as KV
caches, remain in the measured prefill graph. ``--backend onnxruntime`` can
measure ORT alone when the accelerated runtime is unavailable.

At batch one, FP32 logits output storage falls from 16,384,000 to 128,000
bytes for ``S=128, V=32000``. The projection performs 128 times less matrix
arithmetic; neither prefill nor projection latency is promised to improve
128-fold. Keep these results separate from the existing all-token ASV result,
prepacking work, and activation-fusion work.

An illustrative synthetic run on 2026-11-09 (ORT 1.31.1, one thread, two warmups,
ten measured calls) gave:

.. list-table::
   :header-rows: 1

   * - Output/projection mode
     - Prefill proxy (ms)
     - Projection (ms)
     - Logits bytes
   * - All tokens
     - 38.41
     - 38.05
     - 16,384,000
   * - Selection after projection
     - 39.00
     - 38.27
     - 128,000
   * - Selection before projection
     - 2.30
     - 2.16
     - 128,000

These are shared-runner observations, not performance gates. Accelerated CPU
measurements were unavailable in that environment: the released onnx-light
0.2.6 wheel lacked APIs required to build this checkout against its runtime.
Run the default two-backend command with a matching onnx-light build to collect
that comparison; no accelerated CPU or full TinyLLM latency is claimed here.
