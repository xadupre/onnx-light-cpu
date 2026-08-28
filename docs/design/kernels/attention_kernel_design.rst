Attention Kernel Design
=======================

The registered ``Attention`` kernel separates node configuration, concrete
shape planning, and compute. It supports ONNX Attention opsets 23 and 24 for
FLOAT, FLOAT16, and BFLOAT16.

Descriptor and invocation plan
------------------------------

``AttentionDescriptor`` records attributes and optional input/output wiring,
then validates opset rules, head counts, ``qk_matmul_output_mode``, cache
pairs, and ``nonpad_kv_seqlen`` availability. The registered adapter currently
rebuilds this descriptor from the node on every invocation. Because the
adapter does not receive the model's opset directly, it infers opset 24 when a
seventh input is present and opset 23 otherwise.

``AttentionPlan`` is lightweight and rebuilt for each invocation because
sequence lengths and strides may change:

.. code-block:: text

   NodeProto ----------------> AttentionDescriptor
                                      |
   Q/K/V/mask/cache shapes ----------+
                                      v
                               AttentionPlan
                               - layout and strides
                               - head mapping
                               - mask broadcasting
                               - total KV length
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                  materialized path          streaming path

Layouts and semantics
---------------------

Rank-four tensors use ``[B, H, L, D]``. Rank-three tensors use
``[B, L, H * D]`` and require explicit Q and KV head counts. MHA, GQA, and MQA
share one plan; ``group_size = q_num_heads / kv_num_heads`` maps Q heads to K/V
heads without physically repeating K or V.

The plan supports boolean and FLOAT additive broadcast masks, bottom-right
causal masking, ``softcap``, tensor ``past_key``/``past_value``, opset-24
``nonpad_kv_seqlen``, optional ``present`` outputs, and all four
``qk_matmul_output_mode`` values. A V head dimension may differ from the Q/K
head dimension.

Those optional outputs are plan-level capabilities used by the low-level
compute and tests. The registered adapter currently stores output 0 (``Y``)
only. Declaring ``present`` or ``qk_matmul_output`` on a FLOAT node forces the
materialized compute path, but the adapter does not publish those tensors.

Compute paths
-------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Path
     - Selection
     - Storage
   * - Materialized
     - FLOAT only; selected when ``present`` or ``qk_matmul_output`` is wired.
     - Builds the complete score/probability tensor, applies masks and softmax,
       then multiplies by V.
   * - Streaming
     - Used for FLOAT without optional observable tensors and unconditionally
       for accepted FLOAT16/BFLOAT16 nodes.
     - Visits KV blocks with online softmax and retains one score tile plus row
       accumulators instead of an ``Lq x Lkv`` matrix.

The streaming recurrence maintains a running maximum, denominator, and
unnormalized output for each query row. Causal and padding frontiers skip
entire unavailable blocks; an all-false boolean-mask block is skipped as well.
Arbitrary additive masks remain fully evaluated.

Scheduling, precision, and invariants
-------------------------------------

Outer batch/head/query-row ranges are submitted through the session executor.
Prefill exposes many independent rows, while short-query and decode shapes
avoid forced parallel overhead. FP16 and BF16 streaming paths accumulate in
FP32 and narrow only the final output; the currently supported explicit
softmax precision is FP32.

Q, K, and V types must match, and cache tensors must match those types.
Rank-three head counts must be positive, with the Q count divisible by the KV
count. ``nonpad_kv_seqlen`` must be INT64. FLOAT16/BFLOAT16 nodes reject
observable ``present`` or ``qk_matmul_output`` wiring. Invalid optional-input
pairings and unsupported combinations fail explicitly. A fully masked query
row produces zeros rather than NaN.
