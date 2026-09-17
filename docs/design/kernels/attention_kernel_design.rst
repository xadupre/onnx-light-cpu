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

The registered adapter publishes ``Y`` and the requested optional outputs.
``present_key`` and ``present_value`` concatenate past and current state in
the original storage type and always use rank-four layout.

Compute paths
-------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Path
     - Selection
     - Storage
   * - Materialized
     - Observable ``qk_matmul_output`` or explicit FP64 softmax; FLOAT also
       selects this path for ``present`` outputs.
     - Retains a full KV score row, and the complete QK output when requested.
       Half-storage inputs and outputs use full FP32 conversion buffers.
   * - Tiled
     - FLOAT/FLOAT16, no past state, at least 16 query rows and more than one
       key, when materialized dispatch is not required.
     - Online softmax over bounded query/KV tiles with FP32 accumulation.
   * - Streaming
     - Short queries, tensor-cache inputs, and BFLOAT16, when materialized
       dispatch is not required.
     - Visits KV blocks with online softmax and retains one score tile plus row
       accumulators instead of an ``Lq x Lkv`` matrix.
   * - Single key
     - Exactly one effective key, when materialized dispatch is not required.
     - Copies the value directly, or writes zero for a masked row.

The streaming recurrence maintains a running maximum, denominator, and
unnormalized output for each query row. Causal and padding frontiers skip
entire unavailable blocks; an all-false boolean-mask block is skipped as well.
Arbitrary additive masks remain fully evaluated.

Conversion and packing
----------------------

Tiled FP16 execution keeps Q/K in their original storage type. It converts V
into a worker-local KV tile immediately before the value product, retains
FP32 accumulation, and narrows the final query tile directly into Y.
Rank-three Q/K/V rows are packed only when their sequence stride requires it;
packing and output scatter are tile-local for both FP16 and FP32. There is no
immutable-input cache: every invocation reads the current source tensors.

FP16 streaming converts the current query row and the visited K/V blocks,
including blocks spanning past and current state, rather than converting
complete tensors. For short contexts, up to four score blocks are converted
once per worker/head and reused across query rows within that invocation.
Longer contexts retain one conversion block. BF16 streaming retains its
FP32-accumulating codec path.
Half-storage materialized execution deliberately retains its full conversion
buffers for observable QK output and FP64 softmax.

The benchmark's memory model bounds active attention scratch per worker. It
does not include vector allocator capacity retained from earlier calls or
internal GEMM packing panels, and should not be interpreted as measured RSS.

Scheduling, precision, and invariants
-------------------------------------

Outer batch/head/query-row ranges are submitted through the session executor.
Prefill exposes many independent rows, while short-query and decode shapes
avoid forced parallel overhead. FP16 and BF16 streaming paths accumulate in
FP32 and narrow only the final output. Explicit FP64 softmax uses the
materialized path rather than weakening its precision contract.

Q, K, and V types must match, and cache tensors must match those types.
Rank-three head counts must be positive, with the Q count divisible by the KV
count. ``nonpad_kv_seqlen`` must be INT64. Invalid optional-input pairings and
unsupported combinations fail explicitly. A fully masked query row produces
zeros rather than NaN.

Execution telemetry
-------------------

Enable ``set_kernel_usage_recording(session, True)`` and inspect
``used_kernel_paths(session)`` to distinguish ``Attention.tiled``,
``Attention.streaming``, ``Attention.materialized``, and
``Attention.single_key``. Additional ``Attention.conversion.*``,
``Attention.packing.*``, and ``Attention.output.*`` records describe the
selected storage decisions. The ordinary ``used_kernel_names(session)``
inventory still reports only ``onnx_light_cpu::Attention``.
``Attention.output.tile`` describes the final row/tile write granularity,
not the presence of an extra Y buffer: contiguous FP32 single-block products
can write directly into Y. The conversion and packing records describe
explicit attention scratch, not internal GEMM panels.
BF16 streaming reports ``Attention.conversion.element`` for its scalar codec
and FP32 query-row cache rather than claiming conversion-free execution.

``tools/benchmark_attention_parity.py`` retains these records alongside raw
alternating CPU/ONNX Runtime samples. Recording is disabled during timing.
ORT worker spinning is disabled so its idle workers do not compete with the
CPU implementation between alternating samples.
The corpus includes aligned q128/kv128 cases, q129/kv257 tile tails, head
dimensions 63/64/128/256, and short-query streaming cases for FP16 and FP32.

Native half-precision arithmetic
-------------------------------

QK products reuse the existing half-storage GEMM dispatch; softmax and value
accumulation stay in FP32. AVX2/F16C accelerates conversion but does not provide
native FP16 arithmetic. The validation host for this change has no AVX-512
FP16/BF16 support, so no throughput claim is made for those instructions.
FP16 accumulation is not substituted for FP32 accumulation, and probabilities
are not rounded to BF16 simply to use a native BF16 value product.
