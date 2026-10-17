Custom operators
----------------

.. py:function:: custom_op_schemas(op_type="", init_doc=True)

   Returns ``LightOpSchema`` records for supported ``com.microsoft`` operators.

.. py:function:: operator_schema_lookup(op_type)

   Returns standard ONNX schemas, Microsoft custom schemas, and this package's
   experimental compatibility schemas. Pass it as
   ``GraphBuilder(..., schema_lookup=operator_schema_lookup)``.

.. py:function:: experimental_op_schemas(op_type="", init_doc=True)

   Returns experimental ``ai.onnx`` compatibility ``LightOpSchema`` records.
   These adapters are separate from standardized ONNX schemas and from the
   Microsoft-only :func:`custom_op_schemas` provider.

Microsoft GroupQueryAttention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``com.microsoft::GroupQueryAttention`` supports causal local attention for
``FLOAT``, ``FLOAT16``, and ``BFLOAT16``. ``local_window_size`` defaults to
``-1`` (full attention); a positive INT32 value counts attended tokens,
including the current token. For example, ``2048`` attends the current
position and up to 2047 preceding positions. Zero, values below ``-1``,
values above ``INT32_MAX``, and local windows with ``causal=0`` are rejected.

Windowing restricts only attention scores: ``present_key`` and
``present_value`` still contain the entire past cache followed by the current
step. RoPE uses absolute positions as before. Full and sliding attention nodes
can coexist in the same model; ``sliding_window_cache`` remains unsupported.

The optional twelfth input, ``head_sink``, has shape ``(num_heads,)`` and the
same element type as the query. Each query head's sink is a virtual softmax
logit: it contributes to the normalization denominator, but contributes no
value to the output. Causal and local-window masks still apply only to real
keys. Omitting the input preserves ordinary attention normalization.

Microsoft MoE
~~~~~~~~~~~~~

``com.microsoft::MoE`` version 1 routes rank-2 packed tokens ``[T,H]`` or
rank-3 decoder activations ``[B,S,H]`` using rank-2 ``[T,E]`` router **logits**.
It applies a softmax over all experts, selects the top ``k`` (default 1), and
optionally renormalizes the selected weights before summing expert outputs.
Expert weights are ``[E,I,H]`` for FC1 and ``[E,H,I]`` for FC2. Optional
expert-major biases occupy inputs 3 and 5. Interleaved SwiGLU
(``activation_type="swiglu"``, ``swiglu_fusion=1``) uses FC1 ``[E,2*I,H]``
with alternating gate and linear rows, and supports ``swiglu_limit``,
``activation_alpha``, and ``activation_beta``. The other supported activations
are ``relu`` (default), ``gelu``, ``silu``, and ``identity``.

Input and output types must match and be ``FLOAT``, ``FLOAT16``, or
``BFLOAT16``. Low-precision inputs and weights are converted in bounded row
tiles and accumulated in FLOAT before the output is narrowed. Separate FC3
weights and biases, sparse mixer routing, and noninterleaved SwiGLU are not
supported by the CPU kernel. The output retains the input shape.
See `ONNX Runtime's MoE CPU implementation
<https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cpu/moe/moe_cpu.cc>`_
for the upstream compatibility contract.

Microsoft QMoE
~~~~~~~~~~~~~~

``com.microsoft::QMoE`` version 1 accepts the block-wise CPU expert tensors emitted by
`mbext's MoE builder
<https://github.com/xadupre/mbext/blob/68a6860cc8fe2c3381f5248d215b7217a36822e3/modelbuilder/builders/base.py#L3870-L4010>`_.
Activations, router logits, optional biases, and output share one of
``FLOAT``, ``FLOAT16``, or ``BFLOAT16``. Scales may independently use any of
these three floating-point types. Input/output and routing shapes,
top-k softmax normalization, and activation parameters follow ``MoE`` above.

Inputs are, in order: input, router logits, FC1 weights, FC1 scales,
optional FC1 bias, FC2 weights, FC2 scales, optional FC2 bias, optional FC3
weights, optional FC3 scales, optional FC3 bias, and three optional zero-point
slots. FC3 weights and scales must appear together. Nonempty zero points,
sparse mixer routing, and CUDA-prepacked weights are rejected.

``expert_weight_bits`` supports **4 and 8**. INT4 stores adjacent last-axis
values in low-then-high nibbles with implicit zero point 8; INT8 stores signed
two's-complement values in ``UINT8`` tensors. ``block_size=0`` (default)
uses raw per-last-axis-column scales ``[E,C]``. Supported positive block sizes
are powers of two of at least 16, including **16, 32, 64, and 128**;
scales then have shape
``[E,R,ceil(C/block_size)]`` for a stored logical matrix ``[E,R,C]``.
The final block and an odd INT4 column count need not be padded in the
weight tensor: its shape is ``[E,R,ceil(C/2)]`` for INT4 or ``[E,R,C]`` for INT8.

The fused builder emits output-major FC1 ``[E,2*I,H]`` and FC2 ``[E,H,I]``
logical matrices with interleaved SwiGLU gate/up rows
(``activation_type="swiglu"``, ``swiglu_fusion=1``).
The split builder supplies FC3 and emits input-major FC1/FC3 ``[E,H,I]``
and FC2 ``[E,I,H]`` logical matrices. Its expert function is
``FC2(activation(FC1(input)) * FC3(input))``, including optional biases.
Separate FC3 and fused SwiGLU cannot be combined. Without FC3, ordinary
``relu``, ``gelu``, ``silu``, and ``identity`` projections are also supported
in input-major layout.

The kernel validates both orientations against all projection and scale
shapes, preferring the builder's orientation when dimensions are ambiguous.
It also accepts ONNX Runtime's raw output-major, row-scaled tensor format
(``block_size=0``, scales ``[E,R]``) when the per-column formats do not fit.
That format uses an implicit INT8 zero point of 128 rather than signed
bytes. See the `ONNX Runtime CPU dequantizer
<https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cpu/moe/moe_quantization_cpu.cc>`_.

This is a portable correctness implementation: expert matrices are
dequantized into execution-allocator scratch and projections accumulate in
FLOAT. It does not interpret CPU weights as CUDA's MatMulNBits block packing
or TensorRT's shuffled prepacked format. Quantized weights are inference-only;
no QMoE gradient or graph-fusion rule is registered.

The cited builder's legacy ``qmoe_block_size=0`` fallback selects
`TensorRT-LLM's preprocessed quantizer output
<https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/thop/weightOnlyQuantOp.cpp>`_,
not its raw CPU tensor output. That
architecture-dependent CUDA format is not the raw tensor layout described
above, even though it has the same tensor shape. Re-export those models with
a positive CPU ``qmoe_block_size`` (for example, 32). Untagged CUDA-prepacked
contents cannot be identified from tensor shapes alone; do not pass them as
raw CPU tensors.

Microsoft MatMulNBits
~~~~~~~~~~~~~~~~~~~~~

``com.microsoft::MatMulNBits`` version 1 multiplies floating-point activations by a
block-quantized right-hand matrix without materializing the complete
dequantized matrix. The CPU contract implements packed INT2, INT4, and INT8
formats with required ``K`` and ``N`` attributes, ``bits`` equal to 2, 4, or
8, ``block_size=32``, ``accuracy_level`` 0 or 4, and
``weight_prepacked=0``. Input ``B`` has shape
``[N, ceil(K / 32), 4 * bits]`` and scales have shape
``[N, ceil(K / 32)]`` or the equivalent flat shape. Implicit zero points are
2, 8, and 128. Activations, scales, optional bias, and output have one matching
``FLOAT``, ``FLOAT16``, or ``BFLOAT16`` type; ``DOUBLE`` is not supported.

Input ``B`` may instead be an ``EncodedValueProto`` using onnx-light's
``ORT_MATMULNBITS_INT2``, ``ORT_MATMULNBITS_INT4``, or
``ORT_MATMULNBITS_INT8`` layout. In that form, omit the ``scales`` input: the
kernel validates the encoded geometry, dtype, layout, padding, and implicit
zero points, then borrows its packed ``B`` and scales without copying them per
inference. It never converts the encoded weights to the VNNI preparation layout. Encoded layouts
with explicit zero points, shared parameter references, incompatible
attributes, or any other unsupported representation fail with an exception.
This encoded form is inference-only; automatic differentiation continues to
require separate tensor ``B`` and ``scales`` inputs.

Genuine three-bit weights use the separate onnx-light
``MATMULNBITS_INT3`` format (struct name
``onnx_light.quantization.v1/matmulnbits_int3``), **not** an ORT layout
or a ``UINT8`` tensor ``B``. Set ``bits=3``, ``block_size=32`` and omit
the ``scales`` node input. Its ``EncodedValueProto.logical_type`` is
``[K,N]`` with ``FLOAT``, ``FLOAT16``, or ``BFLOAT16`` element type.
The inline struct has, in order, an ``INT64[2]`` constant named
``parameters`` containing ``[3,32]``, a ``UINT8[N,ceil(K/32),12]``
field named ``B``, a ``[N,ceil(K/32)]`` field named ``scales`` with
the logical element type, and optionally a ``UINT8[N,ceil(K/32)]``
field named ``zero_points``. Concatenate these fields (excluding the
constant) in that order into ``raw_data``; no external payload or shared
parameters are permitted. Block order is output column first, then K
block. In each 12-byte block, code ``q[p]`` occupies bits
``[3*p, 3*p+3)`` in little-endian bit order (least-significant bit
first within each byte). Unused codes and bits after the final K
element in each column must be zero. For example, codes
``[1,2,5]`` begin with bytes ``51 01`` (the low byte of
``1 | (2 << 3) | (5 << 6)`` is ``0x51``; the third code's high
bit is in the second byte). The remaining
ten bytes of this block are zero. An omitted zero-point field means
code 4 for every block; explicit zero points must each be in ``[0,7]``.
Weights are reconstructed as ``(q - zero_point) * scale`` without
expanding a full INT8 matrix. INT3 is not ternary: it represents eight
distinct codes.
The low-level C++ INT3 entry point also rejects overflowing activation,
packed-weight, scale, and output sizes before dispatch or buffer access.

The activation may have any positive rank with final dimension ``K``. Shape
inference preserves every leading concrete or symbolic dimension and replaces
the final dimension with ``N``. INT4 uses bounded panels of 8 rows, 32 output
columns, and 32 reduction elements, with scalar, AVX2, or AVX-512 dispatch.
Scratch is 6,144 bytes per active callback (4,096 decoded weight bytes,
1,024 activation bytes, and 1,024 accumulator bytes), independent of matrix
size. On AVX-512 VNNI/BW systems, FP32 INT4 with ``accuracy_level=4`` keeps
the weights packed at four bits in a 16-column VNNI layout, reorders the
scales, and precomputes weight sums. This uses
``K*N/2 + 8*N*ceil(K/32)`` persistent bytes, dynamically quantizes each
32-value activation block to INT8, and accumulates in INT32 with ``vpdpbusd``.
Each active eight-row tile uses ``8*K + 32*ceil(K/32)`` temporary bytes.
Other combinations retain the bounded panel or allocation-free scalar path.
Inputs and outputs are excluded from scratch accounting. The projection
benchmark reports both worker-local and prepared storage.

Explicit zero points in the legacy layouts or node input, deprecated ``g_idx``,
provider-prepacked weights, other bit widths, block sizes, mixed floating-point types, and ``DOUBLE`` are
rejected rather than silently using a different layout. The registered gradient differentiates the
activation and optional bias while treating packed weights and scales as
constants. The bias fusion applies only to an exclusively consumed
``MatMulNBits`` output followed by a compatible rank-one ``Add``.

Muse-Glimmer projections
^^^^^^^^^^^^^^^^^^^^^^^^

The synthetic projection suite uses ``M=1,8,128`` for decode, short prompts,
and prefill. Its ``K -> N`` families are Q/attention gate ``6656 -> 4096``,
K/V ``6656 -> 256``, attention output ``4096 -> 6656``, MLP gate/up
``6656 -> 19968``, MLP down ``19968 -> 6656``, and vocabulary
``6656 -> 202048``. The vocabulary is the `Transformers reference default
<https://github.com/huggingface/transformers/blob/bdb4cc00d5c76659f46585cd552b238c4d7bec54/src/transformers/models/muse_glimmer/configuration_muse_glimmer.py#L120-L146>`_,
not a verified checkpoint value.

The `mbext source plan
<https://github.com/xadupre/mbext/blob/cc81bff49c8045ad734a059979076882cabb1aa2/docs/next_steps/2026-08_muse_glimmer.rst>`_
does not supply an ONNX artifact. The generic `exporter
<https://github.com/xadupre/mbext/blob/cc81bff49c8045ad734a059979076882cabb1aa2/modelbuilder/helpers/quantization.py#L260-L359>`_
packs consecutive K values low-nibble first into UINT8 ``[N,ceil(K/G),G/2]``
and stores matching floating-point scales ``[N,ceil(K/G)]``.
Symmetric default/RTN export omits zero points (INT4 midpoint 8);
``G=32`` is the `configurable builder default
<https://github.com/xadupre/mbext/blob/cc81bff49c8045ad734a059979076882cabb1aa2/modelbuilder/builders/base.py#L440-L500>`_.
The `activation dtype policy
<https://github.com/xadupre/mbext/blob/cc81bff49c8045ad734a059979076882cabb1aa2/modelbuilder/builder.py#L112-L144>`_
uses FP32 for CPU integer-weight export; CUDA normally uses FP16, or BF16 when requested.
These are source-audited policies, not observations of a Muse model.

**Separate unsupported contract:** asymmetric and ``k_quant`` exports can
provide explicit packed zero points ``[N,ceil(ceil(K/G)/2)]``; exporters
can also select a block size other than 32. Neither is enabled by this
optimization. The actual artifact must be inspected for these attributes,
zero-point inputs, dtypes, projection fusion, and whether the LM head is
quantized before claiming model compatibility.

Run the reproducible parity/latency suite with
``python -m tools.benchmark_matmul_nbits_parity --help``. It separates
preparation from repeated invocations using constant initializers, compares
against ONNX Runtime, and reports scratch/copy accounting separately from
process memory. Pass ``--accuracy-level 4`` to exercise the INT8-accumulation
path. Large vocabulary cases are opt-in because model serialization and ONNX
Runtime preparation can require several GB.
For isolated single-thread FP32 kernel measurements, build with
``-DONNX_LIGHT_CPU_BUILD_BENCHMARKS=ON`` and run
``matmul_nbits_throughput M K N repeats [bits=4|8] [--generic]``. This reports input
preparation, first-use packing time, median steady-state seconds, source packed
bytes, persistent prepared bytes, per-callback scratch bytes, and the selected
implementation.
For example, ``matmul_nbits_throughput 1 512 32000 21 8 --generic`` measures
the generic INT8 panel path, not INT4. Both modes use one participant;
the generic mode allocates no persistent prepared-weight buffers.
The bit width remains optional with ``--generic``:
``matmul_nbits_throughput 1 512 32000 21 --generic`` uses the default INT4 path.

The prepared INT8 path processes four activation rows per eight-column tile,
reusing packed weights across rows and keeping dot products and accumulation
inside the ISA-specific implementation. Activation blocks use the full signed
INT8 range ``[-127,127]``; AVX2 splits the unsigned activation's high bit before
pairwise multiplication to avoid 16-bit saturation. AVX-512 VNNI uses all 512
bits to process two weight quads at once.

Disjoint scratch of ``5*K`` bytes per participant is allocated through the
runtime execution allocator before dispatch; range callbacks do not allocate.
For decode with ``M=1,2,3,4``, activations are quantized once into shared,
read-only scratch of ``5*M*K/4`` bytes and workers own disjoint groups of
eight output columns. The parallel threshold uses the actual ``M*N``
output count, excluding padded lanes. For a zero reduction dimension, the
low-level entry point returns the bias, or zero when no bias is supplied,
without dispatch.
The output-based parallel threshold, target block size, and participant cap
apply to prepared execution as well as the generic path. Activation scales,
scale products, and biased accumulation remain in double precision until each
updated block accumulator is converted to FLOAT. Nonfinite activations or
weight scales retain per-element floating-point dequantization semantics.
Implementation labels remain available through ``used_kernel_paths()``, but
are excluded from ``used_kernel_names()``.

Run ``python -m tools.benchmark_matmul_nbits_parity --qwen-int8`` for all six
synthetic Qwen2 prefill projections at four threads. Both sessions use the
same graph and input values; CPU timings feed native backend tensors without
NumPy adaptation. Preparation and first-run times are separate from the
alternating steady-state samples. These isolated nodes do not measure the
exported end-to-end decoder.
Use ``--qwen-int8-decode`` instead for all six projections at ``M=1,2,4``
with one and four threads.

On an Intel Xeon Platinum 8480C with affinity restricted to four CPUs,
GCC 13.2 Release and ONNX Runtime 1.32.0, the ``--qwen-int8`` command with
``--repeat 15 --warmup 5`` measured these median steady-state milliseconds
(``M=128``, FP32, INT8, block size 32, accuracy level 4). The selected CPU
implementation was ``int8_avx512_vnni``; maximum absolute error against ORT
across these six cases was below ``1.9e-5``.

.. list-table::
   :header-rows: 1

   * - Projection
     - K
     - N
     - CPU before (``51e760e``)
     - CPU tiled
     - ORT
   * - LM head
     - 512
     - 32000
     - 21.044
     - 6.812
     - 6.725
   * - MLP gate
     - 512
     - 1376
     - 0.900
     - 0.337
     - 0.401
   * - MLP up
     - 512
     - 1376
     - 0.900
     - 0.325
     - 0.421
   * - MLP down
     - 1376
     - 512
     - 0.951
     - 0.328
     - 0.484
   * - Attention QKV
     - 512
     - 1024
     - 0.704
     - 0.251
     - 0.337
   * - Attention output
     - 512
     - 512
     - 0.376
     - 0.137
     - 0.228

Both CPU revisions used native backend feeds, the same deterministic inputs,
four threads, five warmups, and fifteen alternating samples. Preparation is
excluded. The old seven-bit activation quantization also failed ORT parity
on these cases (maximum LM-head error about ``0.285``); the timings above
do not imply that its numerical behavior was equivalent. These measurements
are not portable speedup thresholds, and the LM head remains slightly slower
than ORT on this runner.

ONNX Runtime 1.30.0 CPU does not implement native BF16 MatMulNBits. The
BF16 oracle therefore runs ORT's FP32 operator on BF16-rounded activations
and scales and rounds its output back to BF16. This validates the CPU
BF16 path without duplicating the quantized operator in Python, but is
not a claim of native ORT BF16 support.

Validation on this runner completed all 81 full-sized ORT comparisons:
nine projection families at ``M=1,8,128``, for all three activation types,
including vocabulary prefill. The separate attention-gate case has the
same dimensions and deterministic inputs as Q. Reproduce these checks
through ``--full`` or the ``OLC_MUSE_INT4_FULL=1`` test matrix.
Maximum absolute errors across the completed comparisons were
``4.816e-5`` (FP32), ``0.00390625`` (FP16), and ``0.015625`` (BF16 oracle).

Representative kernel measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On an AMD EPYC 9V74 runner, GCC 13.3 Release, one runtime participant,
FP32 ``K=6656,N=256``, the standalone benchmark measured the following
seconds (one warmup, median of five runs). The baseline is the foundation
implementation at ``a16de32cd15d4c4ac3ba9d7644a92dc80b7c2e22``;
the optimized binary reported ``int4_panel_avx512``.
These are representative measurements, not portable performance thresholds.

.. list-table::
   :header-rows: 1

   * - M
     - Input preparation
     - Baseline steady state
     - Panel steady state
     - Panel scratch bytes
     - Packed-weight copy bytes/run
   * - 1
     - 0.00101
     - 0.00334
     - 0.00154
     - 6144
     - 0
   * - 8
     - 0.00114
     - 0.0266
     - 0.00213
     - 6144
     - 0
   * - 128
     - 0.00285
     - 0.425
     - 0.0295
     - 6144
     - 0

Preparation above generates inputs and allocates output storage; the kernel
has no packing/preparation phase. Python session construction and first-use
costs are separate from these low-level timings. Packed weights occupy
851,968 bytes in this case; a full FP32 weight matrix would occupy
6,815,744 bytes. The same panel bound applies to the reference vocabulary
projection, whose full FP32 weight matrix would occupy 5,379,325,952 bytes.

Experimental SimplifiedLayerNormalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SimplifiedLayerNormalization`` is an experimental ONNX Runtime compatibility
operator in the default domain (``""``, also spelled ``"ai.onnx"``), since
version 1. Use :func:`operator_schema_lookup` to load or incrementally construct
its graph with ``GraphBuilder.make_node``. Register shape and memory support
with :func:`register_operator_support`; all global and session-local kernel
registration helpers also register this support.

``X`` and ``Scale`` independently support ``FLOAT``, ``FLOAT16``, ``DOUBLE``,
and ``BFLOAT16`` (all sixteen pairs). Mandatory ``Y`` has the shape of ``X``
and the element type of ``Scale``. ``Scale`` broadcasts right-aligned to the
**entire** shape of ``X``, not only the normalized suffix, and cannot expand
the shape of ``X``. ``axis=-1`` selects the first normalized dimension and
``epsilon=1e-5`` is added to the mean square. IEEE epsilon values, including
negative and non-finite values, are permitted.

The second output, ``inv_std_var``, is optional and may be omitted or named
with an empty string. ``stash_type=1`` saves it as ``FLOAT``; ``stash_type=11``
saves it as ``DOUBLE``. Other stash types are rejected. Its runtime-compatible
shape is ``X[:axis] + [1] * (rank(X) - axis)`` after normalizing negative
``axis``: **every** normalized dimension is one, unlike upstream schema
inference which only replaces the axis dimension. ``X`` must have positive
rank and a nonempty normalized suffix; empty outer rows are supported.

``stash_type`` selects arithmetic precision as well as the statistics type:
``1`` uses FP32 and ``11`` uses FP64, independently of input types, broadcasting,
and whether statistics are requested. Inputs and scale are converted to this
precision for the reduction, normalization, and scaling; the result is then
converted to the type of ``Scale``. In particular, FP64 inputs with the default
FP32 stash can overflow when converted or squared. There is no intermediate
low-precision rounding before multiplication by ``Scale``.

The FP32 suffix path reuses the optimized RMS mean-square and affine engine.
FP16 with FP32 stash uses shared F16C primitives with a single final rounding;
FP64 uses AVX primitives for either stash type. Contiguous scale suffixes use
one broadcast index per row or block, including outer-broadcast scales.
Unsupported SIMD/type/alignment combinations retain portable fallbacks.
Optional statistics are saved without repeating the reduction. Benchmark
``cpu_kernel_paths`` diagnostics distinguish the RMS, FP16/F16C, FP64/AVX,
scalar, and generic paths, together with their scale layout.

CPU tensor scratch memory is zero, excluding inputs and outputs (broadcast
metadata is proportional to rank). This is inference compatibility support:
no gradient rules or fusion patterns are registered for this operator.

Microsoft SkipLayerNormalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``com.microsoft::SkipLayerNormalization`` version 1 adds ``input``, ``skip``,
and optional ``bias`` before centering the residual sum over its last axis,
scaling by the inverse standard deviation and ``gamma``, and adding optional
``beta``. All inputs have one matching ``FLOAT``, ``FLOAT16``, or ``BFLOAT16``
type. ``input`` has shape ``[S, H]`` or ``[B, S, H]``; for rank-three input,
``skip`` may also have shape ``[S, H]`` or ``[1, S, H]``. ``gamma``, optional
``beta``, and optional ``bias`` are rank-one tensors of length ``H``.

Output slot 0 has the input shape and type. Optional slots 1 and 2 are FLOAT
mean and inverse-standard-deviation tensors with final dimension one. Optional
slot 3 is the typed residual sum. ``epsilon`` defaults to ``1e-12`` and must be
finite and nonnegative. The allocation-free CPU kernel uses FP32 arithmetic
for narrow inputs and runtime-owned row scheduling. This follows the
`ONNX Runtime CPU contract
<https://github.com/microsoft/onnxruntime/blob/9c87bdf2e6676eee0073c711a32e75183dc41199/onnxruntime/contrib_ops/cpu/skip_layer_norm.cc>`_.
Shape inference preserves symbolic dimensions and scratch-memory accounting is
zero, excluding inputs and outputs.

Microsoft SkipSimplifiedLayerNormalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``com.microsoft::SkipSimplifiedLayerNormalization`` version 1 adds ``input``,
``skip``, and optional ``bias``, then normalizes the residual sum by
``sqrt(mean(sum * sum, axis=-1, keepdims=True) + epsilon)`` and multiplies
by ``gamma``. All inputs have the same ``FLOAT``, ``FLOAT16``, or ``BFLOAT16`` type.
``input`` has rank two or three, with shape ``[S, H]`` or ``[B, S, H]``.
``skip`` matches that shape; for rank-three input only, ``[S, H]`` and
``[1, S, H]`` also broadcast over the batch dimension. Sequence and hidden
dimensions do not broadcast. ``gamma`` and optional ``bias`` are rank-one
tensors of length ``H``. ``0 < H <= INT_MAX``; empty outer dimensions are valid.
``epsilon`` defaults to ``1e-12`` and must be finite and nonnegative.

Mandatory output slot 0 has the input shape and type. Optional output slot 3,
``input_skip_bias_sum``, contains the residual sum with the same shape and
type. Request it with the ONNX output list
``["Y", "", "", "input_skip_bias_sum"]``. Optional slot 1, ``mean``, contains
``FLOAT`` zeros; optional slot 2, ``inv_std_var``, contains the ``FLOAT``
inverse RMS. Both statistics have the input shape with the final dimension
replaced by one. They are independently optional; for example,
``["Y", "", "inv_std_var"]`` requests only the inverse RMS.
The bias may be omitted or represented by an empty input name. Trailing
optional output slots may likewise be omitted or left empty.

Arithmetic matches the ONNX Runtime CPU contract: the inputs are widened to
FP32 before residual addition, reduction, normalization, and scaling.
``FLOAT16`` and ``BFLOAT16`` values are narrowed only when storing the final
normalized and residual outputs, never before computing the mean square.
Statistics use this unrounded FP32 residual. The mean output is zero because
RMS normalization does not subtract a mean. This follows the
`pinned ONNX Runtime CPU implementation
<https://github.com/microsoft/onnxruntime/blob/f26e546fe25b04e737ec460f05a7d57f4938f23c/onnxruntime/contrib_ops/cpu/skip_layer_norm.cc>`_;
older ONNX Runtime versions and GPU implementations may not support these
statistics or BFLOAT16.

Use :func:`custom_op_schemas` or :func:`operator_schema_lookup` as the native
``GraphBuilder`` schema lookup callback and :func:`register_operator_support`
to register shape and memory support. Global and session-local kernel
registration helpers also register this metadata. Shape inference preserves
symbolic dimensions and records equality and batch-broadcast constraints.
CPU scratch memory is zero, excluding inputs and outputs: the FLOAT path uses
the normalized output buffer for the residual intermediate, while low-precision
paths recompute the widened residual rather than allocating a temporary buffer.
Saved statistics do not imply automatic differentiation support: no gradient
rules or fusion patterns are registered.

Support inventory
~~~~~~~~~~~~~~~~~

.. py:class:: OperatorSupport

   Immutable ``NamedTuple`` describing shape inference, peak memory, fusion
   patterns, and gradient availability for one custom or experimental operator.

.. py:function:: operator_support() -> tuple[OperatorSupport, ...]

   Returns the custom and experimental operator support inventory without registering or
   executing an implementation.

.. py:function:: register_operator_support() -> None

   Registers custom/experimental shape and peak-memory support and custom fusion patterns.

.. py:function:: register_custom_gradients(registry=None)

   Adds the ``CDist``, ``BiasGelu``, ``GroupQueryAttention``, and
   ``MatMulNBits`` backward rules to an independent ``GradRegistry`` and
   returns it.
