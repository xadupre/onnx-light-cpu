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

The activation may have any positive rank with final dimension ``K``. Shape
inference preserves every leading concrete or symbolic dimension and replaces
the final dimension with ``N``. INT4 uses bounded panels of 8 rows, 32 output
columns, and 32 reduction elements, with scalar, AVX2, or AVX-512 dispatch.
Scratch is 6,144 bytes per active callback (4,096 decoded weight bytes,
1,024 activation bytes, and 1,024 accumulator bytes), independent of matrix
size. INT2/INT8 retain their allocation-free scalar path. Packed constants are
borrowed directly on every invocation: there is no kernel preparation,
persistent repacking, packed-weight copy, or full floating-point weight matrix.
Inputs and outputs are excluded from scratch accounting. The registered
peak-memory function still returns zero heap scratch; the bounded worker
stack storage is reported separately by the projection benchmark.

Explicit zero points, deprecated ``g_idx``, provider-prepacked weights, other
bit widths, block sizes, mixed floating-point types, and ``DOUBLE`` are
rejected rather than silently using a different layout. The registered gradient differentiates the
activation and optional bias while treating packed weights and scales as
constants. The bias fusion applies only to an exclusively consumed
``MatMulNBits`` output followed by a compatible rank-one ``Add``.

Muse-Glimmer projections
^^^^^^^^^^^^^^^^^^^^^^^

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
process memory. Large vocabulary cases are opt-in because model serialization
and ONNX Runtime preparation can require several GB.
For isolated single-thread FP32 kernel measurements, build with
``-DONNX_LIGHT_CPU_BUILD_BENCHMARKS=ON`` and run
``matmul_nbits_throughput M K N repeats``. This reports input preparation,
median steady-state seconds, packed bytes, zero kernel weight-copy bytes,
scratch bytes, and the selected implementation.

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
