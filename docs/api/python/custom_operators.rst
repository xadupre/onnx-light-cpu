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

``stash_type`` does not select arithmetic precision. Work uses FP64 if either
``X`` or ``Scale`` is ``DOUBLE``, or if the shape of ``Scale`` is not exactly
the normalized suffix ``X.shape[axis:]``; otherwise it uses FP32. There is no intermediate
low-precision rounding before multiplication by ``Scale``. The FP32 suffix
path reuses the optimized RMS mean-square and affine engine, saving optional
statistics without repeating the reduction. CPU scratch memory is zero,
excluding inputs and outputs. This is inference compatibility support: no
gradient rules or fusion patterns are registered for this operator.

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

   Adds the ``CDist`` and ``BiasGelu`` backward rules to an independent
   ``GradRegistry`` and returns it.
