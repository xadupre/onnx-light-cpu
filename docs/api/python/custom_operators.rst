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
