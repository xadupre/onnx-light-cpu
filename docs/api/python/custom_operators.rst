Custom operators
----------------

.. py:function:: custom_op_schemas(op_type="", init_doc=True)

   Returns ``LightOpSchema`` records for supported ``com.microsoft`` operators.

.. py:function:: operator_schema_lookup(op_type)

   Returns standard ONNX schemas and this package's custom schemas. Pass it as
   ``GraphBuilder(..., schema_lookup=operator_schema_lookup)``.

.. py:class:: OperatorSupport

   Immutable ``NamedTuple`` describing shape inference, peak memory, fusion
   patterns, and gradient availability for one custom operator.

.. py:function:: operator_support() -> tuple[OperatorSupport, ...]

   Returns the custom operator support inventory without registering or
   executing an implementation.

.. py:function:: register_operator_support() -> None

   Registers custom shape-inference, peak-memory, and fusion-pattern support.

.. py:function:: register_custom_gradients(registry=None)

   Adds the ``CDist`` and ``BiasGelu`` backward rules to an independent
   ``GradRegistry`` and returns it.
