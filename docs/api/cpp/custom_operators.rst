Custom operators
----------------

The custom operator support inventory and its public implementations are
available through these APIs.

.. doxygenfunction:: onnx_light_cpu::CollectOperatorSupport
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterCustomOperatorGradients
   :project: onnx_light_cpu

.. doxygenclass:: onnx_light_cpu::BiasGeluFusionPattern
   :project: onnx_light_cpu

.. doxygenclass:: onnx_light_cpu::CDistFusionPattern
   :project: onnx_light_cpu

.. doxygenclass:: onnx_light_cpu::GroupQueryAttentionFusionPattern
   :project: onnx_light_cpu
