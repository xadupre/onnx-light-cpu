Kernel registration
-------------------

When onnx-light-cpu is built with ``-DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``
(which requires the `onnx-light <https://github.com/xadupre/onnx-light>`_ C++
package), it produces ``lib_onnx_light_cpu_kernels``. Its runtime inventory is
available from :cpp:func:`CollectRegisteredKernels` and the generated
:ref:`ByOp catalogue <l-cpu-by-op>`.

.. doxygenfunction:: onnx_light_cpu::RegisterAbsKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterExpKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterLogKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterGemmKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterNotKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterAllKernels()
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterAllKernels(MicrosoftKernelImplementation)
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterKernelGlobal
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterAllKernelsGlobal
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterKernelForSession
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterAllKernelsForSession
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterMicrosoftKernels
   :project: onnx_light_cpu

.. doxygenenum:: onnx_light_cpu::MicrosoftKernelImplementation
   :project: onnx_light_cpu
