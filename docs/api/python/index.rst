Python API
----------

.. py:module:: onnx_light_cpu

.. py:class:: SimdLevel

   An ``IntEnum`` naming the instruction-set levels used for CPU kernel
   dispatch: ``NONE``, ``SSE2``, ``AVX``, ``AVX2``, and ``AVX512``.

.. py:function:: detect_simd_level() -> SimdLevel

   Returns the highest SIMD instruction-set level available to this process.
   The result remains compatible with integer comparisons.

.. py:function:: has_cpu_kernels() -> bool

   Returns whether the compiled onnx-light-cpu kernel extension is available.

.. toctree::
   :maxdepth: 1

   registration
   kernel_inventory
   custom_operators
   processor_performance
