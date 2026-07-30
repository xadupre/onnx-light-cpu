API Reference
=============

C++ API
-------

The public C++ API is declared in ``onnx_light_cpu/cpu_kernels.h``. Every
kernel dispatches at runtime to the best available SIMD path.

.. code-block:: cpp

   namespace onnx_light_cpu {

   enum class SimdLevel : int {
     kNone = 0,   // Scalar fallback (no SIMD).
     kSSE2 = 1,   // SSE2 (128-bit).
     kAVX = 2,    // AVX (256-bit).
     kAVX2 = 3,   // AVX2 (256-bit with FMA, integer ops).
     kAVX512 = 4, // AVX-512F (512-bit).
   };

   SimdLevel DetectSimdLevel();

   void AbsFloat32(const float *input, float *output, std::size_t count);
   void AbsFloat64(const double *input, double *output, std::size_t count);
   void AbsInt32(const std::int32_t *input, std::int32_t *output, std::size_t count);
   void AbsInt64(const std::int64_t *input, std::int64_t *output, std::size_t count);

   } // namespace onnx_light_cpu

Python API
----------

.. py:module:: onnx_light_cpu.onnx_py._cpukernels

.. py:function:: detect_simd_level() -> int

   Returns the detected SIMD level: ``0=None``, ``1=SSE2``, ``2=AVX``,
   ``3=AVX2``, ``4=AVX512``.

.. py:function:: abs(input)

   Computes the elementwise absolute value of a 1-D array using the optimized
   SIMD dispatch. ``input`` must be a contiguous CPU array with dtype
   ``float32``, ``float64``, ``int32`` or ``int64``; the function dispatches on
   the dtype and returns a new array of the same dtype, like :func:`numpy.abs`.

.. py:function:: has_cpu_kernels() -> bool

   Returns ``True`` when the CPU kernel extension is available.

Registering kernels with onnx-light
-----------------------------------

.. py:module:: onnx_light_cpu

.. py:function:: register_kernels(sess, domain="")

   Registers the onnx-light-cpu kernels on an ``onnx-light``
   ``ReferenceEvaluator`` (any object exposing a compatible
   ``register_custom_kernel(domain, op_type, fn)`` method). After this call,
   every ``Abs`` node evaluated by ``sess`` dispatches to the
   SIMD-accelerated onnx-light-cpu kernel instead of the built-in one, so any
   ONNX model using ``Abs`` benefits from the optimized kernel. Returns
   ``sess`` so calls can be chained.

   .. code-block:: python

      import numpy as np
      from onnx_light.onnx.reference import ReferenceEvaluator
      from onnx_light_cpu import register_kernels

      sess = ReferenceEvaluator(model)
      register_kernels(sess)
      (y,) = sess.run(None, {"x": np.array([-1.0, 2.0], dtype=np.float32)})
