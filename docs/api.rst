API Reference
=============

C++ API
-------

The public C++ API is declared in ``onnx_light_cpu/impl/math/math_kernels.h``
and ``onnx_light_cpu/impl/logical/logical_kernels.h``. The shared SIMD
dispatch primitives (``SimdLevel`` and ``DetectSimdLevel``) live in the neutral
``onnx_light_cpu/impl/simd_level.h`` header, which both kernel families
include. Every kernel dispatches at runtime to the best available SIMD path.

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

onnx-light kernel class
~~~~~~~~~~~~~~~~~~~~~~~~

When onnx-light-cpu is built with ``-DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON`` (which
requires the `onnx-light <https://github.com/xadupre/onnx-light>`_ C++ package),
an additional library ``lib_onnx_light_cpu_kernels`` is produced. It declares
``onnx_light_cpu/kernels/math/abs_kernel.h``,
``onnx_light_cpu/kernels/math/exp_log_kernel.h`` and
``onnx_light_cpu/kernels/logical/not_kernel.h``:

.. code-block:: cpp

   namespace onnx_light_cpu {

   // Derives from onnx_light::core::runtime::KernelBase and delegates to the
   // SIMD Abs* routines above.
   class AbsKernel : public onnx_light::core::runtime::KernelBase { ... };

   // Exp/Log equivalents delegating to the SIMD Exp*/Log* routines.
   class ExpKernel : public onnx_light::core::runtime::KernelBase { ... };
   class LogKernel : public onnx_light::core::runtime::KernelBase { ... };

   // Not equivalent delegating to the SIMD NotBool routine.
   class NotKernel : public onnx_light::core::runtime::KernelBase { ... };

   // Registers the kernels into onnx-light's shared KernelDispatchTable for the
   // default ONNX domain / CPU device, overriding the built-in operators.
   void RegisterKernels();     // Abs
   void RegisterExpKernel();   // Exp
   void RegisterLogKernel();   // Log
   void RegisterNotKernel();   // Not

   } // namespace onnx_light_cpu

``AbsKernel``, ``ExpKernel``, ``LogKernel`` and ``NotKernel`` are full
``KernelBase`` subclasses, so once the matching ``Register*`` function has run
every ``Abs``/``Exp``/``Log``/``Not`` node executed by onnx-light's runtime (and
therefore any model run through ``ReferenceEvaluator``) resolves to the
SIMD-accelerated kernel.

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

.. py:function:: logical_not(input)

   Computes the elementwise logical negation of a 1-D ``bool`` array using the
   optimized SIMD dispatch and returns a new ``bool`` array, like
   :func:`numpy.logical_not`.

.. py:function:: has_cpu_kernels() -> bool

   Returns ``True`` when the CPU kernel extension is available.

Registering kernels with onnx-light
-----------------------------------

.. py:module:: onnx_light_cpu

.. py:function:: register_kernels(sess, domain="")

   Registers the onnx-light-cpu kernels on an ``onnx-light``
   ``ReferenceEvaluator`` (any object exposing a compatible
   ``register_custom_kernel(domain, op_type, fn)`` method). After this call,
   every ``Abs``, ``Exp``, ``Log`` and ``Not`` node evaluated by ``sess``
   dispatches to the SIMD-accelerated onnx-light-cpu kernel instead of the
   built-in one, so any ONNX model using those operators benefits from the
   optimized kernel. Returns ``sess`` so calls can be chained.

   .. code-block:: python

      import numpy as np
      from onnx_light.onnx.reference import ReferenceEvaluator
      from onnx_light_cpu import register_kernels

      sess = ReferenceEvaluator(model)
      register_kernels(sess)
      (y,) = sess.run(None, {"x": np.array([-1.0, 2.0], dtype=np.float32)})
