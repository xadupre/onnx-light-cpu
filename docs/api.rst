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

Parallel iteration helper
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``onnx_light_cpu/impl/parallel_for.h`` provides a header-only, cost-aware
``ParallelFor`` built on a persistent thread pool (workers are created once and
reused). Before dispatching any worker threads it consults a small cost model,
``ParallelForBlockCount``, that combines the processor (hardware thread count)
with an estimate of the loop cost (iteration count times ``cost_per_element``).
Ranges that are too small — or whose per-element work is too cheap — run inline
on the calling thread, because waking worker threads would cost more than the
work saved. Every block is disjoint and covers the range exactly once, so
element-wise results are independent of the thread count (bit-exact).

When a kernel vectorises its inner loop it processes several values per SIMD
instruction (e.g. AVX handles 8 ``float`` at a time, AVX-512 handles 16).
``ParallelFor`` therefore accepts a ``block_multiple`` argument and rounds every
block size up to a multiple of it, so each block is a whole number of SIMD
vectors and the vectorised loop never processes a scalar remainder in the middle
of the range. ``ParallelForSimdLanes<T>()`` returns the lane count for element
type ``T`` at the widest supported register (AVX-512, 64 bytes): 16 for
``float``, 8 for ``double``, 32 for a 2-byte half, 64 for ``std::int8_t``.

.. code-block:: cpp

   namespace onnx_light_cpu {

   // Number of threads ParallelFor may use (>= 1, includes the caller).
   std::int64_t ParallelForThreadCount() noexcept;

   // Number of T elements the widest SIMD register processes at once (>= 1).
   template <typename T> constexpr std::int64_t ParallelForSimdLanes() noexcept;

   // Cost model: number of blocks to split [0, total) into given a relative
   // per-iteration cost. Returns 1 to mean "run inline, do not parallelize".
   std::int64_t ParallelForBlockCount(std::int64_t total,
                                      double cost_per_element = 1.0) noexcept;

   // Runs fn(begin, end) over disjoint sub-ranges covering [0, total). The
   // cost_per_element overload lets heavier kernels parallelize smaller ranges;
   // block_multiple keeps each block a whole number of SIMD vectors.
   template <typename Fn> void ParallelFor(std::int64_t total, Fn fn);
   template <typename Fn>
   void ParallelFor(std::int64_t total, double cost_per_element, Fn fn);
   template <typename Fn>
   void ParallelFor(std::int64_t total, double cost_per_element,
                    std::int64_t block_multiple, Fn fn);

   } // namespace onnx_light_cpu

Every public kernel (``Abs*``, ``Exp*``/``Log*`` and ``NotBool``) already routes
its work through ``ParallelFor``. The memory-bandwidth-bound ``Abs``/``Not``
kernels pass ``cost_per_element = 1`` (so they only parallelize on large arrays),
while the compute-bound ``Exp``/``Log`` kernels pass a higher cost so the same
sized ranges parallelize sooner. Each kernel also passes
``ParallelForSimdLanes<T>()`` as ``block_multiple`` so the parallel blocks align
with its SIMD vectors. Because every block is disjoint the results are
unchanged relative to the single-threaded kernels.


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

   // Gemm equivalent delegating to the register-blocked SIMD Gemm* routines.
   class GemmKernel : public onnx_light::core::runtime::KernelBase { ... };

   // Registers the kernels into onnx-light's shared KernelDispatchTable for the
   // default ONNX domain / CPU device, overriding the built-in operators.
   void RegisterAbsKernel();   // Abs
   void RegisterExpKernel();   // Exp
   void RegisterLogKernel();   // Log
   void RegisterGemmKernel();  // Gemm
   void RegisterNotKernel();   // Not

   // Convenience wrapper registering all of the kernels above in one call.
   void RegisterAllKernels();

   } // namespace onnx_light_cpu

``AbsKernel``, ``ExpKernel``, ``LogKernel``, ``GemmKernel`` and ``NotKernel`` are
full ``KernelBase`` subclasses, so once the matching ``Register*`` function (or
``RegisterAllKernels``) has run every ``Abs``/``Exp``/``Log``/``Gemm``/``Not``
node executed by onnx-light's runtime (and therefore any model run through
``ReferenceEvaluator``) resolves to the SIMD-accelerated kernel.

Python API
----------

.. py:module:: onnx_light_cpu.onnx_py._cpukernels

.. py:function:: detect_simd_level() -> int

   Returns the detected SIMD level: ``0=None``, ``1=SSE2``, ``2=AVX``,
   ``3=AVX2``, ``4=AVX512``.

.. py:function:: has_cpu_kernels() -> bool

   Returns ``True`` when the CPU kernel extension is available.

The compiled kernels themselves (``Abs``, ``Exp``, ``Log``, ``Gemm``, ``Not``)
are not exposed as numpy-like Python functions; they are reachable through
onnx-light's runtime after registration (see :func:`register_all_kernels` and
:func:`onnx_light_cpu.register_kernels`).

.. py:module:: onnx_light_cpu.onnx_py._cpuregister

.. py:function:: register_all_kernels() -> None

   Registers every onnx-light-cpu kernel class (``Abs``, ``Exp``, ``Log``,
   ``Gemm`` and ``Not``) into onnx-light's C++ ``KernelDispatchTable`` for the
   CPU device, replacing the corresponding built-in entries for the default
   ONNX domain.
   This is the Python binding for the C++ :cpp:func:`RegisterAllKernels`
   function and is only available in builds compiled with the onnx-light
   integration enabled (``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``).


Registering kernels with onnx-light
-----------------------------------

.. py:module:: onnx_light_cpu

.. py:function:: register_kernels(sess=None)

   Registers the onnx-light-cpu kernels into onnx-light's shared C++
   ``KernelDispatchTable`` for the CPU device by calling
   :func:`register_all_kernels`. After this call, every ``Abs``, ``Exp``,
   ``Log``, ``Gemm`` and ``Not`` node executed by onnx-light's runtime (and
   therefore any model run through ``ReferenceEvaluator``) dispatches to the
   SIMD-accelerated onnx-light-cpu kernel instead of the built-in one. The
   registration is global, so ``sess`` is optional and only returned unchanged
   so calls can be chained.

   .. code-block:: python

      import numpy as np
      from onnx_light.onnx.reference import ReferenceEvaluator
      from onnx_light_cpu import register_kernels

      register_kernels()
      sess = ReferenceEvaluator(model)
      (y,) = sess.run(None, {"x": np.array([-1.0, 2.0], dtype=np.float32)})
