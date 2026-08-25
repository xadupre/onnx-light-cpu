onnx-light-cpu
==============

Highly optimized CPU kernels for
`onnx-light <https://github.com/xadupre/onnx-light>`_.

It implements ONNX operators with SIMD-accelerated kernels that dispatch at
runtime to the best available instruction set (AVX-512, AVX2, AVX, SSE2, or a
scalar fallback). The optimal implementation is selected once through
CPUID-based CPU feature detection and cached, so the dispatch overhead is paid
only once.

The kernels can be used directly from C++ or installed into onnx-light's shared
C++ kernel dispatch table so any ONNX model using a supported operator runs the
optimized kernel when evaluated through a ``ReferenceEvaluator``. See
:doc:`getting_started` to install the package and run your first model, the
:doc:`kernels` page for the list of operators provided by this repository, the
:doc:`examples` gallery for runnable examples, the :ref:`benchmarks-gallery`
gallery for performance comparisons against other back-ends, and the
:ref:`processor-performance-gallery` for a profile of the current host.

Registering the optimized kernels
---------------------------------

Registration installs every shipped ``onnx-light-cpu`` kernel into
``onnx-light``'s shared CPU dispatch table. It is process-wide, so call it once
before creating the evaluators or runtime sessions that should use the
accelerated implementations.

.. tab-set::

   .. tab-item:: Python

      Build with the ``onnx-light`` integration enabled, then call
      ``onnx_light_cpu.register_kernels()`` before constructing the evaluator:

      .. code-block:: python

         from onnx_light.onnx.reference import ReferenceEvaluator
         from onnx_light_cpu import register_kernels

         register_kernels()
         session = ReferenceEvaluator(model)
         outputs = session.run(None, feeds)

      The Python entry point is available in builds configured with
      ``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``.

   .. tab-item:: C++

      Link ``onnx_light_cpu::lib_onnx_light_cpu_kernels`` and register the
      kernels before constructing an ``onnx-light`` runtime session:

      .. code-block:: cpp

         #include <onnx_light_cpu/kernels/register_kernels.h>

         int main() {
           onnx_light_cpu::RegisterAllKernels();

           // RuntimeSession and ReferenceEvaluator now resolve supported
           // CPU operators to the onnx-light-cpu implementations.
         }

      The native integration is built with
      ``-DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``.

Both entry points update the same shared C++ ``KernelDispatchTable``. See
:doc:`design/registering_kernels` for per-session overrides, kernel-usage
inspection, custom registrations, and troubleshooting when two builds link
different copies of ``lib_onnx_core``.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   getting_started
   kernels
   design/index
   api/index
   examples
   next_steps/index
