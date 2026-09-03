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
:doc:`standalone_cpp` page for a complete standalone C++ inference program, the
:doc:`byop` page for the list of operators provided by this repository, the
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

      To override an operator for one evaluator only, leave the shared table
      unchanged and register a Python kernel on that session:

      .. code-block:: python

         import numpy as np
         from onnx_light.onnx.reference import ReferenceEvaluator

         session = ReferenceEvaluator(model)
         session.register_custom_kernel("", "Abs", lambda node, x: np.abs(x))
         outputs = session.run(None, feeds)

      Other evaluators keep their existing ``Abs`` implementation.

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

      To override an operator for one C++ runtime context only, register its
      callback on that context before the session's first ``Run``:

      .. code-block:: cpp

         #include <onnx_core/runtime/kernels/kernel_context.h>
         #include <onnx_core/runtime/memory/simple_tensor.h>
         #include <onnx_core/runtime/runtime_context.h>
         #include <onnx_core/runtime/runtime_session.h>

         #include <cmath>
         #include <cstddef>
         #include <cstdint>
         #include <stdexcept>
         #include <vector>

         namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;

         void RegisterAbsForSession(rt::RuntimeContext &context) {
           context.RegisterCustomKernel(
               "", "Abs", [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                              rt::RuntimeContext &context) {
                 const rt::Tensor &input = context.Get(node.input(0));
                 if (input.data_type != static_cast<std::int32_t>(rt::DataType::FLOAT)) {
                   throw std::invalid_argument("This example Abs kernel requires FLOAT input.");
                 }
                 std::vector<float> output(static_cast<std::size_t>(input.element_count()));
                 for (std::size_t i = 0; i < output.size(); ++i) {
                   output[i] = std::fabs(input.AsFloat()[i]);
                 }
                 context.Set(node.output(0),
                             rt::Tensor::FromFloat(node.output(0), input.shape, output));
               });
         }

         rt::RuntimeContext context(rt::KernelContext(rt::DefaultOpset(18)));
         RegisterAbsForSession(context);
         rt::RuntimeSession session(context.GetExecutionPlan(graph));
         session.Run(context);

      ``RuntimeSession`` caches resolved kernels on its first run, so register
      the callback before then. Other contexts continue to use their existing
      ``Abs`` implementation.

The shipped-kernel Python and C++ entry points above update the same shared C++
``KernelDispatchTable``. See :doc:`design/registering_kernels` for per-session
overrides, kernel-usage inspection, custom registrations, and troubleshooting
when two builds link different copies of ``lib_onnx_core``.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   getting_started
   standalone_cpp
   design/index
   byop
   api/index
   examples
   next_steps/index
