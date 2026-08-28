Getting Started
===============

Installation
------------

Install from source:

.. code-block:: bash

   pip install .

Or with Pixi:

.. code-block:: bash

   pixi install
   pixi run install

Or build with CMake (C++ only):

.. code-block:: bash

   cmake -S . -B build -DONNX_LIGHT_CPU_BUILD_TESTS=ON \
         -DONNX_LIGHT_CPU_BUILD_PYTHON=OFF \
         -DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON
   cmake --build build

The C++ quick start requires the onnx-light C++ package to be installed so
``find_package(onnx_light)`` can locate it.

Quick Start (C++)
-----------------

Register the optimized kernels, then execute the graph through onnx-light's
runtime. Runtime dispatch selects the registered onnx-light-cpu kernel and its
best available SIMD path:

.. code-block:: cpp

   #include <onnx_light_cpu/kernels/register_kernels.h>

   #include <onnx_core/runtime/kernels/kernel_context.h>
   #include <onnx_core/runtime/memory/simple_tensor.h>
   #include <onnx_core/runtime/runtime_context.h>
   #include <onnx_core/runtime/runtime_session.h>
   #include <onnx_proto/onnx_helper.h>

   int main() {
       namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;

       onnx_light_cpu::RegisterAllKernels();

       ONNX_LIGHT_NAMESPACE::GraphProto graph;
       graph.ref_node().push_back(
           ONNX_LIGHT_NAMESPACE::MakeNode("Abs", {"x"}, {"y"}));

       rt::RuntimeContext context(rt::KernelContext(rt::DefaultOpset(18)));
       context.Set(
           "x", rt::Tensor::FromFloat("x", {4}, {-1.0f, 2.0f, -3.0f, 4.0f}));
       rt::RuntimeSession session(context.GetExecutionPlan(graph));
       session.Run(context);

       const float *output = context.Get("y").AsFloat();
       // output = {1.0f, 2.0f, 3.0f, 4.0f}
   }

Quick Start (Python)
--------------------

.. code-block:: python

   from onnx_light_cpu import register_kernels
   from onnx_light_cpu.onnx_py._cpukernels import detect_simd_level, has_cpu_kernels

   # Check what SIMD level is available:
   # 0=None, 1=SSE2, 2=AVX, 3=AVX2, 4=AVX512
   print("CPU kernels available:", has_cpu_kernels())
   print("SIMD level:", detect_simd_level())

   # When onnx-light is installed, register the optimized kernels globally so
   # ReferenceEvaluator uses them for supported ONNX operators.
   register_kernels()
