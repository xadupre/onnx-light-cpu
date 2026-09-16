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

Diagnosing native-extension import failures
------------------------------------------

Before attributing an undefined symbol to import order, rebuild both projects
from clean source trees with no existing native artifacts. Reusing an in-place
build can mix extensions and shared libraries from different configurations.
Use absolute source paths and fresh build directories; do not install either
project into site-packages or use editable installs for this check.

For example, on Linux, with the Python build dependencies already available:

.. code-block:: bash

   export ONNX_SOURCE=/absolute/path/to/onnx-light
   export CPU_SOURCE=/absolute/path/to/onnx-light-cpu
   export BUILD_ROOT="$(mktemp -d)"

   cmake -S "$ONNX_SOURCE" -B "$BUILD_ROOT/onnx-light" \
       -DCMAKE_BUILD_TYPE=Release -DONNX_LIGHT_INSTALL=OFF \
       -DPython_EXECUTABLE="$(command -v python)"
   cmake --build "$BUILD_ROOT/onnx-light" --parallel 4
   cmake --install "$BUILD_ROOT/onnx-light" --prefix "$ONNX_SOURCE"

   export PYTHONPATH="$CPU_SOURCE:$ONNX_SOURCE"
   python "$CPU_SOURCE/setup.py" build_ext --inplace --onnx-light-source \
       --build-temp "$BUILD_ROOT/onnx-light-cpu" --parallel 4

   cd "$CPU_SOURCE"
   python -m unittest \
       unittests.python.test_kernels_e2e.TestExtensionImportOrder -v

Here ``cmake --install`` places the dependency's freshly built libraries in its
source tree, not in site-packages. ``--onnx-light-source`` links against those
same runtime libraries rather than rebuilding a second copy.

The regression test imports ``_cpukernels`` then ``_cpuregister``, and the
reverse order, in separate Python subprocesses. It runs from both the checkout
and a temporary working directory, verifies that the CPU extensions came from
the checkout, and exercises kernel registration. A failure includes the child
process's output and traceback. If a clean build passes but reused artifacts
fail, record both source revisions and build commands with the report rather
than changing symbol visibility or preloading libraries to mask the failure.

Quick Start
-----------

Register the optimized kernels, then execute the graph through onnx-light's
runtime. Runtime dispatch selects the registered onnx-light-cpu kernel and its
best available SIMD path:

.. tab-set::

   .. tab-item:: Python

      .. code-block:: python

         from onnx_light_cpu import detect_simd_level, has_cpu_kernels, register_kernels

         # Check what SIMD level is available:
         print("CPU kernels available:", has_cpu_kernels())
         print("SIMD level:", detect_simd_level())

         # When onnx-light is installed, register the optimized kernels globally so
         # ReferenceEvaluator uses them for supported ONNX operators.
         register_kernels()

   .. tab-item:: C++

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
