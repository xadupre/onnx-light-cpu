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
         -DONNX_LIGHT_CPU_BUILD_PYTHON=OFF
   cmake --build build

Quick Start (C++)
-----------------

Include the public header and call one of the kernel functions; the best
available SIMD path is selected automatically at runtime:

.. code-block:: cpp

   #include <onnx_light_cpu/impl/math/math_kernels.h>

   int main() {
       float input[] = {-1.0f, 2.0f, -3.0f, 4.0f};
       float output[4];
       onnx_light_cpu::AbsFloat32(input, output, 4);
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
