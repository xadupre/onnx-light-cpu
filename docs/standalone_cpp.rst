Standalone C++ inference
========================

The repository includes a complete, independent CMake consumer in
``examples/cpp/standalone_inference``. It registers the
``onnx-light-cpu`` kernels, runs an ONNX ``Abs`` model through
``onnx-light``'s native runtime, and verifies that
``onnx_light_cpu::Abs`` produced the output.

Build and install ``onnx-light`` first. Then configure and install this
project with its native integration enabled into the same prefix:

.. code-block:: bash

    cmake -S . -B build-install \
          -DCMAKE_BUILD_TYPE=Release \
          -DONNX_LIGHT_CPU_BUILD_PYTHON=OFF \
          -DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON \
          -DCMAKE_PREFIX_PATH=/path/to/prefix \
          -DCMAKE_INSTALL_PREFIX=/path/to/prefix
    cmake --build build-install --parallel
    cmake --install build-install

The example is deliberately built as a separate downstream project:

.. code-block:: bash

    cmake -S examples/cpp/standalone_inference -B build-standalone \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/path/to/prefix
    cmake --build build-standalone --parallel
    ./build-standalone/onnx_light_cpu_inference

Without an argument, the executable creates a minimal ONNX ``Abs`` model in
memory and evaluates ``[-1, 2, -3.5, 4]``. A compatible ONNX file can be
loaded instead:

.. code-block:: bash

    ./build-standalone/onnx_light_cpu_inference model.onnx

The input model must have one FLOAT input of shape ``[4]`` and one FLOAT
output. The executable uses the model's first declared input name and prints
the output and the dispatched kernel name.
