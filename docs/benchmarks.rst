Benchmarks
==========

A gallery of benchmarks comparing the SIMD-accelerated CPU kernels provided by
``onnx-light-cpu`` against other back-ends such as ``numpy``, ``onnxruntime``
and ``onnx-light``'s built-in reference kernels.

Reproducible Gemm and Attention baseline
----------------------------------------

The Gemm and Attention corpora used by the
:doc:`Gemm and MatMul roadmap <next_steps/2026_08_gemm_matmul>` are implemented
as C++ backend cases in ``TestMode::BENCHMARK``. This is the benchmark framework
provided by ``onnx-light``: cases are generated lazily in C++, exposed through
``CollectTestCases``, and consumed by the common benchmark recorder. The Gemm
cases live in
``onnx_light_cpu/backend_test/cases/math/cases_gemm.cc``; the Attention cases
live in ``onnx-light``'s C++ backend-test registry.

The Gemm corpus contains shape-forced cases for every prepared algorithm:
``direct`` (small K), ``skinny_m``, ``skinny_n``, ``split_k`` (large K with a
small output), and square/transformer shapes (general five-loop). Run the
recorder in separate processes to compare one thread with the configured pool
while keeping the same binary and inputs:

.. code-block:: bash

   ONNX_LIGHT_CPU_NUM_THREADS=1 <benchmark-command>
   ONNX_LIGHT_CPU_NUM_THREADS=6 <benchmark-command>

The requested count cannot exceed the build-time
``ONNX_LIGHT_CPU_MAX_THREADS`` limit. With no request, the pool uses physical
cores before SMT siblings and prioritizes P-cores on hybrid processors. Worker
affinity is applied on Linux and Windows. ``ONNX_LIGHT_CPU_SPIN_COUNT`` selects
the bounded spin-before-park budget; use ``0`` to measure immediate parking.
Both environment settings are read once when the shared pool is initialized,
so changing them inside a running process has no effect.

.. toctree::
    :maxdepth: 1

    auto_examples/benchmarks/index
