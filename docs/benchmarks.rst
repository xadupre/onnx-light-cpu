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

.. toctree::
    :maxdepth: 1

    auto_benchmarks/index
