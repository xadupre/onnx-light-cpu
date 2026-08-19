.. _benchmarks-gallery:

Benchmarks
==========

A gallery of benchmarks comparing the SIMD-accelerated CPU kernels provided by
``onnx-light-cpu`` against other back-ends such as ``numpy``, ``onnxruntime``
and ``onnx-light``'s built-in reference kernels.

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
small output), and square/transformer shapes (general five-loop). Every shape
is registered for each element type the ``GemmKernel`` implements -- ``float32``,
``float16`` and ``bfloat16`` -- so the corpus also measures the fp16/bf16
widen/round-trip overhead. Standalone recorders run on their calling thread.
Integration benchmarks compare participant counts through the ``onnx-light``
session ``cpu_execution`` policy, which owns thread count, affinity, and spin
behavior. ``onnx-light-cpu`` does not create workers or read thread-control
environment variables.
