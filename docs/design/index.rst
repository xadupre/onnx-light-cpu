.. _l-design:

Design
======

``onnx-light-cpu`` installs its SIMD-accelerated kernels into onnx-light's
shared C++ dispatch table. The pages below explain how the kernels are
registered, how benchmarks must be constructed, and how the main kernel
families are designed.

.. toctree::
    :maxdepth: 1

    registering_kernels
    benchmark_methodology
    kernels/index
