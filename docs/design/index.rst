.. _l-design:

Design
======

``onnx-light-cpu`` installs its SIMD-accelerated kernels into onnx-light's
shared C++ dispatch table. The pages below explain how the kernels are
registered and how the matrix-multiplication kernels are designed.

.. toctree::
    :maxdepth: 1

    registering_kernels
    gemm_kernel_design
