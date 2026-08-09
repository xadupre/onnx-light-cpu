Kernels
=======

The table below lists the kernels **provided by this repository**
(``onnx-light-cpu``). It does not include the kernels that come with onnx-light
itself. The list is generated automatically at documentation build time by
scanning this repository's public C++ headers
(``onnx_light_cpu/impl/math/math_kernels.h`` and
``onnx_light_cpu/impl/logical/logical_kernels.h``), so it always reflects the
kernels this repository actually provides.

.. registered-kernels::

Each kernel implements an ONNX operator for the given element data type. The
``Abs`` kernels compute the elementwise absolute value, the ``Exp`` kernels the
elementwise natural exponential, the ``Log`` kernels the elementwise natural
logarithm, the ``Gemm`` kernels the general matrix multiplication
``alpha * op(A) @ op(B) + beta * C``, and the ``Not`` kernel the elementwise
logical negation of a ``bool`` tensor. Each kernel dispatches at runtime to the
best available instruction set (AVX-512, AVX2, AVX, SSE2, or a scalar fallback),
selected once through CPUID-based CPU feature detection.
