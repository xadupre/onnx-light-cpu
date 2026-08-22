Kernels
=======

The pages below list the kernels **provided by this repository**
(``onnx-light-cpu``). They do not include the kernels that come with
onnx-light itself. They are generated automatically at documentation build
time from ``onnx_light_cpu.registered_kernels()``, the public inventory of the
C++ registrations the runtime actually executes, so they always reflect the
kernels this repository actually provides -- one stable page per registration,
without any documentation-side operator list to keep in sync.

.. toctree::
   :maxdepth: 1

   kernels_generated/index

Each kernel implements an ONNX operator for the given element data type. The
``Abs`` kernels compute the elementwise absolute value, the ``Exp`` kernels the
elementwise natural exponential, the ``Log`` kernels the elementwise natural
logarithm, the ``Gemm`` kernels the general matrix multiplication
``alpha * op(A) @ op(B) + beta * C``, and the ``Not`` kernel the elementwise
logical negation of a ``bool`` tensor. Each kernel dispatches at runtime to the
best available instruction set (AVX-512, AVX2, AVX, SSE2, or a scalar fallback),
selected once through CPUID-based CPU feature detection.
