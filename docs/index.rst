onnx-light-cpu
==============

Highly optimized CPU kernels for
`onnx-light <https://github.com/xadupre/onnx-light>`_.

It implements ONNX operators with SIMD-accelerated kernels that dispatch at
runtime to the best available instruction set (AVX-512, AVX2, AVX, SSE2, or a
scalar fallback). The optimal implementation is selected once through
CPUID-based CPU feature detection and cached, so the dispatch overhead is paid
only once.

The kernels can be used directly from C++ or installed into onnx-light's shared
C++ kernel dispatch table so any ONNX model using a supported operator runs the
optimized kernel when evaluated through a ``ReferenceEvaluator``. See
:doc:`getting_started` to install the package and run your first model, the
:doc:`kernels` page for the list of operators provided by this repository, the
:doc:`examples` gallery for runnable examples, and the :ref:`benchmarks-gallery` gallery
for performance comparisons against other back-ends.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   getting_started
   kernels
   design/index
   api/index
   examples
   next_steps/index
