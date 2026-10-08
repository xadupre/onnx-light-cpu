C++ API
-------

The public C++ API is declared in ``onnx_light_cpu/kernels``. Signatures are
extracted from public header comments by Doxygen and rendered through Breathe.
The kernel and custom-operator pages are generated from Doxygen XML at build
time; implementation notes remain in ``kernels_notes.rst``.

.. toctree::
   :maxdepth: 1

   kernels
   registration
   custom_operators
   runtime_and_simd
