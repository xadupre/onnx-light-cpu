.. _l-design:

Design
======

``onnx-light-cpu`` is an extension of `onnx-light
<https://github.com/xadupre/onnx-light>`_: it installs fast CPU kernels into
onnx-light's shared C++ dispatch table. It accelerates standard ONNX operators
in the ``ai.onnx`` domain and operators in the ``com.microsoft`` domain. The
pages below explain registration, runtime parallel execution, implementation
selection, benchmarks, and the main kernel families.

.. toctree::
    :maxdepth: 1

    registering_kernels
    parallel_execution
    kernel_selection
    benchmark_methodology
    kernels/index
