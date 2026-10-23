Completed Plans — 2026
======================

Completed performance roadmaps from 2026. Unfinished work remains on the
:doc:`next-steps page <../index>`.

.. toctree::
    :maxdepth: 1
    :hidden:

    2026_08_gemm_matmul
    2026_08_attention
    2026_08_exp_log_parity
    2026_08_binary_elementwise
    2026_08_binary_elementwise_performance
    2026_08_tree_ensemble
    2026_08_runtime_execution_controls
    2026_08_processor_performance_profile
    2026_08_registered_kernel_documentation
    2026_08_kernel_performance_improvements
    2026_08_com_microsoft_domain

Completed
---------

.. list-table::
    :header-rows: 1
    :widths: 35 65
    :class: sphinx-datatable

    * - Plan
      - Contribution
    * - :doc:`Exp and Log parity <2026_08_exp_log_parity>`
      - Brings the transcendental kernels to their published ONNX Runtime
        parity and numerical gates.
    * - :doc:`Gemm and MatMul <2026_08_gemm_matmul>`
      - Delivers the shared matrix engine, typed and compact paths, scheduling,
        packing, AVX-512 tuning, and the corrective work through #605 and #608.
    * - :doc:`Attention <2026_08_attention>`
      - Delivers materialized and bounded-memory streaming Attention, including
        the AVX-512 optimization pass and the AVX2 decode and scheduling
        foundations through #599, #605, and #608.
    * - :doc:`Binary elementwise performance
        <2026_08_binary_elementwise_performance>`
      - Delivers prepared traversal, typed bulk execution, AVX-512 arithmetic,
        low-precision comparison, integer validation, and dispatch
        optimizations through #599.
    * - :doc:`Binary elementwise <2026_08_binary_elementwise>`
      - Supplies the shared prepared broadcast engine, generated correctness
        corpus, and parity gate for all 19 registered binary operators.
    * - :doc:`TreeEnsemble <2026_08_tree_ensemble>`
      - Provides the typed in-place v5 engine, compact prepared storage, and
        final correctness, parity, and memory gates, with the large-batch
        row-parallel follow-up tracked in #580.
    * - :doc:`Runtime execution controls
        <2026_08_runtime_execution_controls>`
      - Supplies the typed session policy and shared executor used by every
        registered kernel.
    * - :doc:`Processor performance profile
        <2026_08_processor_performance_profile>`
      - Measures cache and RAM bandwidth and latency plus sustained arithmetic
        throughput through one explicit Python API.
    * - :doc:`Registered kernel documentation
        <2026_08_registered_kernel_documentation>`
      - Derives the Python inventory and generated reference directly from the
        actual C++ registrations.
    * - :doc:`Recent kernel performance improvements
        <2026_08_kernel_performance_improvements>`
      - Consolidates the August sprint and the August 30--September 2 follow-up
        across unary, binary, matrix, Attention, normalization,
        ``com.microsoft``, and TreeEnsemble kernels.
    * - :doc:`com.microsoft domain support
        <2026_08_com_microsoft_domain>`
      - Introduces ``CDist`` and ``BiasGelu`` end to end, including schemas,
        runtime kernels, gradients, fusion patterns, tests, documentation, and
        the latency follow-ups in #562 and #564.
