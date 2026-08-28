.. _l-next-steps:

Next Steps
==========

Performance roadmaps for the kernels that ``onnx-light-cpu`` plans to
optimize next.

.. toctree::
    :maxdepth: 1
    :hidden:

    2026/2026_08_gemm_matmul
    2026/2026_08_attention
    2026/2026_08_exp_log_parity
    2026/2026_08_qwen3_inference
    2026/2026_08_conv
    2026/2026_08_unary_elementwise
    2026/2026_08_binary_elementwise
    2026/2026_08_binary_elementwise_performance
    2026/2026_08_elementwise_kernel_tuning
    2026/2026_08_tree_ensemble
    2026/2026_08_svm
    2026/2026_08_runtime_execution_controls
    2026/2026_08_processor_performance_profile
    2026/2026_08_kv_cache
    2026/2026_08_registered_kernel_documentation
    2026/2026_08_kernel_performance_improvements
    2026/2026_08_com_microsoft_domain

Started
-------

.. list-table::
    :header-rows: 1
    :widths: 35 65
    :class: sphinx-datatable

    * - Plan
      - Contribution
    * - :doc:`Gemm and MatMul <2026/2026_08_gemm_matmul>`
      - Closes the remaining transformer down-projection GEMM gap, then reruns
        stable-affinity ``MatMulInteger`` AMX/VNNI parity and the complete
        chained-GEMM gate.
    * - :doc:`Qwen3 inference <2026/2026_08_qwen3_inference>`
      - Establishes a frozen end-to-end model benchmark, then builds the
        canonical batch-1 QDQ INT4, shared GQA, and persistent-decode path.
    * - :doc:`Binary elementwise performance
        <2026/2026_08_binary_elementwise_performance>`
      - Completes low-precision and predicate bulk kernels, specializes
        priority broadcasts, calibrates scheduling, and closes the
        large-tensor parity gap.

Completed
---------

2026
^^^^

.. list-table::
    :header-rows: 1
    :widths: 35 65
    :class: sphinx-datatable

    * - Plan
      - Contribution
    * - :doc:`Attention <2026/2026_08_attention>`
      - Delivers materialized and bounded-memory MHA, GQA/MQA, and rank-3
        execution with a reproducible ONNX Runtime parity gate.
    * - :doc:`Exp and Log parity <2026/2026_08_exp_log_parity>`
      - Brings the transcendental kernels to their published ONNX Runtime
        parity and numerical gates.
    * - :doc:`Binary elementwise <2026/2026_08_binary_elementwise>`
      - Supplies the shared prepared broadcast engine, generated correctness
        corpus, and parity gate for all 19 registered binary operators.
    * - :doc:`TreeEnsemble <2026/2026_08_tree_ensemble>`
      - Provides the typed in-place v5 engine, compact prepared storage, and
        final correctness, parity, and memory gates.
    * - :doc:`Runtime execution controls
        <2026/2026_08_runtime_execution_controls>`
      - Supplies the typed session policy and shared executor used by every
        registered kernel.
    * - :doc:`Processor performance profile
        <2026/2026_08_processor_performance_profile>`
      - Measures cache and RAM bandwidth and latency plus sustained arithmetic
        throughput through one explicit Python API.
    * - :doc:`Registered kernel documentation
        <2026/2026_08_registered_kernel_documentation>`
      - Derives the Python inventory and generated reference directly from the
        actual C++ registrations.
    * - :doc:`Recent kernel performance improvements
        <2026/2026_08_kernel_performance_improvements>`
      - Consolidates the August optimization sprint across unary, binary,
        matrix, Attention, normalization, and TreeEnsemble kernels.
    * - :doc:`com.microsoft domain support
        <2026/2026_08_com_microsoft_domain>`
      - Introduces ``CDist`` and ``BiasGelu`` end to end, including schemas,
        runtime kernels, gradients, fusion patterns, tests, and documentation.

Discussion
----------

.. list-table::
    :header-rows: 1
    :widths: 35 65
    :class: sphinx-datatable

    * - Plan
      - Contribution
    * - :doc:`Conv <2026/2026_08_conv>`
      - Defines prepared convolution plans and specialized algorithms that
        avoid a universal materialized ``im2col`` path.
    * - :doc:`Unary elementwise <2026/2026_08_unary_elementwise>`
      - Unifies scalar, SIMD, traversal, and scheduling implementations across
        unary operators.
    * - :doc:`Processor-aware tuning
        <2026/2026_08_elementwise_kernel_tuning>`
      - Defines calibration and persistence of processor-specific elementwise
        thresholds.
    * - :doc:`SVM <2026/2026_08_svm>`
      - Defines prepared SVM classification and regression kernels.
    * - :doc:`Persistent KV cache <2026/2026_08_kv_cache>`
      - Defines mutable, paged, and optionally quantized cache storage that
        avoids copying complete past K/V tensors during decode.
