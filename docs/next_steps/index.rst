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
    2026/2026_09_qwen3_operator_slice
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
      - Incorporates the medium-GEMM, MatMul, bias, and dynamic-packing work
        delivered through #560, #561, #566, #567, and #575, then reruns the
        stable-affinity parity gate.
    * - :doc:`Attention <2026/2026_08_attention>`
      - Tracks the single-key, short-query, tiled FP32, and Qwen3.6 FP16 work
        delivered through #559, #569, and #578; the dedicated-machine ONNX
        Runtime parity gate remains.
    * - :doc:`Binary elementwise performance
        <2026/2026_08_binary_elementwise_performance>`
      - Includes the AVX-512 arithmetic and integer ``Div`` validation work
        from #563 and #577; dedicated-machine acceptance remains.

Planned
-------

.. list-table::
    :header-rows: 1
    :widths: 35 65
    :class: sphinx-datatable

    * - Plan
      - Contribution
    * - :doc:`Qwen3 inference <2026/2026_08_qwen3_inference>`
      - Freezes the audited end-to-end model benchmark, then builds the
        canonical batch-1 INT4, shared GQA, and persistent-decode path.
    * - :doc:`Qwen3 non-MatMulNBits operators
        <2026/2026_09_qwen3_operator_slice>`
      - Implements the metadata, input, layout, activation, and normalization
        slice required by the audited Qwen3 graph independently of packed
        INT4 projections.

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
    * - :doc:`Exp and Log parity <2026/2026_08_exp_log_parity>`
      - Brings the transcendental kernels to their published ONNX Runtime
        parity and numerical gates.
    * - :doc:`Binary elementwise <2026/2026_08_binary_elementwise>`
      - Supplies the shared prepared broadcast engine, generated correctness
        corpus, and parity gate for all 19 registered binary operators.
    * - :doc:`TreeEnsemble <2026/2026_08_tree_ensemble>`
      - Provides the typed in-place v5 engine, compact prepared storage, and
        final correctness, parity, and memory gates, with the large-batch
        row-parallel follow-up tracked in #580.
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
      - Consolidates the August sprint and the August 30--September 2 follow-up
        across unary, binary, matrix, Attention, normalization,
        ``com.microsoft``, and TreeEnsemble kernels.
    * - :doc:`com.microsoft domain support
        <2026/2026_08_com_microsoft_domain>`
      - Introduces ``CDist`` and ``BiasGelu`` end to end, including schemas,
        runtime kernels, gradients, fusion patterns, tests, documentation, and
        the latency follow-ups in #562 and #564.

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
