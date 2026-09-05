.. _l-next-steps:

Next Steps
==========

Performance roadmaps for the kernels that ``onnx-light-cpu`` plans to
optimize next.

.. toctree::
    :maxdepth: 1
    :hidden:

    2026/2026_08_gemm_matmul
    2026/2026_09_avx2_gemm_matmul
    2026/2026_08_attention
    2026/2026_08_exp_log_parity
    2026/2026_08_qwen3_inference
    2026/2026_09_qwen3_operator_slice
    2026/2026_09_qwen3_missing_kernels
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
    2026/2026_09_avx2_performance

Started
-------

.. list-table::
    :header-rows: 1
    :widths: 35 65
    :class: sphinx-datatable

    * - Plan
      - Contribution
    * - :doc:`AVX2 performance <2026/2026_09_avx2_performance>`
      - Uses the explicit AVX2 SIMD ceiling to measure and rank the remaining
        gaps below the completed AVX-512 paths before optimizing matrix,
        Attention, activation, normalization, unary, and binary workloads.
    * - :doc:`AVX2 Gemm and MatMul gap closure <2026/2026_09_avx2_gemm_matmul>`
      - Measures the AVX2-ceiling FP32/FP64 Gemm/MatMul corpus from #633
        (AVX2 PR02a); dedicated AVX2-only hardware and an onnx-light checkout
        remain required for a genuine AVX2-vs-AVX2 ONNX Runtime parity gate.

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
        canonical batch-1 INT4 and persistent-decode path on top of the
        delivered GQA, RMSNormalization, and Sigmoid primitives.
    * - :doc:`Qwen3 missing kernels <2026/2026_09_qwen3_missing_kernels>`
      - Splits the remaining INT4 projections, input/layout kernels, and
        normalization adapters into independent PRs, with runtime metadata
        and persistent-cache ownership kept in onnx-light.
    * - :doc:`Qwen3 non-MatMulNBits operators
        <2026/2026_09_qwen3_operator_slice>`
      - Implements the metadata, input, layout, activation, and normalization
        slice required by the audited Qwen3 graph independently of packed
        INT4 projections; Sigmoid and the shared RMS engine are already
        available.

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
    * - :doc:`Gemm and MatMul <2026/2026_08_gemm_matmul>`
      - Delivers the shared matrix engine, typed and compact paths, scheduling,
        packing, AVX-512 tuning, and the corrective work through #605 and #608.
    * - :doc:`Attention <2026/2026_08_attention>`
      - Delivers materialized and bounded-memory streaming Attention, including
        the AVX-512 optimization pass and the AVX2 decode and scheduling
        foundations through #599, #605, and #608.
    * - :doc:`Binary elementwise performance
        <2026/2026_08_binary_elementwise_performance>`
      - Delivers prepared traversal, typed bulk execution, AVX-512 arithmetic,
        low-precision comparison, integer validation, and dispatch
        optimizations through #599.
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
