.. _l-next-steps:

Next Steps
==========

Performance roadmaps for the kernels that ``onnx-light-cpu`` plans to
optimize next.
Completed plans are kept in the :doc:`2026 archive <2026/index>`.

.. toctree::
    :maxdepth: 1
    :hidden:

    2026/2026_09_avx2_gemm_matmul
    2026/2026_08_qwen3_inference
    2026/2026_09_qwen3_operator_slice
    2026/2026_09_avx2_activation_normalization
    2026/2026_09_qwen3_missing_kernels
    2026/2026_08_conv
    2026/2026_08_unary_elementwise
    2026/2026_08_elementwise_kernel_tuning
    2026/2026_08_svm
    2026/2026_08_kv_cache
    2026/2026_09_avx2_performance
    2026/2026_09_avx2_diagnostic_baseline
    2026/2026_09_avx2_matrix_kernel_improvements
    2026/index

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
        The :doc:`isolated-runtime diagnostic baseline
        <2026/2026_09_avx2_diagnostic_baseline>` identifies matrix and Attention
        priorities and a thread-pool interference bias in the existing runner.
        The :doc:`matrix kernel follow-up
        <2026/2026_09_avx2_matrix_kernel_improvements>` delivers measured
        FP16 packing and FP16/FP32 single-row improvements; full ORT parity
        remains open.
    * - :doc:`AVX2 activation and normalization gap closure
        <2026/2026_09_avx2_activation_normalization>`
      - Extends the #604 AVX2 activation work (AVX2 PR04a): measured
        ``Sigmoid``, ``Softmax``, ``BiasGelu``, and ``RMSNormalization``
        against ONNX Runtime and closed the one found gap, a serialized
        Float16 RMSNormalization reduction; the full onnx-light
        backend-corpus acceptance run remains.
    * - :doc:`AVX2 Gemm and MatMul gap closure <2026/2026_09_avx2_gemm_matmul>`
      - Measures the AVX2-ceiling FP32/FP64 Gemm/MatMul corpus from #633
        (AVX2 PR02a); dedicated AVX2-only hardware and an onnx-light checkout
        remain required for a genuine AVX2-vs-AVX2 ONNX Runtime parity gate.
    * - :doc:`Qwen3 missing kernels <2026/2026_09_qwen3_missing_kernels>`
      - Tracks optimized INT4 projections and integration work after the
        portable MatMulNBits foundation. Gather and
        SimplifiedLayerNormalization are complete; Cast, Split, and
        SkipSimplifiedLayerNormalization have delivered their principal
        kernels, with runtime metadata and persistent-cache ownership kept in
        onnx-light.

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
    * - :doc:`Qwen3 non-MatMulNBits operators
        <2026/2026_09_qwen3_operator_slice>`
      - Implements the metadata, input, layout, activation, and normalization
        slice required by the audited Qwen3 graph independently of packed
        INT4 projections; Sigmoid and the shared RMS engine are already
        available.

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
