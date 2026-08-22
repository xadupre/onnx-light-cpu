.. _l-next-steps:

Next Steps
==========

Performance roadmaps for the kernels that ``onnx-light-cpu`` plans to
optimize next.

Use the search field to filter by status or text, and select a column heading
to sort the table.

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
    2026/2026_08_elementwise_kernel_tuning
    2026/2026_08_tree_ensemble
    2026/2026_08_svm
    2026/2026_08_runtime_execution_controls
    2026/2026_08_kv_cache
    2026/2026_08_registered_kernel_documentation

.. list-table::
    :header-rows: 1
    :widths: 12 24 32 32
    :class: sphinx-datatable

    * - Status
      - Next step
      - Planned work
      - Why
    * - Completed
      - :doc:`Gemm and MatMul <2026/2026_08_gemm_matmul>`
      - Delivers the shared matrix engine, low-precision paths, tuning, and
        reproducible parity tooling.
      - Matrix multiplication is the foundation for attention and dense models.
    * - Discussed
      - :doc:`Attention <2026/2026_08_attention>`
      - Add materialized and streaming MHA, GQA, and MQA CPU kernels.
      - Transformer inference needs bounded-memory attention without avoidable
        score and head materialization.
    * - Completed
      - :doc:`Exp and Log parity <2026/2026_08_exp_log_parity>`
      - Reach the published ONNX Runtime parity and numerical gates.
      - Transcendental kernels are common and expose SIMD and scheduling gaps.
    * - Discussed
      - :doc:`Qwen3 inference <2026/2026_08_qwen3_inference>`
      - Build the batch-1 INT4, GQA, and decode critical path.
      - A frozen dense model provides an end-to-end target for kernel priorities.
    * - Discussed
      - :doc:`Conv <2026/2026_08_conv>`
      - Add prepared convolution plans and specialized algorithms.
      - A universal materialized ``im2col`` path wastes memory and bandwidth.
    * - Discussed
      - :doc:`Unary elementwise <2026/2026_08_unary_elementwise>`
      - Unify scalar, SIMD, traversal, and scheduling for unary operators.
      - Individual kernels duplicate dispatch logic and leave broad gaps.
    * - Discussed
      - :doc:`Binary elementwise <2026/2026_08_binary_elementwise>`
      - Add a prepared SIMD engine for binary operators and broadcasting.
      - Generic rank-aware offset computation is too expensive per element.
    * - Discussed
      - :doc:`Processor-aware tuning <2026/2026_08_elementwise_kernel_tuning>`
      - Calibrate and persist processor-specific elementwise thresholds.
      - Fixed worker limits and crossovers do not transfer across processors.
    * - Completed
      - :doc:`TreeEnsemble <2026/2026_08_tree_ensemble>`
      - Deliver the prepared v5 engine, tuning, and final parity gate.
      - ONNX-ML forests require predictable low-latency traversal and batching.
    * - Discussed
      - :doc:`SVM <2026/2026_08_svm>`
      - Add prepared SVM classification and regression kernels.
      - ``onnx-light-cpu`` does not yet register optimized SVM operators.
    * - Completed
      - :doc:`Runtime execution controls <2026/2026_08_runtime_execution_controls>`
      - Use the typed session policy and executor for every registered kernel.
      - A second CPU scheduler would invalidate runtime limits and diagnostics.
    * - Planned
      - :doc:`Registered kernel documentation <2026/2026_08_registered_kernel_documentation>`
      - Derive the Python inventory and generated reference from actual C++
        registrations.
      - Separate hard-coded lists and source scans become stale when kernels
        change.
    * - Discussed
      - :doc:`Persistent KV cache <2026/2026_08_kv_cache>`
      - Add mutable, paged, and optionally quantized cache storage.
      - Copying complete past K/V tensors makes decode cost grow with context.
