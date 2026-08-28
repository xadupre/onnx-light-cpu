Kernels
=======

The registered CPU kernels share a common runtime contract: onnx-light creates
one ``KernelBase`` adapter per graph node, the adapter validates concrete
tensors and delegates to an immutable or per-invocation execution plan, and
parallel work is submitted only through the current session's ``CpuExecutor``.
No kernel family owns a private worker pool.

The pages below describe five implemented kernel families. Other registered
families, including variadic elementwise, integer MatMul, normalization,
SwiGLU, TreeEnsemble, and custom-domain kernels, are not yet covered here.
Performance targets and future work remain in :doc:`../../next_steps/index`.

.. toctree::
    :maxdepth: 1

    unary_kernel_design
    binary_kernel_design
    gemm_kernel_design
    matmul_kernel_design
    attention_kernel_design
