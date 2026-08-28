Recent Kernel Performance Improvements
======================================

:Date: 2026-08

**complete** (August 25--27 optimization sprint)

Objective
---------

The objective was to improve the kernels exercised by the expanded backend
benchmark corpus before adding more operators. The work targeted measured hot
paths across unary and binary elementwise operations, matrix multiplication,
Attention, normalization, and TreeEnsemble while preserving the existing ONNX
semantics and runtime registration contract.

This completed step records the cross-cutting optimization sprint. It does not
replace the dedicated roadmaps or declare every final ONNX Runtime parity gate
complete; remaining measured gaps continue in their operator-specific plans.

Delivered work
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 48 28

   * - Area
     - Contribution
     - Pull requests
   * - Unary kernels
     - Reworked ``Abs``, ``Exp``, and ``Log`` execution and tuning, added
       vectorized paths and tails, and refined thresholds from backend
       measurements.
     - `#442 <https://github.com/xadupre/onnx-light-cpu/pull/442>`_,
       `#459 <https://github.com/xadupre/onnx-light-cpu/pull/459>`_,
       `#460 <https://github.com/xadupre/onnx-light-cpu/pull/460>`_
   * - Binary kernels
     - Specialized broadcast plans and arithmetic dispatch, expanded bulk
       execution and benchmark coverage, and optimized FP16/BF16 ``Pow`` for
       contiguous, scalar-broadcast, and mixed-exponent signatures.
     - `#443 <https://github.com/xadupre/onnx-light-cpu/pull/443>`_--
       `#446 <https://github.com/xadupre/onnx-light-cpu/pull/446>`_,
       `#475 <https://github.com/xadupre/onnx-light-cpu/pull/475>`_
   * - GEMM and MatMul
     - Improved blocking, packing, scheduling, and benchmark coverage; added
       specialized square and large-K float64 paths plus native half and
       skinny integer MatMul execution.
     - `#447 <https://github.com/xadupre/onnx-light-cpu/pull/447>`_,
       `#467 <https://github.com/xadupre/onnx-light-cpu/pull/467>`_,
       `#468 <https://github.com/xadupre/onnx-light-cpu/pull/468>`_,
       `#473 <https://github.com/xadupre/onnx-light-cpu/pull/473>`_
   * - Higher-level kernels
     - Applied shared execution improvements across registered kernels,
       streamlined TreeEnsemble traversal, fused FP16 RMS normalization, and
       widened FP16 Attention into the optimized streaming and tiled paths.
     - `#451 <https://github.com/xadupre/onnx-light-cpu/pull/451>`_,
       `#463 <https://github.com/xadupre/onnx-light-cpu/pull/463>`_,
       `#466 <https://github.com/xadupre/onnx-light-cpu/pull/466>`_,
       `#472 <https://github.com/xadupre/onnx-light-cpu/pull/472>`_

Validation and completion
-------------------------

Each optimization retained a portable fallback and added focused coverage for
the affected data types, vector tails, broadcast or matrix shapes, and runtime
dispatch. The backend benchmarks supplied reproducible filters, warmups,
repeats, raw timings, and ONNX Runtime comparisons for the changed paths.

The step is complete because all listed implementations and their correctness
coverage are merged. Dedicated parity roadmaps remain authoritative where a
kernel family still has an explicit minimum-speedup or cross-machine gate.
