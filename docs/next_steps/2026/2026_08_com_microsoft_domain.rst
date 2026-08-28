com.microsoft Domain Support
============================

:Date: 2026-08

**complete** (`#479 <https://github.com/xadupre/onnx-light-cpu/pull/479>`_)

Objective
---------

The objective was to establish the first non-standard ONNX domain supported
end to end by ``onnx-light-cpu``. The implementation introduces
``com.microsoft::CDist`` and ``com.microsoft::BiasGelu`` through the same
schema, graph-building, runtime, differentiation, optimization, testing, and
documentation surfaces used by standard operators.

Operator contract
-----------------

``CDist``
    Computes pairwise distances between rows of two rank-two tensors. Version
    1 supports ``FLOAT`` and ``DOUBLE`` with ``sqeuclidean`` and ``euclidean``
    metrics and produces an ``[M, K]`` result from ``[M, N]`` and ``[K, N]``
    inputs.

``BiasGelu``
    Adds a rank-one last-dimension bias and applies exact GELU. Version 1
    supports ``FLOAT16``, ``FLOAT``, ``DOUBLE``, and ``BFLOAT16`` while
    preserving the input shape and type.

Delivered integration
---------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Surface
     - Contribution
   * - Schemas
     - Adds lightweight versioned schemas and a combined schema lookup so
       graph builders can resolve standard ONNX and ``com.microsoft``
       operators through one callback.
   * - Shape and memory
     - Registers symbolic shape inference and zero-scratch peak-memory
       functions for both operators.
   * - CPU runtime
     - Adds portable typed implementations, registered ``KernelBase`` adapters,
       tuning schemas, usage names, and structured inventory metadata.
   * - Fusion patterns
     - Fuses exact ``Add`` plus ``Gelu`` into ``BiasGelu`` and the
       squared-distance ``Unsqueeze``/``Sub``/``Mul``/``ReduceSum`` expansion
       into ``CDist(metric="sqeuclidean")``.
   * - Gradients
     - Registers standard-ONNX backward graphs for both operators. The C++
       documentation added in
       `#484 <https://github.com/xadupre/onnx-light-cpu/pull/484>`_ describes
       the complete gradient data flow.
   * - Python API
     - Exposes custom schemas, combined schema lookup, operator-support
       inventory, support registration, and gradient registration alongside
       the existing kernel registration entry point.
   * - Tests and examples
     - Adds low-level, runtime, registration, usage, gradient, fusion, backend
       TEST/BENCHMARK, gallery, and ONNX Runtime comparison coverage.

Validation and completion
-------------------------

The completion gate requires both operators to build through their custom
schemas, infer shapes, report peak memory, execute through the registered
onnx-light runtime, differentiate into standard ONNX nodes, and participate in
their guarded fusion patterns. Their registrations must also appear in the
public kernel inventory and backend corpus.

All layers and their focused C++ and Python tests were delivered in #479.
This establishes the domain-oriented directory and registration pattern to
follow when another custom domain or operator is added.
