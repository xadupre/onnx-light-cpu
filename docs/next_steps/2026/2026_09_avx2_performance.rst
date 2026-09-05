AVX2 performance baseline
=========================

Issue :issue:`631` establishes the measurement baseline for the AVX2 performance
roadmap. It does not change kernel implementations.

Reproducible run
----------------

Update the target branch before collecting results, then build the Python package
in Release mode with the AVX2 ceiling:

.. code-block:: bash

    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2" \
      python setup.py build_ext --inplace --onnx-light-source
    PYTHONPATH=. python -c \
      "from onnx_light_cpu import detect_simd_level; print(detect_simd_level().name)"
    python tools/benchmark_avx2_parity.py \
      --environment pinned --output avx2-parity.json

The detected level must be ``AVX2``. The fixed corpus covers GEMM/MatMul,
Attention, activation and normalization, unary, and binary elementwise cases in
float32 and float16. It runs both one-thread and process-visible physical-core policies with
identical onnx-light-cpu and ONNX Runtime thread counts. Each runtime receives a
separate timing phase, and the first runtime alternates between consecutive
cases.

Results
-------

The JSON records every raw sample, medians and dispersion, shapes, data types,
loop families, CPU and affinity, SIMD ceiling and detected level, compiler,
package versions, and timing order. Results are ranked by positive absolute
latency gap and speedup. Qwen decode and prefill rows are labelled explicitly.
The companion Markdown groups rows into ``<0.5x``, ``0.5x-0.9x``,
``0.9x-1.0x``, and ``>=1.0x`` ONNX Runtime.

Run the ``AVX2 parity baseline`` workflow to publish the JSON, Markdown, and
environment capture as one artifact. Generated results are not committed.
Results from shared runners are diagnostic, especially within 5--10% of parity;
only ``--environment pinned`` results collected on pinned native AVX2 hardware
may make a final parity decision. Follow-up issues should be opened only for the
ranked measured bottlenecks listed by the report.
