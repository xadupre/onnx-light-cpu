RotaryEmbedding
===============

Both ``ai.onnx::RotaryEmbedding`` and ``com.microsoft::RotaryEmbedding`` use
the same validated adapter. Split-half FLOAT heads use eight-pair AVX
arithmetic; FLOAT16 heads use AVX2/F16C conversion followed by FLOAT
arithmetic and round-to-nearest-even narrowing. F16C is checked independently
of AVX2. Dispatch is selected once per invocation with cached CPU detection,
and ISA-specific sources retain their per-source compiler flags.
The selector takes an explicit storage ``DataType`` and returns no SIMD
implementation for unsupported types. Heads with fewer than eight pairs
bypass SIMD dispatch and use the scalar loop directly.

The existing scalar loop handles vector tails, interleaved rotation, BFLOAT16,
and CPUs/builds without the required instructions. Multiplication and
addition/subtraction remain separate, without introducing FMA. Full-head
rotation writes every output element directly; partial rotation retains the
input copy to preserve the unrotated suffix. Position validation, cache
indexing, rank-three/rank-four layouts, and runtime-owned row scheduling are
unchanged. No temporary tensor or worker allocation is introduced.

AVX2 investigation, 2026-10-23
------------------------------

The four cases from issue #977 reproduced a repeatable gap on an AMD EPYC
9V74 80-Core Processor VM (four visible logical CPUs, two physical cores).
This is **not** the dashboard's EPYC 7763. The CPU library was compiled with
an AVX2 ceiling and reported SIMD level 3; ORT retained its own CPU dispatch.

Build provenance:

* onnx-light-cpu 0.2.0, original adapter at
  ``f73c8e63e2bf5d11a33f4af6af612ff7fc98cfcc``, optimized at ``56f7be9``;
  GCC 13.3.0, CMake Release, Python 3.13.16.
* onnx-light 0.2.0 published CPython 3.13 Linux x86-64 wheel and its included
  headers; release tag points to
  ``6400f0f812b13bc32469e405bd2f762ba3cba7f0``.
* ONNX Runtime 1.31.0 Linux wheel, Release build reporting commit ``402979a``;
  NumPy 2.5.4. The CPU bindings used nanobind 3.1.0 and
  ``ONNX_LIGHT_CPU_PYTHON_STABLE_ABI=OFF`` to match the onnx-light wheel.

The controlled comparison pinned the process to logical CPU 0 and used one
thread in each backend, sequential ORT execution, disabled ORT intra/inter-op
spinning, and CPU ``spin_policy="park_immediately"``. Each trial used 100
warmups and 1,000 measured calls per backend, alternating backend order on
successive iterations, with a nonbinding ten-second measurement cap.
Reported values are medians of three trial medians. No compilation or tests
ran concurrently. The optimized adapter was measured first, then the
byte-identical original adapter was rebuilt in the same tree and measured
with the same protocol. After restoring and rebuilding the optimized adapter,
executing the reproduction snippet below confirmed 13.85--14.69 microseconds
for FLOAT and 12.05--12.13 microseconds for FLOAT16.

Inputs were the unchanged registered
``test_cpu_[microsoft_]rotary_embedding_b1_s128_h8_d64_{dtype}_benchmark``
cases: X ``[1,128,512]``, caches ``[128,32]``, and INT64 positions
``[1,128]``. CPU sessions received backend tensors directly; ORT received
the same data as NumPy arrays. Session creation and input adaptation were
outside the timed calls; execution and output handling remained inside.

.. list-table:: Controlled latency in microseconds
   :header-rows: 1

   * - Domain / dtype
     - CPU before
     - CPU after
     - ORT before
     - ORT after
     - Before / after
   * - com.microsoft / float32
     - 97.133
     - 14.211
     - 22.423
     - 22.313
     - 6.84x
   * - com.microsoft / float16
     - 615.208
     - 12.348
     - 23.079
     - 20.400
     - 49.82x
   * - ai.onnx / float32
     - 97.029
     - 14.432
     - 22.688
     - 22.533
     - 6.72x
   * - ai.onnx / float16
     - 614.788
     - 12.458
     - 23.039
     - 20.520
     - 49.35x

The optimized CPU/ORT latency ratios are 0.61--0.64, below one for all four
cases. Separate unpinned runs with the standard backend benchmark API
confirmed improvements with both one and two threads; two-thread ORT
timings were noisier. These shared-VM measurements are diagnostic, not a
claim of identical performance on the original processor.

Hardware profiling was unavailable (``perf_event_paranoid=4``). Source and
disassembly inspection identified the scalar pair loop and branch-heavy
software half conversions; the optimized object contains ``vcvtph2ps``,
``vcvtps2ph``, and vector multiply/add/subtract, without FMA.
No sampled profile or per-phase percentage is claimed.

Correctness checks cover every vector remainder, unaligned buffers, signed
zero, subnormals, infinities, NaNs, partial rotation, both layouts, and
BFLOAT16/interleaved fallback. FLOAT comparisons allow rounding-level
differences while checking signed zeros and infinities explicitly. Half-type
outputs and unrotated suffixes retain bit-exact checks.
The four original benchmark outputs also
matched ORT: FLOAT16 was bit-identical, and FLOAT maximum absolute error
was below 1e-6.

Reproduction
------------

Build against the exact onnx-light runtime selected through ``PYTHONPATH``:

.. code-block:: bash

   CMAKE_ARGS="-DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2 -DONNX_LIGHT_CPU_PYTHON_STABLE_ABI=OFF" \
     python setup.py build_ext --inplace --onnx-light-source --parallel "$(nproc)"

The following uses existing benchmark helpers and the registered inputs.
Run it at the original and optimized revisions under the same environment:

.. code-block:: bash

   taskset -c 0 python - <<'PY'
   import statistics
   import onnxruntime as ort
   from onnx_light.onnx.backend import TestMode, collect_test_cases_by_name
   from onnx_light.onnx.reference import ReferenceEvaluator
   from onnx_light_cpu import register_backend_test_cases, register_kernels
   from onnx_light_cpu._benchmark import _prepare_feeds
   from tools.benchmark_attention_parity import measure_alternating

   register_backend_test_cases()
   register_kernels()
   pattern = r"^test_cpu_(microsoft_)?rotary_embedding_b1_s128_h8_d64_float(16|32)_benchmark$"
   for case in collect_test_cases_by_name(
       pattern, mode=TestMode.BENCHMARK, generate_benchmark_expected_outputs=False
   ):
       model = case.model
       cpu_feeds, ort_feeds = _prepare_feeds(
           [value.name for value in model.graph.input], case.data_sets
       )
       cpu = ReferenceEvaluator(model, cpu_execution={
           "num_threads": 1, "affinity_policy": "none", "spin_policy": "park_immediately",
       })
       options = ort.SessionOptions()
       options.intra_op_num_threads = options.inter_op_num_threads = 1
       options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
       options.add_session_config_entry("session.intra_op.allow_spinning", "0")
       options.add_session_config_entry("session.inter_op.allow_spinning", "0")
       session = ort.InferenceSession(
           model.SerializeToString(), sess_options=options,
           providers=["CPUExecutionProvider"],
       )
       for trial in range(3):
           samples, _ = measure_alternating(
               (lambda: cpu.run(None, cpu_feeds[0]),
                lambda: session.run(None, ort_feeds[0])),
               repeat=1000, warmup=100, max_repeat_time=10.0,
           )
           print(case.name, trial, [statistics.median(s) * 1e6 for s in samples])
       case.unload()
   PY
