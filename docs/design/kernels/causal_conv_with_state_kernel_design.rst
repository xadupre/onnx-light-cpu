CausalConvWithState
===================

The ``com.microsoft::CausalConvWithState`` adapter validates geometry, types and
buffer capacities before dispatching. The general implementation handles FLOAT,
FLOAT16 and BFLOAT16, both layouts, optional bias and past state, dilation and
SiLU/Swish activation.

FLOAT16 three-tap path
----------------------

For a three-tap kernel, unit dilation, no activation and sequence length at least
two, the adapter uses a bounded, allocation-free tiled routine in
``ml_cpu_compute``. Channels-first tiles contain 64 time positions plus two
history positions. Channels-last tiles contain up to 16 channels and rotate three
history buffers. Channel tiles never cross batch boundaries, including partial
tiles.

Weights are converted once per channel tile, and each input value is converted
once rather than once per tap. The existing bulk half converters select F16C
where available and retain portable scalar fallbacks. The float32 arithmetic
retains bias followed by the three products in tap order; there is no
half-precision accumulation or approximate activation. Present state is copied
from the original input bits, preserving signed zero and NaN payloads.

Independent channel rows/tiles use ``ExecuteRanges`` and the caller's executor.
The work estimate counts converted output elements rather than repeating the
general kernel's per-tap conversion cost, avoiding worker wake-ups for short
prefills.
Scratch is fixed-size stack storage, not a full-tensor conversion, persistent
cache, worker allocation or private thread pool. Unsupported configurations
continue to use the general implementation.

Validation and measurement
--------------------------

The focused C++ suite is ``test_onnx_light_causal_conv_with_state_kernel``.
It compares against the independent naive kernel across types, layouts,
dilations, activations and state boundaries. Additional FLOAT16 cases cover tile
tails, the benchmark lengths, multiple batches, special values, exact state bits
and serial/parallel/nested scheduling.

The four issue-979 workloads are the existing backend cases matching
``^test_cpu_causal_conv_c64_l(128|513)_float16_(first|last)_benchmark$``.
Their inputs, weights, output obligations and expected-value generation are
unchanged. For a quick check with an AVX2-ceiling Release build, from the
repository root run:

.. code-block:: bash

   python -m onnx_light_cpu benchmark \
     --test '^test_cpu_causal_conv_c64_l(128|513)_float16_(first|last)_benchmark$' \
     --dtype float16 --onnxruntime --threads 1 \
     --warmup 100 --repeat 1000 --max-repeat-time 3 \
     --output /tmp/causal-conv.xlsx

Use the same thread limit for both runtimes and for before/after measurements.
The original dashboard does not provide per-case build provenance or complete
execution settings; local results are not a reconstruction of that campaign.

EPYC 7763 measurements, 2026-10-24
----------------------------------

Environment:

* AMD EPYC 7763 virtual runner: two visible physical cores, four logical CPUs,
  process CPU set 0-3; Linux 6.17.0-1022-azure.
* GCC 13.3.0, C++20 Release, Python 3.13.16, NumPy 2.5.4,
  ``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2``; runtime ``simd_level=3``.
* onnx-light-cpu 0.2.0: baseline ``f6bdfdfdb4882d1e59bf6c83632456bed0a4abfd``,
  optimized implementation ``885576b3bb14f54b4f7c5f1903a01b0172ef24c7``.
* onnx-light 0.2.1 source
  ``9707261506e7d25aa475eb6ec881d51338eb59c1``, nanobind 3.0.1,
  stable ABI enabled in both builds; selected through ``PYTHONPATH``.
* ONNX Runtime 1.31.0, Release wheel, reported build commit ``402979a``,
  ``CPUExecutionProvider``.

The tables use three passes per case, each with 100 warm-ups and 1,000 measured
calls per backend (three-second cap, not reached). Values are medians of the
three pass medians. The existing
``tools.benchmark_bias_gelu_parity.measure_alternating`` timing helper alternates
backend order on every iteration, disables Python GC during measurement and
times complete evaluator/session calls, including output materialization.
The CPU evaluator receives the backend tensors directly; ORT receives NumPy
views of the same data. Both outputs are compared before timing, with unchanged
FLOAT16 tolerances (``rtol=2e-2``, ``atol=1e-5``), and kernel recording confirms
``onnx_light_cpu::CausalConvWithState`` before recording is disabled for timing.

Unlike the quick CLI check above, these measurements explicitly disable
spinning in both runtimes: CPU ``spin_policy="park_immediately"`` and ORT session
entries ``session.intra_op.allow_spinning=0`` and
``session.inter_op.allow_spinning=0``. Neither runtime pins workers. Both use
the listed intra-op participant limit; ORT uses sequential execution and one
inter-op thread. All settings are identical before and after.

.. csv-table:: One participant; latency in milliseconds
   :header: "Case", "CPU before", "CPU after", "ORT before", "ORT after"

   "L128 first", 0.184744, 0.017032, 0.040596, 0.040997
   "L128 last", 0.186232, 0.019612, 0.051491, 0.051316
   "L513 first", 0.717508, 0.032836, 0.113061, 0.109103
   "L513 last", 0.734309, 0.043767, 0.160901, 0.154067

.. csv-table:: Four-participant limit; latency in milliseconds
   :header: "Case", "CPU before", "CPU after", "ORT before", "ORT after"

   "L128 first", 0.183933, 0.017258, 0.040847, 0.040495
   "L128 last", 0.185521, 0.019622, 0.051652, 0.051376
   "L513 first", 0.461956, 0.033022, 0.120870, 0.108693
   "L513 last", 0.480931, 0.044182, 0.165799, 0.154539

Every baseline pass reproduced an ORT advantage; every optimized pass reversed
it. Serial CPU latency improves 9.50-21.85x, ending 2.41-3.52x faster than ORT.
With four participants permitted, improvements are 9.45-13.99x. The optimized
cost estimate keeps these small workloads serial; the larger scheduling tests
still exercise the runtime executor. The four-participant baseline L513-first
pass medians varied from 0.457 to 0.490 ms, well outside near-parity with ORT.

Profiling limitations and validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``perf stat -e cycles`` was denied by the runner's
``perf_event_paranoid=4`` policy, so no native sampled-profile percentages are
claimed. A 1,000-call ``cProfile`` pass per serial case attributed over 99% of
time to the evaluator's native execution call, not Python feed preparation.
Source inspection identified the repeated software half conversions in the
scalar tap loop. Tiling removes those conversions without moving allocation,
packing or other work outside the timed invocation.

Baseline parity and special-value checks passed before enabling the fast path.
The final focused C++ suite passes all 12 tests, including exact finite FLOAT16
comparisons, both operator outputs, special values, fallback configurations and
the updated executor scheduling checks. No AVX-512 host was available and a
separate scalar-only build was not run; this result is specifically for the
documented AVX2 host.
