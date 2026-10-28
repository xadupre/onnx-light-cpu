Attention Kernel Design
=======================

The registered ``Attention`` kernel separates node configuration, concrete
shape planning, and compute. It supports ONNX Attention opsets 23 and 24 for
FLOAT, FLOAT16, and BFLOAT16.

Descriptor and invocation plan
------------------------------

``AttentionDescriptor`` records attributes and optional input/output wiring,
then validates opset rules, head counts, ``qk_matmul_output_mode``, cache
pairs, and ``nonpad_kv_seqlen`` availability. The registered adapter currently
rebuilds this descriptor from the node on every invocation. Because the
adapter does not receive the model's opset directly, it infers opset 24 when a
seventh input is present and opset 23 otherwise.

``AttentionPlan`` is lightweight and rebuilt for each invocation because
sequence lengths and strides may change:

.. code-block:: text

   NodeProto ----------------> AttentionDescriptor
                                      |
   Q/K/V/mask/cache shapes ----------+
                                      v
                               AttentionPlan
                               - layout and strides
                               - head mapping
                               - mask broadcasting
                               - total KV length
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                  materialized path          streaming path

Layouts and semantics
---------------------

Rank-four tensors use ``[B, H, L, D]``. Rank-three tensors use
``[B, L, H * D]`` and require explicit Q and KV head counts. MHA, GQA, and MQA
share one plan; ``group_size = q_num_heads / kv_num_heads`` maps Q heads to K/V
heads without physically repeating K or V.

The plan supports boolean and FLOAT additive broadcast masks, bottom-right
causal masking, ``softcap``, tensor ``past_key``/``past_value``, opset-24
``nonpad_kv_seqlen``, optional ``present`` outputs, and all four
``qk_matmul_output_mode`` values. A V head dimension may differ from the Q/K
head dimension.

The registered adapter publishes ``Y`` and the requested optional outputs.
``present_key`` and ``present_value`` concatenate past and current state in
the original storage type and always use rank-four layout.

Compute paths
-------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Path
     - Selection
     - Storage
   * - Materialized
     - Observable ``qk_matmul_output`` or explicit FP64 softmax; FLOAT also
       selects this path for ``present`` outputs.
     - Retains a full KV score row, and the complete QK output when requested.
       Half-storage inputs and outputs use full FP32 conversion buffers.
   * - Tiled
     - FLOAT/FLOAT16, no past state, at least 16 query rows and more than one
       key, when materialized dispatch is not required. AVX2/FMA also uses
       tiles for 8--15 rows with Q/K and V head dimensions in [16, 256]
       and finite V in the KV rows consumed by the query tile.
     - Online softmax over bounded query/KV tiles with FP32 accumulation.
   * - Streaming
     - Short queries, tensor-cache inputs, and BFLOAT16, when materialized
       dispatch is not required.
     - Visits KV blocks with online softmax and retains one score tile plus row
       accumulators instead of an ``Lq x Lkv`` matrix.
   * - Single key
     - Exactly one effective key, when materialized dispatch is not required.
     - Copies the value directly, or writes zero for a masked row.

The streaming recurrence maintains a running maximum, denominator, and
unnormalized output for each query row. Causal and padding frontiers skip
entire unavailable blocks; an all-false boolean-mask block is skipped as well.
Arbitrary additive masks remain fully evaluated.

Conversion and packing
----------------------

Tiled FP16 execution keeps Q/K in their original storage type. It converts V
into a worker-local KV tile immediately before the value product, retains
FP32 accumulation, and narrows the final query tile directly into Y.
Rank-three Q/K/V rows are packed only when their sequence stride requires it;
packing and output scatter are tile-local for both FP16 and FP32. There is no
immutable-input cache: every invocation reads the current source tensors.

FP16 streaming converts the current query row and the visited K/V blocks,
including blocks spanning past and current state, rather than converting
complete tensors. For short contexts, up to four score blocks are converted
once per worker/head and reused across query rows within that invocation.
Longer contexts retain one conversion block. BF16 streaming retains its
FP32-accumulating codec path.
Half-storage materialized execution deliberately retains its full conversion
buffers for observable QK output and FP64 softmax.

The benchmark's memory model bounds active attention scratch per worker. It
does not include vector allocator capacity retained from earlier calls or
internal GEMM packing panels, and should not be interpreted as measured RSS.

Scheduling, precision, and invariants
-------------------------------------

Outer batch/head/query-row ranges are submitted through the session executor.
Prefill exposes many independent rows, while short-query and decode shapes
avoid forced parallel overhead. FP16 and BF16 streaming paths accumulate in
FP32 and narrow only the final output. Explicit FP64 softmax uses the
materialized path rather than weakening its precision contract.

Q, K, and V types must match, and cache tensors must match those types.
Rank-three head counts must be positive, with the Q count divisible by the KV
count. ``nonpad_kv_seqlen`` must be INT64. Invalid optional-input pairings and
unsupported combinations fail explicitly. A fully masked query row produces
zeros rather than NaN.

Execution telemetry
-------------------

Enable ``set_kernel_usage_recording(session, True)`` and inspect
``used_kernel_paths(session)`` to distinguish ``Attention.tiled``,
``Attention.streaming``, ``Attention.materialized``, and
``Attention.single_key``. Additional ``Attention.conversion.*``,
``Attention.packing.*``, and ``Attention.output.*`` records describe the
selected storage decisions. The ordinary ``used_kernel_names(session)``
inventory still reports only ``onnx_light_cpu::Attention``.
``Attention.output.tile`` describes the final row/tile write granularity,
not the presence of an extra Y buffer: contiguous FP32 single-block products
can write directly into Y. The conversion and packing records describe
explicit attention scratch, not internal GEMM panels.
BF16 streaming reports ``Attention.conversion.element`` for its scalar codec
and FP32 query-row cache rather than claiming conversion-free execution.

``tools/benchmark_attention_parity.py`` retains these records alongside raw
alternating CPU/ONNX Runtime samples. Recording is disabled during timing.
ORT worker spinning is disabled so its idle workers do not compete with the
CPU implementation between alternating samples.
The corpus includes aligned q128/kv128 cases, q129/kv257 tile tails, head
dimensions 63/64/128/256, and short-query streaming cases for FP16 and FP32.

Native half-precision arithmetic
--------------------------------

QK products reuse the existing half-storage GEMM dispatch; softmax and value
accumulation stay in FP32. AVX2/F16C accelerates conversion but does not provide
native FP16 arithmetic. The validation host for this change has no AVX-512
FP16/BF16 support, so no throughput claim is made for those instructions.
FP16 accumulation is not substituted for FP32 accumulation, and probabilities
are not rounded to BF16 simply to use a native BF16 value product.

AVX2 measurements and remaining gap
-----------------------------------

A controlled 2026-09-17 campaign used an AMD EPYC 7763, GCC 13 Release
build with an AVX2 ceiling, Python 3.13, onnx-light 0.1.27 and ONNX Runtime
1.30.0. Both runtimes used one thread, affinity CPU 0, and disabled ORT
worker spinning. Each case used 20 warmups and up to 100 alternating samples
(0.25 seconds per candidate). Compilation and scanning did not overlap these
measurements.

An additional A/B/A check isolated the FP16 tiled softmax dispatch: its scores
already have FP32 storage, but the old codec gate unnecessarily excluded the
AVX2/FMA softmax. Removing that gate, while retaining the CPU, mask, softcap
and window restrictions, produced:

.. list-table:: FP16, batch 1, 12 heads, q128/kv128/head dimension 64
   :header-rows: 1
   :widths: 30 20 20 15 15

   * - Layout / mask
     - Before (ms)
     - After (ms)
     - Before / after
     - ORT / after
   * - Rank 4 / none
     - 2.523
     - 1.100
     - 2.29x
     - 0.742x
   * - Rank 4 / causal
     - 1.828
     - 1.083
     - 1.69x
     - 0.787x
   * - Rank 3 / none
     - 2.547
     - 1.124
     - 2.27x
     - 1.303x

These are softmax-dispatch A/B measurements on the bounded-conversion
implementation, not a claim that conversion removal alone gives those gains.
Repeating the before candidate reproduced 2.516/1.822/2.544 ms. Short-query
measurements also caught a regression from converting K/V independently for
each query; bounded reuse corrected it (q8/kv257/hd128: approximately
0.639 to 0.514 ms; q8/kv1024/hd64: approximately 1.312 to 1.305 ms).

A separate 28-case original/optimized campaign used the same host, affinity
and threading policy with 3 warmups, 20 samples and a 0.1-second cap. Its
median original/optimized latency ratio was 1.31x for FP16 and 0.98x for FP32.
FP16 rank-four q128/kv128/hd64 improved from 2.471 to 1.059 ms (2.33x);
the rank-three equivalent improved from 2.508 to 1.122 ms (2.24x).
Longer A/B/A confirmation runs put the apparent FP32 hd128/256 regressions
at 0.993x/0.995x, consistent with unchanged performance. Contended samples
were discarded; no four-thread speedup claim is made here.

The 0.9x parity target is **not met for every comparable case**. In particular,
rank-four tiled cases and wide-head tails remain below it. The remaining work
is not a missing registered-kernel dispatch or a full-tensor conversion:

* Streaming still evaluates scores and accumulates values per query row;
  contexts longer than the bounded reuse window revisit and reconvert K/V.
* Tiled attention still submits separate QK and value GEMMs for every tile.
  GEMM panel packing, skinny/tail kernels, online-softmax bookkeeping and
  exponentials remain, even when explicit attention packing is tile-local.
* A native ``gprof`` run of the differential tests confirms AVX2/FMA score,
  value-accumulation and GEMM functions execute. Their sampled costs dominate
  the remaining explicit conversion helpers. This profile includes oracle
  computation and test-data generation; its percentages are not end-to-end
  benchmark phase percentages. Whole-test self-time shares were 37.62% for
  the materialized oracle, 11.54% for AVX2 value accumulation and 7.74% for
  AVX2 scores. Inlined work cannot be attributed separately to conversion
  merely by summing named conversion helpers.

Reproduce the focused registered-kernel campaign with:

.. code-block:: bash

   python tools/benchmark_attention_parity.py --threads 1 --cpus 0 \
     --case '^test_cpu_attention_opset23_rank[34]_mha_q128_kv128_hd64_(none|causal)_stateless_float16_benchmark$' \
     --repeat 100 --warmup 20 --max-repeat-time 0.25 \
     --output /tmp/attention-parity.json

For native attribution, configure the existing ``test_attention_plan`` target
with ``-DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg`` and an AVX2 ceiling,
then profile tiled and streaming test filters separately using ``gprof``.
Hardware ``perf`` counters were unavailable on this host.

Issue 982: AVX2 tile products and short queries
-----------------------------------------------

The 2026-10-24 investigation reproduced all 37 named gaps on an AMD EPYC
7763 virtual machine (four visible logical CPUs). Baseline code was
``f9adcb9``; optimized code was ``b33e854``. Both used onnx-light-cpu 0.2.0,
GCC 13.3.0, CMake Release, and ``ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2``.
The runtime dependencies were the onnx-light 0.2.0 CPython 3.13 Linux
release wheel (tag commit ``6400f0f812b13bc32469e405bd2f762ba3cba7f0``),
ONNX Runtime 1.31.0, NumPy 2.4.3, and ml-dtypes 0.5.4; Python was 3.13.16.
The CPU extension linked the exact wheel libraries using matching tag
headers and ``ONNX_LIGHT_CPU_PYTHON_STABLE_ABI=OFF``.

Measurements used the registered backend cases without changing their
models, inputs, or tolerances. Both backends used one thread and affinity
CPU 0, with 20 warmups and up to 100 alternating samples (one second per
candidate). ORT used sequential execution, one inter-op thread, and disabled
intra/inter-op spinning. With one effective thread neither backend has idle
worker threads competing between samples. Compilation and profiling did not
overlap timed runs. The runner checked every output against ORT before timing.

Two bottlenecks were isolated by dispatch A/B measurements:

* The tiled probability/value product repacked already contiguous attention
  tiles through general GEMM. Calling the existing six-row AVX2/FMA microkernel
  directly avoids that packing and inner scheduling. The fast path is bounded
  to V dimensions 16--256 and at most 512 keys. Full 128-row tiles with V
  dimensions above 64 retain general GEMM, since measurements did not establish
  a benefit there. Outer runtime scheduling is unchanged.
* Short queries reread K/V row by row. Eligible FP32/FP16 queries with 8--15
  rows now reuse tiled products; aligned FP16 queries with 16 rows no longer
  take the old streaming override on AVX2/FMA. An invocation-local vectorized
  finite-value check preserves the old streaming behavior for NaN/Inf in V:
  streaming skips zero probabilities, whereas dense GEMM evaluates zero times
  NaN/Inf. This scan is inside the measured invocation; it is not cached.
  It shares the tiled product's causal, window and per-batch nonpad bounds,
  so unavailable V prefixes and suffixes are not scanned. Rank-three scans
  include the interleaved KV heads only within the selected rows.

Both decisions cache CPU feature detection and leave unsupported CPUs,
dimensions, tensor-cache inputs, sink attention, and materialized dispatch
on the existing paths. Products retain FP32 accumulation and the existing
softmax. No new packing buffer or executor is introduced.

Contiguous additive masks retain AVX2 decode for finite rows. Only rows with
reachable NaN mask entries fall back to scalar streaming. Tiled AVX2 softmax
also preserves NaNs computed by QK: a check limited to its otherwise-zero
branch distinguishes these from fully masked rows. Normal finite blocks
do not incur an extra score scan.

A ``gprof`` run of the new differential test, repeated 100 times, confirmed
the six-row AVX2/FMA GEMM microkernel and AVX2 softmax execute. This includes
materialized-oracle, data-generation, and assertion costs, so its self-time
percentages are not end-to-end phase percentages. Hardware and software
``perf`` events were blocked by ``perf_event_paranoid=4``.

The table reports median milliseconds for **every issue case**. Restore the
full name by adding ``test_cpu_attention_`` and ``_benchmark`` to the case
column. ``Repeat`` is a second baseline-equivalent campaign with both new
dispatch gates disabled, before restoring the final implementation. ``ORT``
is from the final optimized campaign.

.. csv-table::
   :header: "Case", "Before", "After", "Repeat", "ORT"
   :widths: 64 9 9 9 9

   "opset23_rank4_mha_q128_kv128_hd64_none_stateless_float32", "1.0400", "1.0048", "1.0389", "0.6975"
   "opset23_rank4_mha_q128_kv128_hd64_causal_stateless_float32", "1.0109", "0.8456", "1.0100", "0.7237"
   "opset23_rank4_mha_q128_kv128_hd64_bool_stateless_float32", "1.0998", "1.0579", "1.0845", "0.7719"
   "opset23_rank4_mha_q128_kv128_hd64_additive_stateless_float32", "1.1059", "1.0660", "1.0963", "0.7206"
   "opset23_rank4_gqa_q128_kv128_hd64_causal_stateless_float32", "1.3368", "1.1152", "1.3296", "0.9517"
   "opset23_rank4_mqa_q128_kv128_hd64_causal_stateless_float32", "1.3450", "1.1170", "1.3398", "0.9478"
   "opset23_rank3_mha_q128_kv128_hd64_none_stateless_float32", "1.1128", "1.0903", "1.0945", "0.7019"
   "opset23_rank4_mha_q8_kv128_hd64_none_stateless_float32", "0.1420", "0.1283", "0.1546", "0.0936"
   "opset23_rank4_mha_q16_kv128_hd64_none_stateless_float32", "0.1943", "0.1723", "0.1948", "0.1272"
   "opset23_rank4_mha_q512_kv128_hd64_none_stateless_float32", "4.2331", "4.0650", "4.0970", "2.6250"
   "opset23_rank4_mha_q128_kv1024_hd64_none_stateless_float32", "8.1838", "7.5957", "8.2445", "5.2784"
   "opset23_rank4_mha_q128_kv4096_hd64_none_stateless_float32", "33.5599", "31.7309", "33.8006", "22.2247"
   "opset23_rank4_mha_q128_kv8192_hd64_none_stateless_float32", "67.0251", "63.8923", "67.6594", "44.4586"
   "opset23_rank4_mha_q128_kv128_hd128_causal_stateless_float32", "1.5140", "1.3283", "1.4905", "1.2738"
   "opset24_rank4_mha_q8_kv1024_hd64_causal_nonpad_float32", "1.0805", "0.7461", "1.1581", "0.6090"
   "opset23_rank4_mha_q128_kv128_hd64_none_stateless_float16", "1.0505", "1.0647", "1.0984", "0.8203"
   "opset23_rank4_mha_q128_kv128_hd64_causal_stateless_float16", "1.0591", "0.8821", "1.0750", "0.8467"
   "opset23_rank4_mha_q128_kv128_hd64_bool_stateless_float16", "1.0835", "1.0922", "1.1246", "0.8683"
   "opset23_rank4_gqa_q128_kv128_hd64_causal_stateless_float16", "1.3663", "1.1383", "1.3694", "1.1274"
   "opset23_rank4_mqa_q128_kv128_hd64_causal_stateless_float16", "1.3617", "1.1357", "1.3641", "1.1164"
   "opset23_rank4_mha_q16_kv128_hd64_none_stateless_float16", "0.3177", "0.1814", "0.2990", "0.1664"
   "opset23_rank4_mha_q512_kv128_hd64_none_stateless_float16", "4.1415", "4.1963", "4.3767", "3.0960"
   "opset23_rank4_mha_q128_kv1024_hd64_none_stateless_float16", "8.1760", "7.6773", "8.3561", "6.3736"
   "opset23_rank4_mha_q128_kv4096_hd64_none_stateless_float16", "32.9396", "30.9296", "33.5952", "25.6175"
   "opset23_rank4_mha_q128_kv8192_hd64_none_stateless_float16", "66.0318", "61.6873", "66.4951", "51.4994"
   "opset24_rank4_mha_q8_kv1024_hd64_causal_nonpad_float16", "1.3200", "0.7385", "1.2315", "0.7577"
   "opset23_rank4_mha_q16_kv257_hd63_none_stateless_float32", "0.3838", "0.3290", "0.3798", "0.2449"
   "opset23_rank4_mha_q128_kv128_hd128_none_stateless_float32", "1.8693", "1.8063", "1.8225", "1.2474"
   "opset23_rank4_mha_q128_kv128_hd256_none_stateless_float32", "3.4873", "3.4405", "3.4513", "2.3648"
   "opset23_rank3_mha_q129_kv257_hd64_none_stateless_float32", "2.4668", "2.2694", "2.4674", "1.4101"
   "opset23_rank4_mha_q129_kv257_hd256_none_stateless_float32", "7.7981", "7.2656", "7.7610", "4.8871"
   "opset23_rank4_mha_q8_kv257_hd128_none_stateless_float32", "0.4851", "0.3845", "0.4899", "0.3047"
   "opset23_rank4_mha_q16_kv257_hd63_none_stateless_float16", "0.4018", "0.3496", "0.4027", "0.3125"
   "opset23_rank4_mha_q128_kv128_hd128_none_stateless_float16", "1.8976", "1.9155", "1.9752", "1.4050"
   "opset23_rank4_mha_q128_kv128_hd256_none_stateless_float16", "3.5762", "3.5596", "3.7049", "2.6132"
   "opset23_rank4_mha_q129_kv257_hd256_none_stateless_float16", "7.8824", "7.2710", "7.9491", "5.4344"
   "opset23_rank4_mha_q8_kv257_hd128_none_stateless_float16", "0.5517", "0.3718", "0.5447", "0.4047"

The median before/after ratio across all 37 cases is 1.077x. The largest
improvement is 1.787x for FP16 causal/nonpad q8/kv1024/hd64. Repeated baselines
confirm the short-query and causal improvements. Small dense-case differences
overlap baseline variation: in particular no repeatable improvement is claimed
for rank-three q128 or full-width FP16 hd128/256. The latter retain the original
value-product dispatch. FP16 dense hd64 and q512 also remain effectively
unchanged rather than establishing a regression.

**Parity is not achieved:** ORT remains faster on 35 of 37 final measurements.
Even the FP16 nonpad case that crosses parity is near enough to require caution.
All original gaps reproduced; no missing or failed timing was counted as a win.
The absolute latencies are not interchangeable with the dashboard campaign.

To repeat a table row, use its full case name as an anchored ``--case`` filter
with the existing runner, and apply the same settings to each row:

.. code-block:: bash

   REPO=/home/runner/work/onnx-light-cpu/onnx-light-cpu
   python "$REPO/tools/benchmark_attention_parity.py" --large \
     --threads 1 --cpus 0 --warmup 20 --repeat 100 --max-repeat-time 1 \
     --case '^test_cpu_attention_opset24_rank4_mha_q8_kv1024_hd64_causal_nonpad_float16_benchmark$' \
     --output /tmp/attention-case.json

Validation comprises all 90 ``test_attention_plan`` tests, the 37 runner
correctness comparisons, and clang-format checks. New tests cover tile and
eligibility boundaries, both layouts, grouped heads, empty/single-key inputs,
nonpad and fully masked rows, runtime-owned scheduling without nesting, and
every finite-check tail through 33 elements including NaN and both infinities.
They also verify that unavailable V rows do not disable short-query tiles under
causal, window and per-batch nonpad bounds. The test target mirrors the library's
AVX2/FMA availability definition while retaining runtime CPU feature checks.
This build system does not expose a scalar-only ceiling; no scalar or
AVX-512 performance claim is made.

Issue 986: short causal GroupQueryAttention on AVX2
---------------------------------------------------

The 2026-10-25 reproduction used an AMD EPYC 7763 VM with four visible logical
CPUs, GCC 13.3.0 Release, Python 3.13.16, and an AVX2 ceiling (reported SIMD
level 3). Baseline production code was ``c945d91``; the optimized code is
``b50c0fb``. Dependencies were onnx-light-cpu 0.2.0, the onnx-light 0.2.0
CPython 3.13 Linux release wheel (tag commit
``6400f0f812b13bc32469e405bd2f762ba3cba7f0``), ONNX Runtime 1.31.0 (build
commit ``402979a``), NumPy 2.4.4, and ml-dtypes 0.6.0. The CPU extension linked
the wheel's libraries with matching ``ONNX_LIGHT_CPU_PYTHON_STABLE_ABI=OFF``.

All four reported gaps reproduced twice before optimization. Measurements
used the exact registered ``com.microsoft::GroupQueryAttention`` benchmark
models and inputs, not replacement workloads. Both backends used one thread,
affinity CPU 0, 100 warmups, and 1,000 measured iterations per case, with a
three-second cap that was never reached. The existing benchmark CLI measures
the CPU phase before constructing ORT, then measures ORT; it does not
alternate individual samples. CPU feeds were backend tensors, ORT feeds were
NumPy arrays, and recording was disabled during timing. ORT used sequential
execution and one inter-op thread. Spin defaults were unchanged: onnx-light
resolved adaptive spinning with 10,000 iterations and ORT retained its default
spinning. With one effective thread neither has idle workers to spin.
Compilation and profiling did not overlap timed runs.

Short grouped prefill was paying packing and GEMM setup costs for every
16-row query tile. AVX2/FMA now uses up to 32 rows for causal rank-three grouped
attention with query/KV lengths at most 64 and both head dimensions 64.
The executor, input conversions, accumulation precision, and softmax are
unchanged. A 64-row experiment was rejected: GQA FP32/FP16 measured
0.439922/0.430304 ms versus 0.419815/0.398470 ms for the corresponding
32-row experiment without the safety scan.

Wider tiles evaluate additional masked products. An invocation-local
vectorized V check retains the original 16-row tiles for NaN/Inf and unsafe
FP32 magnitudes, preserving earlier causal rows that must not acquire
zero-times-NaN/Inf. BF16 retains its old tile size. The check is inside the
measured call, does not cache input contents, and adds no scratch allocation.
Execution telemetry reports the actual tile size, including fallback.

Callgrind 3.22.0 profiled
``ComputeAttentionFloat32Streaming.SmallCausalGqaCapsWorkerCount`` repeated
20 times. Original versus final QK GEMM and transpose-packing calls fell from
1,280 to 640 each. Transpose-packing instructions fell from 2,901,760 to
1,840,000; the final finite-value scan ran 20 times. AVX2/FMA microkernels
and softmax were present in both profiles. Wider tiles do more masked
arithmetic, so total instructions increased: this is an amortization/latency
improvement, not a FLOP reduction. These native-test profiles include data
generation and assertions and are not end-to-end phase percentages.
Hardware ``perf`` events were blocked by ``perf_event_paranoid=4``.

The following are median milliseconds. Restore each full benchmark name with
the prefix ``test_cpu_group_query_attention_`` and suffix ``_benchmark``.
``Before repeat`` restores the 16-row target after the initial experiments;
``After`` and ``After repeat`` both include the safety scan. ``ORT`` is from
the final repeated campaign.

.. csv-table::
   :header: "Case", "Before", "Before repeat", "After", "After repeat", "ORT"
   :widths: 57 9 9 9 9 9

   "mqa_b4_s32_qh8_kvh1_hd64_causal_float32", "0.336033", "0.337977", "0.315289", "0.316967", "0.198666"
   "mqa_b4_s32_qh8_kvh1_hd64_causal_float16", "0.351492", "0.355936", "0.323084", "0.322213", "0.226002"
   "gqa_b2_s64_qh8_kvh2_hd64_causal_float32", "0.478579", "0.480137", "0.432013", "0.431441", "0.291990"
   "gqa_b2_s64_qh8_kvh2_hd64_causal_float16", "0.428896", "0.432323", "0.402422", "0.401309", "0.313496"

Latency falls approximately 6--10% on all four single-thread cases.
**ORT remains faster on all four**; parity is not claimed. Absolute timings
are not interchangeable with the dashboard's unspecified thread/build
provenance.

A separate four-thread check used affinity CPUs 0--3 with the same warmup,
repetition, phase order, and unchanged spin defaults. These are before/after
CPU milliseconds, with ORT from the final run:

.. csv-table::
   :header: "Case", "Before", "After", "ORT"
   :widths: 64 12 12 12

   "MQA FP32", "0.163416", "0.154810", "0.134000"
   "MQA FP16", "0.174170", "0.173730", "0.185992"
   "GQA FP32", "0.228642", "0.215218", "0.185827"
   "GQA FP16", "0.221760", "0.209146", "0.213048"

The four-thread MQA FP16 difference is within noise; no improvement is
claimed there. Its dashboard gap does not reproduce at four threads even
before optimization. GQA FP16 finishes near parity, which requires caution
rather than a general parity claim. No failed or missing measurement was
counted as a win.

Reproduce all four cases with the existing CLI:

.. code-block:: bash

   REPO=/home/runner/work/onnx-light-cpu/onnx-light-cpu
   export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
   CMAKE_ARGS='-DONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2 -DONNX_LIGHT_CPU_BUILD_TESTS=ON -DONNX_LIGHT_CPU_PYTHON_STABLE_ABI=OFF' \
     python "$REPO/setup.py" build_ext --inplace --onnx-light-source
   taskset -c 0 python -m onnx_light_cpu benchmark \
     --test '^test_cpu_group_query_attention_(mqa_b4_s32_qh8_kvh1|gqa_b2_s64_qh8_kvh2)_hd64_causal_float(16|32)_benchmark$' \
     --dtype float32 float16 --threads 1 --warmup 100 --repeat 1000 \
     --max-repeat-time 3 --onnxruntime --output /tmp/gqa.xlsx

Validation passed all 91 shared Attention tests, 20 GQA adapter tests,
10 naive-GQA tests, and 12 MHA adapter tests. The exact four registered
benchmark inputs also matched ORT (FP32 rtol/atol 2e-4/2e-5; FP16
2e-2/2e-3). Added differential cases cover lengths 31/32/33 and 63/64/65
in both layouts and types, causal/nonpad, boolean/additive and fully masked
rows, and runtime scheduling without nesting. The unsafe-value regression
checks NaN, both infinities, and extreme FP32 values, then changes V back to
finite values to verify invocation-local selection. No AVX-512 or scalar
performance claim is made.

Issue 1029: retained AVX2 dense-softmax and strided-output optimization
--------------------------------------------------------------------

The 2026-10-27 follow-up retains two computation changes, rather than only
documenting attribution:

* For a finite, unmasked, noncausal single-KV-block tile, reuse the existing
  AVX2/FMA softmax polynomial and range selection, but fuse subtraction,
  exponentiation and denominator reduction. Leave probabilities
  **unnormalized**: the value product still accumulates in FP32 before final
  output normalization. Masks, nonpad lengths, windows, softcap, multiple KV
  blocks and nonfinite scores retain the existing online-softmax path.
* Let the existing direct AVX2 value microkernel write through the output
  sequence stride. Rank-three FP32 no longer needs a contiguous output
  temporary followed by a scatter for eligible single-block products.
  The value-kernel eligibility is unchanged and shared by dispatch and
  destination selection; general GEMM still receives contiguous output.

There are no new scratch buffers, worker allocations, packing caches,
threads, ISA flags or wider value-GEMM dispatch gates. Scalar and AVX-512
dispatch remain unchanged. The shared normalized Softmax entry point uses
the same implementation with normalization enabled.

Measurements
~~~~~~~~~~~~

The baseline is remote main ``233c60976a6e0de7a183abb48631106fdfd4a186``,
refreshed with ``git fetch origin`` before building. This is a different
machine from the previous campaigns: Intel Xeon Platinum 8480C, GCC 13.2.0,
CMake 4.1.1, Release ``-O3 -DNDEBUG``, and an **AVX2 ceiling**. The dependency
is the onnx-light source checkout at
``6d6364cc44bf060eafe913bfa6e6b03b68dc5778``, linked against the libraries
reported by ``get_cpp_build_info()``, with the Python stable ABI enabled.
Python is 3.12.15, NumPy 2.5.3 and ml-dtypes 0.5.4. The imported ORT is a
local 1.32.0 Release build at ``782dc0c5b5562b52319d735ae31a4d6e8349e656``,
on branch ``moe-fp16-cpu-offload-and-adaptive-swap``. Installed distribution
metadata reports 1.29.0, but does not describe the runtime imported through
``PYTHONPATH``; the ORT version above comes from ``onnxruntime.__version__``
and ``get_build_info()``.

ORT is not capped at AVX2. A debugger trace of the rank-four FP16
128/128/256 case confirms that its Attention FP32 GEMM fallback reaches
``MlasGemmFloatKernelAvx512F``. These timings therefore compare our
AVX2-ceiling build with unrestricted ORT, not ISA-matched implementations.

Separate-process timings drifted substantially, including ORT controls.
Consequently, the controlled A/B campaign used the **same ReferenceEvaluator,
registered models and feeds in one process**, pinned to CPU 2, one runtime
thread, 40 warmups and 400 measured samples per candidate. A temporary
invocation-level boolean enabled/disabled both new decisions; it was set
outside each session run. Disabling it restored online softmax and buffered
rank-three output. The boolean and its checks are not retained in production.
The repeat rotated old/new/ORT order every iteration, with ORT sequential,
one intra/inter-op thread and both spinning options disabled. Compilation
and other timed campaigns did not overlap. Correctness comparisons used the
existing runner's unchanged tolerances.

Median milliseconds below are from that repeated three-candidate campaign.
``Speedup`` is before/after, not ORT/CPU. ``First speedup`` is the preceding
400-sample alternating old/new campaign, included to expose repeatability.

.. csv-table::
   :header: "Layout/type, Q/KV/D", "Before", "After", "ORT", "Speedup", "First speedup"

   "rank3 FP32, 128/128/64", "0.83794", "0.75006", "0.46113", "1.117x", "1.092x"
   "rank3 FP32, 129/257/64", "1.76365", "1.63098", "0.99367", "1.081x", "1.072x"
   "rank4 FP16, 128/128/64", "0.77298", "0.72870", "0.63915", "1.061x", "1.063x"
   "rank4 FP16, 16/128/64", "0.14813", "0.14210", "0.13243", "1.042x", "1.038x"
   "rank4 FP16, 16/257/63", "0.29808", "0.28766", "0.27288", "1.036x", "1.032x"
   "rank4 FP16, 128/128/128", "1.74193", "1.69199", "1.27978", "1.030x", "1.029x"
   "rank4 FP16, 128/128/256", "3.29573", "3.25455", "2.37042", "1.013x", "1.023x"
   "rank4 FP16, 129/257/256", "7.50962", "7.42169", "4.96428", "1.012x", "1.015x"

The rank-three reduction is repeatable (approximately 7--10% lower latency).
FP16 benefits diminish as head dimension increases: the hd256 differences
are small, and **the wide-head bottleneck is not resolved**. ORT remains
faster on every row; issue 1029 must not be considered fully fixed.
Direct multi-query dot products and alternative value microtiles were also
tried, but their regressions or inconsistent benefits did not justify
retaining them.

Reproduce the production-path ORT comparison with the normal ``AUTO``
configuration and binaries installed only into the source checkout.
The historical measurements above used an explicit AVX2 ceiling instead.

.. code-block:: bash

   cmake -S . -B build/temp -UONNX_LIGHT_CPU_MAX_SIMD_LEVEL \
     -DML_CPU_COMPUTE_MAX_SIMD_LEVEL=AUTO
   cmake --build build/temp --parallel 48 \
     --target test_attention_plan test_exp_log_kernel _cpuregister _cpukernels
   cmake --install build/temp --prefix "$PWD"
   python tools/benchmark_attention_parity.py \
     --threads 1 --cpus 2 --warmup 40 --repeat 400 --max-repeat-time 4 \
     --case '^test_cpu_attention_opset23_rank3_mha_q(128|129)_kv(128|257)_hd64_none_stateless_float32_benchmark$' \
     --case '^test_cpu_attention_opset23_rank4_mha_q(16|128|129)_kv(128|257)_hd(63|64|128|256)_none_stateless_float16_benchmark$' \
     --output /tmp/attention-1029-auto.json

Validation includes all 93 Attention tests, 36 shared math tests and 18
Sigmoid/Softmax adapter tests under the AVX2 ceiling. An AVX-512-ceiling
build passes 89 Attention tests (four AVX2-only tests skip) and all 39 shared
math tests. New coverage compares the
fused unnormalized primitive with online softmax across vector tails and
normal/subnormal/extreme ranges, verifies nonfinite-score fallback followed
by finite reuse, and extends materialized-oracle comparisons to wide heads,
q128/q129, KV257/KV513, both layouts and precisions, masks and runtime-owned
scheduling. All 48 default-corpus stateless/nonpad FP32/FP16 cases also
pass the existing ORT correctness comparison in the final production build;
that short correctness run is not performance evidence and does not pass
the runner's ORT performance-parity thresholds. No AVX-512 or scalar
performance improvement is claimed.

Corrected AUTO/AVX-512 comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The follow-up removes the installed editable distribution and its import
hook. Normal Python imports now load both native CPU modules from the source
checkout. ``ML_CPU_COMPUTE_MAX_SIMD_LEVEL=AUTO`` reports ``AVX512`` on this
machine; no AVX2 ceiling is retained. The dependency checkout has meanwhile
advanced to ``34c79978604659f7910ed234b687e4402c68451e`` (onnx-light 0.2.1).
Its stale ``_onnxpyprotoop`` module required an incremental rebuild before
the benchmark could import successfully.

The runner now feeds backend tensors directly to onnx-light-cpu, as the
shared backend benchmark does; only ORT receives NumPy feeds. It records
imported module versions, native module paths, the detected CPU SIMD level
and ORT's build information instead of trusting stale installed-distribution
versions. ORT remains the local 1.32.0 build identified above.

Two independent campaigns use CPU 2, one thread per runtime, alternating
CPU/ORT order, 40 warmups and exactly 400 samples per runtime and case.
The table gives second-campaign medians in milliseconds and both campaigns'
``ORT/CPU`` ratios; a ratio above one favors onnx-light-cpu.

.. csv-table::
   :header: "Layout/type, Q/KV/D", "CPU", "ORT", "First ORT/CPU", "Second ORT/CPU"

   "rank3 FP32, 128/128/64", "0.65581", "0.55289", "0.887x", "0.843x"
   "rank3 FP32, 129/257/64", "1.93380", "1.17566", "0.611x", "0.608x"
   "rank4 FP16, 128/128/64", "0.69998", "0.74784", "1.082x", "1.068x"
   "rank4 FP16, 16/128/64", "0.36938", "0.18143", "0.501x", "0.491x"
   "rank4 FP16, 16/257/63", "0.57118", "0.35217", "0.625x", "0.617x"
   "rank4 FP16, 128/128/128", "1.27021", "1.29494", "1.051x", "1.019x"
   "rank4 FP16, 128/128/256", "2.34545", "2.35102", "1.057x", "1.002x"
   "rank4 FP16, 129/257/256", "5.87519", "4.54726", "0.777x", "0.774x"

A separate 100-sample, 20-warmup campaign checks all 48 original
FP32/FP16 stateless/nonpad cases. Numerical comparisons all pass with
unchanged tolerances. Median ORT/CPU ratios are 0.910 for FP32 and 1.071
for FP16, but individual minima are 0.426 and 0.516 respectively.
Performance parity is therefore **not achieved**. BF16 is not measured
in these campaigns. Absolute timings drift: second-campaign interquartile
ranges are approximately 3--11% of CPU medians and 4--13% of ORT medians,
so near-parity cases do not establish a reliable small advantage.

This changes the diagnosis:

* Square FP16 hd256 is approximately at parity, rather than a uniformly
  unresolved wide-head failure. The 129/257 tail case still trails ORT
  by approximately 29% in latency.
* Aligned FP16 Q16/KV128/D64 selects ``Attention.streaming`` and takes
  approximately twice ORT's latency. The existing aligned-short gate
  excludes tiling on an AVX-512 host because its AVX2-only short-tile
  exception requires an exact AVX2 SIMD level.
* Rank-three FP32 129/257 remains approximately 65% slower than ORT.
  The existing AVX-512 dense-softmax shortcut only covers FP32 KV128
  with Q8/Q128, not this case. Its recorded path still packs tiles and
  writes tile output; recording alone does not quantify those costs.
* The retained optimization gates explicitly require an AVX2 ceiling.
  These new measurements therefore do **not** demonstrate a benefit
  from that optimization on the normal AVX-512 configuration.

Changes in dependency revision, input adaptation and runtime binaries mean
the old and new tables are not a controlled AVX2-versus-AVX-512 experiment.
No ISA-only speedup or new hot-symbol attribution is claimed. Issue 1029
remains open; default-configuration dispatch and tail handling still need
optimization.

Issue 1028: retained Intel AVX2 dense FP32 improvement
------------------------------------------------------------

The 2026-10-28 follow-up starts from refreshed remote main
``cc34d0d8fe6d2372cf6472466e3438a8bfae25e6``. Unlike the closed profiling-only
PR 1058, this change retains two production optimizations:

* QK uses the actual attention tile depth for GEMM blocking and six register
  rows. The general GEMM profile's nominal KC otherwise selects unnecessarily
  narrow packed column panels for these short dot products.
* P*V with 128 value columns uses a three-row, 32-column register tile.
  Twelve accumulators, three broadcast probabilities and one loaded value
  vector fit in sixteen YMM registers, without packing or allocating scratch.
  Existing GEMM micro-tiles handle the remaining one or two query rows.

Both require the Intel Core GEMM profile, AVX2 with FMA, rank-four FP32,
at least 128 queries, no mask, causal/window bound, softcap or nonpad length.
Individual tile bounds retain the existing fallback for small tiles and
unsupported head sizes. Scheduling ownership, score-tile sizes, accumulation
precision and tolerances do not change. AMD, generic profiles, rank-three,
FP16/BF16 and the default AVX-512 path remain unchanged: the rejected AMD
blocking experiments in PR 1058 do not justify enabling these gates there.

Matched, repeated A/B measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The host is an Intel Xeon Platinum 8480C, GCC 13.2 Release, Python 3.12.15,
NumPy 2.5.3 and ml-dtypes 0.5.4. The onnx-light 0.2.2 source checkout is
``06912a3676fa2c58a47ce51d5acd124d8e0bedd4`` with stable ABI enabled.
ORT 1.32.0 is the local Release build at ``782dc0c5b5``.
The CPU build reports AVX2 using ``ML_CPU_COMPUTE_MAX_SIMD_LEVEL=AVX2``;
ORT is unrestricted on this AVX-512-capable host. These are not ISA-parity
measurements, nor a reproduction on the original AMD runner.

Both independent campaigns use the original five registered models and
inputs, CPU-0 affinity, one thread per backend, sequential ORT, one inter-op
thread and disabled ORT spinning. Baseline, candidate and ORT calls rotate
order in the same process, reusing the same sessions and feeds. A temporary
invocation-level gate selects the original or optimized path outside timing;
it is removed from the retained code. There are 20 warmups and up to 100
samples per path, stopping once every path has accumulated two seconds.
The long-KV case collects 63 and 60 samples per path; the other cases collect
100. Model construction, data generation, correctness and compilation are
outside timing. CPU feeds are backend tensors, not NumPy arrays.

All values below are median milliseconds. Case names use the prefix
``test_cpu_attention_opset23_rank4_mha_`` and suffix
``_none_stateless_float32_benchmark``.

.. csv-table::
   :header: "Case", "Before", "After", "ORT", "Before repeat", "After repeat", "ORT repeat"

   "q128_kv128_hd64", "0.641815", "0.586902", "0.413437", "0.647782", "0.597532", "0.427517"
   "q512_kv128_hd64", "2.484012", "2.294868", "1.499381", "2.731780", "2.532912", "1.656027"
   "q128_kv8192_hd64", "43.550382", "42.091625", "30.219244", "45.617714", "45.018527", "31.222460"
   "q128_kv128_hd128", "1.315575", "1.201075", "0.838056", "1.212978", "1.108685", "0.758204"
   "q128_kv128_hd256", "2.258434", "2.182405", "1.388872", "2.256909", "2.175080", "1.368227"

The four shorter cases improve by 3.4--8.7% in both campaigns. The long-KV
reduction is only 1.3--3.4%; its smaller, less stable gain should not be
extrapolated. No selected case regresses in either matched comparison.
Final production-only measurements, after removing the temporary gate,
give the following additional CPU/ORT medians under the same settings:

.. csv-table::
   :header: "Q/KV/D", "Retained CPU", "ORT"

   "128/128/64", "0.675312", "0.446491"
   "512/128/64", "2.326355", "1.447188"
   "128/8192/64", "41.828070", "27.840577"
   "128/128/128", "1.147282", "0.774157"
   "128/128/256", "2.204210", "1.347139"

Separate-process drift remains visible, particularly in the first case;
the gain claims use the matched A/B campaigns, not differences between
separate runs. A separate native ``perf record -e cpu-clock:u`` run confirms
both the six-row AVX2 GEMM and ``AttentionValue128Float32_AVX2_FMA`` execute.
Its samples include startup and ORT, so no kernel phase-time share is claimed.
Hardware cycles/instructions are unsupported on this VM.

All 108 Attention C++ tests pass with the AVX2 ceiling. Coverage includes
the five original shapes, eligibility boundaries, head/query/KV tails,
multi-block accumulation, unaligned operands and padded output strides.
The five registered cases also pass ORT comparisons with one and two runtime
threads at the unchanged FP32 rtol/atol ``2e-4/2e-5``.
**ORT remains faster on all five cases, and issue 1028 remains open.**
The original AMD gap and default-ISA parity are not resolved by these
Intel AVX2-specific improvements.

The retained path can be reproduced from a source-linked build with:

.. code-block:: bash

   CMAKE_ARGS='-DML_CPU_COMPUTE_MAX_SIMD_LEVEL=AVX2' \
     python setup.py build_ext --inplace --onnx-light-source --parallel 48
   python tools/benchmark_attention_parity.py --large \
     --threads 1 --cpus 0 --warmup 20 --repeat 100 --max-repeat-time 2 \
     --case '^test_cpu_attention_opset23_rank4_mha_(q128_kv128_hd(64|128|256)|q512_kv128_hd64|q128_kv8192_hd64)_none_stateless_float32_benchmark$' \
     --output /tmp/attention-1028.json
