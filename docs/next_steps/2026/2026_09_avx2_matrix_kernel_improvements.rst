AVX2 Matrix Kernel Improvements
===============================

:Date: 2026-09-06

**Implemented; full ORT parity remains in progress**

This implementation follows the isolated diagnostic measurements in
`#645 <https://github.com/xadupre/onnx-light-cpu/pull/645>`_. It changes
production kernels rather than treating the earlier AVX2 issue closures as
evidence of parity.

Kernel changes
--------------

* FP16 and FP32 single-row products with non-transposed B use a depth-blocked
  AVX2/FMA range kernel. Eight vector accumulators cover 64 output columns;
  32-row depth slabs and bounded prefetching avoid repeatedly walking the
  entire weight matrix's page translations for every narrow output tile.
* The runtime executor partitions disjoint column ranges. There is no private
  thread pool, expanded weight tensor or new persistent prepack state.
  FP16 additionally requires F16C; wider-ISA paths retain their existing
  dispatch. FP32 bias/output aliasing keeps the existing fallback.
* General AVX2 FP16 products widen bounded panels once while packing, then
  reuse the FP32 micro-kernel instead of repeatedly converting compact
  operands inside every row tile. Fused packing and compute tasks avoid a
  full expanded B allocation and additional packing barriers.
* The existing two-column FP16 skinny kernel is now restricted to an actual
  physical N=2 matrix. A two-column remainder in a wider matrix must not read
  adjacent rows as if their physical stride were two.

Coverage includes depth and column tails, transposed A, alpha/beta and bias
aliasing, empty depth, NaN/infinity, direct F16C range bounds, runtime
participant limits, and the existing no-expanded-operand allocation gate.
The GEMM kernel and plan executables pass in AVX2-ceiling and automatic-ISA
builds; registered Gemm/MatMul integration also passes. This machine cannot
execute the AVX-512-specific cases, so native AVX-512 performance remains a
separate non-regression requirement.

Controlled before/after measurements
-------------------------------------

The baseline shared library was rebuilt independently from ``292bfb8`` with
the same Release compiler configuration and AVX2 ceiling as the modified
library. Each process recorded its actual loaded CPU library path to ensure
that the comparison selected the intended binary. Both libraries used the
same registered backend models, inputs and linked onnx-light runtime.

The machine was a Core i7-13800H under WSL2, with GCC 14.2.0, Python 3.12.3
and ONNX Runtime 1.28.0. One-thread processes used guest CPU 0; ten-thread
processes used ``0,2,4,6,8,10,12,14,16,18``. These are guest affinities, not
a claim about native P-core placement.

Baseline CPU, modified CPU and ORT ran in separate sequential processes,
preserving their normal thread-pool policies. The confirmation used up to
1,000 samples or two seconds per case and requested 20 warmups; the existing
CPU helper also caps warmup time. Variant order was modified/ORT/baseline
for one thread and baseline/ORT/modified for ten threads.

All numbers below are median milliseconds. **Ratio = ORT / modified CPU**;
values greater than 1 favor onnx-light-cpu.

.. list-table:: One thread
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Workload
     - Before
     - After
     - ORT
     - Ratio
   * - MatMul FP16, 64 square
     - 0.044
     - 0.034
     - 0.034
     - 1.01
   * - MatMul FP16, 128 square
     - 0.219
     - 0.086
     - 0.067
     - 0.77
   * - MatMul FP16, 512 square
     - 11.643
     - 4.191
     - 3.903
     - 0.93
   * - MatMul FP16, 1024 square
     - 91.686
     - 32.867
     - 32.546
     - 0.99
   * - MatMul FP32, M=1, K=N=4096
     - 14.560
     - 7.485
     - 3.837
     - 0.51
   * - Qwen QKV FP16, M=1, K=4096, N=6144
     - 33.150
     - 3.676
     - 16.536
     - 4.50
   * - Qwen gate/up FP16, M=1, K=4096, N=12288
     - 58.880
     - 6.768
     - 33.283
     - 4.92
   * - Gemm transformer projection FP16
     - 25.862
     - 10.371
     - 11.169
     - 1.08

.. list-table:: Ten threads
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Workload
     - Before
     - After
     - ORT
     - Ratio
   * - MatMul FP16, 64 square
     - 0.046
     - 0.034
     - 0.025
     - 0.74
   * - MatMul FP16, 128 square
     - 0.190
     - 0.089
     - 0.072
     - 0.81
   * - MatMul FP16, 512 square
     - 6.415
     - 2.077
     - 1.114
     - 0.54
   * - MatMul FP16, 1024 square
     - 21.665
     - 8.120
     - 9.160
     - 1.13
   * - MatMul FP32, M=1, K=N=4096
     - 3.845
     - 2.106
     - 2.082
     - 0.99
   * - Qwen QKV FP16, M=1, K=4096, N=6144
     - 41.807
     - 1.397
     - 6.318
     - 4.52
   * - Qwen gate/up FP16, M=1, K=4096, N=12288
     - 110.167
     - 3.683
     - 12.015
     - 3.26
   * - Gemm transformer projection FP16
     - 10.199
     - 3.364
     - 2.236
     - 0.66

The Qwen projections show substantial gains in repeated isolated runs, but
the exact multipliers are not a dedicated-hardware guarantee. The shared
environment remains variable: modified CPU p90/p10 reached 2.73 in this
confirmation. Do not compare these absolute times directly with older
reports that used a different one-thread affinity or the biased shared-pool
protocol.

Remaining work
--------------

The selected Qwen FP16 projections are now ahead of ORT, and several larger
FP16 cases are near parity. This is not family-wide or full-model parity:
FP32 M=1 with one thread, small FP16 matrices and medium multithread FP16
products still have gaps. Integer matrices, FP64, Attention and the full
isolated acceptance corpus remain separate work.

Retain the raw samples and actual binary provenance with future measurements.
Repeat the priority cases on dedicated hardware before accepting the final
``1.0x`` family median / ``0.9x`` minimum parity gate.
