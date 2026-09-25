Command Line
============

The package installs the ``onnx-light-cpu`` command. It can also be invoked
through the Python module:

.. code-block:: bash

   onnx-light-cpu --help
   python -m onnx_light_cpu --help

Benchmark backend test cases
----------------------------

The ``benchmark`` command runs selected ``TestMode.BENCHMARK`` backend test
cases with the onnx-light-cpu kernels and writes the measurements to an Excel
workbook:

.. code-block:: bash

   onnx-light-cpu benchmark \
       --tests "^test_cpu_(abs|gemm)_" \
       --dtypes float32 float64 \
       --repeat 100 \
       --warmup 10 \
       --max-repeat-time 2 \
       --threads 4 \
       --onnxruntime \
       --pr 623 \
       --output benchmark.xlsx

``--test`` (or ``--tests``)
   One or more regular expressions matched against backend case names. The
   option may be repeated. The default is ``^test_cpu_``.

``--dtype`` (or ``--dtypes``)
   One or more data types, supplied separately or as a comma-separated list.
   The option may be repeated. Supported values are ``bfloat16``, ``float16``,
   ``float32``, ``float64``, signed and unsigned 8-, 16-, 32-, and 64-bit
   integers, and ``bool``. The default, ``all``, selects every supported type
   and cannot be combined with another type.

``--compare-dtypes BASELINE COMPARISON``
   Selects two distinct data types and compares onnx-light-cpu median times for
   matching backend tests. It cannot be combined with ``--dtype``/``--dtypes``.
   For example, compare float16 and bfloat16 for the same Abs workloads:

   .. code-block:: bash

     onnx-light-cpu benchmark \
         --tests "^test_cpu_abs_" \
         --compare-dtypes float16 bfloat16 \
         --output float16_bfloat16.xlsx \
         --markdown float16_bfloat16.md

   The workbook gains a ``dtype_comparison`` sheet containing both median
   times in seconds and ``speedup = baseline_median_s / comparison_median_s``.
   A value above one means the second dtype (bfloat16 here) is faster.
   The full Markdown report gains the same comparison table; ``--pr`` and
   ``--pr-markdown`` show dtype speedups instead of ONNX Runtime speedups,
   with the ordered dtype pair in each test name.

   Cases are paired only when their names differ solely in dtype tokens and
   their operators and input shapes match. Homogeneous binary signatures such
   as ``float16xfloat16_to_float16`` are supported. The test regular expression
   must match both variants. Missing counterparts have blank timings and
   speedups (``None`` in Markdown). Mixed-dtype cases such as Cast conversions
   are excluded from the comparison table, but remain in the measurements.
   Raw and aggregated sheets remain unchanged, and ``--onnxruntime`` can still
   be used to include its measurements there. When inferring tests from a pull
   request, this option overrides the inferred dtypes.

``-r``, ``--repeat``
   Maximum number of measured iterations per case. The default is ten times
   the number of logical CPUs.

``-w``, ``--warmup``
   Maximum number of warm-up iterations per case. The default is twice the
   number of logical CPUs.

``-t``, ``--max-repeat-time``
   Maximum time in seconds for each of the warm-up and measurement phases of a
   case. The default is one second.

``--threads``
   Session thread limit. The default, ``0``, lets the runtime derive the
   available participants from the processor topology. Each kernel then selects
   how many participants to use from that capacity according to its own work
   size, thresholds, preferred participant count, and maximum participant count.
   A positive value imposes an explicit common limit. Workers are unpinned,
   matching ONNX Runtime, whose ``intra_op_num_threads`` receives the same value.

``--onnxruntime``
   Also measures ONNX Runtime with the same number of threads and reports its
   latency and the speedup of onnx-light-cpu over ONNX Runtime. The
   onnx-light-cpu phase completes before the ONNX Runtime session is created,
   so idle ONNX Runtime workers cannot interfere with its measurements.

``--pr [NUMBER_OR_URL]``
   Adds the aggregated Markdown table as a pull request comment using GitHub
   CLI. If the number or URL is omitted, GitHub CLI selects the pull request
   associated with the current branch. When neither ``--tests`` nor ``--dtypes``
   is given, this pull request is also used to infer the modified operator and
   data type.

``--from-pr [NUMBER_OR_URL]``
   Inspects the pull request's changed kernel files and diff to infer the
   operator and data type without posting results. Explicit ``--tests`` and
   ``--dtypes`` filters override this inference.

The Linux ``onnx-light main`` job in the ``ci-core`` workflow invokes this
command after its existing build and tests when a pull request modifies kernel
implementation or backend benchmark case files. A separate report job updates
the latest benchmark comment instead of adding a new comment after every push.
The comment displays speedup with two decimal places and input shapes without
input names: for example, ``2x3x4,5x6``. A scalar is written as ``scalar``;
multiple input datasets are separated by ``; ``. This compact presentation also
applies to ``--pr`` and ``--pr-markdown``. The workbook and full Markdown report
retain the original precision and named input shapes.

``-o``, ``--output``
   Output workbook path. It must end in ``.xlsx`` and defaults to
   ``onnx_light_cpu_benchmark.xlsx``. Parent directories are created when
   needed.

The ``raw`` sheet contains each measured duration. The ``aggregated`` sheet
contains the requested repeat, warm-up, thread count, input shapes, maximum
repeat time, sample count, mean, standard deviation, minimum, 10th percentile,
median, 90th percentile, and maximum latency for every selected case. When
``--onnxruntime`` is enabled, it also contains ONNX Runtime latency and speedup.

Both sheets and the full Markdown report include ``cpu_kernel_paths``, the
kernel names and implementation paths recorded during the untimed verification
run. For example, ``Cast.float32_to_float16.f16c`` identifies the F16C fast
path; ``Cast.scalar``, ``Cast.copy``, ``Cast.empty`` and ``Cast.builtin``
identify the other Cast paths. SIMD paths still use the scalar codec for
tails and, for float16-to-float32 F16C, vectors containing NaNs. Recording is
disabled while timing, and these diagnostics always describe onnx-light-cpu,
not ONNX Runtime. See :ref:`l-cast-simd-measurements` for a measured example.
