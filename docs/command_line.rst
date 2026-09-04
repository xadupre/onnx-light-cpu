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
   Number of onnx-light-cpu worker threads. The default is the number of CPUs
   available to the process. Workers are unpinned, matching ONNX Runtime when
   ``intra_op_num_threads`` is set explicitly.

``--onnxruntime``
   Also measures ONNX Runtime with the same number of threads and reports its
   latency and the speedup of onnx-light-cpu over ONNX Runtime.

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

``-o``, ``--output``
   Output workbook path. It must end in ``.xlsx`` and defaults to
   ``onnx_light_cpu_benchmark.xlsx``. Parent directories are created when
   needed.

The ``raw`` sheet contains each measured duration. The ``aggregated`` sheet
contains the requested repeat, warm-up, thread count, input shapes, maximum
repeat time, sample count, mean, standard deviation, minimum, 10th percentile,
median, 90th percentile, and maximum latency for every selected case. When
``--onnxruntime`` is enabled, it also contains ONNX Runtime latency and speedup.
