Runtime Execution Controls Roadmap
==================================

:Date: 2026-08

**complete**

Objective
---------

The objective is to expose a truthful, typed, inspectable CPU execution policy
covering thread count, spin-before-park, affinity, topology selection, nesting,
and kernel-specific limits. The same resolved policy must describe the workers
that actually execute a registered ``onnx-light-cpu`` kernel.

This requires coordinated changes in both repositories:

``onnx-light``
    Own the per-session public execution policy, session pool, Python API,
    calibration execution descriptor, and runtime diagnostics.

``onnx-light-cpu``
    Own topology and ISA-aware defaults for standalone kernels, consume the
    session policy when registered with ``onnx-light``, and avoid creating or
    nesting a second pool in that path.

This roadmap supplies the execution infrastructure required by
:doc:`processor-aware elementwise kernel tuning
<2026_08_elementwise_kernel_tuning>`. Runtime PR01 through Runtime PR04 should
land before that roadmap's tuning schemas and calibration callbacks depend on
session-owned parallel execution.

Resolved ownership
------------------

``onnx-light-cpu`` no longer owns a process-wide pool. Registered kernels
receive the exact ``CpuExecutor`` leased by their ``onnx-light`` session, and
standalone entry points execute synchronously on the calling thread. The CPU
library therefore has no independent thread count, spin budget, affinity
assignment, or scheduler lifecycle that can disagree with runtime diagnostics.

Public policy model in onnx-light
---------------------------------

Add a typed ``CpuExecutionPolicy`` (name illustrative) to ``onnx-light`` and
carry it through ``RuntimeSessionOptions`` and ``ReferenceEvaluator``:

``num_threads``
    ``0`` selects the topology-derived default; ``1`` is serial; values above
    one request an explicit participant count including the caller.

``spin_policy``
    An enum such as ``adaptive``, ``fixed_iterations``, ``fixed_duration``, and
    ``park_immediately``. A duration is more portable than an iteration count
    across microarchitectures.

``spin_budget``
    Validated duration or iteration count according to ``spin_policy``.

``affinity_policy``
    ``none``, ``physical_cores``, ``performance_cores``,
    ``physical_then_smt``, or ``explicit``.

``cpu_set``
    Optional explicit logical-processor identifiers, including processor group
    on Windows. Validate against the process-visible CPU set.

``idle_policy``
    Whether workers retain affinity while parked and whether the pool may
    release workers after a long idle period.

``allow_nested_parallelism``
    Default ``false``. External-region guards and session workers must make the
    effective behavior observable.

``maximum_threads_per_kernel``
    A prepared kernel may lower, but never exceed, the session participant
    count. Processor-aware tuning profiles can resolve this value per kernel
    and dtype.

Resolve the policy once during session preparation. Store both the request and
an immutable ``ResolvedCpuExecutionPolicy`` containing effective threads,
selected logical processors, physical-core identities, P/E classification,
SMT use, spin policy, and all fallback diagnostics.

Pool ownership
--------------

``onnx-light`` should own one pool per distinct resolved session policy, either
directly per session or through a safely shared pool registry keyed by the
complete policy. Sharing by thread count alone is incorrect because affinity
and spin behavior are observable.

Registered ``onnx-light-cpu`` adapters should receive a parallel-range executor
and effective execution descriptor from the session. They execute serial SIMD
range functions inside that executor and must not wake the private
``onnx-light-cpu`` pool. Standalone kernel entry points retain a standalone
pool configured by the CPU library policy.

Nested calls from session workers, application-owned pools, OpenMP, and BLAS
must fall back to serial execution unless an explicit composition policy proves
that additional workers are safe.

Affinity contract
-----------------

Topology detection must respect the process-visible CPU set, containers, Linux
cpusets, Windows processor groups, and macOS limitations. Defaults use one
logical processor per physical core and prefer performance cores, without
assuming adjacent processor identifiers are siblings.

Explicit affinity must:

* reject unavailable processors instead of silently widening the set;
* distinguish an unsupported platform from an empty request;
* preserve the calling thread unless the policy explicitly pins it;
* report every failed worker pin;
* avoid assigning a worker to the calling thread's physical core when enough
  other cores exist;
* define behavior when the process CPU set changes after session creation.

The no-affinity policy remains available for embedding applications that own
placement externally.

Spinning contract
-----------------

Spin applies in two places: workers waiting for a new generation and the caller
waiting for workers to complete. Expose them separately only if measurements
show different optimal policies; otherwise one policy keeps the API smaller.

The adaptive default should consider workload cadence, participant count,
oversubscription, power mode, and whether the process is running in a shared or
latency-sensitive environment. It must remain bounded and eventually park.

Diagnostics must report cumulative spins, parks, wakeups, dispatches, caller
waits, and worker-active time without adding overhead when disabled. These
counters are essential to distinguish arithmetic cost from scheduler wakeup
latency.

Standalone compatibility
------------------------

The former ``ONNX_LIGHT_CPU_NUM_THREADS``, ``ONNX_LIGHT_CPU_SPIN_COUNT``, and
``ONNX_LIGHT_CPU_MAX_THREADS`` controls are removed with the private scheduler.
Standalone callers that need parallelism partition their inputs with their own
executor; registered kernels use the typed ``onnx-light`` session policy.

Python and C++ APIs
-------------------

The C++ API should expose request, resolution, and inspection types without
requiring Python or environment variables. Python should provide equivalent
keyword arguments and a read-only resolved-policy object:

.. code-block:: python

    session = ReferenceEvaluator(
        model,
        cpu_execution={
            "num_threads": 0,
            "spin_policy": "adaptive",
            "affinity_policy": "physical_cores",
        },
    )
    print(session.cpu_execution_policy)

Exact spelling belongs in the ``onnx-light`` API design. Unknown keys, invalid
processor identifiers, negative budgets, and impossible combinations must
raise explicit errors.

Tuning and cache identity
-------------------------

Kernel calibration must use the session's resolved executor and descriptor.
Cache compatibility must include every execution property that can change the
winner: at minimum effective threads, affinity class, SMT use, and stable spin
policy class. Raw CPU identifiers and transient scheduler counters do not
belong in persistent keys.

A calibration request for a thread count or affinity that the active executor
does not use must fail before measurement. A kernel-specific maximum participant
count is part of the calibrated parameter set, not a second hidden pool limit.

Benchmark and validation plan
-----------------------------

Follow :doc:`the benchmark methodology <../../design/benchmark_methodology>`.
Cover:

* serial, 2, 4, physical-core, and logical-thread counts;
* no affinity, physical-core affinity, P-core preference, and explicit CPU sets;
* immediate park, fixed spin budgets, and adaptive spin;
* isolated calls, bursty calls, and sustained throughput;
* tiny elementwise work, memory-bound large tensors, GEMM, and nested calls;
* Linux cpusets, Windows processor groups, hybrid CPUs, SMT, and unsupported
  affinity platforms;
* several sessions with different policies in one process;
* registered-kernel and standalone-kernel execution.

Correctness tests must include pool destruction, concurrent session calls,
exceptions during preparation, fork/process boundaries where supported, and
thread-sanitizer runs. Performance gates report latency, throughput, dispersion,
CPU time, wakeups, and power where available.

Pull-request sequence
---------------------

.. list-table::
   :header-rows: 1
   :widths: 9 24 43 14 10

   * - PR
     - Repository and scope
     - Merge criterion
     - Depends on
     - Status
   * - Runtime PR01
     - ``onnx-light``: typed policy and resolution.
     - C++ request/resolved types validate threads, spin, affinity, and CPU sets;
       topology fallbacks and diagnostics are deterministic and fully tested.
     - None
     - Completed
   * - Runtime PR02
     - ``onnx-light``: policy-owned executor.
     - Sessions with different policies execute concurrently without sharing an
       incompatible pool; nesting is serial and lifecycle/thread-sanitizer tests
       pass.
     - PR01
     - Completed
   * - Runtime PR03
     - ``onnx-light``: Python and inspection API.
     - ``ReferenceEvaluator`` accepts typed execution options and exposes the
       immutable resolved policy and optional zero-overhead-disabled counters.
     - PR02
     - Completed
   * - Runtime PR04
     - ``onnx-light-cpu``: executor adapter.
     - Registered kernels use the session executor; standalone entry points
       remain serial.
     - PR02
     - Completed (`Pool PR05
       <https://github.com/xadupre/onnx-light-cpu/pull/270>`_)
   * - Runtime PR05
     - ``onnx-light-cpu``: standalone policy and environment compatibility.
     - Superseded by Runtime PR08: standalone execution is serial and the
       private scheduler controls are removed.
     - PR01
     - Superseded
   * - Runtime PR06
     - Both: tuning identity and kernel participant limits.
     - Calibration uses the actual executor; cache identity is truthful; per-
       kernel maximum threads cannot exceed the session policy.
     - PR03, PR04
     - Completed (`Pool PR06
       <https://github.com/xadupre/onnx-light/pull/4594>`_)
   * - Runtime PR07
     - Both: private-scheduler removal and compatibility gate.
     - The complete policy matrix passes correctness tests; default latency and
       throughput do not regress; ``onnx-light-cpu`` owns no worker pool after
       `#314 <https://github.com/xadupre/onnx-light-cpu/pull/314>`_ removes the
       adapter and registered kernels use only the session executor.
     - PR05, PR06
     - Merged (`Pool PR07
       <https://github.com/xadupre/onnx-light-cpu/pull/271>`_)

Runtime PR07 completed the roadmap.
