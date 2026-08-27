API
===

C++ API
-------

The public C++ API is declared throughout ``onnx_light_cpu/impl`` and
``onnx_light_cpu/kernels``. The shared SIMD
dispatch primitives (``SimdLevel`` and ``DetectSimdLevel``) live in the
``onnx_light_cpu/impl/simd_level.h`` header, which both kernel families
include. Every kernel dispatches at runtime to the best available SIMD path.

The signatures below are extracted from the header comments by Doxygen (see
``docs/Doxyfile``) and rendered through the Breathe extension, so they always
reflect the current state of the project.

.. doxygenenum:: onnx_light_cpu::SimdLevel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::DetectSimdLevel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::AbsFloat32
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::AbsFloat64
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::AbsFloat16
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::AbsInt8
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::AbsInt32
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::AbsInt64
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::ExpFloat32
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::ExpFloat64
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::ExpFloat16
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::LogFloat32
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::LogFloat64
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::LogFloat16
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::GemmFloat32
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::GemmFloat64
   :project: onnx_light_cpu

The ``onnx_light_cpu/impl/logical/logical_kernels.h`` header declares the
logical family, currently the elementwise logical negation used by the ``Not``
operator (ONNX ``bool`` tensors are stored one byte per element, so the buffers
are the raw ``uint8_t`` byte patterns):

.. doxygenfunction:: onnx_light_cpu::NotBool
   :project: onnx_light_cpu

Execution ownership
~~~~~~~~~~~~~~~~~~~

``onnx-light-cpu`` owns SIMD computation, not thread scheduling. Direct C++
kernel calls execute synchronously on the calling thread and do not create
workers. When the kernels are registered with ``onnx-light``, the registration
adapter injects the session ``CpuExecutor``. Large ranges are then split into
disjoint SIMD-aligned blocks and dispatched by that executor.

Consequently, participant count, affinity, spin policy, nesting, lifecycle, and
diagnostics all come from the ``onnx-light`` session policy. There are no
``ONNX_LIGHT_CPU_NUM_THREADS`` or ``ONNX_LIGHT_CPU_SPIN_COUNT`` settings and no
second pool that can oversubscribe the runtime.


onnx-light kernel class
~~~~~~~~~~~~~~~~~~~~~~~~

When onnx-light-cpu is built with ``-DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON`` (which
requires the `onnx-light <https://github.com/xadupre/onnx-light>`_ C++ package),
an additional library ``lib_onnx_light_cpu_kernels`` is produced. Its complete
runtime inventory is exposed by :cpp:func:`CollectRegisteredKernels` and by the
generated :ref:`ByOp catalogue <l-cpu-by-op>`.

.. doxygenclass:: onnx_light_cpu::AbsKernel
   :project: onnx_light_cpu
   :members:

.. doxygenclass:: onnx_light_cpu::ExpKernel
   :project: onnx_light_cpu
   :members:

.. doxygenclass:: onnx_light_cpu::LogKernel
   :project: onnx_light_cpu
   :members:

.. doxygenclass:: onnx_light_cpu::GemmKernel
   :project: onnx_light_cpu
   :members:

.. doxygenclass:: onnx_light_cpu::NotKernel
   :project: onnx_light_cpu
   :members:

.. doxygenfunction:: onnx_light_cpu::RegisterAbsKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterExpKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterLogKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterGemmKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterNotKernel
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterAllKernels
   :project: onnx_light_cpu

The custom operator support inventory and its implementations are also public:

.. doxygenfunction:: onnx_light_cpu::CollectOperatorSupport
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::ComputeShapeCDist
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::ComputeShapeBiasGelu
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::ComputePeakMemoryCDist
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::ComputePeakMemoryBiasGelu
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions
   :project: onnx_light_cpu

.. doxygenfunction:: onnx_light_cpu::RegisterCustomOperatorGradients
   :project: onnx_light_cpu

.. doxygenclass:: onnx_light_cpu::BiasGeluFusionPattern
   :project: onnx_light_cpu

.. doxygenclass:: onnx_light_cpu::CDistFusionPattern
   :project: onnx_light_cpu

Python API
----------

.. py:module:: onnx_light_cpu.onnx_py._cpukernels

.. py:function:: detect_simd_level() -> int

   Returns the detected SIMD level: ``0=None``, ``1=SSE2``, ``2=AVX``,
   ``3=AVX2``, ``4=AVX512``.

.. py:function:: has_cpu_kernels() -> bool

   Returns ``True`` when the CPU kernel extension is available.

The compiled kernels themselves (``Abs``, ``Exp``, ``Log``, ``Gemm``, ``Not``)
are not exposed as numpy-like Python functions; they are reachable through
onnx-light's runtime after registration (see :func:`register_all_kernels` and
:func:`onnx_light_cpu.register_kernels`).

.. py:function:: benchmark_processor_performance_raw(thread_policies, repeats, minimum_duration_ms, memory_budget_bytes, include_latency, explicit_single_affinity=None)

   Runs the versioned processor performance profile (effective memory
   bandwidth/latency plus register-resident compute throughput) and returns it
   as a plain nested-tuple structure. Raises ``ValueError`` before allocating
   or timing anything when an option is invalid. This binding links neither
   onnx-light nor any kernel dispatch table. It is wrapped by
   :func:`onnx_light_cpu.benchmark_processor_performance`, which converts the
   raw tuples into immutable, documented result objects.

.. py:module:: onnx_light_cpu.onnx_py._cpuregister

.. py:function:: register_all_kernels() -> None

   Registers every onnx-light-cpu kernel class (``Abs``, ``Exp``, ``Log``,
   ``Gemm`` and ``Not``) into onnx-light's C++ ``KernelDispatchTable`` for the
   CPU device, replacing the corresponding built-in entries for the default
   ONNX domain.
   This is the Python binding for the C++ :cpp:func:`RegisterAllKernels`
   function and is only available in builds compiled with the onnx-light
   integration enabled (``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``).

.. py:function:: registered_kernel_names() -> list[tuple[str, str]]

   Returns the ``(op_type, kernel name)`` pairs of every onnx-light-cpu kernel,
   for example ``("Abs", "onnx_light_cpu::Abs")``. The kernel name is the
   library-qualified name each kernel records when it runs, so callers can check
   that the accelerated kernels are the ones actually used.

.. py:function:: registered_kernels() -> list[tuple[str, str, str, str, list[str], int | None, int | None]]

   Returns one ``(domain, op_type, device, kernel_name, types, since_version,
   until_version)`` tuple per onnx-light-cpu kernel registration, collected
   from the C++ :cpp:func:`CollectRegisteredKernels` inventory without
   installing, replacing, or executing any kernel. Sorted by ``(domain,
   op_type, device, kernel_name)``. This is the raw binding wrapped by
   :func:`onnx_light_cpu.registered_kernels`, which converts each tuple into an
   immutable :class:`onnx_light_cpu.RegisteredKernel` record.

.. py:function:: operator_support() -> list[tuple[str, str, str, str, list[str], bool]]

   Returns the read-only custom operator support inventory: domain, operator,
   shape-inference function, peak-memory function, fusion patterns, and
   gradient availability. This is the raw binding wrapped by
   :func:`onnx_light_cpu.operator_support`.

.. py:function:: microsoft_op_schemas(op_type="", init_doc=True)

   Returns the ``LightOpSchema`` history supplied for ``com.microsoft``
   operators.

.. py:function:: register_custom_operator_support() -> None

   Registers the ``com.microsoft`` shape-inference, peak-memory, and fusion
   pattern implementations.

.. py:function:: register_custom_gradients(registry) -> None

   Registers the ``com.microsoft`` gradient implementations in ``registry``.

.. py:function:: used_kernel_names() -> list[str]

   Returns the library-qualified names of the onnx-light-cpu kernels that ran
   since the last :func:`clear_used_kernel_names` call, in invocation order.

.. py:function:: clear_used_kernel_names() -> None

   Clears the record of onnx-light-cpu kernels that have run.

.. py:function:: set_kernel_usage_recording(enabled) -> None

   Enables or disables per-invocation kernel usage recording.


Registering kernels with onnx-light
-----------------------------------

.. py:module:: onnx_light_cpu

.. py:function:: register_kernels(sess=None)

   Registers the onnx-light-cpu kernels into onnx-light's shared C++
   ``KernelDispatchTable`` for the CPU device by calling
   :func:`register_all_kernels`. After this call, every ``Abs``, ``Exp``,
   ``Log``, ``Gemm`` and ``Not`` node executed by onnx-light's runtime (and
   therefore any model run through ``ReferenceEvaluator``) dispatches to the
   optimized onnx-light-cpu kernel instead of the built-in one. It also
   installs the ``com.microsoft`` ``CDist`` and ``BiasGelu`` kernels, their
   symbolic shape and peak-memory functions, and their fusion patterns. The
   registration is global, so ``sess`` is optional and only returned unchanged
   so calls can be chained.

   .. code-block:: python

      import numpy as np
      from onnx_light.onnx.reference import ReferenceEvaluator
      from onnx_light_cpu import register_kernels

      register_kernels()
      sess = ReferenceEvaluator(model)
      (y,) = sess.run(None, {"x": np.array([-1.0, 2.0], dtype=np.float32)})

.. py:function:: custom_op_schemas(op_type="", init_doc=True)

   Returns the :class:`LightOpSchema` records for the supported
   ``com.microsoft`` operators.

.. py:function:: operator_schema_lookup(op_type)

   Returns the standard ONNX schemas and this package's custom schemas. Pass it
   as ``GraphBuilder(..., schema_lookup=operator_schema_lookup)``.

.. py:class:: OperatorSupport

   Immutable ``NamedTuple`` describing shape inference, peak memory, fusion
   patterns, and gradient availability for one custom operator.

.. py:function:: operator_support() -> tuple[OperatorSupport, ...]

   Returns the read-only custom operator support inventory without registering
   or executing any implementation.

.. py:function:: register_operator_support() -> None

   Registers custom shape-inference, peak-memory, and fusion-pattern support.

.. py:function:: register_custom_gradients(registry=None)

   Adds the ``CDist`` and ``BiasGelu`` backward rules to an independent
   ``GradRegistry`` and returns that registry.

.. py:function:: registered_kernel_names() -> dict[str, str]

   Returns a ``{op_type: kernel name}`` dictionary mapping each ONNX ``op_type``
   onnx-light-cpu overrides to the library-qualified name of the accelerated
   kernel installed for it (for example ``{"Abs": "onnx_light_cpu::Abs"}``). Use
   it to confirm the accelerated kernels — rather than onnx-light's built-in
   ones — are registered. Derived from :func:`registered_kernels` instead of
   maintaining a second operator list.

.. py:class:: RegisteredKernel

   Immutable ``NamedTuple`` record describing one onnx-light-cpu kernel
   registration, as returned by :func:`registered_kernels`.

   .. py:attribute:: domain
      :type: str

      ONNX operator domain, e.g. ``"ai.onnx"``.

   .. py:attribute:: op_type
      :type: str

      ONNX operator type name, e.g. ``"Abs"``.

   .. py:attribute:: device
      :type: str

      Device the kernel runs on, e.g. ``"CPU"``.

   .. py:attribute:: kernel_name
      :type: str

      Library-qualified C++ kernel class name, e.g. ``"onnx_light_cpu::Abs"``.

   .. py:attribute:: types
      :type: tuple[str, ...]

      Element type names (``TensorProto::DataType`` names, e.g. ``"FLOAT"``)
      the kernel accepts for its primary tensor operands.

   .. py:attribute:: since_version
      :type: int | None

      Inclusive opset lower bound, or ``None`` when there is no lower bound.

   .. py:attribute:: until_version
      :type: int | None

      Inclusive opset upper bound, or ``None`` when there is no upper bound.

.. py:function:: registered_kernels() -> tuple[RegisteredKernel, ...]

   Returns one immutable :class:`RegisteredKernel` record per onnx-light-cpu
   kernel registration, collected from the C++
   :cpp:func:`CollectRegisteredKernels` inventory without installing,
   replacing, or executing any kernel. Records are sorted deterministically by
   ``(domain, op_type, device, kernel_name)``, and
   :func:`registered_kernel_names` derives its ``{op_type: kernel name}``
   mapping from this same inventory. Wraps
   :func:`onnx_light_cpu.onnx_py._cpuregister.registered_kernels`.

.. py:function:: used_kernel_names() -> list[str]

   Returns, in invocation order, the library-qualified names of the
   onnx-light-cpu kernels that have run since the last
   :func:`clear_used_kernel_names` call. After running a model through a
   ``ReferenceEvaluator`` this reports which accelerated kernels the runtime
   actually dispatched to. Wraps
   :func:`onnx_light_cpu.onnx_py._cpuregister.used_kernel_names`.

.. py:function:: clear_used_kernel_names() -> None

   Clears the record of onnx-light-cpu kernels that have run, so a subsequent
   :func:`used_kernel_names` call only reports the kernels used after this call.
   Wraps :func:`onnx_light_cpu.onnx_py._cpuregister.clear_used_kernel_names`.

.. py:function:: set_kernel_usage_recording(enabled) -> None

   Enables or disables per-invocation kernel usage recording.

.. py:function:: register_backend_test_cases() -> None

   Registers the onnx-light-cpu ``test_cpu_*`` backend test cases into
   onnx-light's shared C++ backend test registry, so they are returned by
   :func:`onnx_light.onnx.backend.collect_test_cases` alongside onnx-light's own
   cases and can be driven through the regular ``ReferenceEvaluator`` API. The
   registration is process-wide and idempotent, and only usable when
   :func:`has_backend_test_cases` reports ``True``.

.. py:function:: has_backend_test_cases() -> bool

   Returns whether the ``register_backend_test_cases`` binding is available. It
   is only built when the ``_cpuregister`` extension links onnx-light's backend
   test registry (``lib_onnx_backend_test``). When ``False``,
   :func:`register_backend_test_cases` is not usable.

Processor performance profile
------------------------------

.. py:function:: benchmark_processor_performance(thread_policies=("single", "physical"), repeats=7, minimum_duration_ms=20.0, memory_budget_bytes=512 * 1024 * 1024, include_latency=True, explicit_single_affinity=None) -> ProcessorPerformanceProfile

   Measures and returns one immutable, versioned :class:`ProcessorPerformanceProfile`:
   effective L1/L2/L3/RAM bandwidth and dependent-load latency (see
   ``onnx_light_cpu/impl/memory_traffic_profile.h``) together with
   register-resident FP32/FP64/FP16/BF16/INT8 arithmetic throughput (see
   ``onnx_light_cpu/impl/compute_arithmetic_profile.h``). This is an explicit,
   expensive action: it is never called during import, session creation,
   calibration lookup, or inference.

   Every option is validated before any allocation or timing happens; invalid
   ``thread_policies``, ``repeats``, ``minimum_duration_ms``,
   ``memory_budget_bytes``, or ``explicit_single_affinity`` raise
   ``ValueError``. A memory level or compute element type that cannot be
   measured truthfully (for example because the host has no matching cache
   level, or no compiled and detected native low-precision path) is absent
   from the result rather than represented by a zero or fabricated value, and
   is explained in :attr:`ProcessorPerformanceProfile.warnings`.

   .. code-block:: python

      from onnx_light_cpu import benchmark_processor_performance

      profile = benchmark_processor_performance(
          thread_policies=("single", "physical"),
          repeats=7,
          minimum_duration_ms=50,
          memory_budget_bytes=512 * 1024 * 1024,
          include_latency=True,
      )

      print(profile.memory["L1"]["single"].read.median_gbps)
      print(profile.memory["RAM"]["physical"].copy.median_gbps)
      print(profile.compute["float32"]["physical"].median_gops)

.. py:class:: ProcessorPerformanceProfile

   Immutable, versioned processor performance profile returned by
   :func:`benchmark_processor_performance`. ``to_dict()`` returns a
   deterministic, JSON-compatible serialization including
   :attr:`ProcessorProfileMetadata.schema_version`.

   .. py:attribute:: metadata
      :type: ProcessorProfileMetadata

   .. py:attribute:: topology
      :type: ProcessorProfileTopology

   .. py:attribute:: memory
      :type: dict[str, dict[str, MemoryLevelMeasurement]]

      Keyed by memory level (``"L1"``, ``"L2"``, ``"L3"``, ``"RAM"``), then by
      thread policy (``"single"``, ``"physical"``). A missing level or policy
      means it could not be measured truthfully; see ``warnings``.

   .. py:attribute:: compute
      :type: dict[str, dict[str, ComputeMeasurement]]

      Keyed by element type (``"float32"``, ``"float64"``, ``"float16"``,
      ``"bfloat16"``, ``"int8"``), then by thread policy. A missing element
      type means no compiled and runtime-detected native arithmetic path
      exists for it; see ``warnings``.

   .. py:attribute:: roofline
      :type: dict[str, dict[str, dict[str, RooflineMeasurement]]]

      Keyed by element type, then thread policy, then memory level. Each
      entry is the arithmetic-intensity crossover derived from that
      policy/element type's compute throughput and that policy/level's read
      bandwidth.

   .. py:attribute:: warnings
      :type: tuple[str, ...]

      Explicit unavailable, inferred, noisy, unpinned, or memory-budget
      limited conditions encountered while assembling this profile.

.. py:class:: ProcessorProfileMetadata

   Schema version, timestamp, platform/compiler identity, the resolved
   options, and the shared timer identity for one profile run.

   .. py:attribute:: schema_version
      :type: int

   .. py:attribute:: unix_timestamp_ns
      :type: int

   .. py:attribute:: platform
      :type: str

   .. py:attribute:: compiler
      :type: str

   .. py:attribute:: timer_name
      :type: str

   .. py:attribute:: options
      :type: ProcessorProfileOptionsEcho

   .. py:attribute:: diagnostics
      :type: tuple[str, ...]

.. py:class:: ProcessorProfileOptionsEcho

   Immutable echo of the options a profile run was measured with.

.. py:class:: ProcessorProfileTopology

   Process-visible logical/physical topology and cache descriptors, reused
   from ``onnx_light_cpu::GetCpuTopology`` and
   ``onnx_light_cpu::GetCpuCacheTopology``.

.. py:class:: CacheDescriptor

   One reusable cache level descriptor, mirroring
   ``onnx_light_cpu::CpuCacheDescriptor``.

.. py:class:: BandwidthMeasurement

   One available bandwidth measurement (read, write, copy, or
   read-modify-write) for one memory level and thread policy.

.. py:class:: LatencyMeasurement

   One available dependent-load pointer-chase latency measurement.

.. py:class:: MemoryLevelMeasurement

   One memory level's measurements for one thread policy. Each of ``read``,
   ``write``, ``copy``, ``read_modify_write``, and ``latency`` is ``None``
   exactly when the underlying engine reported it unavailable.

.. py:class:: ComputeMeasurement

   One available register-resident arithmetic throughput measurement.

.. py:class:: RooflineMeasurement

   One derived Roofline crossover point for one element type, thread policy,
   and memory level.

.. py:class:: ExplicitAffinity

   One explicit logical-processor affinity ``(group, index)``, used to pin
   the ``"single"`` thread policy's lone participant.
