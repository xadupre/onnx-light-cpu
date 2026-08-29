Processor performance profile
-----------------------------

.. py:function:: benchmark_processor_performance(thread_policies=("single", "physical"), repeats=7, minimum_duration_ms=20.0, memory_budget_bytes=512 * 1024 * 1024, include_latency=True, explicit_single_affinity=None) -> ProcessorPerformanceProfile

   Measures an immutable, versioned :class:`ProcessorPerformanceProfile`. This
   is an explicit, expensive action and is never run during import, session
   creation, calibration lookup, or inference.

   Every option is validated before allocation or timing; invalid
   ``thread_policies``, ``repeats``, ``minimum_duration_ms``,
   ``memory_budget_bytes``, or ``explicit_single_affinity`` raises
   ``ValueError``. A memory level or compute element type that cannot be
   measured truthfully is absent rather than represented by zero or a
   fabricated value; see :attr:`ProcessorPerformanceProfile.warnings`.

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

   Immutable, versioned processor performance profile.

   .. py:attribute:: metadata
      :type: ProcessorProfileMetadata

   .. py:attribute:: topology
      :type: ProcessorProfileTopology

   .. py:attribute:: memory
      :type: dict[str, dict[str, MemoryLevelMeasurement]]

      Keyed by memory level, then by thread policy. Missing entries could not
      be measured truthfully; see ``warnings``.

   .. py:attribute:: compute
      :type: dict[str, dict[str, ComputeMeasurement]]

      Keyed by element type, then by thread policy. Missing entries have no
      compiled and runtime-detected native arithmetic path.

   .. py:attribute:: roofline
      :type: dict[str, dict[str, dict[str, RooflineMeasurement]]]

      Keyed by element type, thread policy, then memory level.

   .. py:attribute:: warnings
      :type: tuple[str, ...]

      Unavailable, inferred, noisy, unpinned, or memory-budget-limited
      conditions from assembling the profile.

.. py:class:: ProcessorProfileMetadata

   Schema version, timestamp, platform/compiler identity, resolved options, and
   shared timer identity for a profile run.

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

   Immutable echo of the measured options.

.. py:class:: ProcessorProfileTopology

   Process-visible logical/physical topology and cache descriptors.

.. py:class:: CacheDescriptor

   One reusable cache level descriptor.

.. py:class:: BandwidthMeasurement

   One available bandwidth measurement.

.. py:class:: LatencyMeasurement

   One available dependent-load pointer-chase latency measurement.

.. py:class:: MemoryLevelMeasurement

   Measurements for one memory level and thread policy.

.. py:class:: ComputeMeasurement

   One available register-resident arithmetic throughput measurement.

.. py:class:: RooflineMeasurement

   One derived Roofline crossover point.

.. py:class:: ExplicitAffinity

   One explicit logical-processor affinity ``(group, index)``.
