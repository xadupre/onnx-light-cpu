Processor performance profile
-----------------------------

.. py:function:: benchmark_processor_performance(thread_policies=("single", "physical"), repeats=7, minimum_duration_ms=20.0, memory_budget_bytes=512 * 1024 * 1024, include_latency=True, explicit_single_affinity=None) -> ProcessorPerformanceProfile

   Measures an immutable, versioned :class:`ProcessorPerformanceProfile`. This
   is an explicit, expensive action and is never run during import, session
   creation, calibration lookup, or inference.

.. py:class:: ProcessorPerformanceProfile

   Immutable, versioned processor performance profile.

.. py:class:: ProcessorProfileMetadata

   Schema version, timestamp, platform/compiler identity, resolved options, and
   shared timer identity for a profile run.

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
