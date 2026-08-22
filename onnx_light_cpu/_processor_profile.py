# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Versioned processor performance profile (Processor Profile PR04).

Exposes :func:`benchmark_processor_performance`, which measures effective
memory bandwidth/latency (see ``onnx_light_cpu/impl/memory_traffic_profile.h``)
and register-resident arithmetic throughput (see
``onnx_light_cpu/impl/compute_arithmetic_profile.h``) and returns one
immutable, versioned :class:`ProcessorPerformanceProfile`.

This module never runs a benchmark on import: the expensive measurement only
happens inside an explicit call to :func:`benchmark_processor_performance`.
See ``docs/next_steps/2026/2026_08_processor_performance_profile.rst`` for the
full measurement contract every result honors -- in particular, a memory
level or compute element type that could not be measured truthfully is
*absent* from the result and explained in :attr:`ProcessorPerformanceProfile.warnings`
instead of being represented by a zero or fabricated value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple

#: Schema version of the ``to_dict()`` shape returned by
#: :class:`ProcessorPerformanceProfile`. Mirrors
#: ``onnx_light_cpu::kProcessorPerformanceProfileSchemaVersion``.
SCHEMA_VERSION = 1


class ExplicitAffinity(NamedTuple):
    """One explicit logical-processor affinity (processor group, index)."""

    group: int
    index: int

    def to_dict(self) -> dict[str, int]:
        return {"group": self.group, "index": self.index}


@dataclass(frozen=True)
class ProcessorProfileOptionsEcho:
    """Immutable echo of the options a profile run was measured with."""

    thread_policies: tuple[str, ...]
    repeats: int
    minimum_duration_ms: float
    memory_budget_bytes: int
    include_latency: bool
    explicit_single_affinity: ExplicitAffinity | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_policies": list(self.thread_policies),
            "repeats": self.repeats,
            "minimum_duration_ms": self.minimum_duration_ms,
            "memory_budget_bytes": self.memory_budget_bytes,
            "include_latency": self.include_latency,
            "explicit_single_affinity": (
                self.explicit_single_affinity.to_dict()
                if self.explicit_single_affinity is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ProcessorProfileMetadata:
    """Schema version, timestamp, platform/compiler identity, and options."""

    schema_version: int
    unix_timestamp_ns: int
    platform: str
    compiler: str
    timer_name: str
    options: ProcessorProfileOptionsEcho
    diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "unix_timestamp_ns": self.unix_timestamp_ns,
            "platform": self.platform,
            "compiler": self.compiler,
            "timer_name": self.timer_name,
            "options": self.options.to_dict(),
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class CacheDescriptor:
    """One reusable cache level descriptor (mirrors ``CpuCacheDescriptor``)."""

    level: int
    kind: str
    size_bytes: int
    line_size_bytes: int
    sharing_thread_count: int
    confidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "line_size_bytes": self.line_size_bytes,
            "sharing_thread_count": self.sharing_thread_count,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ProcessorProfileTopology:
    """Process-visible logical/physical topology and cache descriptors."""

    logical_thread_count: int
    physical_core_count: int
    performance_core_count: int
    efficiency_core_count: int
    caches: tuple[CacheDescriptor, ...]
    cache_topology_detected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_thread_count": self.logical_thread_count,
            "physical_core_count": self.physical_core_count,
            "performance_core_count": self.performance_core_count,
            "efficiency_core_count": self.efficiency_core_count,
            "caches": [cache.to_dict() for cache in self.caches],
            "cache_topology_detected": self.cache_topology_detected,
        }


@dataclass(frozen=True)
class BandwidthMeasurement:
    """One available bandwidth measurement for one traffic mode."""

    working_set_bytes: int
    participant_count: int
    affinity_pinned: bool
    timer_name: str
    useful_bytes_per_pass_per_participant: int
    raw_gbps_samples: tuple[float, ...]
    median_gbps: float
    dispersion_gbps: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "working_set_bytes": self.working_set_bytes,
            "participant_count": self.participant_count,
            "affinity_pinned": self.affinity_pinned,
            "timer_name": self.timer_name,
            "useful_bytes_per_pass_per_participant": self.useful_bytes_per_pass_per_participant,
            "raw_gbps_samples": list(self.raw_gbps_samples),
            "median_gbps": self.median_gbps,
            "dispersion_gbps": self.dispersion_gbps,
        }


@dataclass(frozen=True)
class LatencyMeasurement:
    """One available dependent-load latency measurement."""

    working_set_bytes: int
    participant_count: int
    affinity_pinned: bool
    timer_name: str
    raw_ns_per_load_samples: tuple[float, ...]
    median_ns_per_load: float
    dispersion_ns_per_load: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "working_set_bytes": self.working_set_bytes,
            "participant_count": self.participant_count,
            "affinity_pinned": self.affinity_pinned,
            "timer_name": self.timer_name,
            "raw_ns_per_load_samples": list(self.raw_ns_per_load_samples),
            "median_ns_per_load": self.median_ns_per_load,
            "dispersion_ns_per_load": self.dispersion_ns_per_load,
        }


@dataclass(frozen=True)
class MemoryLevelMeasurement:
    """One memory level's measurements for one thread policy.

    Each field is ``None`` exactly when the underlying engine reported that
    traffic mode (or latency) unavailable; see
    :attr:`ProcessorPerformanceProfile.warnings` for why.
    """

    level: str
    policy: str
    read: BandwidthMeasurement | None
    write: BandwidthMeasurement | None
    copy: BandwidthMeasurement | None
    read_modify_write: BandwidthMeasurement | None
    latency: LatencyMeasurement | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "policy": self.policy,
            "read": self.read.to_dict() if self.read is not None else None,
            "write": self.write.to_dict() if self.write is not None else None,
            "copy": self.copy.to_dict() if self.copy is not None else None,
            "read_modify_write": (
                self.read_modify_write.to_dict() if self.read_modify_write is not None else None
            ),
            "latency": self.latency.to_dict() if self.latency is not None else None,
        }


@dataclass(frozen=True)
class ComputeMeasurement:
    """One available register-resident arithmetic throughput measurement."""

    element_type: str
    policy: str
    implementation_name: str
    participant_count: int
    affinity_pinned: bool
    timer_name: str
    operations_per_pass_per_participant: int
    dot_product_length: int
    raw_gops_samples: tuple[float, ...]
    median_gops: float
    dispersion_gops: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_type": self.element_type,
            "policy": self.policy,
            "implementation_name": self.implementation_name,
            "participant_count": self.participant_count,
            "affinity_pinned": self.affinity_pinned,
            "timer_name": self.timer_name,
            "operations_per_pass_per_participant": self.operations_per_pass_per_participant,
            "dot_product_length": self.dot_product_length,
            "raw_gops_samples": list(self.raw_gops_samples),
            "median_gops": self.median_gops,
            "dispersion_gops": self.dispersion_gops,
        }


@dataclass(frozen=True)
class RooflineMeasurement:
    """One derived Roofline crossover point for one element type/policy/level."""

    element_type: str
    policy: str
    level: str
    compute_gops: float
    memory_read_gbps: float
    arithmetic_intensity_crossover: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_type": self.element_type,
            "policy": self.policy,
            "level": self.level,
            "compute_gops": self.compute_gops,
            "memory_read_gbps": self.memory_read_gbps,
            "arithmetic_intensity_crossover": self.arithmetic_intensity_crossover,
        }


@dataclass(frozen=True)
class ProcessorPerformanceProfile:
    """Immutable, versioned processor performance profile.

    ``memory`` and ``compute`` are nested mappings keyed first by memory
    level / element type, then by thread policy, e.g.
    ``profile.memory["L1"]["single"].read.median_gbps`` or
    ``profile.compute["float32"]["physical"].median_gops``. ``roofline`` is
    keyed by element type, then policy, then memory level. A missing key at
    any level means that combination could not be measured truthfully; see
    ``warnings`` for the explanation.
    """

    metadata: ProcessorProfileMetadata
    topology: ProcessorProfileTopology
    memory: Mapping[str, Mapping[str, MemoryLevelMeasurement]]
    compute: Mapping[str, Mapping[str, ComputeMeasurement]]
    roofline: Mapping[str, Mapping[str, Mapping[str, RooflineMeasurement]]]
    warnings: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        """Deterministic, JSON-compatible serialization of this profile."""
        return {
            "metadata": self.metadata.to_dict(),
            "topology": self.topology.to_dict(),
            "memory": {
                level: {policy: entry.to_dict() for policy, entry in policies.items()}
                for level, policies in self.memory.items()
            },
            "compute": {
                element_type: {policy: entry.to_dict() for policy, entry in policies.items()}
                for element_type, policies in self.compute.items()
            },
            "roofline": {
                element_type: {
                    policy: {level: entry.to_dict() for level, entry in levels.items()}
                    for policy, levels in policies.items()
                }
                for element_type, policies in self.roofline.items()
            },
            "warnings": list(self.warnings),
        }


def _parse_explicit_single_affinity(
    explicit_single_affinity: tuple[int, int] | ExplicitAffinity | None,
) -> tuple[int, int] | None:
    if explicit_single_affinity is None:
        return None
    group, index = explicit_single_affinity
    return (int(group), int(index))


def benchmark_processor_performance(
    thread_policies: tuple[str, ...] = ("single", "physical"),
    repeats: int = 7,
    minimum_duration_ms: float = 20.0,
    memory_budget_bytes: int = 512 * 1024 * 1024,
    include_latency: bool = True,
    explicit_single_affinity: tuple[int, int] | ExplicitAffinity | None = None,
) -> ProcessorPerformanceProfile:
    """Measures and returns one versioned :class:`ProcessorPerformanceProfile`.

    This is an explicit, expensive action: it allocates memory, pins threads
    where supported, and runs the full memory bandwidth/latency and
    register-resident compute measurement engines. It is never called during
    import, session creation, calibration lookup, or inference.

    Parameters
    ----------
    thread_policies:
        Which participant policies to measure: any combination of
        ``"single"`` (one participant) and ``"physical"`` (one participant
        per physical core). Must not be empty.
    repeats:
        Number of recorded samples after warmup, for every measurement. Must
        be at least 1.
    minimum_duration_ms:
        Minimum wall-clock duration, per recorded sample, used to size the
        number of internal passes. Must be positive.
    memory_budget_bytes:
        Total memory budget available across every participant for the
        memory engine. Must be positive.
    include_latency:
        Whether to measure dependent-load latency in addition to bandwidth.
    explicit_single_affinity:
        Optional ``(group, index)`` logical-processor affinity used to pin
        the lone participant of the ``"single"`` policy. Must reference a
        logical processor visible to this process.

    Returns
    -------
    An immutable :class:`ProcessorPerformanceProfile`.

    Raises
    ------
    ValueError
        If any option is invalid. Validation happens before any allocation
        or timing.
    """
    from .onnx_py._cpukernels import (  # pyrefly: ignore[missing-import]
        benchmark_processor_performance_raw,
    )

    (
        raw_metadata,
        raw_topology,
        raw_memory,
        raw_compute,
        raw_roofline,
        raw_warnings,
    ) = benchmark_processor_performance_raw(
        list(thread_policies),
        int(repeats),
        float(minimum_duration_ms),
        int(memory_budget_bytes),
        bool(include_latency),
        _parse_explicit_single_affinity(explicit_single_affinity),
    )

    (
        schema_version,
        unix_timestamp_ns,
        platform,
        compiler,
        timer_name,
        raw_options,
        diagnostics,
    ) = raw_metadata
    (
        policies,
        options_repeats,
        options_min_duration_ms,
        options_memory_budget_bytes,
        options_include_latency,
        options_affinity,
    ) = raw_options
    metadata = ProcessorProfileMetadata(
        schema_version=schema_version,
        unix_timestamp_ns=unix_timestamp_ns,
        platform=platform,
        compiler=compiler,
        timer_name=timer_name,
        options=ProcessorProfileOptionsEcho(
            thread_policies=tuple(policies),
            repeats=options_repeats,
            minimum_duration_ms=options_min_duration_ms,
            memory_budget_bytes=options_memory_budget_bytes,
            include_latency=options_include_latency,
            explicit_single_affinity=(
                ExplicitAffinity(*options_affinity) if options_affinity is not None else None
            ),
        ),
        diagnostics=tuple(diagnostics),
    )

    (
        logical_thread_count,
        physical_core_count,
        performance_core_count,
        efficiency_core_count,
        raw_caches,
        cache_topology_detected,
    ) = raw_topology
    topology = ProcessorProfileTopology(
        logical_thread_count=logical_thread_count,
        physical_core_count=physical_core_count,
        performance_core_count=performance_core_count,
        efficiency_core_count=efficiency_core_count,
        caches=tuple(
            CacheDescriptor(
                level=level,
                kind=kind,
                size_bytes=size_bytes,
                line_size_bytes=line_size_bytes,
                sharing_thread_count=sharing_thread_count,
                confidence=confidence,
            )
            for level, kind, size_bytes, line_size_bytes, sharing_thread_count, confidence in (
                raw_caches
            )
        ),
        cache_topology_detected=cache_topology_detected,
    )

    def _bandwidth(raw: tuple | None) -> BandwidthMeasurement | None:
        if raw is None:
            return None
        (
            working_set_bytes,
            participant_count,
            affinity_pinned,
            timer,
            useful_bytes_per_pass_per_participant,
            raw_samples,
            median,
            dispersion,
        ) = raw
        return BandwidthMeasurement(
            working_set_bytes=working_set_bytes,
            participant_count=participant_count,
            affinity_pinned=affinity_pinned,
            timer_name=timer,
            useful_bytes_per_pass_per_participant=useful_bytes_per_pass_per_participant,
            raw_gbps_samples=tuple(raw_samples),
            median_gbps=median,
            dispersion_gbps=dispersion,
        )

    def _latency(raw: tuple | None) -> LatencyMeasurement | None:
        if raw is None:
            return None
        (
            working_set_bytes,
            participant_count,
            affinity_pinned,
            timer,
            raw_samples,
            median,
            dispersion,
        ) = raw
        return LatencyMeasurement(
            working_set_bytes=working_set_bytes,
            participant_count=participant_count,
            affinity_pinned=affinity_pinned,
            timer_name=timer,
            raw_ns_per_load_samples=tuple(raw_samples),
            median_ns_per_load=median,
            dispersion_ns_per_load=dispersion,
        )

    memory: dict[str, dict[str, MemoryLevelMeasurement]] = {}
    for level, policy, read, write, copy, read_modify_write, latency in raw_memory:
        entry = MemoryLevelMeasurement(
            level=level,
            policy=policy,
            read=_bandwidth(read),
            write=_bandwidth(write),
            copy=_bandwidth(copy),
            read_modify_write=_bandwidth(read_modify_write),
            latency=_latency(latency),
        )
        memory.setdefault(level, {})[policy] = entry

    compute: dict[str, dict[str, ComputeMeasurement]] = {}
    for (
        element_type,
        policy,
        implementation_name,
        participant_count,
        affinity_pinned,
        timer,
        operations_per_pass_per_participant,
        dot_product_length,
        raw_samples,
        median,
        dispersion,
    ) in raw_compute:
        entry = ComputeMeasurement(
            element_type=element_type,
            policy=policy,
            implementation_name=implementation_name,
            participant_count=participant_count,
            affinity_pinned=affinity_pinned,
            timer_name=timer,
            operations_per_pass_per_participant=operations_per_pass_per_participant,
            dot_product_length=dot_product_length,
            raw_gops_samples=tuple(raw_samples),
            median_gops=median,
            dispersion_gops=dispersion,
        )
        compute.setdefault(element_type, {})[policy] = entry

    roofline: dict[str, dict[str, dict[str, RooflineMeasurement]]] = {}
    for (
        element_type,
        policy,
        level,
        compute_gops,
        memory_read_gbps,
        arithmetic_intensity_crossover,
    ) in raw_roofline:
        entry = RooflineMeasurement(
            element_type=element_type,
            policy=policy,
            level=level,
            compute_gops=compute_gops,
            memory_read_gbps=memory_read_gbps,
            arithmetic_intensity_crossover=arithmetic_intensity_crossover,
        )
        roofline.setdefault(element_type, {}).setdefault(policy, {})[level] = entry

    return ProcessorPerformanceProfile(
        metadata=metadata,
        topology=topology,
        memory=MappingProxyType(
            {level: MappingProxyType(policies) for level, policies in memory.items()}
        ),
        compute=MappingProxyType(
            {
                element_type: MappingProxyType(policies)
                for element_type, policies in compute.items()
            }
        ),
        roofline=MappingProxyType(
            {
                element_type: MappingProxyType(
                    {policy: MappingProxyType(levels) for policy, levels in policies.items()}
                )
                for element_type, policies in roofline.items()
            }
        ),
        warnings=tuple(raw_warnings),
    )
