# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`onnx_light_cpu.benchmark_processor_performance`.

The compiled ``_cpukernels`` extension (``benchmark_processor_performance_raw``)
links neither onnx-light nor any kernel dispatch table, so most tests here
exercise the real binding directly with small bounded options (mirroring the
bounded C++ tests in ``unittests/cc/test_processor_performance_profile.cc``).
A few tests patch the raw binding to deterministically exercise Python-side
wrapping (immutability, serialization, and partial-platform handling) without
depending on the host's actual topology.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest import mock

import pytest

import onnx_light_cpu
from onnx_light_cpu import (
    ExplicitAffinity,
    ProcessorPerformanceProfile,
    benchmark_processor_performance,
)


def _bounded_kwargs(**overrides):
    kwargs = {
        "thread_policies": ("single",),
        "repeats": 2,
        "minimum_duration_ms": 1.0,
        "memory_budget_bytes": 8 * 1024 * 1024,
        "include_latency": True,
    }
    kwargs.update(overrides)
    return kwargs


class TestImportTimeInactivity:
    def test_import_does_not_touch_the_extension(self):
        extension = ModuleType("onnx_light_cpu.onnx_py._cpukernels")
        extension.benchmark_processor_performance_raw = mock.Mock()

        with mock.patch.dict(sys.modules, {"onnx_light_cpu.onnx_py._cpukernels": extension}):
            import importlib

            importlib.reload(onnx_light_cpu)

        extension.benchmark_processor_performance_raw.assert_not_called()


class TestValidation:
    def test_empty_thread_policies_raises_before_measuring(self):
        with pytest.raises(ValueError):
            benchmark_processor_performance(**_bounded_kwargs(thread_policies=()))

    def test_zero_repeats_raises_before_measuring(self):
        with pytest.raises(ValueError):
            benchmark_processor_performance(**_bounded_kwargs(repeats=0))

    def test_non_positive_duration_raises_before_measuring(self):
        with pytest.raises(ValueError):
            benchmark_processor_performance(**_bounded_kwargs(minimum_duration_ms=0.0))

    def test_zero_memory_budget_raises_before_measuring(self):
        with pytest.raises(ValueError):
            benchmark_processor_performance(**_bounded_kwargs(memory_budget_bytes=0))

    def test_unknown_thread_policy_raises(self):
        with pytest.raises(ValueError):
            benchmark_processor_performance(**_bounded_kwargs(thread_policies=("both",)))

    def test_implausible_explicit_affinity_raises(self):
        with pytest.raises(ValueError):
            benchmark_processor_performance(
                **_bounded_kwargs(explicit_single_affinity=(0xFFFF, 0xFFFF))
            )


class TestBoundedSinglePolicyProfile:
    """Exercises the real extension with small bounded options."""

    def test_returns_coherent_profile(self):
        profile = benchmark_processor_performance(**_bounded_kwargs())

        assert isinstance(profile, ProcessorPerformanceProfile)
        assert profile.metadata.schema_version == 2
        assert profile.metadata.platform
        assert profile.metadata.compiler
        assert profile.topology.logical_thread_count >= 1
        assert profile.topology.physical_core_count >= 1

        # L1 must be measurable on any host running these tests.
        assert "L1" in profile.memory
        assert "single" in profile.memory["L1"]
        l1_single = profile.memory["L1"]["single"]
        assert l1_single.read is not None
        assert l1_single.read.median_gbps > 0.0
        assert l1_single.latency is not None
        assert l1_single.latency.median_ns_per_load > 0.0

        # Float32 compute is always available.
        assert "float32" in profile.compute
        assert "single" in profile.compute["float32"]
        assert profile.compute["float32"]["single"].median_gops > 0.0

    def test_physical_policy_reports_a_participant(self):
        profile = benchmark_processor_performance(
            **_bounded_kwargs(thread_policies=("physical",))
        )
        assert "physical" in profile.compute["float32"]
        scaling = profile.memory["L1"]["physical"].read_scaling
        assert scaling[0].participant_count == 1
        assert scaling[-1].participant_count == profile.topology.physical_core_count

    def test_include_latency_false_omits_every_latency_measurement(self):
        profile = benchmark_processor_performance(**_bounded_kwargs(include_latency=False))
        for policies in profile.memory.values():
            for entry in policies.values():
                assert entry.latency is None

    def test_impossible_ram_budget_is_absent_and_warned(self):
        profile = benchmark_processor_performance(**_bounded_kwargs(memory_budget_bytes=1))
        assert "RAM" not in profile.memory
        assert profile.warnings

    def test_missing_native_element_type_is_absent_not_zero(self):
        profile = benchmark_processor_performance(**_bounded_kwargs())
        for element_type in ("float16", "bfloat16", "int8"):
            if element_type not in profile.compute:
                # Absent, not represented by zero -- and explained.
                assert any(element_type in warning for warning in profile.warnings)


# ---------------------------------------------------------------------------
# Python-side wrapping: immutability, serialization, and partial platforms,
# exercised deterministically against a hand-built raw tuple so it does not
# depend on the host's actual topology or ISA.
# ---------------------------------------------------------------------------


def _raw_profile_stub():
    metadata = (
        2,
        1_700_000_000_000_000_000,
        "linux",
        "gcc",
        "std::chrono::steady_clock",
        (["single"], 2, 1.0, 8 * 1024 * 1024, True, None),
        [],
    )
    topology = (4, 2, 0, 0, [(1, "data", 32768, 64, 2, "detected")], True)
    bandwidth = (16384, 1, False, "std::chrono::steady_clock", 16384, [10.0, 11.0], 10.5, 0.5)
    latency = (16384, 1, False, "std::chrono::steady_clock", [1.0, 1.1], 1.05, 0.05)
    memory = [("L1", "single", bandwidth, bandwidth, bandwidth, bandwidth, latency, [bandwidth])]
    compute = [
        (
            "float32",
            "single",
            "AVX2",
            1,
            False,
            "std::chrono::steady_clock",
            8192,
            0,
            [100.0, 101.0],
            100.5,
            0.5,
        )
    ]
    roofline = [("float32", "single", "L1", 100.5, 10.5, 9.5714)]
    warnings = [
        "compute float16 (single): no compiled and runtime-detected native arithmetic path"
    ]
    return metadata, topology, memory, compute, roofline, warnings


class TestPythonWrapping:
    def _patch_binding(self, raw_profile):
        extension = ModuleType("onnx_light_cpu.onnx_py._cpukernels")
        extension.benchmark_processor_performance_raw = mock.Mock(return_value=raw_profile)
        return mock.patch.dict(sys.modules, {"onnx_light_cpu.onnx_py._cpukernels": extension})

    def test_wraps_raw_tuple_into_immutable_profile(self):
        with self._patch_binding(_raw_profile_stub()):
            profile = benchmark_processor_performance(
                thread_policies=("single",),
                repeats=2,
                minimum_duration_ms=1.0,
                memory_budget_bytes=8 * 1024 * 1024,
            )

        assert profile.memory["L1"]["single"].read.median_gbps == 10.5
        assert profile.memory["L1"]["single"].read_scaling[0].participant_count == 1
        assert profile.compute["float32"]["single"].median_gops == 100.5
        assert (
            profile.roofline["float32"]["single"]["L1"].arithmetic_intensity_crossover == 9.5714
        )

        with pytest.raises(AttributeError):
            profile.metadata.schema_version = 2  # frozen dataclass

        with pytest.raises(TypeError):
            profile.memory["L1"]["single"] = profile.memory["L1"]["single"]  # read-only mapping

        with pytest.raises(TypeError):
            profile.memory["L1"]["extra"] = profile.memory["L1"]["single"]  # read-only mapping

    def test_to_dict_is_json_serializable_and_stable(self):
        with self._patch_binding(_raw_profile_stub()):
            profile = benchmark_processor_performance(
                thread_policies=("single",),
                repeats=2,
                minimum_duration_ms=1.0,
                memory_budget_bytes=8 * 1024 * 1024,
            )

        as_dict = profile.to_dict()
        serialized = json.dumps(as_dict)
        assert json.loads(serialized) == as_dict
        assert as_dict["metadata"]["schema_version"] == 2
        assert as_dict["memory"]["L1"]["single"]["read"]["median_gbps"] == 10.5
        assert as_dict["memory"]["L1"]["single"]["read_scaling"][0]["median_gbps"] == 10.5
        assert as_dict["compute"]["float32"]["single"]["median_gops"] == 100.5
        assert as_dict["warnings"] == [
            "compute float16 (single): no compiled and runtime-detected native arithmetic path"
        ]

    def test_partial_platform_omits_unavailable_sections(self):
        metadata, topology, memory, compute, roofline, warnings = _raw_profile_stub()
        # Simulate a host with no measurable RAM level and no INT8 dot-product
        # path: they are simply absent from the raw tuples, exactly like the
        # real binding would report.
        with self._patch_binding((metadata, topology, memory, compute, roofline, warnings)):
            profile = benchmark_processor_performance(
                thread_policies=("single",),
                repeats=2,
                minimum_duration_ms=1.0,
                memory_budget_bytes=8 * 1024 * 1024,
            )

        assert "RAM" not in profile.memory
        assert "int8" not in profile.compute
        assert profile.warnings == tuple(warnings)

    def test_explicit_single_affinity_round_trips(self):
        metadata, topology, memory, compute, roofline, warnings = _raw_profile_stub()
        metadata = (
            metadata[0],
            metadata[1],
            metadata[2],
            metadata[3],
            metadata[4],
            (["single"], 2, 1.0, 8 * 1024 * 1024, True, (0, 3)),
            metadata[6],
        )
        with self._patch_binding((metadata, topology, memory, compute, roofline, warnings)):
            profile = benchmark_processor_performance(
                thread_policies=("single",), explicit_single_affinity=(0, 3)
            )

        assert profile.metadata.options.explicit_single_affinity == ExplicitAffinity(0, 3)
