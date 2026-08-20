# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that thread scheduling remains owned by onnx-light."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_private_parallel_for_is_absent():
    assert not (_ROOT / "onnx_light_cpu" / "impl" / "parallel_for.h").exists()
    assert not (_ROOT / "onnx_light_cpu" / "impl" / "parallel_for.cc").exists()
    assert not (_ROOT / "onnx_light_cpu" / "kernels" / "session_executor_adapter.h").exists()


def test_private_scheduler_controls_are_absent():
    cmake = (_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    bindings = (_ROOT / "onnx_light_cpu" / "onnx_py" / "_cpupy_math_kernels.cc").read_text(
        encoding="utf-8"
    )
    for removed in (
        "ONNX_LIGHT_CPU_MAX_THREADS",
        "ONNX_LIGHT_CPU_NUM_THREADS",
        "ONNX_LIGHT_CPU_SPIN_COUNT",
        "parallel_for_thread_count",
    ):
        assert removed not in cmake
        assert removed not in bindings
