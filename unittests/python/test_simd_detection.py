# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the runtime helpers exposed by the Python bindings.

The Python extension ``onnx_light_cpu.onnx_py._cpukernels`` intentionally
exposes only detection and configuration helpers; the kernels themselves are
reachable through onnx-light's runtime after registration, not as numpy-like
Python functions.
"""

from onnx_light_cpu.onnx_py._cpukernels import (
    detect_simd_level,
    has_cpu_kernels,
    parallel_for_thread_count,
)


class TestDetection:
    def test_has_cpu_kernels(self):
        assert has_cpu_kernels() is True

    def test_detect_simd_level(self):
        level = detect_simd_level()
        assert isinstance(level, int)
        assert 0 <= level <= 4

    def test_parallel_for_thread_count(self):
        thread_count = parallel_for_thread_count()
        assert isinstance(thread_count, int)
        assert thread_count >= 1
