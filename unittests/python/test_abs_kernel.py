# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from onnx_light_cpu.onnx_py._cpukernels import (
    abs_float32,
    abs_float64,
    abs_int32,
    abs_int64,
    detect_simd_level,
    has_cpu_kernels,
)


class TestDetection:
    def test_has_cpu_kernels(self):
        assert has_cpu_kernels() is True

    def test_detect_simd_level(self):
        level = detect_simd_level()
        assert isinstance(level, int)
        assert level >= 0


class TestAbsFloat32:
    def test_basic(self):
        inp = np.array([-1.0, 0.0, 3.0, -7.5], dtype=np.float32)
        out = np.zeros_like(inp)
        abs_float32(inp, out)
        np.testing.assert_array_equal(out, np.abs(inp))

    def test_empty(self):
        inp = np.array([], dtype=np.float32)
        out = np.array([], dtype=np.float32)
        abs_float32(inp, out)

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1000])
    def test_various_sizes(self, size):
        inp = np.linspace(-100, 100, size, dtype=np.float32)
        out = np.zeros(size, dtype=np.float32)
        abs_float32(inp, out)
        np.testing.assert_allclose(out, np.abs(inp))

    def test_negative_zero(self):
        inp = np.array([-0.0], dtype=np.float32)
        out = np.zeros(1, dtype=np.float32)
        abs_float32(inp, out)
        assert out[0] == 0.0
        assert not np.signbit(out[0])


class TestAbsFloat64:
    def test_basic(self):
        inp = np.array([-1.0, 0.0, 3.0, -7.5], dtype=np.float64)
        out = np.zeros_like(inp)
        abs_float64(inp, out)
        np.testing.assert_array_equal(out, np.abs(inp))

    @pytest.mark.parametrize("size", [1, 3, 4, 5, 7, 8, 9, 100, 256])
    def test_various_sizes(self, size):
        inp = np.linspace(-100, 100, size, dtype=np.float64)
        out = np.zeros(size, dtype=np.float64)
        abs_float64(inp, out)
        np.testing.assert_allclose(out, np.abs(inp))


class TestAbsInt32:
    def test_basic(self):
        inp = np.array([-1, 0, 3, -7, 100], dtype=np.int32)
        out = np.zeros_like(inp)
        abs_int32(inp, out)
        np.testing.assert_array_equal(out, np.abs(inp))

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 100, 256])
    def test_various_sizes(self, size):
        inp = np.arange(-size // 2, size // 2, dtype=np.int32)
        out = np.zeros_like(inp)
        abs_int32(inp, out)
        np.testing.assert_array_equal(out, np.abs(inp))


class TestAbsInt64:
    def test_basic(self):
        inp = np.array([-1, 0, 3, -7, 100], dtype=np.int64)
        out = np.zeros_like(inp)
        abs_int64(inp, out)
        np.testing.assert_array_equal(out, np.abs(inp))

    @pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 8, 9, 100, 256])
    def test_various_sizes(self, size):
        inp = np.arange(-size // 2, size // 2, dtype=np.int64)
        out = np.zeros_like(inp)
        abs_int64(inp, out)
        np.testing.assert_array_equal(out, np.abs(inp))
