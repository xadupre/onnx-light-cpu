# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from onnx_light_cpu.onnx_py._cpukernels import (
    abs,
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
        out = abs(inp)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, np.abs(inp))

    def test_empty(self):
        inp = np.array([], dtype=np.float32)
        out = abs(inp)
        assert out.shape == (0,)

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1000])
    def test_various_sizes(self, size):
        inp = np.linspace(-100, 100, size, dtype=np.float32)
        out = abs(inp)
        np.testing.assert_allclose(out, np.abs(inp))

    def test_negative_zero(self):
        inp = np.array([-0.0], dtype=np.float32)
        out = abs(inp)
        assert out[0] == 0.0
        assert not np.signbit(out[0])


class TestAbsFloat64:
    def test_basic(self):
        inp = np.array([-1.0, 0.0, 3.0, -7.5], dtype=np.float64)
        out = abs(inp)
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, np.abs(inp))

    @pytest.mark.parametrize("size", [1, 3, 4, 5, 7, 8, 9, 100, 256])
    def test_various_sizes(self, size):
        inp = np.linspace(-100, 100, size, dtype=np.float64)
        out = abs(inp)
        np.testing.assert_allclose(out, np.abs(inp))


class TestAbsFloat16:
    def test_basic(self):
        inp = np.array([-1.0, 0.0, 3.0, -7.5], dtype=np.float16)
        out = abs(inp)
        assert out.dtype == np.float16
        np.testing.assert_array_equal(out, np.abs(inp))

    def test_empty(self):
        inp = np.array([], dtype=np.float16)
        out = abs(inp)
        assert out.dtype == np.float16
        assert out.shape == (0,)

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1000])
    def test_various_sizes(self, size):
        inp = np.linspace(-100, 100, size, dtype=np.float16)
        out = abs(inp)
        np.testing.assert_array_equal(out, np.abs(inp))

    def test_negative_zero(self):
        inp = np.array([-0.0], dtype=np.float16)
        out = abs(inp)
        assert out[0] == 0.0
        assert not np.signbit(out[0])

    def test_special_values(self):
        inp = np.array([np.inf, -np.inf, np.nan, -65504.0], dtype=np.float16)
        out = abs(inp)
        assert out[0] == np.inf
        assert out[1] == np.inf
        assert np.isnan(out[2])
        assert out[3] == 65504.0


class TestAbsInt8:
    def test_basic(self):
        inp = np.array([-1, 0, 3, -7, 100, 127], dtype=np.int8)
        out = abs(inp)
        assert out.dtype == np.int8
        np.testing.assert_array_equal(out, np.abs(inp))

    def test_int8_min_wraps(self):
        # |-128| overflows int8 and wraps to -128, matching numpy.abs.
        inp = np.array([-128], dtype=np.int8)
        out = abs(inp)
        assert out[0] == -128

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256])
    def test_various_sizes(self, size):
        inp = np.arange(-size // 2, size // 2, dtype=np.int8)
        out = abs(inp)
        np.testing.assert_array_equal(out, np.abs(inp))


class TestAbsInt32:
    def test_basic(self):
        inp = np.array([-1, 0, 3, -7, 100], dtype=np.int32)
        out = abs(inp)
        assert out.dtype == np.int32
        np.testing.assert_array_equal(out, np.abs(inp))

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 100, 256])
    def test_various_sizes(self, size):
        inp = np.arange(-size // 2, size // 2, dtype=np.int32)
        out = abs(inp)
        np.testing.assert_array_equal(out, np.abs(inp))


class TestAbsInt64:
    def test_basic(self):
        inp = np.array([-1, 0, 3, -7, 100], dtype=np.int64)
        out = abs(inp)
        assert out.dtype == np.int64
        np.testing.assert_array_equal(out, np.abs(inp))

    @pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 8, 9, 100, 256])
    def test_various_sizes(self, size):
        inp = np.arange(-size // 2, size // 2, dtype=np.int64)
        out = abs(inp)
        np.testing.assert_array_equal(out, np.abs(inp))


class TestAbsDispatch:
    def test_unsupported_dtype(self):
        inp = np.array([-1, 2, -3], dtype=np.int16)
        with pytest.raises(ValueError):
            abs(inp)
