# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from onnx_light_cpu.onnx_py._cpukernels import exp, log


class TestExpFloat32:
    def test_basic(self):
        inp = np.array([-1.0, 0.0, 1.0, 2.5], dtype=np.float32)
        out = exp(inp)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, np.exp(inp), rtol=1e-5, atol=1e-6)

    def test_empty(self):
        inp = np.array([], dtype=np.float32)
        out = exp(inp)
        assert out.shape == (0,)

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1000])
    def test_various_sizes(self, size):
        inp = np.linspace(-20, 20, size, dtype=np.float32)
        out = exp(inp)
        np.testing.assert_allclose(out, np.exp(inp), rtol=1e-5, atol=1e-6)

    def test_special_values(self):
        inp = np.array([np.inf, -np.inf, np.nan, 0.0, 200.0, -200.0], dtype=np.float32)
        out = exp(inp)
        assert out[0] == np.inf
        assert out[1] == 0.0
        assert np.isnan(out[2])
        assert out[3] == 1.0
        assert out[4] == np.inf
        assert out[5] == 0.0


class TestExpFloat64:
    def test_basic(self):
        inp = np.array([-1.0, 0.0, 1.0, 2.5], dtype=np.float64)
        out = exp(inp)
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, np.exp(inp), rtol=1e-12)

    @pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 8, 9, 100, 256])
    def test_various_sizes(self, size):
        inp = np.linspace(-30, 30, size, dtype=np.float64)
        out = exp(inp)
        np.testing.assert_allclose(out, np.exp(inp), rtol=1e-12)

    def test_special_values(self):
        inp = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float64)
        out = exp(inp)
        assert out[0] == np.inf
        assert out[1] == 0.0
        assert np.isnan(out[2])
        assert out[3] == 1.0


class TestExpFloat16:
    def test_basic(self):
        inp = np.array([-1.0, 0.0, 1.0, 2.5], dtype=np.float16)
        out = exp(inp)
        assert out.dtype == np.float16
        np.testing.assert_allclose(out, np.exp(inp), rtol=1e-2, atol=1e-3)

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 33])
    def test_various_sizes(self, size):
        inp = np.linspace(-5, 5, size, dtype=np.float16)
        out = exp(inp)
        np.testing.assert_allclose(out, np.exp(inp), rtol=1e-2, atol=1e-3)


class TestLogFloat32:
    def test_basic(self):
        inp = np.array([0.5, 1.0, 2.0, 10.0], dtype=np.float32)
        out = log(inp)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, np.log(inp), rtol=1e-5, atol=1e-6)

    def test_empty(self):
        inp = np.array([], dtype=np.float32)
        out = log(inp)
        assert out.shape == (0,)

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1000])
    def test_various_sizes(self, size):
        inp = np.linspace(1e-3, 1000, size, dtype=np.float32)
        out = log(inp)
        np.testing.assert_allclose(out, np.log(inp), rtol=1e-5, atol=1e-6)

    def test_special_values(self):
        inp = np.array([1.0, 0.0, -1.0, np.inf, np.nan], dtype=np.float32)
        out = log(inp)
        assert out[0] == 0.0
        assert out[1] == -np.inf
        assert np.isnan(out[2])
        assert out[3] == np.inf
        assert np.isnan(out[4])


class TestLogFloat64:
    def test_basic(self):
        inp = np.array([0.5, 1.0, 2.0, 10.0], dtype=np.float64)
        out = log(inp)
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, np.log(inp), rtol=1e-12)

    @pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 8, 9, 100, 256])
    def test_various_sizes(self, size):
        inp = np.linspace(1e-6, 1e6, size, dtype=np.float64)
        out = log(inp)
        np.testing.assert_allclose(out, np.log(inp), rtol=1e-12)

    def test_special_values(self):
        inp = np.array([1.0, 0.0, -1.0, np.inf, np.nan], dtype=np.float64)
        out = log(inp)
        assert out[0] == 0.0
        assert out[1] == -np.inf
        assert np.isnan(out[2])
        assert out[3] == np.inf
        assert np.isnan(out[4])


class TestLogFloat16:
    def test_basic(self):
        inp = np.array([0.5, 1.0, 2.0, 10.0], dtype=np.float16)
        out = log(inp)
        assert out.dtype == np.float16
        np.testing.assert_allclose(out, np.log(inp), rtol=1e-2, atol=1e-3)

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 33])
    def test_various_sizes(self, size):
        inp = np.linspace(0.5, 100, size, dtype=np.float16)
        out = log(inp)
        np.testing.assert_allclose(out, np.log(inp), rtol=1e-2, atol=1e-3)


class TestExpLogDispatch:
    @pytest.mark.parametrize("fn", [exp, log])
    def test_unsupported_dtype(self, fn):
        inp = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(ValueError):
            fn(inp)
