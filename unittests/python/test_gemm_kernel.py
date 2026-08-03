# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from onnx_light_cpu.onnx_py._cpukernels import gemm


class TestGemmFloat32:
    def test_basic_matmul(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((5, 3)).astype(np.float32)
        b = rng.standard_normal((3, 7)).astype(np.float32)
        out = gemm(a, b, beta=0.0)
        assert out.dtype == np.float32
        assert out.shape == (5, 7)
        np.testing.assert_allclose(out, a @ b, rtol=1e-4, atol=1e-4)

    def test_alpha_beta_bias(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((4, 6)).astype(np.float32)
        b = rng.standard_normal((6, 8)).astype(np.float32)
        c = rng.standard_normal((4, 8)).astype(np.float32)
        out = gemm(a, b, c, alpha=0.5, beta=2.0)
        np.testing.assert_allclose(out, 0.5 * (a @ b) + 2.0 * c, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("trans_a", [False, True])
    @pytest.mark.parametrize("trans_b", [False, True])
    def test_transpose_variants(self, trans_a, trans_b):
        rng = np.random.default_rng(2)
        m, k, n = 6, 4, 5
        a = rng.standard_normal((k, m) if trans_a else (m, k)).astype(np.float32)
        b = rng.standard_normal((n, k) if trans_b else (k, n)).astype(np.float32)
        out = gemm(a, b, trans_a=trans_a, trans_b=trans_b, beta=0.0)
        op_a = a.T if trans_a else a
        op_b = b.T if trans_b else b
        np.testing.assert_allclose(out, op_a @ op_b, rtol=1e-4, atol=1e-4)

    def test_beta_zero_ignores_bias(self):
        rng = np.random.default_rng(3)
        a = rng.standard_normal((3, 3)).astype(np.float32)
        b = rng.standard_normal((3, 3)).astype(np.float32)
        c = np.full((3, 3), np.nan, dtype=np.float32)
        out = gemm(a, b, c, beta=0.0)
        np.testing.assert_allclose(out, a @ b, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("m,k,n", [(1, 1, 1), (17, 33, 9), (64, 48, 96)])
    def test_various_sizes(self, m, k, n):
        rng = np.random.default_rng(m * 100 + n)
        a = rng.standard_normal((m, k)).astype(np.float32)
        b = rng.standard_normal((k, n)).astype(np.float32)
        out = gemm(a, b, beta=0.0)
        np.testing.assert_allclose(out, a @ b, rtol=1e-3, atol=1e-3)


class TestGemmFloat64:
    def test_basic_matmul(self):
        rng = np.random.default_rng(4)
        a = rng.standard_normal((5, 9)).astype(np.float64)
        b = rng.standard_normal((9, 7)).astype(np.float64)
        c = rng.standard_normal((5, 7)).astype(np.float64)
        out = gemm(a, b, c, alpha=1.5, beta=-0.5)
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, 1.5 * (a @ b) - 0.5 * c, rtol=1e-10, atol=1e-10)


class TestGemmErrors:
    def test_shape_mismatch(self):
        a = np.zeros((3, 4), dtype=np.float32)
        b = np.zeros((5, 6), dtype=np.float32)
        with pytest.raises(ValueError):
            gemm(a, b)

    def test_bias_shape_mismatch(self):
        a = np.zeros((3, 4), dtype=np.float32)
        b = np.zeros((4, 5), dtype=np.float32)
        c = np.zeros((3, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            gemm(a, b, c)

    def test_dtype_mismatch(self):
        a = np.zeros((3, 4), dtype=np.float32)
        b = np.zeros((4, 5), dtype=np.float64)
        with pytest.raises(ValueError):
            gemm(a, b)

    def test_unsupported_dtype(self):
        a = np.zeros((3, 4), dtype=np.int32)
        b = np.zeros((4, 5), dtype=np.int32)
        with pytest.raises(ValueError):
            gemm(a, b)
