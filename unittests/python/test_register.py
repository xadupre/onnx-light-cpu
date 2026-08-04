# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from onnx_light_cpu import register_kernels
from onnx_light_cpu._register import (
    _abs_kernel,
    _exp_kernel,
    _gemm_kernel,
    _log_kernel,
    _not_kernel,
)


class FakeEvaluator:
    """Minimal stand-in for onnx-light's ReferenceEvaluator.

    Records the kernels registered through ``register_custom_kernel`` so the
    registration behaviour can be tested without installing onnx-light.
    """

    def __init__(self):
        self.kernels = {}

    def register_custom_kernel(self, domain, op_type, fn):
        self.kernels[(domain, op_type)] = fn


class TestRegisterKernels:
    def test_registers_abs_on_default_domain(self):
        sess = FakeEvaluator()
        result = register_kernels(sess)
        assert result is sess
        assert ("", "Abs") in sess.kernels

    def test_registers_exp_and_log_on_default_domain(self):
        sess = FakeEvaluator()
        register_kernels(sess)
        assert ("", "Exp") in sess.kernels
        assert ("", "Log") in sess.kernels

    def test_registers_not_on_default_domain(self):
        sess = FakeEvaluator()
        register_kernels(sess)
        assert ("", "Not") in sess.kernels

    def test_registers_gemm_on_default_domain(self):
        sess = FakeEvaluator()
        register_kernels(sess)
        assert ("", "Gemm") in sess.kernels

    def test_custom_domain(self):
        sess = FakeEvaluator()
        register_kernels(sess, domain="ai.onnx")
        assert ("ai.onnx", "Abs") in sess.kernels
        assert ("ai.onnx", "Exp") in sess.kernels
        assert ("ai.onnx", "Log") in sess.kernels
        assert ("ai.onnx", "Not") in sess.kernels
        assert ("ai.onnx", "Gemm") in sess.kernels

    def test_registered_kernel_computes_abs(self):
        sess = FakeEvaluator()
        register_kernels(sess)
        fn = sess.kernels[("", "Abs")]
        inp = np.array([-1.0, 2.0, -3.5], dtype=np.float32)
        np.testing.assert_array_equal(fn(None, inp), np.abs(inp))


class TestAbsKernel:
    @pytest.mark.parametrize(
        "dtype", [np.float16, np.float32, np.float64, np.int8, np.int32, np.int64]
    )
    def test_supported_dtypes(self, dtype):
        inp = np.array([-1, 0, 3, -7], dtype=dtype)
        out = _abs_kernel(None, inp)
        assert out.dtype == inp.dtype
        np.testing.assert_array_equal(out, np.abs(inp))

    def test_multidimensional_input(self):
        inp = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
        out = _abs_kernel(None, inp)
        assert out.shape == inp.shape
        np.testing.assert_array_equal(out, np.abs(inp))

    def test_non_contiguous_input(self):
        base = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
        view = base.T  # non-contiguous
        out = _abs_kernel(None, view)
        assert out.shape == view.shape
        np.testing.assert_array_equal(out, np.abs(view))

    def test_empty_input(self):
        inp = np.array([], dtype=np.float32)
        out = _abs_kernel(None, inp)
        assert out.shape == (0,)

    def test_unsupported_dtype_falls_back_to_numpy(self):
        inp = np.array([-1, 2, -3], dtype=np.int16)
        out = _abs_kernel(None, inp)
        assert out.dtype == np.int16
        np.testing.assert_array_equal(out, np.abs(inp))

    def test_read_only_input(self):
        inp = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
        inp.setflags(write=False)
        out = _abs_kernel(None, inp)
        assert out.shape == inp.shape
        np.testing.assert_array_equal(out, np.abs(inp))
        assert inp[0, 0] == -1.0


class TestExpKernel:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_supported_dtypes(self, dtype):
        inp = np.array([-1.0, 0.0, 1.0, 2.0], dtype=dtype)
        out = _exp_kernel(None, inp)
        assert out.dtype == inp.dtype
        np.testing.assert_allclose(out, np.exp(inp), rtol=1e-2, atol=1e-3)

    def test_unsupported_dtype_falls_back_to_numpy(self):
        inp = np.array([1, 2, 3], dtype=np.int32)
        out = _exp_kernel(None, inp)
        np.testing.assert_allclose(out, np.exp(inp))

    def test_read_only_input(self):
        inp = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        inp.setflags(write=False)
        out = _exp_kernel(None, inp)
        np.testing.assert_allclose(out, np.exp(inp), rtol=1e-2, atol=1e-3)


class TestLogKernel:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_supported_dtypes(self, dtype):
        inp = np.array([0.5, 1.0, 2.0, 10.0], dtype=dtype)
        out = _log_kernel(None, inp)
        assert out.dtype == inp.dtype
        np.testing.assert_allclose(out, np.log(inp), rtol=1e-2, atol=1e-3)

    def test_unsupported_dtype_falls_back_to_numpy(self):
        inp = np.array([1, 2, 3], dtype=np.int32)
        out = _log_kernel(None, inp)
        np.testing.assert_allclose(out, np.log(inp))

    def test_read_only_input(self):
        inp = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        inp.setflags(write=False)
        out = _log_kernel(None, inp)
        np.testing.assert_allclose(out, np.log(inp), rtol=1e-2, atol=1e-3)


class TestNotKernel:
    def test_supported_dtype(self):
        inp = np.array([True, False, True, False], dtype=np.bool_)
        out = _not_kernel(None, inp)
        assert out.dtype == np.bool_
        np.testing.assert_array_equal(out, np.logical_not(inp))

    def test_multidimensional_input(self):
        inp = np.array([[True, False], [False, True]], dtype=np.bool_)
        out = _not_kernel(None, inp)
        assert out.shape == inp.shape
        np.testing.assert_array_equal(out, np.logical_not(inp))

    def test_non_contiguous_input(self):
        base = np.array([[True, False], [False, True]], dtype=np.bool_)
        view = base.T  # non-contiguous
        out = _not_kernel(None, view)
        assert out.shape == view.shape
        np.testing.assert_array_equal(out, np.logical_not(view))

    def test_empty_input(self):
        inp = np.array([], dtype=np.bool_)
        out = _not_kernel(None, inp)
        assert out.shape == (0,)

    def test_unsupported_dtype_falls_back_to_numpy(self):
        inp = np.array([0, 1, 2], dtype=np.int32)
        out = _not_kernel(None, inp)
        np.testing.assert_array_equal(out, np.logical_not(inp))

    def test_read_only_input(self):
        inp = np.array([True, False, True], dtype=np.bool_)
        inp.setflags(write=False)
        out = _not_kernel(None, inp)
        np.testing.assert_array_equal(out, np.logical_not(inp))
        assert inp[0]


class _FakeAttribute:
    """Minimal stand-in for onnx's ``AttributeProto`` scalar attributes."""

    # AttributeProto.AttributeType codes: FLOAT=1, INT=2.
    def __init__(self, name, *, f=None, i=None):
        self.name = name
        if f is not None:
            self.type = 1
            self.f = f
            self.i = 0
        else:
            self.type = 2
            self.i = i
            self.f = 0.0


class _FakeNode:
    """Minimal stand-in for onnx's ``NodeProto`` carrying Gemm attributes."""

    def __init__(self, **attrs):
        attributes = []
        for name, value in attrs.items():
            if isinstance(value, float):
                attributes.append(_FakeAttribute(name, f=value))
            else:
                attributes.append(_FakeAttribute(name, i=int(value)))
        self.attribute = attributes


class TestGemmKernel:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_supported_dtypes(self, dtype):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((3, 4)).astype(dtype)
        b = rng.standard_normal((4, 5)).astype(dtype)
        out = _gemm_kernel(_FakeNode(), a, b)
        assert out.dtype == dtype
        np.testing.assert_allclose(out, a @ b, rtol=1e-4, atol=1e-4)

    def test_alpha_beta_and_bias(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((2, 3)).astype(np.float32)
        b = rng.standard_normal((3, 4)).astype(np.float32)
        c = rng.standard_normal((2, 4)).astype(np.float32)
        node = _FakeNode(alpha=2.0, beta=0.5)
        out = _gemm_kernel(node, a, b, c)
        np.testing.assert_allclose(out, 2.0 * (a @ b) + 0.5 * c, rtol=1e-4, atol=1e-4)

    def test_transpose_a_and_b(self):
        rng = np.random.default_rng(2)
        a = rng.standard_normal((3, 2)).astype(np.float32)  # op(A) is 2x3
        b = rng.standard_normal((4, 3)).astype(np.float32)  # op(B) is 3x4
        node = _FakeNode(transA=1, transB=1)
        out = _gemm_kernel(node, a, b)
        np.testing.assert_allclose(out, a.T @ b.T, rtol=1e-4, atol=1e-4)

    def test_unsupported_dtype_falls_back_to_numpy(self):
        a = np.arange(6, dtype=np.int64).reshape(2, 3)
        b = np.arange(12, dtype=np.int64).reshape(3, 4)
        node = _FakeNode(alpha=2.0)
        out = _gemm_kernel(node, a, b)
        np.testing.assert_array_equal(out, 2.0 * (a @ b))

    def test_mixed_dtype_falls_back_to_numpy(self):
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((3, 2), dtype=np.float64)
        out = _gemm_kernel(_FakeNode(), a, b)
        np.testing.assert_allclose(out, a @ b)
