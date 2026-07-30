# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from onnx_light_cpu import register_kernels
from onnx_light_cpu._register import _abs_kernel


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

    def test_custom_domain(self):
        sess = FakeEvaluator()
        register_kernels(sess, domain="ai.onnx")
        assert ("ai.onnx", "Abs") in sess.kernels

    def test_registered_kernel_computes_abs(self):
        sess = FakeEvaluator()
        register_kernels(sess)
        fn = sess.kernels[("", "Abs")]
        inp = np.array([-1.0, 2.0, -3.5], dtype=np.float32)
        np.testing.assert_array_equal(fn(None, inp), np.abs(inp))


class TestAbsKernel:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
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
