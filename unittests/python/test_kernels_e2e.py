# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Python tests for every implemented onnx-light-cpu kernel.

The Python surface no longer exposes numpy-like kernel functions; the
SIMD-accelerated ``Abs``, ``Exp``, ``Log``, ``Gemm`` and ``Not`` kernels are
only reachable through onnx-light's runtime after
:func:`onnx_light_cpu.register_kernels` installs them into onnx-light's shared
C++ ``KernelDispatchTable``. These tests therefore build a single-node ONNX
model per operator and run it through onnx-light's ``ReferenceEvaluator`` to
check that each kernel produces the expected result.

onnx-light and the ``_cpuregister`` extension (built with
``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``) are assumed to be available.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import register_kernels


def _run(node, inputs, output_types):
    """Builds a single-node model, registers the kernels and runs it.

    ``inputs`` maps input names to numpy arrays; ``output_types`` maps output
    names to their ``TensorProto`` element types.
    """
    register_kernels()
    graph_inputs = [
        helper.make_tensor_value_info(name, _np_to_tp(arr.dtype), list(arr.shape))
        for name, arr in inputs.items()
    ]
    graph_outputs = [
        helper.make_tensor_value_info(name, tp, None) for name, tp in output_types.items()
    ]
    graph = helper.make_graph([node], "test", graph_inputs, graph_outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    sess = ReferenceEvaluator(model)
    return sess.run(None, inputs)


def _np_to_tp(dtype):
    mapping = {
        np.dtype(np.float32): TensorProto.FLOAT,
        np.dtype(np.float64): TensorProto.DOUBLE,
        np.dtype(np.int64): TensorProto.INT64,
        np.dtype(np.bool_): TensorProto.BOOL,
    }
    return mapping[np.dtype(dtype)]


class TestKernelsEndToEnd:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
    def test_abs(self, dtype):
        x = np.array([-3, -1, 0, 2, 5], dtype=dtype)
        node = helper.make_node("Abs", ["X"], ["Y"])
        (y,) = _run(node, {"X": x}, {"Y": _np_to_tp(dtype)})
        np.testing.assert_array_equal(y, np.abs(x))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_exp(self, dtype):
        x = np.array([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=dtype)
        node = helper.make_node("Exp", ["X"], ["Y"])
        (y,) = _run(node, {"X": x}, {"Y": _np_to_tp(dtype)})
        np.testing.assert_allclose(y, np.exp(x), rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_log(self, dtype):
        x = np.array([0.1, 0.5, 1.0, 2.0, 10.0], dtype=dtype)
        node = helper.make_node("Log", ["X"], ["Y"])
        (y,) = _run(node, {"X": x}, {"Y": _np_to_tp(dtype)})
        np.testing.assert_allclose(y, np.log(x), rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_gemm(self, dtype):
        a = np.arange(6, dtype=dtype).reshape(2, 3)
        b = np.arange(12, dtype=dtype).reshape(3, 4)
        node = helper.make_node("Gemm", ["A", "B"], ["Y"])
        (y,) = _run(node, {"A": a, "B": b}, {"Y": _np_to_tp(dtype)})
        np.testing.assert_allclose(y, a @ b, rtol=1e-5)

    def test_not(self):
        x = np.array([True, False, True, False], dtype=np.bool_)
        node = helper.make_node("Not", ["X"], ["Y"])
        (y,) = _run(node, {"X": x}, {"Y": TensorProto.BOOL})
        np.testing.assert_array_equal(y, np.logical_not(x))
