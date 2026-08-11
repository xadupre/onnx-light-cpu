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
check that each kernel produces the expected result **and** that the model
actually dispatches to (uses) the onnx-light-cpu kernel, for every element type
combination the kernel implements.

onnx-light and the ``_cpuregister`` extension (built with
``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``) are required; when either is missing the
whole module is skipped via :func:`pytest.importorskip`. ``ml_dtypes`` provides
the ``bfloat16`` numpy dtype and is likewise imported through
:func:`pytest.importorskip`.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnx_light")
pytest.importorskip("onnx_light_cpu.onnx_py._cpuregister")
pytest.importorskip("ml_dtypes")

import ml_dtypes
from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import (
    clear_used_kernel_names,
    register_kernels,
    registered_kernel_names,
    used_kernel_names,
)

# Element types implemented by each kernel (see the ``switch`` statements in
# ``onnx_light_cpu/kernels/math/*.cc`` and ``onnx_light_cpu/kernels/logical``).
ABS_DTYPES = [
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    ml_dtypes.bfloat16,
]
UNARY_FLOAT_DTYPES = [np.float32, np.float64, np.float16, ml_dtypes.bfloat16]
GEMM_DTYPES = [np.float32, np.float64, np.float16, ml_dtypes.bfloat16]


def _np_to_tp(dtype):
    mapping = {
        np.dtype(np.float32): TensorProto.FLOAT,
        np.dtype(np.float64): TensorProto.DOUBLE,
        np.dtype(np.int8): TensorProto.INT8,
        np.dtype(np.int16): TensorProto.INT16,
        np.dtype(np.int32): TensorProto.INT32,
        np.dtype(np.int64): TensorProto.INT64,
        np.dtype(np.float16): TensorProto.FLOAT16,
        np.dtype(ml_dtypes.bfloat16): TensorProto.BFLOAT16,
        np.dtype(np.bool_): TensorProto.BOOL,
    }
    return mapping[np.dtype(dtype)]


def _dtype_id(dtype):
    return np.dtype(dtype).name


def _tolerances(dtype):
    """Relative/absolute tolerances suited to the precision of ``dtype``."""
    npd = np.dtype(dtype)
    if npd == np.dtype(ml_dtypes.bfloat16):
        return {"rtol": 5e-1, "atol": 5e-1}
    if npd == np.dtype(np.float16):
        return {"rtol": 5e-2, "atol": 5e-2}
    if npd == np.dtype(np.float32):
        return {"rtol": 1e-5, "atol": 0.0}
    return {"rtol": 1e-6, "atol": 0.0}


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


class TestKernelsEndToEnd:
    @pytest.mark.parametrize("dtype", ABS_DTYPES, ids=_dtype_id)
    def test_abs(self, dtype):
        x = np.array([-3, -1, 0, 2, 5], dtype=dtype)
        node = helper.make_node("Abs", ["X"], ["Y"])
        (y,) = _run(node, {"X": x}, {"Y": _np_to_tp(dtype)})
        np.testing.assert_array_equal(y.astype(np.float64), np.abs(x).astype(np.float64))

    @pytest.mark.parametrize("dtype", UNARY_FLOAT_DTYPES, ids=_dtype_id)
    def test_exp(self, dtype):
        x = np.array([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=dtype)
        node = helper.make_node("Exp", ["X"], ["Y"])
        (y,) = _run(node, {"X": x}, {"Y": _np_to_tp(dtype)})
        np.testing.assert_allclose(
            y.astype(np.float64),
            np.exp(x.astype(np.float64)),
            **_tolerances(dtype),
        )

    @pytest.mark.parametrize("dtype", UNARY_FLOAT_DTYPES, ids=_dtype_id)
    def test_log(self, dtype):
        x = np.array([0.1, 0.5, 1.0, 2.0, 10.0], dtype=dtype)
        node = helper.make_node("Log", ["X"], ["Y"])
        (y,) = _run(node, {"X": x}, {"Y": _np_to_tp(dtype)})
        np.testing.assert_allclose(
            y.astype(np.float64),
            np.log(x.astype(np.float64)),
            **_tolerances(dtype),
        )

    @pytest.mark.parametrize("dtype", GEMM_DTYPES, ids=_dtype_id)
    def test_gemm(self, dtype):
        a = np.arange(6, dtype=dtype).reshape(2, 3)
        b = np.arange(12, dtype=dtype).reshape(3, 4)
        node = helper.make_node("Gemm", ["A", "B"], ["Y"])
        (y,) = _run(node, {"A": a, "B": b}, {"Y": _np_to_tp(dtype)})
        np.testing.assert_allclose(
            y.astype(np.float64),
            a.astype(np.float64) @ b.astype(np.float64),
            **_tolerances(dtype),
        )

    def test_not(self):
        x = np.array([True, False, True, False], dtype=np.bool_)
        node = helper.make_node("Not", ["X"], ["Y"])
        (y,) = _run(node, {"X": x}, {"Y": TensorProto.BOOL})
        np.testing.assert_array_equal(y, np.logical_not(x))


class TestUsedKernelNames:
    """Checks the kernels a model actually dispatches to are the onnx-light-cpu
    ones, identified by the library-qualified name each kernel records when it
    runs (rather than onnx-light's identically-behaving built-in kernels), for
    every element type combination each kernel implements."""

    def test_registered_kernel_names(self):
        names = registered_kernel_names()
        assert names == {
            "Abs": "onnx_light_cpu::Abs",
            "Exp": "onnx_light_cpu::Exp",
            "Log": "onnx_light_cpu::Log",
            "Gemm": "onnx_light_cpu::Gemm",
            "Not": "onnx_light_cpu::Not",
        }

    @pytest.mark.parametrize("dtype", ABS_DTYPES, ids=_dtype_id)
    def test_abs_uses_accelerated_kernel(self, dtype):
        register_kernels()
        clear_used_kernel_names()
        x = np.array([-1, 2], dtype=dtype)
        node = helper.make_node("Abs", ["X"], ["Y"])
        _run(node, {"X": x}, {"Y": _np_to_tp(dtype)})
        assert used_kernel_names() == ["onnx_light_cpu::Abs"]

    @pytest.mark.parametrize("dtype", UNARY_FLOAT_DTYPES, ids=_dtype_id)
    def test_exp_uses_accelerated_kernel(self, dtype):
        register_kernels()
        clear_used_kernel_names()
        x = np.array([0.0, 1.0], dtype=dtype)
        node = helper.make_node("Exp", ["X"], ["Y"])
        _run(node, {"X": x}, {"Y": _np_to_tp(dtype)})
        assert used_kernel_names() == ["onnx_light_cpu::Exp"]

    @pytest.mark.parametrize("dtype", UNARY_FLOAT_DTYPES, ids=_dtype_id)
    def test_log_uses_accelerated_kernel(self, dtype):
        register_kernels()
        clear_used_kernel_names()
        x = np.array([1.0, 2.0], dtype=dtype)
        node = helper.make_node("Log", ["X"], ["Y"])
        _run(node, {"X": x}, {"Y": _np_to_tp(dtype)})
        assert used_kernel_names() == ["onnx_light_cpu::Log"]

    @pytest.mark.parametrize("dtype", GEMM_DTYPES, ids=_dtype_id)
    def test_gemm_uses_accelerated_kernel(self, dtype):
        register_kernels()
        clear_used_kernel_names()
        a = np.arange(6, dtype=dtype).reshape(2, 3)
        b = np.arange(12, dtype=dtype).reshape(3, 4)
        node = helper.make_node("Gemm", ["A", "B"], ["Y"])
        _run(node, {"A": a, "B": b}, {"Y": _np_to_tp(dtype)})
        assert used_kernel_names() == ["onnx_light_cpu::Gemm"]

    def test_not_uses_accelerated_kernel(self):
        register_kernels()
        clear_used_kernel_names()
        x = np.array([True, False], dtype=np.bool_)
        node = helper.make_node("Not", ["X"], ["Y"])
        _run(node, {"X": x}, {"Y": TensorProto.BOOL})
        assert used_kernel_names() == ["onnx_light_cpu::Not"]

    def test_multiple_nodes_record_every_kernel(self):
        register_kernels()
        clear_used_kernel_names()
        x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
        nodes = [
            helper.make_node("Abs", ["X"], ["A"]),
            helper.make_node("Exp", ["A"], ["E"]),
            helper.make_node("Log", ["E"], ["Y"]),
        ]
        graph = helper.make_graph(
            nodes,
            "chain",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        sess = ReferenceEvaluator(model)
        sess.run(None, {"X": x})
        assert sorted(used_kernel_names()) == sorted(
            [
                "onnx_light_cpu::Abs",
                "onnx_light_cpu::Exp",
                "onnx_light_cpu::Log",
            ]
        )
