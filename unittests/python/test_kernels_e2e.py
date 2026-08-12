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
``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``) are required; when either is missing the
whole module is skipped via :func:`pytest.importorskip`.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnx_light")
pytest.importorskip("onnx_light_cpu.onnx_py._cpuregister")

from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import (
    clear_used_kernel_names,
    register_kernels,
    registered_kernel_names,
    used_kernel_names,
)


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


class TestUsedKernelNames:
    """Checks the kernels a model actually dispatches to are the onnx-light-cpu
    ones, identified by the library-qualified name each kernel records when it
    runs (rather than onnx-light's identically-behaving built-in kernels)."""

    def test_registered_kernel_names(self):
        names = registered_kernel_names()
        assert names == {
            "Abs": "onnx_light_cpu::Abs",
            "Exp": "onnx_light_cpu::Exp",
            "Log": "onnx_light_cpu::Log",
            "Gemm": "onnx_light_cpu::Gemm",
            "Not": "onnx_light_cpu::Not",
        }

    @pytest.mark.parametrize(
        ("op_type", "inputs", "output_type", "expected_name"),
        [
            (
                "Abs",
                {"X": np.array([-1.0, 2.0], dtype=np.float32)},
                TensorProto.FLOAT,
                "onnx_light_cpu::Abs",
            ),
            (
                "Exp",
                {"X": np.array([0.0, 1.0], dtype=np.float32)},
                TensorProto.FLOAT,
                "onnx_light_cpu::Exp",
            ),
            (
                "Log",
                {"X": np.array([1.0, 2.0], dtype=np.float32)},
                TensorProto.FLOAT,
                "onnx_light_cpu::Log",
            ),
            (
                "Not",
                {"X": np.array([True, False], dtype=np.bool_)},
                TensorProto.BOOL,
                "onnx_light_cpu::Not",
            ),
        ],
    )
    def test_single_op_uses_accelerated_kernel(self, op_type, inputs, output_type, expected_name):
        register_kernels()
        clear_used_kernel_names()
        node = helper.make_node(op_type, list(inputs), ["Y"])
        _run(node, inputs, {"Y": output_type})
        assert used_kernel_names() == [expected_name]

    def test_gemm_uses_accelerated_kernel(self):
        register_kernels()
        clear_used_kernel_names()
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        b = np.arange(12, dtype=np.float32).reshape(3, 4)
        node = helper.make_node("Gemm", ["A", "B"], ["Y"])
        _run(node, {"A": a, "B": b}, {"Y": TensorProto.FLOAT})
        assert used_kernel_names() == ["onnx_light_cpu::Gemm"]

    @pytest.mark.parametrize("size", [16, 64, 128])
    def test_gemm_uses_accelerated_kernel_benchmark_sizes(self, size):
        # Backs docs/examples/plot_gemm_benchmark.py: the timed ``Gemm`` curve
        # must dispatch to onnx-light-cpu (not onnx-light's built-in kernel) for
        # the same square shapes the benchmark measures, otherwise its timings
        # would be indistinguishable from the built-in baseline. The benchmark
        # verifies dispatch the same way, via ``registered_kernel_names``.
        register_kernels()
        assert registered_kernel_names()["Gemm"] == "onnx_light_cpu::Gemm"
        rng = np.random.default_rng(0)
        a = rng.standard_normal((size, size)).astype(np.float32)
        b = rng.standard_normal((size, size)).astype(np.float32)
        node = helper.make_node("Gemm", ["A", "B"], ["Y"], alpha=1.0, beta=1.0)
        (y,) = _run(node, {"A": a, "B": b}, {"Y": TensorProto.FLOAT})
        np.testing.assert_allclose(y, a @ b, rtol=1e-2, atol=1e-2)

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
