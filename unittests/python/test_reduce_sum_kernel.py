# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests INT64 ReduceSum dispatch through the registered CPU kernel."""

import numpy as np

import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import (
    register_kernels,
    registered_kernels,
    set_kernel_usage_recording,
    used_kernel_names,
)

_MODEL = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("ReduceSum", ["attention_mask", "axes"], ["sum"], keepdims=1),
            oh.make_node("Abs", ["x"], ["absolute"]),
        ],
        "reduce_sum_int64_cpu",
        [
            oh.make_tensor_value_info("attention_mask", TensorProto.INT64, [2, 3]),
            oh.make_tensor_value_info("axes", TensorProto.INT64, [1]),
            oh.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
        ],
        [
            oh.make_tensor_value_info("sum", TensorProto.INT64, [2, 1]),
            oh.make_tensor_value_info("absolute", TensorProto.FLOAT, [2]),
        ],
    ),
    opset_imports=[oh.make_opsetid("", 13)],
    ir_version=10,
)
_FEEDS = {
    "attention_mask": np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int64),
    "axes": np.array([1], dtype=np.int64),
    "x": np.array([-1.0, 2.0], dtype=np.float32),
}
_EXPECTED = ReferenceEvaluator(_MODEL).run(None, _FEEDS)
register_kernels()


class TestReduceSumKernel(ExtTestCase):
    def test_int64_reduce_sum_with_cpu_kernels(self):
        reduce_sum = next(r for r in registered_kernels() if r.op_type == "ReduceSum")
        self.assertEqual(reduce_sum.types, ("INT64",))
        session = ReferenceEvaluator(_MODEL)
        set_kernel_usage_recording(session, True)

        actual = session.run(None, _FEEDS)
        self.assertEqual(actual[0].dtype, np.int64)
        self.assertEqual(actual[0].shape, (2, 1))
        self.assertEqualArray(actual[0], _EXPECTED[0])
        self.assertEqualArray(actual[1], _EXPECTED[1])
        self.assertIn("onnx_light_cpu::Abs", used_kernel_names(session))
        self.assertIn("onnx_light_cpu::ReduceSum", used_kernel_names(session))
