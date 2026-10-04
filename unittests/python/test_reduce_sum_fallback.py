# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""INT64 ReduceSum dispatches to onnx-light-cpu when CPU kernels are registered."""

import numpy as np

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import (
    register_kernels,
    registered_kernels,
    set_kernel_usage_recording,
    used_kernel_names,
)

_MODEL = helper.make_model(
    helper.make_graph(
        [
            helper.make_node("ReduceSum", ["attention_mask", "axes"], ["sum"], keepdims=1),
            helper.make_node("Abs", ["x"], ["absolute"]),
        ],
        "reduce_sum_int64_fallback",
        [
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [2, 3]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("sum", TensorProto.INT64, [2, 1]),
            helper.make_tensor_value_info("absolute", TensorProto.FLOAT, [2]),
        ],
    ),
    opset_imports=[helper.make_opsetid("", 13)],
    ir_version=10,
)
_FEEDS = {
    "attention_mask": np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int64),
    "axes": np.array([1], dtype=np.int64),
    "x": np.array([-1.0, 2.0], dtype=np.float32),
}


class TestReduceSumFallback(ExtTestCase):
    def test_int64_reduce_sum_with_cpu_kernels(self):
        reduce_sum = next(r for r in registered_kernels() if r.op_type == "ReduceSum")
        self.assertEqual(reduce_sum.types, ("INT64",))
        register_kernels()
        session = ReferenceEvaluator(_MODEL)
        set_kernel_usage_recording(session, True)

        actual = session.run(None, _FEEDS)
        self.assertEqual(actual[0].dtype, np.int64)
        self.assertEqual(actual[0].shape, (2, 1))
        self.assertEqualArray(actual[0], np.array([[2], [2]], dtype=np.int64))
        self.assertEqualArray(actual[1], np.array([1.0, 2.0], dtype=np.float32))
        self.assertIn("onnx_light_cpu::Abs", used_kernel_names(session))
        self.assertIn("onnx_light_cpu::ReduceSum", used_kernel_names(session))
