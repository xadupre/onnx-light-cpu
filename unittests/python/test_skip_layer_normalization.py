# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Checks the fused residual layer-normalization decoder path."""

import unittest

import ml_dtypes
import numpy as np
import onnxruntime

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto, helper, numpy_helper
from onnx_light.onnx.backend import TestMode, collect_test_cases_by_name
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import (
    register_backend_test_cases,
    register_kernel_for_session,
)

_OP = "SkipLayerNormalization"
_PREFIX = "^test_cpu_skip_layer_normalization_"


def _decoder_model(dtype, shape=(1, 2, 4)):
    storage = np.float32 if dtype == TensorProto.FLOAT else np.float16
    weight = np.eye(shape[-1], dtype=storage)
    node = helper.make_node(
        _OP,
        ["input", "skip", "gamma", "beta", "bias"],
        ["normalized", "", "", "residual"],
        domain="com.microsoft",
        epsilon=1.0e-5,
    )
    return helper.make_model(
        helper.make_graph(
            [
                node,
                helper.make_node("MatMul", ["normalized", "weight"], ["decoded"]),
            ],
            "decoder-skip-layer-normalization",
            [
                helper.make_tensor_value_info("input", dtype, shape),
                helper.make_tensor_value_info("skip", dtype, shape),
                helper.make_tensor_value_info("gamma", dtype, [shape[-1]]),
                helper.make_tensor_value_info("beta", dtype, [shape[-1]]),
                helper.make_tensor_value_info("bias", dtype, [shape[-1]]),
            ],
            [
                helper.make_tensor_value_info("decoded", dtype, shape),
                helper.make_tensor_value_info("residual", dtype, shape),
            ],
            [numpy_helper.from_array(weight, name="weight")],
        ),
        opset_imports=[helper.make_opsetid("", 23), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )


def _array(tensor):
    dtype = {
        int(TensorProto.FLOAT): np.float32,
        int(TensorProto.FLOAT16): np.float16,
        int(TensorProto.BFLOAT16): ml_dtypes.bfloat16,
    }[int(tensor.data_type)]
    return np.frombuffer(tensor.raw_data(), dtype=dtype).reshape(tuple(tensor.shape))


class TestSkipLayerNormalization(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        register_backend_test_cases()

    def test_decoder_fixture_matches_onnxruntime(self):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        for dtype, tensor_type in (
            (np.float32, TensorProto.FLOAT),
            (np.float16, TensorProto.FLOAT16),
        ):
            with self.subTest(dtype=dtype):
                model = _decoder_model(tensor_type)
                feeds = {
                    "input": np.arange(8, dtype=dtype).reshape(1, 2, 4),
                    "skip": np.linspace(-1, 1, 8, dtype=dtype).reshape(1, 2, 4),
                    "gamma": np.array([1, 0.5, -1, 2], dtype=dtype),
                    "beta": np.array([0.25, -0.5, 1, -1], dtype=dtype),
                    "bias": np.array([0.125, -0.25, 0.5, -0.75], dtype=dtype),
                }
                expected = onnxruntime.InferenceSession(
                    model.SerializeToString(),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                ).run(None, feeds)
                session = ReferenceEvaluator(model)
                register_kernel_for_session(session, "com.microsoft", _OP)
                actual = session.run(None, feeds)
                tolerance = 2e-3 if dtype == np.float16 else 3e-5
                for output, reference in zip(actual, expected, strict=True):
                    np.testing.assert_allclose(output, reference, rtol=tolerance, atol=tolerance)

    def test_backend_cases_execute(self):
        cases = collect_test_cases_by_name(_PREFIX, mode=TestMode.TEST)
        self.assertEqual(len(cases), 15)
        for case in cases:
            with self.subTest(case=case.name):
                session = ReferenceEvaluator(case.model)
                register_kernel_for_session(session, "com.microsoft", _OP)
                for dataset in case.data_sets:
                    feeds = {
                        value.name: _array(tensor)
                        for value, tensor in zip(
                            case.model.graph.input, dataset.inputs, strict=True
                        )
                    }
                    actual = session.run(None, feeds)
                    for output, expected_tensor in zip(actual, dataset.outputs, strict=True):
                        expected = _array(expected_tensor)
                        tolerance = 2e-3 if expected.dtype == np.float16 else 3e-5
                        np.testing.assert_allclose(
                            output.astype(np.float32),
                            expected.astype(np.float32),
                            rtol=tolerance,
                            atol=tolerance,
                        )


if __name__ == "__main__":
    unittest.main()
