# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Checks packed INT4 MatMulNBits backend cases against ONNX Runtime."""

import unittest

import numpy as np
import onnxruntime

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto
from onnx_light.onnx.backend import TestMode, collect_test_cases
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import (
    register_backend_test_cases,
    register_kernel_for_session,
)


def _array(tensor):
    """Decodes a backend tensor in its stored precision."""
    dtype = {
        int(TensorProto.FLOAT): np.float32,
        int(TensorProto.UINT8): np.uint8,
    }[int(tensor.data_type)]
    return np.frombuffer(tensor.raw_data(), dtype=dtype).reshape(tuple(tensor.shape))


class TestMatMulNBits(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        register_backend_test_cases()

    def test_backend_cases_match_onnxruntime(self):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        cases = [
            case
            for case in collect_test_cases("MatMulNBits", mode=TestMode.TEST)
            if case.name.startswith("test_cpu_matmulnbits_")
        ]
        self.assertEqual(len(cases), 2)
        for case in cases:
            with self.subTest(case=case.name):
                model = type(case.model)()
                model.ParseFromString(case.model.SerializeToString())
                model.ir_version = 10
                oracle = onnxruntime.InferenceSession(
                    model.SerializeToString(),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                session = ReferenceEvaluator(model)
                register_kernel_for_session(session, "com.microsoft", "MatMulNBits")
                for dataset in case.data_sets:
                    feeds = {
                        value.name: _array(tensor)
                        for value, tensor in zip(model.graph.input, dataset.inputs, strict=True)
                    }
                    expected = oracle.run(None, feeds)
                    actual = session.run(None, feeds)
                    for output, reference in zip(actual, expected, strict=True):
                        np.testing.assert_allclose(output, reference, rtol=3e-5, atol=3e-5)


if __name__ == "__main__":
    unittest.main()
