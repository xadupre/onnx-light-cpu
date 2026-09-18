# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import unittest

import ml_dtypes
import numpy as np
import onnxruntime as ort
from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import (
    register_kernel_for_session,
    registered_kernels,
    set_kernel_usage_recording,
    used_kernel_names,
)


def _model(dtype, shape, softcap=False):
    nodes = [helper.make_node("Tanh", ["X"], ["Y"])]
    initializers = []
    if softcap:
        initializers = [
            helper.make_tensor("scale", dtype, [], [0.19611613513818404]),
            helper.make_tensor("cap", dtype, [], [20]),
        ]
        nodes = [
            helper.make_node("Mul", ["X", "scale"], ["scaled"]),
            helper.make_node("Div", ["scaled", "cap"], ["uncapped"]),
            helper.make_node("Tanh", ["uncapped"], ["capped"]),
            helper.make_node("Mul", ["capped", "cap"], ["Y"]),
        ]
    return helper.make_model(
        helper.make_graph(
            nodes,
            "tanh",
            [helper.make_tensor_value_info("X", dtype, shape)],
            [helper.make_tensor_value_info("Y", dtype, shape)],
            initializer=initializers,
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )


def _ort_session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


class TestTanh(unittest.TestCase):
    def _run(self, model, x):
        session = ReferenceEvaluator(model)
        register_kernel_for_session(session, "", "Tanh")
        set_kernel_usage_recording(session, True)
        result = session.run(None, {"X": x})[0]
        self.assertIn("onnx_light_cpu::Tanh", used_kernel_names(session))
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.shape, x.shape)
        return result

    def test_registration(self):
        kernel = next(k for k in registered_kernels() if k.op_type == "Tanh")
        self.assertEqual(kernel.domain, "ai.onnx")
        self.assertEqual(set(kernel.types), {"FLOAT", "FLOAT16", "BFLOAT16"})

    def test_onnxruntime_parity(self):
        for dtype, proto in (
            (np.float32, TensorProto.FLOAT),
            (np.float16, TensorProto.FLOAT16),
            (ml_dtypes.bfloat16, TensorProto.BFLOAT16),
        ):
            # ORT's CPU provider has no BF16 Tanh: promote the already-rounded
            # BF16 input and round ORT's FP32 result back to BF16.
            oracle_type = TensorProto.FLOAT if proto == TensorProto.BFLOAT16 else proto
            oracle = _ort_session(_model(oracle_type, [None]))
            tiny = np.nextafter(dtype(0), dtype(1), dtype=dtype)
            values = np.array(
                [
                    -np.inf,
                    -100,
                    -10,
                    -1,
                    -0.25,
                    -tiny,
                    -0.0,
                    0,
                    tiny,
                    0.25,
                    1,
                    10,
                    100,
                    np.inf,
                    np.nan,
                ],
                dtype=dtype,
            )
            values = np.concatenate((values, np.linspace(-9, 9, 1001).astype(dtype)))
            for size in (*range(1, 34), 255, 256, 257, 202048):
                with self.subTest(dtype=dtype, size=size):
                    x = np.resize(values, size)
                    actual = self._run(_model(proto, [None]), x)
                    feeds = x.astype(np.float32) if proto == TensorProto.BFLOAT16 else x
                    expected = oracle.run(None, {"X": feeds})[0].astype(dtype)
                    tolerance = 2e-6 if proto == TensorProto.FLOAT else 8e-3
                    # Some ORT CPU implementations flush FP32 subnormals.
                    # Keep normal-value parity strict and test our subnormal
                    # preservation bit-for-bit below.
                    underflow = np.abs(x.astype(np.float32)) < np.finfo(np.float32).tiny
                    np.testing.assert_allclose(
                        actual[~underflow].astype(np.float32),
                        expected[~underflow].astype(np.float32),
                        rtol=tolerance,
                        atol=0,
                        equal_nan=True,
                    )
                    np.testing.assert_allclose(
                        actual[underflow].astype(np.float32),
                        expected[underflow].astype(np.float32),
                        rtol=0,
                        atol=np.finfo(np.float32).tiny,
                    )
                    finite = np.isfinite(x)
                    np.testing.assert_array_equal(
                        np.signbit(actual[finite]), np.signbit(x[finite])
                    )
                    small = np.abs(x.astype(np.float32)) <= float(tiny)
                    np.testing.assert_array_equal(
                        actual[small].view("u" + str(x.itemsize)),
                        x[small].view("u" + str(x.itemsize)),
                    )

    def test_shapes_and_empty(self):
        for dtype, proto in (
            (np.float32, TensorProto.FLOAT),
            (np.float16, TensorProto.FLOAT16),
            (ml_dtypes.bfloat16, TensorProto.BFLOAT16),
        ):
            for shape in ([], [0], [2, 0, 3], [2, 3, 4]):
                with self.subTest(dtype=dtype, shape=shape):
                    x = np.zeros(shape, dtype=dtype)
                    np.testing.assert_array_equal(self._run(_model(proto, shape), x), x)

    def test_muse_softcap(self):
        for tokens in (1, 16):
            shape = [1, tokens, 202048]
            x = np.random.default_rng(42).uniform(-2000, 2000, shape).astype(np.float32)
            model = _model(TensorProto.FLOAT, shape, softcap=True)
            actual = self._run(model, x)
            expected = _ort_session(model).run(None, {"X": x})[0]
            np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-6)
            self.assertLessEqual(float(np.max(np.abs(actual))), 20)


if __name__ == "__main__":
    unittest.main()
