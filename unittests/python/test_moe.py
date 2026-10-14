# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime parity and reduced GPT-OSS-style decoder generation with MoE."""

import unittest

import numpy as np
import onnxruntime

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto, helper, numpy_helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import register_kernel_for_session


def _moe_node(inputs, activation="swiglu", k=2, normalize=True):
    attributes = {
        "k": k,
        "activation_type": activation,
        "normalize_routing_weights": int(normalize),
    }
    if activation == "swiglu":
        attributes.update(
            swiglu_fusion=1,
            swiglu_limit=7.0,
            activation_alpha=1.702,
            activation_beta=1.0,
        )
    return helper.make_node("MoE", inputs, ["output"], domain="com.microsoft", **attributes)


def _model(shape, dtype, activation="swiglu", k=2, normalize=True):
    rng = np.random.default_rng(897)
    numpy_dtype = np.float32 if dtype == TensorProto.FLOAT else np.float16
    tokens = int(np.prod(shape[:-1]))
    experts, intermediate, hidden = 3, 5, shape[-1]
    arrays = {
        "input": rng.normal(size=shape).astype(numpy_dtype),
        "router": rng.normal(size=(tokens, experts)).astype(numpy_dtype),
        "fc1": rng.normal(
            size=(experts, intermediate * (2 if activation == "swiglu" else 1), hidden)
        ).astype(numpy_dtype),
        "bias1": rng.normal(
            size=(experts, intermediate * (2 if activation == "swiglu" else 1))
        ).astype(numpy_dtype),
        "fc2": rng.normal(size=(experts, hidden, intermediate)).astype(numpy_dtype),
        "bias2": rng.normal(size=(experts, hidden)).astype(numpy_dtype),
    }
    names = ["input", "router", "fc1", "bias1", "fc2", "bias2"]
    model = helper.make_model(
        helper.make_graph(
            [_moe_node([*names, "", ""], activation, k, normalize)],
            "moe-parity",
            [helper.make_tensor_value_info(name, dtype, arrays[name].shape) for name in names],
            [helper.make_tensor_value_info("output", dtype, shape)],
        ),
        opset_imports=[helper.make_opsetid("", 23), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    return model, arrays


def _ort_run(model, feeds):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    return onnxruntime.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    ).run(None, feeds)


def _light_run(model, feeds):
    evaluator = ReferenceEvaluator(model)
    register_kernel_for_session(evaluator, "com.microsoft", "MoE")
    return evaluator.run(None, feeds)


def _decoder():
    rng = np.random.default_rng(898)
    hidden, intermediate, experts, vocab = 8, 12, 4, 11
    weights = {
        "embedding": rng.normal(scale=0.2, size=(vocab, hidden)).astype(np.float32),
        "router_weight": rng.normal(scale=0.2, size=(hidden, experts)).astype(np.float32),
        "fc1": rng.normal(scale=0.2, size=(experts, 2 * intermediate, hidden)).astype(np.float32),
        "bias1": rng.normal(scale=0.1, size=(experts, 2 * intermediate)).astype(np.float32),
        "fc2": rng.normal(scale=0.2, size=(experts, hidden, intermediate)).astype(np.float32),
        "bias2": rng.normal(scale=0.1, size=(experts, hidden)).astype(np.float32),
        "lm_head": rng.normal(scale=0.2, size=(hidden, vocab)).astype(np.float32),
        "router_shape": np.array([-1, experts], dtype=np.int64),
    }
    nodes = [
        helper.make_node("Gather", ["embedding", "input_ids"], ["hidden"]),
        helper.make_node("MatMul", ["hidden", "router_weight"], ["router_3d"]),
        helper.make_node("Reshape", ["router_3d", "router_shape"], ["router"]),
        _moe_node(["hidden", "router", "fc1", "bias1", "fc2", "bias2", "", ""]),
        helper.make_node("MatMul", ["output", "lm_head"], ["logits"]),
    ]
    return helper.make_model(
        helper.make_graph(
            nodes,
            "reduced-gpt-oss-decoder",
            [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "sequence"])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "sequence", vocab])],
            [numpy_helper.from_array(value, name=name) for name, value in weights.items()],
        ),
        opset_imports=[helper.make_opsetid("", 23), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )


class TestMoE(ExtTestCase):
    def test_routing_activation_normalization_and_dtype_against_ort(self):
        for dtype in (TensorProto.FLOAT, TensorProto.FLOAT16):
            for shape in ((2, 8), (1, 2, 8)):
                for activation in ("relu", "gelu", "silu", "identity", "swiglu"):
                    for k, normalize in ((1, False), (2, True), (3, False)):
                        with self.subTest(dtype=dtype, shape=shape, activation=activation, k=k):
                            model, feeds = _model(shape, dtype, activation, k, normalize)
                            reference = _ort_run(model, feeds)[0]
                            actual = _light_run(model, feeds)[0]
                            self.assertEqual(actual.dtype, reference.dtype)
                            tolerance = 3e-2 if dtype == TensorProto.FLOAT16 else 1e-4
                            np.testing.assert_allclose(
                                actual, reference, rtol=tolerance, atol=tolerance
                            )

    def test_reduced_decoder_generate(self):
        model = _decoder()
        session = ReferenceEvaluator(model)
        register_kernel_for_session(session, "com.microsoft", "MoE")
        prompt = np.array([[2, 3]], dtype=np.int64)
        np.testing.assert_allclose(
            session.run(None, {"input_ids": prompt})[0],
            _ort_run(model, {"input_ids": prompt})[0],
            rtol=1e-4,
            atol=1e-4,
        )
        expected = prompt.copy()
        for _ in range(3):
            logits = _ort_run(model, {"input_ids": expected})[0]
            token = logits[:, -1, :].argmax(axis=-1).reshape(-1, 1)
            expected = np.concatenate((expected, token), axis=1)
        actual = session.generate({"input_ids": prompt}, max_new_tokens=3)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
