# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Offline random INT4 MoE decoder parity, including native cached generation.

The C++ backend corpus supplies packed weights and independently dequantized
floating weights. Python only assembles graphs and drives decoding; both the
quantized decoder and its floating MoE oracle execute native kernels.
"""

import unittest

import numpy as np

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto, helper, numpy_helper
from onnx_light.onnx.backend import TestMode, collect_test_cases_by_name
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import (
    register_backend_test_cases,
    register_kernels_global,
    registered_kernels,
)


def _array(tensor):
    dtype = {TensorProto.FLOAT: np.float32, TensorProto.UINT8: np.uint8}[int(tensor.data_type)]
    return np.frombuffer(tensor.raw_data(), dtype=dtype).reshape(tuple(tensor.shape)).copy()


def _decoder(moe_node, expert_weights, hidden, experts):
    """Builds one causal self-attention block followed by a residual MoE."""
    rng = np.random.default_rng(20261017)
    vocab = 19
    weights = {
        **expert_weights,
        "embedding": rng.normal(scale=0.3, size=(vocab, hidden)).astype(np.float32),
        "position_embedding": rng.normal(scale=0.1, size=(32, hidden)).astype(np.float32),
        "query_weight": rng.normal(scale=0.2, size=(hidden, hidden)).astype(np.float32),
        "key_weight": rng.normal(scale=0.2, size=(hidden, hidden)).astype(np.float32),
        "value_weight": rng.normal(scale=0.2, size=(hidden, hidden)).astype(np.float32),
        "attention_weight": rng.normal(scale=0.2, size=(hidden, hidden)).astype(np.float32),
        "router_weight": rng.normal(scale=0.2, size=(hidden, experts)).astype(np.float32),
        "lm_head": rng.normal(scale=0.3, size=(hidden, vocab)).astype(np.float32),
        "router_shape": np.array([-1, experts], dtype=np.int64),
        "sequence_axis": np.array(1, dtype=np.int64),
        "zero": np.array(0, dtype=np.int64),
        "one": np.array(1, dtype=np.int64),
        "query_axes": np.array([0, 2], dtype=np.int64),
        "attention_scale": np.array(hidden**-0.5, dtype=np.float32),
        "masked_score": np.array(-10000, dtype=np.float32),
    }
    nodes = [
        helper.make_node("Shape", ["input_ids"], ["input_shape"]),
        helper.make_node("Gather", ["input_shape", "sequence_axis"], ["sequence_length"]),
        helper.make_node("Shape", ["past_key"], ["past_shape"]),
        helper.make_node("Gather", ["past_shape", "sequence_axis"], ["past_length"]),
        helper.make_node("Range", ["zero", "sequence_length", "one"], ["local_positions"]),
        helper.make_node("Add", ["local_positions", "past_length"], ["positions"]),
        helper.make_node("Gather", ["embedding", "input_ids"], ["token_embedding"]),
        helper.make_node("Gather", ["position_embedding", "positions"], ["position_values"]),
        helper.make_node("Add", ["token_embedding", "position_values"], ["embedded"]),
        helper.make_node("MatMul", ["embedded", "query_weight"], ["query"]),
        helper.make_node("MatMul", ["embedded", "key_weight"], ["key"]),
        helper.make_node("MatMul", ["embedded", "value_weight"], ["value"]),
        helper.make_node("Concat", ["past_key", "key"], ["present_key"], axis=1),
        helper.make_node("Concat", ["past_value", "value"], ["present_value"], axis=1),
        helper.make_node("Transpose", ["present_key"], ["key_transposed"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["query", "key_transposed"], ["scores"]),
        helper.make_node("Mul", ["scores", "attention_scale"], ["scaled_scores"]),
        helper.make_node("Add", ["past_length", "sequence_length"], ["total_length"]),
        helper.make_node("Range", ["zero", "total_length", "one"], ["key_positions"]),
        helper.make_node("Unsqueeze", ["positions", "query_axes"], ["query_positions"]),
        helper.make_node("LessOrEqual", ["key_positions", "query_positions"], ["causal_mask"]),
        helper.make_node(
            "Where", ["causal_mask", "scaled_scores", "masked_score"], ["masked_scores"]
        ),
        helper.make_node("Softmax", ["masked_scores"], ["attention_probabilities"], axis=-1),
        helper.make_node("MatMul", ["attention_probabilities", "present_value"], ["context"]),
        helper.make_node("MatMul", ["context", "attention_weight"], ["attention_output"]),
        helper.make_node("Add", ["embedded", "attention_output"], ["hidden"]),
        helper.make_node("MatMul", ["hidden", "router_weight"], ["router_3d"]),
        helper.make_node("Reshape", ["router_3d", "router_shape"], ["router"]),
        moe_node,
        helper.make_node("Add", ["hidden", "moe_output"], ["decoder_output"]),
        helper.make_node("MatMul", ["decoder_output", "lm_head"], ["logits"]),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "offline-random-int4-moe-decoder",
            [
                helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["B", "S"]),
                helper.make_tensor_value_info("past_key", TensorProto.FLOAT, ["B", "P", hidden]),
                helper.make_tensor_value_info(
                    "past_value", TensorProto.FLOAT, ["B", "P", hidden]
                ),
            ],
            [
                helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["B", "S", vocab]),
                helper.make_tensor_value_info(
                    "present_key", TensorProto.FLOAT, ["B", "T", hidden]
                ),
                helper.make_tensor_value_info(
                    "present_value", TensorProto.FLOAT, ["B", "T", hidden]
                ),
            ],
            [numpy_helper.from_array(value, name=name) for name, value in weights.items()],
        ),
        opset_imports=[helper.make_opsetid("", 23), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    for kind in ("key", "value"):
        binding = model.graph.persistent_bindings.add()
        binding.input_name = f"past_{kind}"
        binding.output_name = f"present_{kind}"
    return model


def _feeds(ids, hidden):
    return {
        "input_ids": ids,
        "past_key": np.empty((ids.shape[0], 0, hidden), dtype=np.float32),
        "past_value": np.empty((ids.shape[0], 0, hidden), dtype=np.float32),
    }


class TestQMoEDecoder(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        register_backend_test_cases()
        register_kernels_global()

    def _sessions(self):
        sessions = []
        dimensions = []
        for op, name in (
            ("QMoE", "test_cpu_qmoe_decoder_random_int4_float32"),
            ("MoE", "test_cpu_moe_decoder_random_int4_float32"),
        ):
            cases = collect_test_cases_by_name(f"^{name}$", mode=TestMode.TEST)
            self.assertEqual(len(cases), 1, f"Missing C++ decoder fixture {name}")
            case = cases[0]
            arrays = {
                info.name: _array(tensor)
                for info, tensor in zip(
                    case.model.graph.input, case.data_sets[0].inputs, strict=True
                )
            }
            source = case.model.graph.node[0]
            self.assertEqual(source.op_type, op)
            if op == "QMoE":
                attributes = {
                    attr.name: helper.get_attribute_value(attr) for attr in source.attribute
                }
                self.assertEqual(attributes["expert_weight_bits"], 4)
                self.assertEqual(arrays[source.input[2]].dtype, np.uint8)
            node = type(source)()
            node.CopyFrom(source)
            original_inputs = list(node.input)
            hidden = arrays[original_inputs[0]].shape[-1]
            experts = arrays[original_inputs[1]].shape[-1]
            dimensions.append((hidden, experts))
            node.input[0] = "hidden"
            node.input[1] = "router"
            node.output[0] = "moe_output"
            expert_weights = {name: arrays[name] for name in original_inputs[2:] if name}
            sessions.append(ReferenceEvaluator(_decoder(node, expert_weights, hidden, experts)))
        self.assertEqual(dimensions[0], dimensions[1])
        return *sessions, dimensions[0][0]

    def test_prefill_cached_decode_and_generation(self):
        quantized, reference, hidden = self._sessions()
        quantized.set_kernel_usage_enabled(True)
        reference.set_kernel_usage_enabled(True)
        prompt = np.array([[2, 5, 3], [4, 1, 7]], dtype=np.int64)
        prompt_snapshot = prompt.copy()
        original = _feeds(prompt, hidden)
        actual_feeds, expected_feeds = dict(original), dict(original)
        prefix = prompt.copy()
        for step in range(5):
            actual = quantized.run(None, actual_feeds)
            expected = reference.run(None, expected_feeds)
            for got, want in zip(actual, expected, strict=True):
                np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
            self.assertEqual(actual[0].shape[:2], (2, 3 if step == 0 else 1))
            if step == 0:
                for length in range(1, prompt.shape[1]):
                    causal_logits = reference.run(None, _feeds(prompt[:, :length], hidden))[0]
                    np.testing.assert_allclose(
                        actual[0][:, :length, :], causal_logits, rtol=2e-4, atol=2e-4
                    )
            for cache in actual[1:]:
                self.assertEqual(cache.shape, (2, prompt.shape[1] + step, hidden))
            # Full-prefix floating evaluation independently checks causal masking,
            # positional offsets, and consumption of the accumulated KV cache.
            full_logits = reference.run(None, _feeds(prefix, hidden))[0]
            np.testing.assert_allclose(
                actual[0][:, -1, :], full_logits[:, -1, :], rtol=2e-4, atol=2e-4
            )
            token = expected[0][:, -1, :].argmax(axis=-1).reshape(-1, 1)
            np.testing.assert_array_equal(actual[0][:, -1, :].argmax(axis=-1), token[:, 0])
            prefix = np.concatenate((prefix, token), axis=1)
            actual_feeds = {
                "input_ids": token,
                "past_key": actual[1],
                "past_value": actual[2],
            }
            expected_feeds = {
                "input_ids": token,
                "past_key": expected[1],
                "past_value": expected[2],
            }
        for _ in range(2):
            np.testing.assert_array_equal(quantized.generate(original, max_new_tokens=5), prefix)
            np.testing.assert_array_equal(reference.generate(original, max_new_tokens=5), prefix)
        np.testing.assert_array_equal(original["input_ids"], prompt_snapshot)
        self.assertEqual(original["past_key"].shape, (2, 0, hidden))
        self.assertEqual(original["past_value"].shape, (2, 0, hidden))
        kernel_names = {record.op_type: record.kernel_name for record in registered_kernels()}
        self.assertIn(kernel_names["QMoE"], quantized.get_kernel_usage())
        self.assertNotIn(kernel_names["QMoE"], reference.get_kernel_usage())
        self.assertIn(kernel_names["MoE"], reference.get_kernel_usage())


if __name__ == "__main__":
    unittest.main()
