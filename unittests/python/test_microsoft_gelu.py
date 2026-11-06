# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""ORT contrib Gelu and a bounded, offline Qwen3.5-style vision encoder."""

import unittest

import numpy as np
import onnxruntime

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto, helper, numpy_helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_core.graph_builder import GraphBuilder
from onnx_light_cpu import (
    custom_op_schemas,
    operator_schema_lookup,
    register_kernel_for_session,
    register_kernels_global,
    register_operator_support,
    registered_kernels,
    set_kernel_usage_recording,
    used_kernel_names,
)


def _gelu_model(dtype, shape, domain="com.microsoft", **attributes):
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Gelu", ["X"], ["Y"], domain=domain, **attributes)],
            "gelu",
            [helper.make_tensor_value_info("X", dtype, shape)],
            [helper.make_tensor_value_info("Y", dtype, shape)],
        ),
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)],
    )


def _ort(model):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


# Cache the built-in standard Gelu before any test installs CPU kernels.
_STANDARD_INPUT = np.linspace(-5, 5, 65, dtype=np.float32)
_STANDARD_ORACLES = {}
for _mode in ("none", "tanh"):
    _model = _gelu_model(TensorProto.FLOAT, [65], "", approximate=_mode)
    _session = ReferenceEvaluator(_model)
    _STANDARD_ORACLES[_mode] = _session.run(None, {"X": _STANDARD_INPUT})[0]


def _tiny_vision_model(dtype):
    """One vision block, hidden=32, 2x2 merger, patch=4, temporal patch=2.

    The merger follows modelbuilder/builders/qwen.py::make_patch_merger:
    LayerNormalization -> Reshape(-1, 128) -> linear -> contrib Gelu -> linear.
    Patch embedding is the equivalent flattened 3D convolution projection.
    """
    rng = np.random.default_rng(553)
    nodes = []
    weights = {}
    elem_type = TensorProto.FLOAT if dtype == np.float32 else TensorProto.FLOAT16

    def linear(name, x, in_width, out_width):
        weights[f"{name}.weight"] = rng.normal(0, 0.05, (in_width, out_width)).astype(dtype)
        weights[f"{name}.bias"] = rng.normal(0, 0.01, out_width).astype(dtype)
        nodes.extend(
            [
                helper.make_node("MatMul", [x, f"{name}.weight"], [f"{name}.matmul"]),
                helper.make_node("Add", [f"{name}.matmul", f"{name}.bias"], [name]),
            ]
        )
        return name

    def norm(name, x):
        weights[f"{name}.scale"] = np.ones(32, dtype=dtype)
        weights[f"{name}.bias"] = np.zeros(32, dtype=dtype)
        nodes.append(
            helper.make_node(
                "LayerNormalization",
                [x, f"{name}.scale", f"{name}.bias"],
                [name],
                epsilon=1e-6,
            )
        )
        return name

    hidden = linear("patch_embed", "pixel_values", 3 * 2 * 4 * 4, 32)
    normalized = norm("block.norm1", hidden)
    query = linear("block.query", normalized, 32, 32)
    key = linear("block.key", normalized, 32, 32)
    value = linear("block.value", normalized, 32, 32)
    weights["attention.scale"] = np.array(32**-0.5, dtype=dtype)
    nodes.extend(
        [
            helper.make_node("Transpose", [key], ["key_t"], perm=[1, 0]),
            helper.make_node("MatMul", [query, "key_t"], ["scores"]),
            helper.make_node("Mul", ["scores", "attention.scale"], ["scaled_scores"]),
            helper.make_node("Softmax", ["scaled_scores"], ["probabilities"], axis=-1),
            helper.make_node("MatMul", ["probabilities", value], ["context"]),
        ]
    )
    attention = linear("block.proj", "context", 32, 32)
    nodes.append(helper.make_node("Add", [hidden, attention], ["residual"]))
    normalized = norm("block.norm2", "residual")
    mlp = linear("block.fc1", normalized, 32, 64)
    nodes.append(helper.make_node("Gelu", [mlp], ["block.gelu"], domain="com.microsoft"))
    mlp = linear("block.fc2", "block.gelu", 64, 32)
    nodes.append(helper.make_node("Add", ["residual", mlp], ["block_output"]))
    normalized = norm("merger.norm", "block_output")
    weights["merge_shape"] = np.array([-1, 128], dtype=np.int64)
    nodes.append(helper.make_node("Reshape", [normalized, "merge_shape"], ["merged"]))
    merger_hidden = linear("merger.fc1", "merged", 128, 128)
    nodes.append(
        helper.make_node("Gelu", [merger_hidden], ["merger.gelu"], domain="com.microsoft")
    )
    output = linear("image_features", "merger.gelu", 128, 128)
    return helper.make_model(
        helper.make_graph(
            nodes,
            "qwen35_tiny_vision",
            [helper.make_tensor_value_info("pixel_values", elem_type, ["patches", 96])],
            [helper.make_tensor_value_info(output, elem_type, ["merged_patches", 128])],
            [numpy_helper.from_array(value, name) for name, value in weights.items()],
        ),
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)],
    )


class TestMicrosoftGelu(ExtTestCase):
    def test_schema_and_registration(self):
        schema = custom_op_schemas("Gelu")[0]
        self.assertEqual(schema.since_version, 1)
        entries = [
            r for r in registered_kernels() if r.domain == "com.microsoft" and r.op_type == "Gelu"
        ]
        self.assertEqual(len(entries), 1)
        self.assertEqual(set(entries[0].types), {"FLOAT", "DOUBLE", "FLOAT16", "BFLOAT16"})
        self.assertFalse(
            any(r.domain == "ai.onnx" and r.op_type == "Gelu" for r in registered_kernels())
        )
        model = _gelu_model(TensorProto.FLOAT16, ["tokens", 128])
        model.graph.output[0].type.tensor_type.ClearField("shape")
        register_operator_support()
        builder = GraphBuilder(model, schema_lookup=operator_schema_lookup)
        output = builder.to_onnx().graph.output[0]
        self.assertEqual(
            [dim.dim_param or dim.dim_value for dim in output.type.tensor_type.shape.dim],
            ["tokens", 128],
        )

    def test_ort_parity(self):
        register_kernels_global()
        for dtype, elem_type, tolerance in (
            (np.float32, TensorProto.FLOAT, 2e-6),
            (np.float16, TensorProto.FLOAT16, 2e-3),
        ):
            for shape in ([], [0, 32], [7], [8], [9], [15], [16], [17], [2, 32], [2048, 128]):
                with self.subTest(dtype=dtype, shape=shape):
                    x = np.linspace(-8, 8, int(np.prod(shape)), dtype=dtype).reshape(shape)
                    model = _gelu_model(elem_type, shape)
                    session = ReferenceEvaluator(model)
                    set_kernel_usage_recording(session, True)
                    actual = session.run(None, {"X": x})[0]
                    expected = _ort(model).run(None, {"X": x})[0]
                    self.assertEqual(actual.dtype, x.dtype)
                    self.assertEqual(actual.shape, x.shape)
                    self.assertEqualArray(expected, actual, atol=tolerance, rtol=tolerance)
                    self.assertIn("onnx_light_cpu::MicrosoftGelu", used_kernel_names(session))

    def test_nonfinite_and_signed_zero(self):
        register_kernels_global()
        x = np.array([-np.inf, -20, -0.0, 0.0, 20, np.inf, np.nan], dtype=np.float32)
        model = _gelu_model(TensorProto.FLOAT, [7])
        actual = ReferenceEvaluator(model).run(None, {"X": x})[0]
        expected = _ort(model).run(None, {"X": x})[0]
        np.testing.assert_allclose(actual, expected, equal_nan=True)
        self.assertEqual(np.signbit(actual[2]), np.signbit(expected[2]))

    def test_standard_domain_unchanged(self):
        register_kernels_global()
        for mode, expected in _STANDARD_ORACLES.items():
            model = _gelu_model(TensorProto.FLOAT, [65], "", approximate=mode)
            session = ReferenceEvaluator(model)
            set_kernel_usage_recording(session, True)
            actual = session.run(None, {"X": _STANDARD_INPUT})[0]
            self.assertEqualArray(expected, actual, atol=0, rtol=0)
            self.assertNotIn("onnx_light_cpu::MicrosoftGelu", used_kernel_names(session))

    def test_invalid_attributes_and_types(self):
        for model, x in (
            (_gelu_model(TensorProto.FLOAT, [1], approximate="tanh"), np.ones(1, np.float32)),
            (_gelu_model(TensorProto.INT32, [1]), np.ones(1, np.int32)),
        ):
            session = ReferenceEvaluator(model)
            register_kernel_for_session(session, "com.microsoft", "Gelu")
            with self.assertRaises((ValueError, RuntimeError)):
                session.run(None, {"X": x})

    def test_tiny_qwen35_vision_image_and_video(self):
        register_kernels_global()
        for dtype in (np.float32, np.float16):
            model = _tiny_vision_model(dtype)
            oracle = _ort(model)
            session = ReferenceEvaluator(model)
            set_kernel_usage_recording(session, True)
            for patches in (4, 8):
                with self.subTest(dtype=dtype, patches=patches):
                    x = np.random.default_rng(1111).normal(size=(patches, 96)).astype(dtype)
                    actual = session.run(None, {"pixel_values": x})[0]
                    expected = oracle.run(None, {"pixel_values": x})[0]
                    self.assertEqual(actual.shape, (patches // 4, 128))
                    self.assertEqualArray(expected, actual, atol=3e-3, rtol=3e-3)
                    self.assertIn("onnx_light_cpu::MicrosoftGelu", used_kernel_names(session))


if __name__ == "__main__":
    unittest.main()
