"""
Run com.microsoft custom operators
==================================

This example registers the portable ``CDist``, ``BiasGelu``, ``MatMulNBits``,
and ``LinearAttention`` CPU kernels and runs one model containing the four
operators through onnx-light.
"""

# sphinx_gallery_thumbnail_path = "_static/gallery_thumbnails/custom_operators.png"

import math

import numpy as np

from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import operator_schema_lookup, register_kernels


def make_model():
    """Builds a model containing the custom operators."""
    graph = helper.make_graph(
        [
            helper.make_node(
                "CDist", ["A", "B"], ["distances"], domain="com.microsoft", metric="euclidean"
            ),
            helper.make_node("BiasGelu", ["X", "bias"], ["activated"], domain="com.microsoft"),
            helper.make_node(
                "MatMulNBits",
                ["M", "packed_weights", "scales"],
                ["projection"],
                domain="com.microsoft",
                K=32,
                N=2,
                bits=4,
                block_size=32,
                accuracy_level=4,
            ),
            helper.make_node(
                "LinearAttention",
                ["Q", "K", "V"],
                ["attention", "state"],
                domain="com.microsoft",
                update_rule="linear",
                q_num_heads=1,
                kv_num_heads=1,
                scale=1.0,
            ),
        ],
        "custom-operators",
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [None]),
            helper.make_tensor_value_info("M", TensorProto.FLOAT, [None, 32]),
            helper.make_tensor_value_info("packed_weights", TensorProto.UINT8, [2, 1, 16]),
            helper.make_tensor_value_info("scales", TensorProto.FLOAT, [2, 1]),
            helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, 2, 2]),
            helper.make_tensor_value_info("K", TensorProto.FLOAT, [1, 2, 2]),
            helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, 2, 1]),
        ],
        [
            helper.make_tensor_value_info("distances", TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info("activated", TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info("projection", TensorProto.FLOAT, [None, 2]),
            helper.make_tensor_value_info("attention", TensorProto.FLOAT, [1, 2, 1]),
            helper.make_tensor_value_info("state", TensorProto.FLOAT, [1, 1, 2, 1]),
        ],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 20),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=13,
    )


a = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
b = np.array([[1.0, 1.0], [-1.0, 2.0], [2.0, 2.0]], dtype=np.float32)
x = np.array([[-2.0, -1.0, 0.0], [0.5, 1.0, 3.0]], dtype=np.float32)
bias = np.array([0.25, -0.5, 1.0], dtype=np.float32)
q = np.array([[[1.0, 2.0], [2.0, 1.0]]], dtype=np.float32)
k = np.array([[[3.0, 4.0], [1.0, 2.0]]], dtype=np.float32)
v = np.array([[[2.0], [3.0]]], dtype=np.float32)
m = np.ones((1, 32), dtype=np.float32)
packed_weights = np.concatenate(
    [np.full(16, 0x88, dtype=np.uint8), np.full(16, 0x99, dtype=np.uint8)]
).reshape(2, 1, 16)
scales = np.array([[1.0], [0.5]], dtype=np.float32)

register_kernels()
session = ReferenceEvaluator(make_model())
distances, activated, projection, attention, state = session.run(
    None,
    {
        "A": a,
        "B": b,
        "X": x,
        "bias": bias,
        "M": m,
        "packed_weights": packed_weights,
        "scales": scales,
        "Q": q,
        "K": k,
        "V": v,
    },
)

expected_distances = np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))
z = x + bias
expected_activated = (
    0.5 * z * (1.0 + np.vectorize(math.erf, otypes=[np.float32])(z / np.sqrt(np.float32(2.0))))
)
np.testing.assert_allclose(distances, expected_distances, rtol=1e-6, atol=1e-6)
np.testing.assert_allclose(activated, expected_activated, rtol=1e-6, atol=1e-5)
np.testing.assert_allclose(projection, np.array([[0.0, 16.0]], dtype=np.float32))
np.testing.assert_allclose(attention, np.array([[[22.0], [32.0]]], dtype=np.float32))
np.testing.assert_allclose(state, np.array([[[[9.0], [14.0]]]], dtype=np.float32))

print(
    "Registered custom schemas:",
    [
        op_type
        for op_type in ("CDist", "BiasGelu", "MatMulNBits", "LinearAttention")
        if operator_schema_lookup(op_type)
    ],
)
print("CDist output:\n", distances)
print("BiasGelu output:\n", activated)
print("MatMulNBits output:\n", projection)
print("LinearAttention output:\n", attention)
