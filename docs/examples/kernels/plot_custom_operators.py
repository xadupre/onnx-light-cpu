"""
Run com.microsoft custom operators
==================================

This example registers the portable ``CDist`` and ``BiasGelu`` CPU kernels and
runs one model containing both operators through onnx-light.
"""

import math

import numpy as np

from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import operator_schema_lookup, register_kernels


def make_model():
    """Builds a model containing both custom operators."""
    graph = helper.make_graph(
        [
            helper.make_node(
                "CDist", ["A", "B"], ["distances"], domain="com.microsoft", metric="euclidean"
            ),
            helper.make_node("BiasGelu", ["X", "bias"], ["activated"], domain="com.microsoft"),
        ],
        "custom-operators",
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [None]),
        ],
        [
            helper.make_tensor_value_info("distances", TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info("activated", TensorProto.FLOAT, [None, None]),
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

register_kernels()
session = ReferenceEvaluator(make_model())
distances, activated = session.run(None, {"A": a, "B": b, "X": x, "bias": bias})

expected_distances = np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))
z = x + bias
expected_activated = (
    0.5 * z * (1.0 + np.vectorize(math.erf, otypes=[np.float32])(z / np.sqrt(np.float32(2.0))))
)
np.testing.assert_allclose(distances, expected_distances, rtol=1e-6, atol=1e-6)
np.testing.assert_allclose(activated, expected_activated, rtol=1e-6, atol=1e-6)

print("Registered custom schemas:", [schema.name for schema in operator_schema_lookup("CDist")])
print("CDist output:\n", distances)
print("BiasGelu output:\n", activated)
