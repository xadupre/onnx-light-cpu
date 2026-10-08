# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare both RotaryEmbedding domains with ONNX Runtime."""

import unittest

import numpy as np
import onnxruntime

from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_core.graph_builder import GraphBuilder
from onnx_light.ext_test_case import ExtTestCase
from onnx_light_cpu import (
    custom_op_schemas,
    operator_schema_lookup,
    operator_support,
    register_kernel_for_session,
    register_operator_support,
)


def _model(domain, x_shape, cache_shape, positions_shape, interleaved, rotary_dim, heads):
    microsoft = domain == "com.microsoft"
    names = ["X", "positions", "cos", "sin"] if microsoft else ["X", "cos", "sin", "positions"]
    if positions_shape is None:
        names.pop()
    shapes = {
        "X": x_shape,
        "cos": cache_shape,
        "sin": cache_shape,
        "positions": positions_shape,
    }
    inputs = [
        helper.make_tensor_value_info(
            name,
            TensorProto.INT64 if name == "positions" else TensorProto.FLOAT,
            shapes[name],
        )
        for name in names
    ]
    attrs = {"interleaved": int(interleaved), "rotary_embedding_dim": rotary_dim}
    if heads:
        attrs["num_heads"] = heads
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("RotaryEmbedding", names, ["Y"], domain=domain, **attrs)],
            "rotary-embedding",
            inputs,
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, x_shape)],
        ),
        opset_imports=[helper.make_opsetid("", 23), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )


class TestRotaryEmbedding(ExtTestCase):
    def test_schemas_and_symbolic_shapes(self):
        register_operator_support()
        for domain in ("", "com.microsoft"):
            with self.subTest(domain=domain):
                schemas = operator_schema_lookup("RotaryEmbedding")
                self.assertTrue(any(schema.domain == (domain or "ai.onnx") for schema in schemas))
                self.assertEqual(
                    len(
                        [
                            support
                            for support in operator_support()
                            if support.op_type == "RotaryEmbedding"
                            and support.domain == (domain or "ai.onnx")
                        ]
                    ),
                    1,
                )
                builder = GraphBuilder(
                    _model(domain, ("B", 2, "S", 4), (7, 2), ("B", "S"), False, 0, 0),
                    schema_lookup=operator_schema_lookup,
                )
                result = builder.to_onnx("model").graph.output[0]
                self.assertEqual(
                    [
                        dimension.dim_param or dimension.dim_value
                        for dimension in result.type.tensor_type.shape.dim
                    ],
                    ["B", 2, "S", 4],
                )
        self.assertEqual(len(custom_op_schemas("RotaryEmbedding")), 1)

    def test_symbolic_shape_accepts_partially_unknown_types(self):
        model = _model("", (1, 2, 4), (3, 2), (1, 2), False, 0, 1)
        model.graph.input[0].type.tensor_type.elem_type = TensorProto.UNDEFINED
        builder = GraphBuilder(model, schema_lookup=operator_schema_lookup)
        result = builder.to_onnx("model").graph.output[0]
        self.assertEqual(
            [dimension.dim_value for dimension in result.type.tensor_type.shape.dim],
            [1, 2, 4],
        )

    def test_ort_parity(self):
        cases = (
            ("", (2, 3, 8), (7, 2), (2, 3), False, 4, 2),
            ("", (2, 2, 3, 6), (2, 3, 2), None, True, 4, 0),
            ("", (1, 2, 3, 4), (7, 2), (1, 3), True, 0, 0),
            ("com.microsoft", (2, 3, 8), (7, 2), (2, 3), False, 4, 2),
            ("com.microsoft", (1, 2, 8), (7, 2), (1, 2), True, 2, 2),
            ("com.microsoft", (1, 2, 3, 4), (7, 2), (1,), True, 0, 0),
        )
        for domain, x_shape, cache_shape, positions_shape, interleaved, dim, heads in cases:
            with self.subTest(domain=domain, shape=x_shape, positions=positions_shape):
                model = _model(
                    domain, x_shape, cache_shape, positions_shape, interleaved, dim, heads
                )
                feeds = {
                    "X": np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape) / 10,
                    "cos": np.full(cache_shape, 0.6, dtype=np.float32),
                    "sin": np.full(cache_shape, 0.8, dtype=np.float32),
                }
                if positions_shape == (1,):
                    feeds["positions"] = np.array([1], dtype=np.int64)
                elif positions_shape is not None:
                    feeds["positions"] = (
                        np.arange(np.prod(positions_shape), dtype=np.int64).reshape(
                            positions_shape
                        )
                        % cache_shape[0]
                    )
                session = ReferenceEvaluator(model)
                register_kernel_for_session(session, domain, "RotaryEmbedding")
                actual = session.run(None, feeds)[0]
                expected = onnxruntime.InferenceSession(
                    model.SerializeToString(),
                    providers=["CPUExecutionProvider"],
                ).run(None, feeds)[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
