# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Experimental schema, GraphBuilder, registration, and fixed-value integration tests.

The native backend corpus supplies the independent builtin-operator numerical
reference. These Python checks use exact, fixed expected tensors instead of a
NumPy implementation of the normalization kernel.
"""

import subprocess
import sys
import unittest

import ml_dtypes
import numpy as np

from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_core.graph_builder import GraphBuilder
from onnx_light_cpu import (
    custom_op_schemas,
    experimental_op_schemas,
    operator_schema_lookup,
    operator_support,
    register_kernel_for_session,
    register_kernel_global,
    register_kernels,
    register_kernels_for_session,
    register_kernels_global,
    register_operator_support,
)

_OP = "SimplifiedLayerNormalization"
_TYPES = (
    (TensorProto.FLOAT, np.float32),
    (TensorProto.FLOAT16, np.float16),
    (TensorProto.DOUBLE, np.float64),
    (TensorProto.BFLOAT16, ml_dtypes.bfloat16),
)


def _model(
    x_type=TensorProto.FLOAT,
    scale_type=TensorProto.FLOAT,
    x_shape=("B", 3, 4),
    scale_shape=(1, 4),
    axis=1,
    stash_type=1,
    statistics="inv_std_var",
    domain="",
    epsilon=1.0e-5,
):
    outputs = ["Y"] if statistics is None else ["Y", statistics]
    graph_outputs = [helper.make_tensor_value_info("Y", scale_type, None)]
    if statistics:
        graph_outputs.append(helper.make_tensor_value_info(statistics, stash_type, None))
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    _OP,
                    ["X", "Scale"],
                    outputs,
                    domain=domain,
                    axis=axis,
                    stash_type=stash_type,
                    epsilon=epsilon,
                )
            ],
            "experimental-normalization",
            [
                helper.make_tensor_value_info("X", x_type, list(x_shape)),
                helper.make_tensor_value_info("Scale", scale_type, list(scale_shape)),
            ],
            graph_outputs,
        ),
        opset_imports=[helper.make_opsetid(domain, 23)],
        ir_version=13,
    )


def _shape(value_info):
    return [
        dim.dim_param if dim.dim_param else dim.dim_value
        for dim in value_info.type.tensor_type.shape.dim
    ]


def _check_registration_entry(entry):
    if entry == "support":
        register_operator_support()
    elif entry == "legacy":
        register_kernels()
    elif entry == "single-global":
        register_kernel_global("", _OP)
    elif entry == "all-global":
        register_kernels_global()
    else:
        session = ReferenceEvaluator(_model(x_shape=(2, 2), scale_shape=(2, 1)))
        if entry == "single-session":
            register_kernel_for_session(session, "", _OP)
        else:
            register_kernels_for_session(session)
    builder = GraphBuilder(_model(), schema_lookup=operator_schema_lookup)
    model = builder.to_onnx("model")
    assert _shape(model.graph.output[0]) == ["B", 3, 4]
    assert _shape(model.graph.output[1]) == ["B", 1, 1]


class TestSimplifiedLayerNormalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_operator_support()

    def test_separate_experimental_schema_provider(self):
        schemas = experimental_op_schemas(_OP)
        self.assertIsInstance(schemas, tuple)
        self.assertEqual(len(schemas), 1)
        schema = schemas[0]
        self.assertEqual((schema.name, schema.domain, schema.since_version), (_OP, "ai.onnx", 1))
        self.assertIn("Experimental", schema.doc)
        self.assertFalse(schema.has_function_implementation)
        self.assertEqual((schema.min_output, schema.max_output), (1, 2))
        self.assertEqual([param.type for param in schema.inputs], ["T", "V"])
        self.assertEqual([param.type for param in schema.outputs], ["V", "U"])
        self.assertEqual(experimental_op_schemas(_OP, init_doc=False)[0].doc, "")
        self.assertEqual(experimental_op_schemas("NotAnOperator"), ())
        self.assertEqual(custom_op_schemas(_OP), ())
        self.assertTrue(all(schema.domain == "com.microsoft" for schema in custom_op_schemas()))
        self.assertTrue(operator_schema_lookup("Add"))
        self.assertTrue(operator_schema_lookup("CDist"))
        self.assertTrue(operator_schema_lookup(_OP))
        records = [record for record in operator_support() if record.op_type == _OP]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].domain, "ai.onnx")
        self.assertEqual(records[0].fusion_patterns, ())
        self.assertFalse(records[0].has_gradient)

    def test_loads_symbolic_graph_and_infers_unknown_output_shapes(self):
        for domain in ("", "ai.onnx"):
            for stash_type in (TensorProto.FLOAT, TensorProto.DOUBLE):
                with self.subTest(domain=domain, stash_type=stash_type):
                    builder = GraphBuilder(
                        _model(
                            TensorProto.DOUBLE,
                            TensorProto.FLOAT16,
                            scale_shape=("B", 1, 4),
                            axis=-2,
                            stash_type=stash_type,
                            domain=domain,
                        ),
                        schema_lookup=operator_schema_lookup,
                    )
                    model = builder.to_onnx("model")
                    y, statistics = model.graph.output
                    self.assertEqual(_shape(y), ["B", 3, 4])
                    self.assertEqual(y.type.tensor_type.elem_type, TensorProto.FLOAT16)
                    self.assertEqual(_shape(statistics), ["B", 1, 1])
                    self.assertEqual(statistics.type.tensor_type.elem_type, stash_type)
                    self.assertEqual(model.graph.node[0].op_type, _OP)

    def test_incremental_graph_all_independent_type_pairs(self):
        for x_type, _ in _TYPES:
            for scale_type, _ in _TYPES:
                with self.subTest(x_type=x_type, scale_type=scale_type):
                    builder = GraphBuilder("incremental", schema_lookup=operator_schema_lookup)
                    builder.set_opset_version("", 23)
                    builder.make_input("X", x_type, ["B", "S", 4])
                    builder.make_input("Scale", scale_type, ["B", 1, 4])
                    outputs = builder.make_node(
                        _OP,
                        ["X", "Scale"],
                        outputs=["Y", "inv_std_var"],
                        attributes={"axis": 1, "stash_type": 11},
                    )
                    for name in outputs:
                        builder.make_output(name)
                    y, statistics = builder.to_onnx("model").graph.output
                    self.assertEqual(_shape(y), ["B", "S", 4])
                    self.assertEqual(y.type.tensor_type.elem_type, scale_type)
                    self.assertEqual(_shape(statistics), ["B", 1, 1])
                    self.assertEqual(statistics.type.tensor_type.elem_type, TensorProto.DOUBLE)

    def test_optional_statistics_and_scalar_scale(self):
        for statistics in (None, ""):
            builder = GraphBuilder(
                _model(scale_shape=(), statistics=statistics),
                schema_lookup=operator_schema_lookup,
            )
            model = builder.to_onnx("model")
            self.assertEqual(len(model.graph.output), 1)
            self.assertEqual(_shape(model.graph.output[0]), ["B", 3, 4])
            self.assertEqual(model.graph.node[0].output[0], "Y")
            # GraphBuilder assigns a name to an anonymous optional output slot.
            self.assertEqual(len(model.graph.node[0].output), 1 if statistics is None else 2)

    def test_rejects_invalid_contract(self):
        for arguments in (
            {"stash_type": 10},
            {"axis": 3},
            {"axis": -4},
            {"scale_shape": (5,)},
            {"x_shape": (2, 1), "scale_shape": (3,)},
            {"x_shape": (2, 0), "scale_shape": ()},
            {"x_type": TensorProto.INT32},
            {"scale_type": TensorProto.INT64},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                GraphBuilder(_model(**arguments), schema_lookup=operator_schema_lookup)

    def test_fixed_outputs_all_type_pairs_and_statistics_dtypes(self):
        for x_type, x_dtype in _TYPES:
            for scale_type, scale_dtype in _TYPES:
                for stash_type, stash_dtype in (
                    (TensorProto.FLOAT, np.float32),
                    (TensorProto.DOUBLE, np.float64),
                ):
                    with self.subTest(
                        x_type=x_type, scale_type=scale_type, stash_type=stash_type
                    ):
                        session = ReferenceEvaluator(
                            _model(
                                x_type,
                                scale_type,
                                x_shape=(2, 2),
                                scale_shape=(2, 1),
                                epsilon=0.0,
                                stash_type=stash_type,
                            )
                        )
                        register_kernel_for_session(session, "", _OP)
                        y, statistics = session.run(
                            None,
                            {
                                "X": np.array([[2, -2], [2, -2]], dtype=x_dtype),
                                "Scale": np.array([[1], [2]], dtype=scale_dtype),
                            },
                        )
                        self.assertEqual(y.dtype, np.dtype(scale_dtype))
                        self.assertEqual(statistics.dtype, np.dtype(stash_dtype))
                        np.testing.assert_array_equal(
                            y, np.array([[1, -1], [2, -2]], dtype=scale_dtype)
                        )
                        np.testing.assert_array_equal(
                            statistics, np.array([[0.5], [0.5]], dtype=stash_dtype)
                        )

    def test_each_registration_entry_installs_shape_support(self):
        for entry in (
            "support",
            "legacy",
            "single-global",
            "all-global",
            "single-session",
            "all-session",
        ):
            with self.subTest(entry=entry):
                result = subprocess.run(
                    [sys.executable, __file__, "--registration-entry", entry],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--registration-entry":
        _check_registration_entry(sys.argv[2])
    else:
        unittest.main()
