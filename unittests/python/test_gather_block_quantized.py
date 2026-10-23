# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Packed embedding parity with ONNX Runtime, including native generation."""

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
)
from onnx_light_cpu._benchmark import _case_dtype, infer_pr_benchmark_selection


def _embedding(
    packed_shape=(7, 64),
    indices_shape=(2, 3),
    bits=2,
    block_size=128,
    dtype=np.float32,
    index_dtype=np.int64,
    zero_points=True,
    **attributes,
):
    rng = np.random.default_rng(972)
    groups = (packed_shape[-1] * (8 // bits) + block_size - 1) // block_size
    scales_shape = (*packed_shape[:-1], groups)
    weights = {
        "weight": rng.integers(0, 256, packed_shape, dtype=np.uint8),
        "scales": rng.uniform(0.01, 1, scales_shape).astype(dtype),
    }
    names = ["weight", "input_ids", "scales"]
    if zero_points:
        weights["zeros"] = rng.integers(
            0, 256, (*packed_shape[:-1], (groups + 8 // bits - 1) // (8 // bits)), dtype=np.uint8
        )
        names.append("zeros")
    output_shape = [
        *indices_shape,
        *packed_shape[1:-1],
        packed_shape[-1] * (8 // bits),
    ]
    output_type = TensorProto.FLOAT if dtype == np.float32 else TensorProto.FLOAT16
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "GatherBlockQuantized",
                    names,
                    ["embeddings"],
                    domain="com.microsoft",
                    bits=bits,
                    block_size=block_size,
                    **attributes,
                )
            ],
            "packed-embedding",
            [
                helper.make_tensor_value_info(
                    "input_ids",
                    TensorProto.INT64 if index_dtype == np.int64 else TensorProto.INT32,
                    list(indices_shape),
                )
            ],
            [helper.make_tensor_value_info("embeddings", output_type, output_shape)],
            [numpy_helper.from_array(value, name) for name, value in weights.items()],
        ),
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    return model


def _ort(model):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    return onnxruntime.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def _cpu(model):
    session = ReferenceEvaluator(model)
    register_kernel_for_session(session, "com.microsoft", "GatherBlockQuantized")
    return session


class TestGatherBlockQuantized(ExtTestCase):
    def test_packed_dtype_benchmark_selection(self):
        from onnx_light.onnx.backend import TestMode, collect_test_cases
        from onnx_light_cpu import register_backend_test_cases, registered_kernels

        register_backend_test_cases()
        _, dtypes = infer_pr_benchmark_selection(
            ["onnx_light_cpu/kernels/com_microsoft/gather_block_quantized_kernel.cc"],
            "",
            [
                record
                for record in registered_kernels()
                if record.op_type == "GatherBlockQuantized"
            ],
        )
        self.assertEqual(dtypes, ["uint8"])
        cases = collect_test_cases("GatherBlockQuantized", mode=TestMode.BENCHMARK)
        self.assertEqual(len(cases), 3)
        for case in cases:
            self.assertEqual(_case_dtype(case.name), "uint8")

    def test_ort_parity(self):
        for bits in (2, 4, 8):
            for dtype in (np.float32, np.float16):
                for index_dtype in (np.int32, np.int64):
                    for zeros in (False, True):
                        with self.subTest(
                            bits=bits, dtype=dtype, index_dtype=index_dtype, zeros=zeros
                        ):
                            # Five groups leave padding in each packed zero-point row.
                            model = _embedding(
                                (7, 65 * bits // 2),
                                bits=bits,
                                block_size=64,
                                dtype=dtype,
                                index_dtype=index_dtype,
                                zero_points=zeros,
                                gather_axis=-2,
                                quantize_axis=-1,
                            )
                            feeds = {
                                "input_ids": np.array([[0, 6, -1], [-7, 3, 3]], dtype=index_dtype)
                            }
                            expected = _ort(model).run(None, feeds)[0]
                            actual = _cpu(model).run(None, feeds)[0]
                            self.assertEqual(actual.dtype, dtype)
                            np.testing.assert_array_equal(actual, expected)

    def test_default_attributes(self):
        model = _embedding(bits=4)
        model.graph.node[0].ClearField("attribute")
        feeds = {"input_ids": np.array([[0, -1, 2], [4, 3, 4]], dtype=np.int64)}
        np.testing.assert_array_equal(
            _cpu(model).run(None, feeds)[0], _ort(model).run(None, feeds)[0]
        )

    def test_scalar_empty_and_out_of_range_indices(self):
        for indices in (
            np.array(-1, dtype=np.int64),
            np.empty((2, 0), dtype=np.int64),
        ):
            with self.subTest(shape=indices.shape):
                model = _embedding(indices_shape=indices.shape)
                feeds = {"input_ids": indices}
                np.testing.assert_array_equal(
                    _cpu(model).run(None, feeds)[0], _ort(model).run(None, feeds)[0]
                )
        indices = np.array([[-8, 7, np.iinfo(np.int64).min]], dtype=np.int64)
        model = _embedding(indices_shape=indices.shape)
        np.testing.assert_array_equal(
            _cpu(model).run(None, {"input_ids": indices})[0],
            np.zeros((1, 3, 256), dtype=np.float32),
        )

    def test_rank_three_data(self):
        model = _embedding((7, 2, 17), quantize_axis=-1, block_size=16)
        feeds = {"input_ids": np.array([[1, 5, -1], [0, 2, 1]], dtype=np.int64)}
        np.testing.assert_array_equal(
            _cpu(model).run(None, feeds)[0], _ort(model).run(None, feeds)[0]
        )

    def test_reported_reproduction_with_global_registration(self):
        model = _embedding((3, 64), (1, 2))
        for tensor in model.graph.initializer:
            array = {
                "weight": np.zeros((3, 64), dtype=np.uint8),
                "scales": np.ones((3, 2), dtype=np.float32),
                "zeros": np.zeros((3, 1), dtype=np.uint8),
            }[tensor.name]
            tensor.CopyFrom(numpy_helper.from_array(array, tensor.name))
        register_kernels_global()
        actual = ReferenceEvaluator(model).run(
            None, {"input_ids": np.array([[0, 1]], dtype=np.int64)}
        )[0]
        np.testing.assert_array_equal(actual, np.zeros((1, 2, 256), dtype=np.float32))

    def test_symbolic_shape_and_schema(self):
        model = _embedding(indices_shape=("batch", "sequence"))
        output = model.graph.output[0]
        output.type.tensor_type.ClearField("shape")
        builder = GraphBuilder(model, schema_lookup=operator_schema_lookup)
        result = builder.to_onnx().graph.output[0]
        self.assertEqual(
            [dim.dim_param or dim.dim_value for dim in result.type.tensor_type.shape.dim],
            ["batch", "sequence", 256],
        )
        self.assertEqual(len(custom_op_schemas("GatherBlockQuantized")), 1)

    def test_invalid_attributes_and_parameter_shapes(self):
        for attributes in (
            {"bits": 3},
            {"block_size": 8},
            {"block_size": 24},
            {"gather_axis": 1},
            {"quantize_axis": 0},
        ):
            with self.subTest(attributes=attributes):
                model = _embedding(**attributes)
                with self.assertRaises((ValueError, RuntimeError)):
                    _cpu(model).run(None, {"input_ids": np.zeros((2, 3), dtype=np.int64)})
        for name, array in (
            ("scales", np.ones((7, 1), dtype=np.float32)),
            ("zeros", np.zeros((7, 2), dtype=np.uint8)),
            ("zeros", np.zeros((7, 1), dtype=np.float32)),
        ):
            with self.subTest(parameter=name, shape=array.shape, dtype=array.dtype):
                model = _embedding()
                tensor = next(value for value in model.graph.initializer if value.name == name)
                tensor.CopyFrom(numpy_helper.from_array(array, name))
                with self.assertRaises((ValueError, RuntimeError)):
                    _cpu(model).run(None, {"input_ids": np.zeros((2, 3), dtype=np.int64)})

    def test_split_embedding_native_generation(self):
        rng = np.random.default_rng(480)
        for bits in (2, 4):
            for image in (False, True):
                with self.subTest(bits=bits, image=image):
                    embedding = _embedding((256, 256 // (8 // bits)), (1, "sequence"), bits=bits)
                    nodes = []
                    decoder_inputs = [
                        helper.make_tensor_value_info(
                            "embeddings", TensorProto.FLOAT, [1, "sequence", 256]
                        ),
                        helper.make_tensor_value_info(
                            "position_ids", TensorProto.INT64, [3, 1, "sequence"]
                        ),
                    ]
                    hidden = "embeddings"
                    if image:
                        decoder_inputs.append(
                            helper.make_tensor_value_info(
                                "image_features", TensorProto.FLOAT, [1, 1, 256]
                            )
                        )
                        nodes.append(
                            helper.make_node(
                                "Add", ["embeddings", "image_features"], ["image_hidden"]
                            )
                        )
                        hidden = "image_hidden"
                    nodes.append(helper.make_node("MatMul", [hidden, "head"], ["logits"]))
                    decoder = helper.make_model(
                        helper.make_graph(
                            nodes,
                            "split-decoder",
                            decoder_inputs,
                            [
                                helper.make_tensor_value_info(
                                    "logits", TensorProto.FLOAT, [1, "sequence", 256]
                                )
                            ],
                            [
                                numpy_helper.from_array(
                                    rng.normal(size=(256, 256)).astype(np.float32), "head"
                                )
                            ],
                        ),
                        ir_version=10,
                        opset_imports=[helper.make_opsetid("", 21)],
                    )
                    ort_embedding, ort_decoder = _ort(embedding), _ort(decoder)
                    prompt = np.array([[2, 5]], dtype=np.int64)
                    extra = (
                        {"image_features": rng.normal(size=(1, 1, 256)).astype(np.float32)}
                        if image
                        else {}
                    )
                    expected = prompt.copy()
                    for _ in range(3):
                        embeds = ort_embedding.run(None, {"input_ids": expected})[0]
                        positions = np.broadcast_to(
                            np.arange(expected.shape[1], dtype=np.int64),
                            (3, 1, expected.shape[1]),
                        ).copy()
                        logits = ort_decoder.run(
                            None, {"embeddings": embeds, "position_ids": positions, **extra}
                        )[0]
                        expected = np.concatenate(
                            (expected, logits[:, -1].argmax(axis=-1).reshape(1, 1)), axis=1
                        )
                    session = _cpu(decoder)
                    actual = session.generate(
                        {"input_ids": prompt, **extra},
                        embedding=ReferenceEvaluator(embedding),
                        max_new_tokens=3,
                    )
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(prompt, [[2, 5]])
                    self.assertEqual(embedding.graph.initializer[0].data_type, TensorProto.UINT8)


if __name__ == "__main__":
    unittest.main()
