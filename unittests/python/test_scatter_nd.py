# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""ScatterND byte-copy parity, dynamic embeddings, and session ownership."""

import ml_dtypes
import numpy as np
import onnxruntime

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.backend import TestMode as BackendTestMode, collect_test_cases_by_name
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import (
    register_backend_test_cases,
    register_kernels_for_session,
    registered_kernels,
    set_kernel_usage_recording,
    used_kernel_names,
)

_KERNEL = "onnx_light_cpu::ScatterND"
_DTYPES = {
    TensorProto.FLOAT: np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.BFLOAT16: ml_dtypes.bfloat16,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
}


def _numpy(tensor):
    return np.frombuffer(tensor.raw_data(), dtype=_DTYPES[int(tensor.data_type)]).reshape(
        tuple(tensor.shape)
    )


def _model(
    data_shape, indices_shape, updates_shape, index_type=TensorProto.INT64, reduction=None
):
    attributes = {} if reduction is None else {"reduction": reduction}
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "ScatterND", ["data", "indices", "updates"], ["output"], **attributes
                )
            ],
            "scatternd",
            [
                helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
                helper.make_tensor_value_info("indices", index_type, indices_shape),
                helper.make_tensor_value_info("updates", TensorProto.FLOAT, updates_shape),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, data_shape)],
        ),
        opset_imports=[helper.make_opsetid("", 18 if reduction is not None else 11)],
        ir_version=10,
    )


def _session(model):
    session = ReferenceEvaluator(model)
    register_kernels_for_session(session)
    set_kernel_usage_recording(session, True)
    return session


def _ort_session(model):
    """Keep INT32 feeds and BF16 rounding while using ORT's supported ScatterND schema."""
    adapted = type(model)()
    adapted.CopyFrom(model)
    for opset in adapted.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = max(opset.version, 13)
    nodes = []
    renames = {}
    for value in adapted.graph.input:
        name = value.name
        data_type = value.type.tensor_type.elem_type
        if data_type == TensorProto.INT32:
            target = name + "_int64"
            nodes.append(helper.make_node("Cast", [name], [target], to=TensorProto.INT64))
            renames[name] = target
        elif data_type == TensorProto.BFLOAT16:
            # ORT's NumPy interface cannot consume BF16. Explicitly round to BF16,
            # then copy FLOAT payloads and round the output back through BF16.
            value.type.tensor_type.elem_type = TensorProto.FLOAT
            rounded, target = name + "_bf16", name + "_float"
            nodes.extend(
                [
                    helper.make_node("Cast", [name], [rounded], to=TensorProto.BFLOAT16),
                    helper.make_node("Cast", [rounded], [target], to=TensorProto.FLOAT),
                ]
            )
            renames[name] = target
    for node in adapted.graph.node:
        inputs = [renames.get(name, name) for name in node.input]
        node.input.clear()
        node.input.extend(inputs)
        nodes.append(node)
    for value in adapted.graph.output:
        if value.type.tensor_type.elem_type == TensorProto.BFLOAT16:
            name = value.name
            target = name + "_float_result"
            for node in nodes:
                outputs = [target if output == name else output for output in node.output]
                node.output.clear()
                node.output.extend(outputs)
            nodes.extend(
                [
                    helper.make_node("Cast", [target], [name + "_bf16"], to=TensorProto.BFLOAT16),
                    helper.make_node("Cast", [name + "_bf16"], [name], to=TensorProto.FLOAT),
                ]
            )
            value.type.tensor_type.elem_type = TensorProto.FLOAT
    graph = helper.make_graph(
        nodes, adapted.graph.name, list(adapted.graph.input), list(adapted.graph.output)
    )
    adapted = helper.make_model(
        graph, opset_imports=list(adapted.opset_import), ir_version=adapted.ir_version
    )
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        adapted.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _ort_feeds(feeds):
    return {
        name: value.astype(np.float32) if value.dtype == ml_dtypes.bfloat16 else value
        for name, value in feeds.items()
    }


class TestScatterND(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        register_backend_test_cases()

    def test_registration(self):
        records = [record for record in registered_kernels() if record.op_type == "ScatterND"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].domain, "ai.onnx")
        self.assertEqual(records[0].kernel_name, _KERNEL)
        self.assertEqual(records[0].since_version, 11)
        for dtype in ("FLOAT", "FLOAT16", "BFLOAT16", "INT32", "INT64"):
            self.assertIn(dtype, records[0].types)

    def test_backend_fixtures_and_onnx_runtime(self):
        cases = collect_test_cases_by_name("^test_cpu_scatternd_", mode=BackendTestMode.TEST)
        self.assertEqual(len(cases), 57)
        for case in cases:
            with self.subTest(case=case.name):
                model = case.model
                session = _session(model)
                ort = _ort_session(model)
                previous = []
                for data_set in case.data_sets:
                    feeds = {
                        info.name: _numpy(tensor).copy()
                        for info, tensor in zip(model.graph.input, data_set.inputs, strict=True)
                    }
                    original = {name: value.copy() for name, value in feeds.items()}
                    actual = session.run(None, feeds)[0]
                    expected = _numpy(data_set.outputs[0])
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.tobytes(), expected.tobytes())
                    ort_expected = ort.run(None, _ort_feeds(feeds))[0].astype(expected.dtype)
                    self.assertEqual(actual.tobytes(), ort_expected.tobytes())
                    for name, value in feeds.items():
                        self.assertEqual(value.tobytes(), original[name].tobytes())
                        self.assertFalse(np.shares_memory(actual, value))
                    for old, snapshot in previous:
                        self.assertEqual(old.tobytes(), snapshot)
                        self.assertFalse(np.shares_memory(actual, old))
                    previous.append((actual, actual.tobytes()))
                self.assertIn(_KERNEL, used_kernel_names(session))
                if "multimodal" in case.name:
                    for kernel in ("Gather", "NonZero"):
                        self.assertIn(f"onnx_light_cpu::{kernel}", used_kernel_names(session))

    def test_multimodal_fixture_dynamic_visual_counts(self):
        cases = collect_test_cases_by_name(
            "^test_cpu_scatternd_multimodal_", mode=BackendTestMode.TEST
        )
        self.assertEqual(len(cases), 3)
        for case in cases:
            self.assertEqual(
                [node.op_type for node in case.model.graph.node],
                ["Gather", "Equal", "NonZero", "Transpose", "ScatterND"],
            )
            self.assertEqual(
                [tuple(data.inputs[3].shape) for data in case.data_sets],
                [(0, 6656), (5, 6656), (10, 6656), (0, 6656)],
            )
            visual_shape = case.model.graph.input[3].type.tensor_type.shape.dim
            self.assertEqual(visual_shape[0].dim_param, "visual_count")

    def test_dynamic_shapes_and_output_ownership(self):
        model = _model(["rows", "width"], ["count", 1], ["count", "width"])
        session, ort = _session(model), _ort_session(model)
        previous = []
        for rows, width, count in ((4, 3, 2), (2, 6656, 1), (0, 4, 0), (3, 0, 2), (4, 3, 0)):
            feeds = {
                "data": np.arange(rows * width, dtype=np.float32).reshape(rows, width),
                "indices": np.arange(count, dtype=np.int64).reshape(count, 1),
                "updates": np.full((count, width), -17, dtype=np.float32),
            }
            original = {name: value.tobytes() for name, value in feeds.items()}
            output = session.run(None, feeds)[0]
            self.assertEqualArray(output, ort.run(None, feeds)[0])
            for name, value in feeds.items():
                self.assertEqual(value.tobytes(), original[name])
                self.assertFalse(np.shares_memory(output, value))
            for old, snapshot in previous:
                self.assertEqual(old.tobytes(), snapshot)
                self.assertFalse(np.shares_memory(output, old))
            previous.append((output, output.tobytes()))
        self.assertIn(_KERNEL, used_kernel_names(session))

    def test_duplicate_indices_equal_updates_ort_parity(self):
        for index_type, dtype in ((TensorProto.INT32, np.int32), (TensorProto.INT64, np.int64)):
            model = _model([3, 2], [3, 1], [3, 2], index_type)
            session = _session(model)
            feeds = {
                "data": np.arange(6, dtype=np.float32).reshape(3, 2),
                "indices": np.array([[1], [-2], [1]], dtype=dtype),
                "updates": np.array([[9, 10]] * 3, dtype=np.float32),
            }
            self.assertEqualArray(
                session.run(None, feeds)[0], _ort_session(model).run(None, feeds)[0]
            )
            self.assertIn(_KERNEL, used_kernel_names(session))

    def test_duplicate_indices_last_tuple_wins_local_contract(self):
        # ONNX leaves conflicting duplicate ordering undefined; do not ask ORT
        # to validate this stronger local guarantee.
        model = _model([3, 2], [3, 1], [3, 2])
        session = _session(model)
        feeds = {
            "data": np.arange(6, dtype=np.float32).reshape(3, 2),
            "indices": np.array([[1], [-2], [1]], dtype=np.int64),
            "updates": np.array([[9, 10], [11, 12], [13, 14]], dtype=np.float32),
        }
        output = session.run(None, feeds)[0]
        self.assertEqualArray(output, np.array([[0, 1], [13, 14], [4, 5]], dtype=np.float32))
        self.assertIn(_KERNEL, used_kernel_names(session))

    def test_bounds_rejected_including_zero_width_slices(self):
        for width in (0, 2):
            for index in (3, -4):
                for index_type, dtype in (
                    (TensorProto.INT32, np.int32),
                    (TensorProto.INT64, np.int64),
                ):
                    with self.subTest(width=width, index=index, dtype=dtype):
                        model = _model([3, width], [1, 1], [1, width], index_type)
                        session = _session(model)
                        with self.assertRaisesRegex(
                            (ValueError, RuntimeError), "(?i)(index|indices|bounds|range)"
                        ):
                            session.run(
                                None,
                                {
                                    "data": np.zeros((3, width), dtype=np.float32),
                                    "indices": np.array([[index]], dtype=dtype),
                                    "updates": np.zeros((1, width), dtype=np.float32),
                                },
                            )

    def test_explicit_none_and_unsupported_reductions(self):
        feeds = {
            "data": np.zeros((3, 2), dtype=np.float32),
            "indices": np.array([[1]], dtype=np.int64),
            "updates": np.ones((1, 2), dtype=np.float32),
        }
        model = _model([3, 2], [1, 1], [1, 2], reduction="none")
        session = _session(model)
        self.assertEqualArray(
            session.run(None, feeds)[0], _ort_session(model).run(None, feeds)[0]
        )
        self.assertIn(_KERNEL, used_kernel_names(session))
        for reduction in ("add", "mul", "min", "max", "unknown"):
            with self.subTest(reduction=reduction):
                model = _model([3, 2], [1, 1], [1, 2], reduction=reduction)
                with self.assertRaisesRegex((ValueError, RuntimeError), "(?i)reduction"):
                    _session(model).run(None, feeds)

    def test_benchmarks_have_lazy_expected_outputs(self):
        cases = collect_test_cases_by_name("^test_cpu_scatternd_", mode=BackendTestMode.BENCHMARK)
        self.assertEqual(len(cases), 18)
        for case in cases:
            self.assertFalse(case.has_expected_outputs)
            self.assertTrue(all(not data.outputs for data in case.data_sets))
