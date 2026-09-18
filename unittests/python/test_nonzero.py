# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""NonZero backend parity, dynamic allocation, and symbolic shape integration."""

import numpy as np
import onnxruntime

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.backend import TestMode as BackendTestMode, collect_test_cases_by_name
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_core.graph_builder import GraphBuilder
from onnx_light_cpu import (
    register_backend_test_cases,
    register_kernels_for_session,
    registered_kernels,
    set_kernel_usage_recording,
    used_kernel_names,
)


def _model(shape):
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("NonZero", ["X"], ["Y"])],
            "nonzero",
            [helper.make_tensor_value_info("X", TensorProto.BOOL, shape)],
            [helper.make_tensor_value_info("Y", TensorProto.INT64, None)],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )


def _ort_session(model):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _numpy(tensor):
    dtype = {TensorProto.BOOL: np.bool_, TensorProto.INT64: np.int64}[int(tensor.data_type)]
    return np.frombuffer(tensor.raw_data(), dtype=dtype).reshape(tuple(tensor.shape))


class TestNonZero(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        register_backend_test_cases()

    def test_registration(self):
        records = [r for r in registered_kernels() if r.op_type == "NonZero"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].domain, "ai.onnx")
        self.assertEqual(records[0].kernel_name, "onnx_light_cpu::NonZero")
        self.assertEqual(records[0].types, ("BOOL",))
        self.assertEqual(records[0].since_version, 9)

    def test_backend_fixtures_and_onnx_runtime(self):
        cases = collect_test_cases_by_name("^test_cpu_nonzero_", mode=BackendTestMode.TEST)
        self.assertEqual(len(cases), 22)
        for case in cases:
            with self.subTest(case=case.name):
                model = case.model
                session = ReferenceEvaluator(model)
                register_kernels_for_session(session)
                set_kernel_usage_recording(session, True)
                ort_session = _ort_session(model)
                for data_set in case.data_sets:
                    feeds = {
                        info.name: _numpy(tensor)
                        for info, tensor in zip(model.graph.input, data_set.inputs, strict=True)
                    }
                    actual = session.run(None, feeds)[0]
                    expected = _numpy(data_set.outputs[0])
                    self.assertEqual(actual.dtype, np.int64)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqualArray(actual, expected)
                    ort_expected = ort_session.run(None, feeds)[0]
                    if len(model.graph.node) == 1 and feeds["X"].ndim == 0:
                        # ORT 1.30 uses [1, count] for scalars, contrary to ONNX's
                        # [rank, count] contract. Compare counts, preserving rank 0.
                        self.assertEqual(actual.shape[0], 0)
                        self.assertEqual(actual.shape[1], ort_expected.shape[1])
                    else:
                        self.assertEqual(actual.shape, ort_expected.shape)
                        self.assertEqualArray(actual, ort_expected)
                self.assertIn("onnx_light_cpu::NonZero", used_kernel_names(session))

    def test_counts_and_shapes_change_in_one_session(self):
        model = _model(["batch", "sequence"])
        session = ReferenceEvaluator(model)
        register_kernels_for_session(session)
        ort_session = _ort_session(model)
        for shape in ((2, 3), (1, 17), (2, 0), (3, 5), (0, 4), (2, 3)):
            for value in (False, True, False):
                feeds = {"X": np.full(shape, value, dtype=np.bool_)}
                actual = session.run(None, feeds)[0]
                expected = ort_session.run(None, feeds)[0]
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqualArray(actual, expected)

    def test_shape_inference_keeps_count_dynamic(self):
        for shape in ([], [7], [2, 3], ["batch", "sequence"], [2, 0, 4]):
            with self.subTest(shape=shape):
                model = GraphBuilder(_model(shape)).to_onnx("model")
                output = model.graph.output[0].type.tensor_type
                self.assertEqual(output.elem_type, TensorProto.INT64)
                self.assertEqual(len(output.shape.dim), 2)
                self.assertEqual(output.shape.dim[0].dim_value, len(shape))
                self.assertTrue(output.shape.dim[1].dim_param)

    def test_embedding_fixture_has_dynamic_positions(self):
        case = collect_test_cases_by_name(
            "^test_cpu_nonzero_multimodal_token_positions_bool$", mode=BackendTestMode.TEST
        )[0]
        self.assertEqual(
            [node.op_type for node in case.model.graph.node], ["Equal", "NonZero", "Transpose"]
        )
        self.assertEqual(
            [tuple(data.outputs[0].shape) for data in case.data_sets],
            [(0, 2), (1, 2), (6, 2), (10, 2), (16, 2), (0, 2)],
        )
        model = GraphBuilder(case.model).to_onnx("model")
        shape = model.graph.output[0].type.tensor_type.shape.dim
        self.assertTrue(shape[0].dim_param)
        self.assertEqual(shape[1].dim_value, 2)

    def test_benchmarks_have_lazy_expected_outputs(self):
        cases = collect_test_cases_by_name("^test_cpu_nonzero_", mode=BackendTestMode.BENCHMARK)
        self.assertEqual(len(cases), 3)
        for case in cases:
            self.assertFalse(case.has_expected_outputs)
            self.assertTrue(all(not data.outputs for data in case.data_sets))
