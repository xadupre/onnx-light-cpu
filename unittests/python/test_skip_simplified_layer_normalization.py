# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Checks fused residual RMS normalization through native and ONNX Runtime sessions."""

import unittest

import ml_dtypes
import numpy as np
import onnxruntime

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.backend import TestMode, collect_test_cases_by_name
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import (
    MicrosoftKernelImplementation,
    clear_used_kernel_names,
    register_backend_test_cases,
    register_kernel_for_session,
    set_kernel_usage_recording,
    used_kernel_names,
)

_OP = "SkipSimplifiedLayerNormalization"
_PREFIX = "^test_cpu_skip_simplified_layer_normalization_"
_IMPLEMENTATIONS = (
    MicrosoftKernelImplementation.OPTIMIZED,
    MicrosoftKernelImplementation.NAIVE,
)


def _model(dtype=TensorProto.FLOAT, shape=(2, 2), bias=False, residual=True, epsilon=0.0):
    """Builds a model preserving omitted bias and statistics slots."""
    inputs = [
        helper.make_tensor_value_info("input", dtype, shape),
        helper.make_tensor_value_info("skip", dtype, shape),
        helper.make_tensor_value_info("gamma", dtype, [shape[-1]]),
    ]
    if bias:
        inputs.append(helper.make_tensor_value_info("bias", dtype, [shape[-1]]))
    outputs = [helper.make_tensor_value_info("output", dtype, shape)]
    if residual:
        outputs.append(helper.make_tensor_value_info("sum", dtype, shape))
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    _OP,
                    ["input", "skip", "gamma", "bias" if bias else ""],
                    ["output", "", "", "sum"] if residual else ["output"],
                    domain="com.microsoft",
                    epsilon=epsilon,
                )
            ],
            "skip-rms-normalization",
            inputs,
            outputs,
        ),
        opset_imports=[helper.make_opsetid("", 23), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )


def _array(tensor):
    """Decodes a backend tensor without changing its stored precision."""
    dtype = {
        int(TensorProto.FLOAT): np.float32,
        int(TensorProto.FLOAT16): np.float16,
        int(TensorProto.BFLOAT16): ml_dtypes.bfloat16,
    }[int(tensor.data_type)]
    return np.frombuffer(tensor.raw_data(), dtype=dtype).reshape(tuple(tensor.shape))


def _session(model, implementation):
    """Creates a session-local native implementation without changing global dispatch."""
    session = ReferenceEvaluator(model)
    register_kernel_for_session(
        session, "com.microsoft", _OP, microsoft_implementation=implementation
    )
    return session


class TestSkipSimplifiedLayerNormalization(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        register_backend_test_cases()

    def test_analytical_outputs_both_implementations_and_dtypes(self):
        for implementation in _IMPLEMENTATIONS:
            for dtype, tensor_type in (
                (np.float32, TensorProto.FLOAT),
                (np.float16, TensorProto.FLOAT16),
                (ml_dtypes.bfloat16, TensorProto.BFLOAT16),
            ):
                for bias in (False, True):
                    for residual in (False, True):
                        with self.subTest(
                            implementation=implementation,
                            dtype=dtype,
                            bias=bias,
                            residual=residual,
                        ):
                            feeds = {
                                "input": np.array([[1, -3], [3, -5]], dtype=dtype),
                                "skip": np.ones((2, 2), dtype=dtype),
                                "gamma": np.array([2, -3], dtype=dtype),
                            }
                            if bias:
                                feeds["skip"] *= dtype(0.5)
                                feeds["bias"] = np.full(2, 0.5, dtype=dtype)
                            originals = {name: value.copy() for name, value in feeds.items()}
                            session = _session(
                                _model(tensor_type, bias=bias, residual=residual),
                                implementation,
                            )
                            set_kernel_usage_recording(session, True)
                            clear_used_kernel_names(session)
                            result = session.run(None, feeds)
                            dispatched = used_kernel_names(session)
                            set_kernel_usage_recording(session, False)
                            self.assertEqual(len(result), 2 if residual else 1)
                            self.assertEqual(result[0].dtype, np.dtype(dtype))
                            np.testing.assert_array_equal(
                                result[0], np.array([[2, 3], [2, 3]], dtype=dtype)
                            )
                            if residual:
                                self.assertEqual(result[1].dtype, np.dtype(dtype))
                                np.testing.assert_array_equal(
                                    result[1], np.array([[2, -2], [4, -4]], dtype=dtype)
                                )
                            expected_name = (
                                "onnx_light_cpu::Naive" + _OP
                                if implementation == MicrosoftKernelImplementation.NAIVE
                                else "onnx_light_cpu::" + _OP
                            )
                            self.assertIn(expected_name, dispatched)
                            for name, original in originals.items():
                                np.testing.assert_array_equal(feeds[name], original)

    def test_backend_naive_oracle_corpus(self):
        cases = collect_test_cases_by_name(_PREFIX, mode=TestMode.TEST)
        self.assertEqual(len(cases), 15)
        self.assertEqual(
            {int(case.model.graph.input[0].type.tensor_type.elem_type) for case in cases},
            {int(TensorProto.FLOAT), int(TensorProto.FLOAT16), int(TensorProto.BFLOAT16)},
        )
        for case in cases:
            for implementation in _IMPLEMENTATIONS:
                with self.subTest(case=case.name, implementation=implementation):
                    session = _session(case.model, implementation)
                    for dataset in case.data_sets:
                        feeds = {
                            value.name: _array(tensor)
                            for value, tensor in zip(
                                case.model.graph.input, dataset.inputs, strict=True
                            )
                        }
                        actual = session.run(None, feeds)
                        self.assertEqual(len(actual), len(dataset.outputs))
                        for output, reference in zip(actual, dataset.outputs, strict=True):
                            expected = _array(reference)
                            self.assertEqual(output.dtype, expected.dtype)
                            np.testing.assert_allclose(
                                output.astype(np.float32),
                                expected.astype(np.float32),
                                rtol=2e-3,
                                atol=2e-3,
                            )

    def test_benchmarks_are_lazy_and_expected_outputs_are_opt_in(self):
        for requested in (False, True):
            cases = collect_test_cases_by_name(
                _PREFIX,
                mode=TestMode.BENCHMARK,
                include_big=True,
                generate_benchmark_expected_outputs=requested,
            )
            self.assertEqual(len(cases), 6)
            for case in cases:
                self.assertEqual(case.has_expected_outputs, requested)
                self.assertEqual(list(case.model.graph.node[0].output)[1:3], ["", ""])
                for dataset in case.data_sets:
                    self.assertEqual(dataset.expected_outputs_generated, requested)
                    self.assertEqual(len(dataset.outputs), 2 if requested else 0)
                    self.assertIn(len(dataset.inputs), (3, 4))
                    if requested:
                        feeds = {
                            value.name: _array(tensor)
                            for value, tensor in zip(
                                case.model.graph.input, dataset.inputs, strict=True
                            )
                        }
                        actual = _session(case.model, MicrosoftKernelImplementation.NAIVE).run(
                            None, feeds
                        )
                        for output, expected in zip(actual, dataset.outputs, strict=True):
                            reference = _array(expected)
                            np.testing.assert_allclose(
                                output.astype(np.float32),
                                reference.astype(np.float32),
                                rtol=(1 / 128 if reference.dtype == ml_dtypes.bfloat16 else 2e-3),
                                atol=2e-3,
                            )

    def test_cpu_onnxruntime_parity(self):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        for mode in (TestMode.TEST, TestMode.BENCHMARK):
            cases = collect_test_cases_by_name(
                _PREFIX + ".*_float(16|32)", mode=mode, include_big=True
            )
            self.assertEqual(len(cases), 10 if mode == TestMode.TEST else 4)
            for case in cases:
                with self.subTest(case=case.name):
                    model = type(case.model)()
                    model.ParseFromString(case.model.SerializeToString())
                    model.ir_version = 10
                    # Older ORT CPU builds do not populate the statistics outputs.
                    node = model.graph.node[0]
                    if len(node.output) == 4:
                        names = list(node.output)
                        names[1:3] = ["", ""]
                        node.output.clear()
                        node.output.extend(names)
                        outputs = [
                            output
                            for output in model.graph.output
                            if output.name in (node.output[0], node.output[3])
                        ]
                        model = helper.make_model(
                            helper.make_graph(
                                [node], model.graph.name, list(model.graph.input), outputs
                            ),
                            opset_imports=list(model.opset_import),
                            ir_version=10,
                        )
                    oracle = onnxruntime.InferenceSession(
                        model.SerializeToString(),
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
                    for dataset in case.data_sets:
                        feeds = {
                            value.name: _array(tensor)
                            for value, tensor in zip(
                                model.graph.input, dataset.inputs, strict=True
                            )
                        }
                        expected = oracle.run(None, feeds)
                        for implementation in _IMPLEMENTATIONS:
                            actual = _session(model, implementation).run(None, feeds)
                            for output, reference in zip(actual, expected, strict=True):
                                tolerance = 2e-3 if reference.dtype == np.float16 else 3e-5
                                np.testing.assert_allclose(
                                    output, reference, rtol=tolerance, atol=tolerance
                                )


if __name__ == "__main__":
    unittest.main()
