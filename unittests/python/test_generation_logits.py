# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Generation projection selection preserves the exported output contract."""

import unittest

import numpy as np
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import ModelProto, TensorProto, checker, helper, numpy_helper
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import has_cpu_kernels, register_kernels_for_session
from onnx_light_cpu.generation import optimize_generation_logits


def make_model(dtype=np.float32, selected=True, expose_full=False):
    proto_dtype = {
        np.float16: TensorProto.FLOAT16,
        np.float32: TensorProto.FLOAT,
        np.float64: TensorProto.DOUBLE,
    }[dtype]
    weights = np.random.default_rng(42).standard_normal((8, 17)).astype(dtype)
    nodes = [helper.make_node("MatMul", ["hidden", "weight"], ["full"])]
    inputs = [helper.make_tensor_value_info("hidden", proto_dtype, ["B", "S", 8])]
    outputs = []
    if selected:
        nodes.append(
            helper.make_node("GatherND", ["full", "positions"], ["logits"], batch_dims=1)
        )
        inputs.append(helper.make_tensor_value_info("positions", TensorProto.INT64, ["B", 1, 1]))
        outputs.append(helper.make_tensor_value_info("logits", proto_dtype, ["B", 1, 17]))
    if not selected or expose_full:
        outputs.append(helper.make_tensor_value_info("full", proto_dtype, ["B", "S", 17]))
    return helper.make_model(
        helper.make_graph(
            nodes,
            "generation_projection",
            inputs,
            outputs,
            [numpy_helper.from_array(weights, "weight")],
            value_info=[helper.make_tensor_value_info("full", proto_dtype, ["B", "S", 17])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )


# Cache the built-in MatMul before accelerated kernels can be registered.
_FULL = {}
for _dtype in (np.float16, np.float32, np.float64):
    _session = ReferenceEvaluator(make_model(_dtype, selected=False))
    _session.run(None, {"hidden": np.zeros((1, 1, 8), dtype=_dtype)})
    _FULL[_dtype] = _session


class TestGenerationLogits(ExtTestCase):
    def test_selected_hidden_states_match_full_logits(self):
        self._check_selected_hidden_states(accelerated=False)

    @unittest.skipUnless(has_cpu_kernels(), "CPU runtime extension is not built")
    def test_accelerated_selected_hidden_states_match_full_logits(self):
        self._check_selected_hidden_states(accelerated=True)

    def _check_selected_hidden_states(self, accelerated):
        import onnxruntime as ort

        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                original = make_model(dtype)
                before = original.SerializeToString()
                reduced = optimize_generation_logits(original)
                self.assertEqual(original.SerializeToString(), before)
                self.assertEqual([n.op_type for n in reduced.graph.node], ["GatherND", "MatMul"])
                self.assertEqual(reduced.graph.node[0].input[0], "hidden")
                checker.check_model(reduced)
                self.assertEqual(
                    optimize_generation_logits(reduced).SerializeToString(),
                    reduced.SerializeToString(),
                )
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                runtime = ort.InferenceSession(
                    reduced.SerializeToString(), options, providers=["CPUExecutionProvider"]
                )
                full_session = ReferenceEvaluator(make_model(dtype, selected=False))
                if accelerated:
                    register_kernels_for_session(full_session)
                # Construct two sessions from the same model, without consuming weights.
                for _ in range(2):
                    session = ReferenceEvaluator(reduced)
                    if accelerated:
                        self.assertGreater(register_kernels_for_session(session), 0)
                    for sequence, positions in (
                        (5, [4]),
                        (5, [2, 4]),  # right-padded, unequal valid lengths
                        (5, [4, 4]),  # left-padded
                        (5, [0, -1]),  # original negative-index semantics
                        (1, [0, 0]),  # repeated decode after prefill
                        (3, [1, 2]),  # another prefill on the same sessions
                    ):
                        hidden = (
                            np.random.default_rng(sequence)
                            .standard_normal((len(positions), sequence, 8))
                            .astype(dtype)
                        )
                        feeds = {
                            "hidden": hidden,
                            "positions": np.array(positions, dtype=np.int64).reshape(-1, 1, 1),
                        }
                        full = _FULL[dtype].run(None, {"hidden": hidden})[0]
                        default_logits = full_session.run(None, {"hidden": hidden})[0]
                        self.assertEqual(default_logits.shape, full.shape)
                        np.testing.assert_allclose(default_logits, full, rtol=2e-3, atol=2e-3)
                        expected = np.stack(
                            [
                                full[b, p : p + 1] if p >= 0 else full[b, -1:]
                                for b, p in enumerate(positions)
                            ]
                        )
                        for actual in (session.run(None, feeds)[0], runtime.run(None, feeds)[0]):
                            self.assertEqual(actual.shape, (len(positions), 1, 17))
                            self.assertEqual(actual.dtype, np.dtype(dtype))
                            np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)

    def test_full_logits_and_other_consumers_are_not_rewritten(self):
        for model in (make_model(selected=False), make_model(expose_full=True)):
            self.assertEqual(
                optimize_generation_logits(model).SerializeToString(), model.SerializeToString()
            )
        model = make_model()
        model.graph.node.append(helper.make_node("Identity", ["full"], ["other"]))
        model.graph.output.append(
            helper.make_tensor_value_info("other", TensorProto.FLOAT, ["B", "S", 17])
        )
        self.assertEqual(
            optimize_generation_logits(model).SerializeToString(), model.SerializeToString()
        )

    def test_missing_or_unsupported_shapes_are_not_rewritten(self):
        for shape in (["B", "S", "H"], ["S", 8], ["B", "S", 7]):
            model = make_model()
            model.graph.input[0].CopyFrom(
                helper.make_tensor_value_info("hidden", TensorProto.FLOAT, shape)
            )
            self.assertEqual(
                optimize_generation_logits(model).SerializeToString(), model.SerializeToString()
            )
        for shape in (["B", 2, 1], ["B", 1, 2]):
            model = make_model()
            model.graph.input[1].CopyFrom(
                helper.make_tensor_value_info("positions", TensorProto.INT64, shape)
            )
            self.assertEqual(
                optimize_generation_logits(model).SerializeToString(), model.SerializeToString()
            )

    def test_subgraph_capture_is_not_rewritten(self):
        model = make_model()
        branch = helper.make_graph(
            [helper.make_node("Identity", ["full"], ["captured"])],
            "capture",
            [],
            [helper.make_tensor_value_info("captured", TensorProto.FLOAT, ["B", "S", 17])],
        )
        model.graph.node.append(
            helper.make_node(
                "If", ["condition"], ["other"], then_branch=branch, else_branch=branch
            )
        )
        model.graph.initializer.append(numpy_helper.from_array(np.array(True), "condition"))
        model.graph.output.append(
            helper.make_tensor_value_info("other", TensorProto.FLOAT, ["B", "S", 17])
        )
        checker.check_model(model)
        self.assertEqual(
            optimize_generation_logits(model).SerializeToString(), model.SerializeToString()
        )

    def test_overridable_initializers_require_input_shape_proof(self):
        model = make_model()
        model.graph.input.append(helper.make_tensor_value_info("weight", TensorProto.FLOAT, None))
        self.assertEqual(
            optimize_generation_logits(model).SerializeToString(), model.SerializeToString()
        )
        model = make_model()
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([[[0]]], dtype=np.int64), "positions")
        )
        model.graph.input[1].CopyFrom(
            helper.make_tensor_value_info("positions", TensorProto.INT64, ["B", 1, "index_rank"])
        )
        self.assertEqual(
            optimize_generation_logits(model).SerializeToString(), model.SerializeToString()
        )

    def test_dynamic_index_computation_stays_before_projection(self):
        model = make_model()
        projection, selection = list(model.graph.node)
        model.graph.node.clear()
        model.graph.node.extend(
            [
                projection,
                helper.make_node("Identity", ["positions"], ["computed_positions"]),
                selection,
            ]
        )
        model.graph.node[2].input.clear()
        model.graph.node[2].input.extend(("full", "computed_positions"))
        model.graph.value_info.append(
            helper.make_tensor_value_info("computed_positions", TensorProto.INT64, ["B", 1, 1])
        )
        reduced = optimize_generation_logits(model)
        checker.check_model(reduced)
        self.assertEqual(
            [node.op_type for node in reduced.graph.node], ["Identity", "GatherND", "MatMul"]
        )
        feeds = {
            "hidden": np.ones((2, 3, 8), dtype=np.float32),
            "positions": np.array([[[1]], [[2]]], dtype=np.int64),
        }
        np.testing.assert_allclose(
            ReferenceEvaluator(reduced).run(None, feeds)[0],
            ReferenceEvaluator(model).run(None, feeds)[0],
            rtol=2e-3,
            atol=2e-3,
        )

    def test_benchmark_reports_prefill_projection_and_output_memory(self):
        from tools.benchmark_generation_logits import benchmark, make_model

        model, feeds = make_model(batch=2, sequence=5, hidden=8, vocab=17)
        rows = benchmark(model, feeds, repeat=1, warmup=0, backends=("onnxruntime",))
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertIn(row["phase"], ("prefill", "projection"))
            self.assertGreater(row["median_s"], 0)
            self.assertEqual(
                row["logits_bytes"], 2 * 17 * 4 * (5 if row["contract"] == "all_tokens" else 1)
            )
            self.assertEqual(row["logits_bytes_saved"], 2 * 5 * 17 * 4 - row["logits_bytes"])

    @unittest.skipUnless(has_cpu_kernels(), "CPU runtime extension is not built")
    def test_accelerated_benchmark_uses_the_same_reduced_model(self):
        from tools.benchmark_generation_logits import benchmark, make_model

        model, feeds = make_model(batch=2, sequence=5, hidden=8, vocab=17)
        rows = benchmark(model, feeds, repeat=1, warmup=0)
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["backend"] for row in rows}, {"onnx-light-cpu", "onnxruntime"})

    def test_repeated_generation_from_the_same_model(self):
        model = make_model()
        model.graph.input.clear()
        model.graph.input.append(
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "S"])
        )
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(
                    np.random.default_rng(44).standard_normal((17, 8)).astype(np.float32),
                    "embedding",
                ),
                numpy_helper.from_array(np.array([[[-1]]], dtype=np.int64), "positions"),
            ]
        )
        nodes = list(model.graph.node)
        model.graph.node.clear()
        model.graph.node.append(
            helper.make_node("Gather", ["embedding", "input_ids"], ["hidden"], axis=0)
        )
        model.graph.node.extend(nodes)
        model.graph.value_info.append(
            helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, "S", 8])
        )
        reduced = optimize_generation_logits(model)
        self.assertEqual(
            [node.op_type for node in reduced.graph.node], ["Gather", "GatherND", "MatMul"]
        )
        full = ModelProto()
        full.CopyFrom(model)
        nodes = list(full.graph.node)
        full.graph.node.clear()
        full.graph.node.extend(nodes[:2])
        full.graph.node[1].output.clear()
        full.graph.node[1].output.append("logits")
        full.graph.output[0].CopyFrom(
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "S", 17])
        )
        full_session = ReferenceEvaluator(full)
        if has_cpu_kernels():
            register_kernels_for_session(full_session)
        # Repeated generate calls and repeated session creation must retain weights.
        for _ in range(2):
            session = ReferenceEvaluator(reduced)
            if has_cpu_kernels():
                register_kernels_for_session(session)
            for prompt in ([1, 2, 3], [4], [5, 6]):
                feeds = {"input_ids": np.array([prompt], dtype=np.int64)}
                expected = full_session.generate(feeds, max_new_tokens=3)
                actual = session.generate(feeds, max_new_tokens=3)
                np.testing.assert_array_equal(actual, expected)

    def test_zero_length_and_out_of_bounds_positions(self):
        import onnxruntime as ort

        original = make_model()
        reduced = optimize_generation_logits(original)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        for model in (original, reduced):
            session = ReferenceEvaluator(model)
            if has_cpu_kernels():
                register_kernels_for_session(session)
            runtime = ort.InferenceSession(
                model.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
            for sequence, position in ((0, 0), (3, 3), (3, -4)):
                feeds = {
                    "hidden": np.zeros((1, sequence, 8), dtype=np.float32),
                    "positions": np.array([[[position]]], dtype=np.int64),
                }
                with self.assertRaises((RuntimeError, ValueError)):
                    session.run(None, feeds)
                with self.assertRaises(ort.capi.onnxruntime_pybind11_state.InvalidArgument):
                    runtime.run(None, feeds)


if __name__ == "__main__":
    unittest.main()
