# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Backend-test driven end-to-end tests for the onnx-light-cpu kernels.

onnx-light exposes a global ONNX backend test case registry through
:func:`onnx_light.onnx.backend.collect_test_cases`. onnx-light-cpu ships its own
backend test cases -- named ``test_cpu_*`` and covering every element type each
accelerated kernel implements -- in a dedicated C++ *registration* library
(``lib_onnx_light_cpu_backend_test``). Those cases are installed into that same
shared registry by :func:`onnx_light_cpu.register_backend_test_cases`.

These tests follow the same steps as the C++ unit test, using onnx-light's
regular Python API:

* register the onnx-light-cpu backend test cases
  (:func:`onnx_light_cpu.register_backend_test_cases`),
* register the accelerated kernels (:func:`onnx_light_cpu.register_kernels`), and
* for every collected ``test_cpu_*`` case, run its single-node model through
  onnx-light's ``ReferenceEvaluator`` and check that
    - the accelerated onnx-light-cpu kernel is the one actually dispatched to
      (via :func:`onnx_light_cpu.used_kernel_names`), and
    - its outputs match the reference outputs shipped with the case (using the
      case's ``rtol``/``atol``).

onnx-light, its backend-test extension (exposed via the ``_cpuregister``
extension's ``register_backend_test_cases`` binding, built with
``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON`` and onnx-light's ``lib_onnx_backend_test``
available) are required. The selected backend cases use dtypes supported by
NumPy or ``ml_dtypes``, including ``BFLOAT16``.
"""

from __future__ import annotations

from unittest import TestCase

import ml_dtypes
import numpy as np

from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.backend import TestMode, collect_test_cases, collect_test_cases_by_name
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import (
    RegisteredKernel,
    clear_used_kernel_names,
    has_backend_test_cases,
    register_backend_test_cases,
    register_kernels,
    registered_kernel_names,
    registered_kernels,
    used_kernel_names,
)

assert has_backend_test_cases(), (
    "onnx-light-cpu must be built with onnx-light's backend test registry "
    "(register_backend_test_cases binding unavailable)."
)

# Operators whose onnx-light-cpu kernel we validate against every ``test_cpu_*``
# backend test case, mapped to the library-qualified name each kernel records
# when it runs.
_REGISTERED_KERNELS = {
    "Abs": "onnx_light_cpu::Abs",
    "Add": "onnx_light_cpu::Add",
    "And": "onnx_light_cpu::And",
    "Attention": "onnx_light_cpu::Attention",
    "BatchNormalization": "onnx_light_cpu::BatchNormalization",
    "BiasGelu": "onnx_light_cpu::BiasGelu",
    "BitShift": "onnx_light_cpu::BitShift",
    "BitwiseAnd": "onnx_light_cpu::BitwiseAnd",
    "BitwiseOr": "onnx_light_cpu::BitwiseOr",
    "BitwiseXor": "onnx_light_cpu::BitwiseXor",
    "CDist": "onnx_light_cpu::CDist",
    "Div": "onnx_light_cpu::Div",
    "Equal": "onnx_light_cpu::Equal",
    "Exp": "onnx_light_cpu::Exp",
    "Gemm": "onnx_light_cpu::Gemm",
    "GroupNormalization": "onnx_light_cpu::GroupNormalization",
    "InstanceNormalization": "onnx_light_cpu::InstanceNormalization",
    "LayerNormalization": "onnx_light_cpu::LayerNormalization",
    "Greater": "onnx_light_cpu::Greater",
    "GreaterOrEqual": "onnx_light_cpu::GreaterOrEqual",
    "Less": "onnx_light_cpu::Less",
    "LessOrEqual": "onnx_light_cpu::LessOrEqual",
    "Log": "onnx_light_cpu::Log",
    "LpNormalization": "onnx_light_cpu::LpNormalization",
    "MatMul": "onnx_light_cpu::MatMul",
    "MatMulInteger": "onnx_light_cpu::MatMulInteger",
    "Max": "onnx_light_cpu::Max",
    "Mean": "onnx_light_cpu::Mean",
    "MeanVarianceNormalization": "onnx_light_cpu::MeanVarianceNormalization",
    "Min": "onnx_light_cpu::Min",
    "Mod": "onnx_light_cpu::Mod",
    "Mul": "onnx_light_cpu::Mul",
    "Not": "onnx_light_cpu::Not",
    "Or": "onnx_light_cpu::Or",
    "PRelu": "onnx_light_cpu::PRelu",
    "Pow": "onnx_light_cpu::Pow",
    "QLinearMatMul": "onnx_light_cpu::QLinearMatMul",
    "RMSNormalization": "onnx_light_cpu::RMSNormalization",
    "Sub": "onnx_light_cpu::Sub",
    "Sum": "onnx_light_cpu::Sum",
    "SwiGLU": "onnx_light_cpu::SwiGLU",
    "TreeEnsemble": "onnx_light_cpu::TreeEnsemble",
    "Xor": "onnx_light_cpu::Xor",
}

_TARGET_KERNELS = {
    op_type: kernel_name
    for op_type, kernel_name in _REGISTERED_KERNELS.items()
    if op_type not in {"Max", "Mean", "Min", "QLinearMatMul", "Sum"}
}

_BENCHMARK_TYPE_SUFFIXES = dict.fromkeys(_TARGET_KERNELS, "float32")
for _op_type in ("And", "Not", "Or", "Xor"):
    _BENCHMARK_TYPE_SUFFIXES[_op_type] = "bool"
for _op_type in ("BitwiseAnd", "BitwiseOr", "BitwiseXor", "MatMulInteger"):
    _BENCHMARK_TYPE_SUFFIXES[_op_type] = "int8"
_BENCHMARK_TYPE_SUFFIXES["BitShift"] = "uint8"
for _op_type in {
    "BatchNormalization",
    "GroupNormalization",
    "InstanceNormalization",
    "LayerNormalization",
    "LpNormalization",
    "MeanVarianceNormalization",
    "RMSNormalization",
}:
    _BENCHMARK_TYPE_SUFFIXES[_op_type] = "(?:float32|float16|bfloat16)"
_BENCHMARK_OP_TAGS = {"RMSNormalization": "rms_normalization"}
_BENCHMARK_NAME_PATTERN = (
    "^test_cpu_(?:"
    + "|".join(
        f"{_BENCHMARK_OP_TAGS.get(op_type, op_type.lower())}_.*_{suffix}"
        for op_type, suffix in _BENCHMARK_TYPE_SUFFIXES.items()
    )
    + ").*_benchmark$"
)

# ``TensorProto`` element type -> numpy dtype used to decode a backend test
# case ``Tensor``'s raw little-endian row-major buffer.
_TP_TO_NP = {
    int(TensorProto.FLOAT): np.float32,
    int(TensorProto.DOUBLE): np.float64,
    int(TensorProto.INT8): np.int8,
    int(TensorProto.INT16): np.int16,
    int(TensorProto.INT32): np.int32,
    int(TensorProto.INT64): np.int64,
    int(TensorProto.UINT8): np.uint8,
    int(TensorProto.UINT16): np.uint16,
    int(TensorProto.UINT32): np.uint32,
    int(TensorProto.UINT64): np.uint64,
    int(TensorProto.BOOL): np.bool_,
    int(TensorProto.FLOAT16): np.float16,
    int(TensorProto.BFLOAT16): ml_dtypes.bfloat16,
}


def _to_numpy(tensor):
    """Decodes a backend test case ``Tensor`` into a numpy array."""
    dtype = _TP_TO_NP[int(tensor.data_type)]
    shape = tuple(int(d) for d in tensor.shape)
    return np.frombuffer(tensor.raw_data(), dtype=dtype).reshape(shape)


def _single_node_op_type(tc):
    """Returns the op_type when ``tc``'s graph is a single node, else ``None``."""
    nodes = list(tc.model.graph.node)
    if len(nodes) != 1:
        return None
    return nodes[0].op_type


def _uses_supported_dtypes(tc):
    return all(
        int(tensor.data_type) in _TP_TO_NP
        for data_set in tc.data_sets
        for tensor in (*data_set.inputs, *data_set.outputs)
    )


def _collect_cpu_cases():
    """Registers and collects the onnx-light-cpu ``test_cpu_*`` backend cases."""
    register_backend_test_cases()
    cases = []
    for op_type in _TARGET_KERNELS:
        for tc in collect_test_cases(op_type):
            if (
                tc.name.startswith("test_cpu_")
                and _single_node_op_type(tc) == op_type
                and tc.data_sets
                and _uses_supported_dtypes(tc)
            ):
                cases.append(tc)
    return cases


_CASES = _collect_cpu_cases()


def _assert_close(actual, expected, rtol, atol):
    if expected.dtype == np.bool_ or np.issubdtype(expected.dtype, np.integer):
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(
            actual.astype(np.float64), expected.astype(np.float64), rtol=rtol, atol=atol
        )


def _normalization_function_reference_model(onnx, model, op_type, dtype):
    """Expands low-precision normalization while preserving every graph type."""
    node = model.graph.node[0]
    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute
    }
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    nodes = []
    x_name = node.input[0]
    y_name = node.output[0]

    def constant(name, value, elem_type):
        tensor = onnx.TensorProto()
        tensor.name = name
        tensor.data_type = elem_type
        constant_dtype = np.float32 if elem_type == int(onnx.TensorProto.FLOAT) else dtype
        tensor.raw_data = np.asarray(value, dtype=constant_dtype).tobytes()
        nodes.append(onnx.helper.make_node("Constant", [], [name], value=tensor))

    def axes_input(axes):
        tensor = onnx.numpy_helper.from_array(np.asarray(axes, dtype=np.int64), "AxesValue")
        nodes.append(onnx.helper.make_node("Constant", [], ["Axes"], value=tensor))

    x_type = int(inputs[0].type.tensor_type.elem_type)
    if op_type == "MeanVarianceNormalization":
        axes = attributes.get("axes", [0, 2, 3])
        axes_input(axes)
        constant("Epsilon", 1.0e-9, int(onnx.TensorProto.FLOAT))
        nodes.extend(
            [
                onnx.helper.make_node(
                    "Cast", [x_name], ["XFloat"], to=int(onnx.TensorProto.FLOAT)
                ),
                onnx.helper.make_node("ReduceMean", ["XFloat", "Axes"], ["Mean"], keepdims=1),
                onnx.helper.make_node("Mul", ["XFloat", "XFloat"], ["Squared"]),
                onnx.helper.make_node(
                    "ReduceMean", ["Squared", "Axes"], ["MeanSquared"], keepdims=1
                ),
                onnx.helper.make_node("Mul", ["Mean", "Mean"], ["SquaredMean"]),
                onnx.helper.make_node("Sub", ["MeanSquared", "SquaredMean"], ["Variance"]),
                onnx.helper.make_node("Sqrt", ["Variance"], ["StdDev"]),
                onnx.helper.make_node("Add", ["StdDev", "Epsilon"], ["Denominator"]),
                onnx.helper.make_node("Sub", ["XFloat", "Mean"], ["Centered"]),
                onnx.helper.make_node("Div", ["Centered", "Denominator"], ["YFloat"]),
                onnx.helper.make_node("Cast", ["YFloat"], [y_name], to=x_type),
            ]
        )
    else:
        rank = len(inputs[0].type.tensor_type.shape.dim)
        axis = int(attributes.get("axis", -1))
        if axis < 0:
            axis += rank
        axes_input(range(axis, rank))
        epsilon = float(attributes.get("epsilon", 1.0e-5))
        constant("Epsilon", epsilon, int(onnx.TensorProto.FLOAT))
        nodes.append(
            onnx.helper.make_node("Cast", [x_name], ["XFloat"], to=int(onnx.TensorProto.FLOAT))
        )
        if op_type == "LayerNormalization":
            nodes.extend(
                [
                    onnx.helper.make_node("ReduceMean", ["XFloat", "Axes"], ["Mean"], keepdims=1),
                    onnx.helper.make_node("Sub", ["XFloat", "Mean"], ["Centered"]),
                    onnx.helper.make_node("Mul", ["Centered", "Centered"], ["Squared"]),
                    onnx.helper.make_node(
                        "ReduceMean", ["Squared", "Axes"], ["Variance"], keepdims=1
                    ),
                ]
            )
        else:
            nodes.extend(
                [
                    onnx.helper.make_node("Mul", ["XFloat", "XFloat"], ["Squared"]),
                    onnx.helper.make_node(
                        "ReduceMean", ["Squared", "Axes"], ["Variance"], keepdims=1
                    ),
                ]
            )
        nodes.extend(
            [
                onnx.helper.make_node("Add", ["Variance", "Epsilon"], ["Adjusted"]),
                onnx.helper.make_node("Sqrt", ["Adjusted"], ["Denominator"]),
                onnx.helper.make_node(
                    "Div",
                    ["Centered" if op_type == "LayerNormalization" else "XFloat", "Denominator"],
                    ["NormalizedFloat"],
                ),
                onnx.helper.make_node("Cast", ["NormalizedFloat"], ["NormalizedX"], to=x_type),
            ]
        )
        scale_type = int(inputs[1].type.tensor_type.elem_type)
        normalized = "NormalizedX"
        if scale_type != x_type:
            nodes.append(
                onnx.helper.make_node("Cast", [normalized], ["NormalizedScale"], to=scale_type)
            )
            normalized = "NormalizedScale"
        nodes.append(onnx.helper.make_node("Mul", [normalized, node.input[1]], ["Scaled"]))
        if op_type == "LayerNormalization" and len(inputs) == 3:
            nodes.append(onnx.helper.make_node("Add", ["Scaled", node.input[2]], [y_name]))
        else:
            nodes.append(onnx.helper.make_node("Identity", ["Scaled"], [y_name]))

    graph = onnx.helper.make_graph(nodes, f"{op_type}Reference", inputs, outputs)
    reference_model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 23)]
    )
    reference_model.ir_version = model.ir_version
    return reference_model


class TestBackendCases(TestCase):
    def setUp(self):
        register_kernels()

    def test_registered_kernel_names(self):
        assert registered_kernel_names() == _REGISTERED_KERNELS

    def test_registered_kernels_parity_with_names(self):
        records = registered_kernels()
        assert isinstance(records, tuple)
        assert all(isinstance(record, RegisteredKernel) for record in records)

        # registered_kernel_names() is derived from registered_kernels(), so
        # every op_type/kernel_name pair must match exactly.
        assert {record.op_type: record.kernel_name for record in records} == (
            registered_kernel_names()
        )

    def test_registered_kernels_is_deterministically_ordered(self):
        first = registered_kernels()
        second = registered_kernels()
        assert first == second
        sort_keys = [(r.domain, r.op_type, r.device, r.kernel_name) for r in first]
        assert sort_keys == sorted(sort_keys)

    def test_registered_kernels_are_immutable_and_complete(self):
        records = registered_kernels()
        assert records, "expected at least one registered kernel record"
        for record in records:
            with self.assertRaises(AttributeError):
                record.op_type = "Other"  # type: ignore[misc]
            if record.op_type == "TreeEnsemble":
                expected_domain = "ai.onnx.ml"
            elif record.op_type in {"BiasGelu", "CDist"}:
                expected_domain = "com.microsoft"
            else:
                expected_domain = "ai.onnx"
            assert record.domain == expected_domain
            assert record.device == "CPU"
            assert isinstance(record.types, tuple)
            assert record.types, record
            if record.op_type in {
                "Add",
                "And",
                "BiasGelu",
                "BitShift",
                "BitwiseAnd",
                "BitwiseOr",
                "BitwiseXor",
                "CDist",
                "Div",
                "Equal",
                "Greater",
                "GreaterOrEqual",
                "Less",
                "LessOrEqual",
                "Max",
                "Mean",
                "Min",
                "Mod",
                "Mul",
                "Or",
                "PRelu",
                "Pow",
                "Sub",
                "Sum",
                "Xor",
            }:
                assert isinstance(record.since_version, int)
                assert record.since_version >= 1
            elif record.op_type in {"Attention", "RMSNormalization"}:
                assert record.since_version == 23
            elif record.op_type == "BatchNormalization":
                assert record.since_version == 15
            elif record.op_type == "GroupNormalization":
                assert record.since_version == 18
            elif record.op_type in {"InstanceNormalization", "LpNormalization"}:
                assert record.since_version == 22
            elif record.op_type == "LayerNormalization":
                assert record.since_version == 17
            elif record.op_type == "MeanVarianceNormalization":
                assert record.since_version == 13
            elif record.op_type == "SwiGLU":
                assert record.since_version == 28
            elif record.op_type == "TreeEnsemble":
                assert record.since_version == 5
            else:
                assert record.since_version is None
            assert record.until_version is None

    def test_every_cpu_benchmark_has_a_registered_kernel(self):
        # Benchmark models are not necessarily single-node: the GEMM corpus
        # also chains several nodes in one model.
        benchmark_ops = {
            node.op_type
            for tc in collect_test_cases_by_name(
                _BENCHMARK_NAME_PATTERN, include_big=True, mode=TestMode.BENCHMARK
            )
            for node in tc.model.graph.node
        }
        assert benchmark_ops
        assert benchmark_ops <= set(_REGISTERED_KERNELS)

    def test_normalization_benchmarks_match_onnx_references(self):
        import onnx
        from onnx.reference import ReferenceEvaluator as OnnxReferenceEvaluator

        try:
            import onnxruntime as ort
        except ImportError:
            ort = None

        normalization_ops = {
            "BatchNormalization",
            "GroupNormalization",
            "InstanceNormalization",
            "LayerNormalization",
            "LpNormalization",
            "MeanVarianceNormalization",
            "RMSNormalization",
        }
        options = None
        if ort is not None:
            from onnxruntime.capi.onnxruntime_pybind11_state import (
                Fail as OrtFail,
                NotImplemented as OrtNotImplemented,
            )

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
        covered_dtypes = {op_type: set() for op_type in normalization_ops}
        for op_type in sorted(normalization_ops):
            cases = collect_test_cases(op_type, include_big=False, mode=TestMode.BENCHMARK)
            for tc in cases:
                dtype = next(
                    (
                        candidate
                        for candidate in ("float32", "float16", "bfloat16")
                        if tc.name.endswith(f"_{candidate}_benchmark")
                    ),
                    None,
                )
                if not tc.name.startswith("test_cpu_") or dtype is None:
                    continue
                with self.subTest(tc=tc.name):
                    covered_dtypes[op_type].add(dtype)
                    assert len(tc.model.graph.initializer) == 0
                    input_names = [vi.name for vi in tc.model.graph.input]
                    light_session = ReferenceEvaluator(tc.model)
                    model_bytes = tc.model.SerializeToString()
                    ort_session = None
                    if ort is not None:
                        try:
                            # GroupNormalization may be expanded from its ONNX function.
                            ort_session = ort.InferenceSession(
                                model_bytes,
                                sess_options=options,
                                providers=["CPUExecutionProvider"],
                            )
                        except OrtNotImplemented:
                            ort_session = None
                        except OrtFail as exc:
                            if "Type Error:" not in str(exc):
                                raise
                            ort_session = None
                    reference_session = None
                    for data_set in tc.data_sets:
                        feeds = {
                            name: _to_numpy(tensor)
                            for name, tensor in zip(input_names, data_set.inputs, strict=True)
                        }
                        got = light_session.run(None, feeds)
                        expected = None
                        if ort_session is not None:
                            try:
                                expected = ort_session.run(None, feeds)
                            except OrtNotImplemented:
                                ort_session = None
                            except OrtFail as exc:
                                if "Type Error:" not in str(exc):
                                    raise
                                ort_session = None
                        if expected is None:
                            if reference_session is None:
                                reference_model = onnx.load_from_string(model_bytes)
                                if (
                                    op_type == "MeanVarianceNormalization" and dtype != "float32"
                                ) or (
                                    op_type in {"LayerNormalization", "RMSNormalization"}
                                    and dtype == "bfloat16"
                                ):
                                    # The stock evaluator's MVN function has FLOAT
                                    # constants incompatible with low-precision X, and
                                    # its BF16 Layer/RMS implementations do not honor
                                    # FLOAT stash arithmetic. Expand the schema
                                    # computation without changing graph element types.
                                    reference_model = _normalization_function_reference_model(
                                        onnx, reference_model, op_type, feeds["X"].dtype
                                    )
                                reference_session = OnnxReferenceEvaluator(reference_model)
                            expected = reference_session.run(None, feeds)
                        assert len(got) == len(expected)
                        tolerance = 2e-5
                        if dtype in {"float16", "bfloat16"}:
                            tolerance = 2e-2
                        for actual, reference in zip(got, expected, strict=True):
                            _assert_close(actual, reference, rtol=tolerance, atol=tolerance)
        for op_type, dtypes in covered_dtypes.items():
            assert dtypes == {"float32", "float16", "bfloat16"}, (op_type, dtypes)

    def test_low_precision_affine_rounding_matches_onnx_functions(self):
        import onnx
        from onnx.reference import ReferenceEvaluator as OnnxReferenceEvaluator

        for op_type, version, x_shape, parameter_shape, attributes in (
            ("GroupNormalization", 21, [1, 1, 4], [1], {"num_groups": 1}),
            ("LayerNormalization", 17, [1, 4], [4], {}),
        ):
            for tensor_type, dtype in (
                (TensorProto.FLOAT16, np.float16),
                (TensorProto.BFLOAT16, ml_dtypes.bfloat16),
            ):
                with self.subTest(op_type=op_type, dtype=dtype):
                    node = helper.make_node(op_type, ["X", "Scale", "B"], ["Y"], **attributes)
                    graph = helper.make_graph(
                        [node],
                        "low_precision_affine_rounding",
                        [
                            helper.make_tensor_value_info("X", tensor_type, x_shape),
                            helper.make_tensor_value_info("Scale", tensor_type, parameter_shape),
                            helper.make_tensor_value_info("B", tensor_type, parameter_shape),
                        ],
                        [helper.make_tensor_value_info("Y", tensor_type, x_shape)],
                    )
                    model = helper.make_model(
                        graph, opset_imports=[helper.make_opsetid("", version)]
                    )
                    feeds = {
                        "X": np.asarray([-3.0, -2.0, -1.0, -0.5], dtype=dtype).reshape(x_shape),
                        "Scale": np.full(parameter_shape, 0.05, dtype=dtype),
                        "B": np.full(parameter_shape, -0.25, dtype=dtype),
                    }
                    actual = ReferenceEvaluator(model).run(None, feeds)[0]
                    reference_model = onnx.load_from_string(model.SerializeToString())
                    if op_type == "LayerNormalization":
                        reference_model = _normalization_function_reference_model(
                            onnx, reference_model, op_type, dtype
                        )
                    expected = OnnxReferenceEvaluator(reference_model).run(None, feeds)[0]
                    np.testing.assert_array_equal(
                        actual.view(np.uint16), expected.view(np.uint16)
                    )

    def test_group_normalization_opset18_double_matches_onnx_reference(self):
        import onnx
        from onnx.reference import ReferenceEvaluator as OnnxReferenceEvaluator

        node = helper.make_node(
            "GroupNormalization",
            ["X", "scale", "bias"],
            ["Y"],
            num_groups=1,
            epsilon=1.0,
        )
        graph = helper.make_graph(
            [node],
            "group_normalization_opset18_double",
            [
                helper.make_tensor_value_info("X", TensorProto.DOUBLE, [1, 1, 4]),
                helper.make_tensor_value_info("scale", TensorProto.DOUBLE, [1]),
                helper.make_tensor_value_info("bias", TensorProto.DOUBLE, [1]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.DOUBLE, [1, 1, 4])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        feeds = {
            "X": np.asarray(
                [10000000.1, 10000000.2, 10000000.3, 10000000.4],
                dtype=np.float64,
            ).reshape(1, 1, 4),
            "scale": np.asarray([1.0], dtype=np.float64),
            "bias": np.asarray([0.0], dtype=np.float64),
        }
        actual = ReferenceEvaluator(model).run(None, feeds)[0]
        expected = OnnxReferenceEvaluator(onnx.load_from_string(model.SerializeToString())).run(
            None, feeds
        )[0]
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3.0e-4)
        assert np.any(actual != 0.0)

    def test_every_target_op_has_backend_cases(self):
        """Guards against the parametrized test silently collecting nothing."""
        counts = dict.fromkeys(_TARGET_KERNELS, 0)
        for tc in _CASES:
            counts[_single_node_op_type(tc)] += 1
        for op_type, count in counts.items():
            assert count > 0, f"no onnx-light-cpu backend test cases collected for {op_type}"

    def test_backend_cases(self):
        """Runs one onnx-light-cpu backend test case through the accelerated
        kernels and checks the outputs match the reference and the accelerated
        kernel ran.
        """
        for tc in _CASES:
            with self.subTest(tc=tc.name):
                self._run_backend_case(tc)

    def _run_backend_case(self, tc):
        op_type = _single_node_op_type(tc)
        expected_kernel = _TARGET_KERNELS[op_type]

        sess = ReferenceEvaluator(tc.model)
        input_names = [vi.name for vi in tc.model.graph.input]
        rtol = tc.rtol if tc.rtol is not None else 1e-5
        atol = tc.atol if tc.atol is not None else 1e-6

        for ds in tc.data_sets:
            assert len(ds.inputs) == len(input_names)
            feeds = {name: _to_numpy(t) for name, t in zip(input_names, ds.inputs, strict=True)}

            clear_used_kernel_names()
            got = sess.run(None, feeds)
            assert expected_kernel in used_kernel_names()

            assert len(got) == len(ds.outputs)
            for actual, expected in zip(got, ds.outputs, strict=True):
                _assert_close(actual, _to_numpy(expected), rtol, atol)

    def test_gemm_uses_accelerated_kernel_benchmark_sizes(self):
        # Backs docs/examples/benchmarks/plot_gemm_benchmark.py: the timed ``Gemm`` curve
        # must dispatch to onnx-light-cpu (not onnx-light's built-in kernel) for
        # the same square shapes the benchmark measures, otherwise its timings
        # would be indistinguishable from the built-in baseline. The benchmark
        # verifies dispatch the same way, via onnx-light's per-session
        # ``ReferenceEvaluator.used_kernels()``.
        for size in (16, 64, 128):
            with self.subTest(size=size):
                register_kernels()
                rng = np.random.default_rng(0)
                a = rng.standard_normal((size, size)).astype(np.float32)
                b = rng.standard_normal((size, size)).astype(np.float32)
                graph = helper.make_graph(
                    [helper.make_node("Gemm", ["A", "B"], ["Y"], alpha=1.0, beta=1.0)],
                    "gemm",
                    [
                        helper.make_tensor_value_info("A", TensorProto.FLOAT, [size, size]),
                        helper.make_tensor_value_info("B", TensorProto.FLOAT, [size, size]),
                    ],
                    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                )
                model = helper.make_model(
                    graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=13
                )
                sess = ReferenceEvaluator(model)
                (y,) = sess.run(None, {"A": a, "B": b})
                # onnx-light-cpu overrides "Gemm" in onnx-light's dispatch table, so the
                # session's resolved kernels (reported as "<domain>:<op_type>") must
                # include the "Gemm" op onnx-light-cpu installed a kernel for.
                used_ops = {key.rsplit(":", 1)[-1] for key in sess.used_kernels()}
                assert "Gemm" in used_ops & set(registered_kernel_names()), sess.used_kernels()
                np.testing.assert_allclose(y, a @ b, rtol=1e-2, atol=1e-2)
