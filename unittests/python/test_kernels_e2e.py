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
    "GroupQueryAttention": "onnx_light_cpu::GroupQueryAttention",
    "Greater": "onnx_light_cpu::Greater",
    "GreaterOrEqual": "onnx_light_cpu::GreaterOrEqual",
    "InstanceNormalization": "onnx_light_cpu::InstanceNormalization",
    "LayerNormalization": "onnx_light_cpu::LayerNormalization",
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


def _node_attributes(node):
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _batch_normalization_reference(feeds, input_names, attributes):
    """Ports the ONNX ``BatchNormalization`` reference math (no stash_type)."""
    x, scale, bias, mean, var = (feeds[name] for name in input_names[:5])
    epsilon = float(attributes.get("epsilon", 1.0e-5))
    dim_ones = (1,) * (x.ndim - 2)

    def test_mode(m, v):
        s = scale.reshape(-1, *dim_ones)
        b = bias.reshape(-1, *dim_ones)
        m = m.reshape(-1, *dim_ones)
        v = v.reshape(-1, *dim_ones)
        return (s * (x - m) / np.sqrt(v + epsilon) + b).astype(x.dtype)

    if not int(attributes.get("training_mode", 0)):
        return [test_mode(mean, var)]

    reduce_axes = tuple(np.delete(np.arange(x.ndim), 1))
    saved_mean = x.mean(axis=reduce_axes)
    saved_var = x.var(axis=reduce_axes)
    momentum = float(attributes.get("momentum", 0.9))
    output_mean = mean * momentum + saved_mean * (1 - momentum)
    output_var = var * momentum + saved_var * (1 - momentum)
    y = test_mode(saved_mean, saved_var)
    return [y, output_mean.astype(x.dtype), output_var.astype(x.dtype)]


def _group_normalization_reference(feeds, input_names, attributes, version):
    """Ports the ONNX ``GroupNormalization`` function body (v18 or v21)."""
    x, scale, bias = (feeds[name] for name in input_names[:3])
    num_groups = int(attributes["num_groups"])
    epsilon = float(attributes.get("epsilon", 1.0e-5))
    n = x.shape[0]
    x_dtype = x.dtype
    # Stage one (mean/variance/normalization) always runs at FLOAT precision
    # for opset 21+ (default stash_type); opset 18 has no stash_type and
    # computes stage one directly at the input's own precision.
    stage_one = x.astype(np.float32) if version >= 21 else x
    x3d = stage_one.reshape(n, num_groups, -1)
    mean = x3d.mean(axis=2, keepdims=True)
    mean_of_square = (x3d * x3d).mean(axis=2, keepdims=True)
    variance = mean_of_square - mean * mean
    stddev = np.sqrt(variance + np.asarray(epsilon, dtype=stage_one.dtype))
    normalized = (x3d - mean) / stddev
    if version >= 21:
        # Stage two applies scale/bias per channel (shape (C,)), independent
        # of stash_type.
        normalized = normalized.reshape(x.shape).astype(x_dtype)
        affine_shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
    else:
        # Opset 18 applies scale/bias per group (shape (num_groups,)) before
        # reshaping back to the original channel layout.
        affine_shape = (1, num_groups, 1)
    y = normalized * scale.reshape(affine_shape) + bias.reshape(affine_shape)
    return [y.reshape(x.shape).astype(x_dtype)]


def _instance_normalization_reference(feeds, input_names, attributes):
    """Ports the ONNX ``InstanceNormalization`` reference math (no stash_type)."""
    x, scale, bias = (feeds[name] for name in input_names[:3])
    epsilon = float(attributes.get("epsilon", 1.0e-5))
    axes = tuple(range(2, x.ndim))
    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.var(x, axis=axes, keepdims=True)
    dim_ones = (1,) * (x.ndim - 2)
    s = scale.reshape(-1, *dim_ones)
    b = bias.reshape(-1, *dim_ones)
    y = s * (x - mean) / np.sqrt(var + epsilon) + b
    return [y.astype(x.dtype)]


def _layer_normalization_reference(feeds, input_names, attributes):
    """Ports ``LayerNormalization``'s stash_type=FLOAT default semantics."""
    x = feeds[input_names[0]]
    scale = feeds[input_names[1]]
    bias = feeds[input_names[2]] if len(input_names) > 2 else None
    axis = int(attributes.get("axis", -1))
    epsilon = float(attributes.get("epsilon", 1.0e-5))
    if axis < 0:
        axis += x.ndim
    axes = tuple(range(axis, x.ndim))
    x_dtype = x.dtype
    x32 = x.astype(np.float32)
    mean = x32.mean(axis=axes, keepdims=True)
    centered = x32 - mean
    variance = (centered * centered).mean(axis=axes, keepdims=True)
    denominator = np.sqrt(variance + np.float32(epsilon))
    normalized = (centered / denominator).astype(x_dtype)
    if scale.dtype != x_dtype:
        normalized = normalized.astype(scale.dtype)
    y = normalized * scale
    if bias is not None:
        y = y + bias
    return [y]


def _rms_normalization_reference(feeds, input_names, attributes):
    """Ports ``RMSNormalization``'s stash_type=FLOAT default semantics."""
    x = feeds[input_names[0]]
    scale = feeds[input_names[1]]
    axis = int(attributes.get("axis", -1))
    epsilon = float(attributes.get("epsilon", 1.0e-5))
    if axis < 0:
        axis += x.ndim
    axes = tuple(range(axis, x.ndim))
    x_dtype = x.dtype
    x32 = x.astype(np.float32)
    variance = (x32 * x32).mean(axis=axes, keepdims=True)
    denominator = np.sqrt(variance + np.float32(epsilon))
    normalized = (x32 / denominator).astype(x_dtype)
    if scale.dtype != x_dtype:
        normalized = normalized.astype(scale.dtype)
    return [normalized * scale]


def _lp_normalization_reference(feeds, input_names, attributes):
    """Ports the ONNX ``LpNormalization`` reference math."""
    x = feeds[input_names[0]]
    axis = int(attributes.get("axis", -1))
    p = int(attributes.get("p", 2))
    norm = np.power(np.power(x, p).sum(axis=axis), 1.0 / p)
    norm = np.expand_dims(norm, axis)
    y = np.where(norm == 0, 0, x / norm)
    return [y.astype(x.dtype)]


def _mean_variance_normalization_reference(feeds, input_names, attributes):
    """Ports the ONNX ``MeanVarianceNormalization`` function body (FLOAT stage)."""
    x = feeds[input_names[0]]
    axes = tuple(int(axis) for axis in attributes.get("axes", (0, 2, 3)))
    x_dtype = x.dtype
    x32 = x.astype(np.float32)
    mean = x32.mean(axis=axes, keepdims=True)
    mean_of_square = (x32 * x32).mean(axis=axes, keepdims=True)
    variance = mean_of_square - mean * mean
    std_dev = np.sqrt(variance)
    y = (x32 - mean) / (std_dev + np.float32(1.0e-9))
    return [y.astype(x_dtype)]


_NORMALIZATION_REFERENCES = {
    "BatchNormalization": _batch_normalization_reference,
    "InstanceNormalization": _instance_normalization_reference,
    "LayerNormalization": _layer_normalization_reference,
    "LpNormalization": _lp_normalization_reference,
    "MeanVarianceNormalization": _mean_variance_normalization_reference,
    "RMSNormalization": _rms_normalization_reference,
}


def _normalization_reference(op_type, node, feeds, group_normalization_version=21):
    """Computes the standard ONNX reference outputs for one normalization node.

    This mirrors the exact math from ONNX's native reference kernels (for ops
    with one, such as ``BatchNormalization``/``InstanceNormalization``) or from
    their ``FunctionBody``/``SetContextDependentFunctionBodyBuilder``
    decomposition (for ops expanded from an ONNX function, such as
    ``GroupNormalization``/``MeanVarianceNormalization``), computed directly
    with NumPy instead of building and running an ONNX graph.
    """
    attributes = _node_attributes(node)
    input_names = list(node.input)
    if op_type == "GroupNormalization":
        return _group_normalization_reference(
            feeds, input_names, attributes, group_normalization_version
        )
    return _NORMALIZATION_REFERENCES[op_type](feeds, input_names, attributes)


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
            elif record.op_type in {"BiasGelu", "CDist", "GroupQueryAttention"}:
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
                "GroupQueryAttention",
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
                    node = tc.model.graph.node[0]
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
                            expected = _normalization_reference(op_type, node, feeds)
                        assert len(got) == len(expected)
                        tolerance = 2e-5
                        if dtype in {"float16", "bfloat16"}:
                            tolerance = 2e-2
                        for actual, reference in zip(got, expected, strict=True):
                            _assert_close(actual, reference, rtol=tolerance, atol=tolerance)
        for op_type, dtypes in covered_dtypes.items():
            assert dtypes == {"float32", "float16", "bfloat16"}, (op_type, dtypes)

    def test_low_precision_affine_rounding_matches_onnx_functions(self):
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
                    expected = _normalization_reference(
                        op_type, node, feeds, group_normalization_version=version
                    )[0]
                    np.testing.assert_array_equal(
                        actual.view(np.uint16), expected.view(np.uint16)
                    )

    def test_group_normalization_opset18_double_matches_onnx_reference(self):
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
        expected = _normalization_reference(
            "GroupNormalization", node, feeds, group_normalization_version=18
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
