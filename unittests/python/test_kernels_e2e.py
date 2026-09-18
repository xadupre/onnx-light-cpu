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
* generate one test per collected ``test_cpu_*`` case with
  :func:`onnx_light.onnx.backend.make_test_class`, run its model through
  onnx-light's ``ReferenceEvaluator``, and check that
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

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from threading import Barrier
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import onnxruntime

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto, helper, inliner
from onnx_light.onnx.backend import (
    TestMode,
    collect_test_cases,
    collect_test_cases_by_name,
    make_test_class,
)
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import (
    RegisteredKernel,
    clear_used_kernel_names,
    has_backend_test_cases,
    register_backend_test_cases,
    register_kernel_for_session,
    register_kernels,
    registered_kernel_names,
    registered_kernels,
    run_backend_correctness_tests,
    set_kernel_usage_recording,
    used_kernel_names,
)

assert has_backend_test_cases(), (
    "onnx-light-cpu must be built with onnx-light's backend test registry "
    "(register_backend_test_cases binding unavailable)."
)

_REGISTERED_KERNELS = registered_kernel_names()

_TARGET_KERNELS = dict(_REGISTERED_KERNELS)
_DOMAIN_SPECIFIC_TARGET_KERNELS = {
    ("ai.onnx", "LinearAttention"): _TARGET_KERNELS["LinearAttention"],
    (
        "com.microsoft",
        "LinearAttention",
    ): _TARGET_KERNELS["com.microsoft::LinearAttention"],
}

_BENCHMARK_TYPE_SUFFIXES = dict.fromkeys(_TARGET_KERNELS, "float32")
for _op_type in ("And", "Not", "Or", "Xor"):
    _BENCHMARK_TYPE_SUFFIXES[_op_type] = "bool"
for _op_type in ("BitwiseAnd", "BitwiseOr", "BitwiseXor", "MatMulInteger"):
    _BENCHMARK_TYPE_SUFFIXES[_op_type] = "int8"
_BENCHMARK_TYPE_SUFFIXES["BitShift"] = "uint8"
_BENCHMARK_TYPE_SUFFIXES["QLinearMatMul"] = "(?:int8|uint8)"
_BENCHMARK_TYPE_SUFFIXES["Sum"] = "(?:float32|float64)"
_BENCHMARK_TYPE_SUFFIXES["Mean"] = "(?:float32|float64)"
_BENCHMARK_TYPE_SUFFIXES["com.microsoft::LinearAttention"] = "float32"
for _op_type in {
    "BatchNormalization",
    "GroupNormalization",
    "GroupQueryAttention",
    "InstanceNormalization",
    "LayerNormalization",
    "LinearAttention",
    "LpNormalization",
    "MeanVarianceNormalization",
    "RMSNormalization",
}:
    _BENCHMARK_TYPE_SUFFIXES[_op_type] = "(?:float32|float16|bfloat16)"
_BENCHMARK_TYPE_SUFFIXES["SimplifiedLayerNormalization"] = "(?:float32|float64|float16|bfloat16)"
_BENCHMARK_TYPE_SUFFIXES["SkipSimplifiedLayerNormalization"] = "(?:float32|float16|bfloat16)"
_BENCHMARK_OP_TAGS = {
    "com.microsoft::LinearAttention": "microsoft_linear_attention",
    "GroupQueryAttention": "group_query_attention",
    "RMSNormalization": "rms_normalization",
    "SimplifiedLayerNormalization": "simplified_layer_normalization",
    "SkipSimplifiedLayerNormalization": "skip_simplified_layer_normalization",
}
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


register_backend_test_cases()


def _assert_close(actual, expected, rtol, atol):
    if expected.dtype == np.bool_ or np.issubdtype(expected.dtype, np.integer):
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(
            actual.astype(np.float64), expected.astype(np.float64), rtol=rtol, atol=atol
        )


def _matmul_nbits_numpy_reference(model, feeds):
    node = model.graph.node[0]
    attributes = {
        attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute
    }
    bits = attributes["bits"]
    block_size = attributes["block_size"]
    k = attributes["K"]
    n = attributes["N"]
    values_per_byte = 8 // bits
    mask = (1 << bits) - 1
    zero_point = 1 << (bits - 1)
    packed = feeds["B"]
    scales = np.asarray(feeds["scales"], dtype=np.float32)
    weights = np.empty((n, k), dtype=np.float32)
    for column in range(n):
        for index in range(k):
            block = index // block_size
            block_offset = index % block_size
            byte = packed[column, block, block_offset // values_per_byte]
            shift = (block_offset % values_per_byte) * bits
            quantized = (int(byte) >> shift) & mask
            weights[column, index] = (quantized - zero_point) * scales[column, block]
    output = np.asarray(feeds["A"], dtype=np.float32) @ weights.T
    if "bias" in feeds:
        output += np.asarray(feeds["bias"], dtype=np.float32)
    return output.astype(feeds["A"].dtype)


def _warm_builtin_session(model, feeds):
    """Builds a ``ReferenceEvaluator`` for ``model`` and runs it once on ``feeds``.

    ``onnx_light_cpu.register_kernels()`` permanently overrides onnx-light's
    process-wide ``KernelDispatchTable`` entries, but a session only resolves
    and caches which kernel it dispatches to on its *first* run (see
    ``docs/examples/benchmarks/plot_abs_benchmark.py``, which uses this same
    technique to compare onnx-light-cpu's accelerated kernels against
    onnx-light's own built-in ones). This module fully imports -- running this
    function at module scope -- before any test's ``setUp`` calls
    :func:`onnx_light_cpu.register_kernels`, so the returned session stays
    resolved to onnx-light's built-in reference kernel forever, regardless of
    later registrations. That built-in kernel is the oracle these tests
    compare onnx-light-cpu's accelerated kernels against; no kernel math is
    reimplemented here.
    """
    session = ReferenceEvaluator(model)
    session.run(None, feeds)
    return session


def _make_float_oracle_model(model):
    """Clones a single-node model with FLOAT inputs and outputs."""
    float_model = type(model)()
    float_model.ParseFromString(model.SerializeToString())
    for value_info in (*float_model.graph.input, *float_model.graph.output):
        value_info.type.tensor_type.elem_type = TensorProto.FLOAT
    return float_model


def _collect_normalization_benchmark_builtin_sessions():
    """Pre-warms a built-in FLOAT oracle per normalization benchmark case."""
    normalization_ops = {
        "BatchNormalization",
        "GroupNormalization",
        "InstanceNormalization",
        "LayerNormalization",
        "LpNormalization",
        "MeanVarianceNormalization",
        "RMSNormalization",
    }
    sessions = {}
    covered_dtypes = {op_type: set() for op_type in normalization_ops}
    for op_type in sorted(normalization_ops):
        for tc in collect_test_cases(op_type, include_big=False, mode=TestMode.BENCHMARK):
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
            covered_dtypes[op_type].add(dtype)
            input_names = [vi.name for vi in tc.model.graph.input]
            feeds = {
                name: _to_numpy(tensor)
                for name, tensor in zip(input_names, tc.data_sets[0].inputs, strict=True)
            }
            use_float_oracle = dtype != "float32"
            oracle_model = _make_float_oracle_model(tc.model) if use_float_oracle else tc.model
            oracle_feeds = (
                {name: value.astype(np.float32) for name, value in feeds.items()}
                if use_float_oracle
                else feeds
            )
            sessions[tc.name] = (
                _warm_builtin_session(oracle_model, oracle_feeds),
                use_float_oracle,
            )
    return sessions, covered_dtypes


(
    _NORMALIZATION_BENCHMARK_BUILTIN_SESSIONS,
    _NORMALIZATION_BENCHMARK_COVERED_DTYPES,
) = _collect_normalization_benchmark_builtin_sessions()


_LOW_PRECISION_AFFINE_CASES = (
    ("GroupNormalization", 21, [1, 1, 4], [1], {"num_groups": 1}),
    ("LayerNormalization", 17, [1, 4], [4], {}),
)
_ORT_MAX_RELEASED_ONNX_OPSET = 26
_ORT_MODEL_IR_VERSION = 13
_LOW_PRECISION_AFFINE_DTYPES = (
    (TensorProto.FLOAT16, np.float16),
    (TensorProto.BFLOAT16, ml_dtypes.bfloat16),
)


def _make_low_precision_affine_model_and_feeds(
    op_type, version, x_shape, parameter_shape, attributes, tensor_type, dtype
):
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", version)])
    feeds = {
        "X": np.asarray([-3.0, -2.0, -1.0, -0.5], dtype=dtype).reshape(x_shape),
        "Scale": np.full(parameter_shape, 0.05, dtype=dtype),
        "B": np.full(parameter_shape, -0.25, dtype=dtype),
    }
    return model, feeds


def _make_ort_compatible_model(model):
    ort_model = type(model)()
    ort_model.CopyFrom(model)
    if ort_model.ir_version > _ORT_MODEL_IR_VERSION:
        ort_model.ir_version = _ORT_MODEL_IR_VERSION
    for opset in ort_model.opset_import:
        if opset.domain in {"", "ai.onnx"} and opset.version > _ORT_MAX_RELEASED_ONNX_OPSET:
            opset.version = _ORT_MAX_RELEASED_ONNX_OPSET
    return ort_model


def _muse_ort_session(model):
    """ORT CPU lacks BF16 GQA: promote the already-rounded BF16 inputs to FLOAT.

    Muse fixtures disable RoPE, avoiding a different BF16 rotary rounding point.
    Integer sequence metadata stays INT32; no attention math is implemented here.
    """
    ort_model = _make_ort_compatible_model(model)
    for value_info in (*ort_model.graph.input, *ort_model.graph.output):
        if value_info.type.tensor_type.elem_type == TensorProto.BFLOAT16:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        ort_model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _muse_ort_feeds(feeds):
    return {
        name: value.astype(np.float32) if value.dtype == ml_dtypes.bfloat16 else value
        for name, value in feeds.items()
    }


def _muse_tolerances(dtype):
    if dtype == ml_dtypes.bfloat16:
        return 1e-2, 1e-3
    if dtype == np.float16:
        return 2e-3, 2e-4
    return 2e-4, 2e-5


def _collect_low_precision_affine_builtin_sessions():
    sessions = {}
    for op_type, version, x_shape, parameter_shape, attributes in _LOW_PRECISION_AFFINE_CASES:
        for tensor_type, dtype in _LOW_PRECISION_AFFINE_DTYPES:
            model, feeds = _make_low_precision_affine_model_and_feeds(
                op_type, version, x_shape, parameter_shape, attributes, tensor_type, dtype
            )
            function_model = inliner.inline_selected_functions(
                model, [("", op_type)], inline_schema_functions=True
            )
            sessions[op_type, dtype] = _warm_builtin_session(function_model, feeds)
    return sessions


_LOW_PRECISION_AFFINE_BUILTIN_SESSIONS = _collect_low_precision_affine_builtin_sessions()


def _make_group_normalization_opset18_double_model_and_feeds():
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
    return model, feeds


(
    _GROUP_NORMALIZATION_OPSET18_DOUBLE_MODEL,
    _GROUP_NORMALIZATION_OPSET18_DOUBLE_FEEDS,
) = _make_group_normalization_opset18_double_model_and_feeds()
_GROUP_NORMALIZATION_OPSET18_DOUBLE_FUNCTION_MODEL = inliner.inline_selected_functions(
    _GROUP_NORMALIZATION_OPSET18_DOUBLE_MODEL,
    [("", "GroupNormalization")],
    inline_schema_functions=True,
)
_GROUP_NORMALIZATION_OPSET18_DOUBLE_FUNCTION_MODEL.ir_version = 13


def _cpu_backend(model, *inputs):
    """Runs one generated backend case and verifies accelerated dispatch."""
    register_kernels()
    session = ReferenceEvaluator(model)
    feeds = dict(zip(session.input_names, inputs, strict=True))
    set_kernel_usage_recording(session, True)
    clear_used_kernel_names(session)
    outputs = session.run(None, feeds)
    dispatched = used_kernel_names(session)
    set_kernel_usage_recording(session, False)
    expected_kernels = []
    for node in model.graph.node:
        domain = node.domain or "ai.onnx"
        expected_kernel = _DOMAIN_SPECIFIC_TARGET_KERNELS.get(
            (domain, node.op_type), _TARGET_KERNELS.get(node.op_type)
        )
        assert expected_kernel is not None, (domain, node.op_type)
        expected_kernels.append(expected_kernel)
    assert dispatched == expected_kernels
    return outputs


TestCpuBackend = make_test_class(
    _cpu_backend,
    include_regex=["^test_cpu_"],
    exclude_regex=["^test_cpu_treeensemble(?:classifier|regressor)_"],
)


class TestExtensionImportOrder(ExtTestCase):
    def test_native_extension_import_orders(self):
        root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root), *(str(Path(path).resolve()) for path in sys.path)]
        )
        script = dedent("""
            import sys
            from importlib import import_module
            from pathlib import Path

            root = Path(sys.argv[1])
            for name in sys.argv[2:]:
                assert name not in sys.modules, f"{name} was loaded prematurely"
                module = import_module(name)
                print(name, module.__file__, flush=True)
                assert Path(module.__file__).resolve().parent == (
                    root / "onnx_light_cpu" / "onnx_py"
                ), module.__file__

            from onnx_light_cpu import has_cpu_kernels, register_kernels, registered_kernels

            assert has_cpu_kernels()
            register_kernels()
            assert registered_kernels()
            """)
        modules = (
            "onnx_light_cpu.onnx_py._cpukernels",
            "onnx_light_cpu.onnx_py._cpuregister",
        )
        # The temporary cwd ensures PYTHONPATH, not the working directory,
        # selects the source checkout. Each order gets a fresh dynamic loader.
        with TemporaryDirectory() as temporary:
            for cwd in (root, Path(temporary)):
                for order in (modules, modules[::-1]):
                    with self.subTest(cwd=str(cwd), order=order):
                        result = subprocess.run(
                            [sys.executable, "-c", script, str(root), *order],
                            cwd=cwd,
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=60,
                            check=False,
                        )
                        self.assertEqual(
                            result.returncode, 0, f"{result.stdout}\n{result.stderr}"
                        )


class TestKernelUsage(ExtTestCase):
    @staticmethod
    def _session(op_type="Abs"):
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node(op_type, ["X"], ["Y"])],
                "kernel_usage",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        session = ReferenceEvaluator(model, cpu_execution={"num_threads": 1})
        register_kernel_for_session(session, "", op_type)
        return session

    @staticmethod
    def _run(session):
        return session.run(None, {"X": np.array([-1, 2], dtype=np.float32)})[0]

    def test_recording_lifecycle_and_snapshot(self):
        session = self._session()
        np.testing.assert_array_equal(self._run(session), [1, 2])
        self.assertEqual(used_kernel_names(session), [])

        set_kernel_usage_recording(session, True)
        self._run(session)
        snapshot = used_kernel_names(session)
        self.assertEqual(snapshot, ["onnx_light_cpu::Abs"])
        snapshot.append("not a recorded kernel")
        self.assertEqual(used_kernel_names(session), ["onnx_light_cpu::Abs"])

        clear_used_kernel_names(session)
        self.assertEqual(used_kernel_names(session), [])
        self._run(session)
        self.assertEqual(used_kernel_names(session), ["onnx_light_cpu::Abs"])

        set_kernel_usage_recording(session, False)
        self._run(session)
        self.assertEqual(used_kernel_names(session), ["onnx_light_cpu::Abs"])
        clear_used_kernel_names(session)
        self._run(session)
        self.assertEqual(used_kernel_names(session), [])
        set_kernel_usage_recording(session, True)
        self._run(session)
        self.assertEqual(used_kernel_names(session), ["onnx_light_cpu::Abs"])

    def test_independent_sessions_do_not_share_recording_state(self):
        first = self._session()
        set_kernel_usage_recording(first, True)
        self._run(first)
        second = self._session("Exp")
        self._run(second)
        self.assertEqual(used_kernel_names(first), ["onnx_light_cpu::Abs"])
        self.assertEqual(used_kernel_names(second), [])

        set_kernel_usage_recording(second, True)
        self._run(second)
        clear_used_kernel_names(first)
        set_kernel_usage_recording(first, False)
        self._run(first)
        self._run(second)
        self.assertEqual(used_kernel_names(first), [])
        self.assertEqual(used_kernel_names(second), ["onnx_light_cpu::Exp"] * 2)

        clear_used_kernel_names(second)
        set_kernel_usage_recording(first, True)
        self._run(first)
        self.assertEqual(used_kernel_names(first), ["onnx_light_cpu::Abs"])
        self.assertEqual(used_kernel_names(second), [])

    def test_concurrent_sessions_clear_only_their_own_log(self):
        first = self._session()
        second = self._session("Exp")
        set_kernel_usage_recording(first, True)
        set_kernel_usage_recording(second, True)
        barrier = Barrier(2, timeout=30)

        def run(session, clear):
            for index in range(32):
                barrier.wait()
                self._run(session)
                if clear and index == 15:
                    clear_used_kernel_names(session)
            return used_kernel_names(session)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_result = executor.submit(run, first, True)
            second_result = executor.submit(run, second, False)
            self.assertEqual(first_result.result(), ["onnx_light_cpu::Abs"] * 16)
            self.assertEqual(second_result.result(), ["onnx_light_cpu::Exp"] * 32)

    def test_recording_retains_first_1024_invocations_until_clear(self):
        session = self._session()
        set_kernel_usage_recording(session, True)
        for _ in range(1030):
            self._run(session)
        self.assertEqual(used_kernel_names(session), ["onnx_light_cpu::Abs"] * 1024)
        clear_used_kernel_names(session)
        self._run(session)
        self.assertEqual(used_kernel_names(session), ["onnx_light_cpu::Abs"])

    def test_recording_requires_an_explicit_valid_session(self):
        for function in (used_kernel_names, clear_used_kernel_names):
            with self.subTest(function=function.__name__), self.assertRaises(TypeError):
                function()
        with self.assertRaises(TypeError):
            set_kernel_usage_recording()
        with self.assertRaises(TypeError):
            set_kernel_usage_recording(True)
        for session in (None, object(), True, SimpleNamespace(_ctx=object(), _runner=object())):
            for function, args in (
                (used_kernel_names, (session,)),
                (clear_used_kernel_names, (session,)),
                (set_kernel_usage_recording, (session, True)),
            ):
                with (
                    self.subTest(function=function.__name__, session=session),
                    self.assertRaises(TypeError),
                ):
                    function(*args)


class TestBackendCases(ExtTestCase):
    def setUp(self):
        register_kernels()

    def test_bias_gelu_benchmark_matches_onnx_runtime(self):
        from tools.benchmark_bias_gelu_parity import parse_args, run

        args = parse_args(
            [
                "--case",
                "empty",
                "--case",
                "avx512_after",
                "--case",
                "exceptional",
                "--repeat",
                "1",
                "--warmup",
                "0",
            ]
        )

        report = run(args)

        assert [row["case"] for row in report["results"]] == [
            "empty",
            "avx512_after",
            "exceptional",
        ]

    def test_matmul_nbits_backend_cases_match_reference(self):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        cases = [
            case
            for case in collect_test_cases("MatMulNBits", mode=TestMode.TEST)
            if case.name.startswith("test_cpu_matmulnbits_")
        ]
        self.assertEqual(len(cases), 9)
        for case in cases:
            with self.subTest(case=case.name):
                model = type(case.model)()
                model.ParseFromString(case.model.SerializeToString())
                model.ir_version = 10
                use_onnxruntime = (
                    model.graph.input[0].type.tensor_type.elem_type != TensorProto.BFLOAT16
                )
                oracle = (
                    onnxruntime.InferenceSession(
                        model.SerializeToString(),
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
                    if use_onnxruntime
                    else None
                )
                session = ReferenceEvaluator(model)
                register_kernel_for_session(session, "com.microsoft", "MatMulNBits")
                for dataset in case.data_sets:
                    feeds = {
                        value.name: _to_numpy(tensor)
                        for value, tensor in zip(model.graph.input, dataset.inputs, strict=True)
                    }
                    expected = (
                        oracle.run(None, feeds)
                        if oracle is not None
                        else [_matmul_nbits_numpy_reference(model, feeds)]
                    )
                    actual = session.run(None, feeds)
                    for output, reference in zip(actual, expected, strict=True):
                        tolerance = 2e-2 if output.dtype != np.float32 else 3e-5
                        _assert_close(output, reference, rtol=tolerance, atol=tolerance)

    def test_all_registered_kernels_pass_regular_backend_correctness_corpus(self):
        report = run_backend_correctness_tests()
        assert report.executed == report.passed
        assert not report.failed, report

    def test_registered_kernels_parity_with_names(self):
        records = registered_kernels()
        assert isinstance(records, tuple)
        assert all(isinstance(record, RegisteredKernel) for record in records)

        names = registered_kernel_names()
        for record in records:
            key = (
                f"{record.domain}::{record.op_type}"
                if f"{record.domain}::{record.op_type}" in names
                else record.op_type
            )
            assert names[key] == record.kernel_name

    def test_registered_kernels_is_deterministically_ordered(self):
        first = registered_kernels()
        second = registered_kernels()
        assert first == second
        sort_keys = [(r.domain, r.op_type, r.device, r.kernel_name) for r in first]
        assert sort_keys == sorted(sort_keys)

    def test_registered_kernels_are_immutable_and_complete(self):
        records = registered_kernels()
        assert records, "expected at least one registered kernel record"
        version_bounded_ops = {
            "Add",
            "And",
            "BitShift",
            "BitwiseAnd",
            "BitwiseOr",
            "BitwiseXor",
            "Div",
            "Equal",
            "Greater",
            "GreaterOrEqual",
            "Less",
            "LessOrEqual",
            "Mod",
            "MatMulNBits",
            "Mul",
            "Or",
            "PRelu",
            "Pow",
            "Sub",
            "Xor",
        }
        for record in records:
            with self.assertRaises(AttributeError):
                record.op_type = "Other"  # type: ignore[misc]
            if record.op_type == "TreeEnsemble":
                expected_domain = "ai.onnx.ml"
            elif record.domain == "com.microsoft":
                expected_domain = "com.microsoft"
            else:
                expected_domain = "ai.onnx"
            assert record.domain == expected_domain
            assert record.device == "CPU"
            assert isinstance(record.types, tuple)
            assert record.types, record
            if (record.domain == "com.microsoft" and record.op_type == "LinearAttention") or (
                record.op_type
                in {
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
                    "MatMulNBits",
                    "Mean",
                    "Min",
                    "Mod",
                    "Mul",
                    "Or",
                    "PRelu",
                    "Pow",
                    "SkipSimplifiedLayerNormalization",
                    "Sub",
                    "Sum",
                    "Xor",
                }
            ):
                assert isinstance(record.since_version, int)
                assert record.since_version >= 1
            elif record.op_type == "SimplifiedLayerNormalization":
                assert record.since_version == 1
            elif record.op_type in {"Attention", "RMSNormalization"}:
                assert record.since_version == 23
            elif record.op_type == "LinearAttention" and record.domain != "com.microsoft":
                assert record.since_version == 27
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
            if record.until_version is not None:
                assert record.op_type in version_bounded_ops
                assert isinstance(record.until_version, int)
                assert record.until_version >= record.since_version

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

    def test_benchmark_expected_outputs_are_opt_in(self):
        kwargs = {
            "mode": TestMode.BENCHMARK,
        }
        input_only = collect_test_cases_by_name("^test_cpu_not_", **kwargs)
        requested = collect_test_cases_by_name(
            "^test_cpu_not_", generate_benchmark_expected_outputs=True, **kwargs
        )
        assert input_only
        assert len(input_only) == len(requested)
        for case in input_only:
            assert not case.has_expected_outputs
            assert all(
                not data_set.expected_outputs_generated and not data_set.outputs
                for data_set in case.data_sets
            )
        for case in requested:
            assert case.has_expected_outputs
            assert all(
                data_set.expected_outputs_generated and data_set.outputs
                for data_set in case.data_sets
            )

    def test_group_query_attention_backend_case_matrix(self):
        supported_types = {
            int(TensorProto.FLOAT),
            int(TensorProto.FLOAT16),
            int(TensorProto.BFLOAT16),
        }
        for mode, expected_count in ((TestMode.TEST, 17), (TestMode.BENCHMARK, 15)):
            cases = [
                tc
                for tc in collect_test_cases("GroupQueryAttention", mode=mode)
                if tc.name.startswith("test_cpu_group_query_attention_")
            ]
            assert len(cases) == expected_count
            assert {int(tc.model.graph.input[0].type.tensor_type.elem_type) for tc in cases} == (
                supported_types
            )
            if mode == TestMode.BENCHMARK:
                names = {tc.name for tc in cases}
                for sequence in (1, 16, 128):
                    assert (
                        "test_cpu_group_query_attention_model_qwen3_8b_int4_mb_prefill_"
                        f"b1_s{sequence}_qh32_kvh8_hd128_pastlen0_causal_float32_benchmark"
                    ) in names
                for past_length in (16, 128, 1024):
                    assert (
                        "test_cpu_group_query_attention_model_qwen3_8b_int4_mb_decode_"
                        f"b1_s1_qh32_kvh8_hd128_pastlen{past_length}_causal_float32_benchmark"
                    ) in names
            else:
                names = {tc.name for tc in cases}
                assert "test_cpu_group_query_attention_cached_rotary_float32" in names
                cached_case = next(
                    tc
                    for tc in cases
                    if tc.name == "test_cpu_group_query_attention_cached_rotary_float32"
                )
                # The exact model contract: query/key/value, past_key/past_value,
                # seqlens_k/total_sequence_length, cos_cache/sin_cache wired, and
                # output/present_key/present_value all produced.
                assert len(cached_case.model.graph.input) == 9
                assert len(cached_case.model.graph.output) == 3

    def test_group_query_attention_benchmarks_match_onnx_runtime(self):
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime is unavailable")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        cases = [
            tc
            for tc in collect_test_cases("GroupQueryAttention", mode=TestMode.BENCHMARK)
            if tc.name.startswith("test_cpu_group_query_attention_model_qwen3_8b_int4_mb_")
        ]
        assert len(cases) == 6
        for tc in cases:
            with self.subTest(tc=tc.name):
                input_names = [vi.name for vi in tc.model.graph.input]
                light_session = ReferenceEvaluator(tc.model)
                ort_session = ort.InferenceSession(
                    _make_ort_compatible_model(tc.model).SerializeToString(),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                for data_set in tc.data_sets:
                    feeds = {
                        name: _to_numpy(tensor)
                        for name, tensor in zip(input_names, data_set.inputs, strict=True)
                    }
                    got = light_session.run(None, feeds)
                    expected = ort_session.run(None, feeds)
                    assert len(got) == len(expected) == 3
                    for actual, reference in zip(got, expected, strict=True):
                        _assert_close(actual, reference, rtol=2e-4, atol=2e-4)

    def test_group_query_attention_muse_fixtures_match_onnx_runtime(self):
        cases = collect_test_cases_by_name(
            "^test_cpu_group_query_attention_model_muse_", mode=TestMode.TEST
        )
        assert len(cases) == 12
        outputs = {}
        for tc in cases:
            with self.subTest(tc=tc.name):
                attributes = {
                    attr.name: helper.get_attribute_value(attr)
                    for attr in tc.model.graph.node[0].attribute
                }
                assert attributes["num_heads"] == 32
                assert attributes["kv_num_heads"] == 2
                assert attributes["local_window_size"] in (2048, -1)
                light_session = ReferenceEvaluator(tc.model)
                set_kernel_usage_recording(light_session, True)
                ort_session = _muse_ort_session(tc.model)
                for data_set in tc.data_sets:
                    feeds = {
                        vi.name: _to_numpy(tensor)
                        for vi, tensor in zip(tc.model.graph.input, data_set.inputs, strict=True)
                    }
                    rtol, atol = _muse_tolerances(feeds["query"].dtype)
                    clear_used_kernel_names()
                    got = light_session.run(None, feeds)
                    assert _TARGET_KERNELS["GroupQueryAttention"] in used_kernel_names()
                    expected = ort_session.run(None, _muse_ort_feeds(feeds))
                    assert len(got) == len(expected) == 3
                    for actual, reference, naive in zip(
                        got, expected, data_set.outputs, strict=True
                    ):
                        assert actual.dtype == feeds["query"].dtype
                        _assert_close(actual, reference, rtol=rtol, atol=atol)
                        _assert_close(actual, _to_numpy(naive), rtol=rtol, atol=atol)
                    past_length = feeds["past_key"].shape[2]
                    total_length = int(feeds["total_sequence_length"])
                    for actual, name in zip(got[1:], ("past_key", "past_value"), strict=True):
                        assert actual.shape == (1, 2, total_length, 128)
                        np.testing.assert_array_equal(actual[:, :, :past_length], feeds[name])
                    outputs[tc.name] = got[0].astype(np.float32)
        for name, local in outputs.items():
            if "_window2048_" not in name:
                continue
            full = outputs[name.replace("_window2048_", "_windowfull_")]
            assert np.max(np.abs(local - full)) > 0.01
            if "_cached_prefill_" in name:
                # The first row sees exactly 2048 keys, the later rows drop old keys.
                np.testing.assert_array_equal(local[:, 0], full[:, 0])

    def test_group_query_attention_muse_mixed_windows_consecutive_decode(self):
        cases = collect_test_cases_by_name(
            "^test_cpu_group_query_attention_model_muse_cached_prefill_.*_window2048_",
            mode=TestMode.TEST,
        )
        assert len(cases) == 3
        for tc in cases:
            with self.subTest(tc=tc.name):
                tensor_type = tc.model.graph.input[0].type.tensor_type.elem_type
                fixture_feeds = {
                    vi.name: _to_numpy(tensor)
                    for vi, tensor in zip(
                        tc.model.graph.input, tc.data_sets[0].inputs, strict=True
                    )
                }
                inputs = [
                    helper.make_tensor_value_info("query", tensor_type, [1, "sequence", 4096]),
                    helper.make_tensor_value_info("key", tensor_type, [1, "sequence", 256]),
                    helper.make_tensor_value_info("value", tensor_type, [1, "sequence", 256]),
                    helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [1]),
                    helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, []),
                ]
                nodes, outputs = [], []
                feeds = {
                    name: value
                    for name, value in fixture_feeds.items()
                    if name not in ("past_key", "past_value")
                }
                for layer, window in enumerate((2048, 2048, 2048, -1)):
                    node = type(tc.model.graph.node[0])()
                    node.CopyFrom(tc.model.graph.node[0])
                    for attr in node.attribute:
                        if attr.name == "local_window_size":
                            attr.i = window
                    for slot, name in ((3, "past_key"), (4, "past_value")):
                        node.input[slot] = f"{name}_{layer}"
                        inputs.append(
                            helper.make_tensor_value_info(
                                node.input[slot], tensor_type, [1, 2, "past_length", 128]
                            )
                        )
                        feeds[node.input[slot]] = fixture_feeds[name]
                    for slot, name in enumerate(("output", "present_key", "present_value")):
                        node.output[slot] = f"{name}_{layer}"
                        shape = (
                            [1, "sequence", 4096] if slot == 0 else [1, 2, "total_length", 128]
                        )
                        outputs.append(
                            helper.make_tensor_value_info(node.output[slot], tensor_type, shape)
                        )
                    nodes.append(node)
                model = helper.make_model(
                    helper.make_graph(nodes, "muse_mixed_windows", inputs, outputs),
                    opset_imports=[
                        helper.make_opsetid("", 23),
                        helper.make_opsetid("com.microsoft", 1),
                    ],
                )
                light_session = ReferenceEvaluator(model)
                ort_session = _muse_ort_session(model)
                ort_feeds = _muse_ort_feeds(feeds)
                rtol, atol = _muse_tolerances(feeds["query"].dtype)
                for step in range(3):
                    with self.subTest(step=step):
                        got = light_session.run(None, feeds)
                        expected = ort_session.run(None, ort_feeds)
                        assert len(got) == len(expected) == 12
                        for actual, reference in zip(got, expected, strict=True):
                            _assert_close(actual, reference, rtol=rtol, atol=atol)
                        for layer in (1, 2):
                            np.testing.assert_array_equal(got[0], got[3 * layer])
                        assert (
                            np.max(np.abs(got[0].astype(np.float32) - got[9].astype(np.float32)))
                            > 0.01
                        )
                        total_length = int(feeds["total_sequence_length"])
                        for layer in range(4):
                            for slot, name in ((1, "past_key"), (2, "past_value")):
                                cache = got[3 * layer + slot]
                                previous = feeds[f"{name}_{layer}"]
                                assert cache.shape == (1, 2, total_length, 128)
                                np.testing.assert_array_equal(
                                    cache[:, :, : previous.shape[2]], previous
                                )
                                np.testing.assert_array_equal(
                                    cache.astype(np.float32),
                                    expected[3 * layer + slot].astype(np.float32),
                                )
                                feeds[f"{name}_{layer}"] = cache
                                ort_feeds[f"{name}_{layer}"] = expected[3 * layer + slot]
                        # Feed each implementation's full present cache back to the
                        # same sessions for two single-token decode steps.
                        for name in ("query", "key", "value"):
                            feeds[name] = fixture_feeds[name][:, -1:].copy()
                            ort_feeds[name] = _muse_ort_feeds({name: feeds[name]})[name]
                        feeds["seqlens_k"] = np.array([total_length], dtype=np.int32)
                        feeds["total_sequence_length"] = np.array(
                            total_length + 1, dtype=np.int32
                        )
                        ort_feeds["seqlens_k"] = feeds["seqlens_k"]
                        ort_feeds["total_sequence_length"] = feeds["total_sequence_length"]

    def test_normalization_benchmarks_match_onnx_references(self):
        try:
            import onnxruntime as ort
        except ImportError:
            ort = None

        options = None
        if ort is not None:
            from onnxruntime.capi.onnxruntime_pybind11_state import (
                Fail as OrtFail,
                NotImplemented as OrtNotImplemented,
            )

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
        for op_type in sorted(_NORMALIZATION_BENCHMARK_COVERED_DTYPES):
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
                    assert len(tc.model.graph.initializer) == 0
                    input_names = [vi.name for vi in tc.model.graph.input]
                    light_session = ReferenceEvaluator(tc.model)
                    model_bytes = tc.model.SerializeToString()
                    ort_session = None
                    # ORT's NumPy run API cannot accept ml_dtypes.bfloat16 feeds.
                    # Use the pre-warmed onnx-light oracle for those cases.
                    if ort is not None and dtype != "bfloat16":
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
                    (
                        builtin_session,
                        builtin_uses_float,
                    ) = _NORMALIZATION_BENCHMARK_BUILTIN_SESSIONS[tc.name]
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
                            except RuntimeError as exc:
                                if "can't be converted to MLDataType" not in str(exc):
                                    raise
                                ort_session = None
                        if expected is None:
                            # Fall back to onnx-light's own built-in (un-accelerated)
                            # kernel, resolved and cached before onnx-light-cpu's
                            # kernels were ever registered (see
                            # ``_warm_builtin_session``).
                            builtin_feeds = (
                                {name: value.astype(np.float32) for name, value in feeds.items()}
                                if builtin_uses_float
                                else feeds
                            )
                            expected = builtin_session.run(None, builtin_feeds)
                        assert len(got) == len(expected)
                        tolerance = 2e-5
                        if dtype in {"float16", "bfloat16"}:
                            tolerance = 2e-2
                        for actual, reference in zip(got, expected, strict=True):
                            _assert_close(actual, reference, rtol=tolerance, atol=tolerance)
        for op_type, dtypes in _NORMALIZATION_BENCHMARK_COVERED_DTYPES.items():
            assert dtypes == {"float32", "float16", "bfloat16"}, (op_type, dtypes)

    def test_low_precision_affine_rounding_matches_onnx_functions(self):
        for op_type, version, x_shape, parameter_shape, attributes in _LOW_PRECISION_AFFINE_CASES:
            for tensor_type, dtype in _LOW_PRECISION_AFFINE_DTYPES:
                with self.subTest(op_type=op_type, dtype=dtype):
                    model, feeds = _make_low_precision_affine_model_and_feeds(
                        op_type, version, x_shape, parameter_shape, attributes, tensor_type, dtype
                    )
                    actual = ReferenceEvaluator(model).run(None, feeds)[0]
                    # Reference: onnx-light's own built-in (un-accelerated) kernel,
                    # resolved and cached before onnx-light-cpu's kernels were ever
                    # registered (see ``_warm_builtin_session``).
                    builtin_session = _LOW_PRECISION_AFFINE_BUILTIN_SESSIONS[op_type, dtype]
                    expected = builtin_session.run(None, feeds)[0]
                    np.testing.assert_array_equal(
                        actual.view(np.uint16), expected.view(np.uint16)
                    )

    def test_group_normalization_opset18_double_matches_onnx_reference(self):
        model = _GROUP_NORMALIZATION_OPSET18_DOUBLE_MODEL
        feeds = _GROUP_NORMALIZATION_OPSET18_DOUBLE_FEEDS
        actual = ReferenceEvaluator(model).run(None, feeds)[0]
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime is unavailable")
        expected = ort.InferenceSession(
            _GROUP_NORMALIZATION_OPSET18_DOUBLE_FUNCTION_MODEL.SerializeToString(),
            providers=["CPUExecutionProvider"],
        ).run(None, feeds)[0]
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3.0e-4)
        assert np.any(actual != 0.0)

    def test_log_benchmark_inputs_are_positive(self):
        cases = collect_test_cases_by_name(
            "^test_cpu_log_n65536_float32_benchmark$", mode=TestMode.BENCHMARK
        )
        assert len(cases) == 1
        values = _to_numpy(cases[0].data_sets[0].inputs[0])
        assert np.all(values > 0)

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
