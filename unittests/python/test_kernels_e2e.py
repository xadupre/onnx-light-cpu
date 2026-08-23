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
NumPy; ``BFLOAT16`` cases are excluded.
"""

from __future__ import annotations

import numpy as np
import pytest

from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.backend import collect_test_cases
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
    "BitShift": "onnx_light_cpu::BitShift",
    "BitwiseAnd": "onnx_light_cpu::BitwiseAnd",
    "BitwiseOr": "onnx_light_cpu::BitwiseOr",
    "BitwiseXor": "onnx_light_cpu::BitwiseXor",
    "Div": "onnx_light_cpu::Div",
    "Equal": "onnx_light_cpu::Equal",
    "Exp": "onnx_light_cpu::Exp",
    "Gemm": "onnx_light_cpu::Gemm",
    "Greater": "onnx_light_cpu::Greater",
    "GreaterOrEqual": "onnx_light_cpu::GreaterOrEqual",
    "Less": "onnx_light_cpu::Less",
    "LessOrEqual": "onnx_light_cpu::LessOrEqual",
    "Log": "onnx_light_cpu::Log",
    "MatMul": "onnx_light_cpu::MatMul",
    "MatMulInteger": "onnx_light_cpu::MatMulInteger",
    "Mod": "onnx_light_cpu::Mod",
    "Mul": "onnx_light_cpu::Mul",
    "Not": "onnx_light_cpu::Not",
    "Or": "onnx_light_cpu::Or",
    "PRelu": "onnx_light_cpu::PRelu",
    "Pow": "onnx_light_cpu::Pow",
    "QLinearMatMul": "onnx_light_cpu::QLinearMatMul",
    "Sub": "onnx_light_cpu::Sub",
    "Xor": "onnx_light_cpu::Xor",
}

_TARGET_KERNELS = {
    op_type: kernel_name
    for op_type, kernel_name in _REGISTERED_KERNELS.items()
    if op_type != "QLinearMatMul"
}

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


class TestBackendCases:
    def setup_method(self):
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
            with pytest.raises(AttributeError):
                record.op_type = "Other"  # type: ignore[misc]
            assert record.domain == "ai.onnx"
            assert record.device == "CPU"
            assert isinstance(record.types, tuple)
            assert record.types, record
            if record.op_type in {
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
                "Mul",
                "Or",
                "PRelu",
                "Pow",
                "Sub",
                "Xor",
            }:
                assert isinstance(record.since_version, int)
                assert record.since_version >= 1
            else:
                assert record.since_version is None
            assert record.until_version is None

    def test_every_target_op_has_backend_cases(self):
        """Guards against the parametrized test silently collecting nothing."""
        counts = dict.fromkeys(_TARGET_KERNELS, 0)
        for tc in _CASES:
            counts[_single_node_op_type(tc)] += 1
        for op_type, count in counts.items():
            assert count > 0, f"no onnx-light-cpu backend test cases collected for {op_type}"

    @pytest.mark.parametrize("tc", _CASES, ids=lambda tc: tc.name)
    def test_backend_case(self, tc):
        """Runs one onnx-light-cpu backend test case through the accelerated
        kernels and checks the outputs match the reference and the accelerated
        kernel ran.
        """
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

    @pytest.mark.parametrize("size", [16, 64, 128])
    def test_gemm_uses_accelerated_kernel_benchmark_sizes(self, size):
        # Backs docs/examples/benchmarks/plot_gemm_benchmark.py: the timed ``Gemm`` curve
        # must dispatch to onnx-light-cpu (not onnx-light's built-in kernel) for
        # the same square shapes the benchmark measures, otherwise its timings
        # would be indistinguishable from the built-in baseline. The benchmark
        # verifies dispatch the same way, via onnx-light's per-session
        # ``ReferenceEvaluator.used_kernels()``.
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
