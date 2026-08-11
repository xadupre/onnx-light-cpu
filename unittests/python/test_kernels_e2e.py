# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Backend-test driven end-to-end tests for the onnx-light-cpu kernels.

onnx-light ships its ONNX backend test cases as a C++-registered registry
(``onnx_light/onnx_extensions/backend_test/cases/``) exposed to Python through
:func:`onnx_light.onnx.backend.collect_test_cases`. Each entry is a
``TestCase`` holding a ``ModelProto`` plus one or more reference input/output
``DataSet`` computed by onnx-light itself.

These tests take the ``Abs``, ``Exp``, ``Log`` and ``Gemm`` backend test cases
(covering every element-type combination onnx-light registers for those ops),
run each single-node model through onnx-light's ``ReferenceEvaluator`` after
:func:`onnx_light_cpu.register_kernels` has installed the SIMD-accelerated
kernels, and check that:

* the accelerated onnx-light-cpu kernel is the one actually dispatched to
  (via :func:`onnx_light_cpu.used_kernel_names`), and
* its outputs match the reference outputs shipped with the backend test case
  (using each case's ``rtol``/``atol``).

onnx-light, its backend-test extension, the ``_cpuregister`` extension (built
with ``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``) and ``ml_dtypes`` (for the
``bfloat16`` numpy dtype) are required; when any is missing the whole module is
skipped via :func:`pytest.importorskip`.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnx_light")
pytest.importorskip("onnx_light_cpu.onnx_py._cpuregister")
pytest.importorskip("ml_dtypes")
pytest.importorskip("onnx_light.onnx.backend")

import ml_dtypes
from onnx_light.onnx import TensorProto
from onnx_light.onnx.backend import collect_test_cases
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import (
    clear_used_kernel_names,
    register_kernels,
    registered_kernel_names,
    used_kernel_names,
)

# Operators whose onnx-light-cpu kernel we want to validate against every
# backend test case onnx-light registers for them, mapped to the
# library-qualified name each kernel records when it runs.
_TARGET_KERNELS = {
    "Abs": "onnx_light_cpu::Abs",
    "Exp": "onnx_light_cpu::Exp",
    "Log": "onnx_light_cpu::Log",
    "Gemm": "onnx_light_cpu::Gemm",
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


def _collect_cases():
    """Collects the single-node backend test cases for every target op."""
    cases = []
    for op_type in _TARGET_KERNELS:
        for tc in collect_test_cases(op_type):
            if _single_node_op_type(tc) == op_type and tc.data_sets:
                cases.append(tc)
    return cases


_CASES = _collect_cases()


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
        assert registered_kernel_names() == {
            "Abs": "onnx_light_cpu::Abs",
            "Exp": "onnx_light_cpu::Exp",
            "Log": "onnx_light_cpu::Log",
            "Gemm": "onnx_light_cpu::Gemm",
            "Not": "onnx_light_cpu::Not",
        }

    def test_every_target_op_has_backend_cases(self):
        """Guards against the parametrized test silently collecting nothing."""
        counts = dict.fromkeys(_TARGET_KERNELS, 0)
        for tc in _CASES:
            counts[_single_node_op_type(tc)] += 1
        for op_type, count in counts.items():
            assert count > 0, f"no single-node backend test cases collected for {op_type}"

    @pytest.mark.parametrize("tc", _CASES, ids=lambda tc: tc.name)
    def test_backend_case(self, tc):
        """Runs one backend test case through the accelerated kernels and
        checks the outputs match the reference and the accelerated kernel ran.
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
