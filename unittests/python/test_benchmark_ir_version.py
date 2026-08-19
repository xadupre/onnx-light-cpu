# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Guard the ONNX IR version used by the benchmark example models.

The gallery benchmark scripts build their ONNX models with
:func:`onnx.helper.make_model`. When ``ir_version`` is not passed explicitly the
model inherits ``onnx.IR_VERSION`` from whatever ``onnx`` release is installed,
which is ``14`` on ``onnx>=1.23``. onnx-light (and the accelerated
onnx-light-cpu kernels) target IR version ``13``, so every generated model must
pin ``ir_version=13`` rather than relying on the default.

This test parses the benchmark example sources (without importing/running them)
and asserts that every ``make_model`` call explicitly sets ``ir_version=13``.
"""

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

# Benchmark scripts that generate ONNX models through ``helper.make_model``.
_BENCHMARK_SOURCES = [
    _ROOT / "docs" / "examples" / "benchmarks" / "plot_abs_benchmark.py",
    _ROOT / "docs" / "examples" / "benchmarks" / "plot_gemm_benchmark.py",
    _ROOT / "docs" / "examples" / "benchmarks" / "plot_gemm_dtype_benchmark.py",
    _ROOT / "tools" / "benchmark_gemm_parity.py",
]

_EXPECTED_IR_VERSION = 13


def _make_model_calls(tree):
    # Match attribute calls ending in ``.make_model`` (i.e. ``helper.make_model``
    # / ``onnx.helper.make_model``). Bare ``make_model(...)`` name calls are the
    # gallery scripts' own local wrappers; they are skipped on purpose because
    # the ``helper.make_model`` call *inside* such a wrapper is itself an
    # attribute call that this walk still reaches and checks, so no coverage is
    # lost. Every real ONNX-model construction therefore gets asserted.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "make_model":
                yield node


@pytest.mark.parametrize("source", _BENCHMARK_SOURCES, ids=lambda p: p.name)
def test_benchmark_pins_ir_version_13(source):
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    calls = list(_make_model_calls(tree))
    assert calls, f"No make_model call found in {source.name}"
    for call in calls:
        keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
        assert "ir_version" in keywords, (
            f"{source.name}:{call.lineno} make_model must pass ir_version explicitly "
            f"(otherwise it defaults to onnx.IR_VERSION, e.g. 14 on onnx>=1.23)"
        )
        value = keywords["ir_version"]
        pinned_to_13 = isinstance(value, ast.Constant) and value.value == _EXPECTED_IR_VERSION
        message = (
            f"{source.name}:{call.lineno} make_model must set ir_version={_EXPECTED_IR_VERSION}"
        )
        assert pinned_to_13, message
