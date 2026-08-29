# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the ``plot_backend_cases_benchmark`` plot-data prep.

``plot_backend_cases_benchmark.py`` is a top-level script (sphinx-gallery
requires its example code at module scope, not tucked inside functions), so
importing it outright would run the whole benchmark, which needs onnx-light
and ONNX Runtime. Instead, the ``NoPlottableCasesError``/``PlotData``/
``prepare_plot_data`` definitions -- the pure part of the script responsible
for what the published chart draws -- are extracted with ``ast`` and
executed in isolation, so this test exercises the actual code from the
example without needing onnx-light or ONNX Runtime.
"""

import ast
from pathlib import Path
from unittest import TestCase

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_PATH = _ROOT / "docs" / "examples" / "benchmarks" / "plot_backend_cases_benchmark.py"
_PLOT_DATA_NAMES = {"NoPlottableCasesError", "PlotData", "_short_label", "prepare_plot_data"}


def _load_plot_data_helpers():
    tree = ast.parse(_EXAMPLE_PATH.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _PLOT_DATA_NAMES
    ]
    missing = _PLOT_DATA_NAMES - {node.name for node in nodes}
    assert not missing, f"{_EXAMPLE_PATH} no longer defines: {sorted(missing)}"
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    import matplotlib.pyplot as plt
    import numpy as np
    from dataclasses import dataclass

    # A single dict is used for both globals and locals (instead of a separate
    # locals dict) so the extracted definitions can resolve each other by
    # name -- e.g. ``prepare_plot_data`` looking up ``NoPlottableCasesError``
    # -- the same way they do as top-level statements in a real module.
    namespace = {"__builtins__": __builtins__, "dataclass": dataclass, "np": np, "plt": plt}
    exec(  # noqa: S102 -- extracting the example's own plot-data helpers, not user input.
        compile(module, str(_EXAMPLE_PATH), "exec"), namespace
    )
    return namespace


_helpers = _load_plot_data_helpers()
NoPlottableCasesError = _helpers["NoPlottableCasesError"]
prepare_plot_data = _helpers["prepare_plot_data"]


def _row(op_type, name, light_time=1.0, ort_time=None, ort_error=None):
    return (op_type, name, "1x1", "float32", light_time, ort_time, ort_error)


class TestPrepareBackendCasesPlotData(TestCase):
    def test_mixed_supported_and_unsupported_cases(self):
        rows = [
            _row("Abs", "test_cpu_abs_float32_benchmark", light_time=1.0, ort_time=2.0),
            _row(
                "Attention",
                "test_cpu_attention_streaming_benchmark",
                ort_error="unsupported input",
            ),
            _row("Gemm", "test_cpu_gemm_float64_benchmark", light_time=2.0, ort_time=1.0),
        ]

        plot_data = prepare_plot_data(rows)

        # The unsupported Attention row is excluded from the plotted values...
        self.assertEqual(len(plot_data.plotted_rows), 2)
        self.assertEqual(plot_data.labels, ["abs_float32", "gemm_float64"])
        self.assertEqual(list(plot_data.speedups), [2.0, 0.5])
        self.assertEqual(sorted(plot_data.colors_by_op_type), ["Abs", "Gemm"])
        # ... but it is not lost: callers keep the full ``rows`` list (with its
        # error) for the textual/table output.
        self.assertEqual(rows[1][6], "unsupported input")

    def test_zero_plottable_rows_raises(self):
        rows = [
            _row("Attention", "test_cpu_attention_a_benchmark", ort_error="rejected: a"),
            _row("Attention", "test_cpu_attention_b_benchmark", ort_error="rejected: b"),
        ]

        with self.assertRaises(NoPlottableCasesError) as ctx:
            prepare_plot_data(rows)

        message = str(ctx.exception)
        self.assertIn("test_cpu_attention_a_benchmark", message)
        self.assertIn("rejected: a", message)
        self.assertIn("test_cpu_attention_b_benchmark", message)
        self.assertIn("rejected: b", message)

    def test_no_rows_at_all_raises(self):
        with self.assertRaises(NoPlottableCasesError):
            prepare_plot_data([])

    def test_single_operator_still_produces_legend_entry(self):
        rows = [_row("Abs", "test_cpu_abs_float32_benchmark", light_time=1.0, ort_time=3.0)]

        plot_data = prepare_plot_data(rows)

        self.assertEqual(len(plot_data.plotted_rows), 1)
        self.assertEqual(list(plot_data.colors_by_op_type), ["Abs"])
