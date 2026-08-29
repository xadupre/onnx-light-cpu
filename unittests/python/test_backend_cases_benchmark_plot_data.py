# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the ``plot_backend_cases_benchmark`` plot-data prep.

These exercise :func:`prepare_plot_data` directly -- the pure function that
turns collected benchmark rows into what the published chart draws -- without
needing onnx-light or ONNX Runtime, unlike the example itself (see
``test_documentation_examples.py``).
"""

import importlib.util
import sys
from pathlib import Path
from unittest import TestCase

_ROOT = Path(__file__).resolve().parents[2]
_HELPER_PATH = _ROOT / "docs" / "examples" / "benchmarks" / "_bench_case_plot_data.py"


def _load_helper_module():
    spec = importlib.util.spec_from_file_location("_bench_case_plot_data", _HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_bench_case_plot_data = _load_helper_module()
NoPlottableCasesError = _bench_case_plot_data.NoPlottableCasesError
prepare_plot_data = _bench_case_plot_data.prepare_plot_data


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
