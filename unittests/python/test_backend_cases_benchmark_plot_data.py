# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the ``plot_backend_cases_benchmark`` plot-data prep.

``plot_backend_cases_benchmark.py`` is a top-level script (sphinx-gallery
requires its example code at module scope, not tucked inside functions), so
importing it outright would run the whole benchmark, which needs onnx-light
and ONNX Runtime. Instead, the ``NoPlottableCasesError``/``PlotData``/
``prepare_plot_data``/``_case_group_key`` definitions -- the pure parts of
the script responsible for what the published chart draws and for which
cases ``--max-cases`` keeps -- are extracted with ``ast`` and executed in
isolation, so this test exercises the actual code from the example without
needing onnx-light or ONNX Runtime.
"""

import ast
import re
import time
from pathlib import Path
from onnx_light.ext_test_case import ExtTestCase

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_PATH = _ROOT / "docs" / "examples" / "benchmarks" / "plot_backend_cases_benchmark.py"
_PLOT_DATA_NAMES = {
    "NoPlottableCasesError",
    "PlotData",
    "_short_label",
    "prepare_plot_data",
    "_case_group_key",
    "_format_no_cases_message",
    "_CASE_GROUP_SUFFIXES",
    "measure",
    "timing_summary",
}


def _load_plot_data_helpers():
    tree = ast.parse(_EXAMPLE_PATH.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign))
        and (
            node.name in _PLOT_DATA_NAMES
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            else any(
                isinstance(target, ast.Name) and target.id in _PLOT_DATA_NAMES
                for target in node.targets
            )
        )
    ]
    found_names = {
        node.name for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    } | {
        target.id
        for node in nodes
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    missing = _PLOT_DATA_NAMES - found_names
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
    namespace = {
        "__builtins__": __builtins__,
        "dataclass": dataclass,
        "np": np,
        "plt": plt,
        "re": re,
        "time": time,
    }
    exec(  # noqa: S102 -- extracting the example's own plot-data helpers, not user input.
        compile(module, str(_EXAMPLE_PATH), "exec"), namespace
    )
    return namespace


_helpers = _load_plot_data_helpers()
NoPlottableCasesError = _helpers["NoPlottableCasesError"]
prepare_plot_data = _helpers["prepare_plot_data"]
_case_group_key = _helpers["_case_group_key"]
_format_no_cases_message = _helpers["_format_no_cases_message"]
measure = _helpers["measure"]
timing_summary = _helpers["timing_summary"]


def _row(
    op_type,
    name,
    light_time=1.0,
    ort_time=None,
    ort_error=None,
    light_p10=None,
    ort_p10=None,
    light_p90=None,
    ort_p90=None,
):
    return (
        op_type,
        name,
        "1x1",
        "float32",
        light_time,
        ort_time,
        ort_error,
        light_time if light_p10 is None else light_p10,
        light_time if light_p90 is None else light_p90,
        ort_time if ort_p10 is None else ort_p10,
        ort_time if ort_p90 is None else ort_p90,
        1,
        1 if ort_time is not None else None,
    )


class TestPrepareBackendCasesPlotData(ExtTestCase):
    def test_mixed_supported_and_unsupported_cases(self):
        rows = [
            _row(
                "Abs",
                "test_cpu_abs_float32_benchmark",
                light_time=1.0,
                ort_time=2.0,
                light_p10=0.5,
                ort_p10=0.75,
                light_p90=1.5,
                ort_p90=3.75,
            ),
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
        self.assertEqual(plot_data.p10_speedups[0], 1.5)
        self.assertEqual(plot_data.p10_speedups[1], 0.5)
        self.assertEqual(plot_data.p90_speedups[0], 2.5)
        self.assertEqual(plot_data.p90_speedups[1], 0.5)
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


class TestCaseGroupKey(ExtTestCase):
    """``_case_group_key`` must stay name-only: it is used to spread
    ``--max-cases`` truncation across operators *before* any case's ONNX
    model (and the tensors it references) is built, so it must not need
    ``tc.model`` -- doing so for every collected case (instead of only the
    small ``--max-cases`` subset) previously exhausted memory on CI runners.
    """

    def test_strips_dtype_and_shape_suffixes(self):
        self.assertEqual(_case_group_key("test_cpu_abs_float32_benchmark"), "abs")
        self.assertEqual(_case_group_key("test_cpu_abs_float64_benchmark"), "abs")
        self.assertEqual(_case_group_key("test_cpu_gemm_square_1024_benchmark"), "gemm_square")
        self.assertEqual(_case_group_key("test_cpu_gemm_skinny_m_benchmark"), "gemm_skinny_m")
        self.assertEqual(
            _case_group_key("test_cpu_not_n512_bool_benchmark"),
            "not",
        )

    def test_distinguishes_unrelated_operators(self):
        keys = {
            _case_group_key(name)
            for name in (
                "test_cpu_attention_streaming_float32_benchmark",
                "test_cpu_gemm_float32_benchmark",
                "test_cpu_matmul_float32_benchmark",
                "test_cpu_tree_ensemble_float32_benchmark",
            )
        }
        self.assertEqual(len(keys), 4)

    def test_no_model_access_required(self):
        # A plain string in, string out call: nothing here should ever touch
        # a ``TestCase`` object or its ``.model`` property.
        self.assertIsInstance(_case_group_key("test_cpu_abs_float32_benchmark"), str)


class TestNoCasesMessage(ExtTestCase):
    def test_uppercase_filter_suggests_lowercase(self):
        message = _format_no_cases_message(".*L.*")

        self.assertIn("usually lowercase", message)
        self.assertIn("case-sensitive", message)

    def test_lowercase_filter_has_no_case_hint(self):
        message = _format_no_cases_message(".*log.*")

        self.assertNotIn("usually lowercase", message)


class TestStableTiming(ExtTestCase):
    def test_batches_short_calls(self):
        calls = []

        def run():
            calls.append(None)

        samples, calls_per_sample = measure(
            run,
            repeat=3,
            warmup=1,
            max_duration=1.0,
            min_sample_duration=1e-9,
        )

        self.assertEqual(calls_per_sample, 1)
        self.assertEqual(len(samples), 3)
        self.assertEqual(len(calls), 7)

    def test_timing_summary_reports_median_and_percentiles(self):
        median, p10, p90 = timing_summary([1.0, 2.0, 3.0])

        self.assertEqual(median, 2.0)
        self.assertEqual(p10, 1.2)
        self.assertEqual(p90, 2.8)
