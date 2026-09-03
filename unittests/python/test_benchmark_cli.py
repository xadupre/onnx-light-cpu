# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``python -m onnx_light_cpu benchmark``."""

import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from onnx_light.ext_test_case import ExtTestCase
from openpyxl import load_workbook

from onnx_light_cpu.__main__ import _build_parser, main
from onnx_light_cpu import _benchmark
from onnx_light_cpu._benchmark import normalize_dtypes, write_benchmark_workbook


class TestBenchmarkCli(ExtTestCase):
    def test_parser_accepts_test_and_dtype_lists(self):
        args = _build_parser().parse_args(
            [
                "benchmark",
                "--test",
                "^test_cpu_abs_",
                "^test_cpu_gemm_",
                "--dtype",
                "float32,float64",
                "int64",
            ]
        )
        self.assertEqual(args.tests, ["^test_cpu_abs_", "^test_cpu_gemm_"])
        self.assertEqual(args.dtypes, ["float32,float64", "int64"])
        self.assertEqual(
            normalize_dtypes(args.dtypes),
            ("float32", "float64", "int64"),
        )

    def test_rejects_unknown_dtype(self):
        with self.assertRaisesRegex(ValueError, "unknown dtype"):
            normalize_dtypes(["complex128"])

    def test_rejects_invalid_test_regular_expression(self):
        with self.assertRaisesRegex(re.PatternError, "unterminated"):
            _benchmark.run_backend_benchmark(
                tests=["("],
                dtypes=["all"],
                repeat=1,
                warmup=0,
                max_repeat_time=1.0,
                threads=1,
            )

    def test_selects_backend_tests_and_dtypes(self):
        cases = [
            SimpleNamespace(name="test_cpu_abs_float32_benchmark", unload=mock.Mock()),
            SimpleNamespace(name="test_cpu_abs_float64_benchmark", unload=mock.Mock()),
            SimpleNamespace(name="test_cpu_gemm_float32_benchmark", unload=mock.Mock()),
        ]
        measured = (
            [{"case": cases[0].name, "duration_us": 1.0}],
            {"case": cases[0].name, "mean_us": 1.0},
        )
        with (
            mock.patch("onnx_light_cpu._benchmark.register_backend_test_cases"),
            mock.patch("onnx_light_cpu._benchmark.register_kernels"),
            mock.patch(
                "onnx_light.onnx.backend.collect_test_cases_by_name",
                return_value=cases,
            ),
            mock.patch(
                "onnx_light_cpu._benchmark._measure_case", return_value=measured
            ) as measure,
        ):
            raw, aggregated = _benchmark.run_backend_benchmark(
                tests=[r"^test_cpu_(abs|log)_"],
                dtypes=["float32"],
                repeat=1,
                warmup=0,
                max_repeat_time=1.0,
                threads=1,
            )
        self.assertEqual(raw, measured[0])
        self.assertEqual(aggregated, [measured[1]])
        measure.assert_called_once_with(cases[0], 1, 0, 1.0, 1)

    def test_writes_raw_and_aggregated_sheets(self):
        raw = [
            {
                "case": "test_cpu_abs_float32_benchmark",
                "operator": "Abs",
                "dtype": "float32",
                "iteration": 1,
                "duration_us": 12.5,
            }
        ]
        aggregated = [
            {
                "case": "test_cpu_abs_float32_benchmark",
                "operator": "Abs",
                "dtype": "float32",
                "samples": 1,
                "mean_us": 12.5,
                "stdev_us": 0.0,
                "min_us": 12.5,
                "p10_us": 12.5,
                "median_us": 12.5,
                "p90_us": 12.5,
                "max_us": 12.5,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "benchmark.xlsx"
            write_benchmark_workbook(output, raw, aggregated)
            workbook = load_workbook(output, read_only=True)
            self.assertEqual(workbook.sheetnames, ["raw", "aggregated"])
            self.assertEqual(workbook["raw"]["A2"].value, raw[0]["case"])
            self.assertEqual(workbook["aggregated"]["E2"].value, 12.5)
            workbook.close()

    def test_main_runs_benchmark_and_writes_output(self):
        rows = ([{"duration_us": 1.0}], [{"case": "one"}])
        with (
            mock.patch(
                "onnx_light_cpu.__main__.run_backend_benchmark",
                return_value=rows,
            ) as run,
            mock.patch("onnx_light_cpu.__main__.write_benchmark_workbook") as write,
        ):
            result = main(
                [
                    "benchmark",
                    "--tests",
                    r"^test_cpu_(abs|log)_",
                    "--dtypes",
                    "float32",
                    "--repeat",
                    "2",
                    "--warmup",
                    "0",
                    "--threads",
                    "1",
                    "--output",
                    "results.xlsx",
                ]
            )
        self.assertEqual(result, 0)
        run.assert_called_once_with(
            tests=[r"^test_cpu_(abs|log)_"],
            dtypes=["float32"],
            repeat=2,
            warmup=0,
            max_repeat_time=1.0,
            threads=1,
        )
        write.assert_called_once_with("results.xlsx", *rows)


if __name__ == "__main__":
    import unittest

    unittest.main()
