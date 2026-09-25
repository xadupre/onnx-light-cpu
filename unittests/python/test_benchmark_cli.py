# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``python -m onnx_light_cpu benchmark``."""

import io
import re
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from onnx_light.ext_test_case import ExtTestCase
from openpyxl import load_workbook

from onnx_light_cpu.__main__ import _build_parser, main
from onnx_light_cpu import __main__ as cli
from onnx_light_cpu import _benchmark
from onnx_light_cpu._benchmark import (
    compare_benchmark_dtypes,
    infer_pr_benchmark_selection,
    normalize_dtypes,
    post_benchmark_markdown,
    pull_request_benchmark_selection,
    write_benchmark_markdown,
    write_pr_benchmark_markdown,
    write_benchmark_workbook,
)
from onnx_light_cpu._register import _kernel_names_only


class TestBenchmarkCli(ExtTestCase):
    @staticmethod
    def _kernel(op_type, types):
        return SimpleNamespace(op_type=op_type, types=types)

    @staticmethod
    def _registration(domain, op_type, kernel_name):
        return SimpleNamespace(domain=domain, op_type=op_type, kernel_name=kernel_name)

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
                "--onnxruntime",
                "--markdown",
                "results.md",
                "--pr-markdown",
                "pr-results.md",
                "--pr",
                "623",
                "--from-pr",
                "624",
            ]
        )
        self.assertEqual(args.tests, ["^test_cpu_abs_", "^test_cpu_gemm_"])
        self.assertEqual(args.dtypes, ["float32,float64", "int64"])
        self.assertTrue(args.onnxruntime)
        self.assertEqual(args.markdown, "results.md")
        self.assertEqual(args.pr_markdown, "pr-results.md")
        self.assertEqual(args.pr, "623")
        self.assertEqual(args.from_pr, "624")
        self.assertEqual(
            normalize_dtypes(args.dtypes),
            ("float32", "float64", "int64"),
        )
        self.assertEqual(_build_parser().parse_args(["benchmark", "--pr"]).pr, "")

    def test_parser_uses_automatic_thread_count_by_default(self):
        parser = _build_parser()
        self.assertEqual(parser.parse_args(["benchmark"]).threads, 0)
        self.assertEqual(parser.parse_args(["benchmark", "--threads", "0"]).threads, 0)
        help_output = io.StringIO()
        with redirect_stdout(help_output), self.assertRaises(SystemExit):
            parser.parse_args(["benchmark", "--help"])
        self.assertIn("kernel select defaults (default: 0)", help_output.getvalue())

    def test_parser_compares_two_dtypes(self):
        args = _build_parser().parse_args(
            ["benchmark", "--compare-dtypes", "float16", "bfloat16"]
        )
        self.assertEqual(args.compare_dtypes, ["float16", "bfloat16"])
        self.assertEqual(args.dtypes, [])
        self.assertIsNone(_build_parser().parse_args(["benchmark"]).compare_dtypes)
        for options in (
            ["--compare-dtypes", "float16"],
            ["--compare-dtypes", "float16", "all"],
            ["--compare-dtypes", "float16", "complex128"],
            ["--compare-dtypes", "float16", "float16"],
            ["--compare-dtypes", "float16", "bfloat16", "--dtype", "float32"],
        ):
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["benchmark", *options])
                self.assertEqual(error.exception.code, 2)

    @staticmethod
    def _dtype_rows():
        return [
            {
                **dict.fromkeys(_benchmark._AGGREGATED_COLUMNS),
                "case": f"test_cpu_abs_n1024_{dtype}_benchmark",
                "operator": "Abs",
                "input_shapes": '[{"x": [1024]}]',
                "dtype": dtype,
                "median_s": median,
                "onnxruntime_median_s": 100.0,
                "speedup": 50.0,
            }
            for dtype, median in (("bfloat16", 2.0), ("float16", 6.0), ("float32", 1.0))
        ]

    def test_compares_matching_dtype_medians(self):
        rows = self._dtype_rows()
        original = [row.copy() for row in rows]
        comparison = compare_benchmark_dtypes(rows, "float16", "bfloat16")
        self.assertEqual(
            comparison,
            [
                {
                    "case": "test_cpu_abs_n1024_float16/bfloat16_benchmark",
                    "operator": "Abs",
                    "input_shapes": '[{"x": [1024]}]',
                    "baseline_dtype": "float16",
                    "comparison_dtype": "bfloat16",
                    "baseline_median_s": 6.0,
                    "comparison_median_s": 2.0,
                    "speedup": 3.0,
                }
            ],
        )
        self.assertEqual(rows, original)
        reverse = compare_benchmark_dtypes(rows, "bfloat16", "float16")
        self.assertEqual(reverse[0]["speedup"], 1 / 3)
        self.assertEqual(compare_benchmark_dtypes([], "float16", "bfloat16"), [])
        for baseline, comparison in (("float16", "float16"), ("all", "bfloat16")):
            with self.assertRaisesRegex(ValueError, "two different supported dtypes"):
                compare_benchmark_dtypes(rows, baseline, comparison)

    def test_dtype_comparison_does_not_pair_different_cases_or_shapes(self):
        for changes in (
            {"case": "test_cpu_abs_n2048_float16_benchmark"},
            {"input_shapes": '[{"x": [2048]}]'},
            {"operator": "Neg"},
        ):
            rows = self._dtype_rows()[:2]
            rows[1].update(changes)
            with self.subTest(changes=changes):
                comparison = compare_benchmark_dtypes(rows, "float16", "bfloat16")
                self.assertEqual(len(comparison), 2)
                self.assertTrue(all(row["speedup"] is None for row in comparison))
                self.assertIsNone(comparison[0]["baseline_median_s"])
                self.assertIsNone(comparison[1]["comparison_median_s"])

    def test_dtype_comparison_excludes_mixed_dtype_cases(self):
        rows = [
            {
                **row,
                "case": row["case"].replace("abs_n1024", "cast_float32_to"),
                "operator": "Cast",
            }
            for row in self._dtype_rows()
        ]
        self.assertEqual(compare_benchmark_dtypes(rows, "float16", "bfloat16"), [])

    def test_dtype_comparison_pairs_homogeneous_binary_cases(self):
        rows = [
            {
                **row,
                "case": (
                    f"test_cpu_add_v14_equal_{row['dtype']}x{row['dtype']}"
                    f"_to_{row['dtype']}_n1024_benchmark"
                ),
                "operator": "Add",
            }
            for row in self._dtype_rows()
        ]
        comparison = compare_benchmark_dtypes(rows, "float16", "bfloat16")
        self.assertEqual(len(comparison), 1)
        self.assertEqual(comparison[0]["speedup"], 3.0)
        for row in rows:
            row["case"] = row["case"].replace(f"{row['dtype']}x", "int32x")
        self.assertEqual(compare_benchmark_dtypes(rows, "float16", "bfloat16"), [])

    def test_dtype_comparison_zero_duration(self):
        rows = self._dtype_rows()
        rows[0]["median_s"] = 0.0
        comparison = compare_benchmark_dtypes(rows, "float16", "bfloat16")
        self.assertEqual(comparison[0]["speedup"], float("inf"))

    def test_main_writes_dtype_comparison_reports(self):
        calls = []
        rows = self._dtype_rows()[:2]

        def run(**kwargs):
            calls.append(kwargs)
            return [], rows

        def infer(pull_request):
            self.assertEqual(pull_request, "763")
            return ["^test_cpu_abs_"], ["float32"]

        self.addCleanup(setattr, cli, "run_backend_benchmark", cli.run_backend_benchmark)
        self.addCleanup(
            setattr, cli, "pull_request_benchmark_selection", cli.pull_request_benchmark_selection
        )
        cli.run_backend_benchmark = run
        cli.pull_request_benchmark_selection = infer
        for selection in (["--tests", "^test_cpu_abs_"], ["--from-pr", "763"]):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "benchmark.xlsx"
                markdown = Path(temporary) / "benchmark.md"
                pr_markdown = Path(temporary) / "pr.md"
                with redirect_stdout(io.StringIO()):
                    result = main(
                        [
                            "benchmark",
                            *selection,
                            "--compare-dtypes",
                            "float16",
                            "bfloat16",
                            "--onnxruntime",
                            "--output",
                            str(output),
                            "--markdown",
                            str(markdown),
                            "--pr-markdown",
                            str(pr_markdown),
                        ]
                    )
                self.assertEqual(result, 0)
                self.assertEqual(calls[-1]["dtypes"], ["float16", "bfloat16"])
                self.assertEqual(calls[-1]["tests"], ["^test_cpu_abs_"])
                self.assertTrue(calls[-1]["with_onnxruntime"])
                workbook = load_workbook(output, read_only=True)
                self.assertEqual(workbook.sheetnames, ["raw", "aggregated", "dtype_comparison"])
                values = list(workbook["dtype_comparison"].values)
                self.assertEqual(values[0], _benchmark._DTYPE_COMPARISON_COLUMNS)
                self.assertEqual(values[1][-3:], (6.0, 2.0, 3.0))
                self.assertEqual(workbook["aggregated"].max_row, 3)
                workbook.close()
                report = markdown.read_text(encoding="utf-8")
                self.assertIn("## Dtype comparison (baseline / comparison)", report)
                self.assertIn("| float16 | bfloat16 | 6.0 | 2.0 | 3.0 |", report)
                self.assertIn(
                    "| 3.00 | test_cpu_abs_n1024_float16/bfloat16_benchmark |",
                    pr_markdown.read_text(encoding="utf-8"),
                )

    def test_rejects_unknown_dtype(self):
        with self.assertRaisesRegex(ValueError, "unknown dtype"):
            normalize_dtypes(["complex128"])

    def test_kernel_names_exclude_implementation_paths(self):
        recorded = [
            "onnx_light_cpu::Cast",
            "Cast.float32_to_float16.f16c",
            "Binary.Greater.compare64.avx2",
            "onnx_light_cpu::SimplifiedLayerNormalization/avx/row-scale",
            "onnx_light_cpu::Abs",
            "onnx_light_cpu::Attention",
            "Attention.tiled",
            "Attention.conversion.tile",
            "Attention.packing.tile",
        ]
        self.assertEqual(
            _kernel_names_only(recorded),
            ["onnx_light_cpu::Cast", "onnx_light_cpu::Abs", "onnx_light_cpu::Attention"],
        )

    def test_rejects_invalid_test_regular_expression(self):
        with self.assertRaisesRegex(re.PatternError, "unterminated"):
            _benchmark.run_backend_benchmark(
                tests=["("],
                dtypes=["all"],
                repeat=1,
                warmup=0,
                max_repeat_time=1.0,
                threads=1,
                with_onnxruntime=False,
            )

    def test_no_matching_backend_tests_returns_empty_results(self):
        with (
            mock.patch("onnx_light_cpu._benchmark.register_backend_test_cases"),
            mock.patch("onnx_light_cpu._benchmark.register_kernels"),
            mock.patch("onnx_light.onnx.backend.collect_test_cases_by_name", return_value=[]),
        ):
            rows = _benchmark.run_backend_benchmark(
                tests=[r"^test_cpu_no_such_operator_"],
                dtypes=["all"],
                repeat=1,
                warmup=0,
                max_repeat_time=1.0,
                threads=0,
                with_onnxruntime=False,
                allow_empty=True,
            )
        self.assertEqual(rows, ([], []))

    def test_no_matching_explicit_backend_tests_raise(self):
        with (
            mock.patch("onnx_light_cpu._benchmark.register_backend_test_cases"),
            mock.patch("onnx_light_cpu._benchmark.register_kernels"),
            mock.patch("onnx_light.onnx.backend.collect_test_cases_by_name", return_value=[]),
            self.assertRaisesRegex(ValueError, "no benchmark backend test matches"),
        ):
            _benchmark.run_backend_benchmark(
                tests=[r"^test_cpu_no_such_operator_"],
                dtypes=["all"],
                repeat=1,
                warmup=0,
                max_repeat_time=1.0,
                threads=1,
                with_onnxruntime=False,
            )

    def test_selects_backend_tests_and_dtypes(self):
        cases = [
            SimpleNamespace(name="test_cpu_abs_float32_benchmark", unload=mock.Mock()),
            SimpleNamespace(name="test_cpu_abs_float64_benchmark", unload=mock.Mock()),
            SimpleNamespace(name="test_cpu_gemm_float32_benchmark", unload=mock.Mock()),
        ]
        measured = (
            [{"case": cases[0].name, "duration_s": 1.0}],
            {"case": cases[0].name, "mean_s": 1.0},
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
            progress = io.StringIO()
            with redirect_stderr(progress):
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
        self.assertIn("[1/1] test_cpu_abs_float32_benchmark", progress.getvalue())
        measure.assert_called_once_with(cases[0], 1, 0, 1.0, 1, False)

    def test_infers_pull_request_benchmark_selection(self):
        kernels = [
            self._kernel("Abs", ("FLOAT", "DOUBLE")),
            self._kernel("Log", ("FLOAT", "DOUBLE")),
            self._kernel("GroupQueryAttention", ("FLOAT", "FLOAT16")),
        ]
        tests, dtypes = infer_pr_benchmark_selection(
            [
                "onnx_light_cpu/impl/math/abs_kernel.cc",
                "docs/command_line.rst",
            ],
            "+  const float value = input[i];",
            kernels,
        )
        self.assertEqual(tests, ["^test_cpu_(abs)_"])
        self.assertEqual(dtypes, ["float32"])

        tests, dtypes = infer_pr_benchmark_selection(
            ["onnx_light_cpu/kernels/com_microsoft/group_query_attention_kernel.cc"],
            "+  RunAttention(input);",
            kernels,
        )
        self.assertEqual(tests, ["^test_cpu_(group_?query_?attention)_"])
        self.assertEqual(dtypes, ["float16", "float32"])

    def test_infers_operator_from_generic_kernel_diff(self):
        kernels = [
            self._kernel("Add", ("FLOAT",)),
            self._kernel("Pow", ("FLOAT",)),
        ]
        tests, dtypes = infer_pr_benchmark_selection(
            ["onnx_light_cpu/impl/math/binary/binary_kernel_descriptor.cc"],
            "+ void BulkFloatIntegerPowRightScalar();\n+ float value;",
            kernels,
        )
        self.assertEqual(tests, ["^test_cpu_(pow)_"])
        self.assertEqual(dtypes, ["float32"])

    def test_kernel_path_takes_precedence_over_unrelated_symbols(self):
        kernels = [
            self._kernel("Add", ("FLOAT",)),
            self._kernel("And", ("BOOL",)),
            self._kernel("TreeEnsemble", ("FLOAT",)),
        ]
        tests, dtypes = infer_pr_benchmark_selection(
            ["onnx_light_cpu/impl/traditionalml/tree_ensemble.cc"],
            "+ partial_stride = SaturatingAdd(rows, lanes);\n+ if (left && right) {}",
            kernels,
        )
        self.assertEqual(tests, ["^test_cpu_(tree_?ensemble)_"])
        self.assertEqual(dtypes, ["float32"])

    def test_reads_pull_request_benchmark_selection(self):
        completed = [
            SimpleNamespace(stdout="onnx_light_cpu/impl/math/abs_kernel.cc\n"),
            SimpleNamespace(stdout="+ DataType::FLOAT\n"),
        ]
        kernels = [self._kernel("Abs", ("FLOAT", "DOUBLE"))]
        with (
            mock.patch("onnx_light_cpu._benchmark.subprocess.run", side_effect=completed) as run,
            mock.patch("onnx_light_cpu._register.registered_kernels", return_value=kernels),
        ):
            selection = pull_request_benchmark_selection("623")
        self.assertEqual(selection, (["^test_cpu_(abs)_"], ["float32"]))
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                ["gh", "pr", "diff", "623", "--name-only"],
                ["gh", "pr", "diff", "623"],
            ],
        )

    def test_explicit_threads_use_onnxruntime_affinity_default(self):
        case = SimpleNamespace(
            name="test_cpu_abs_float32_benchmark",
            model=SimpleNamespace(
                graph=SimpleNamespace(
                    node=[SimpleNamespace(domain="", op_type="Abs")],
                    input=[],
                )
            ),
            data_sets=[],
        )
        with (
            mock.patch("onnx_light.onnx.reference.ReferenceEvaluator") as evaluator,
            mock.patch("onnx_light_cpu._benchmark.clear_used_kernel_names") as clear,
            mock.patch("onnx_light_cpu._benchmark.set_kernel_usage_recording") as recording,
            mock.patch(
                "onnx_light_cpu._register.registered_kernels",
                return_value=(self._registration("ai.onnx", "Abs", "onnx_light_cpu::Abs"),),
            ),
            mock.patch("onnx_light_cpu._benchmark.platform.processor", return_value="test CPU"),
            mock.patch(
                "onnx_light_cpu._benchmark.used_kernel_paths",
                return_value=("onnx_light_cpu::Abs",),
            ) as used,
        ):
            raw, _ = _benchmark._measure_case(case, 1, 0, 1.0, 3)
        evaluator.assert_called_once_with(
            case.model,
            cpu_execution={"num_threads": 3, "affinity_policy": "none"},
        )
        self.assertEqual(raw[0]["run"], 1)
        self.assertEqual(raw[0]["runtime"], "onnx-light-cpu")
        self.assertEqual(raw[0]["processor"], "test CPU")
        self.assertEqual(raw[0]["cpu_kernel_paths"], "onnx_light_cpu::Abs")
        self.assertIn("duration_s", raw[0])
        self.assertEqual(
            recording.call_args_list,
            [mock.call(evaluator.return_value, True), mock.call(evaluator.return_value, False)],
        )
        clear.assert_called_once_with(evaluator.return_value)
        used.assert_called_once_with(evaluator.return_value)

    def test_domain_specific_kernel_name_is_checked(self):
        kernels = (
            self._registration("ai.onnx", "LinearAttention", "onnx_light_cpu::LinearAttention"),
            self._registration(
                "com.microsoft",
                "LinearAttention",
                "onnx_light_cpu::MicrosoftLinearAttention",
            ),
        )
        self.assertEqual(
            _benchmark._kernel_names_for_operator(kernels, "com.microsoft", "LinearAttention"),
            ("onnx_light_cpu::MicrosoftLinearAttention",),
        )

    def test_onnxruntime_unsupported_case_is_reported(self):
        model = SimpleNamespace(
            graph=SimpleNamespace(
                node=[SimpleNamespace(domain="", op_type="Abs")],
                input=[],
            ),
            SerializeToString=mock.Mock(return_value=b"model"),
        )
        case = SimpleNamespace(
            name="test_cpu_abs_float32_benchmark",
            model=model,
            data_sets=[],
        )
        onnxruntime = SimpleNamespace(
            SessionOptions=lambda: SimpleNamespace(),
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
            InferenceSession=mock.Mock(side_effect=RuntimeError("unsupported model")),
        )
        with (
            mock.patch.dict(sys.modules, {"onnxruntime": onnxruntime}),
            mock.patch("onnx_light.onnx.reference.ReferenceEvaluator"),
            mock.patch("onnx_light_cpu._benchmark.clear_used_kernel_names"),
            mock.patch("onnx_light_cpu._benchmark.set_kernel_usage_recording"),
            mock.patch(
                "onnx_light_cpu._register.registered_kernels",
                return_value=(self._registration("ai.onnx", "Abs", "onnx_light_cpu::Abs"),),
            ),
            mock.patch(
                "onnx_light_cpu._benchmark.used_kernel_paths",
                return_value=("onnx_light_cpu::Abs",),
            ),
        ):
            raw, aggregated = _benchmark._measure_case(case, 1, 0, 1.0, 1, True)
        self.assertEqual(aggregated["onnxruntime_error"], "unsupported model")
        self.assertIsNone(aggregated["onnxruntime_median_s"])
        self.assertIsNone(aggregated["speedup"])
        self.assertEqual([row["runtime"] for row in raw], ["onnx-light-cpu"])

    def test_onnxruntime_run_failure_is_reported(self):
        model = SimpleNamespace(
            graph=SimpleNamespace(
                node=[SimpleNamespace(domain="", op_type="Abs")],
                input=[SimpleNamespace(name="x")],
            ),
            SerializeToString=mock.Mock(return_value=b"model"),
        )
        case = SimpleNamespace(
            name="test_cpu_abs_float32_benchmark",
            model=model,
            data_sets=[SimpleNamespace(inputs=[SimpleNamespace()])],
        )
        ort_session = SimpleNamespace(
            run=mock.Mock(side_effect=RuntimeError("unsupported input"))
        )
        onnxruntime = SimpleNamespace(
            SessionOptions=lambda: SimpleNamespace(),
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
            InferenceSession=mock.Mock(return_value=ort_session),
        )
        with (
            mock.patch.dict(sys.modules, {"onnxruntime": onnxruntime}),
            mock.patch(
                "onnx_light_cpu._benchmark._to_numpy",
                return_value=SimpleNamespace(shape=(1,)),
            ),
            mock.patch("onnx_light.onnx.reference.ReferenceEvaluator"),
            mock.patch("onnx_light_cpu._benchmark.clear_used_kernel_names"),
            mock.patch("onnx_light_cpu._benchmark.set_kernel_usage_recording"),
            mock.patch(
                "onnx_light_cpu._register.registered_kernels",
                return_value=(self._registration("ai.onnx", "Abs", "onnx_light_cpu::Abs"),),
            ),
            mock.patch(
                "onnx_light_cpu._benchmark.used_kernel_paths",
                return_value=("onnx_light_cpu::Abs",),
            ),
        ):
            raw, aggregated = _benchmark._measure_case(case, 1, 0, 1.0, 1, True)
        self.assertEqual(aggregated["onnxruntime_error"], "unsupported input")
        self.assertEqual([row["runtime"] for row in raw], ["onnx-light-cpu"])

    def test_cpu_is_measured_before_onnxruntime_is_created(self):
        events = []

        def cpu_run():
            return None

        def ort_run():
            return None

        def measure(run):
            events.append(("measure", run))
            return [1.0, 2.0]

        def create_onnxruntime():
            events.append(("create", None))
            return SimpleNamespace(run=ort_run), None

        runners, measured, error = _benchmark._measure_runtime_phases(
            cpu_run, create_onnxruntime, measure
        )
        self.assertEqual(
            events,
            [("measure", cpu_run), ("create", None), ("measure", ort_run)],
        )
        self.assertEqual([runtime for runtime, _ in runners], ["onnx-light-cpu", "onnxruntime"])
        self.assertEqual(
            measured,
            {"onnx-light-cpu": [1.0, 2.0], "onnxruntime": [1.0, 2.0]},
        )
        self.assertIsNone(error)

    def test_writes_raw_and_aggregated_sheets(self):
        raw = [
            {
                "case": "test_cpu_abs_float32_benchmark",
                "operator": "Abs",
                "dtype": "float32",
                "repeat": 2,
                "warmup": 1,
                "threads": 3,
                "processor": "test CPU",
                "input_shapes": '[{"x": [2, 3]}]',
                "cpu_kernel_paths": "onnx_light_cpu::Abs",
                "max_repeat_time": 1.0,
                "run": 1,
                "runtime": "onnx-light-cpu",
                "duration_s": 0.0000125,
            }
        ]
        aggregated = [
            {
                "case": "test_cpu_abs_float32_benchmark",
                "operator": "Abs",
                "dtype": "float32",
                "repeat": 2,
                "warmup": 1,
                "threads": 3,
                "processor": "test CPU",
                "input_shapes": '[{"x": [2, 3]}]',
                "cpu_kernel_paths": "onnx_light_cpu::Abs",
                "max_repeat_time": 1.0,
                "samples": 1,
                "mean_s": 0.0000125,
                "stdev_s": 0.0,
                "min_repeat_s": 0.0000125,
                "p10_s": 0.0000125,
                "median_s": 0.0000125,
                "p90_s": 0.0000125,
                "max_repeat_s": 0.0000125,
                "onnxruntime_samples": None,
                "onnxruntime_mean_s": None,
                "onnxruntime_median_s": None,
                "onnxruntime_error": None,
                "speedup": None,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "benchmark.xlsx"
            write_benchmark_workbook(output, raw, aggregated)
            workbook = load_workbook(output, read_only=True)
            self.assertEqual(workbook.sheetnames, ["raw", "aggregated"])
            self.assertEqual(workbook["raw"]["A2"].value, raw[0]["case"])
            raw_headers = [cell.value for cell in workbook["raw"][1]]
            self.assertIn("run", raw_headers)
            self.assertIn("runtime", raw_headers)
            self.assertIn("processor", raw_headers)
            self.assertNotIn("duration_us", raw_headers)
            self.assertEqual(workbook["aggregated"]["I2"].value, "onnx_light_cpu::Abs")
            self.assertEqual(workbook["aggregated"]["K2"].value, 1)
            workbook.close()

    def test_writes_aggregated_markdown(self):
        aggregated = [
            {
                column: "test|value" if column == "case" else 1
                for column in _benchmark._AGGREGATED_COLUMNS
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "benchmark.md"
            write_benchmark_markdown(output, aggregated)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "| case | operator | dtype | repeat | warmup | threads | processor | input_shapes"
                " | cpu_kernel_paths | max_repeat_time | samples | mean_s | stdev_s"
                " | min_repeat_s | p10_s | median_s | p90_s | max_repeat_s | onnxruntime_samples"
                " | onnxruntime_mean_s | onnxruntime_median_s | onnxruntime_error | speedup |\n"
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---"
                " | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
                "| test\\|value | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1"
                " | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |\n",
            )

    def test_posts_aggregated_markdown_to_pull_request(self):
        aggregated = [
            {
                "case": "test|value",
                "speedup": 1.23456,
                "input_shapes": '[{"z": [2, 3, 4], "a": [5, 6]}]',
            }
        ]
        for pull_request, reference in [("623", ["623"]), ("", [])]:
            with self.subTest(pull_request=pull_request):
                with mock.patch("onnx_light_cpu._benchmark.subprocess.run") as run:
                    post_benchmark_markdown(pull_request, aggregated)
                run.assert_called_once_with(
                    [
                        "gh",
                        "pr",
                        "comment",
                        *reference,
                        "--edit-last",
                        "--create-if-none",
                        "--body-file",
                        "-",
                    ],
                    input=mock.ANY,
                    text=True,
                    check=True,
                )
                self.assertEqual(
                    run.call_args.kwargs["input"],
                    '<div style="max-height: 500px; overflow-x: auto; overflow-y: auto;">\n\n'
                    "| speedup | test_name |\n| --- | --- |\n| 1.23 | test\\|value |\n"
                    "\n</div>\n",
                )

    def test_writes_pull_request_markdown(self):
        aggregated = [
            {
                "case": "test|value",
                "speedup": 1,
                "input_shapes": '[{"x": [2, 3]}]',
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "benchmark-pr.md"
            write_pr_benchmark_markdown(output, aggregated)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                '<div style="max-height: 500px; overflow-x: auto; overflow-y: auto;">\n\n'
                "| speedup | test_name |\n| --- | --- |\n| 1.00 | test\\|value |\n"
                "\n</div>\n",
            )

    def test_writes_no_matching_backend_tests_message(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "benchmark-pr.md"
            write_pr_benchmark_markdown(output, [], [r"^test_cpu_(tree_?ensemble)_"])
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "No corresponding backend tests were found for regular expression "
                "`^test_cpu_(tree_?ensemble)_`.\n",
            )

    def test_pull_request_formatting_sorts_by_increasing_speedup(self):
        aggregated = [
            {
                "case": "multiple_datasets",
                "speedup": 1.236,
                "input_shapes": '[{"x": [2, 3], "bias": []}, {"x": [0, 3], "bias": [3]}]',
            },
            {"case": "unsupported", "speedup": None, "input_shapes": '[{"x": [1]}]'},
            {"case": "slow", "speedup": 0.004, "input_shapes": '[{"x": [2, 3, 4]}]'},
        ]
        self.assertEqual(
            _benchmark._pr_benchmark_markdown(aggregated),
            '<div style="max-height: 500px; overflow-x: auto; overflow-y: auto;">\n\n'
            "| speedup | test_name |\n"
            "| --- | --- |\n"
            "| 0.00 | slow |\n"
            "| 1.24 | multiple_datasets |\n"
            "| None | unsupported |\n"
            "\n</div>\n",
        )
        self.assertEqual(aggregated[0]["speedup"], 1.236)
        self.assertEqual(
            aggregated[0]["input_shapes"],
            '[{"x": [2, 3], "bias": []}, {"x": [0, 3], "bias": [3]}]',
        )

    def test_pull_request_report_is_truncated_to_github_comment_limit(self):
        aggregated = [
            {"case": f"test_{index}_{'x' * 80}", "speedup": index} for index in range(20)
        ]
        with mock.patch.object(_benchmark, "_PR_COMMENT_MAX_LENGTH", 500):
            report = _benchmark._pr_benchmark_markdown(aggregated)
        self.assertLessEqual(len(report), 500)
        self.assertIn("Report truncated", report)
        self.assertIn("test_0_", report)
        self.assertNotIn("test_19_", report)

    def test_main_runs_benchmark_and_writes_output(self):
        rows = ([{"duration_s": 1.0}], [{"case": "one"}])
        with (
            mock.patch(
                "onnx_light_cpu.__main__.run_backend_benchmark",
                return_value=rows,
            ) as run,
            mock.patch("onnx_light_cpu.__main__.write_benchmark_workbook") as write,
            mock.patch("onnx_light_cpu.__main__.write_benchmark_markdown") as markdown,
            mock.patch("onnx_light_cpu.__main__.write_pr_benchmark_markdown") as pr_markdown,
            mock.patch("onnx_light_cpu.__main__.post_benchmark_markdown") as post,
            mock.patch(
                "onnx_light_cpu.__main__.pull_request_benchmark_selection",
                return_value=([r"^test_cpu_(abs)_"], ["float32"]),
            ) as infer,
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
                    "--markdown",
                    "results.md",
                    "--pr-markdown",
                    "pr-results.md",
                    "--pr",
                    "623",
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
            with_onnxruntime=False,
            allow_empty=False,
        )
        write.assert_called_once_with("results.xlsx", *rows)
        markdown.assert_called_once_with("results.md", rows[1])
        pr_markdown.assert_called_once_with("pr-results.md", rows[1], [r"^test_cpu_(abs|log)_"])
        post.assert_called_once_with("623", rows[1], [r"^test_cpu_(abs|log)_"])
        infer.assert_not_called()

    def test_main_infers_filters_from_pull_request(self):
        rows = ([{"duration_s": 1.0}], [{"case": "one"}])
        with (
            mock.patch(
                "onnx_light_cpu.__main__.pull_request_benchmark_selection",
                return_value=([r"^test_cpu_(abs)_"], ["float32"]),
            ) as infer,
            mock.patch("onnx_light_cpu.__main__.run_backend_benchmark", return_value=rows) as run,
            mock.patch("onnx_light_cpu.__main__.write_benchmark_workbook"),
            mock.patch("onnx_light_cpu.__main__.post_benchmark_markdown"),
        ):
            main(["benchmark", "--from-pr", "623"])
        infer.assert_called_once_with("623")
        self.assertEqual(run.call_args.kwargs["tests"], [r"^test_cpu_(abs)_"])
        self.assertEqual(run.call_args.kwargs["dtypes"], ["float32"])
        self.assertTrue(run.call_args.kwargs["allow_empty"])


if __name__ == "__main__":
    import unittest

    unittest.main()
