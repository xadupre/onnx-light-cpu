"""Command-line interface for onnx-light-cpu."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence

from ._benchmark import (
    compare_benchmark_dtypes,
    normalize_dtypes,
    post_benchmark_markdown,
    pull_request_benchmark_selection,
    run_backend_benchmark,
    write_benchmark_markdown,
    write_pr_benchmark_markdown,
    write_benchmark_workbook,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite and greater than 0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onnx-light-cpu")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser(
        "benchmark", help="benchmark selected onnx-light-cpu backend test cases"
    )
    benchmark.add_argument(
        "--test",
        "--tests",
        dest="tests",
        action="extend",
        nargs="+",
        default=[],
        help="backend case name regular expression; may be repeated (default: ^test_cpu_)",
    )
    dtype_selection = benchmark.add_mutually_exclusive_group()
    dtype_selection.add_argument(
        "--dtype",
        "--dtypes",
        dest="dtypes",
        action="extend",
        nargs="+",
        default=[],
        help="dtype or comma-separated dtypes; may be repeated (default: all)",
    )
    dtype_selection.add_argument(
        "--compare-dtypes",
        nargs=2,
        choices=normalize_dtypes(["all"]),
        metavar=("BASELINE", "COMPARISON"),
        help=(
            "benchmark two dtypes for matching cases and compare their median times "
            "(e.g. float16 bfloat16); speedup is baseline / comparison"
        ),
    )
    benchmark.add_argument(
        "-o",
        "--output",
        default="onnx_light_cpu_benchmark.xlsx",
        help="output Excel workbook",
    )
    benchmark.add_argument(
        "-r",
        "--repeat",
        type=_positive_int,
        default=10 * (os.cpu_count() or 1),
        help="maximum measured iterations per case",
    )
    benchmark.add_argument(
        "--markdown",
        help="path of an optional Markdown file with the aggregated figures",
    )
    benchmark.add_argument(
        "--pr-markdown",
        help="path of an optional concise Markdown file for a pull request comment",
    )
    benchmark.add_argument(
        "--pr",
        nargs="?",
        const="",
        metavar="NUMBER_OR_URL",
        help="add the aggregated figures to a pull request (default: current branch's PR)",
    )
    benchmark.add_argument(
        "--from-pr",
        nargs="?",
        const="",
        metavar="NUMBER_OR_URL",
        help="infer test and dtype filters from a pull request without posting results",
    )
    benchmark.add_argument(
        "-w",
        "--warmup",
        type=_non_negative_int,
        default=2 * (os.cpu_count() or 1),
        help="maximum warm-up iterations per case",
    )
    benchmark.add_argument(
        "-t",
        "--max-repeat-time",
        type=_positive_float,
        default=1.0,
        help="maximum measurement and warm-up time per case in seconds",
    )
    benchmark.add_argument(
        "--threads",
        type=_non_negative_int,
        default=0,
        help=(
            "session thread limit; 0 lets the runtime and each kernel select defaults "
            "(default: 0)"
        ),
    )
    benchmark.add_argument(
        "--onnxruntime",
        action="store_true",
        help="also benchmark ONNX Runtime and report the speedup",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the onnx-light-cpu command line."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        if args.compare_dtypes and args.compare_dtypes[0] == args.compare_dtypes[1]:
            parser.error("--compare-dtypes requires two different dtypes")
        tests = args.tests or ["^test_cpu_"]
        dtypes = args.dtypes or ["all"]
        selection_pr = args.from_pr if args.from_pr is not None else args.pr
        selection_inferred = False
        if selection_pr is not None and not args.tests and not args.dtypes:
            tests, dtypes = pull_request_benchmark_selection(selection_pr)
            if not tests:
                raise ValueError("no benchmark operator could be inferred from the pull request")
            selection_inferred = True
        if args.compare_dtypes:
            dtypes = args.compare_dtypes
        raw_rows, aggregated_rows = run_backend_benchmark(
            tests=tests,
            dtypes=dtypes,
            repeat=args.repeat,
            warmup=args.warmup,
            max_repeat_time=args.max_repeat_time,
            threads=args.threads,
            with_onnxruntime=args.onnxruntime,
            allow_empty=selection_inferred,
        )
        comparison = (
            {"comparison_rows": compare_benchmark_dtypes(aggregated_rows, *args.compare_dtypes)}
            if args.compare_dtypes
            else {}
        )
        write_benchmark_workbook(args.output, raw_rows, aggregated_rows, **comparison)
        if args.markdown:
            write_benchmark_markdown(args.markdown, aggregated_rows, **comparison)
        report_rows = comparison.get("comparison_rows", aggregated_rows)
        empty_message = (
            f"No comparable benchmark cases were found for dtypes "
            f"`{args.compare_dtypes[0]}` and `{args.compare_dtypes[1]}`.\n"
            if args.compare_dtypes and aggregated_rows and not report_rows
            else None
        )
        empty_report = {"empty_message": empty_message} if empty_message is not None else {}
        if args.pr_markdown:
            write_pr_benchmark_markdown(args.pr_markdown, report_rows, tests, **empty_report)
        if args.pr is not None:
            post_benchmark_markdown(args.pr, report_rows, tests, **empty_report)
        print(
            f"Wrote {len(raw_rows)} measurements for {len(aggregated_rows)} cases "
            f"to {args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
