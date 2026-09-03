"""Command-line interface for onnx-light-cpu."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence

from ._benchmark import (
    post_benchmark_markdown,
    run_backend_benchmark,
    write_benchmark_markdown,
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
    default_threads = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
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
    benchmark.add_argument(
        "--dtype",
        "--dtypes",
        dest="dtypes",
        action="extend",
        nargs="+",
        default=[],
        help="dtype or comma-separated dtypes; may be repeated (default: all)",
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
        "--pr",
        nargs="?",
        const="",
        metavar="NUMBER_OR_URL",
        help="add the aggregated figures to a pull request (default: current branch's PR)",
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
        type=_positive_int,
        default=default_threads,
        help=f"number of onnx-light-cpu worker threads (default: {default_threads})",
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
        tests = args.tests or ["^test_cpu_"]
        dtypes = args.dtypes or ["all"]
        raw_rows, aggregated_rows = run_backend_benchmark(
            tests=tests,
            dtypes=dtypes,
            repeat=args.repeat,
            warmup=args.warmup,
            max_repeat_time=args.max_repeat_time,
            threads=args.threads,
            with_onnxruntime=args.onnxruntime,
        )
        write_benchmark_workbook(args.output, raw_rows, aggregated_rows)
        if args.markdown:
            write_benchmark_markdown(args.markdown, aggregated_rows)
        if args.pr is not None:
            post_benchmark_markdown(args.pr, aggregated_rows)
        print(
            f"Wrote {len(raw_rows)} measurements for {len(aggregated_rows)} cases "
            f"to {args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
