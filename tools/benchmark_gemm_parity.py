#!/usr/bin/env python3
"""Measure the FP32/FP64/FP16 Gemm parity gate against ONNX Runtime."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class GemmCase:
    name: str
    m: int
    n: int
    k: int
    trans_a: bool = False
    trans_b: bool = False
    constant_b: bool = False
    bias: str = "none"

    @property
    def operations(self) -> int:
        return 2 * self.m * self.n * self.k


PRIORITY_CASES = (
    GemmCase("direct", 32, 128, 16),
    GemmCase("tiny_dynamic", 1, 64, 64),
    GemmCase("tiny_constant", 1, 64, 64, constant_b=True),
    GemmCase("square_128", 128, 128, 128),
    GemmCase("square_512", 512, 512, 512),
    GemmCase("square_1024", 1024, 1024, 1024),
    GemmCase("scalar_bias_256", 256, 256, 256, bias="scalar"),
    GemmCase("row_bias_256", 256, 256, 256, bias="row"),
    GemmCase("column_bias_256", 256, 256, 256, bias="column"),
    GemmCase("matrix_bias_256", 256, 256, 256, bias="matrix"),
    GemmCase("skinny_m", 1, 1024, 1024),
    GemmCase("skinny_n", 1024, 1, 1024),
    GemmCase("large_k", 32, 32, 4096),
    GemmCase("split_k", 2, 2, 4096),
    GemmCase("trans_a", 128, 128, 128, trans_a=True),
    GemmCase("trans_b", 128, 128, 128, trans_b=True),
    GemmCase("trans_ab", 128, 128, 128, trans_a=True, trans_b=True),
    GemmCase("transformer_projection", 128, 3072, 768, constant_b=True),
)

PARITY_DTYPES = ("float32", "float64", "float16")


def repeat_count(case: GemmCase, minimum: int, maximum: int) -> int:
    target_operations = 300_000_000
    return max(minimum, min(maximum, target_operations // max(1, case.operations)))


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_dtype: dict[str, dict[str, Any]] = {}
    passed = True
    for dtype in sorted({str(result["dtype"]) for result in results}):
        speedups = [float(result["speedup"]) for result in results if result["dtype"] == dtype]
        median = statistics.median(speedups)
        minimum = min(speedups)
        dtype_passed = median >= 1.0 and minimum >= 0.9
        passed = passed and dtype_passed
        by_dtype[dtype] = {
            "median_speedup": median,
            "minimum_speedup": minimum,
            "passed": dtype_passed,
        }
    return {"passed": passed, "thresholds": {"median": 1.0, "minimum": 0.9}, "by_dtype": by_dtype}


def _gflops(operations: int, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return operations / seconds / 1e9


def render_comparison_table(results: Sequence[dict[str, Any]]) -> str:
    """Render a simple side-by-side onnx-light-cpu vs ONNX Runtime table."""
    header = (
        "dtype",
        "case",
        "M",
        "N",
        "K",
        "onnx-light-cpu (GFLOP/s)",
        "onnxruntime (GFLOP/s)",
        "speedup",
    )
    rows: list[tuple[str, ...]] = [header]
    for result in results:
        operations = 2 * int(result["m"]) * int(result["n"]) * int(result["k"])
        rows.append(
            (
                str(result["dtype"]),
                str(result["name"]),
                str(result["m"]),
                str(result["n"]),
                str(result["k"]),
                f"{_gflops(operations, float(result['cpu_median_seconds'])):.2f}",
                f"{_gflops(operations, float(result['ort_median_seconds'])):.2f}",
                f"{float(result['speedup']):.3f}x",
            )
        )

    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    lines = []
    for index, row in enumerate(rows):
        cells = " | ".join(cell.ljust(widths[column]) for column, cell in enumerate(row))
        lines.append(f"| {cells} |")
        if index == 0:
            separators = " | ".join("-" * widths[column] for column in range(len(header)))
            lines.append(f"| {separators} |")
    return "\n".join(lines)


def measure_alternating(
    functions: Sequence[Callable[[], Any]], repeat: int, warmup: int
) -> tuple[list[float], ...]:
    for iteration in range(warmup):
        for offset in range(len(functions)):
            functions[(iteration + offset) % len(functions)]()

    timings: tuple[list[float], ...] = tuple([] for _ in functions)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for iteration in range(repeat):
            for offset in range(len(functions)):
                index = (iteration + offset) % len(functions)
                start = time.perf_counter_ns()
                functions[index]()
                timings[index].append((time.perf_counter_ns() - start) / 1e9)
    finally:
        if gc_enabled:
            gc.enable()
    return timings


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith(("model name", "Model")) and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _compiler() -> str:
    command = shlex.split(os.environ.get("CXX", "c++"))
    try:
        output = subprocess.run(
            [*command, "--version"], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return output.splitlines()[0] if output else "unknown"


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "onnx-light", "onnx-light-cpu", "onnxruntime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _bias_shape(case: GemmCase) -> tuple[int, ...]:
    if case.bias == "scalar":
        return ()
    if case.bias == "row":
        return (1, case.n)
    if case.bias == "column":
        return (case.m, 1)
    if case.bias == "matrix":
        return (case.m, case.n)
    if case.bias != "none":
        raise ValueError(f"Unsupported bias layout {case.bias!r}.")
    return ()


def _build_case(case: GemmCase, dtype_name: str, rng: Any) -> tuple[bytes, dict[str, Any]]:
    import numpy as np
    from onnx_light.onnx import TensorProto, checker, helper, numpy_helper

    dtype = {
        "float32": np.float32,
        "float64": np.float64,
        "float16": np.float16,
    }[dtype_name]
    tensor_type = {
        "float32": TensorProto.FLOAT,
        "float64": TensorProto.DOUBLE,
        "float16": TensorProto.FLOAT16,
    }[dtype_name]
    a_shape = (case.k, case.m) if case.trans_a else (case.m, case.k)
    b_shape = (case.n, case.k) if case.trans_b else (case.k, case.n)
    a = rng.standard_normal(a_shape).astype(dtype)
    b = rng.standard_normal(b_shape).astype(dtype)

    inputs = [helper.make_tensor_value_info("A", tensor_type, a_shape)]
    feeds: dict[str, Any] = {"A": a}
    initializers = []
    if case.constant_b:
        initializers.append(numpy_helper.from_array(b, name="B"))
    else:
        inputs.append(helper.make_tensor_value_info("B", tensor_type, b_shape))
        feeds["B"] = b

    node_inputs = ["A", "B"]
    if case.bias != "none":
        bias_shape = _bias_shape(case)
        bias = rng.standard_normal(bias_shape or ()).astype(dtype)
        inputs.append(helper.make_tensor_value_info("C", tensor_type, bias_shape))
        feeds["C"] = bias
        node_inputs.append("C")

    node = helper.make_node(
        "Gemm",
        node_inputs,
        ["Y"],
        transA=int(case.trans_a),
        transB=int(case.trans_b),
    )
    graph = helper.make_graph(
        [node],
        f"gemm_parity_{case.name}_{dtype_name}",
        inputs,
        [helper.make_tensor_value_info("Y", tensor_type, (case.m, case.n))],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=13)
    checker.check_model(model)
    return model.SerializeToString(), feeds


def _parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.split(","):
        if "-" in item:
            begin, end = (int(part) for part in item.split("-", 1))
            cpus.update(range(begin, end + 1))
        elif item:
            cpus.add(int(item))
    if not cpus:
        raise ValueError("CPU affinity list must not be empty.")
    return cpus


def _metadata(requested_threads: int, actual_threads: int, simd_level: int) -> dict[str, Any]:
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = []
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "affinity": affinity,
        "requested_threads": requested_threads,
        "actual_threads": actual_threads,
        "affinity_policy": "none",
        "simd_level": simd_level,
        "compiler": _compiler(),
        "versions": _package_versions(),
        "python": platform.python_version(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cpus:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("--cpus is only supported on platforms with sched_setaffinity().")
        os.sched_setaffinity(0, _parse_cpu_list(args.cpus))
    import numpy as np
    import onnxruntime
    from onnx_light.onnx.reference import ReferenceEvaluator
    from onnx_light_cpu import register_kernels
    from onnx_light_cpu.onnx_py._cpukernels import detect_simd_level
    from onnx_light_cpu.onnx_py._cpuregister import set_kernel_usage_recording

    register_kernels()
    set_kernel_usage_recording(False)
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = args.threads
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    results: list[dict[str, Any]] = []
    dtype_names = PARITY_DTYPES if args.dtype == "all" else (args.dtype,)
    selected_cases = (
        PRIORITY_CASES
        if not args.case
        else tuple(case for case in PRIORITY_CASES if case.name in args.case)
    )
    unknown_cases = set(args.case) - {case.name for case in selected_cases}
    if unknown_cases:
        raise ValueError(f"Unknown cases: {', '.join(sorted(unknown_cases))}.")

    actual_threads = None
    for dtype_index, dtype_name in enumerate(dtype_names):
        for case_index, case in enumerate(selected_cases):
            rng = np.random.default_rng(1000 * dtype_index + case_index)
            model_bytes, feeds = _build_case(case, dtype_name, rng)
            cpu_session = ReferenceEvaluator(
                model_bytes,
                cpu_execution={
                    "num_threads": args.threads,
                    "affinity_policy": "none",
                },
            )
            session_threads = cpu_session.cpu_execution_resolution.effective_threads
            if actual_threads is None:
                actual_threads = session_threads
            elif actual_threads != session_threads:
                raise RuntimeError(
                    "onnx-light resolved inconsistent participant counts across sessions."
                )
            ort_session = onnxruntime.InferenceSession(
                model_bytes, sess_options=session_options, providers=["CPUExecutionProvider"]
            )

            def cpu_run(session=cpu_session, current_feeds=feeds):
                return session.run(None, current_feeds)[0]

            def ort_run(session=ort_session, current_feeds=feeds):
                return session.run(None, current_feeds)[0]

            cpu_output = cpu_run()
            ort_output = ort_run()
            tolerance = {
                "float16": 2e-2,
                "float32": 2e-2,
                "float64": 1e-9,
            }[dtype_name]
            np.testing.assert_allclose(cpu_output, ort_output, rtol=tolerance, atol=tolerance)

            repeat = repeat_count(case, args.minimum_repeats, args.maximum_repeats)
            cpu_samples, ort_samples = measure_alternating(
                (cpu_run, ort_run), repeat=repeat, warmup=args.warmup
            )
            cpu_median = statistics.median(cpu_samples)
            ort_median = statistics.median(ort_samples)
            result = {
                **asdict(case),
                "dtype": dtype_name,
                "repeat": repeat,
                "cpu_samples_seconds": cpu_samples,
                "ort_samples_seconds": ort_samples,
                "cpu_median_seconds": cpu_median,
                "ort_median_seconds": ort_median,
                "cpu_iqr_seconds": _percentile(cpu_samples, 0.75)
                - _percentile(cpu_samples, 0.25),
                "ort_iqr_seconds": _percentile(ort_samples, 0.75)
                - _percentile(ort_samples, 0.25),
                "speedup": ort_median / cpu_median,
            }
            results.append(result)
            print(
                f"{dtype_name:>7} {case.name:<24} "
                f"cpu={cpu_median * 1e6:10.2f} us "
                f"ort={ort_median * 1e6:10.2f} us "
                f"speedup={result['speedup']:.3f}x",
                flush=True,
            )

    if actual_threads is None:
        raise RuntimeError("No benchmark cases were selected.")
    report = {
        "metadata": _metadata(args.threads, actual_threads, detect_simd_level()),
        "results": results,
        "summary": summarize(results),
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--cpus", default="", help="Linux CPU affinity, for example 0-3 or 0,2,4."
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--dtype", choices=("all", *PARITY_DTYPES), default="all")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only this named case; may be repeated.",
    )
    parser.add_argument("--minimum-repeats", type=int, default=7)
    parser.add_argument("--maximum-repeats", type=int, default=31)
    parser.add_argument("--output", type=Path, default=Path("gemm_fp_parity_results.json"))
    parser.add_argument(
        "--enforce", action="store_true", help="Exit nonzero when the gate fails."
    )
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive.")
    if args.minimum_repeats < 1 or args.maximum_repeats < args.minimum_repeats:
        parser.error("repeat bounds must be positive and ordered.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = render_comparison_table(report["results"])
    table_path = args.output.with_suffix(".md")
    table_path.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(json.dumps(report["summary"], indent=2))
    print(f"raw results: {args.output}")
    print(f"comparison table: {table_path}")
    return int(args.enforce and not report["summary"]["passed"])


if __name__ == "__main__":
    sys.exit(main())
