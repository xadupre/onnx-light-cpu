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
    operator: str = "Gemm"
    batch_shape: tuple[int, ...] = ()
    b_batch_shape: tuple[int, ...] = ()
    vector_a: bool = False
    vector_b: bool = False

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
    GemmCase("transformer_projection_dynamic", 128, 3072, 768),
    GemmCase("transformer_projection", 128, 3072, 768, constant_b=True),
    GemmCase("transformer_down_projection_dynamic", 128, 768, 3072),
)

MATMUL_PRIORITY_CASES = (
    GemmCase("batched_direct", 32, 128, 16, operator="MatMul", batch_shape=(4,)),
    GemmCase("batched_square", 128, 128, 128, operator="MatMul", batch_shape=(2,)),
    GemmCase(
        "broadcast_b",
        128,
        128,
        128,
        operator="MatMul",
        batch_shape=(4,),
        b_batch_shape=(1,),
    ),
    GemmCase("vector_matrix", 1, 128, 128, operator="MatMul", vector_a=True),
    GemmCase("matrix_vector", 128, 1, 128, operator="MatMul", vector_b=True),
    GemmCase("large_k", 32, 32, 4096, operator="MatMul", batch_shape=(2,)),
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
        "onnx-light-cpu p95 (us)",
        "onnxruntime p95 (us)",
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
                f"{float(result['cpu_p95_seconds']) * 1e6:.2f}",
                f"{float(result['ort_p95_seconds']) * 1e6:.2f}",
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
    functions: Sequence[Callable[[], Any]],
    repeat: int,
    warmup: int,
    max_repeat_time: float = 1.0,
) -> tuple[list[float], ...]:
    warmup_durations = [0.0] * len(functions)
    for iteration in range(warmup):
        for offset in range(len(functions)):
            index = (iteration + offset) % len(functions)
            if warmup_durations[index] >= max_repeat_time:
                continue
            start = time.perf_counter_ns()
            functions[index]()
            warmup_durations[index] += (time.perf_counter_ns() - start) / 1e9
        if all(duration >= max_repeat_time for duration in warmup_durations):
            break

    timings: tuple[list[float], ...] = tuple([] for _ in functions)
    total_durations = [0.0] * len(functions)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for iteration in range(repeat):
            for offset in range(len(functions)):
                index = (iteration + offset) % len(functions)
                if total_durations[index] >= max_repeat_time:
                    continue
                start = time.perf_counter_ns()
                functions[index]()
                duration = (time.perf_counter_ns() - start) / 1e9
                timings[index].append(duration)
                total_durations[index] += duration
            if all(duration >= max_repeat_time for duration in total_durations):
                break
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


def _matmul_operand_shape(
    batch_shape: tuple[int, ...], matrix_shape: tuple[int, ...], vector: bool, k: int
) -> tuple[int, ...]:
    if vector:
        if batch_shape:
            raise ValueError("MatMul vector operands cannot have batch dimensions.")
        return (k,)
    return batch_shape + matrix_shape


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
    a_matrix_shape = (case.k, case.m) if case.trans_a else (case.m, case.k)
    b_matrix_shape = (case.n, case.k) if case.trans_b else (case.k, case.n)
    a_shape = _matmul_operand_shape(case.batch_shape, a_matrix_shape, case.vector_a, case.k)
    b_shape = _matmul_operand_shape(case.b_batch_shape, b_matrix_shape, case.vector_b, case.k)
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

    node_attributes = {}
    if case.operator == "Gemm":
        node_attributes = {"transA": int(case.trans_a), "transB": int(case.trans_b)}
    node = helper.make_node(case.operator, node_inputs, ["Y"], **node_attributes)
    output_shape = None if case.operator == "MatMul" else (case.m, case.n)
    graph = helper.make_graph(
        [node],
        f"gemm_parity_{case.name}_{dtype_name}",
        inputs,
        [helper.make_tensor_value_info("Y", tensor_type, output_shape)],
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


def _metadata(
    requested_threads: int | None,
    actual_threads: int,
    affinity_policy: str,
    simd_level: int,
) -> dict[str, Any]:
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
        "onnx_light_effective_threads": actual_threads,
        "onnx_light_affinity_policy": affinity_policy,
        "onnxruntime_intra_op_threads": requested_threads,
        "backend_isolation": "separate session lifetimes; order alternates by case",
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
    from onnx_light_cpu import detect_simd_level, register_kernels, set_kernel_usage_recording

    register_kernels()

    set_kernel_usage_recording(False)
    session_options = None
    cpu_execution = None
    if args.threads is not None:
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = args.threads
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        cpu_execution = {"num_threads": args.threads, "affinity_policy": "none"}

    results: list[dict[str, Any]] = []
    dtype_names = PARITY_DTYPES if args.dtype == "all" else (args.dtype,)
    case_corpus = (*PRIORITY_CASES, *MATMUL_PRIORITY_CASES)
    if args.operator == "gemm":
        case_corpus = PRIORITY_CASES
    elif args.operator == "matmul":
        case_corpus = MATMUL_PRIORITY_CASES
    selected_cases = (
        case_corpus
        if not args.case
        else tuple(case for case in case_corpus if case.name in args.case)
    )
    unknown_cases = set(args.case) - {case.name for case in case_corpus}
    if unknown_cases:
        raise ValueError(f"Unknown cases: {', '.join(sorted(unknown_cases))}.")

    actual_threads = None
    affinity_policy = None

    def make_cpu_execution(model_bytes, feeds):
        session = ReferenceEvaluator(model_bytes, cpu_execution=cpu_execution)
        resolution = session.cpu_execution_resolution

        def execute():
            return session.run(None, feeds)[0]

        return (
            execute,
            resolution.effective_threads,
            resolution.request.affinity_policy.name.lower(),
        )

    def make_ort_execution(model_bytes, feeds):
        session = onnxruntime.InferenceSession(
            model_bytes,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        def execute():
            return session.run(None, feeds)[0]

        return execute

    for dtype_index, dtype_name in enumerate(dtype_names):
        for case_index, case in enumerate(selected_cases):
            rng = np.random.default_rng(1000 * dtype_index + case_index)
            model_bytes, feeds = _build_case(case, dtype_name, rng)
            repeat = args.repeat
            cpu_execute, session_threads, session_affinity = make_cpu_execution(
                model_bytes, feeds
            )
            ort_execute = make_ort_execution(model_bytes, feeds)
            if (dtype_index + case_index) % 2 == 0:
                cpu_output = cpu_execute().copy()
                ort_output = ort_execute().copy()
                cpu_samples, ort_samples = measure_alternating(
                    (cpu_execute, ort_execute), repeat, args.warmup, args.max_repeat_time
                )
            else:
                ort_output = ort_execute().copy()
                cpu_output = cpu_execute().copy()
                ort_samples, cpu_samples = measure_alternating(
                    (ort_execute, cpu_execute), repeat, args.warmup, args.max_repeat_time
                )
            gc.collect()
            if actual_threads is None:
                actual_threads = session_threads
                affinity_policy = session_affinity
            elif actual_threads != session_threads or affinity_policy != session_affinity:
                raise RuntimeError(
                    "onnx-light resolved inconsistent CPU policies across sessions."
                )
            tolerance = {
                "float16": 2e-2,
                "float32": 2e-2,
                "float64": 1e-9,
            }[dtype_name]
            np.testing.assert_allclose(cpu_output, ort_output, rtol=tolerance, atol=tolerance)

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
                "cpu_p95_seconds": _percentile(cpu_samples, 0.95),
                "ort_p95_seconds": _percentile(ort_samples, 0.95),
                "speedup": ort_median / cpu_median,
            }
            results.append(result)
            print(
                f"{dtype_name:>7} {case.name:<24} "
                f"cpu={cpu_median * 1e6:10.2f} us "
                f"ort={ort_median * 1e6:10.2f} us "
                f"cpu_p95={result['cpu_p95_seconds'] * 1e6:10.2f} us "
                f"ort_p95={result['ort_p95_seconds'] * 1e6:10.2f} us "
                f"speedup={result['speedup']:.3f}x",
                flush=True,
            )

    if actual_threads is None:
        raise RuntimeError("No benchmark cases were selected.")
    report = {
        "metadata": _metadata(args.threads, actual_threads, affinity_policy, detect_simd_level()),
        "results": results,
        "summary": summarize(results),
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Thread count shared by both runtimes (default: 1).",
    )
    parser.add_argument(
        "--cpus", default="", help="Linux CPU affinity, for example 0-3 or 0,2,4."
    )
    parser.add_argument("-r", "--repeat", type=int, default=100 * (os.cpu_count() or 1))
    parser.add_argument("-w", "--warmup", type=int, default=100 * 20)
    parser.add_argument("-t", "--max-repeat-time", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("all", *PARITY_DTYPES), default="all")
    parser.add_argument("--operator", choices=("all", "gemm", "matmul"), default="all")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only this named case; may be repeated.",
    )
    parser.add_argument("--output", type=Path, default=Path("gemm_fp_parity_results.json"))
    parser.add_argument(
        "--enforce", action="store_true", help="Exit nonzero when the gate fails."
    )
    args = parser.parse_args(argv)
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive.")
    if args.repeat < 1 or args.warmup < 0 or args.max_repeat_time <= 0:
        parser.error("repeat and max-repeat-time must be positive and warmup non-negative.")
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
