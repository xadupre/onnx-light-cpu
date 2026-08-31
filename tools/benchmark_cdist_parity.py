#!/usr/bin/env python3
"""Measure com.microsoft::CDist end-to-end latency against ONNX Runtime."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shlex
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.benchmark_bias_gelu_parity import (  # noqa: E402
    _command_output,
    _ort_node_profile,
    _package_versions,
    _parse_cpu_list,
    _profile_python,
    percentile,
)


@dataclass(frozen=True)
class CDistCase:
    name: str
    m: int
    k: int
    n: int
    input_kind: str = "random"

    @property
    def work(self) -> int:
        return self.m * self.k * self.n


CASES = (
    CDistCase("ort_float_vector", 4, 3, 2, "ort_float"),
    CDistCase("ort_double_vector", 2, 3, 3, "ort_double"),
    CDistCase("empty_a", 0, 3, 17),
    CDistCase("empty_b", 3, 0, 17),
    CDistCase("singleton", 1, 1, 1),
    CDistCase("rectangular", 7, 11, 13),
    CDistCase("vector_before", 3, 5, 15),
    CDistCase("vector_aligned", 3, 5, 16),
    CDistCase("vector_after", 3, 5, 17),
    CDistCase("repeated_points", 8, 7, 33, "repeated"),
    CDistCase("near_identical", 4, 4, 65, "near_identical"),
    CDistCase("large_feature_tail", 4, 5, 4097),
    CDistCase("threshold_before", 511, 32, 64),
    CDistCase("threshold_at", 512, 32, 64),
    CDistCase("threshold_after", 513, 32, 64),
    CDistCase("representative", 256, 128, 64),
)
DTYPES = ("float32", "float64")
METRICS = ("sqeuclidean", "euclidean")


def measure_alternating(
    functions: Sequence[Callable[[], Any]], repeat: int, warmup: int, max_repeat_time: float
) -> tuple[tuple[list[float], ...], list[list[str]]]:
    """Warm equally and alternate candidates to limit ordering bias."""
    labels = ("onnx-light-cpu", "onnxruntime")
    if len(functions) != len(labels):
        raise ValueError("CDist parity measurement requires exactly two candidates.")
    for iteration in range(warmup):
        for offset in range(len(functions)):
            functions[(iteration + offset) % len(functions)]()
    samples: tuple[list[float], ...] = tuple([] for _ in functions)
    elapsed = [0.0] * len(functions)
    orders: list[list[str]] = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for iteration in range(repeat):
            if any(duration >= max_repeat_time for duration in elapsed):
                break
            order = [(iteration + offset) % len(functions) for offset in range(len(functions))]
            orders.append([labels[index] for index in order])
            for index in order:
                start = time.perf_counter_ns()
                functions[index]()
                duration = (time.perf_counter_ns() - start) / 1e9
                samples[index].append(duration)
                elapsed[index] += duration
    finally:
        if enabled:
            gc.enable()
    return samples, orders


def numerical_tolerance(dtype: str, metric: str, n: int, scale: float) -> float:
    """Bound cancellation in ORT's expanded squared-distance formulation."""
    epsilon = 2.0**-23 if dtype == "float32" else 2.0**-52
    squared_bound = 8.0 * epsilon * max(n, 1) * scale * scale
    return squared_bound**0.5 if metric == "euclidean" else squared_bound


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the dedicated-machine median and per-case tail latency gate."""
    by_dtype: dict[str, dict[str, Any]] = {}
    passed = bool(results)
    for dtype in sorted({str(result["dtype"]) for result in results}):
        selected = [result for result in results if result["dtype"] == dtype]
        speedups = [float(result["speedup"]) for result in selected]
        tail_speedups = [float(result["tail_speedup"]) for result in selected]
        median = statistics.median(speedups)
        minimum = min(speedups)
        minimum_tail = min(tail_speedups)
        dtype_passed = median >= 1.0 and minimum >= 0.9 and minimum_tail >= 0.9
        passed = passed and dtype_passed
        by_dtype[dtype] = {
            "median_speedup": median,
            "minimum_speedup": minimum,
            "minimum_tail_speedup": minimum_tail,
            "passed": dtype_passed,
        }
    return {
        "passed": passed and set(by_dtype) == set(DTYPES),
        "thresholds": {"median": 1.0, "minimum": 0.9, "minimum_tail": 0.9},
        "by_dtype": by_dtype,
    }


def _make_inputs(case: CDistCase, dtype: str, random: Any) -> tuple[Any, Any]:
    import numpy as np

    numpy_dtype = np.float32 if dtype == "float32" else np.float64
    a = random.standard_normal((case.m, case.n)).astype(numpy_dtype)
    b = random.standard_normal((case.k, case.n)).astype(numpy_dtype)
    if case.input_kind == "ort_float":
        a = np.array(
            [
                [-1.0856307, 0.99734545],
                [0.2829785, -1.5062947],
                [-0.5786002, 1.6514366],
                [-2.4266791, -0.42891264],
            ],
            dtype=numpy_dtype,
        )
        b = np.array(
            [[1.2659363, -0.8667404], [-0.6788862, -0.09470897], [1.4913896, -0.638902]],
            dtype=numpy_dtype,
        )
    elif case.input_kind == "ort_double":
        a = np.array(
            [[0.17251948, 1.6354825, 0.0373364], [-0.8841497, -1.1431923, -0.621366]],
            dtype=numpy_dtype,
        )
        b = np.array(
            [
                [-1.3486496, -0.81973106, -0.1342539],
                [1.5996001, -0.28360364, -0.5063398],
                [0.06890842, 1.4522595, -1.6390957],
            ],
            dtype=numpy_dtype,
        )
    elif case.input_kind == "repeated" and case.m and case.k:
        b[:] = a[0]
    elif case.input_kind == "near_identical":
        scale = numpy_dtype(1024 if dtype == "float32" else 1 << 24)
        a = np.full((case.m, case.n), scale, dtype=numpy_dtype)
        b = a[: case.k].copy()
        b[:, -1] = np.nextafter(b[:, -1], numpy_dtype(np.inf))
    return a, b


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the selected matrix with identical inputs and runtime policies."""
    selected_cpus = _parse_cpu_list(args.cpus) if args.cpus else None
    if selected_cpus is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("--cpus requires os.sched_setaffinity().")
        os.sched_setaffinity(0, selected_cpus)

    import numpy as np
    import onnxruntime
    from onnx_light.onnx import TensorProto, helper
    from onnx_light.onnx.reference import ReferenceEvaluator
    from onnx_light_cpu import clear_used_kernel_names, register_kernels, used_kernel_names

    register_kernels()
    random = np.random.default_rng(args.seed)
    rows = []
    for case in CASES:
        if args.case and case.name not in args.case:
            continue
        for dtype in DTYPES:
            if args.dtype and dtype not in args.dtype:
                continue
            tensor_type = TensorProto.FLOAT if dtype == "float32" else TensorProto.DOUBLE
            for metric in METRICS:
                if args.metric and metric not in args.metric:
                    continue
                a, b = _make_inputs(case, dtype, random)
                model = helper.make_model(
                    helper.make_graph(
                        [
                            helper.make_node(
                                "CDist", ["A", "B"], ["C"], domain="com.microsoft", metric=metric
                            )
                        ],
                        f"cdist_{case.name}_{dtype}_{metric}",
                        [
                            helper.make_tensor_value_info("A", tensor_type, [case.m, case.n]),
                            helper.make_tensor_value_info("B", tensor_type, [case.k, case.n]),
                        ],
                        [helper.make_tensor_value_info("C", tensor_type, [case.m, case.k])],
                    ),
                    opset_imports=[
                        helper.make_opsetid("", 18),
                        helper.make_opsetid("com.microsoft", 1),
                    ],
                    ir_version=13,
                )
                model_bytes = model.SerializeToString()
                feeds = {"A": a, "B": b}
                cpu = ReferenceEvaluator(model_bytes, cpu_execution={"num_threads": args.threads})

                def make_ort_session(enable_profiling: bool, current_model=model_bytes):
                    options = onnxruntime.SessionOptions()
                    options.intra_op_num_threads = args.threads
                    options.inter_op_num_threads = 1
                    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
                    options.enable_mem_pattern = True
                    options.enable_cpu_mem_arena = True
                    options.enable_profiling = enable_profiling
                    return onnxruntime.InferenceSession(
                        current_model, sess_options=options, providers=["CPUExecutionProvider"]
                    )

                ort = make_ort_session(False)

                def cpu_run(session=cpu, current_feeds=feeds):
                    return session.run(None, current_feeds)[0]

                def ort_run(session=ort, current_feeds=feeds):
                    return session.run(None, current_feeds)[0]

                clear_used_kernel_names()
                cpu_output = cpu_run()
                if "onnx_light_cpu::CDist" not in used_kernel_names():
                    raise RuntimeError(
                        f"{case.name} did not dispatch the optimized CDist kernel."
                    )
                ort_output = ort_run()
                scale = max(
                    float(np.max(np.abs(a), initial=0)),
                    float(np.max(np.abs(b), initial=0)),
                    1.0,
                )
                tolerance = args.atol
                if case.input_kind in {"near_identical", "repeated"}:
                    tolerance = max(tolerance, numerical_tolerance(dtype, metric, case.n, scale))
                np.testing.assert_allclose(
                    cpu_output, ort_output, rtol=args.rtol, atol=tolerance, equal_nan=True
                )
                (cpu_samples, ort_samples), orders = measure_alternating(
                    (cpu_run, ort_run), args.repeat, args.warmup, args.max_repeat_time
                )
                cpu_median = statistics.median(cpu_samples)
                ort_median = statistics.median(ort_samples)
                cpu_p90 = percentile(cpu_samples, 0.9)
                ort_p90 = percentile(ort_samples, 0.9)
                profile = {}
                if args.profile_runs:
                    profile["onnx_light_cpu_python"] = _profile_python(cpu_run, args.profile_runs)
                    profile_ort = make_ort_session(True)
                    for _ in range(args.profile_runs):
                        profile_ort.run(None, feeds)
                    profile["onnxruntime_nodes"] = _ort_node_profile(profile_ort.end_profiling())
                absolute_error = (
                    float(np.max(np.abs(cpu_output - ort_output), initial=0))
                    if cpu_output.size
                    else 0.0
                )
                rows.append(
                    {
                        **asdict(case),
                        "dtype": dtype,
                        "metric": metric,
                        "numerical_atol": tolerance,
                        "maximum_absolute_error": absolute_error,
                        "cpu_samples_seconds": cpu_samples,
                        "ort_samples_seconds": ort_samples,
                        "candidate_order": orders,
                        "cpu_median_seconds": cpu_median,
                        "cpu_p90_seconds": cpu_p90,
                        "ort_median_seconds": ort_median,
                        "ort_p90_seconds": ort_p90,
                        "cpu_iqr_seconds": percentile(cpu_samples, 0.75)
                        - percentile(cpu_samples, 0.25),
                        "ort_iqr_seconds": percentile(ort_samples, 0.75)
                        - percentile(ort_samples, 0.25),
                        "speedup": ort_median / cpu_median,
                        "tail_speedup": ort_p90 / cpu_p90,
                        "profile": profile,
                    }
                )
    if not rows:
        raise RuntimeError("No CDist benchmark cases were selected.")
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_revision": _command_output(("git", "rev-parse", "HEAD")),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "affinity": (
                sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
            ),
            "threads": args.threads,
            "ort_execution_mode": "sequential",
            "ort_inter_op_threads": 1,
            "allocator_policy": (
                "default arenas and memory patterns; outputs allocated by session.run"
            ),
            "input_policy": "same NumPy buffers reused by both runtimes",
            "warmup": args.warmup,
            "seed": args.seed,
            "compiler": _command_output(
                (*shlex.split(os.environ.get("CXX", "c++")), "--version")
            ),
            "compiler_flags": os.environ.get("CXXFLAGS", ""),
            "python": platform.python_version(),
            "versions": _package_versions(),
        },
        "results": rows,
    }
    report["summary"] = summarize(rows)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", choices=[case.name for case in CASES], default=[]
    )
    parser.add_argument("--dtype", action="append", choices=DTYPES, default=[])
    parser.add_argument("--metric", action="append", choices=METRICS, default=[])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpus", default="", help="Linux CPU affinity, for example 0-3 or 0,2.")
    parser.add_argument("--seed", type=int, default=549)
    parser.add_argument("-r", "--repeat", type=int, default=1000)
    parser.add_argument("-w", "--warmup", type=int, default=100)
    parser.add_argument("-t", "--max-repeat-time", type=float, default=1.0)
    parser.add_argument("--profile-runs", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--output", type=Path, default=Path("cdist_parity_results.json"))
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.threads < 1
        or args.repeat < 1
        or args.warmup < 0
        or args.max_repeat_time <= 0
        or args.profile_runs < 0
        or args.rtol < 0
        or args.atol < 0
    ):
        parser.error("counts/time must be positive and tolerances must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["results"]:
        print(
            f"{row['dtype']} {row['metric']} {row['name']} "
            f"({row['m']},{row['k']},{row['n']}): "
            f"cpu={row['cpu_median_seconds'] * 1e6:.2f}us "
            f"ort={row['ort_median_seconds'] * 1e6:.2f}us "
            f"speedup={row['speedup']:.3f}x "
            f"p90={row['cpu_p90_seconds'] * 1e6:.2f}/"
            f"{row['ort_p90_seconds'] * 1e6:.2f}us"
        )
    print(f"raw results: {args.output}")
    print(json.dumps(report["summary"], indent=2))
    return int(args.enforce and not report["summary"]["passed"])


if __name__ == "__main__":
    sys.exit(main())
