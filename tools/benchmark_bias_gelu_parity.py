#!/usr/bin/env python3
"""Measure com.microsoft::BiasGelu end-to-end latency against ONNX Runtime."""

from __future__ import annotations

import argparse
import cProfile
import gc
import importlib.metadata
import json
import os
import platform
import pstats
import shlex
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

CASES = (
    ("empty", (0, 17)),
    ("scalar", (1, 1)),
    ("avx2_before", (1, 7)),
    ("avx2_aligned", (1, 8)),
    ("avx2_after", (1, 9)),
    ("avx512_before", (1, 15)),
    ("avx512_aligned", (1, 16)),
    ("avx512_after", (1, 17)),
    ("exceptional", (1, 9)),
    ("singleton_dims", (1, 1, 63)),
    ("transformer", (128, 768)),
    ("threshold_before", (255, 256)),
    ("threshold_at", (256, 256)),
    ("threshold_after", (257, 256)),
    ("large_outer", (4096, 256)),
)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile."""
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def measure_alternating(
    functions: Sequence[Callable[[], Any]], repeat: int, warmup: int, max_repeat_time: float
) -> tuple[tuple[list[float], ...], list[list[str]]]:
    """Warm equally and alternate candidates to limit ordering bias."""
    labels = ("onnx-light-cpu", "onnxruntime")
    if len(functions) != len(labels):
        raise ValueError("BiasGelu parity measurement requires exactly two candidates.")
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


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the dedicated-machine latency gate."""
    speedups = [float(result["speedup"]) for result in results]
    tail_speedups = [float(result["tail_speedup"]) for result in results]
    median = statistics.median(speedups)
    minimum = min(speedups)
    minimum_tail = min(tail_speedups)
    return {
        "passed": median >= 1.0 and minimum >= 0.9 and minimum_tail >= 0.9,
        "thresholds": {"median": 1.0, "minimum": 0.9, "minimum_tail": 0.9},
        "median_speedup": median,
        "minimum_speedup": minimum,
        "minimum_tail_speedup": minimum_tail,
    }


def _parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.split(","):
        if "-" in item:
            begin, end = (int(part) for part in item.split("-", 1))
            if end < begin:
                raise ValueError(f"Invalid CPU range {item!r}.")
            cpus.update(range(begin, end + 1))
        elif item:
            cpus.add(int(item))
    if not cpus:
        raise ValueError("CPU affinity list must not be empty.")
    return cpus


def _command_output(command: Sequence[str]) -> str:
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
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


def _profile_python(function: Callable[[], Any], repeat: int) -> list[dict[str, Any]]:
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(repeat):
        function()
    profile.disable()
    stats = pstats.Stats(profile)
    rows = []
    for (filename, line, name), values in stats.stats.items():
        primitive_calls, total_calls, total_time, cumulative_time, _ = values
        rows.append(
            {
                "function": f"{filename}:{line}({name})",
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_seconds": total_time,
                "cumulative_seconds": cumulative_time,
            }
        )
    return sorted(rows, key=lambda row: row["cumulative_seconds"], reverse=True)[:10]


def _ort_node_profile(path: str) -> list[dict[str, Any]]:
    events = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for event in events:
        if event.get("cat") == "Node" and event.get("dur") is not None:
            rows.append(
                {
                    "name": event.get("name", ""),
                    "duration_microseconds": event["dur"],
                    "provider": event.get("args", {}).get("provider", ""),
                }
            )
    Path(path).unlink()
    return rows


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
    rows = []
    random = np.random.default_rng(args.seed)
    for case_name, shape in CASES:
        if args.case and case_name not in args.case:
            continue
        inner = shape[-1]
        a = random.standard_normal(shape, dtype=np.float32)
        bias = random.standard_normal(inner, dtype=np.float32)
        if case_name == "exceptional":
            a[0] = np.array(
                [0.0, -0.0, 1.0, -1.0, 5.99, -5.99, np.inf, -np.inf, np.nan],
                dtype=np.float32,
            )
            bias.fill(0)
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("BiasGelu", ["A", "B"], ["C"], domain="com.microsoft")],
                f"bias_gelu_{case_name}",
                [
                    helper.make_tensor_value_info("A", TensorProto.FLOAT, shape),
                    helper.make_tensor_value_info("B", TensorProto.FLOAT, [inner]),
                ],
                [helper.make_tensor_value_info("C", TensorProto.FLOAT, shape)],
            ),
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
            ir_version=13,
        )
        model_bytes = model.SerializeToString()
        feeds = {"A": a, "B": bias}
        cpu = ReferenceEvaluator(model_bytes, cpu_execution={"num_threads": args.threads})
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = True
        options.enable_cpu_mem_arena = True
        if args.profile_runs:
            options.enable_profiling = True
        ort = onnxruntime.InferenceSession(
            model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
        )

        def cpu_run(session=cpu, current_feeds=feeds):
            return session.run(None, current_feeds)[0]

        def ort_run(session=ort, current_feeds=feeds):
            return session.run(None, current_feeds)[0]

        clear_used_kernel_names()
        cpu_output = cpu_run()
        if "onnx_light_cpu::BiasGelu" not in used_kernel_names():
            raise RuntimeError(f"{case_name} did not dispatch the optimized BiasGelu kernel.")
        np.testing.assert_allclose(
            cpu_output, ort_run(), rtol=args.rtol, atol=args.atol, equal_nan=True
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
            for _ in range(args.profile_runs):
                ort_run()
            profile["onnxruntime_nodes"] = _ort_node_profile(ort.end_profiling())
        rows.append(
            {
                "case": case_name,
                "shape": shape,
                "inner": inner,
                "elements": int(a.size),
                "cpu_samples_seconds": cpu_samples,
                "ort_samples_seconds": ort_samples,
                "candidate_order": orders,
                "cpu_median_seconds": cpu_median,
                "cpu_p90_seconds": cpu_p90,
                "ort_median_seconds": ort_median,
                "ort_p90_seconds": ort_p90,
                "cpu_iqr_seconds": percentile(cpu_samples, 0.75) - percentile(cpu_samples, 0.25),
                "ort_iqr_seconds": percentile(ort_samples, 0.75) - percentile(ort_samples, 0.25),
                "speedup": ort_median / cpu_median,
                "tail_speedup": ort_p90 / cpu_p90,
                "profile": profile,
            }
        )
    if not rows:
        raise RuntimeError("No BiasGelu benchmark cases were selected.")
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
        "--case", action="append", choices=[name for name, _ in CASES], default=[]
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpus", default="", help="Linux CPU affinity, for example 0-3 or 0,2.")
    parser.add_argument("--seed", type=int, default=548)
    parser.add_argument("-r", "--repeat", type=int, default=1000)
    parser.add_argument("-w", "--warmup", type=int, default=100)
    parser.add_argument("-t", "--max-repeat-time", type=float, default=1.0)
    parser.add_argument("--profile-runs", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=Path("bias_gelu_parity_results.json"))
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.threads < 1
        or args.repeat < 1
        or args.warmup < 0
        or args.max_repeat_time <= 0
        or args.profile_runs < 0
    ):
        parser.error(
            "threads/repeat/time must be positive; warmup/profile-runs must be non-negative"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["results"]:
        print(
            f"{row['case']} {tuple(row['shape'])}: "
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
