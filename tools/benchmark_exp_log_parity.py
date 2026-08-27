#!/usr/bin/env python3
"""Measure float32 Exp/Log end-to-end parity against ONNX Runtime."""

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

SIZES = (100, 1_000, 10_000, 100_000, 1_000_000, 4_194_304, 10_000_000, 100_000_000)
THREAD_COUNTS = ("1", "2", "4", "physical")


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
    samples: tuple[list[float], ...] = tuple([] for _ in functions)
    total_durations = [0.0] * len(functions)
    enabled = gc.isenabled()
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
                samples[index].append(duration)
                total_durations[index] += duration
            if all(duration >= max_repeat_time for duration in total_durations):
                break
    finally:
        if enabled:
            gc.enable()
    return samples


def _physical_threads() -> int:
    topology = Path("/sys/devices/system/cpu")
    cores = {
        path.read_text(encoding="utf-8").strip()
        for path in topology.glob("cpu[0-9]*/topology/core_id")
    }
    return len(cores) or max(1, os.cpu_count() or 1)


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


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith(("model name", "Model")) and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


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


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    speedups = [float(result["speedup"]) for result in results]
    large_model_speedups = {
        str(result["operator"]): float(result["speedup"])
        for result in results
        if result["size"] == 4_194_304
    }
    median = statistics.median(speedups)
    minimum = min(speedups)
    large_models_passed = (
        set(large_model_speedups) == {"Exp", "Log"} and min(large_model_speedups.values()) >= 0.95
    )
    return {
        "passed": median >= 1.0 and minimum >= 0.9 and large_models_passed,
        "thresholds": {"median": 1.0, "minimum": 0.9, "large_model": 0.95},
        "median_speedup": median,
        "minimum_speedup": minimum,
        "large_model_speedups": large_model_speedups,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    selected_cpus = _parse_cpu_list(args.cpus) if args.cpus else None
    if selected_cpus is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("--cpus is only supported on platforms with sched_setaffinity().")
        os.sched_setaffinity(0, selected_cpus)
    import numpy as np
    import onnxruntime
    from onnx_light.onnx import TensorProto, helper
    from onnx_light.onnx.reference import ReferenceEvaluator
    from onnx_light_cpu import register_kernels

    register_kernels()
    physical_threads = _physical_threads()
    requested = physical_threads if args.threads == "physical" else int(args.threads)
    rows: list[dict[str, Any]] = []
    actual_threads = None
    affinity_policy = None
    for operator in ("Exp", "Log"):
        for size in args.sizes:
            values = np.linspace(-8, 8, size, dtype=np.float32)
            if operator == "Log":
                values = np.exp(values)
            model = helper.make_model(
                helper.make_graph(
                    [helper.make_node(operator, ["X"], ["Y"])],
                    f"{operator.lower()}_parity_{size}",
                    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [size])],
                    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [size])],
                ),
                opset_imports=[helper.make_opsetid("", 18)],
                ir_version=13,  # Freeze the model for the onnx-light runtime.
            )
            feeds = {"X": values}
            # Session construction and output allocation are intentionally
            # outside the timed call; dispatch and execution remain measured.
            cpu_execution: dict[str, Any] = {"num_threads": requested}
            if requested > physical_threads:
                cpu_execution["affinity_policy"] = "physical_then_smt"
            cpu = ReferenceEvaluator(model.SerializeToString(), cpu_execution=cpu_execution)
            resolution = cpu.cpu_execution_resolution
            session_threads = resolution.effective_threads
            session_affinity = resolution.request.affinity_policy.name.lower()
            if actual_threads is None:
                actual_threads = session_threads
                affinity_policy = session_affinity
            elif actual_threads != session_threads or affinity_policy != session_affinity:
                raise RuntimeError(
                    "onnx-light resolved inconsistent CPU policies across sessions."
                )
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = requested
            options.inter_op_num_threads = 1
            options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            ort = onnxruntime.InferenceSession(
                model.SerializeToString(),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

            def cpu_run(session=cpu, current_feeds=feeds):
                return session.run(None, current_feeds)[0]

            def ort_run(session=ort, current_feeds=feeds):
                return session.run(None, current_feeds)[0]

            np.testing.assert_allclose(cpu_run(), ort_run(), rtol=2e-5, atol=2e-6)
            cpu_samples, ort_samples = measure_alternating(
                (cpu_run, ort_run), args.repeat, args.warmup, args.max_repeat_time
            )
            cpu_median = statistics.median(cpu_samples)
            ort_median = statistics.median(ort_samples)
            rows.append(
                {
                    "operator": operator,
                    "size": size,
                    "requested_threads": requested,
                    "cpu_samples_seconds": cpu_samples,
                    "ort_samples_seconds": ort_samples,
                    "cpu_median_seconds": cpu_median,
                    "ort_median_seconds": ort_median,
                    "speedup": ort_median / cpu_median if cpu_median else float("inf"),
                    "allocation_and_dispatch_included": True,
                    "sample_order": "alternating; onnx-light-cpu first on even samples",
                }
            )
    if actual_threads is None:
        raise RuntimeError("No benchmark cases were selected.")
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_revision": _command_output(("git", "rev-parse", "HEAD")),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "logical_cpus": os.cpu_count(),
            "physical_cores": physical_threads,
            "affinity": (
                sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
            ),
            "requested_threads": requested,
            "actual_threads": actual_threads,
            "affinity_policy": affinity_policy,
            "configured_thread_counts": THREAD_COUNTS,
            "compiler_flags": os.environ.get("CXXFLAGS", ""),
            "compiler": _command_output(
                (*shlex.split(os.environ.get("CXX", "c++")), "--version")
            ),
            "python": platform.python_version(),
            "versions": _package_versions(),
            "ort_execution_mode": "sequential",
            "ort_inter_op_threads": 1,
        },
        "results": rows,
    }
    report["summary"] = summarize(rows)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", choices=THREAD_COUNTS, default="1")
    parser.add_argument(
        "--cpus", default="", help="Linux CPU affinity, for example 0-3 or 0,2,4."
    )
    parser.add_argument("--size", dest="sizes", action="append", type=int, default=None)
    parser.add_argument("-r", "--repeat", type=int, default=100 * (os.cpu_count() or 1))
    parser.add_argument("-w", "--warmup", type=int, default=100 * 20)
    parser.add_argument("-t", "--max-repeat-time", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("exp_log_parity_results.json"))
    parser.add_argument(
        "--enforce", action="store_true", help="Exit nonzero when the parity gate fails."
    )
    args = parser.parse_args(argv)
    args.sizes = tuple(args.sizes or SIZES)
    if args.repeat < 1 or args.warmup < 0 or args.max_repeat_time <= 0:
        parser.error("repeat and max-repeat-time must be positive and warmup non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["results"]:
        print(
            f"{row['operator']:>3} {row['size']:>10} "
            f"cpu={row['cpu_median_seconds'] * 1e3:.3f}ms "
            f"ort={row['ort_median_seconds'] * 1e3:.3f}ms "
            f"speedup={row['speedup']:.3f}x"
        )
    print(f"raw results: {args.output}")
    print(json.dumps(report["summary"], indent=2))
    return int(args.enforce and not report["summary"]["passed"])


if __name__ == "__main__":
    sys.exit(main())
