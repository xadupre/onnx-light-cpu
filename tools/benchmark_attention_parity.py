#!/usr/bin/env python3
"""Measure the registered Attention priority corpus against ONNX Runtime."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

KV_BLOCK = 128
DTYPES = ("float32", "float16", "bfloat16")
_CASE_PATTERN = re.compile(
    r"^test_cpu_attention_opset(?P<opset>23|24)_"
    r"(?P<layout>rank3|rank4)_(?P<geometry>mha|gqa|mqa)_"
    r"q(?P<q_length>\d+)_kv(?P<kv_length>\d+)_hd(?P<head_dim>\d+)_"
    r"(?P<mask>none|causal|bool|additive)_"
    r"(?P<cache>stateless|internal_cache|nonpad)_"
    r"(?P<dtype>float32|float16|bfloat16)_benchmark$"
)


def parse_case_name(name: str) -> dict[str, Any]:
    """Decode the globally unique backend case name used in dashboard rows."""
    match = _CASE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid Attention benchmark case name: {name!r}")
    case: dict[str, Any] = match.groupdict()
    for field in ("opset", "q_length", "kv_length", "head_dim"):
        case[field] = int(case[field])
    return case


def measure_alternating(
    functions: Sequence[Callable[[], Any]],
    repeat: int,
    warmup: int,
    max_repeat_time: float = 1.0,
) -> tuple[tuple[list[float], ...], list[list[str]]]:
    warmup_durations = [0.0] * len(functions)
    labels = ("onnx-light-cpu", "onnxruntime")
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
    orders: list[list[str]] = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for iteration in range(repeat):
            order = [(iteration + offset) % len(functions) for offset in range(len(functions))]
            order = [index for index in order if total_durations[index] < max_repeat_time]
            if not order:
                break
            orders.append([labels[index] for index in order])
            for index in order:
                start = time.perf_counter_ns()
                functions[index]()
                duration = (time.perf_counter_ns() - start) / 1e9
                samples[index].append(duration)
                total_durations[index] += duration
    finally:
        if enabled:
            gc.enable()
    return samples, orders


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_dtype: dict[str, dict[str, Any]] = {}
    passed = bool(results)
    for dtype in sorted({str(result["dtype"]) for result in results}):
        selected = [result for result in results if result["dtype"] == dtype]
        speedups = [float(result["speedup"]) for result in selected]
        memory_passed = all(bool(result["memory_gate_passed"]) for result in selected)
        median = statistics.median(speedups)
        minimum = min(speedups)
        dtype_passed = median >= 1.0 and minimum >= 0.9 and memory_passed
        passed = passed and dtype_passed
        by_dtype[dtype] = {
            "median_speedup": median,
            "minimum_speedup": minimum,
            "memory_gate_passed": memory_passed,
            "passed": dtype_passed,
        }
    return {
        "passed": passed and set(by_dtype) == set(DTYPES),
        "thresholds": {"median": 1.0, "minimum": 0.9},
        "by_dtype": by_dtype,
    }


def _tensor_numpy(tensor: Any) -> Any:
    import ml_dtypes
    import numpy as np
    from onnx_light.onnx import TensorProto

    data_type = int(tensor.data_type)
    dtypes = {
        int(TensorProto.FLOAT): np.float32,
        int(TensorProto.FLOAT16): np.float16,
        int(TensorProto.BFLOAT16): ml_dtypes.bfloat16,
        int(TensorProto.BOOL): np.bool_,
        int(TensorProto.INT64): np.int64,
    }
    if data_type not in dtypes:
        raise TypeError(f"Unsupported Attention tensor type {data_type}.")
    shape = tuple(int(dimension) for dimension in tensor.shape)
    return np.frombuffer(tensor.raw_data(), dtype=dtypes[data_type]).reshape(shape)


def _cpu_model_and_isa() -> tuple[str, list[str]]:
    cpuinfo = Path("/proc/cpuinfo")
    model = platform.processor() or "unknown"
    isa: list[str] = []
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith(("model name", "Model")) and ":" in line:
                model = line.split(":", 1)[1].strip()
            if line.startswith(("flags", "Features")) and ":" in line:
                isa = line.split(":", 1)[1].strip().split()
                break
    return model, isa


def _cache_metadata() -> list[dict[str, str]]:
    root = Path("/sys/devices/system/cpu/cpu0/cache")
    caches = []
    for index in sorted(root.glob("index*")):
        values = {}
        for field in ("level", "type", "size", "coherency_line_size", "shared_cpu_list"):
            path = index / field
            values[field] = (
                path.read_text(encoding="utf-8").strip() if path.exists() else "unknown"
            )
        caches.append(values)
    return caches


def _command_output(command: Sequence[str]) -> str:
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return output.splitlines()[0] if output else "unknown"


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "ml-dtypes", "onnx-light", "onnx-light-cpu", "onnxruntime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _collect_cases(include_large: bool, patterns: Sequence[str]) -> list[Any]:
    from onnx_light.onnx.backend import TestMode, collect_test_cases_by_name
    from onnx_light_cpu import register_backend_test_cases

    register_backend_test_cases()
    cases = []
    for test_case in collect_test_cases_by_name("^test_cpu_attention_", mode=TestMode.BENCHMARK):
        metadata = parse_case_name(test_case.name)
        if metadata["kv_length"] == 8192 and not include_large:
            continue
        if patterns and not any(re.search(pattern, test_case.name) for pattern in patterns):
            continue
        if not test_case.data_sets:
            raise RuntimeError(f"{test_case.name} did not materialize a data set.")
        cases.append(test_case)
    if not cases:
        raise RuntimeError("No Attention benchmark cases were selected.")
    return cases


def _repeat_count(case: dict[str, Any], minimum: int, maximum: int) -> int:
    work = case["q_length"] * case["kv_length"] * case["head_dim"]
    return max(minimum, min(maximum, 100_000_000 // max(work, 1)))


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import onnxruntime
    from onnx_light.onnx.reference import ReferenceEvaluator
    from onnx_light_cpu import clear_used_kernel_names, register_kernels, used_kernel_names

    if args.cpus:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("--cpus requires os.sched_setaffinity().")
        os.sched_setaffinity(
            0, {int(cpu) for part in args.cpus.split(",") for cpu in _expand_cpu_part(part)}
        )

    cases = _collect_cases(args.large, args.case)
    register_kernels()
    rows = []
    resolved_threads: int | None = None
    for test_case in cases:
        case = parse_case_name(test_case.name)
        model_bytes = test_case.model.SerializeToString()
        initializer_names = {
            initializer.name for initializer in test_case.model.graph.initializer
        }
        input_names = [
            value.name
            for value in test_case.model.graph.input
            if value.name not in initializer_names
        ]
        data_set = test_case.data_sets[0]
        feeds = {
            name: _tensor_numpy(tensor)
            for name, tensor in zip(input_names, data_set.inputs, strict=True)
        }

        cpu = (
            ReferenceEvaluator(test_case.model)
            if args.threads is None
            else ReferenceEvaluator(test_case.model, cpu_execution={"num_threads": args.threads})
        )
        resolution = cpu.cpu_execution_resolution
        current_threads = int(resolution.effective_threads)
        if resolved_threads is None:
            resolved_threads = current_threads
        elif resolved_threads != current_threads:
            raise RuntimeError("onnx-light resolved inconsistent thread counts.")

        if args.threads is None:
            ort = onnxruntime.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        else:
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = args.threads
            options.inter_op_num_threads = 1
            options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            ort = onnxruntime.InferenceSession(
                model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
            )

        def cpu_run(session=cpu, current_feeds=feeds):
            return session.run(None, current_feeds)

        def ort_run(session=ort, current_feeds=feeds):
            return session.run(None, current_feeds)

        clear_used_kernel_names()
        cpu_outputs = cpu_run()
        if "onnx_light_cpu::Attention" not in used_kernel_names():
            raise RuntimeError(
                f"{test_case.name} did not dispatch onnx_light_cpu::Attention: "
                f"{used_kernel_names()}"
            )
        ort_outputs = ort_run()
        tolerance = {"float32": (2e-4, 2e-5), "float16": (2e-2, 2e-3), "bfloat16": (4e-2, 4e-3)}
        rtol, atol = tolerance[case["dtype"]]
        for actual, expected in zip(cpu_outputs, ort_outputs, strict=True):
            np.testing.assert_allclose(
                actual.astype(np.float32),
                expected.astype(np.float32),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )

        (cpu_samples, ort_samples), orders = measure_alternating(
            (cpu_run, ort_run), args.repeat, args.warmup, args.max_repeat_time
        )
        cpu_median = statistics.median(cpu_samples)
        ort_median = statistics.median(ort_samples)
        workers = current_threads
        score_tile_bytes = workers * min(case["kv_length"], KV_BLOCK) * 4
        peak_temporary_bytes = score_tile_bytes + workers * case["head_dim"] * 8
        element_bytes = 4 if case["dtype"] == "float32" else 2
        q_shape = feeds["Q"].shape
        q_heads = (
            int(q_shape[1]) if case["layout"] == "rank4" else int(q_shape[2]) // case["head_dim"]
        )
        kv_bytes = (
            q_heads * case["q_length"] * case["kv_length"] * case["head_dim"] * 2 * element_bytes
        )
        rows.append(
            {
                **case,
                "backend_case_name": test_case.name,
                "cpu_samples_seconds": cpu_samples,
                "ort_samples_seconds": ort_samples,
                "candidate_order": orders,
                "cpu_median_seconds": cpu_median,
                "ort_median_seconds": ort_median,
                "cpu_iqr_seconds": _percentile(cpu_samples, 0.75)
                - _percentile(cpu_samples, 0.25),
                "ort_iqr_seconds": _percentile(ort_samples, 0.75)
                - _percentile(ort_samples, 0.25),
                "speedup": ort_median / cpu_median,
                "peak_temporary_bytes": peak_temporary_bytes,
                "peak_score_tile_bytes": score_tile_bytes,
                "temporary_memory_accounting": (
                    "streaming allocation model: worker_count * "
                    "(Bc + head_dim + v_head_dim) * sizeof(float)"
                ),
                "score_block": {"Br": 1, "Bc": min(case["kv_length"], KV_BLOCK)},
                "worker_count": workers,
                "full_score_or_probability_materialized": False,
                "memory_gate_passed": score_tile_bytes <= workers * KV_BLOCK * 4,
                "tensor_cache_bytes_copied": 0,
                "effective_kv_bandwidth_gbps": kv_bytes / cpu_median / 1e9,
                "tokens_per_second": case["q_length"] / cpu_median,
                "latency_per_token_seconds": cpu_median / case["q_length"],
            }
        )

    cpu_model, isa = _cpu_model_and_isa()
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "policy": "default" if args.threads is None else "equal-thread diagnostic",
            "requested_threads": args.threads,
            "effective_threads": resolved_threads,
            "affinity": (
                sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
            ),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": cpu_model,
            "logical_cpus": os.cpu_count(),
            "cache": _cache_metadata(),
            "isa": isa,
            "compiler": _command_output(
                (*shlex.split(os.environ.get("CXX", "c++")), "--version")
            ),
            "compiler_flags": os.environ.get("CXXFLAGS", ""),
            "git_revision": _command_output(("git", "rev-parse", "HEAD")),
            "python": platform.python_version(),
            "versions": _package_versions(),
            "large_corpus": args.large,
        },
        "results": rows,
    }
    report["summary"] = summarize(rows)
    return report


def _expand_cpu_part(part: str) -> range:
    if "-" not in part:
        value = int(part)
        return range(value, value + 1)
    begin, end = (int(value) for value in part.split("-", 1))
    if end < begin:
        raise ValueError(f"Invalid CPU range {part!r}.")
    return range(begin, end + 1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", default=[], help="Case-name regular expression."
    )
    parser.add_argument("--large", action="store_true", help="Include opt-in KV=8192 cases.")
    parser.add_argument("--threads", type=int, default=None, help="Equal-thread diagnostic.")
    parser.add_argument("--cpus", default="", help="Linux CPU affinity, for example 0-3,8.")
    parser.add_argument("-r", "--repeat", type=int, default=10 * (os.cpu_count() or 1))
    parser.add_argument("-w", "--warmup", type=int, default=2 * (os.cpu_count() or 1))
    parser.add_argument("-t", "--max-repeat-time", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("attention_parity_results.json"))
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.warmup < 0 or args.max_repeat_time <= 0:
        parser.error("repeat and max-repeat-time must be positive and warmup non-negative")
    if args.threads is not None and args.threads < 1:
        parser.error("threads must be positive")
    if args.enforce and args.threads is not None:
        parser.error("--enforce applies only to the default-policy corpus")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["results"]:
        print(
            f"{row['backend_case_name']} cpu={row['cpu_median_seconds'] * 1e3:.3f}ms "
            f"ort={row['ort_median_seconds'] * 1e3:.3f}ms speedup={row['speedup']:.3f}x"
        )
    print(f"raw results: {args.output}")
    print(json.dumps(report["summary"], indent=2))
    return int(args.enforce and not report["summary"]["passed"])


if __name__ == "__main__":
    sys.exit(main())
