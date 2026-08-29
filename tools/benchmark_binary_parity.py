#!/usr/bin/env python3
"""Run the fixed binary-elementwise parity matrix against ONNX Runtime."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
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

ELEMENT_COUNTS = (4_096, 65_536, 1_048_576, 4_194_304)
THREAD_POLICIES = ("1", "physical")
SHAPE_FAMILIES = (
    "contiguous",
    "left_scalar",
    "right_scalar",
    "row",
    "per_channel",
    "outer",
    "general",
)
PRIORITY_SIGNATURES = {
    ("Add", "float32"),
    ("Add", "bfloat16"),
    ("Sub", "float32"),
    ("Sub", "bfloat16"),
    ("Mul", "float32"),
    ("Mul", "bfloat16"),
    ("Div", "float32"),
    ("Div", "bfloat16"),
    ("PRelu", "float32"),
    ("PRelu", "bfloat16"),
    ("Equal", "float32"),
    ("Equal", "int32"),
    ("Less", "float32"),
    ("Less", "int32"),
    ("And", "bool"),
    ("BitwiseAnd", "int32"),
    ("Pow", "float32"),
    ("Mod", "float32"),
}
ORT_UNSUPPORTED_SIGNATURES = {
    ("Add", "bfloat16"),
    ("Sub", "bfloat16"),
    ("Mul", "bfloat16"),
    ("Div", "bfloat16"),
    ("PRelu", "bfloat16"),
}
_CASE_PATTERN = re.compile(
    r"^test_cpu_(?P<operator>[a-z]+)_v(?P<opset>\d+)_"
    r"(?P<shape_family>contiguous|left_scalar|right_scalar|row|per_channel|outer|general)_"
    r"(?P<left_type>[a-z0-9]+)x(?P<right_type>[a-z0-9]+)_to_(?P<output_type>[a-z0-9]+)"
    r"(?P<attributes>_(?:fmod[01]|left|right))?_n(?P<element_count>\d+)_benchmark$"
)
_OPERATOR_NAMES = {operator.lower(): operator for operator, _ in PRIORITY_SIGNATURES}
_DTYPE_SIZES = {
    "bool": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
}
_GROUP_FIELDS = ("operator", "left_type", "shape_family")


def parse_case_name(name: str) -> dict[str, Any]:
    match = _CASE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid binary benchmark case name: {name!r}")
    case: dict[str, Any] = match.groupdict()
    case["operator"] = _OPERATOR_NAMES.get(case["operator"], case["operator"])
    case["opset"] = int(case["opset"])
    case["element_count"] = int(case["element_count"])
    case["attributes"] = (case["attributes"] or "").removeprefix("_")
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


def summarize(
    results: Sequence[dict[str, Any]],
    unsupported_cases: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    speedups = [float(result["speedup"]) for result in results]
    actual_cases = {
        (
            str(row["operator"]),
            str(row["left_type"]),
            str(row["shape_family"]),
            int(row["element_count"]),
            str(row["thread_policy"]),
        )
        for row in results
    }
    expected_cases = {
        (operator, dtype, family, count, policy)
        for operator, dtype in PRIORITY_SIGNATURES
        for family in SHAPE_FAMILIES
        for count in ELEMENT_COUNTS
        for policy in THREAD_POLICIES
    }
    unsupported_gate_cases = {
        (
            str(row["operator"]),
            str(row["left_type"]),
            str(row["shape_family"]),
            int(row["element_count"]),
            str(row["thread_policy"]),
        )
        for row in unsupported_cases
    }
    unexpected_unsupported_cases = sorted(
        case for case in unsupported_gate_cases if case[:2] not in ORT_UNSUPPORTED_SIGNATURES
    )
    complete = actual_cases | unsupported_gate_cases == expected_cases
    comparable_complete = actual_cases == expected_cases - unsupported_gate_cases
    median = statistics.median(speedups) if speedups else 0.0
    minimum = min(speedups, default=0.0)
    groups: dict[tuple[str, str, str], list[float]] = {}
    for result in results:
        key = tuple(str(result[field]) for field in _GROUP_FIELDS)
        groups.setdefault(key, []).append(float(result["speedup"]))
    group_results = [
        {
            **dict(zip(_GROUP_FIELDS, key, strict=True)),
            "median_speedup": statistics.median(values),
            "minimum_speedup": min(values),
            "case_count": len(values),
            "passed": statistics.median(values) >= 1.0,
        }
        for key, values in sorted(groups.items())
    ]
    groups_passed = bool(group_results) and all(group["passed"] for group in group_results)

    serial_p90 = {
        _comparison_key(row): float(row["cpu_p90_seconds"])
        for row in results
        if row["thread_policy"] == "1" and "cpu_p90_seconds" in row
    }
    small_physical = [
        row
        for row in results
        if row["thread_policy"] == "physical" and int(row["element_count"]) <= 65_536
    ]
    small_p90_complete = bool(small_physical) and all(
        _comparison_key(row) in serial_p90 and "cpu_p90_seconds" in row for row in small_physical
    )
    small_p90_regressions = [
        {
            "backend_case_name": row.get("backend_case_name", ""),
            "serial_p90_seconds": serial_p90[_comparison_key(row)],
            "physical_p90_seconds": float(row["cpu_p90_seconds"]),
            "regression": float(row["cpu_p90_seconds"]) / serial_p90[_comparison_key(row)] - 1.0,
        }
        for row in small_physical
        if _comparison_key(row) in serial_p90
        and "cpu_p90_seconds" in row
        and float(row["cpu_p90_seconds"]) > serial_p90[_comparison_key(row)] * 1.02
    ]
    return {
        "passed": (
            complete
            and comparable_complete
            and not unexpected_unsupported_cases
            and groups_passed
            and minimum >= 0.9
            and small_p90_complete
            and not small_p90_regressions
        ),
        "matrix_complete": complete,
        "comparable_matrix_complete": comparable_complete,
        "thresholds": {
            "group_median": 1.0,
            "minimum": 0.9,
            "small_p90_maximum_regression": 0.02,
        },
        "median_speedup": median,
        "minimum_speedup": minimum,
        "case_count": len(results),
        "groups_passed": groups_passed,
        "groups": group_results,
        "small_p90_complete": small_p90_complete,
        "small_p90_passed": small_p90_complete and not small_p90_regressions,
        "small_p90_regressions": small_p90_regressions,
        "unsupported_case_count": len(unsupported_cases),
        "unexpected_unsupported_cases": unexpected_unsupported_cases,
    }


def _comparison_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["operator"],
        row["left_type"],
        row.get("right_type", row["left_type"]),
        row.get("output_type", row["left_type"]),
        row.get("attributes", ""),
        row["shape_family"],
        int(row["element_count"]),
    )


def _tensor_element_count(tensor: Any) -> int:
    return math.prod(int(dimension) for dimension in tensor.shape)


def _physical_threads() -> int:
    topology = Path("/sys/devices/system/cpu")
    cores = set()
    for core_path in topology.glob("cpu[0-9]*/topology/core_id"):
        package_path = core_path.with_name("physical_package_id")
        package = (
            package_path.read_text(encoding="utf-8").strip() if package_path.exists() else "0"
        )
        cores.add((package, core_path.read_text(encoding="utf-8").strip()))
    return len(cores) or max(1, os.cpu_count() or 1)


def _tensor_numpy(tensor: Any) -> Any:
    import ml_dtypes
    import numpy as np
    from onnx_light.onnx import TensorProto

    dtypes = {
        int(TensorProto.BOOL): np.bool_,
        int(TensorProto.FLOAT): np.float32,
        int(TensorProto.FLOAT16): np.float16,
        int(TensorProto.BFLOAT16): ml_dtypes.bfloat16,
        int(TensorProto.INT32): np.int32,
    }
    data_type = int(tensor.data_type)
    if data_type not in dtypes:
        raise TypeError(f"Unsupported priority binary tensor type {data_type}.")
    shape = tuple(int(dimension) for dimension in tensor.shape)
    return np.frombuffer(tensor.raw_data(), dtype=dtypes[data_type]).reshape(shape)


def _collect_cases(patterns: Sequence[str]) -> list[Any]:
    from onnx_light.onnx.backend import TestMode, collect_test_cases_by_name
    from onnx_light_cpu import register_backend_test_cases

    register_backend_test_cases()
    cases = []
    for test_case in collect_test_cases_by_name(
        "^test_cpu_", mode=TestMode.BENCHMARK, generate_benchmark_expected_outputs=False
    ):
        if "_benchmark" not in test_case.name or "_swapped_" in test_case.name:
            continue
        try:
            metadata = parse_case_name(test_case.name)
        except ValueError:
            continue
        signature = (metadata["operator"], metadata["left_type"])
        if signature not in PRIORITY_SIGNATURES:
            continue
        if metadata["left_type"] != metadata["right_type"]:
            continue
        if patterns and not any(re.search(pattern, test_case.name) for pattern in patterns):
            continue
        cases.append(test_case)
    if not cases:
        raise RuntimeError("No binary benchmark cases were selected.")
    return cases


def _shape_metrics(case: dict[str, Any]) -> tuple[str, int, int]:
    count = int(case["element_count"])
    family = str(case["shape_family"])
    scale = count // 4096
    if family in {"contiguous", "left_scalar", "right_scalar"}:
        return family, count, count
    if family == "row":
        return "repeated_contiguous_block", 16 * scale, 256
    if family == "per_channel":
        return "repeated_contiguous_block", 128 * scale, 32
    if family == "outer":
        return "outer_broadcast", 64 * scale, 64
    return "general_strided", 64 * scale, 64


def _selected_isa(case: dict[str, Any], flags: Sequence[str]) -> str:
    if case["left_type"] != "float32" or case["operator"] not in {"Add", "Sub", "Mul", "Div"}:
        return "scalar"
    available = set(flags)
    if "avx512f" in available:
        return "avx512"
    if "avx2" in available:
        return "avx2"
    if "sse2" in available:
        return "sse2"
    if "sve" in available:
        return "sve"
    if platform.machine().lower() in {"aarch64", "arm64"}:
        return "neon"
    return "scalar"


def _cpu_model_and_flags() -> tuple[str, list[str]]:
    cpuinfo = Path("/proc/cpuinfo")
    model = platform.processor() or "unknown"
    flags: list[str] = []
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith(("model name", "Model")) and ":" in line:
                model = line.split(":", 1)[1].strip()
            if line.startswith(("flags", "Features")) and ":" in line:
                flags = line.split(":", 1)[1].strip().split()
                break
    return model, flags


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


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else "unknown"


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import onnxruntime
    from onnxruntime.capi.onnxruntime_pybind11_state import (
        NotImplemented as OrtNotImplemented,
    )
    from onnx_light.onnx.reference import ReferenceEvaluator
    from onnx_light.onnx_py._onnxpykernels.runtime import CpuExecutionPolicy
    from onnx_light.kernel_tuning import calibrate_kernel_tuning
    from onnx_light_cpu import (
        clear_used_kernel_names,
        register_kernels,
        set_kernel_usage_recording,
        used_kernel_names,
    )

    if args.cpus:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("--cpus requires os.sched_setaffinity().")
        os.sched_setaffinity(0, _parse_cpu_list(args.cpus))
    cases = _collect_cases(args.case)
    register_kernels()
    physical_threads = _physical_threads()
    calibration_reports = []
    if args.calibrate:
        from onnx_light.onnx import TensorProto

        element_types = {
            "bool": int(TensorProto.BOOL),
            "float32": int(TensorProto.FLOAT),
            "bfloat16": int(TensorProto.BFLOAT16),
            "int32": int(TensorProto.INT32),
        }
        policy = CpuExecutionPolicy()
        policy.num_threads = physical_threads
        for operator, dtype in sorted(PRIORITY_SIGNATURES):
            calibration_report = calibrate_kernel_tuning(
                operator,
                element_types=[element_types[dtype]],
                library="onnx_light_cpu",
                implementation="broadcast_plan",
                maximum_duration_ms=args.calibration_duration_ms,
                maximum_memory_bytes=args.calibration_memory_bytes,
                save=False,
                cpu_execution=policy,
            )
            if not calibration_report.get("calibrated"):
                raise RuntimeError(
                    f"Calibration failed for {operator}/{dtype}: {calibration_report!r}."
                )
            calibration_reports.append(calibration_report)
    cpu_model, flags = _cpu_model_and_flags()
    rows = []
    unsupported_cases = []
    for policy in args.threads:
        requested_threads = 1 if policy == "1" else physical_threads
        for test_case in cases:
            case = parse_case_name(test_case.name)
            data_set = test_case.data_sets[0]
            feeds = {
                value.name: _tensor_numpy(tensor)
                for value, tensor in zip(
                    test_case.model.graph.input, data_set.inputs, strict=True
                )
            }
            model_bytes = test_case.model.SerializeToString()
            cpu = ReferenceEvaluator(
                test_case.model, cpu_execution={"num_threads": requested_threads}
            )
            effective_threads = int(cpu.cpu_execution_resolution.effective_threads)
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = requested_threads
            options.inter_op_num_threads = 1
            options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            try:
                ort = onnxruntime.InferenceSession(
                    model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
                )
            except OrtNotImplemented as exc:
                unsupported_cases.append(
                    {
                        **case,
                        "backend_case_name": test_case.name,
                        "thread_policy": policy,
                        "reason": str(exc),
                    }
                )
                continue

            def cpu_run(session=cpu, current_feeds=feeds):
                return session.run(None, current_feeds)

            def ort_run(session=ort, current_feeds=feeds):
                return session.run(None, current_feeds)

            set_kernel_usage_recording(True)
            try:
                clear_used_kernel_names()
                cpu_outputs = cpu_run()
                operation_identity = f"onnx_light_cpu::{case['operator']}"
                if operation_identity not in used_kernel_names():
                    raise RuntimeError(f"{test_case.name} dispatched {used_kernel_names()}.")
            finally:
                set_kernel_usage_recording(False)
            ort_outputs = ort_run()
            for actual, expected in zip(cpu_outputs, ort_outputs, strict=True):
                np.testing.assert_allclose(
                    actual.astype(np.float32),
                    expected.astype(np.float32),
                    rtol=2e-2 if case["left_type"] == "bfloat16" else 1e-5,
                    atol=2e-3 if case["left_type"] == "bfloat16" else 1e-6,
                    equal_nan=True,
                )
            (cpu_samples, ort_samples), orders = measure_alternating(
                (cpu_run, ort_run), args.repeat, args.warmup, args.max_repeat_time
            )
            cpu_median = statistics.median(cpu_samples)
            ort_median = statistics.median(ort_samples)
            left_elements, right_elements = (
                _tensor_element_count(data_set.inputs[0]),
                _tensor_element_count(data_set.inputs[1]),
            )
            output_elements = _tensor_element_count(data_set.outputs[0])
            rows.append(
                {
                    **case,
                    "backend_case_name": test_case.name,
                    "thread_policy": policy,
                    "requested_threads": requested_threads,
                    "effective_threads": effective_threads,
                    "operation_identity": operation_identity,
                    "loop_family": _shape_metrics(case)[0],
                    "isa": _selected_isa(case, flags),
                    "unique_tensor_bytes": (
                        left_elements * _DTYPE_SIZES[case["left_type"]]
                        + right_elements * _DTYPE_SIZES[case["right_type"]]
                        + output_elements * _DTYPE_SIZES[case["output_type"]]
                    ),
                    "expanded_operand_bytes": output_elements
                    * (_DTYPE_SIZES[case["left_type"]] + _DTYPE_SIZES[case["right_type"]]),
                    "cpu_samples_seconds": cpu_samples,
                    "ort_samples_seconds": ort_samples,
                    "candidate_order": orders,
                    "cpu_median_seconds": cpu_median,
                    "cpu_p10_seconds": _percentile(cpu_samples, 0.1),
                    "cpu_p90_seconds": _percentile(cpu_samples, 0.9),
                    "cpu_iqr_seconds": _percentile(cpu_samples, 0.75)
                    - _percentile(cpu_samples, 0.25),
                    "ort_median_seconds": ort_median,
                    "ort_p10_seconds": _percentile(ort_samples, 0.1),
                    "ort_p90_seconds": _percentile(ort_samples, 0.9),
                    "ort_iqr_seconds": _percentile(ort_samples, 0.75)
                    - _percentile(ort_samples, 0.25),
                    "output_elements_per_second": output_elements / cpu_median,
                    "speedup": ort_median / cpu_median,
                }
            )
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_revision": _command_output(("git", "rev-parse", "HEAD")),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": cpu_model,
            "isa_flags": flags,
            "kernel": platform.release(),
            "logical_cpus": os.cpu_count(),
            "physical_cores": physical_threads,
            "cpu_governor": _read_text(
                Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            ),
            "numa_nodes": sorted(
                path.name for path in Path("/sys/devices/system/node").glob("node[0-9]*")
            ),
            "affinity": (
                sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
            ),
            "thread_policies": args.threads,
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")
            },
            "compiler": _command_output(
                (*shlex.split(os.environ.get("CXX", "c++")), "--version")
            ),
            "compiler_flags": os.environ.get("CXXFLAGS", ""),
            "python": platform.python_version(),
            "versions": _package_versions(),
            "ort_execution_mode": "sequential",
            "ort_inter_op_threads": 1,
            "sample_order": "alternating",
            "calibration_enabled": args.calibrate,
            "calibration_reports": calibration_reports,
            "byte_model_note": "Byte models are logical counts, not measured DRAM traffic.",
        },
        "results": rows,
        "unsupported_cases": unsupported_cases,
    }
    report["summary"] = summarize(rows, unsupported_cases)
    return report


def _parse_cpu_list(value: str) -> set[int]:
    cpus = set()
    for part in value.split(","):
        if "-" in part:
            begin, end = (int(item) for item in part.split("-", 1))
            if end < begin:
                raise ValueError(f"Invalid CPU range {part!r}.")
            cpus.update(range(begin, end + 1))
        elif part:
            cpus.add(int(part))
    if not cpus:
        raise ValueError("CPU affinity list must not be empty.")
    return cpus


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[], help="Case-name regex.")
    parser.add_argument("--threads", action="append", choices=THREAD_POLICIES, default=None)
    parser.add_argument("--cpus", default="", help="Linux CPU affinity, for example 0-3,8.")
    parser.add_argument("-r", "--repeat", type=int, default=100 * (os.cpu_count() or 1))
    parser.add_argument("-w", "--warmup", type=int, default=100 * 20)
    parser.add_argument("-t", "--max-repeat-time", type=float, default=1.0)
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish in-memory Binary profiles before creating benchmark sessions.",
    )
    parser.add_argument("--calibration-duration-ms", type=int, default=250)
    parser.add_argument("--calibration-memory-bytes", type=int, default=64 << 20)
    parser.add_argument("--output", type=Path, default=Path("binary_parity_results.json"))
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args(argv)
    args.threads = tuple(args.threads or THREAD_POLICIES)
    if (
        args.repeat < 1
        or args.warmup < 0
        or args.max_repeat_time <= 0
        or args.calibration_duration_ms < 0
        or args.calibration_memory_bytes < 0
    ):
        parser.error(
            "repeat must be positive; warmup and calibration budgets must be non-negative"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["results"]:
        print(
            f"{row['backend_case_name']} threads={row['thread_policy']} "
            f"cpu={row['cpu_median_seconds'] * 1e3:.3f}ms "
            f"ort={row['ort_median_seconds'] * 1e3:.3f}ms speedup={row['speedup']:.3f}x"
        )
    print(f"raw results: {args.output}")
    print(json.dumps(report["summary"], indent=2))
    return int(args.enforce and not report["summary"]["passed"])


if __name__ == "__main__":
    sys.exit(main())
