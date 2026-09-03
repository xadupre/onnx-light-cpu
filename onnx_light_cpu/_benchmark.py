# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark onnx-light-cpu backend test cases."""

from __future__ import annotations

import json
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ._register import (
    clear_used_kernel_names,
    register_backend_test_cases,
    register_kernels,
    used_kernel_names,
)

_DTYPES = (
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
)

_METADATA_COLUMNS = (
    "case",
    "operator",
    "dtype",
    "repeat",
    "warmup",
    "threads",
    "processor",
    "input_shapes",
    "max_repeat_time",
)
_RAW_COLUMNS = (*_METADATA_COLUMNS, "run", "runtime", "duration_s")
_AGGREGATED_COLUMNS = (
    *_METADATA_COLUMNS,
    "samples",
    "mean_s",
    "stdev_s",
    "min_repeat_s",
    "p10_s",
    "median_s",
    "p90_s",
    "max_repeat_s",
    "onnxruntime_samples",
    "onnxruntime_mean_s",
    "onnxruntime_median_s",
    "onnxruntime_error",
    "speedup",
)


def normalize_dtypes(values: Sequence[str]) -> tuple[str, ...]:
    """Validates and normalizes dtype command-line values."""
    requested = [value.lower() for item in values for value in item.split(",")]
    if "all" in requested:
        if len(requested) != 1:
            raise ValueError("'all' cannot be combined with other dtypes")
        return _DTYPES
    unknown = sorted(set(requested) - set(_DTYPES))
    if unknown:
        raise ValueError(f"unknown dtype(s): {', '.join(unknown)}")
    return tuple(dict.fromkeys(requested))


def _case_dtype(name: str) -> str | None:
    tokens = name.lower().split("_")
    return next((dtype for dtype in _DTYPES if dtype in tokens), None)


def _matches_case(name: str, expressions: Sequence[re.Pattern[str]]) -> bool:
    return any(expression.search(name) for expression in expressions)


def _to_numpy(tensor: Any) -> np.ndarray:
    from onnx_light.onnx.helper import (  # pyrefly: ignore[missing-import]
        tensor_dtype_to_np_dtype,
    )

    dtype = tensor_dtype_to_np_dtype(int(tensor.data_type))
    shape = tuple(int(dimension) for dimension in tensor.shape)
    return np.frombuffer(tensor.raw_data(), dtype=dtype).reshape(shape)


def _measure_case(
    case: Any,
    repeat: int,
    warmup: int,
    max_repeat_time: float,
    threads: int,
    with_onnxruntime: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from onnx_light.onnx.reference import (  # pyrefly: ignore[missing-import]
        ReferenceEvaluator,
    )

    model = case.model
    operator = model.graph.node[0].op_type
    feeds = []
    input_names = [value.name for value in model.graph.input]
    for data_set in case.data_sets:
        feeds.append(
            {
                name: _to_numpy(tensor)
                for name, tensor in zip(input_names, data_set.inputs, strict=True)
            }
        )
    input_shapes = json.dumps(
        [{name: list(array.shape) for name, array in feed.items()} for feed in feeds]
    )

    evaluator = ReferenceEvaluator(
        model,
        cpu_execution={  # pyrefly: ignore[unexpected-keyword]
            "num_threads": threads,
            # ORT does not pin workers when intra_op_num_threads is explicit.
            "affinity_policy": "none",
        },
    )
    clear_used_kernel_names()
    for feed in feeds:
        evaluator.run(None, feed)
    expected_kernel = f"onnx_light_cpu::{operator}"
    if expected_kernel not in used_kernel_names():
        raise RuntimeError(f"{case.name}: expected kernel {expected_kernel!r} did not run")

    warmup_start = time.perf_counter()
    for _ in range(warmup):
        for feed in feeds:
            evaluator.run(None, feed)
        if time.perf_counter() - warmup_start >= max_repeat_time:
            break

    durations = []
    repeat_start = time.perf_counter()
    for _ in range(repeat):
        start = time.perf_counter_ns()
        for feed in feeds:
            evaluator.run(None, feed)
        duration_s = (time.perf_counter_ns() - start) / 1_000_000_000
        durations.append(duration_s)
        if time.perf_counter() - repeat_start >= max_repeat_time:
            break

    dtype = _case_dtype(case.name) or "unknown"
    metadata = {
        "case": case.name,
        "operator": operator,
        "dtype": dtype,
        "repeat": repeat,
        "warmup": warmup,
        "threads": threads,
        "processor": platform.processor() or platform.machine(),
        "input_shapes": input_shapes,
        "max_repeat_time": max_repeat_time,
    }
    raw = [
        {
            **metadata,
            "run": run,
            "runtime": "onnx-light-cpu",
            "duration_s": duration,
        }
        for run, duration in enumerate(durations, start=1)
    ]
    aggregate: dict[str, Any] = {
        **metadata,
        "samples": len(durations),
        "mean_s": statistics.fmean(durations),
        "stdev_s": statistics.stdev(durations) if len(durations) > 1 else 0.0,
        "min_repeat_s": min(durations),
        "p10_s": float(np.percentile(durations, 10)),
        "median_s": statistics.median(durations),
        "p90_s": float(np.percentile(durations, 90)),
        "max_repeat_s": max(durations),
        "onnxruntime_samples": None,
        "onnxruntime_mean_s": None,
        "onnxruntime_median_s": None,
        "onnxruntime_error": None,
        "speedup": None,
    }
    if with_onnxruntime:
        import onnxruntime  # pyrefly: ignore[missing-import]

        try:
            session_options = onnxruntime.SessionOptions()
            session_options.intra_op_num_threads = threads
            session_options.inter_op_num_threads = 1
            session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            ort_session = onnxruntime.InferenceSession(
                model.SerializeToString(),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            for feed in feeds:
                ort_session.run(None, feed)
            ort_warmup_start = time.perf_counter()
            for _ in range(warmup):
                for feed in feeds:
                    ort_session.run(None, feed)
                if time.perf_counter() - ort_warmup_start >= max_repeat_time:
                    break
            ort_durations = []
            ort_start = time.perf_counter()
            for _ in range(repeat):
                start = time.perf_counter_ns()
                for feed in feeds:
                    ort_session.run(None, feed)
                ort_durations.append((time.perf_counter_ns() - start) / 1_000_000_000)
                if time.perf_counter() - ort_start >= max_repeat_time:
                    break
        except Exception as exc:  # noqa: BLE001 -- unsupported cases remain in the report.
            aggregate["onnxruntime_error"] = str(exc)
        else:
            raw.extend(
                {
                    **metadata,
                    "run": run,
                    "runtime": "onnxruntime",
                    "duration_s": duration,
                }
                for run, duration in enumerate(ort_durations, start=1)
            )
            aggregate["onnxruntime_samples"] = len(ort_durations)
            aggregate["onnxruntime_mean_s"] = statistics.fmean(ort_durations)
            aggregate["onnxruntime_median_s"] = statistics.median(ort_durations)
            aggregate["speedup"] = (
                aggregate["onnxruntime_median_s"] / aggregate["median_s"]
                if aggregate["median_s"]
                else float("inf")
            )
    return raw, aggregate


def run_backend_benchmark(
    tests: Sequence[str],
    dtypes: Sequence[str],
    repeat: int,
    warmup: int,
    max_repeat_time: float,
    threads: int,
    with_onnxruntime: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Runs selected BENCHMARK backend cases and returns raw and aggregate rows."""
    from onnx_light.onnx.backend import (  # pyrefly: ignore[missing-import]
        TestMode,
        collect_test_cases_by_name,
    )

    if repeat <= 0:
        raise ValueError("repeat must be greater than 0")
    if warmup < 0:
        raise ValueError("warmup must be greater than or equal to 0")
    if max_repeat_time <= 0:
        raise ValueError("max_repeat_time must be greater than 0")
    if threads <= 0:
        raise ValueError("threads must be greater than 0")
    if with_onnxruntime:
        import onnxruntime  # noqa: F401  # pyrefly: ignore[missing-import]

    selected_dtypes = set(normalize_dtypes(dtypes))
    test_expressions = [re.compile(expression) for expression in tests]
    register_backend_test_cases()
    register_kernels()
    cases = [
        case
        for case in collect_test_cases_by_name(
            "^test_cpu_.*_benchmark$",
            mode=TestMode.BENCHMARK,
            generate_benchmark_expected_outputs=False,  # pyrefly: ignore[unexpected-keyword]
        )
        if (
            _matches_case(case.name, test_expressions)
            and _case_dtype(case.name) in selected_dtypes
        )
    ]
    if not cases:
        raise ValueError(
            f"no benchmark backend test matches tests={list(tests)!r}, dtypes={list(dtypes)!r}"
        )

    raw_rows = []
    aggregated_rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.name}", file=sys.stderr, flush=True)
        raw, aggregate = _measure_case(
            case, repeat, warmup, max_repeat_time, threads, with_onnxruntime
        )
        raw_rows.extend(raw)
        aggregated_rows.append(aggregate)
        case.unload()
    return raw_rows, aggregated_rows


def write_benchmark_workbook(
    path: str | os.PathLike[str],
    raw_rows: Sequence[dict[str, Any]],
    aggregated_rows: Sequence[dict[str, Any]],
) -> None:
    """Writes raw and aggregated benchmark data to an Excel workbook."""
    from openpyxl import Workbook  # pyrefly: ignore[missing-import]

    output = Path(path)
    if output.suffix.lower() != ".xlsx":
        raise ValueError("benchmark output must have an .xlsx extension")
    output.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    raw_sheet = workbook.active
    assert raw_sheet is not None
    raw_sheet.title = "raw"
    raw_sheet.append(_RAW_COLUMNS)
    for row in raw_rows:
        raw_sheet.append([row[column] for column in _RAW_COLUMNS])

    aggregate_sheet = workbook.create_sheet("aggregated")
    aggregate_sheet.append(_AGGREGATED_COLUMNS)
    for row in aggregated_rows:
        aggregate_sheet.append([row[column] for column in _AGGREGATED_COLUMNS])
    workbook.save(output)


def write_benchmark_markdown(
    path: str | os.PathLike[str], aggregated_rows: Sequence[dict[str, Any]]
) -> None:
    """Writes aggregated benchmark data as a Markdown table."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = _AGGREGATED_COLUMNS
    lines = [
        f"| {' | '.join(columns)} |",
        f"| {' | '.join('---' for _ in columns)} |",
    ]
    lines.extend(
        f"| {' | '.join(str(row[column]).replace('|', r'\\|') for column in columns)} |"
        for row in aggregated_rows
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
