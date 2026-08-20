#!/usr/bin/env python3
"""Measures MatMulInteger parity against ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from tools.benchmark_gemm_parity import (
    _metadata,
    _parse_cpu_list,
    _percentile,
    measure_alternating,
)


@dataclass(frozen=True)
class IntegerGemmCase:
    name: str
    m: int
    n: int
    k: int

    @property
    def operations(self) -> int:
        return 2 * self.m * self.n * self.k


PRIORITY_CASES = (
    IntegerGemmCase("tiny", 1, 64, 64),
    IntegerGemmCase("direct", 32, 128, 16),
    IntegerGemmCase("square_128", 128, 128, 128),
    IntegerGemmCase("square_512", 512, 512, 512),
    IntegerGemmCase("skinny_m", 1, 1024, 1024),
    IntegerGemmCase("skinny_n", 1024, 1, 1024),
    IntegerGemmCase("large_k", 32, 32, 4096),
    IntegerGemmCase("transformer", 128, 3072, 768),
)


def repeat_count(case: IntegerGemmCase, minimum: int, maximum: int) -> int:
    """Returns the operation-scaled repeat count."""
    return max(minimum, min(maximum, 200_000_000 // max(1, case.operations)))


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarizes the integer parity gate."""
    speedups = [float(result["speedup"]) for result in results]
    median = statistics.median(speedups)
    minimum = min(speedups)
    return {
        "passed": median >= 1.0 and minimum >= 0.9,
        "median_speedup": median,
        "minimum_speedup": minimum,
        "thresholds": {"median": 1.0, "minimum": 0.9},
    }


def _build_case(case: IntegerGemmCase, rng: Any) -> tuple[bytes, dict[str, Any]]:
    import numpy
    from onnx_light.onnx import TensorProto, checker, helper

    a = rng.integers(0, 256, size=(case.m, case.k), dtype=numpy.uint8)
    b = rng.integers(-128, 128, size=(case.k, case.n), dtype=numpy.int8)
    a_zero_point = numpy.array(128, dtype=numpy.uint8)
    b_zero_point = numpy.array(0, dtype=numpy.int8)
    inputs = [
        helper.make_tensor_value_info("A", TensorProto.UINT8, a.shape),
        helper.make_tensor_value_info("B", TensorProto.INT8, b.shape),
        helper.make_tensor_value_info("AZ", TensorProto.UINT8, ()),
        helper.make_tensor_value_info("BZ", TensorProto.INT8, ()),
    ]
    node = helper.make_node("MatMulInteger", ["A", "B", "AZ", "BZ"], ["Y"])
    graph = helper.make_graph(
        [node],
        f"integer_gemm_parity_{case.name}",
        inputs,
        [helper.make_tensor_value_info("Y", TensorProto.INT32, (case.m, case.n))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=13)
    checker.check_model(model)
    return model.SerializeToString(), {
        "A": a,
        "B": b,
        "AZ": a_zero_point,
        "BZ": b_zero_point,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Runs the integer parity benchmark."""
    if args.cpus:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("--cpus is only supported on platforms with sched_setaffinity().")
        os.sched_setaffinity(0, _parse_cpu_list(args.cpus))
    import numpy
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
    selected_cases = (
        PRIORITY_CASES
        if not args.case
        else tuple(case for case in PRIORITY_CASES if case.name in args.case)
    )
    unknown_cases = set(args.case) - {case.name for case in selected_cases}
    if unknown_cases:
        raise ValueError(f"Unknown cases: {', '.join(sorted(unknown_cases))}.")

    results: list[dict[str, Any]] = []
    actual_threads = None
    for case_index, case in enumerate(selected_cases):
        model_bytes, feeds = _build_case(case, numpy.random.default_rng(case_index))
        cpu_session = ReferenceEvaluator(
            model_bytes,
            cpu_execution={"num_threads": args.threads, "affinity_policy": "none"},
        )
        session_threads = cpu_session.cpu_execution_resolution.effective_threads
        if actual_threads is None:
            actual_threads = session_threads
        elif actual_threads != session_threads:
            raise RuntimeError(
                "onnx-light resolved inconsistent participant counts across sessions."
            )
        ort_session = onnxruntime.InferenceSession(
            model_bytes,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        def cpu_run(session=cpu_session, current_feeds=feeds):
            return session.run(None, current_feeds)[0]

        def ort_run(session=ort_session, current_feeds=feeds):
            return session.run(None, current_feeds)[0]

        numpy.testing.assert_array_equal(cpu_run(), ort_run())
        repeat = repeat_count(case, args.minimum_repeats, args.maximum_repeats)
        cpu_samples, ort_samples = measure_alternating(
            (cpu_run, ort_run), repeat=repeat, warmup=args.warmup
        )
        cpu_median = statistics.median(cpu_samples)
        ort_median = statistics.median(ort_samples)
        result = {
            **asdict(case),
            "repeat": repeat,
            "cpu_samples_seconds": cpu_samples,
            "ort_samples_seconds": ort_samples,
            "cpu_median_seconds": cpu_median,
            "ort_median_seconds": ort_median,
            "cpu_iqr_seconds": _percentile(cpu_samples, 0.75) - _percentile(cpu_samples, 0.25),
            "ort_iqr_seconds": _percentile(ort_samples, 0.75) - _percentile(ort_samples, 0.25),
            "speedup": ort_median / cpu_median,
        }
        results.append(result)
        print(
            f"{case.name:<16} cpu={cpu_median * 1e6:10.2f} us "
            f"ort={ort_median * 1e6:10.2f} us speedup={result['speedup']:.3f}x",
            flush=True,
        )

    if actual_threads is None:
        raise RuntimeError("No benchmark cases were selected.")
    return {
        "metadata": _metadata(args.threads, actual_threads, detect_simd_level()),
        "results": results,
        "summary": summarize(results),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--cpus", default="", help="Linux CPU affinity, for example 0-3 or 0,2,4."
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--minimum-repeats", type=int, default=7)
    parser.add_argument("--maximum-repeats", type=int, default=31)
    parser.add_argument("--output", type=Path, default=Path("integer_gemm_parity_results.json"))
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the CLI."""
    args = parse_args(argv)
    report = run(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"raw results: {args.output}")
    return int(args.enforce and not report["summary"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
