#!/usr/bin/env python3
"""Measure float32 Exp/Log end-to-end parity against ONNX Runtime."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

SIZES = (100, 1_000, 10_000, 100_000, 1_000_000, 4_194_304, 10_000_000, 100_000_000)
THREAD_COUNTS = ("1", "2", "4", "physical")


def _measure(function: Any, repeat: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        function()
    samples: list[float] = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeat):
            start = time.perf_counter_ns()
            function()
            samples.append((time.perf_counter_ns() - start) / 1e9)
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import onnxruntime
    from onnx_light.onnx import TensorProto, helper
    from onnx_light.onnx.reference import ReferenceEvaluator
    from onnx_light_cpu import register_kernels

    register_kernels()
    requested = _physical_threads() if args.threads == "physical" else int(args.threads)
    rows: list[dict[str, Any]] = []
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
                ir_version=13,
            )
            feeds = {"X": values}
            cpu = ReferenceEvaluator(
                model.SerializeToString(), cpu_execution={"num_threads": requested}
            )
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = requested
            options.inter_op_num_threads = 1
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
            cpu_samples = _measure(cpu_run, args.repeat, args.warmup)
            ort_samples = _measure(ort_run, args.repeat, args.warmup)
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
                }
            )
    return {
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "affinity": (
                sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
            ),
            "requested_threads": requested,
            "configured_thread_counts": THREAD_COUNTS,
            "compiler_flags": os.environ.get("CXXFLAGS", ""),
        },
        "results": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", choices=THREAD_COUNTS, default="1")
    parser.add_argument("--size", dest="sizes", action="append", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("exp_log_parity_results.json"))
    args = parser.parse_args(argv)
    args.sizes = tuple(args.sizes or SIZES)
    if args.repeat < 1 or args.warmup < 0:
        parser.error("repeat must be positive and warmup must be non-negative")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
