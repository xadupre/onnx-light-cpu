#!/usr/bin/env python3
"""Measure the TreeEnsemble-5 end-to-end parity gate against ONNX Runtime."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class TreeEnsembleCase:
    name: str
    task: str
    dtype: str
    rows: int
    trees: int
    depth: int
    features: int
    outputs: int
    branch_distribution: str = "homogeneous"
    membership: bool = False
    transform: str = "NONE"
    labels: str = "zero_based"

    @property
    def visits(self) -> int:
        return self.rows * self.trees * self.depth


PRIORITY_CASES = (
    TreeEnsembleCase("reg_scalar_stump_f32", "regression", "float32", 1, 1, 1, 4, 1),
    TreeEnsembleCase("reg_batch_shallow_f32", "regression", "float32", 32, 4, 2, 8, 1),
    TreeEnsembleCase("reg_deep_mixed_f32", "regression", "float32", 8, 16, 10, 32, 2, "mixed"),
    TreeEnsembleCase(
        "reg_large_membership_f32", "regression", "float32", 1024, 81, 1, 8, 1, membership=True
    ),
    TreeEnsembleCase("reg_many_targets_f32", "regression", "float32", 128, 16, 4, 32, 16),
    TreeEnsembleCase("reg_scalar_large_forest_f64", "regression", "float64", 1, 1024, 1, 8, 1),
    TreeEnsembleCase("reg_batch_deep_f64", "regression", "float64", 128, 16, 8, 32, 2),
    TreeEnsembleCase("reg_large_batch_f64", "regression", "float64", 1024, 81, 2, 16, 1),
    TreeEnsembleCase(
        "cls_scalar_binary_f32",
        "classification",
        "float32",
        1,
        4,
        2,
        8,
        2,
        transform="LOGISTIC",
        labels="zero_based",
    ),
    TreeEnsembleCase(
        "cls_batch_strings_f32",
        "classification",
        "float32",
        32,
        16,
        4,
        16,
        10,
        "mixed",
        transform="SOFTMAX",
        labels="string",
    ),
    TreeEnsembleCase(
        "cls_large_classes_f32",
        "classification",
        "float32",
        128,
        81,
        2,
        32,
        128,
        transform="SOFTMAX",
        labels="int64",
    ),
    TreeEnsembleCase(
        "cls_membership_f64",
        "classification",
        "float64",
        1024,
        4,
        1,
        8,
        2,
        membership=True,
    ),
    TreeEnsembleCase(
        "cls_deep_f64",
        "classification",
        "float64",
        8,
        16,
        10,
        32,
        10,
        "mixed",
        transform="SOFTMAX_ZERO",
        labels="string",
    ),
    TreeEnsembleCase(
        "cls_scalar_large_forest_f64",
        "classification",
        "float64",
        1,
        1024,
        1,
        8,
        2,
        labels="int64",
    ),
)

TREE_ENSEMBLE_GRID_FEATURES = (4, 16, 64, 256, 1024, 4096)
TREE_ENSEMBLE_GRID_BATCHES = (1, 8, 32, 128)
TREE_ENSEMBLE_GRID_TREES = (10, 100, 1000, 10000)
BENCHMARK_GRID_CASES = tuple(
    TreeEnsembleCase(
        f"reg_grid_t{trees}_f{features}_b{rows}_f32",
        "regression",
        "float32",
        rows,
        trees,
        4,
        features,
        1,
    )
    for trees in TREE_ENSEMBLE_GRID_TREES
    for features in TREE_ENSEMBLE_GRID_FEATURES
    for rows in TREE_ENSEMBLE_GRID_BATCHES
)
ALL_CASES = PRIORITY_CASES + BENCHMARK_GRID_CASES

THREAD_POLICIES = ("single", "physical")
PARITY_DTYPES = ("float32", "float64")
MEDIAN_GATE = 1.0
MINIMUM_GATE = 0.9
SINGLE_ROW_REGRESSION_GATE = 1.1
MAX_MEASURE_DURATION = 2.0


def repeat_count(case: TreeEnsembleCase, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, 20_000_000 // max(1, case.visits)))


def measure_one(
    function: Callable[[], Any],
    repeat: int,
    warmup: int,
    max_duration: float = MAX_MEASURE_DURATION,
) -> list[float]:
    for _ in range(warmup):
        function()
    timings = []
    total_duration = 0.0
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeat):
            start = time.perf_counter_ns()
            function()
            duration = (time.perf_counter_ns() - start) / 1e9
            timings.append(duration)
            total_duration += duration
            if total_duration >= max_duration:
                break
    finally:
        if gc_enabled:
            gc.enable()
    return timings


def summarize(
    results: Sequence[dict[str, Any]], baseline: dict[str, float] | None = None
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    passed = bool(results)
    for task in ("regression", "classification"):
        for dtype in PARITY_DTYPES:
            speedups = [
                float(result["speedup"])
                for result in results
                if result["task"] == task and result["dtype"] == dtype
            ]
            group_name = f"{task}_{dtype}"
            if not speedups:
                groups[group_name] = {"passed": False, "case_count": 0}
                passed = False
                continue
            median = statistics.median(speedups)
            minimum = min(speedups)
            group_passed = median >= MEDIAN_GATE and minimum >= MINIMUM_GATE
            groups[group_name] = {
                "median_speedup": median,
                "minimum_speedup": minimum,
                "case_count": len(speedups),
                "passed": group_passed,
            }
            passed = passed and group_passed

    single_row: dict[str, Any] = {
        "available": baseline is not None,
        "passed": baseline is not None,
    }
    if baseline is not None:
        ratios = {}
        for result in results:
            if int(result["rows"]) != 1 or result["name"] not in baseline:
                continue
            ratios[str(result["name"])] = float(result["cpu_median_seconds"]) / float(
                baseline[str(result["name"])]
            )
        single_row["ratios"] = ratios
        single_row["passed"] = bool(ratios) and max(ratios.values()) <= SINGLE_ROW_REGRESSION_GATE
    passed = passed and bool(single_row["passed"])

    correctness_passed = all(bool(result.get("correct", False)) for result in results)
    preparation_passed = all(bool(result.get("preparation_passed", False)) for result in results)
    workspace_passed = all(bool(result.get("workspace_passed", False)) for result in results)
    return {
        "passed": passed and correctness_passed and preparation_passed and workspace_passed,
        "thresholds": {
            "median_speedup": MEDIAN_GATE,
            "minimum_speedup": MINIMUM_GATE,
            "single_row_regression": SINGLE_ROW_REGRESSION_GATE,
        },
        "groups": groups,
        "single_row": single_row,
        "correctness_passed": correctness_passed,
        "preparation_passed": preparation_passed,
        "workspace_passed": workspace_passed,
    }


def render_comparison_table(results: Sequence[dict[str, Any]]) -> str:
    rows = [
        (
            "task",
            "dtype",
            "case",
            "rows",
            "features",
            "trees",
            "depth",
            "onnx-light-cpu (us)",
            "onnxruntime (us)",
            "speedup",
        )
    ]
    for result in results:
        rows.append(
            (
                str(result["task"]),
                str(result["dtype"]),
                str(result["name"]),
                str(result["rows"]),
                str(result["features"]),
                str(result["trees"]),
                str(result["depth"]),
                f"{float(result['cpu_median_seconds']) * 1e6:.2f}",
                f"{float(result['ort_median_seconds']) * 1e6:.2f}",
                f"{float(result['speedup']):.3f}x",
            )
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = []
    for index, row in enumerate(rows):
        lines.append(
            "| "
            + " | ".join(cell.ljust(widths[column]) for column, cell in enumerate(row))
            + " |"
        )
        if index == 0:
            lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    return "\n".join(lines)


def _build_case(case: TreeEnsembleCase, seed: int) -> tuple[bytes, dict[str, Any]]:
    import numpy as np
    from onnx_light.onnx import TensorProto, checker, helper, numpy_helper

    dtype = np.float32 if case.dtype == "float32" else np.float64
    tensor_type = TensorProto.FLOAT if case.dtype == "float32" else TensorProto.DOUBLE
    internal_count = (1 << case.depth) - 1
    leaf_count = 1 << case.depth
    nodes_featureids: list[int] = []
    nodes_splits: list[float] = []
    nodes_modes: list[int] = []
    true_ids: list[int] = []
    false_ids: list[int] = []
    true_leafs: list[int] = []
    false_leafs: list[int] = []
    membership_values: list[float] = []
    leaf_targetids: list[int] = []
    leaf_weights: list[float] = []

    for tree in range(case.trees):
        node_offset = tree * internal_count
        leaf_offset = tree * leaf_count
        for local_node in range(internal_count):
            nodes_featureids.append((tree + local_node) % case.features)
            nodes_splits.append(((local_node % 7) - 3) * 0.25)
            mode = (
                6
                if case.membership
                else (local_node % 6 if case.branch_distribution == "mixed" else 0)
            )
            nodes_modes.append(mode)
            if mode == 6:
                membership_values.extend((-1.0, 0.0, 1.0, float("nan")))
            for child, ids, leafs in (
                (2 * local_node + 1, true_ids, true_leafs),
                (2 * local_node + 2, false_ids, false_leafs),
            ):
                if child >= internal_count:
                    ids.append(leaf_offset + child - internal_count)
                    leafs.append(1)
                else:
                    ids.append(node_offset + child)
                    leafs.append(0)
        for leaf in range(leaf_count):
            for target in range(case.outputs):
                leaf_targetids.append(target)
                leaf_weights.append(
                    ((leaf + 3 * target + tree) % 11 - 5) / (max(case.trees, 1) * 8.0)
                )

    attributes: dict[str, Any] = {
        "tree_roots": [tree * internal_count for tree in range(case.trees)],
        "nodes_featureids": nodes_featureids,
        "nodes_splits": numpy_helper.from_array(np.asarray(nodes_splits, dtype=dtype)),
        "nodes_modes": numpy_helper.from_array(np.asarray(nodes_modes, dtype=np.uint8)),
        "nodes_truenodeids": true_ids,
        "nodes_falsenodeids": false_ids,
        "nodes_trueleafs": true_leafs,
        "nodes_falseleafs": false_leafs,
        "leaf_targetids": leaf_targetids,
        "leaf_weights": numpy_helper.from_array(np.asarray(leaf_weights, dtype=dtype)),
        "n_targets": case.outputs,
        "aggregate_function": 1,
        "post_transform": {
            "NONE": 0,
            "SOFTMAX": 1,
            "LOGISTIC": 2,
            "SOFTMAX_ZERO": 3,
        }[case.transform],
    }
    if membership_values:
        attributes["membership_values"] = numpy_helper.from_array(
            np.asarray(membership_values, dtype=dtype)
        )

    nodes = [
        helper.make_node("TreeEnsemble", ["X"], ["scores"], domain="ai.onnx.ml", **attributes)
    ]
    outputs = []
    if case.task == "regression":
        outputs.append(
            helper.make_tensor_value_info("scores", tensor_type, [case.rows, case.outputs])
        )
    else:
        nodes.append(helper.make_node("ArgMax", ["scores"], ["class_index"], axis=1, keepdims=0))
        label_name = "class_index"
        label_type = TensorProto.INT64
        if case.labels == "string":
            labels = [f"class_{index}" for index in range(case.outputs)]
            nodes.append(
                helper.make_node(
                    "LabelEncoder",
                    ["class_index"],
                    ["label"],
                    domain="ai.onnx.ml",
                    keys_int64s=list(range(case.outputs)),
                    values_strings=labels,
                )
            )
            label_name = "label"
            label_type = TensorProto.STRING
        elif case.labels == "int64":
            nodes.append(helper.make_node("Gather", ["labels", "class_index"], ["label"], axis=0))
            label_name = "label"
            label_type = TensorProto.INT64
        outputs.extend(
            [
                helper.make_tensor_value_info(label_name, label_type, [case.rows]),
                helper.make_tensor_value_info("scores", tensor_type, [case.rows, case.outputs]),
            ]
        )

    rng = np.random.default_rng(seed)
    if case.membership:
        values = rng.integers(-2, 3, size=(case.rows, case.features)).astype(dtype)
    else:
        values = rng.standard_normal((case.rows, case.features)).astype(dtype)
    initializers = []
    if case.task == "classification" and case.labels == "int64":
        initializers.append(
            numpy_helper.from_array(np.arange(100, 100 + case.outputs, dtype=np.int64), "labels")
        )
    graph = helper.make_graph(
        nodes,
        f"tree_parity_{case.name}",
        [helper.make_tensor_value_info("X", tensor_type, [case.rows, case.features])],
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("ai.onnx.ml", 5)],
        ir_version=13,
    )
    checker.check_model(model)
    return model.SerializeToString(), {"X": values}


def _physical_threads() -> int:
    topology = Path("/sys/devices/system/cpu")
    cores = {
        (
            path.parent.joinpath("physical_package_id").read_text(encoding="utf-8").strip(),
            path.read_text(encoding="utf-8").strip(),
        )
        for path in topology.glob("cpu[0-9]*/topology/core_id")
    }
    return len(cores) or max(1, os.cpu_count() or 1)


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "onnx-light", "onnx-light-cpu", "onnxruntime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _metadata(requested_threads: int, actual_threads: int) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_revision": revision,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "physical_cores": _physical_threads(),
        "affinity": affinity,
        "requested_threads": requested_threads,
        "actual_threads": actual_threads,
        "thread_policies": THREAD_POLICIES,
        "environment": {
            name: os.environ.get(name, "")
            for name in ("CXX", "CXXFLAGS", "OMP_NUM_THREADS", "ONNXRUNTIME_NUM_THREADS")
        },
        "versions": _package_versions(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cpus:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("--cpus requires sched_setaffinity().")
        os.sched_setaffinity(
            0,
            {
                cpu
                for item in args.cpus.split(",")
                for cpu in (
                    range(int(item.split("-")[0]), int(item.split("-")[1]) + 1)
                    if "-" in item
                    else (int(item),)
                )
            },
        )

    import numpy as np
    import onnxruntime
    from onnx_light.onnx.reference import ReferenceEvaluator
    from onnx_light_cpu import register_kernels

    register_kernels()
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    selected = (
        PRIORITY_CASES
        if not args.case
        else tuple(case for case in ALL_CASES if case.name in args.case)
    )
    unknown = set(args.case) - {case.name for case in ALL_CASES}
    if unknown:
        raise ValueError(f"Unknown cases: {', '.join(sorted(unknown))}.")

    cpu_records = {}
    actual_threads = None
    for index, case in enumerate(selected):
        model, feeds = _build_case(case, index)

        def prepare_cpu(model_bytes=model):
            return ReferenceEvaluator(
                model_bytes,
                cpu_execution={"num_threads": args.threads, "affinity_policy": "none"},
            )

        cpu_preparation = measure_one(prepare_cpu, args.preparation_repeats, 0)
        cpu_session = prepare_cpu()
        resolved_threads = cpu_session.cpu_execution_resolution.effective_threads
        actual_threads = resolved_threads if actual_threads is None else actual_threads
        if actual_threads != resolved_threads:
            raise RuntimeError("onnx-light resolved inconsistent participant counts.")

        def cpu_run(session=cpu_session, current_feeds=feeds):
            return session.run(None, current_feeds)

        cpu_output = cpu_run()
        repeat = repeat_count(case, args.minimum_repeats, args.maximum_repeats)
        cpu_samples = measure_one(cpu_run, repeat=repeat, warmup=args.warmup)
        cpu_records[case.name] = {
            "output": cpu_output,
            "samples": cpu_samples,
            "preparation": cpu_preparation,
            "repeat": repeat,
            "resolved_threads": resolved_threads,
        }

    del cpu_session
    del cpu_run
    gc.collect()

    results = []
    for index, case in enumerate(selected):
        model, feeds = _build_case(case, index)

        def prepare_ort(model_bytes=model):
            return onnxruntime.InferenceSession(
                model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
            )

        ort_preparation = measure_one(prepare_ort, args.preparation_repeats, 0)
        ort_session = prepare_ort()

        def ort_run(session=ort_session, current_feeds=feeds):
            return session.run(None, current_feeds)

        ort_output = ort_run()
        cpu_record = cpu_records[case.name]
        cpu_output = cpu_record["output"]
        tolerance = 2e-5 if case.dtype == "float32" else 1e-9
        if case.task == "classification":
            np.testing.assert_array_equal(cpu_output[0], ort_output[0])
            np.testing.assert_allclose(
                cpu_output[1], ort_output[1], rtol=tolerance, atol=tolerance
            )
        else:
            np.testing.assert_allclose(
                cpu_output[0], ort_output[0], rtol=tolerance, atol=tolerance
            )
        ort_samples = measure_one(
            ort_run,
            repeat=cpu_record["repeat"],
            warmup=args.warmup,
        )
        cpu_samples = cpu_record["samples"]
        cpu_median = statistics.median(cpu_samples)
        ort_median = statistics.median(ort_samples)
        preparation_ratio = statistics.median(cpu_record["preparation"]) / statistics.median(
            ort_preparation
        )
        workspace_bound = min(case.rows, 128) * case.outputs * 16 * cpu_record["resolved_threads"]
        result = {
            **asdict(case),
            "repeat": cpu_record["repeat"],
            "correct": True,
            "cpu_samples_seconds": cpu_samples,
            "ort_samples_seconds": ort_samples,
            "cpu_median_seconds": cpu_median,
            "ort_median_seconds": ort_median,
            "speedup": ort_median / cpu_median,
            "cpu_preparation_samples_seconds": cpu_record["preparation"],
            "ort_preparation_samples_seconds": ort_preparation,
            "preparation_ratio": preparation_ratio,
            "preparation_passed": preparation_ratio <= args.max_preparation_ratio,
            "workspace_upper_bound_bytes": workspace_bound,
            "workspace_budget_bytes": args.workspace_budget_bytes,
            "workspace_passed": workspace_bound <= args.workspace_budget_bytes,
        }
        results.append(result)
        print(
            f"{case.task[:3]} {case.dtype:>7} {case.name:<32} "
            f"cpu={cpu_median * 1e6:10.2f} us ort={ort_median * 1e6:10.2f} us "
            f"speedup={result['speedup']:.3f}x",
            flush=True,
        )

    if actual_threads is None:
        raise RuntimeError("No benchmark cases were selected.")
    baseline = None
    if args.baseline:
        baseline_report = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline = {
            str(row["name"]): float(row["cpu_median_seconds"])
            for row in baseline_report["inference"]
            if int(row["rows"]) == 1
        }
    summary = summarize(results, baseline)
    return {
        "metadata": _metadata(args.threads, actual_threads),
        "preparation": [
            {
                "name": row["name"],
                "cpu_samples_seconds": row["cpu_preparation_samples_seconds"],
                "ort_samples_seconds": row["ort_preparation_samples_seconds"],
                "ratio": row["preparation_ratio"],
                "passed": row["preparation_passed"],
            }
            for row in results
        ],
        "inference": [
            {
                key: value
                for key, value in row.items()
                if not key.startswith(("cpu_preparation_", "ort_preparation_", "workspace_"))
                and key not in {"preparation_ratio", "preparation_passed"}
            }
            for row in results
        ],
        "workspace": [
            {
                "name": row["name"],
                "upper_bound_bytes": row["workspace_upper_bound_bytes"],
                "budget_bytes": row["workspace_budget_bytes"],
                "passed": row["workspace_passed"],
            }
            for row in results
        ],
        "summary": summary,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--cpus", default="", help="Linux CPU affinity, for example 0-3 or 0,2,4."
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--minimum-repeats", type=int, default=7)
    parser.add_argument("--maximum-repeats", type=int, default=31)
    parser.add_argument("--preparation-repeats", type=int, default=3)
    parser.add_argument("--max-preparation-ratio", type=float, default=2.0)
    parser.add_argument("--workspace-budget-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tree_ensemble_parity_results.json"))
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive.")
    if args.minimum_repeats < 1 or args.maximum_repeats < args.minimum_repeats:
        parser.error("repeat bounds must be positive and ordered.")
    if args.preparation_repeats < 1 or args.max_preparation_ratio <= 0:
        parser.error("preparation repeats and ratio must be positive.")
    if args.workspace_budget_bytes < 0:
        parser.error("workspace budget must be non-negative.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = render_comparison_table(report["inference"])
    table_path = args.output.with_suffix(".md")
    table_path.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(json.dumps(report["summary"], indent=2))
    print(f"raw results: {args.output}")
    print(f"comparison table: {table_path}")
    return int(args.enforce and not report["summary"]["passed"])


if __name__ == "__main__":
    sys.exit(main())
