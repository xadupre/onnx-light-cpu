#!/usr/bin/env python3
"""Measure and rank the AVX2 backend corpus against ONNX Runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

THREAD_POLICIES = ("1", "physical")
AVX2_DTYPES = ("float32", "float16")
AVX2_CORPUS = (
    (r"^test_cpu_(gemm|matmul)_", ("float32", "float16")),
    (r"^test_cpu_(attention|group_query_attention)_", ("float32", "float16")),
    (
        (
            r"^test_cpu_(batchnormalization|groupnormalization|instancenormalization|"
            r"layernormalization|lpnormalization|meanvariancenormalization|sigmoid|softmax)_"
        ),
        ("float32",),
    ),
    (r"^test_cpu_(abs|exp|log)_", ("float32",)),
    (
        (
            r"^test_cpu_(add|sub|mul|div|mod|pow|equal|greater|greater_or_equal|less|"
            r"less_or_equal|and|or|xor|bitwise_and|bitwise_or|bitwise_xor|bitshift|prelu)_"
        ),
        ("float32",),
    ),
)
AVX2_TESTS = tuple(pattern for pattern, _ in AVX2_CORPUS)
SPEEDUP_GROUPS = ("<0.5x", "0.5x-0.9x", "0.9x-1.0x", ">=1.0x")
_LOOP_FAMILIES = (
    "left_scalar",
    "right_scalar",
    "per_channel",
    "contiguous",
    "row",
    "outer",
    "general",
)


def _command_output(command: Sequence[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return (
        completed.stdout.splitlines()[0]
        if completed.returncode == 0 and completed.stdout
        else "unknown"
    )


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "onnx-light", "onnx-light-cpu", "onnxruntime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith(("model name", "Model")) and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def physical_core_count() -> int:
    """Returns the number of process-visible physical cores."""
    affinity = (
        set(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else set(range(os.cpu_count() or 1))
    )
    topology = Path("/sys/devices/system/cpu")
    cores = set()
    for cpu in affinity:
        cpu_path = topology / f"cpu{cpu}" / "topology"
        package_path = cpu_path / "physical_package_id"
        core_path = cpu_path / "core_id"
        if package_path.exists() and core_path.exists():
            cores.add(
                (
                    package_path.read_text(encoding="utf-8").strip(),
                    core_path.read_text(encoding="utf-8").strip(),
                )
            )
    return len(cores) or len(affinity) or 1


def _family(operator: str) -> str:
    if operator in {"Gemm", "MatMul"}:
        return "GEMM/MatMul"
    if operator in {"Attention", "GroupQueryAttention"}:
        return "Attention"
    if operator in {
        "BatchNormalization",
        "GroupNormalization",
        "InstanceNormalization",
        "LayerNormalization",
        "LpNormalization",
        "MeanVarianceNormalization",
        "RMSNormalization",
        "Sigmoid",
        "Softmax",
        "SwiGLU",
    }:
        return "activation/normalization"
    if operator in {"Abs", "Exp", "Log"}:
        return "unary"
    return "binary elementwise"


def _workload(case: str) -> str:
    if "qwen" not in case:
        return "general"
    if "decode" in case or "_m1_" in case:
        return "Qwen decode"
    if "prefill" in case:
        return "Qwen prefill"
    return "Qwen"


def _loop_family(case: str) -> str:
    return next((family for family in _LOOP_FAMILIES if f"_{family}_" in case), "operator")


def speedup_group(speedup: float) -> str:
    """Maps ORT median / onnx-light-cpu median speedup to the published buckets."""
    if speedup < 0.5:
        return "<0.5x"
    if speedup < 0.9:
        return "0.5x-0.9x"
    if speedup < 1.0:
        return "0.9x-1.0x"
    return ">=1.0x"


def build_report(
    raw_rows: Sequence[dict[str, Any]],
    aggregated_rows: Sequence[dict[str, Any]],
    *,
    environment: str,
    command: str,
    simd_level: str,
) -> dict[str, Any]:
    """Builds the ranked JSON report from complete raw measurements."""
    raw_by_case: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in raw_rows:
        key = (str(row["case"]), str(row["thread_policy"]))
        raw_by_case.setdefault(key, {}).setdefault(str(row["runtime"]), []).append(
            float(row["duration_s"])
        )

    results = []
    unsupported = []
    for aggregate in aggregated_rows:
        key = (str(aggregate["case"]), str(aggregate["thread_policy"]))
        samples = raw_by_case.get(key, {})
        if set(samples) != {"onnx-light-cpu", "onnxruntime"}:
            unsupported.append(
                {
                    "case": aggregate["case"],
                    "thread_policy": aggregate["thread_policy"],
                    "threads": aggregate["threads"],
                    "onnxruntime_error": aggregate.get("onnxruntime_error"),
                    "onnx_light_cpu_samples_s": samples.get("onnx-light-cpu", []),
                    "onnxruntime_samples_s": samples.get("onnxruntime", []),
                }
            )
            continue
        cpu_samples = samples["onnx-light-cpu"]
        ort_samples = samples["onnxruntime"]
        cpu_median = statistics.median(cpu_samples)
        ort_median = statistics.median(ort_samples)
        if (
            cpu_median <= 0
            or ort_median <= 0
            or not math.isfinite(cpu_median)
            or not math.isfinite(ort_median)
        ):
            raise RuntimeError(f"{aggregate['case']}: timing samples must be finite and positive")
        speedup = ort_median / cpu_median
        gap = cpu_median - ort_median
        results.append(
            {
                "case": aggregate["case"],
                "family": _family(str(aggregate["operator"])),
                "operator": aggregate["operator"],
                "dtype": aggregate["dtype"],
                "workload": _workload(str(aggregate["case"])),
                "loop_family": _loop_family(str(aggregate["case"])),
                "thread_policy": aggregate["thread_policy"],
                "threads": aggregate["threads"],
                "input_shapes": json.loads(str(aggregate["input_shapes"])),
                "runtime_order": str(aggregate["runtime_order"]).split(","),
                "onnx_light_cpu_samples_s": cpu_samples,
                "onnxruntime_samples_s": ort_samples,
                "onnx_light_cpu_median_s": cpu_median,
                "onnx_light_cpu_stdev_s": (
                    statistics.stdev(cpu_samples) if len(cpu_samples) > 1 else 0.0
                ),
                "onnxruntime_median_s": ort_median,
                "onnxruntime_stdev_s": (
                    statistics.stdev(ort_samples) if len(ort_samples) > 1 else 0.0
                ),
                "latency_gap_s": gap,
                "absolute_latency_contribution_s": max(0.0, gap),
                "speedup": speedup,
                "speedup_group": speedup_group(speedup),
                "decision": "diagnostic" if environment == "shared" else "measured",
            }
        )

    results.sort(
        key=lambda row: (
            -float(row["absolute_latency_contribution_s"]),
            float(row["speedup"]),
        )
    )
    for rank, row in enumerate(results, start=1):
        row["gap_rank"] = rank

    groups = {
        group: [row["gap_rank"] for row in results if row["speedup_group"] == group]
        for group in SPEEDUP_GROUPS
    }
    follow_ups = [
        {
            "gap_rank": row["gap_rank"],
            "case": row["case"],
            "thread_policy": row["thread_policy"],
            "latency_gap_s": row["latency_gap_s"],
            "speedup": row["speedup"],
        }
        for row in results
        if float(row["speedup"]) < 0.9 and float(row["latency_gap_s"]) > 0
    ]
    return {
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "command": command,
            "environment": environment,
            "final_parity_decision": environment == "pinned",
            "simd_ceiling": "AVX2",
            "detected_simd_level": simd_level,
            "cpu_model": _cpu_model(),
            "affinity": (
                sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
            ),
            "logical_cpus": os.cpu_count(),
            "physical_cores": physical_core_count(),
            "platform": platform.platform(),
            "compiler": _command_output(
                (*shlex.split(os.environ.get("CXX", "c++")), "--version")
            ),
            "compiler_flags": os.environ.get("CXXFLAGS", ""),
            "python": platform.python_version(),
            "git_revision": _command_output(("git", "rev-parse", "HEAD")),
            "versions": _package_versions(),
        },
        "groups": groups,
        "complete": not unsupported,
        "unsupported": unsupported,
        "follow_up_candidates": follow_ups,
        "raw": list(raw_rows),
        "results": results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Renders a concise grouped and ranked summary."""
    metadata = report["metadata"]
    lines = ["# AVX2 parity gaps", ""]
    if not metadata["final_parity_decision"]:
        lines.extend(
            [
                (
                    "> **Diagnostic only:** shared-runner measurements do not make final 5-10% "
                    "parity decisions. Rerun on pinned native AVX2 hardware."
                ),
                "",
            ]
        )
    lines.extend(
        [
            (
                "Speedup is ONNX Runtime median latency divided by onnx-light-cpu median "
                "latency; values at or above 1.0x favor onnx-light-cpu."
            ),
            "",
            f"- CPU: `{metadata['cpu_model']}`",
            (
                f"- SIMD ceiling/detected: `{metadata['simd_ceiling']}` / "
                f"`{metadata['detected_simd_level']}`"
            ),
            (
                f"- Physical cores: `{metadata['physical_cores']}`; affinity: "
                f"`{metadata['affinity']}`"
            ),
            "",
        ]
    )
    if report["unsupported"]:
        lines.extend(
            [
                (
                    "> **Incomplete:** ONNX Runtime did not produce samples for "
                    f"{len(report['unsupported'])} selected case(s)."
                ),
                "",
            ]
        )
    by_rank = {row["gap_rank"]: row for row in report["results"]}
    for group in SPEEDUP_GROUPS:
        lines.extend(
            [
                f"## {group} onnx-light-cpu / ONNX Runtime",
                "",
                "| rank | family | workload | case | threads | gap (ms) | speedup |",
                "| ---: | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for rank in report["groups"][group]:
            row = by_rank[rank]
            lines.append(
                f"| {rank} | {row['family']} | {row['workload']} | `{row['case']}` | "
                f"{row['threads']} | {row['latency_gap_s'] * 1e3:.3f} | "
                f"{row['speedup']:.3f}x |"
            )
        if not report["groups"][group]:
            lines.append("| - | - | - | No cases | - | - | - |")
        lines.append("")

    lines.extend(["## Measured follow-up candidates", ""])
    if report["follow_up_candidates"]:
        lines.append("| rank | case | thread policy | gap (ms) | speedup |")
        lines.append("| ---: | --- | --- | ---: | ---: |")
        for row in report["follow_up_candidates"]:
            lines.append(
                f"| {row['gap_rank']} | `{row['case']}` | {row['thread_policy']} | "
                f"{row['latency_gap_s'] * 1e3:.3f} | {row['speedup']:.3f}x |"
            )
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype", "--dtypes", dest="dtypes", action="append", choices=AVX2_DTYPES, default=None
    )
    parser.add_argument(
        "--thread-policy",
        "--thread-policies",
        dest="thread_policies",
        action="append",
        choices=THREAD_POLICIES,
        default=None,
    )
    parser.add_argument("-r", "--repeat", type=int, default=100 * (os.cpu_count() or 1))
    parser.add_argument("-w", "--warmup", type=int, default=20)
    parser.add_argument("-t", "--max-repeat-time", type=float, default=1.0)
    parser.add_argument(
        "--environment",
        choices=("shared", "pinned"),
        default="shared",
        help="Whether results are diagnostic shared-runner data or final pinned-host data.",
    )
    parser.add_argument("--output", type=Path, default=Path("avx2_parity_results.json"))
    args = parser.parse_args(argv)
    args.dtypes = tuple(args.dtypes or AVX2_DTYPES)
    args.thread_policies = tuple(args.thread_policies or THREAD_POLICIES)
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.max_repeat_time <= 0:
        parser.error("--max-repeat-time must be positive")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    from onnx_light_cpu import SimdLevel, detect_simd_level
    from onnx_light_cpu._benchmark import run_backend_benchmark

    simd_level = detect_simd_level()
    if simd_level != SimdLevel.AVX2:
        raise RuntimeError(
            f"AVX2 benchmark requires detected SIMD level AVX2, got {simd_level.name}"
        )

    raw_rows = []
    aggregated_rows = []
    physical_threads = physical_core_count()
    for policy in args.thread_policies:
        threads = 1 if policy == "1" else physical_threads
        for pattern, corpus_dtypes in AVX2_CORPUS:
            selected_dtypes = tuple(dtype for dtype in corpus_dtypes if dtype in args.dtypes)
            if not selected_dtypes:
                continue
            raw, aggregated = run_backend_benchmark(
                tests=(pattern,),
                dtypes=selected_dtypes,
                repeat=args.repeat,
                warmup=args.warmup,
                max_repeat_time=args.max_repeat_time,
                threads=threads,
                with_onnxruntime=True,
                alternate_runtime_order=True,
            )
            for row in (*raw, *aggregated):
                row["thread_policy"] = policy
            raw_rows.extend(raw)
            aggregated_rows.extend(aggregated)

    return build_report(
        raw_rows,
        aggregated_rows,
        environment=args.environment,
        command=shlex.join(sys.argv),
        simd_level=simd_level.name,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"raw results: {args.output}")
    print(f"summary: {markdown}")
    return int(not report["complete"])


if __name__ == "__main__":
    sys.exit(main())
