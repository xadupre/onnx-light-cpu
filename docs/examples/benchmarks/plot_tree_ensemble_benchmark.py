"""
Benchmark TreeEnsemble scheduling scenarios
===========================================

This example visualizes representative TreeEnsemble-5 parity cases against
ONNX Runtime. It invokes the maintained parity runner in a subprocess, which
keeps runtime thread pools isolated and uses the onnx-light reference evaluator
with registered onnx-light-cpu kernels. TreeEnsemble is not currently exposed
as an ``onnx-light-cpu`` backend-test kernel.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np

runner_name = Path("tools") / "benchmark_tree_ensemble_parity.py"
root = None
file_name = globals().get("__file__")
if file_name is not None:
    candidate = Path(file_name).resolve().parents[3]
    if (candidate / runner_name).exists() and (candidate / "CMakeLists.txt").exists():
        root = candidate

if root is None:
    candidate = Path.cwd()
    for parent in [candidate, *candidate.parents]:
        if (parent / runner_name).exists() and (parent / "CMakeLists.txt").exists():
            root = parent
            break
if root is None:
    raise RuntimeError(f"Unable to locate repository root containing {runner_name}.")
runner = root / runner_name
if os.environ.get("UNITTEST_GOING"):
    cases = ["reg_batch_shallow_f32", "reg_large_membership_f32"]
    repeats = "2"
else:
    cases = [
        "reg_batch_shallow_f32",
        "reg_deep_mixed_f32",
        "reg_large_membership_f32",
        "reg_many_targets_f32",
        "reg_large_batch_f64",
    ]
    repeats = "7"

with tempfile.TemporaryDirectory() as temporary:
    output = Path(temporary) / "tree_ensemble.json"
    command = [
        sys.executable,
        str(runner),
        "--threads",
        str(min(4, os.cpu_count() or 1)),
        "--warmup",
        "1",
        "--minimum-repeats",
        repeats,
        "--maximum-repeats",
        repeats,
        "--preparation-repeats",
        "1",
        "--output",
        str(output),
    ]
    for case in cases:
        command.extend(["--case", case])
    subprocess.run(command, check=True, cwd=root)
    rows = json.loads(output.read_text(encoding="utf-8"))["inference"]

labels = [row["name"] for row in rows]
positions = np.arange(len(rows))
cpu = np.array([row["cpu_median_seconds"] for row in rows]) * 1e6
ort = np.array([row["ort_median_seconds"] for row in rows]) * 1e6
speedup = ort / cpu

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
width = 0.4
axes[0].bar(positions - width / 2, cpu, width, label="onnx-light CPU")
axes[0].bar(positions + width / 2, ort, width, label="ONNX Runtime")
axes[0].set_yscale("log")
axes[0].set_ylabel("median time (us)")
axes[0].set_title("TreeEnsemble inference time")
axes[0].legend()
axes[0].grid(True, axis="y")

axes[1].bar(positions, speedup)
axes[1].axhline(1, color="black", linestyle="--")
axes[1].set_ylabel("speedup over ONNX Runtime")
axes[1].set_xticks(positions, labels, rotation=30, ha="right")
axes[1].grid(True, axis="y")
plt.tight_layout()
plt.show()
