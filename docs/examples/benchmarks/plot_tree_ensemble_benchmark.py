"""
Benchmark TreeEnsemble scheduling scenarios
===========================================

This example measures TreeEnsemble-5 regression forests while varying the
number of trees, input features, and batch size. Batch size 1 is represented
explicitly because ONNX Runtime uses a specialized execution path for that
case. The largest forest contains 10,000 trees and the widest input contains
4,096 features, both representative of production models.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
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
    tree_counts = (10, 100)
    feature_counts = (4, 64)
    batch_sizes = (1, 32)
    minimum_repeats = maximum_repeats = "2"
else:
    tree_counts = (10, 100, 1000, 10000)
    feature_counts = (4, 16, 64, 256, 1024, 4096)
    batch_sizes = (1, 8, 32, 128)
    minimum_repeats = "2"
    maximum_repeats = "7"
cases = [
    f"reg_grid_t{trees}_f{features}_b{batch}_f32"
    for trees in tree_counts
    for features in feature_counts
    for batch in batch_sizes
]

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
        minimum_repeats,
        "--maximum-repeats",
        maximum_repeats,
        "--preparation-repeats",
        "1",
        "--output",
        str(output),
    ]
    for case in cases:
        command.extend(["--case", case])
    subprocess.run(command, check=True, cwd=root)
    rows = json.loads(output.read_text(encoding="utf-8"))["inference"]

by_dimensions = {(row["trees"], row["rows"], row["features"]): row for row in rows}
timings = {
    trees: np.array(
        [
            [
                by_dimensions[(trees, batch, features)]["cpu_median_seconds"] * 1e6
                for features in feature_counts
            ]
            for batch in batch_sizes
        ]
    )
    for trees in tree_counts
}
speedups = {
    trees: np.array(
        [
            [by_dimensions[(trees, batch, features)]["speedup"] for features in feature_counts]
            for batch in batch_sizes
        ]
    )
    for trees in tree_counts
}
all_timings = np.concatenate([values.ravel() for values in timings.values()])
all_speedups = np.concatenate([values.ravel() for values in speedups.values()])
timing_norm = LogNorm(vmin=float(all_timings.min()), vmax=float(all_timings.max()))
speedup_norm = TwoSlopeNorm(
    vmin=min(0.5, float(all_speedups.min())),
    vcenter=1.0,
    vmax=max(1.5, float(all_speedups.max())),
)

fig, axes = plt.subplots(
    2,
    len(tree_counts),
    figsize=(5 * len(tree_counts), 9),
    squeeze=False,
    layout="constrained",
)
for column, trees in enumerate(tree_counts):
    timing_image = axes[0, column].imshow(
        timings[trees], aspect="auto", cmap="viridis", norm=timing_norm
    )
    speedup_image = axes[1, column].imshow(
        speedups[trees], aspect="auto", cmap="coolwarm", norm=speedup_norm
    )
    for row_index in range(len(batch_sizes)):
        for feature_index in range(len(feature_counts)):
            axes[0, column].text(
                feature_index,
                row_index,
                f"{timings[trees][row_index, feature_index]:.1f}",
                ha="center",
                va="center",
                color="white",
            )
            axes[1, column].text(
                feature_index,
                row_index,
                f"{speedups[trees][row_index, feature_index]:.2f}x",
                ha="center",
                va="center",
                color="black",
            )
    axes[0, column].set_title(f"{trees:,} trees")

for axis in axes.flat:
    axis.set_xticks(range(len(feature_counts)), feature_counts, rotation=30)
    axis.set_yticks(range(len(batch_sizes)), batch_sizes)
    axis.set_xlabel("number of features")
    axis.set_ylabel("batch size")
fig.colorbar(timing_image, ax=axes[0, :], label="onnx-light CPU median time (us)")
fig.colorbar(speedup_image, ax=axes[1, :], label="speedup over ONNX Runtime")
fig.suptitle("TreeEnsemble: depth 4, float32, one output")
plt.show()
