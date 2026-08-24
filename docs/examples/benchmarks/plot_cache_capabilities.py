"""
L1, L2, and L3 CPU cache capabilities
======================================

This example uses :func:`onnx_light_cpu.benchmark_processor_performance` to
compare the process-visible CPU cache levels. It shows their detected capacity,
effective single-thread bandwidth, and dependent-load latency.

The measurements use working sets chosen safely inside each cache rather than
the exact cache capacity. They describe the effective performance observed on
this host, not a hardware maximum or physical cache-link rate.
"""

# %%
# Measure the cache levels
# ------------------------
# ``UNITTEST_GOING=1`` keeps the gallery test quick while still exercising the
# same public profiling API and plotting code.

import os

import matplotlib.pyplot as plt
import numpy as np

from onnx_light_cpu import benchmark_processor_performance

unit_test_going = os.environ.get("UNITTEST_GOING", "0") in ("1", "true", "True")

profile = benchmark_processor_performance(
    thread_policies=("single",),
    repeats=2 if unit_test_going else 7,
    minimum_duration_ms=1.0 if unit_test_going else 20.0,
    memory_budget_bytes=(8 * 1024 * 1024) if unit_test_going else (256 * 1024 * 1024),
    include_latency=True,
)

cache_levels = [level for level in ("L1", "L2", "L3") if level in profile.memory]
assert cache_levels

# %%
# Inspect the measurements
# ------------------------
# A level can be absent when the host cannot report or measure it reliably. The
# exact working-set size is printed so results remain interpretable.

cache_sizes = {}
for cache in profile.topology.caches:
    level = f"L{cache.level}"
    if level in cache_levels and cache.kind in ("data", "unified"):
        cache_sizes[level] = max(cache_sizes.get(level, 0), cache.size_bytes)

print(
    "{:<5} {:>10} {:>16} {:>11} {:>12}".format(
        "level", "cache KiB", "working-set KiB", "read GB/s", "latency ns"
    )
)
for level in cache_levels:
    measurement = profile.memory[level]["single"]
    reference = measurement.read or measurement.write or measurement.copy
    cache_kib = cache_sizes.get(level, 0) / 1024
    working_set_kib = reference.working_set_bytes / 1024 if reference else float("nan")
    read_gbps = measurement.read.median_gbps if measurement.read else float("nan")
    latency_ns = measurement.latency.median_ns_per_load if measurement.latency else float("nan")
    print(
        f"{level:<5} {cache_kib:>10.0f} {working_set_kib:>16.0f} "
        f"{read_gbps:>11.2f} {latency_ns:>12.2f}"
    )

for warning in profile.warnings:
    print(f"warning: {warning}")

# %%
# Compare capacity, bandwidth, and latency
# ----------------------------------------
# Bandwidth includes all four traffic patterns exposed by the profiler. Missing
# measurements remain ``NaN`` instead of being presented as zero performance.

x = np.arange(len(cache_levels))
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

axes[0].bar(x, [cache_sizes.get(level, float("nan")) / 1024 for level in cache_levels])
axes[0].set_ylabel("detected capacity (KiB)")
axes[0].set_title("cache capacity")

modes = ("read", "write", "copy", "read_modify_write")
width = 0.2
for index, mode in enumerate(modes):
    values = []
    for level in cache_levels:
        measurement = getattr(profile.memory[level]["single"], mode)
        values.append(measurement.median_gbps if measurement else float("nan"))
    axes[1].bar(x + (index - 1.5) * width, values, width, label=mode.replace("_", " "))
axes[1].set_ylabel("effective bandwidth (GB/s)")
axes[1].set_title("single-thread bandwidth")
axes[1].legend(fontsize=7)

latencies = [
    (
        profile.memory[level]["single"].latency.median_ns_per_load
        if profile.memory[level]["single"].latency
        else float("nan")
    )
    for level in cache_levels
]
axes[2].bar(x, latencies)
axes[2].set_ylabel("effective latency (ns/load)")
axes[2].set_title("dependent-load latency")

for axis in axes:
    axis.set_xticks(x)
    axis.set_xticklabels(cache_levels)
    axis.grid(axis="y", alpha=0.25)

fig.suptitle("Process-visible L1/L2/L3 CPU cache capabilities")
fig.tight_layout()
plt.show()
