"""
Processor performance profile: memory, compute, and Roofline
==============================================================

This example calls :func:`onnx_light_cpu.benchmark_processor_performance` --
the single public entry point produced by the
:doc:`processor performance profile roadmap
<../../next_steps/2026/2026_08_processor_performance_profile>` -- and renders
its returned :class:`~onnx_light_cpu.ProcessorPerformanceProfile` end to end:
topology, working-set sizes, warnings, a compact measurement table, memory
bandwidth/latency plots, an arithmetic throughput plot, and a Roofline chart.

It does **not** re-implement any measurement: every number plotted below comes
straight from the profile returned by that one call. Every reported number is
an *effective, working-set measurement* taken on this exact host -- never a
hardware maximum, a physical link rate, or a guaranteed peak (see the roadmap
document linked above for the full measurement contract).
"""

# %%
# Setup
# -----
#
# ``UNITTEST_GOING=1`` shrinks the measurement (fewer repeats, shorter
# duration, a smaller memory budget) so the example still runs quickly as a
# unit test while exercising every public result path: both thread policies,
# latency, an explicit affinity, and every element type this host can supply.

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from onnx_light_cpu import ExplicitAffinity, benchmark_processor_performance

unit_test_going = os.environ.get("UNITTEST_GOING", "0") in ("1", "true", "True")

profile = benchmark_processor_performance(
    thread_policies=("single", "physical"),
    repeats=2 if unit_test_going else 7,
    minimum_duration_ms=1.0 if unit_test_going else 20.0,
    memory_budget_bytes=(8 * 1024 * 1024) if unit_test_going else (256 * 1024 * 1024),
    include_latency=True,
    explicit_single_affinity=ExplicitAffinity(0, 0),
)

# %%
# Topology, selected affinities, and working-set sizes
# -----------------------------------------------------
#
# Everything below is read from ``profile.topology`` and ``profile.memory`` /
# ``profile.compute`` -- no separate measurement is taken here.

topology = profile.topology
print(f"platform={profile.metadata.platform} compiler={profile.metadata.compiler}")
print(f"timer={profile.metadata.timer_name} schema_version={profile.metadata.schema_version}")
print(
    f"logical_threads={topology.logical_thread_count} "
    f"physical_cores={topology.physical_core_count} "
    f"performance_cores={topology.performance_core_count} "
    f"efficiency_cores={topology.efficiency_core_count} "
    f"cache_topology_detected={topology.cache_topology_detected}"
)
for cache in topology.caches:
    print(
        f"  L{cache.level} {cache.kind:<9} size={cache.size_bytes:>10} bytes "
        f"line={cache.line_size_bytes:>3} sharing_threads={cache.sharing_thread_count:>3} "
        f"confidence={cache.confidence}"
    )

print("\nselected affinities and working-set sizes (memory levels):")
for level, policies in profile.memory.items():
    for policy, entry in policies.items():
        reference = entry.read or entry.write or entry.copy or entry.read_modify_write
        if reference is None:
            continue
        print(
            f"  {level:<3} {policy:<8} participants={reference.participant_count:<2} "
            f"affinity_pinned={reference.affinity_pinned!s:<5} "
            f"working_set={reference.working_set_bytes:>10} bytes"
        )

if profile.warnings:
    print("\nwarnings:")
    for warning in profile.warnings:
        print(f"  - {warning}")

# %%
# Compact measurement table
# -------------------------
#
# One line per available (level/policy) bandwidth+latency measurement and per
# (element type/policy) compute measurement. Values are the *median* of the
# raw samples retained in the profile; all figures are effective measurements
# on this host, not a hardware specification.

print("\nmemory bandwidth (effective, median of raw samples):")
print(f"  {'level':<4} {'policy':<8} {'read GB/s':>10} {'write GB/s':>11} {'copy GB/s':>10}")
for level, policies in profile.memory.items():
    for policy, entry in policies.items():
        read = entry.read.median_gbps if entry.read else float("nan")
        write = entry.write.median_gbps if entry.write else float("nan")
        copy = entry.copy.median_gbps if entry.copy else float("nan")
        print(f"  {level:<4} {policy:<8} {read:>10.2f} {write:>11.2f} {copy:>10.2f}")

print("\ncompute throughput (effective, median of raw samples):")
print(f"  {'dtype':<9} {'policy':<8} {'impl':<10} {'GOP/s':>10}")
for element_type, policies in profile.compute.items():
    for policy, entry in policies.items():
        print(
            f"  {element_type:<9} {policy:<8} {entry.implementation_name:<10} "
            f"{entry.median_gops:>10.2f}"
        )

# %%
# Plot 1: bandwidth by memory level
# ----------------------------------
#
# One group of bars per available memory level, one bar per traffic mode
# (read/write/copy), split by thread policy.

levels = list(profile.memory.keys())
policies_order = [
    p for p in ("single", "physical") if any(p in profile.memory[lv] for lv in levels)
]
modes = ("read", "write", "copy")
mode_colors = {"read": "#4a9eff", "write": "#f4a259", "copy": "#5cb85c"}

fig, axes = plt.subplots(
    1, max(1, len(policies_order)), figsize=(5 * max(1, len(policies_order)), 4.2), squeeze=False
)
for ax, policy in zip(axes[0], policies_order, strict=True):
    x = np.arange(len(levels))
    width = 0.25
    for i, mode in enumerate(modes):
        values = []
        for level in levels:
            entry = profile.memory[level].get(policy)
            measurement = getattr(entry, mode, None) if entry is not None else None
            values.append(measurement.median_gbps if measurement is not None else 0.0)
        ax.bar(x + (i - 1) * width, values, width, label=mode, color=mode_colors[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("effective GB/s")
    ax.set_title(f"memory bandwidth ({policy})")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("plot_processor_performance_bandwidth.png")

# %%
# Plot 2: dependent-load latency by memory level
# ------------------------------------------------

fig_lat, ax_lat = plt.subplots(1, 1, figsize=(6, 4.2))
x = np.arange(len(levels))
width = 0.35
for i, policy in enumerate(policies_order):
    values = []
    for level in levels:
        entry = profile.memory[level].get(policy)
        values.append(
            entry.latency.median_ns_per_load if entry is not None and entry.latency else 0.0
        )
    ax_lat.bar(x + (i - 0.5) * width, values, width, label=policy)
ax_lat.set_xticks(x)
ax_lat.set_xticklabels(levels)
ax_lat.set_ylabel("effective ns / dependent load")
ax_lat.set_title("dependent-load latency by memory level")
ax_lat.legend(fontsize=8)
fig_lat.tight_layout()
fig_lat.savefig("plot_processor_performance_latency.png")

# %%
# Plot 3: arithmetic throughput by element type and thread policy
# -------------------------------------------------------------------

element_types = list(profile.compute.keys())
fig_compute, ax_compute = plt.subplots(1, 1, figsize=(6.5, 4.2))
x = np.arange(len(element_types))
width = 0.35
for i, policy in enumerate(policies_order):
    values = [
        (
            profile.compute[element_type][policy].median_gops
            if policy in profile.compute[element_type]
            else 0.0
        )
        for element_type in element_types
    ]
    ax_compute.bar(x + (i - 0.5) * width, values, width, label=policy)
ax_compute.set_xticks(x)
ax_compute.set_xticklabels(element_types)
ax_compute.set_ylabel("effective GOP/s")
ax_compute.set_title("register-resident arithmetic throughput")
ax_compute.legend(fontsize=8)
fig_compute.tight_layout()
fig_compute.savefig("plot_processor_performance_compute.png")

# %%
# Plot 4: Roofline
# ----------------
#
# Every point below is read from ``profile.roofline``: a horizontal ceiling at
# the measured compute throughput for one element type/policy, a diagonal
# ceiling derived from the measured read bandwidth of one memory level, and
# their crossover arithmetic intensity. These are *derived* quantities that
# still link back to the exact compute and memory measurements they were
# computed from -- not an idealized hardware Roofline.

fig_roof, ax_roof = plt.subplots(1, 1, figsize=(6.5, 5))
intensity = np.logspace(-2, 4, 200)
for element_type, per_policy in profile.roofline.items():
    for policy, per_level in per_policy.items():
        for level, point in per_level.items():
            memory_bound = point.memory_read_gbps * intensity
            roofline_curve = np.minimum(memory_bound, point.compute_gops)
            ax_roof.plot(
                intensity,
                roofline_curve,
                label=f"{element_type}/{policy}/{level}",
                linewidth=1.2,
            )
            ax_roof.scatter(
                [point.arithmetic_intensity_crossover],
                [point.compute_gops],
                marker="o",
                s=14,
            )

ax_roof.set_xscale("log")
ax_roof.set_yscale("log")
ax_roof.set_xlabel("arithmetic intensity (operations / byte)")
ax_roof.set_ylabel("effective GOP/s")
ax_roof.set_title("Roofline (effective measurements, not a hardware specification)")
ax_roof.legend(fontsize=6, ncol=2)
fig_roof.tight_layout()
fig_roof.savefig("plot_processor_performance_roofline.png")

plt.show()

# %%
# Serialization
# -------------
#
# ``to_dict()`` gives a stable, versioned JSON-compatible representation of
# the whole profile (metadata, topology, memory, compute, roofline, and
# warnings), suitable for archiving alongside a run or feeding a future
# optimal-transport GEMM tile-placement cost model.

serialized = profile.to_dict()
assert serialized["metadata"]["schema_version"] == profile.metadata.schema_version
print(f"\nserialized profile keys: {sorted(serialized.keys())}")
print(f"serialized JSON size: {len(json.dumps(serialized))} bytes")
