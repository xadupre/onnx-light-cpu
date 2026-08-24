"""
Benchmark backend test cases against ONNX Runtime
====================================================

onnx-light-cpu ships its own ONNX backend test cases -- named ``test_cpu_*``
-- in a dedicated C++ registration library
(``lib_onnx_light_cpu_backend_test``, see :func:`onnx_light_cpu.register_backend_test_cases`).
Every case that has an accelerated kernel also has a ``TestMode.BENCHMARK``
variant: the same operator and attributes, but with inputs large enough that a
single evaluation takes a measurable amount of time. This example collects
every ``test_cpu_*_benchmark`` case -- across every operator and element type
onnx-light-cpu ships one for -- and times each one through ``onnx-light``
(with onnx-light-cpu's accelerated kernel registered) and through ONNX
Runtime, using the exact same generated model and inputs for both.

Each case is also checked for correctness (both runtimes must agree, within
the case's tolerance) and for kernel dispatch (the accelerated onnx-light-cpu
kernel, not onnx-light's built-in reference kernel, must have run), the same
way :mod:`unittests.python.test_kernels_e2e` verifies these backend cases.
"""

# %%
# Setup
# -----
#
# Every ``test_cpu_*_benchmark`` case registered by onnx-light-cpu is
# collected via :func:`onnx_light.onnx.backend.collect_test_cases_by_name`
# (which accepts an ECMAScript regular expression), regardless of operator or
# element type; ``--filter`` further narrows that set down when only a subset
# is of interest.

import argparse
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import pandas as pd
from tqdm import tqdm

from onnx_light.onnx.backend import TestMode, collect_test_cases_by_name
from onnx_light.onnx.helper import tensor_dtype_to_np_dtype
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import (
    clear_used_kernel_names,
    has_backend_test_cases,
    register_backend_test_cases,
    register_kernels,
    used_kernel_names,
)

assert has_backend_test_cases(), (
    "onnx-light-cpu must be built with onnx-light's backend test registry "
    "(register_backend_test_cases binding unavailable)."
)

# ``--filter`` narrows the collected "test_cpu_*_benchmark" cases down to
# those whose name additionally matches a regular expression (e.g. ``--filter
# gemm`` keeps only Gemm cases, ``--filter '_2d_'`` keeps only 2-D cases
# across every operator). ``parse_known_args`` ignores unrelated arguments
# injected by pytest/sphinx-gallery when this file runs as a test or a
# documentation example.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--filter",
    default=None,
    help="regular expression a case name must additionally match, e.g. '^test_cpu_gemm_'",
)
args, _ = parser.parse_known_args()
_name_filter = re.compile(args.filter) if args.filter else None


def _to_numpy(tensor):
    """Decodes a backend test case ``Tensor`` into a numpy array."""
    dtype = tensor_dtype_to_np_dtype(int(tensor.data_type))
    shape = tuple(int(d) for d in tensor.shape)
    return np.frombuffer(tensor.raw_data(), dtype=dtype).reshape(shape)


def _collect_cases():
    """Registers and collects every "test_cpu_*_benchmark" backend test case.

    Every operator and element type onnx-light-cpu ships a BENCHMARK variant
    for is included; ``--filter`` (``_name_filter``) is the only further
    restriction applied here.
    """
    register_backend_test_cases()
    max_per_case_group = (
        2 if os.environ.get("UNITTEST_GOING", "0") in ("1", "true", "True") else None
    )
    counts: dict[str, int] = {}
    cases = []
    for tc in collect_test_cases_by_name("^test_cpu_.*_benchmark$", mode=TestMode.BENCHMARK):
        nodes = list(tc.model.graph.node)
        if len(nodes) != 1 or not tc.data_sets:
            continue
        if _name_filter is not None and not _name_filter.search(tc.name):
            continue
        op_type = nodes[0].op_type
        if max_per_case_group is not None:
            if counts.get(op_type, 0) >= max_per_case_group:
                continue
            counts[op_type] = counts.get(op_type, 0) + 1
        cases.append(tc)
    return cases


_CASES = _collect_cases()
_no_cases_message = (
    f"no onnx-light-cpu BENCHMARK backend test cases were collected (filter={args.filter!r})"
)
assert _CASES, _no_cases_message

# %%
# Timing helper
# -------------
#
# Each candidate gets three untimed warm-up calls, then is called ``repeat``
# times and the median wall-clock time is retained. ``repeat`` shrinks as the
# case's inputs grow but never below three.


def measure(func, repeat, warmup=3):
    for _ in range(warmup):
        func()
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def _case_element_count(tc):
    return max(
        (
            int(np.prod([int(d) for d in tensor.shape]))
            for ds in tc.data_sets
            for tensor in (*ds.inputs, *ds.outputs)
        ),
        default=1,
    )


# %%
# Run every case through onnx-light-cpu and ONNX Runtime
# -------------------------------------------------------
#
# ``register_kernels()`` installs the accelerated kernels in onnx-light's
# process-wide dispatch table before any of the onnx-light-cpu sessions below
# run for the first time.

register_kernels()

rows = []
for tc in tqdm(_CASES, desc="benchmarking backend cases", unit="case"):
    op_type = tc.model.graph.node[0].op_type
    expected_kernel = f"onnx_light_cpu::{op_type}"
    model_bytes = tc.model.SerializeToString()
    # Some Gemm benchmark cases turn "B" into a graph initializer to exercise
    # the constant-B code path; its value is baked into the model rather than
    # fed at run time, so it must be excluded from the runtime feeds below.
    initializer_names = {init.name for init in tc.model.graph.initializer}
    input_names = [vi.name for vi in tc.model.graph.input if vi.name not in initializer_names]

    light_session = ReferenceEvaluator(tc.model)
    ort_session = onnxruntime.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])

    ds = tc.data_sets[0]
    feeds = {name: _to_numpy(t) for name, t in zip(input_names, ds.inputs, strict=True)}
    # The case's own rtol/atol are tuned for comparing against its shipped
    # reference output (computed by the very same accelerated kernel used
    # here); onnxruntime is an independent implementation whose reductions
    # (e.g. Gemm's dot products, Exp/Log's large-input rounding) accumulate in
    # a different order, so a looser, fixed tolerance is used instead.
    rtol, atol = 1e-2, 1e-3

    clear_used_kernel_names()
    light_out = light_session.run(None, feeds)
    assert expected_kernel in used_kernel_names(), used_kernel_names()
    ort_out = ort_session.run(None, feeds)
    for actual, expected in zip(light_out, ort_out, strict=True):
        if expected.dtype == np.bool_:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(
                actual.astype(np.float64),
                expected.astype(np.float64),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )

    # Aim for roughly a constant total element budget per case (~2e7 elements
    # processed across all repeats) so large cases are not re-run too many
    # times, but always repeat at least 3 times and never more than 30.
    repeat = max(3, min(30, 20_000_000 // _case_element_count(tc)))
    light_time = measure(lambda feeds=feeds, sess=light_session: sess.run(None, feeds), repeat)
    ort_time = measure(lambda feeds=feeds, sess=ort_session: sess.run(None, feeds), repeat)
    shapes = ",".join(
        "x".join(str(d) for d in array.shape) or "scalar" for array in feeds.values()
    )
    dtypes = ",".join(str(array.dtype) for array in feeds.values())
    rows.append((op_type, tc.name, shapes, dtypes, light_time, ort_time))

# Print an aligned table once every case has run, since column widths (name,
# input shapes, dtypes) are not known ahead of time.
op_width = max(len(op_type) for op_type, *_ in rows)
name_width = max(len(name) for _, name, *_ in rows)
shapes_width = max(len(shapes) for _, _, shapes, _, _, _ in rows)
dtypes_width = max(len(dtypes) for _, _, _, dtypes, _, _ in rows)
for op_type, name, shapes, dtypes, light_time, ort_time in rows:
    print(
        f"{op_type:>{op_width}} | {name:<{name_width}} | shapes={shapes:<{shapes_width}} | "
        f"dtype={dtypes:<{dtypes_width}} | "
        f"onnx-light-cpu={light_time * 1e6:10.2f} us | "
        f"onnxruntime={ort_time * 1e6:10.2f} us | speed-up={ort_time / light_time:6.2f}x"
    )

# %%
# Excel export
# ------------
#
# The full results table -- one row per benchmark case -- is also written to
# an ``.xlsx`` workbook so it can be inspected, filtered, or archived outside
# this script, alongside the printed table and the plot below.

results_frame = pd.DataFrame(
    [
        {
            "operator": op_type,
            "case": name,
            "input_shapes": shapes,
            "input_dtypes": dtypes,
            "onnx_light_cpu_us": light_time * 1e6,
            "onnxruntime_us": ort_time * 1e6,
            "speed_up": ort_time / light_time,
        }
        for op_type, name, shapes, dtypes, light_time, ort_time in rows
    ]
)
results_frame.to_excel("plot_backend_cases_benchmark.xlsx", index=False)

# %%
# Plot the speed-ups
# -------------------
#
# One bar per case, grouped and colored by operator, showing onnx-light-cpu's
# speed-up over ONNX Runtime (values above 1 mean onnx-light-cpu is faster).
# The x-axis is logarithmic so a speed-up and its reciprocal are equidistant
# from the ``1`` baseline.

_unique_op_types = sorted({op_type for op_type, *_ in rows})
_COLOR_MAP = plt.get_cmap("turbo", len(_unique_op_types))
_COLORS = {op_type: _COLOR_MAP(index) for index, op_type in enumerate(_unique_op_types)}


def _short_label(op_type, name):
    label = name.removeprefix("test_cpu_")
    label = label.removeprefix(op_type.lower() + "_")
    return label.removesuffix("_benchmark")


labels = [_short_label(op_type, name) for op_type, name, _, _, _, _ in rows]
speedups = np.array([ort_time / light_time for _, _, _, _, light_time, ort_time in rows])
colors = [_COLORS[op_type] for op_type, _, _, _, _, _ in rows]

fig, ax = plt.subplots(figsize=(8, max(5, 0.4 * len(rows))))
positions = np.arange(len(rows))
ax.barh(positions, speedups, color=colors)
ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":")
ax.set_xscale("log")
ax.set_yticks(positions, labels, fontsize=7)
ax.set_xlabel("speed-up vs onnxruntime")
ax.set_title("onnx-light-cpu speed-up over onnxruntime on backend cases")

handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in _COLORS.values()]
ax.legend(handles, _COLORS.keys(), title="operator", loc="upper left", fontsize=8, ncols=3)

fig.tight_layout()
fig.savefig("plot_backend_cases_benchmark.png")
plt.show()
