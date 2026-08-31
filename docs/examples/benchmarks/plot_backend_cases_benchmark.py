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
# is of interest and ``--max-cases`` controls how many matching cases run.
# ``--threads`` applies the same explicit thread count to both runtimes.
# onnx-light uses unpinned workers, matching ONNX Runtime when its thread count
# is explicit, so the comparison does not mix different affinity policies.

import argparse
import gc
import os
import re
import time
from dataclasses import dataclass

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
parser.add_argument(
    "--max-case",
    "--max-cases",
    dest="max_cases",
    type=int,
    default=20,
    help="maximum number of matching cases to run (default: 20; 0 disables the limit)",
)
parser.add_argument(
    "--threads",
    type=int,
    default=min(4, os.cpu_count() or 1),
    help="threads used by both runtimes (default: min(4, available CPUs))",
)
parser.add_argument(
    "-r",
    "--repeat",
    type=int,
    default=10 * (os.cpu_count() or 1),
    help="maximum measured calls per runtime and case (default: 10 per CPU)",
)
parser.add_argument(
    "-w",
    "--warmup",
    type=int,
    default=2 * (os.cpu_count() or 1),
    help="untimed warm-up calls per runtime and case (default: 2 per CPU)",
)
parser.add_argument(
    "-t",
    "--max-repeat-time",
    type=float,
    default=1.0,
    help="maximum cumulative measurement time per runtime and case in seconds (default: 1)",
)
args, unknown_args = parser.parse_known_args()
if args.max_cases < 0:
    parser.error("--max-cases must be greater than or equal to 0")
if args.threads <= 0:
    parser.error("--threads must be greater than 0")
if args.repeat <= 0:
    parser.error("--repeat must be greater than 0")
if args.warmup < 0:
    parser.error("--warmup must be greater than or equal to 0")
if args.max_repeat_time <= 0:
    parser.error("--max-repeat-time must be greater than 0")
for unknown_arg in unknown_args:
    if unknown_arg.startswith(("--repeat", "--warm", "--max-repeat-time")):
        parser.error(f"unrecognized timing option: {unknown_arg}")
_name_filter = re.compile(args.filter) if args.filter else None


def _to_numpy(tensor):
    """Decodes a backend test case ``Tensor`` into a numpy array."""
    dtype = tensor_dtype_to_np_dtype(int(tensor.data_type))
    shape = tuple(int(d) for d in tensor.shape)
    return np.frombuffer(tensor.raw_data(), dtype=dtype).reshape(shape)


_CASE_GROUP_SUFFIXES = (
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
    "benchmark",
)


def _case_group_key(name):
    """Derives a cheap, name-only grouping key used to spread ``--max-cases``
    truncation across operators.

    This must not touch ``tc.model``: building the ONNX model (and the
    tensors it references) for every collected case -- rather than only for
    the small subset that ``--max-cases`` keeps -- can exhaust memory on CI
    runners once the backend test corpus grows. The real ``op_type`` for
    each *selected* case is still derived from its model later, once, when
    it is actually measured.
    """
    stem = name
    if stem.startswith("test_cpu_"):
        stem = stem[len("test_cpu_") :]
    tokens = stem.split("_")
    while tokens and (
        tokens[-1] in _CASE_GROUP_SUFFIXES or re.fullmatch(r"[a-z]?\d+", tokens[-1])
    ):
        tokens.pop()
    return "_".join(tokens) or stem


def _collect_cases():
    """Registers and collects every "test_cpu_*_benchmark" backend test case.

    Every operator and element type onnx-light-cpu ships a BENCHMARK variant
    for is included; ``--filter`` and ``--max-cases`` are the only further
    restrictions applied here. When ``--max-cases`` truncates the collected
    cases, the truncation round-robins across operators (instead of taking a
    contiguous prefix of whatever order they were collected in) so a small
    ``--max-cases`` still yields a representative, multi-operator sample
    rather than being dominated by whichever operator happens to sort first
    or have the most benchmark variants -- e.g. Attention, which ONNX Runtime
    rejects for several onnx-light-cpu benchmark cases (see below), would
    otherwise fill the whole default selection and leave nothing comparable
    to plot.
    """
    register_backend_test_cases()
    max_cases = (
        10 if os.environ.get("UNITTEST_GOING", "0") in ("1", "true", "True") else args.max_cases
    )
    cases_by_group = {}
    for tc in collect_test_cases_by_name(
        "^test_cpu_.*_benchmark$",
        mode=TestMode.BENCHMARK,
        generate_benchmark_expected_outputs=False,
    ):
        if _name_filter is not None and not _name_filter.search(tc.name):
            continue
        cases_by_group.setdefault(_case_group_key(tc.name), []).append(tc)
    if not max_cases:
        return [tc for group in cases_by_group.values() for tc in group]
    cases = []
    while len(cases) < max_cases and any(cases_by_group.values()):
        for group_key in list(cases_by_group):
            group = cases_by_group[group_key]
            if not group:
                del cases_by_group[group_key]
                continue
            cases.append(group.pop(0))
            if len(cases) >= max_cases:
                break
    return cases


print("-- _collect_cases")
_CASES = _collect_cases()
_no_cases_message = (
    f"no onnx-light-cpu BENCHMARK backend test cases were collected (filter={args.filter!r})"
)
assert _CASES, _no_cases_message
print(f"-- collected {len(_CASES)} cases")

# %%
# Timing helper
# -------------
#
# Each candidate gets up to ``--warmup`` untimed calls, then up to ``--repeat``
# measured calls. Both phases stop once ``--max-repeat-time`` cumulative
# seconds have elapsed, and the median wall-clock time is retained.


def measure(func, repeat, warmup, max_duration):
    warmup_duration = 0.0
    for _ in range(warmup):
        start = time.perf_counter()
        func()
        warmup_duration += time.perf_counter() - start
        if warmup_duration >= max_duration:
            break
    timings = []
    total_duration = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        duration = time.perf_counter() - start
        timings.append(duration)
        total_duration += duration
        if max_duration is not None and total_duration >= max_duration:
            break
    return float(np.median(timings))


# %%
# Run every case through onnx-light-cpu and ONNX Runtime
# -------------------------------------------------------
#
# ``register_kernels()`` installs the accelerated kernels in onnx-light's
# process-wide dispatch table before any of the onnx-light-cpu sessions below
# run for the first time.

print("-- register_kernels")
register_kernels()

# Some onnx-light-cpu Attention benchmark cases (e.g. streaming past_key/
# past_value without materializing present_key/present_value, or FLOAT16/
# BFLOAT16 inputs) are rejected by ONNX Runtime -- either at session-creation
# time or while running -- even though onnx-light-cpu's kernel handles them
# fine. Rather than special-case those by name, any ONNX Runtime failure is
# caught here and the case is still timed/reported for onnx-light-cpu alone,
# with "n/a" standing in for the ONNX Runtime side.
#
# All onnx-light-cpu cases are measured before any ONNX Runtime session is
# created. Both runtimes therefore keep their default spinning behavior
# without leaving one runtime's live pool to perturb the other's measurements.
print("-- benchmark onnx-light-cpu")
print(
    f"-- timing warmup={args.warmup} repeat={args.repeat} "
    f"max_repeat_time={args.max_repeat_time:g}s"
)
measurements = []
_progress = tqdm(_CASES, desc="benchmarking backend cases", unit="case")
for tc in _progress:
    _progress.set_postfix_str(tc.name)
    op_type = tc.model.graph.node[0].op_type
    expected_kernel = f"onnx_light_cpu::{op_type}"
    if tc.model.graph.initializer:
        raise AssertionError(
            f"{tc.name} contains an initializer; backend benchmarks must time runtime inputs"
        )
    input_names = [vi.name for vi in tc.model.graph.input]

    ds = tc.data_sets[0]
    feeds = {name: _to_numpy(t) for name, t in zip(input_names, ds.inputs, strict=True)}
    light_session = ReferenceEvaluator(
        tc.model, cpu_execution={"num_threads": args.threads, "affinity_policy": "none"}
    )
    clear_used_kernel_names()
    light_out = [np.array(output, copy=True) for output in light_session.run(None, feeds)]
    assert expected_kernel in used_kernel_names(), used_kernel_names()
    light_time = measure(
        lambda feeds=feeds, sess=light_session: sess.run(None, feeds),
        args.repeat,
        warmup=args.warmup,
        max_duration=args.max_repeat_time,
    )
    shapes = ",".join(
        "x".join(str(d) for d in array.shape) or "scalar" for array in feeds.values()
    )
    dtypes = ",".join(str(array.dtype) for array in feeds.values())
    measurements.append(
        {
            "op_type": op_type,
            "name": tc.name,
            "model_bytes": tc.model.SerializeToString(),
            "feeds": feeds,
            "light_out": light_out,
            "light_time": light_time,
            "node_count": len(tc.model.graph.node),
            "shapes": shapes,
            "dtypes": dtypes,
        }
    )

del light_session
gc.collect()

print("-- benchmark ONNX Runtime")
rows = []
_progress = tqdm(measurements, desc="benchmarking backend cases", unit="case")
for measurement in _progress:
    _progress.set_postfix_str(measurement["name"])
    ort_error = None
    ort_time = None
    try:
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = args.threads
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        ort_session = onnxruntime.InferenceSession(
            measurement["model_bytes"],
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        ort_out = ort_session.run(None, measurement["feeds"])
    except Exception as exc:  # noqa: BLE001 -- unsupported cases are reported as "n/a".
        ort_error = str(exc).splitlines()[0][:40]
    else:
        # The case's own tolerance compares against output generated by the
        # accelerated kernel. ONNX Runtime may accumulate reductions in a
        # different order, so the cross-runtime check uses its own tolerance.
        # The rounding noise of a reduction is set by the magnitude of the
        # terms being accumulated, not by the magnitude of the result: an
        # element that cancels down to nearly zero still carries the noise of
        # the whole sum. ``rtol`` cannot see that, and a constant ``atol``
        # would have to be sized for the largest case, so the floor is derived
        # from the tensor's own scale and the length of the reductions. The
        # largest input dimension is a deliberate upper bound on any
        # contraction length in the model, since the exact one is per-operator;
        # over-estimating only widens the floor, which stays orders of
        # magnitude below ``rtol`` for every case registered here.
        reduction = max(
            (max(array.shape) for array in measurement["feeds"].values() if array.ndim > 0),
            default=1,
        )
        for actual, expected in zip(measurement["light_out"], ort_out, strict=True):
            if expected.dtype == np.bool_:
                np.testing.assert_array_equal(actual, expected)
            elif expected.dtype == np.float32 or expected.dtype == np.float64:
                scale = float(np.abs(expected).max(initial=0.0))
                noise = scale * float(np.finfo(expected.dtype).eps) * reduction**0.5
                np.testing.assert_allclose(
                    actual.astype(np.float64),
                    expected.astype(np.float64),
                    rtol=1e-2,
                    atol=max(1e-3, (measurement["node_count"] - 1) * 5e-2, noise),
                    equal_nan=True,
                )
        ort_time = measure(
            lambda feeds=measurement["feeds"], sess=ort_session: sess.run(None, feeds),
            args.repeat,
            warmup=args.warmup,
            max_duration=args.max_repeat_time,
        )
    rows.append(
        (
            measurement["op_type"],
            measurement["name"],
            measurement["shapes"],
            measurement["dtypes"],
            measurement["light_time"],
            ort_time,
            ort_error,
        )
    )

# Print an aligned table once every case has run, since column widths (name,
# input shapes, dtypes) are not known ahead of time. Cases ONNX Runtime could
# not run (ort_time is None) show its error message instead of a timing/
# speed-up.
op_width = max(len(op_type) for op_type, *_ in rows)
name_width = max(len(name) for _, name, *_ in rows)
shapes_width = max(len(shapes) for _, _, shapes, _, _, _, _ in rows)
dtypes_width = max(len(dtypes) for _, _, _, dtypes, _, _, _ in rows)
for op_type, name, shapes, dtypes, light_time, ort_time, ort_error in rows:
    ort_str = f"{ort_time * 1e6:10.2f} us" if ort_time is not None else f"{'error':>13}"
    speedup_str = f"{ort_time / light_time:6.2f}x" if ort_time is not None else f"{'n/a':>7}"
    print(
        f"{op_type:>{op_width}} | {name:<{name_width}} | shapes={shapes:<{shapes_width}} | "
        f"dtype={dtypes:<{dtypes_width}} | "
        f"onnx-light-cpu={light_time * 1e6:10.2f} us | "
        f"onnxruntime={ort_str} | speed-up={speedup_str}"
        + (f" | onnxruntime_error={ort_error}" if ort_error is not None else "")
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
            "onnxruntime_us": ort_time * 1e6 if ort_time is not None else None,
            "speed_up": ort_time / light_time if ort_time is not None else None,
            "onnxruntime_error": ort_error,
        }
        for op_type, name, shapes, dtypes, light_time, ort_time, ort_error in rows
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
# from the ``1`` baseline. Cases ONNX Runtime failed to run (ort_time is
# None, see the try/except above) are left out of the plot -- they have no
# speed-up to show -- but not out of the table/xlsx output above, where their
# error is still reported. If every case was rejected or failed, there is
# nothing to plot: :func:`prepare_plot_data` raises rather than silently
# rendering an empty chart, since that would look identical to a build that
# ran fine.


class NoPlottableCasesError(RuntimeError):
    """Raised when no collected row has a comparable ONNX Runtime timing.

    A chart with zero bars is not a useful (nor a correct) rendering of "no
    comparable case was found": it looks identical to "the build silently
    produced an empty page". Raising here instead turns that situation into a
    build failure with the offending case names/errors attached.
    """


@dataclass
class PlotData:
    """Everything the chart needs, derived once from the plottable rows."""

    plotted_rows: list
    labels: list
    speedups: np.ndarray
    colors: list
    colors_by_op_type: dict


def _short_label(name):
    """Strips the common ``test_cpu_*_benchmark`` case-name affixes."""
    label = name.removeprefix("test_cpu_")
    return label.removesuffix("_benchmark")


def prepare_plot_data(rows):
    """Turns collected benchmark ``rows`` into the values the chart plots.

    Each row is ``(op_type, name, shapes, dtypes, light_time, ort_time,
    ort_error)`` -- the same tuples appended to ``rows`` above. Rows whose
    ``ort_time`` (index 5) is ``None`` are excluded from the returned plot
    data. Raises :class:`NoPlottableCasesError` -- naming every rejected case
    and its error -- when that leaves nothing to plot, rather than letting
    the caller render an empty chart.
    """
    plotted_rows = [row for row in rows if row[5] is not None]
    if not plotted_rows:
        if rows:
            details = "; ".join(f"{row[1]} ({row[6] or 'no error message'})" for row in rows)
        else:
            details = "no benchmark case was collected"
        raise NoPlottableCasesError(
            "no benchmark case produced a comparable ONNX Runtime timing; every "
            f"collected case was rejected or failed: {details}"
        )
    unique_op_types = sorted({row[0] for row in plotted_rows})
    color_map = plt.get_cmap("turbo", len(unique_op_types))
    colors_by_op_type = {
        op_type: color_map(index) for index, op_type in enumerate(unique_op_types)
    }
    labels = [_short_label(row[1]) for row in plotted_rows]
    speedups = np.array([row[5] / row[4] for row in plotted_rows])
    colors = [colors_by_op_type[row[0]] for row in plotted_rows]
    return PlotData(
        plotted_rows=plotted_rows,
        labels=labels,
        speedups=speedups,
        colors=colors,
        colors_by_op_type=colors_by_op_type,
    )


_plot_data = prepare_plot_data(rows)

fig, ax = plt.subplots(figsize=(8, max(5, 0.4 * len(_plot_data.plotted_rows))))
positions = np.arange(len(_plot_data.plotted_rows))
ax.barh(positions, _plot_data.speedups, color=_plot_data.colors)
ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":")
ax.set_xscale("log")
ax.set_yticks(positions, _plot_data.labels, fontsize=7)
for tick_label, speedup in zip(ax.get_yticklabels(), _plot_data.speedups, strict=True):
    if speedup <= 0.5:
        tick_label.set_color("red")
    elif speedup < 0.95:
        tick_label.set_color("orange")
ax.set_xlabel("speed-up vs onnxruntime")
ax.set_title("onnx-light-cpu speed-up over onnxruntime on backend cases")

handles = [
    plt.Rectangle((0, 0), 1, 1, color=color) for color in _plot_data.colors_by_op_type.values()
]
ax.legend(
    handles,
    _plot_data.colors_by_op_type.keys(),
    title="operator",
    loc="upper left",
    fontsize=8,
    ncols=3,
)

fig.tight_layout()
fig.savefig("plot_backend_cases_benchmark.png")
plt.show()
