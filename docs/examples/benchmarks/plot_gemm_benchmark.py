"""
Benchmark Gemm: numpy vs onnxruntime vs onnx-light vs onnx-light-cpu
=====================================================================

This example compares four ways of computing a ``float32`` general matrix
multiplication ``Y = A @ B`` for square matrices of increasing size:

* **numpy** - :func:`numpy.matmul` (``A @ B``), which dispatches to the platform
  BLAS and is used here as the reference baseline.
* **onnxruntime** - running a single-node ``Gemm`` ONNX model.
* **onnx-light (built-in)** - the same single-node model run through
  ``onnx-light``'s ``ReferenceEvaluator`` *without* registering
  ``onnx-light-cpu``, i.e. onnx-light's own (non-SIMD) reference ``Gemm``
  kernel. Only measured for the smaller sizes in the grid -- see "Why the
  built-in curve stops early" below.
* **onnx-light + onnx-light-cpu** - the SIMD-accelerated ``Gemm`` kernel that
  ``onnx-light`` dispatches to after :func:`onnx_light_cpu.register_kernels`
  installs the optimized kernel implementation.

The back-ends compute the same result; the goal is to see how their timings
evolve as the matrices grow.

Why the built-in curve stops early
-----------------------------------

``onnx-light``'s dispatch table is a single, process-wide table:
:func:`onnx_light_cpu.register_kernels` permanently replaces the default
domain's ``Gemm`` entry, and a given ``RuntimeSession`` resolves (and caches)
its kernels on its *first* run. So the only way to observe the un-accelerated,
built-in kernel *and* the accelerated one in the same process is to build a
separate model/session for the built-in curve and run it *before*
:func:`onnx_light_cpu.register_kernels` is ever called -- which is what this
example does. Since the built-in kernel is a plain reference implementation
with no SIMD or blocking/packing, its cost grows much faster than the other
three back-ends; to keep the benchmark's runtime reasonable it is only
measured up to 512 while the accelerated back-ends continue to 4096.
"""

# %%
# Setup
# -----
#
# Report which SIMD level the current CPU provides. The mapping is ``0=None``,
# ``1=SSE2``, ``2=AVX``, ``3=AVX2`` and ``4=AVX512``.

import time

import numpy as np
import onnxruntime

# ``onnx-light`` ships ``onnx_light.onnx`` as a drop-in replacement for the
# ``onnx`` package; use it to build the model so the example depends on
# onnx-light rather than onnx.
from onnx_light.onnx import TensorProto, checker, helper
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import (
    clear_used_kernel_names,
    register_kernels,
    registered_kernel_names,
    used_kernel_names,
)
from onnx_light_cpu.onnx_py._cpukernels import detect_simd_level, has_cpu_kernels
from onnx_light_cpu.onnx_py._cpuregister import set_kernel_usage_recording

_SIMD_NAMES = {0: "scalar", 1: "SSE2", 2: "AVX", 3: "AVX2", 4: "AVX-512"}

assert has_cpu_kernels()
level = detect_simd_level()
simd_name = _SIMD_NAMES.get(level, level)
print(f"CPU kernels available, SIMD level: {level} ({simd_name})")

# %%
# Build the shared ONNX model
# ---------------------------
#
# A single ``Gemm`` node multiplying two 2-D ``float32`` tensors of dynamic
# shape is enough to benchmark both runtimes. The bias input ``C`` is omitted so
# the node computes ``A @ B``. ``make_gemm_model`` is called twice: once for
# the built-in (pre-registration) curve and once for onnxruntime / the
# accelerated onnx-light-cpu curve, so each gets its own model/graph object
# (see "Why the built-in curve stops early" above for why that matters).


def make_gemm_model():
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["A", "B"], ["Y"], alpha=1.0, beta=1.0)],
        "gemm_bench",
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "K"]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, ["K", "N"]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["M", "N"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=13)
    checker.check_model(model)
    return model


model = make_gemm_model()
session = onnxruntime.InferenceSession(
    model.SerializeToString(), providers=["CPUExecutionProvider"]
)

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
#
# Each candidate gets three untimed warm-up calls, then is called ``repeat``
# times and the median wall-clock time is retained. The number of repeats
# shrinks as the matrices grow but never below seven.


def measure(func, repeat, warmup=3):
    for _ in range(warmup):
        func()
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def measure_together(*funcs, repeat, warmup=3):
    timings = tuple([] for _ in funcs)
    for iteration in range(warmup):
        for index in range(len(funcs)):
            funcs[(iteration + index) % len(funcs)]()
    for iteration in range(repeat):
        for offset in range(len(funcs)):
            index = (iteration + offset) % len(funcs)
            start = time.perf_counter()
            funcs[index]()
            timings[index].append(time.perf_counter() - start)
    return tuple(float(np.median(values)) for values in timings)


# %%
# Prime the built-in onnx-light kernel before registering onnx-light-cpu
# ----------------------------------------------------------------------
#
# A dedicated model/session is run once while onnx-light-cpu's accelerated
# ``Gemm`` kernel has not been installed yet, so it resolves and caches
# onnx-light's own built-in reference kernel. It is timed below alongside the
# other runtimes up to size 512.

size_grid = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
alone_sizes = [16, 32, 64, 128, 256, 512]
rng = np.random.default_rng(0)

alone_model = make_gemm_model()
alone_session = ReferenceEvaluator(alone_model)
alone_session.run(
    None,
    {
        "A": np.zeros((1, 1), dtype=np.float32),
        "B": np.zeros((1, 1), dtype=np.float32),
    },
)
alone_times = {}

light_label = "onnx-light + onnx-light-cpu"
alone_label = "onnx-light (built-in)"
# ``register_kernels()`` needs the ``_cpuregister`` extension, which is only
# built with ``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``. When it is missing (as in
# the documentation build) the onnx-light-cpu curve is simply omitted; the
# import above stays unconditional.
register_kernels()
light_session = ReferenceEvaluator(model)


def run_light(a, b):
    return light_session.run(None, {"A": a, "B": b})[0]


accelerated_kernel_name = registered_kernel_names()["Gemm"]

# onnx-light-cpu kernels record their name on every run (a mutex per call);
# only the accelerated curve would pay that cost, so disable recording to keep
# the timings fair. Recording is briefly re-enabled below to verify the exact
# implementation used for every benchmark size.
set_kernel_usage_recording(False)


# %%
# Run the rest of the benchmark
# -------------------------------
#
# For every size the same square inputs are fed to numpy, onnx-light-cpu and
# onnxruntime. Runtimes whose timings form speed-up ratios are measured together,
# rotating their order to reduce cache and scheduling bias. The results are
# checked against :func:`numpy.matmul` so every implementation agrees.

rng = np.random.default_rng(0)

rows = []
for size in size_grid:
    a = rng.standard_normal((size, size)).astype(np.float32)
    b = rng.standard_normal((size, size)).astype(np.float32)
    expected = a @ b

    set_kernel_usage_recording(True)
    clear_used_kernel_names()
    run_light(a, b)
    accelerated_kernel_names = used_kernel_names()
    assert accelerated_kernel_name in accelerated_kernel_names, accelerated_kernel_names
    set_kernel_usage_recording(False)

    repeat = max(7, min(100, 20_000_000 // (size * size * size)))

    numpy_time = measure(lambda a=a, b=b: a @ b, repeat)

    if size in alone_sizes:
        alone_time, cpu_time, ort_time = measure_together(
            lambda a=a, b=b: alone_session.run(None, {"A": a, "B": b}),
            lambda a=a, b=b: run_light(a, b),
            lambda a=a, b=b: session.run(None, {"A": a, "B": b}),
            repeat=repeat,
        )
        alone_times[size] = alone_time
        np.testing.assert_allclose(
            alone_session.run(None, {"A": a, "B": b})[0],
            expected,
            rtol=1e-2,
            atol=1e-2,
        )
    else:
        cpu_time, ort_time = measure_together(
            lambda a=a, b=b: run_light(a, b),
            lambda a=a, b=b: session.run(None, {"A": a, "B": b}),
            repeat=repeat,
        )

    np.testing.assert_allclose(run_light(a, b), expected, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(
        session.run(None, {"A": a, "B": b})[0], expected, rtol=1e-2, atol=1e-2
    )

    rows.append((size, numpy_time, cpu_time, ort_time))
    alone_text = f"{alone_times[size] * 1e6:10.2f} us" if size in alone_times else "not measured"
    print(
        f"size={size:>4}x{size:<4} | numpy={numpy_time * 1e6:10.2f} us | "
        f"onnx-light={alone_text} | "
        f"onnx-light-cpu={cpu_time * 1e6:10.2f} us | "
        f"onnxruntime={ort_time * 1e6:10.2f} us"
    )

print(
    "verified onnx-light-cpu Gemm for every benchmark size: "
    f"accelerated={accelerated_kernel_name}"
)
set_kernel_usage_recording(True)

sizes = np.array([r[0] for r in rows])
numpy_times = np.array([r[1] for r in rows])
cpu_times = np.array([r[2] for r in rows])
ort_times = np.array([r[3] for r in rows])
alone_grid = np.array(alone_sizes)
alone_grid_times = np.array([alone_times[size] for size in alone_sizes])

# %%
# Plot the timings
# ----------------
#
# The left panel shows the raw execution time versus the matrix size on a
# log-log scale. The right panel shows the speed-up relative to
# **onnxruntime** (the baseline): for each back-end the onnxruntime time is
# divided by the back-end time, so values above ``1`` are faster than
# onnxruntime and values below ``1`` are slower. The speed-up is drawn on a
# logarithmic y-axis so that a given ratio and its reciprocal are equidistant
# from the ``1`` baseline. The built-in onnx-light curve
# only has points for the sizes it was measured on (see "Why the built-in
# curve stops early" above).

import matplotlib.pyplot as plt

fig, (ax_time, ax_speedup) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_time.plot(sizes, numpy_times * 1e6, "o--", label="numpy", color="#9b7ec8")
ax_time.plot(
    alone_grid,
    alone_grid_times * 1e6,
    "o--",
    label=alone_label,
    color="#5cb85c",
    linewidth=3.5,
    markersize=9,
    markerfacecolor="none",
    markeredgewidth=2,
    zorder=2,
)
if light_session is not None:
    ax_time.plot(
        sizes,
        cpu_times * 1e6,
        "o-",
        label=light_label,
        color="#4a9eff",
        linewidth=1.5,
        markersize=4,
        zorder=3,
    )
ax_time.plot(sizes, ort_times * 1e6, "o-", label="onnxruntime", color="#f4a259")
ax_time.set_xscale("log")
ax_time.set_yscale("log")
ax_time.set_xlabel("matrix size N (N x N)")
ax_time.set_ylabel("time (microseconds)")
ax_time.set_title(f"Gemm execution time (SIMD: {simd_name})")
ax_time.tick_params(axis="x", labelrotation=20)
ax_time.legend()

ort_times_by_size = dict(zip(sizes.tolist(), ort_times.tolist(), strict=True))
alone_ort_times = np.array([ort_times_by_size[size] for size in alone_sizes])

ax_speedup.plot(sizes, ort_times / numpy_times, "o--", label="numpy", color="#9b7ec8")
ax_speedup.plot(
    alone_grid,
    alone_ort_times / alone_grid_times,
    "o--",
    label=alone_label,
    color="#5cb85c",
    linewidth=3.5,
    markersize=9,
    markerfacecolor="none",
    markeredgewidth=2,
    zorder=2,
)
if light_session is not None:
    cpu_speedup = ort_times / cpu_times
    ax_speedup.plot(
        sizes,
        cpu_speedup,
        "o-",
        label=light_label,
        color="#4a9eff",
        linewidth=1.5,
        markersize=4,
        zorder=3,
    )
    for size, speedup in zip(sizes, cpu_speedup, strict=True):
        ax_speedup.annotate(
            f"{speedup:.2f}x",
            (size, speedup),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#4a9eff",
        )
ax_speedup.plot(sizes, ort_times / ort_times, "o-", label="onnxruntime", color="#f4a259")
ax_speedup.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
ax_speedup.set_xscale("log")
ax_speedup.set_yscale("log")
ax_speedup.set_xlabel("matrix size N (N x N)")
ax_speedup.set_ylabel("speed-up vs onnxruntime")
ax_speedup.set_title("Gemm speed-up (onnxruntime = 1)")
ax_speedup.tick_params(axis="x", labelrotation=20)
ax_speedup.legend()

fig.tight_layout()
fig.savefig("plot_gemm_benchmark.png")
plt.show()
