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
measured for the smaller sizes in ``size_grid`` (all but the last two).
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
    used_kernel_names,
)
from onnx_light_cpu.onnx_py._cpukernels import detect_simd_level, has_cpu_kernels

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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
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
# Each candidate is called ``repeat`` times and the best (minimum) wall-clock
# time is kept to reduce the impact of scheduling noise. The number of repeats
# shrinks as the matrices grow so the whole benchmark stays fast.


def measure(func, repeat):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


# %%
# Built-in onnx-light curve (measured *before* registering onnx-light-cpu)
# ---------------------------------------------------------------------------
#
# A dedicated model/session, run and timed here while onnx-light-cpu's
# accelerated ``Gemm`` kernel has not been installed yet, so it resolves to
# onnx-light's own built-in reference kernel. Skipped for the last two sizes
# in ``size_grid`` (256 and 512) since the built-in kernel's cost grows much
# faster than the other back-ends.

size_grid = [16, 32, 64, 128, 256, 512]
alone_sizes = size_grid[:-2]
rng = np.random.default_rng(0)

alone_model = make_gemm_model()
alone_session = ReferenceEvaluator(alone_model)
alone_times = {}
for size in alone_sizes:
    a = rng.standard_normal((size, size)).astype(np.float32)
    b = rng.standard_normal((size, size)).astype(np.float32)
    expected = a @ b
    repeat = max(3, min(100, 20_000_000 // (size * size * size)))

    def run_alone(a=a, b=b):
        return alone_session.run(None, {"A": a, "B": b})[0]

    alone_times[size] = measure(run_alone, repeat)
    np.testing.assert_allclose(run_alone(), expected, rtol=1e-2, atol=1e-2)
    print(f"size={size:>4}x{size:<4} | onnx-light (built-in)={alone_times[size] * 1e6:10.2f} us")

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


# Confirm the model dispatches to the onnx-light-cpu ``Gemm`` kernel (identified
# by the library-qualified name it records when it runs) rather than
# onnx-light's built-in kernel.
_probe = np.zeros((2, 2), dtype=np.float32)
clear_used_kernel_names()
run_light(_probe, _probe)
assert used_kernel_names() == ["onnx_light_cpu::Gemm"], used_kernel_names()


# %%
# Run the rest of the benchmark
# -------------------------------
#
# For every size the same square inputs are fed to numpy, onnx-light-cpu and
# onnxruntime. The results are checked against :func:`numpy.matmul` so every
# implementation agrees.

rng = np.random.default_rng(0)

rows = []
for size in size_grid:
    a = rng.standard_normal((size, size)).astype(np.float32)
    b = rng.standard_normal((size, size)).astype(np.float32)
    expected = a @ b

    repeat = max(3, min(100, 20_000_000 // (size * size * size)))

    numpy_time = measure(lambda a=a, b=b: a @ b, repeat)

    if light_session is not None:
        cpu_time = measure(lambda a=a, b=b: run_light(a, b), repeat)
        np.testing.assert_allclose(run_light(a, b), expected, rtol=1e-2, atol=1e-2)
    else:
        cpu_time = float("nan")

    ort_time = measure(lambda a=a, b=b: session.run(None, {"A": a, "B": b}), repeat)
    np.testing.assert_allclose(
        session.run(None, {"A": a, "B": b})[0], expected, rtol=1e-2, atol=1e-2
    )

    rows.append((size, numpy_time, cpu_time, ort_time))
    print(
        f"size={size:>4}x{size:<4} | numpy={numpy_time * 1e6:10.2f} us | "
        f"onnx-light-cpu={cpu_time * 1e6:10.2f} us | "
        f"onnxruntime={ort_time * 1e6:10.2f} us"
    )

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
# onnxruntime and values below ``1`` are slower. The built-in onnx-light curve
# only has points for the sizes it was measured on (see "Why the built-in
# curve stops early" above).

import matplotlib.pyplot as plt

fig, (ax_time, ax_speedup) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_time.plot(sizes, numpy_times * 1e6, "o--", label="numpy", color="#9b7ec8")
ax_time.plot(alone_grid, alone_grid_times * 1e6, "o--", label=alone_label, color="#5cb85c")
if light_session is not None:
    ax_time.plot(sizes, cpu_times * 1e6, "o-", label=light_label, color="#4a9eff")
ax_time.plot(sizes, ort_times * 1e6, "o-", label="onnxruntime", color="#f4a259")
ax_time.set_xscale("log")
ax_time.set_yscale("log")
ax_time.set_xlabel("matrix size N (N x N)")
ax_time.set_ylabel("time (microseconds)")
ax_time.set_title(f"Gemm execution time (SIMD: {simd_name})")
ax_time.tick_params(axis="x", labelrotation=45)
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
)
if light_session is not None:
    ax_speedup.plot(sizes, ort_times / cpu_times, "o-", label=light_label, color="#4a9eff")
ax_speedup.plot(sizes, ort_times / ort_times, "o-", label="onnxruntime", color="#f4a259")
ax_speedup.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
ax_speedup.set_xscale("log")
ax_speedup.set_yscale("log")
ax_speedup.set_xlabel("matrix size N (N x N)")
ax_speedup.set_ylabel("speed-up vs onnxruntime")
ax_speedup.set_title("Gemm speed-up (onnxruntime = 1)")
ax_speedup.tick_params(axis="x", labelrotation=45)
ax_speedup.legend()

fig.tight_layout()
fig.savefig("plot_gemm_benchmark.png")
plt.show()
