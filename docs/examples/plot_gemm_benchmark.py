"""
Benchmark Gemm: numpy vs onnxruntime vs onnx-light-cpu
======================================================

This example compares up to three ways of computing a ``float32`` general matrix
multiplication ``Y = A @ B`` for square matrices of increasing size:

* **numpy** - :func:`numpy.matmul` (``A @ B``), which dispatches to the platform
  BLAS and is used here as the reference baseline.
* **onnxruntime** - running a single-node ``Gemm`` ONNX model.
* **onnx-light + onnx-light-cpu** - the SIMD-accelerated ``Gemm`` kernel that
  ``onnx-light`` dispatches to after :func:`onnx_light_cpu.register_kernels`
  installs the optimized kernel implementation.

The back-ends compute the same result; the goal is to see how their timings
evolve as the matrices grow.
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

from onnx_light_cpu import register_kernels
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
# the node computes ``A @ B``.

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

session = onnxruntime.InferenceSession(
    model.SerializeToString(), providers=["CPUExecutionProvider"]
)

light_label = "onnx-light + onnx-light-cpu"
# ``register_kernels()`` needs the ``_cpuregister`` extension, which is only
# built with ``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``. When it is missing (as in
# the documentation build) the onnx-light-cpu curve is simply omitted; the
# import above stays unconditional.
try:
    register_kernels()
    light_session = ReferenceEvaluator(model)
except ImportError:
    light_session = None


def run_light(a, b):
    return light_session.run(None, {"A": a, "B": b})[0]


# %%
# Timing helper
# -------------
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
# Run the benchmark
# -----------------
#
# For every size the same square inputs are fed to the three back-ends. The
# results are checked against :func:`numpy.matmul` so every implementation
# agrees.

sizes = [16, 32, 64, 128, 256, 512]
rng = np.random.default_rng(0)

rows = []
for size in sizes:
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

# %%
# Plot the timings
# ----------------
#
# The left panel shows the raw execution time versus the matrix size on a
# log-log scale. The right panel shows the speed-up relative to
# **onnxruntime** (the baseline): for each back-end the onnxruntime time is
# divided by the back-end time, so values above ``1`` are faster than
# onnxruntime and values below ``1`` are slower.

import matplotlib.pyplot as plt

fig, (ax_time, ax_speedup) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_time.plot(sizes, numpy_times * 1e6, "o--", label="numpy", color="#9b7ec8")
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

ax_speedup.plot(sizes, ort_times / numpy_times, "o--", label="numpy", color="#9b7ec8")
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
plt.show()
