"""
Benchmark Abs: onnxruntime vs onnx-light + onnx-light-cpu
=========================================================

This example compares three ways of computing the elementwise absolute value
of a ``float32`` array across a range of input sizes:

* **onnxruntime** - running a single-node ``Abs`` ONNX model.
* **onnx-light + onnx-light-cpu** - the SIMD-accelerated ``Abs`` kernel that
  ``onnx-light`` dispatches to. ``onnx-light`` picks the fastest kernel
  registered for the operator, and ``onnx-light-cpu`` provides exactly this
  kernel (``abs``) with runtime AVX-512/AVX2/AVX/SSE2 dispatch.
* **numpy** - :func:`numpy.abs`, used as a reference baseline.

The three back-ends compute the same result; the goal here is to see how their
timings evolve as the array grows from a few hundred to a hundred million
elements.
"""

# %%
# Setup
# -----
#
# Import the compiled ``onnx-light-cpu`` kernel and report which SIMD level the
# current CPU provides. The mapping is ``0=None``, ``1=SSE2``, ``2=AVX``,
# ``3=AVX2`` and ``4=AVX512``.

import time

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper

from onnx_light_cpu.onnx_py._cpukernels import (
    abs,
    detect_simd_level,
    has_cpu_kernels,
)

_SIMD_NAMES = {0: "scalar", 1: "SSE2", 2: "AVX", 3: "AVX2", 4: "AVX-512"}

assert has_cpu_kernels()
level = detect_simd_level()
simd_name = _SIMD_NAMES.get(level, level)
print(f"CPU kernels available, SIMD level: {level} ({simd_name})")

# %%
# Build the ONNX model for onnxruntime
# ------------------------------------
#
# A single ``Abs`` node operating on a 1-D ``float32`` tensor of dynamic length
# is enough to benchmark the runtime.

graph = helper.make_graph(
    [helper.make_node("Abs", ["X"], ["Y"])],
    "abs_bench",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N"])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
onnx.checker.check_model(model)

session = onnxruntime.InferenceSession(
    model.SerializeToString(), providers=["CPUExecutionProvider"]
)

# %%
# Timing helper
# -------------
#
# Each candidate is called ``repeat`` times and the best (minimum) wall-clock
# time is kept to reduce the impact of scheduling noise. The number of repeats
# shrinks as the arrays grow so the whole benchmark stays fast.


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
# For every size the same input is fed to the three back-ends. The results are
# checked against :func:`numpy.abs` to make sure every implementation agrees.

sizes = [10**k for k in range(2, 9)]
rng = np.random.default_rng(0)

rows = []
for size in sizes:
    inp = rng.uniform(-100.0, 100.0, size=size).astype(np.float32)
    expected = np.abs(inp)

    repeat = max(3, min(200, 2_000_000 // size))

    numpy_time = measure(lambda inp=inp: np.abs(inp), repeat)

    cpu_time = measure(lambda inp=inp: abs(inp), repeat)
    assert np.array_equal(abs(inp), expected), size

    ort_time = measure(lambda inp=inp: session.run(None, {"X": inp}), repeat)
    assert np.array_equal(session.run(None, {"X": inp})[0], expected), size

    rows.append((size, numpy_time, cpu_time, ort_time))
    print(
        f"size={size:>9} | numpy={numpy_time * 1e6:10.2f} us | "
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
# The left panel shows the raw execution time versus the array size on a
# log-log scale. The right panel shows the throughput (elements processed per
# second), which highlights the fixed per-call overhead at small sizes.

import matplotlib.pyplot as plt

fig, (ax_time, ax_tput) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_time.plot(sizes, numpy_times * 1e6, "o--", label="numpy", color="#9b7ec8")
ax_time.plot(
    sizes,
    cpu_times * 1e6,
    "o-",
    label="onnx-light + onnx-light-cpu",
    color="#4a9eff",
)
ax_time.plot(sizes, ort_times * 1e6, "o-", label="onnxruntime", color="#f4a259")
ax_time.set_xscale("log")
ax_time.set_yscale("log")
ax_time.set_xlabel("array size (elements)")
ax_time.set_ylabel("time (microseconds)")
ax_time.set_title(f"Abs execution time (SIMD: {simd_name})")
ax_time.legend()

ax_tput.plot(sizes, sizes / numpy_times, "o--", label="numpy", color="#9b7ec8")
ax_tput.plot(
    sizes,
    sizes / cpu_times,
    "o-",
    label="onnx-light + onnx-light-cpu",
    color="#4a9eff",
)
ax_tput.plot(sizes, sizes / ort_times, "o-", label="onnxruntime", color="#f4a259")
ax_tput.set_xscale("log")
ax_tput.set_yscale("log")
ax_tput.set_xlabel("array size (elements)")
ax_tput.set_ylabel("throughput (elements / second)")
ax_tput.set_title("Abs throughput")
ax_tput.legend()

fig.tight_layout()
plt.show()
