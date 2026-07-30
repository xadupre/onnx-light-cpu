"""
Elementwise Abs with runtime SIMD dispatch
==========================================

This example exercises the SIMD-accelerated ``Abs`` kernels provided by
``onnx-light-cpu`` from Python. The extension detects the best available
instruction set (AVX-512, AVX2, AVX, SSE2, or a scalar fallback) once at
runtime and dispatches every call to that implementation.

Each kernel computes the elementwise absolute value of a contiguous 1-D array.
The example runs every supported data type (``float32``, ``float64``,
``int32`` and ``int64``), checks the result against :func:`numpy.abs`, and then
plots the input/output of the ``float32`` kernel to illustrate the operation.
"""

# %%
# Setup
# -----
#
# Import the compiled extension and report which SIMD level the current CPU
# provides. The mapping is ``0=None``, ``1=SSE2``, ``2=AVX``, ``3=AVX2`` and
# ``4=AVX512``.

import numpy as np

from onnx_light_cpu.onnx_py._cpukernels import (
    abs_float32,
    abs_float64,
    abs_int32,
    abs_int64,
    detect_simd_level,
    has_cpu_kernels,
)

_SIMD_NAMES = {0: "scalar", 1: "SSE2", 2: "AVX", 3: "AVX2", 4: "AVX-512"}

assert has_cpu_kernels()
level = detect_simd_level()
print(f"CPU kernels available, SIMD level: {level} ({_SIMD_NAMES.get(level, level)})")

# %%
# Run every supported data type
# -----------------------------
#
# Each entry pairs a kernel with the NumPy dtype it operates on. The kernels
# write into a pre-allocated output array of the same length as the input.

cases = [
    ("float32", np.float32, abs_float32),
    ("float64", np.float64, abs_float64),
    ("int32", np.int32, abs_int32),
    ("int64", np.int64, abs_int64),
]

rng = np.random.default_rng(0)

for name, dtype, kernel in cases:
    if np.issubdtype(dtype, np.floating):
        inp = rng.uniform(-100.0, 100.0, size=1000).astype(dtype)
    else:
        inp = rng.integers(-100, 100, size=1000).astype(dtype)
    out = np.empty_like(inp)
    kernel(inp, out)
    assert np.array_equal(out, np.abs(inp)), name
    print(f"{name:<8} {inp.size} elements -> matches numpy.abs")

# %%
# Visualize the float32 kernel
# ----------------------------
#
# Feed a smooth ramp through the ``float32`` kernel and plot the input against
# the computed absolute value.

import matplotlib.pyplot as plt

x = np.linspace(-5.0, 5.0, 201).astype(np.float32)
y = np.empty_like(x)
abs_float32(x, y)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, x, label="input", linestyle="--", color="#9b7ec8")
ax.plot(x, y, label="abs_float32(input)", color="#4a9eff")
ax.axhline(0.0, color="black", linewidth=0.8)
ax.axvline(0.0, color="black", linewidth=0.8)
ax.set_title(f"onnx-light-cpu Abs (SIMD level: {_SIMD_NAMES.get(level, level)})")
ax.set_xlabel("input")
ax.set_ylabel("output")
ax.legend()
fig.tight_layout()
plt.show()
