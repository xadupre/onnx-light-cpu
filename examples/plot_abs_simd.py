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
    abs,
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
# A single ``abs`` entry point dispatches on the array dtype (just like
# :func:`numpy.abs`) and returns a new array of the same dtype.

dtypes = [np.float32, np.float64, np.int32, np.int64]

rng = np.random.default_rng(0)

for dtype in dtypes:
    if np.issubdtype(dtype, np.floating):
        inp = rng.uniform(-100.0, 100.0, size=1000).astype(dtype)
    else:
        inp = rng.integers(-100, 100, size=1000).astype(dtype)
    out = abs(inp)
    assert np.array_equal(out, np.abs(inp)), dtype
    print(f"{np.dtype(dtype).name:<8} {inp.size} elements -> matches numpy.abs")

# %%
# Visualize the float32 kernel
# ----------------------------
#
# Feed a smooth ramp through the ``float32`` kernel and plot the input against
# the computed absolute value.

import matplotlib.pyplot as plt

x = np.linspace(-5.0, 5.0, 201).astype(np.float32)
y = abs(x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, x, label="input", linestyle="--", color="#9b7ec8")
ax.plot(x, y, label="abs(input)", color="#4a9eff")
ax.axhline(0.0, color="black", linewidth=0.8)
ax.axvline(0.0, color="black", linewidth=0.8)
ax.set_title(f"onnx-light-cpu Abs (SIMD level: {_SIMD_NAMES.get(level, level)})")
ax.set_xlabel("input")
ax.set_ylabel("output")
ax.legend()
fig.tight_layout()
plt.show()
