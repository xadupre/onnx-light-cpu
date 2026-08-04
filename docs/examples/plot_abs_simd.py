"""
Elementwise Abs with runtime SIMD dispatch
==========================================

This example exercises the SIMD-accelerated ``Abs`` kernels provided by
``onnx-light-cpu`` through an ``onnx-light`` :class:`ReferenceEvaluator`. The
:func:`onnx_light_cpu.register_kernels` helper registers every optimized kernel
(``Abs``, ``Exp``, ``Log``, ``Not`` and ``Gemm``) on the evaluator so any ONNX
model dispatches to them. The extension detects the best available instruction
set (AVX-512, AVX2, AVX, SSE2, or a scalar fallback) once at runtime and
dispatches every call to that implementation.

A single-node ``Abs`` model is evaluated for every supported data type
(``float32``, ``float64``, ``int32`` and ``int64``), the result is checked
against :func:`numpy.abs`, and then the input/output of the ``float32`` kernel
is plotted to illustrate the operation.
"""

# %%
# Setup
# -----
#
# Import the ``onnx-light-cpu`` registration helper and report which SIMD level
# the current CPU provides. The mapping is ``0=None``, ``1=SSE2``, ``2=AVX``,
# ``3=AVX2`` and ``4=AVX512``.

import numpy as np

# ``onnx-light`` ships ``onnx_light.onnx`` as a drop-in replacement for the
# ``onnx`` package; use it to build the model so the example depends on
# onnx-light rather than onnx.
from onnx_light.onnx import TensorProto, checker, helper
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import register_kernels
from onnx_light_cpu.onnx_py._cpukernels import (
    detect_simd_level,
    has_cpu_kernels,
)

_SIMD_NAMES = {0: "scalar", 1: "SSE2", 2: "AVX", 3: "AVX2", 4: "AVX-512"}

assert has_cpu_kernels()
level = detect_simd_level()
simd_name = _SIMD_NAMES.get(level, level)
print(f"CPU kernels available, SIMD level: {level} ({simd_name})")

# %%
# Build a single-node ``Abs`` model
# ---------------------------------
#
# A single ``Abs`` node operating on a 1-D tensor of dynamic length is enough to
# exercise the kernel. The tensor element type is filled in per dtype below.


def make_abs_model(tensor_type):
    graph = helper.make_graph(
        [helper.make_node("Abs", ["X"], ["Y"])],
        "abs",
        [helper.make_tensor_value_info("X", tensor_type, ["N"])],
        [helper.make_tensor_value_info("Y", tensor_type, ["N"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    checker.check_model(model)
    return model


# %%
# Run every supported data type
# -----------------------------
#
# ``register_kernels`` overrides the built-in ``Abs`` so every ``Abs`` node in
# the model dispatches to the SIMD-accelerated kernel, which handles the array
# dtype (just like :func:`numpy.abs`).

dtypes = [
    (np.float32, TensorProto.FLOAT),
    (np.float64, TensorProto.DOUBLE),
    (np.int32, TensorProto.INT32),
    (np.int64, TensorProto.INT64),
]

rng = np.random.default_rng(0)

for dtype, tensor_type in dtypes:
    sess = ReferenceEvaluator(make_abs_model(tensor_type))
    register_kernels(sess)
    if np.issubdtype(dtype, np.floating):
        inp = rng.uniform(-100.0, 100.0, size=1000).astype(dtype)
    else:
        inp = rng.integers(-100, 100, size=1000).astype(dtype)
    out = sess.run(None, {"X": inp})[0]
    assert np.array_equal(out, np.abs(inp)), dtype
    print(f"{np.dtype(dtype).name:<8} {inp.size} elements -> matches numpy.abs")

# %%
# Visualize the float32 kernel
# ----------------------------
#
# Feed a smooth ramp through the ``float32`` kernel and plot the input against
# the computed absolute value.

import matplotlib.pyplot as plt

float_sess = ReferenceEvaluator(make_abs_model(TensorProto.FLOAT))
register_kernels(float_sess)

x = np.linspace(-5.0, 5.0, 201).astype(np.float32)
y = float_sess.run(None, {"X": x})[0]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, x, label="input", linestyle="--", color="#9b7ec8")
ax.plot(x, y, label="abs(input)", color="#4a9eff")
ax.axhline(0.0, color="black", linewidth=0.8)
ax.axvline(0.0, color="black", linewidth=0.8)
ax.set_title(f"onnx-light-cpu Abs (SIMD level: {simd_name})")
ax.set_xlabel("input")
ax.set_ylabel("output")
ax.legend()
fig.tight_layout()
plt.show()
