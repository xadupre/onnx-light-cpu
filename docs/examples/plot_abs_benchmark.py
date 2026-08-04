"""
Benchmark Abs: onnxruntime vs onnx-light + onnx-light-cpu
=========================================================

This example compares up to three ways of computing the elementwise absolute value
of a ``float32`` array across a range of input sizes:

* **onnxruntime** - running a single-node ``Abs`` ONNX model.
* **onnx-light + onnx-light-cpu** - the SIMD-accelerated ``Abs`` kernel that
  ``onnx-light`` dispatches to. The *same* ONNX model used by onnxruntime is
  evaluated by an ``onnx-light`` :class:`ReferenceEvaluator` on which the
  ``onnx-light-cpu`` ``Abs`` kernel has been registered
  (:func:`onnx_light_cpu.register_kernels`); the kernel provides runtime
  AVX-512/AVX2/AVX/SSE2 dispatch.
* **numpy** - :func:`numpy.abs`, used as a reference baseline.

The back-ends compute the same result; the goal here is to see how their
timings evolve as the array grows from a few hundred to a hundred million
elements.
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
# A single ``Abs`` node operating on a 1-D ``float32`` tensor of dynamic length
# is enough to benchmark the runtimes. The exact same model is fed to
# onnxruntime and to onnx-light so the comparison is apples-to-apples.

graph = helper.make_graph(
    [helper.make_node("Abs", ["X"], ["Y"])],
    "abs_bench",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N"])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
checker.check_model(model)

# Serialize once (outside the timed region) so the setup timing below measures
# only the session construction and not the protobuf serialization.
model_bytes = model.SerializeToString()

_ort_setup_start = time.perf_counter()
session = onnxruntime.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
ort_setup_time = time.perf_counter() - _ort_setup_start

# %%
# Build the onnx-light evaluator
# ------------------------------
#
# ``onnx-light`` evaluates the same model with its C++ runtime. Registering the
# onnx-light-cpu kernels overrides the built-in ``Abs`` so every ``Abs`` node in
# the model dispatches to the SIMD-accelerated kernel.

_light_setup_start = time.perf_counter()
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
light_setup_time = time.perf_counter() - _light_setup_start


def run_light(inp):
    return light_session.run(None, {"X": inp})[0]


# %%
# Setup cost: why the evaluator is as slow to build as onnxruntime
# ----------------------------------------------------------------
#
# Constructing an ``onnx-light`` :class:`ReferenceEvaluator` now costs about as
# much as constructing an :class:`onnxruntime.InferenceSession`. This is expected
# and is a *one-time* cost paid before any :meth:`run`:
#
# * ``onnxruntime`` parses the model and builds an optimized execution plan at
#   construction time.
# * ``onnx-light`` deliberately front-loads the same kind of work into
#   ``ReferenceEvaluator.__init__``: it eagerly builds the opset ``KernelContext``
#   and the persistent ``RuntimeContext`` once (instead of rebuilding them on
#   every ``run`` call), and :func:`onnx_light_cpu.register_kernels` installs the
#   custom kernels. Earlier onnx-light versions did this lazily, so construction
#   looked instantaneous but the first ``run`` absorbed the cost.
#
# The payoff is that the amortized per-call ``run`` time (measured below) stays
# low, because the expensive analysis happens exactly once at setup rather than
# on every invocation.

print(f"setup: onnxruntime InferenceSession = {ort_setup_time * 1e3:.2f} ms")
print(f"setup: onnx-light ReferenceEvaluator = {light_setup_time * 1e3:.2f} ms")

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

    if light_session is not None:
        cpu_time = measure(lambda inp=inp: run_light(inp), repeat)
        assert np.array_equal(run_light(inp), expected), size
    else:
        cpu_time = float("nan")

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
# log-log scale. The right panel shows the speed-up relative to
# **onnxruntime** (the baseline): for each back-end the onnxruntime time is
# divided by the back-end time, so values above ``1`` are faster than
# onnxruntime and values below ``1`` are slower. The onnxruntime curve is a
# flat line at ``1`` by construction.

import matplotlib.pyplot as plt

fig, (ax_time, ax_speedup) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_time.plot(sizes, numpy_times * 1e6, "o--", label="numpy", color="#9b7ec8")
if light_session is not None:
    ax_time.plot(
        sizes,
        cpu_times * 1e6,
        "o-",
        label=light_label,
        color="#4a9eff",
    )
ax_time.plot(sizes, ort_times * 1e6, "o-", label="onnxruntime", color="#f4a259")
ax_time.set_xscale("log")
ax_time.set_yscale("log")
ax_time.set_xlabel("array size (elements)")
ax_time.set_ylabel("time (microseconds)")
ax_time.set_title(f"Abs execution time (SIMD: {simd_name})")
ax_time.legend()

ax_speedup.plot(sizes, ort_times / numpy_times, "o--", label="numpy", color="#9b7ec8")
if light_session is not None:
    ax_speedup.plot(
        sizes,
        ort_times / cpu_times,
        "o-",
        label=light_label,
        color="#4a9eff",
    )
ax_speedup.plot(sizes, ort_times / ort_times, "o-", label="onnxruntime", color="#f4a259")
ax_speedup.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
ax_speedup.set_xscale("log")
ax_speedup.set_yscale("log")
ax_speedup.set_xlabel("array size (elements)")
ax_speedup.set_ylabel("speed-up vs onnxruntime")
ax_speedup.set_title("Abs speed-up (onnxruntime = 1)")
ax_speedup.legend()

fig.tight_layout()
plt.show()
