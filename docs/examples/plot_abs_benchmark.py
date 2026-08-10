"""
Benchmark Abs: onnxruntime vs onnx-light + onnx-light-cpu
=========================================================

This example compares up to four ways of computing the elementwise absolute value
of a ``float32`` array across a range of input sizes:

* **onnxruntime** - running a single-node ``Abs`` ONNX model.
* **onnx-light + onnx-light-cpu** - the SIMD-accelerated ``Abs`` kernel that
  ``onnx-light`` dispatches to. The *same* ONNX model used by onnxruntime is
  evaluated by an ``onnx-light`` :class:`ReferenceEvaluator` on which the
  ``onnx-light-cpu`` ``Abs`` kernel has been registered
  (:func:`onnx_light_cpu.register_kernels`); the kernel provides runtime
  AVX-512/AVX2/AVX/SSE2 dispatch.
* **onnx-light (built-in)** - ``onnx-light``'s own un-accelerated (pure
  reference) ``Abs`` kernel, as a baseline for what ``onnx-light-cpu`` buys on
  top of it. It is measured across the complete size grid.
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
# A single ``Abs`` node operating on a 1-D ``float32`` tensor of dynamic length
# is enough to benchmark the runtimes. The exact same model is fed to
# onnxruntime and to onnx-light so the comparison is apples-to-apples.


def make_abs_model():
    graph = helper.make_graph(
        [helper.make_node("Abs", ["X"], ["Y"])],
        "abs_bench",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    checker.check_model(model)
    return model


model = make_abs_model()

# Serialize once (outside the timed region) so the setup timing below measures
# only the session construction and not the protobuf serialization.
model_bytes = model.SerializeToString()

_ort_setup_start = time.perf_counter()
session = onnxruntime.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
ort_setup_time = time.perf_counter() - _ort_setup_start

# %%
# Sizes benchmarked
# -----------------

size_grid = [10**k for k in range(2, 9)]


def measure(func, repeat, warmup=3):
    for _ in range(warmup):
        func()
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def measure_pair(first, second, repeat, warmup=3):
    for _ in range(warmup):
        first()
        second()
    timings = ([], [])
    funcs = (first, second)
    for iteration in range(repeat):
        order = (0, 1) if iteration % 2 == 0 else (1, 0)
        for index in order:
            start = time.perf_counter()
            funcs[index]()
            timings[index].append(time.perf_counter() - start)
    return tuple(float(np.median(values)) for values in timings)


# %%
# Prepare the built-in (un-accelerated) onnx-light Abs kernel
# -------------------------------------------------------------
#
# ``onnx_light_cpu.register_kernels()`` permanently overrides the process-wide
# ``Abs`` kernel entry, and a session only resolves/caches which kernel it
# uses on its *first* run. This baseline therefore uses its own model/session
# and runs once **before** ``register_kernels()`` is called below. Its cached
# built-in kernel can then be timed alongside the accelerated session, on the
# same inputs and with alternating measurement order.

alone_model = make_abs_model()
alone_session = ReferenceEvaluator(alone_model)
alone_label = "onnx-light (built-in)"
alone_session.run(None, {"X": np.zeros(1, dtype=np.float32)})

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
register_kernels()
light_session = ReferenceEvaluator(model)
light_setup_time = time.perf_counter() - _light_setup_start

# Confirm the model dispatches to the onnx-light-cpu ``Abs`` kernel (identified
# by the library-qualified name it records when it runs) rather than
# onnx-light's built-in kernel.
clear_used_kernel_names()
light_session.run(None, {"X": np.zeros(1, dtype=np.float32)})
assert used_kernel_names() == ["onnx_light_cpu::Abs"], used_kernel_names()


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
# %%
# Run the benchmark
# -----------------
#
# For every size the same input is fed to the four back-ends. Each measurement
# starts with three untimed warm-up calls, then retains the median of the timed
# repetitions. The two onnx-light variants are measured as a pair, alternating
# which one runs first to reduce cache and scheduling bias. At least seven timed
# repetitions are used, including for the largest arrays. The results are checked
# against :func:`numpy.abs` to make sure every implementation agrees.

rng = np.random.default_rng(0)

rows = []
for size in size_grid:
    inp = rng.uniform(-100.0, 100.0, size=size).astype(np.float32)
    expected = np.abs(inp)

    repeat = max(7, min(200, 2_000_000 // size))

    numpy_time = measure(lambda inp=inp: np.abs(inp), repeat)

    if light_session is not None:
        alone_time, cpu_time = measure_pair(
            lambda inp=inp: alone_session.run(None, {"X": inp}),
            lambda inp=inp: run_light(inp),
            repeat,
        )
        assert np.array_equal(alone_session.run(None, {"X": inp})[0], expected), size
        assert np.array_equal(run_light(inp), expected), size
    else:
        alone_time = measure(lambda inp=inp: alone_session.run(None, {"X": inp}), repeat)
        cpu_time = float("nan")

    ort_time = measure(lambda inp=inp: session.run(None, {"X": inp}), repeat)
    assert np.array_equal(session.run(None, {"X": inp})[0], expected), size

    cpu_speedup = alone_time / cpu_time
    rows.append((size, numpy_time, alone_time, cpu_time, ort_time))
    print(
        f"size={size:>9} | numpy={numpy_time * 1e6:10.2f} us | "
        f"onnx-light={alone_time * 1e6:10.2f} us | "
        f"onnx-light-cpu={cpu_time * 1e6:10.2f} us | "
        f"cpu speed-up={cpu_speedup:5.2f}x | "
        f"onnxruntime={ort_time * 1e6:10.2f} us"
    )

sizes = np.array([r[0] for r in rows])
numpy_times = np.array([r[1] for r in rows])
alone_times = np.array([r[2] for r in rows])
cpu_times = np.array([r[3] for r in rows])
ort_times = np.array([r[4] for r in rows])

# %%
# Plot the timings
# ----------------
#
# The left panel shows the raw execution time versus the array size on a
# log-log scale. The middle panel shows the speed-up relative to
# **onnxruntime** (the baseline): for each back-end the onnxruntime time is
# divided by the back-end time, so values above ``1`` are faster than
# onnxruntime and values below ``1`` are slower. The onnxruntime curve is a
# flat line at ``1`` by construction. The right panel isolates the comparison
# of interest: ``onnx-light built-in time / onnx-light-cpu time``. Values above
# ``1`` therefore show directly how much the CPU kernel is faster.

import matplotlib.pyplot as plt

fig, (ax_time, ax_speedup, ax_cpu_gain) = plt.subplots(1, 3, figsize=(16, 4.5))

ax_time.plot(sizes, numpy_times * 1e6, "o--", label="numpy", color="#9b7ec8")
if light_session is not None:
    ax_time.plot(
        sizes,
        cpu_times * 1e6,
        "o-",
        label=light_label,
        color="#4a9eff",
    )
ax_time.plot(
    sizes,
    alone_times * 1e6,
    "o--",
    label=alone_label,
    color="#5cb85c",
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
ax_speedup.plot(
    sizes,
    ort_times / alone_times,
    "o--",
    label=alone_label,
    color="#5cb85c",
)
ax_speedup.plot(sizes, ort_times / ort_times, "o-", label="onnxruntime", color="#f4a259")
ax_speedup.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
ax_speedup.set_xscale("log")
ax_speedup.set_yscale("log")
ax_speedup.set_xlabel("array size (elements)")
ax_speedup.set_ylabel("speed-up vs onnxruntime")
ax_speedup.set_title("Abs speed-up (onnxruntime = 1)")
ax_speedup.legend()

if light_session is not None:
    cpu_gain = alone_times / cpu_times
    ax_cpu_gain.plot(sizes, cpu_gain, "o-", color="#4a9eff")
    for size, gain in zip(sizes, cpu_gain, strict=True):
        ax_cpu_gain.annotate(
            f"{gain:.2f}x",
            (size, gain),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
ax_cpu_gain.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
ax_cpu_gain.set_xscale("log")
ax_cpu_gain.set_xlabel("array size (elements)")
ax_cpu_gain.set_ylabel("speed-up vs onnx-light built-in")
ax_cpu_gain.set_title("onnx-light-cpu gain (built-in = 1)")

fig.tight_layout()
fig.savefig("plot_abs_benchmark.png")
plt.show()
