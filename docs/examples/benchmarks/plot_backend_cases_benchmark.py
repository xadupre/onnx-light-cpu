"""
Benchmark a subset of backend test cases against ONNX Runtime
================================================================

onnx-light-cpu ships its own ONNX backend test cases -- named ``test_cpu_*``
-- in a dedicated C++ registration library
(``lib_onnx_light_cpu_backend_test``, see :func:`onnx_light_cpu.register_backend_test_cases`).
Every case that has an accelerated kernel also has a ``TestMode.BENCHMARK``
variant: the same operator and attributes, but with inputs large enough that a
single evaluation takes a measurable amount of time. This example walks those
benchmark cases -- covering unary, binary, and matrix operators -- and times
each one through ``onnx-light`` (with onnx-light-cpu's accelerated kernel
registered) and through ONNX Runtime, using the exact same generated model and
inputs for both.

Each case is also checked for correctness (both runtimes must agree, within
the case's tolerance) and for kernel dispatch (the accelerated onnx-light-cpu
kernel, not onnx-light's built-in reference kernel, must have run), the same
way :mod:`unittests.python.test_kernels_e2e` verifies these backend cases.
"""

# %%
# Setup
# -----
#
# Only ``float32`` (``bool`` for ``Not``) cases are kept so every candidate can
# be compared directly; onnx-light-cpu also ships ``float16``/``bfloat16``
# benchmark variants of the same cases (see
# ``onnx_light_cpu/backend_test/cases/math/cases_gemm.cc``) which are outside
# the scope of this example.

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime

from onnx_light.onnx import TensorProto
from onnx_light.onnx.backend import TestMode, collect_test_cases_by_name
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

# Operators covered by onnx-light-cpu's "test_cpu_*" backend test cases, mapped
# to the library-qualified kernel name each records when it runs.
_TARGET_KERNELS = {
    "Add": "onnx_light_cpu::Add",
    "And": "onnx_light_cpu::And",
    "Abs": "onnx_light_cpu::Abs",
    "BitShift": "onnx_light_cpu::BitShift",
    "BitwiseAnd": "onnx_light_cpu::BitwiseAnd",
    "BitwiseOr": "onnx_light_cpu::BitwiseOr",
    "BitwiseXor": "onnx_light_cpu::BitwiseXor",
    "Div": "onnx_light_cpu::Div",
    "Equal": "onnx_light_cpu::Equal",
    "Exp": "onnx_light_cpu::Exp",
    "Greater": "onnx_light_cpu::Greater",
    "GreaterOrEqual": "onnx_light_cpu::GreaterOrEqual",
    "Log": "onnx_light_cpu::Log",
    "Less": "onnx_light_cpu::Less",
    "LessOrEqual": "onnx_light_cpu::LessOrEqual",
    "Mod": "onnx_light_cpu::Mod",
    "Mul": "onnx_light_cpu::Mul",
    "Gemm": "onnx_light_cpu::Gemm",
    "Not": "onnx_light_cpu::Not",
    "Or": "onnx_light_cpu::Or",
    "PRelu": "onnx_light_cpu::PRelu",
    "Pow": "onnx_light_cpu::Pow",
    "Sub": "onnx_light_cpu::Sub",
    "Xor": "onnx_light_cpu::Xor",
}

# The input/output type pair kept for each operator. Comparisons consume
# float32 and produce bool; logical and bitwise operators use their natural
# homogeneous types.
_TARGET_DTYPES = dict.fromkeys(_TARGET_KERNELS, (TensorProto.FLOAT, TensorProto.FLOAT))
for _op_type in ("Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"):
    _TARGET_DTYPES[_op_type] = (TensorProto.FLOAT, TensorProto.BOOL)
for _op_type in ("And", "Not", "Or", "Xor"):
    _TARGET_DTYPES[_op_type] = (TensorProto.BOOL, TensorProto.BOOL)
for _op_type in ("BitwiseAnd", "BitwiseOr", "BitwiseXor"):
    _TARGET_DTYPES[_op_type] = (TensorProto.INT8, TensorProto.INT8)
_TARGET_DTYPES["BitShift"] = (TensorProto.UINT8, TensorProto.UINT8)


def _to_numpy(tensor):
    """Decodes a backend test case ``Tensor`` into a numpy array."""
    dtype = {
        int(TensorProto.BOOL): np.bool_,
        int(TensorProto.FLOAT): np.float32,
        int(TensorProto.INT8): np.int8,
        int(TensorProto.UINT8): np.uint8,
    }[int(tensor.data_type)]
    shape = tuple(int(d) for d in tensor.shape)
    return np.frombuffer(tensor.raw_data(), dtype=dtype).reshape(shape)


def _collect_cases():
    """Registers and collects a subset of the "test_cpu_*" BENCHMARK cases.

    A single regular expression -- matching every "test_cpu_<op>_..." name
    for the target operators -- is used instead of collecting per operator.
    """
    register_backend_test_cases()
    pattern = "^test_cpu_(" + "|".join(op.lower() for op in _TARGET_KERNELS) + ")_"
    max_per_op = 2 if os.environ.get("UNITTEST_GOING", "0") in ("1", "true", "True") else None
    counts = dict.fromkeys(_TARGET_KERNELS, 0)
    cases = []
    for tc in collect_test_cases_by_name(pattern, mode=TestMode.BENCHMARK):
        nodes = list(tc.model.graph.node)
        op_type = nodes[0].op_type if len(nodes) == 1 else None
        if op_type not in _TARGET_KERNELS or not tc.data_sets:
            continue
        input_type, output_type = _TARGET_DTYPES[op_type]
        if not all(
            all(int(tensor.data_type) == int(input_type) for tensor in data_set.inputs)
            and all(int(tensor.data_type) == int(output_type) for tensor in data_set.outputs)
            for data_set in tc.data_sets
        ):
            continue
        if max_per_op is not None and counts[op_type] >= max_per_op:
            continue
        counts[op_type] += 1
        cases.append(tc)
    return cases


_CASES = _collect_cases()
assert _CASES, "no onnx-light-cpu BENCHMARK backend test cases were collected"

# %%
# Timing helper
# -------------
#
# Each candidate gets three untimed warm-up calls, then is called ``repeat``
# times and the median wall-clock time is retained. ``repeat`` shrinks as the
# case's inputs grow but never below three.


def measure(func, repeat, warmup=3):
    for _ in range(warmup):
        func()
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def _case_element_count(tc):
    return max(
        (
            int(np.prod([int(d) for d in tensor.shape]))
            for ds in tc.data_sets
            for tensor in (*ds.inputs, *ds.outputs)
        ),
        default=1,
    )


# %%
# Run every case through onnx-light-cpu and ONNX Runtime
# -------------------------------------------------------
#
# ``register_kernels()`` installs the accelerated kernels in onnx-light's
# process-wide dispatch table before any of the onnx-light-cpu sessions below
# run for the first time.

register_kernels()

rows = []
for tc in _CASES:
    op_type = tc.model.graph.node[0].op_type
    expected_kernel = _TARGET_KERNELS[op_type]
    model_bytes = tc.model.SerializeToString()
    # Some Gemm benchmark cases turn "B" into a graph initializer to exercise
    # the constant-B code path; its value is baked into the model rather than
    # fed at run time, so it must be excluded from the runtime feeds below.
    initializer_names = {init.name for init in tc.model.graph.initializer}
    input_names = [vi.name for vi in tc.model.graph.input if vi.name not in initializer_names]

    light_session = ReferenceEvaluator(tc.model)
    ort_session = onnxruntime.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])

    ds = tc.data_sets[0]
    feeds = {name: _to_numpy(t) for name, t in zip(input_names, ds.inputs, strict=True)}
    # The case's own rtol/atol are tuned for comparing against its shipped
    # reference output (computed by the very same accelerated kernel used
    # here); onnxruntime is an independent implementation whose reductions
    # (e.g. Gemm's dot products, Exp/Log's large-input rounding) accumulate in
    # a different order, so a looser, fixed tolerance is used instead.
    rtol, atol = 1e-2, 1e-3

    clear_used_kernel_names()
    light_out = light_session.run(None, feeds)
    assert expected_kernel in used_kernel_names(), used_kernel_names()
    ort_out = ort_session.run(None, feeds)
    for actual, expected in zip(light_out, ort_out, strict=True):
        if expected.dtype == np.bool_:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(
                actual.astype(np.float64),
                expected.astype(np.float64),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )

    # Aim for roughly a constant total element budget per case (~2e7 elements
    # processed across all repeats) so large cases are not re-run too many
    # times, but always repeat at least 3 times and never more than 30.
    repeat = max(3, min(30, 20_000_000 // _case_element_count(tc)))
    light_time = measure(lambda feeds=feeds, sess=light_session: sess.run(None, feeds), repeat)
    ort_time = measure(lambda feeds=feeds, sess=ort_session: sess.run(None, feeds), repeat)
    rows.append((op_type, tc.name, light_time, ort_time))
    print(
        f"{op_type:>5} | {tc.name:<55} | onnx-light-cpu={light_time * 1e6:10.2f} us | "
        f"onnxruntime={ort_time * 1e6:10.2f} us | speed-up={ort_time / light_time:6.2f}x"
    )

# %%
# Plot the speed-ups
# -------------------
#
# One bar per case, grouped and colored by operator, showing onnx-light-cpu's
# speed-up over ONNX Runtime (values above 1 mean onnx-light-cpu is faster).
# The x-axis is logarithmic so a speed-up and its reciprocal are equidistant
# from the ``1`` baseline.

_COLOR_MAP = plt.get_cmap("turbo", len(_TARGET_KERNELS))
_COLORS = {op_type: _COLOR_MAP(index) for index, op_type in enumerate(_TARGET_KERNELS)}


def _short_label(op_type, name):
    label = name.removeprefix("test_cpu_")
    label = label.removeprefix(op_type.lower() + "_")
    return label.removesuffix("_benchmark")


labels = [_short_label(op_type, name) for op_type, name, _, _ in rows]
speedups = np.array([ort_time / light_time for _, _, light_time, ort_time in rows])
colors = [_COLORS[op_type] for op_type, _, _, _ in rows]

fig, ax = plt.subplots(figsize=(8, max(5, 0.4 * len(rows))))
positions = np.arange(len(rows))
ax.barh(positions, speedups, color=colors)
ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":")
ax.set_xscale("log")
ax.set_yticks(positions, labels, fontsize=7)
ax.set_xlabel("speed-up vs onnxruntime")
ax.set_title("onnx-light-cpu speed-up over onnxruntime on backend cases")

handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in _COLORS.values()]
ax.legend(handles, _COLORS.keys(), title="operator", loc="upper left", fontsize=8, ncols=3)

fig.tight_layout()
fig.savefig("plot_backend_cases_benchmark.png")
plt.show()
