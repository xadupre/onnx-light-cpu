"""
Benchmark Exp and Log parallel scheduling
=========================================

This example compares ONNX Runtime, the built-in onnx-light kernels, the
onnx-light-cpu kernels, and NumPy around the scheduling thresholds used by the
Exp and Log CPU kernels. Each runtime is measured in a separate phase so that
persistent worker pools do not perturb measurements of another candidate.
"""

import os
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime

from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import register_kernels


def make_model(op_type):
    """Creates a dynamic one-input ONNX model."""
    return helper.make_model(
        helper.make_graph(
            [helper.make_node(op_type, ["X"], ["Y"])],
            op_type,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])],
        ),
        opset_imports=[helper.make_opsetid("", 20)],
        ir_version=13,
    )


def make_ort_session(model, threads):
    """Creates a sequential ONNX Runtime session."""
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    return onnxruntime.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def measure(run, arrays, repeat):
    """Measures median inference times for all arrays."""
    run(arrays[0])
    results = []
    for array in arrays:
        samples = []
        for _ in range(repeat):
            begin = time.perf_counter()
            run(array)
            samples.append(time.perf_counter() - begin)
        results.append(statistics.median(samples))
    return np.array(results)


threads = min(4, os.cpu_count() or 1)
if os.environ.get("UNITTEST_GOING"):
    sizes = np.array([1024, 65536, 131072], dtype=np.int64)
    repeat = 2
else:
    sizes = np.array(
        [1024, 32768, 65535, 65536, 131071, 131072, 262144, 1048576, 4194304],
        dtype=np.int64,
    )
    repeat = 10

rng = np.random.default_rng(42)
exp_arrays = [rng.uniform(-4, 4, int(size)).astype(np.float32) for size in sizes]
log_arrays = [rng.uniform(0.01, 10, int(size)).astype(np.float32) for size in sizes]
models = {"Exp": make_model("Exp"), "Log": make_model("Log")}

# Sessions using the built-in kernels must be created before the global CPU
# kernel registration.
builtin_sessions = {op_type: ReferenceEvaluator(model) for op_type, model in models.items()}
for session in builtin_sessions.values():
    session.run(None, {"X": np.ones(1, dtype=np.float32)})

register_kernels()
cpu_sessions = {
    op_type: ReferenceEvaluator(
        model, cpu_execution={"num_threads": threads, "affinity_policy": "none"}
    )
    for op_type, model in models.items()
}
ort_sessions = {op_type: make_ort_session(model, threads) for op_type, model in models.items()}

all_times = {}
for op_type, arrays in (("Exp", exp_arrays), ("Log", log_arrays)):
    builtin = builtin_sessions[op_type]
    cpu = cpu_sessions[op_type]
    ort = ort_sessions[op_type]
    expected = builtin.run(None, {"X": arrays[-1]})[0]
    np.testing.assert_allclose(
        cpu.run(None, {"X": arrays[-1]})[0], expected, rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        ort.run(None, {"X": arrays[-1]})[0], expected, rtol=2e-5, atol=2e-6
    )
    numpy_op = np.exp if op_type == "Exp" else np.log

    all_times[op_type] = {
        "onnx-light": measure(
            lambda array, current=builtin: current.run(None, {"X": array}), arrays, repeat
        ),
        "onnx-light-cpu": measure(
            lambda array, current=cpu: current.run(None, {"X": array}), arrays, repeat
        ),
        "ONNX Runtime": measure(
            lambda array, current=ort: current.run(None, {"X": array}), arrays, repeat
        ),
        "NumPy": measure(numpy_op, arrays, repeat),
    }

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
for column, op_type in enumerate(("Exp", "Log")):
    times = all_times[op_type]
    for label, values in times.items():
        axes[0, column].plot(sizes, values * 1e6, marker="o", label=label)
        axes[1, column].plot(sizes, times["ONNX Runtime"] / values, marker="o", label=label)
    threshold = 65536 if op_type == "Exp" else 131072
    for row in range(2):
        axes[row, column].axvline(
            threshold, color="black", linestyle="--", alpha=0.5, label="CPU threshold"
        )
        axes[row, column].set_xscale("log", base=2)
        axes[row, column].grid(True)
    axes[0, column].set_yscale("log")
    axes[0, column].set_title(f"{op_type} inference time")
    axes[1, column].set_title(f"{op_type} speedup over ONNX Runtime")
    axes[1, column].set_xlabel("tensor elements")

axes[0, 0].set_ylabel("median time (us)")
axes[1, 0].set_ylabel("speedup")
axes[0, 0].legend()
plt.tight_layout()
plt.show()
