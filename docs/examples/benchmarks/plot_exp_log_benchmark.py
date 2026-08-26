"""
Benchmark Exp and Log parallel scheduling
=========================================

This example compares ONNX Runtime, the built-in onnx-light kernels, the
onnx-light-cpu kernels, and NumPy around the scheduling thresholds used by the
Exp and Log CPU kernels. Each runtime is measured in a separate phase so that
persistent worker pools do not perturb measurements of another candidate.
"""

import argparse
import gc
import os
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime

from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import register_kernels

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-r", "--repeat", type=int, default=10 * (os.cpu_count() or 1))
parser.add_argument("-w", "--warmup", type=int, default=2 * (os.cpu_count() or 1))
parser.add_argument("-t", "--max-repeat-time", type=float, default=1.0)
args, _ = parser.parse_known_args()
if args.repeat <= 0:
    parser.error("--repeat must be greater than 0")
if args.warmup < 0:
    parser.error("--warmup must be greater than or equal to 0")
if args.max_repeat_time <= 0:
    parser.error("--max-repeat-time must be greater than 0")


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


def measure(run, arrays, repeat, warmup, max_duration):
    """Measures median inference times for all arrays."""
    results = []
    for array in arrays:
        warmup_duration = 0.0
        for _ in range(warmup):
            begin = time.perf_counter()
            run(array)
            warmup_duration += time.perf_counter() - begin
            if warmup_duration >= max_duration:
                break
        samples = []
        total_duration = 0.0
        for _ in range(repeat):
            begin = time.perf_counter()
            run(array)
            duration = time.perf_counter() - begin
            samples.append(duration)
            total_duration += duration
            if total_duration >= max_duration:
                break
        results.append(statistics.median(samples))
    return np.array(results)


threads = min(4, os.cpu_count() or 1)
if os.environ.get("UNITTEST_GOING"):
    sizes = np.array([1024, 65536, 131072], dtype=np.int64)
else:
    sizes = np.array(
        [1024, 32768, 65535, 65536, 131071, 131072, 262144, 1048576, 4194304],
        dtype=np.int64,
    )

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

all_times = {op_type: {} for op_type in models}
for op_type, arrays in (("Exp", exp_arrays), ("Log", log_arrays)):
    builtin = builtin_sessions[op_type]
    cpu = cpu_sessions[op_type]
    numpy_op = np.exp if op_type == "Exp" else np.log
    expected = numpy_op(arrays[-1])
    np.testing.assert_allclose(
        builtin.run(None, {"X": arrays[-1]})[0], expected, rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        cpu.run(None, {"X": arrays[-1]})[0], expected, rtol=2e-5, atol=2e-6
    )
    all_times[op_type]["onnx-light"] = measure(
        lambda array, current=builtin: current.run(None, {"X": array}),
        arrays,
        args.repeat,
        args.warmup,
        args.max_repeat_time,
    )
    all_times[op_type]["onnx-light-cpu"] = measure(
        lambda array, current=cpu: current.run(None, {"X": array}),
        arrays,
        args.repeat,
        args.warmup,
        args.max_repeat_time,
    )

del builtin
del cpu
del builtin_sessions
del cpu_sessions
gc.collect()

for op_type, arrays in (("Exp", exp_arrays), ("Log", log_arrays)):
    numpy_op = np.exp if op_type == "Exp" else np.log
    all_times[op_type]["NumPy"] = measure(
        numpy_op, arrays, args.repeat, args.warmup, args.max_repeat_time
    )

ort_sessions = {op_type: make_ort_session(model, threads) for op_type, model in models.items()}
for op_type, arrays in (("Exp", exp_arrays), ("Log", log_arrays)):
    ort = ort_sessions[op_type]
    numpy_op = np.exp if op_type == "Exp" else np.log
    np.testing.assert_allclose(
        ort.run(None, {"X": arrays[-1]})[0],
        numpy_op(arrays[-1]),
        rtol=2e-5,
        atol=2e-6,
    )
    all_times[op_type]["ONNX Runtime"] = measure(
        lambda array, current=ort: current.run(None, {"X": array}),
        arrays,
        args.repeat,
        args.warmup,
        args.max_repeat_time,
    )

for op_type in ("Exp", "Log"):
    print(f"\n{op_type}:")
    for size, numpy_time, light_time, cpu_time, ort_time in zip(
        sizes,
        all_times[op_type]["NumPy"],
        all_times[op_type]["onnx-light"],
        all_times[op_type]["onnx-light-cpu"],
        all_times[op_type]["ONNX Runtime"],
        strict=True,
    ):
        cpu_speedup = light_time / cpu_time
        ort_speedup = ort_time / cpu_time
        print(
            f"  size={size:>9} | numpy={numpy_time * 1e6:10.2f} us | "
            f"onnx-light={light_time * 1e6:10.2f} us | "
            f"onnx-light-cpu={cpu_time * 1e6:10.2f} us | "
            f"cpu vs built-in={cpu_speedup:5.2f}x | "
            f"onnxruntime={ort_time * 1e6:10.2f} us | "
            f"cpu vs onnxruntime={ort_speedup:5.2f}x"
        )

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
for row, op_type in enumerate(("Exp", "Log")):
    times = all_times[op_type]
    for label, values in times.items():
        axes[row, 0].plot(sizes, values * 1e6, marker="o", label=label)
        axes[row, 1].plot(sizes, times["ONNX Runtime"] / values, marker="o", label=label)
    threshold = 65536 if op_type == "Exp" else 131072
    for column in range(2):
        axes[row, column].axvline(
            threshold, color="black", linestyle="--", alpha=0.5, label="CPU threshold"
        )
        axes[row, column].set_xscale("log", base=2)
        axes[row, column].grid(True)
    axes[row, 0].set_yscale("log")
    axes[row, 0].set_title(f"{op_type} inference time")
    axes[row, 1].set_title(f"{op_type} speedup over ONNX Runtime")
    axes[row, 0].set_ylabel("median time (us)")
    axes[row, 1].set_ylabel("speedup")

axes[1, 0].set_xlabel("tensor elements")
axes[1, 1].set_xlabel("tensor elements")
axes[0, 0].legend()
plt.tight_layout()
fig.savefig("plot_exp_log_benchmark.png")
plt.show()
