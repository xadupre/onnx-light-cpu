"""
Benchmark custom operators against ONNX Runtime
===============================================

This benchmark compares the portable onnx-light-cpu implementations of
``com.microsoft.CDist`` and ``com.microsoft.BiasGelu`` with ONNX Runtime CPU.
"""

import argparse
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
parser.add_argument("-r", "--repeat", type=int, default=100)
parser.add_argument("-w", "--warmup", type=int, default=20)
parser.add_argument("-t", "--max-repeat-time", type=float, default=1.0)
args, _ = parser.parse_known_args()


def make_model(op_type, inputs, output, **attributes):
    """Builds one dynamic custom-operator model."""
    node = helper.make_node(
        op_type,
        [value.name for value in inputs],
        [output.name],
        domain="com.microsoft",
        **attributes,
    )
    return helper.make_model(
        helper.make_graph([node], op_type, inputs, [output]),
        opset_imports=[
            helper.make_opsetid("", 20),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=13,
    )


def measure(run, feeds):
    """Returns the median bounded inference time."""
    for _ in range(args.warmup):
        run(feeds)
    samples = []
    elapsed = 0.0
    for _ in range(args.repeat):
        begin = time.perf_counter()
        run(feeds)
        duration = time.perf_counter() - begin
        samples.append(duration)
        elapsed += duration
        if elapsed >= args.max_repeat_time:
            break
    return statistics.median(samples)


def make_ort(model):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )


sizes = [16, 64] if os.environ.get("UNITTEST_GOING") else [16, 64, 256, 1024]
rng = np.random.default_rng(42)
models = {
    "CDist": make_model(
        "CDist",
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [None, 32]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [None, 32]),
        ],
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None]),
        metric="sqeuclidean",
    ),
    "BiasGelu": make_model(
        "BiasGelu",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 128]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [128]),
        ],
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 128]),
    ),
}
ort_sessions = {name: make_ort(model) for name, model in models.items()}
register_kernels()
light_sessions = {
    name: ReferenceEvaluator(
        model,
        cpu_execution={"num_threads": min(4, os.cpu_count() or 1), "affinity_policy": "none"},
    )
    for name, model in models.items()
}

results = {name: {"onnx-light-cpu": [], "ONNX Runtime": []} for name in models}
for size in sizes:
    feeds_by_op = {
        "CDist": {
            "A": rng.standard_normal((size, 32), dtype=np.float32),
            "B": rng.standard_normal((size, 32), dtype=np.float32),
        },
        "BiasGelu": {
            "X": rng.standard_normal((size, 128), dtype=np.float32),
            "bias": rng.standard_normal(128, dtype=np.float32),
        },
    }
    for op_type, feeds in feeds_by_op.items():
        light = light_sessions[op_type]
        ort = ort_sessions[op_type]
        expected = ort.run(None, feeds)[0]
        actual = light.run(None, feeds)[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        results[op_type]["onnx-light-cpu"].append(
            measure(lambda current_feeds, sess=light: sess.run(None, current_feeds), feeds)
        )
        results[op_type]["ONNX Runtime"].append(
            measure(lambda current_feeds, sess=ort: sess.run(None, current_feeds), feeds)
        )

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for axis, op_type in zip(axes, ("CDist", "BiasGelu"), strict=True):
    for runtime, values in results[op_type].items():
        axis.plot(sizes, np.asarray(values) * 1e6, marker="o", label=runtime)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_title(op_type)
    axis.set_xlabel("rows")
    axis.set_ylabel("median time (us)")
    axis.grid(True)
    axis.legend()
plt.tight_layout()
fig.savefig("plot_custom_operators_benchmark.png")
plt.show()
