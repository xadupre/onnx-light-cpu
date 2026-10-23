"""
Packed embedding gather against ONNX Runtime
===========================================

Compare ``com.microsoft::GatherBlockQuantized`` on packed 2-, 4-, and 8-bit
embedding tables. The C++ backend fixtures retain packed weights: neither
runtime receives an expanded floating-point checkpoint. Check numerical parity
before timing, and feed backend tensors directly to onnx-light-cpu.
"""

import os
import statistics
import time

import numpy as np
import onnxruntime

from onnx_light.onnx.backend import TestMode, collect_test_cases_by_name
from onnx_light.onnx.helper import tensor_dtype_to_np_dtype
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import register_backend_test_cases, register_kernels

register_kernels()
register_backend_test_cases()


def to_numpy(tensor):
    """Views a backend fixture as an ORT input."""
    return np.frombuffer(
        tensor.raw_data(), dtype=tensor_dtype_to_np_dtype(int(tensor.data_type))
    ).reshape(tuple(tensor.shape))


def measure(session, feeds, repeat):
    """Returns the median of a fixed, bounded number of warmed calls."""
    session.run(None, feeds)
    samples = []
    for _ in range(repeat):
        begin = time.perf_counter()
        session.run(None, feeds)
        samples.append(time.perf_counter() - begin)
    return statistics.median(samples)


repeat = 2 if os.environ.get("UNITTEST_GOING") else 30
cases = collect_test_cases_by_name(
    "^test_cpu_gather_block_quantized_.*_benchmark$",
    mode=TestMode.BENCHMARK,
    generate_benchmark_expected_outputs=False,
)
assert cases, "GatherBlockQuantized backend benchmark fixtures must be built."
for case in cases:
    names = [value.name for value in case.model.graph.input]
    cpu_feeds = dict(zip(names, case.data_sets[0].inputs, strict=True))
    ort_feeds = {name: to_numpy(value) for name, value in cpu_feeds.items()}
    cpu = ReferenceEvaluator(
        case.model, cpu_execution={"num_threads": 1, "affinity_policy": "none"}
    )
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    ort = onnxruntime.InferenceSession(
        case.model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    np.testing.assert_array_equal(cpu.run(None, cpu_feeds)[0], ort.run(None, ort_feeds)[0])
    print(
        f"{case.name}: onnx-light-cpu={measure(cpu, cpu_feeds, repeat) * 1e6:.1f} us, "
        f"ORT={measure(ort, ort_feeds, repeat) * 1e6:.1f} us"
    )
