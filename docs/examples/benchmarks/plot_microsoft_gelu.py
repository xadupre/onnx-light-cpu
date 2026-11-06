"""
ONNX Runtime contrib Gelu
========================

Qwen3.5's vision patch merger uses ``com.microsoft::Gelu`` rather than
standard ONNX Gelu. This example validates both float32 and float16 against
ONNX Runtime before timing the portable CPU kernel. Backend tensors are fed
directly to onnx-light during timing, avoiding NumPy adaptation overhead.
"""

import os
from functools import partial
from timeit import timeit

import numpy as np
import onnxruntime
from onnx_light.onnx.backend import TestMode, collect_test_cases
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import register_backend_test_cases, register_kernels_global
from onnx_light_cpu._benchmark import _prepare_feeds

register_backend_test_cases()
register_kernels_global()
small = os.environ.get("UNITTEST_GOING", "0") == "1"
mode = TestMode.TEST if small else TestMode.BENCHMARK
repeat = 2 if small else 20
cases = collect_test_cases("Gelu", mode=mode)

for dtype in ("float32", "float16"):
    shape_tag = "2_32" if small else "4096_128"
    suffix = "" if small else "_benchmark"
    name = f"test_cpu_microsoft_gelu_{shape_tag}_{dtype}{suffix}"
    case = next(case for case in cases if case.name == name)
    model = case.model
    names = [value.name for value in model.graph.input]
    native_feeds, numpy_feeds = _prepare_feeds(names, case.data_sets)
    cpu = ReferenceEvaluator(model, cpu_execution={"num_threads": 1})
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    ort = onnxruntime.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    actual = cpu.run(None, numpy_feeds[0])[0]
    expected = ort.run(None, numpy_feeds[0])[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
    cpu_seconds = timeit(partial(cpu.run, None, native_feeds[0]), number=repeat) / repeat
    ort_seconds = timeit(partial(ort.run, None, numpy_feeds[0]), number=repeat) / repeat
    print(f"{dtype}: onnx-light-cpu={cpu_seconds:.6g}s, ORT={ort_seconds:.6g}s")
