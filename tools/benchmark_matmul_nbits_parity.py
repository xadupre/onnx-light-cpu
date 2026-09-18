#!/usr/bin/env python3
"""Configuration-derived Muse-Glimmer INT4 projections with synthetic packed weights.

Run ``python -m tools.benchmark_matmul_nbits_parity --projection k --m 8``.
Use ``--full`` for all projections, M=1/8/128 and FP32/FP16/BF16. Large
projections are deliberately opt-in. No floating-point weight matrix is built.
These are reference-configuration dimensions, not an audit of an exported Muse graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

from tools.benchmark_gemm_parity import measure_alternating

HIDDEN = 6656
INTERMEDIATE = 19968
QUERY = 4096
KEY_VALUE = 256
VOCABULARY = 202048
BLOCK_SIZE = 32
DTYPES = ("float32", "float16", "bfloat16")
ROWS = (1, 8, 128)


def projections():
    """Return separate projections, not fictitious fused QKV/gate-up shapes."""
    return {
        "q": (HIDDEN, QUERY),
        "k": (HIDDEN, KEY_VALUE),
        "v": (HIDDEN, KEY_VALUE),
        "attention_output": (QUERY, HIDDEN),
        "gate": (HIDDEN, INTERMEDIATE),
        "up": (HIDDEN, INTERMEDIATE),
        "down": (INTERMEDIATE, HIDDEN),
        "lm_head": (HIDDEN, VOCABULARY),
    }


def memory_accounting(m, k, n, dtype, block_size=BLOCK_SIZE):
    """Analytical payloads only: not allocator peaks, process RSS or measurements."""
    itemsize = 4 if dtype == "float32" else 2
    blocks = (k + block_size - 1) // block_size
    return {
        "kind": "analytical; excludes runtime/ORT arenas, model copies and thread stacks",
        "packed_initializer_bytes": n * blocks * (block_size // 2),
        "scale_initializer_bytes": n * blocks * itemsize,
        "input_bytes": m * k * itemsize,
        "output_bytes": m * n * itemsize,
        "avoided_full_float_weight_bytes": k * n * 4,
        "kernel_panel_workspace_bytes_per_worker": (32 * 32 + 8 * 32 + 8 * 32) * 4,
        "kernel_full_weight_copy_bytes_per_run": 0,
        "kernel_full_input_conversion_copy_bytes_per_run": 0,
        "kernel_activation_panel_write_bytes_per_run": m * k * ((n + 31) // 32) * 4,
        "kernel_dequantized_panel_write_bytes_upper_bound": (
            ((m + 7) // 8) * ((n + 31) // 32) * ((k + 31) // 32) * 32 * 32 * 4
        ),
        "copy_scope": "optimized INT4 kernel only; session and Python boundary copies excluded",
    }


def make_inputs(m, k, n, dtype, block_size=BLOCK_SIZE, seed=548):
    import ml_dtypes
    import numpy as np

    storage = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": ml_dtypes.bfloat16,
    }[dtype]
    random = np.random.default_rng(seed)
    blocks = (k + block_size - 1) // block_size
    a = random.uniform(-0.5, 0.5, size=(m, k)).astype(storage)
    # Generate packed bytes directly: no temporary [K,N] or [N,K] float weights.
    packed = random.integers(0, 256, size=(n, blocks, block_size // 2), dtype=np.uint8)
    scales = random.uniform(0.001, 0.02, size=(n, blocks)).astype(storage)
    return a, packed, scales


def make_model(m, k, n, dtype, packed, scales, block_size=BLOCK_SIZE):
    from onnx_light.onnx import TensorProto, helper, numpy_helper

    element = {
        "float32": TensorProto.FLOAT,
        "float16": TensorProto.FLOAT16,
        "bfloat16": TensorProto.BFLOAT16,
    }[dtype]
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "MatMulNBits",
                    ["A", "B", "scales"],
                    ["Y"],
                    domain="com.microsoft",
                    K=k,
                    N=n,
                    bits=4,
                    block_size=block_size,
                    accuracy_level=0,
                )
            ],
            "muse_glimmer_int4_projection",
            [helper.make_tensor_value_info("A", element, [m, k])],
            [helper.make_tensor_value_info("Y", element, [m, n])],
            initializer=[
                numpy_helper.from_array(packed, name="B"),
                numpy_helper.from_array(scales, name="scales"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )


def fingerprint(*arrays):
    """Hash buffers without retaining full-sized comparison copies."""
    return tuple(
        hashlib.sha256(memoryview(array.view("uint8")).cast("B")).hexdigest() for array in arrays
    )


def run_case(name, m, k, n, dtype, *, threads=1, repeat=3, warmup=1):
    """Prepare once, check independent ORT parity, then reuse both sessions."""
    import numpy as np
    import onnxruntime as ort
    from onnx_light.onnx.reference import ReferenceEvaluator
    from onnx_light_cpu import (
        clear_used_kernel_names,
        detect_simd_level,
        register_kernel_for_session,
        set_kernel_usage_recording,
        used_kernel_names,
    )

    started = time.perf_counter()
    a, packed, scales = make_inputs(m, k, n, dtype)
    original = fingerprint(a, packed, scales)
    model = make_model(m, k, n, dtype, packed, scales)
    serialized = model.SerializeToString()
    original_model = hashlib.sha256(serialized).hexdigest()
    input_preparation = time.perf_counter() - started
    started = time.perf_counter()
    cpu = ReferenceEvaluator(serialized, cpu_execution={"num_threads": threads})
    register_kernel_for_session(cpu, "com.microsoft", "MatMulNBits")
    cpu_preparation = time.perf_counter() - started
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    started = time.perf_counter()
    oracle_feeds = {"A": a}
    oracle_kind = "native"
    bf16_rejection = None
    if dtype == "bfloat16":
        # Probe the real schema/kernel rather than masking unrelated session errors.
        try:
            oracle = ort.InferenceSession(
                serialized, sess_options=options, providers=["CPUExecutionProvider"]
            )
        except (
            ort.capi.onnxruntime_pybind11_state.InvalidGraph,
            ort.capi.onnxruntime_pybind11_state.NotImplemented,
        ) as error:
            bf16_rejection = str(error)
            unsupported_kernel = (
                isinstance(error, ort.capi.onnxruntime_pybind11_state.NotImplemented)
                and "MatMulNBits" in bf16_rejection
            )
            if not unsupported_kernel and "bfloat16" not in bf16_rejection.lower():
                raise
            oracle_kind = "float32 with BF16-rounded inputs/scales and final BF16 cast"
            oracle_feeds = {"A": a.astype(np.float32)}
            oracle_model = make_model(m, k, n, "float32", packed, scales.astype(np.float32))
            serialized = oracle_model.SerializeToString()
            oracle = ort.InferenceSession(
                serialized, sess_options=options, providers=["CPUExecutionProvider"]
            )
    else:
        oracle = ort.InferenceSession(
            serialized, sess_options=options, providers=["CPUExecutionProvider"]
        )
    ort_preparation = time.perf_counter() - started
    cpu_feeds = {"A": a}

    def cpu_run():
        return cpu.run(None, cpu_feeds)[0]

    def ort_run():
        return oracle.run(None, oracle_feeds)[0]

    set_kernel_usage_recording(cpu, True)
    clear_used_kernel_names(cpu)
    actual = cpu_run()
    if not any("MatMulNBits" in kernel for kernel in used_kernel_names(cpu)):
        raise AssertionError("MatMulNBits did not dispatch the registered CPU kernel")
    set_kernel_usage_recording(cpu, False)
    expected = ort_run().astype(a.dtype).astype(np.float32)
    rtol, atol = {
        "float32": (3e-4, 2e-4),
        "float16": (1e-2, 1e-2),
        "bfloat16": (2e-2, 3e-2),
    }[dtype]
    np.testing.assert_allclose(actual.astype(np.float32), expected, rtol=rtol, atol=atol)
    np.testing.assert_array_equal(cpu_run(), actual)
    samples = measure_alternating((cpu_run, ort_run), repeat, warmup)
    np.testing.assert_array_equal(cpu_run(), actual)
    assert fingerprint(a, packed, scales) == original, "Input/packed constants were modified"
    assert hashlib.sha256(model.SerializeToString()).hexdigest() == original_model
    return {
        "projection": name,
        "shape_source": "reference configuration; synthetic constants, not audited graph weights",
        "m": m,
        "k": k,
        "n": n,
        "dtype": dtype,
        "block_size": BLOCK_SIZE,
        "accuracy_level": 0,
        "threads": threads,
        "simd_level": str(detect_simd_level()),
        "warmup": warmup,
        "repeat": repeat,
        "ort_version": ort.__version__,
        "oracle": oracle_kind,
        "ort_timing_scope": (
            "FP32 session.run; final BF16 reference cast excluded"
            if bf16_rejection
            else "native session.run"
        ),
        "ort_bfloat16_rejection": bf16_rejection,
        "input_and_model_preparation_seconds": input_preparation,
        "cpu_session_preparation_seconds": cpu_preparation,
        "ort_session_preparation_seconds": ort_preparation,
        "ort_preparation_includes_bf16_probe_and_conversion": bool(bf16_rejection),
        "cpu_samples_seconds": samples[0],
        "ort_samples_seconds": samples[1],
        "cpu_median_seconds": statistics.median(samples[0]),
        "ort_median_seconds": statistics.median(samples[1]),
        "speedup": statistics.median(samples[1]) / statistics.median(samples[0]),
        "timing_scope": "steady-state session.run with output allocation; preparation excluded",
        "max_absolute_error": float(np.max(np.abs(actual.astype(np.float32) - expected))),
        "constants_unchanged": True,
        "memory": memory_accounting(m, k, n, dtype),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", action="append", choices=tuple(projections()))
    parser.add_argument("--m", action="append", type=int, choices=ROWS)
    parser.add_argument("--dtype", action="append", choices=DTYPES)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.repeat < 1 or args.warmup < 0:
        parser.error("threads/repeat must be positive and warmup nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    names = args.projection or (tuple(projections()) if args.full else ("k",))
    rows = args.m or (ROWS if args.full else (1,))
    dtypes = args.dtype or (DTYPES if args.full else ("float32",))
    results = []
    for name in names:
        for m in rows:
            for dtype in dtypes:
                results.append(
                    run_case(
                        name,
                        m,
                        *projections()[name],
                        dtype,
                        threads=args.threads,
                        repeat=args.repeat,
                        warmup=args.warmup,
                    )
                )
                print(f"Parity passed: {name}, M={m}, {dtype}", file=sys.stderr)
                if args.output:
                    args.output.write_text(
                        json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8"
                    )
    text = json.dumps({"results": results}, indent=2)
    print(text)


if __name__ == "__main__":
    main()
