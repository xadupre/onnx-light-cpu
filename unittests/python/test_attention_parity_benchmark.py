import math

import pytest

from tools.benchmark_attention_parity import (
    _expand_cpu_part,
    measure_alternating,
    parse_args,
    parse_case_name,
    summarize,
)


def test_backend_case_name_is_traceable_and_complete():
    case = parse_case_name(
        "test_cpu_attention_opset24_rank3_gqa_q8_kv1024_hd128_causal_nonpad_bfloat16_benchmark"
    )

    assert case == {
        "opset": 24,
        "layout": "rank3",
        "geometry": "gqa",
        "q_length": 8,
        "kv_length": 1024,
        "head_dim": 128,
        "mask": "causal",
        "cache": "nonpad",
        "dtype": "bfloat16",
    }


def test_candidate_order_and_raw_samples_are_retained():
    order = []

    def cpu():
        order.append("cpu")

    def ort():
        order.append("ort")

    samples, candidate_order = measure_alternating((cpu, ort), repeat=3, warmup=0)

    assert order == ["cpu", "ort", "ort", "cpu", "cpu", "ort"]
    assert tuple(map(len, samples)) == (3, 3)
    assert candidate_order == [
        ["onnx-light-cpu", "onnxruntime"],
        ["onnxruntime", "onnx-light-cpu"],
        ["onnx-light-cpu", "onnxruntime"],
    ]


def test_summary_applies_per_type_parity_and_memory_gates():
    results = []
    for dtype in ("float32", "float16", "bfloat16"):
        results.extend(
            [
                {"dtype": dtype, "speedup": 0.9, "memory_gate_passed": True},
                {"dtype": dtype, "speedup": 1.1, "memory_gate_passed": True},
            ]
        )

    summary = summarize(results)

    assert summary["passed"]
    assert all(
        math.isclose(dtype_summary["median_speedup"], 1.0)
        for dtype_summary in summary["by_dtype"].values()
    )


@pytest.mark.parametrize(
    "field,value",
    [("speedup", 0.89), ("memory_gate_passed", False)],
)
def test_summary_rejects_a_priority_regression(field, value):
    results = [
        {"dtype": dtype, "speedup": 1.0, "memory_gate_passed": True}
        for dtype in ("float32", "float16", "bfloat16")
    ]
    results[0][field] = value

    assert not summarize(results)["passed"]


def test_equal_thread_runs_cannot_enforce_default_policy_gate():
    with pytest.raises(SystemExit):
        parse_args(["--threads", "2", "--enforce"])


def test_cpu_ranges_are_validated():
    assert list(_expand_cpu_part("2-4")) == [2, 3, 4]
    with pytest.raises(ValueError):
        _expand_cpu_part("4-2")
