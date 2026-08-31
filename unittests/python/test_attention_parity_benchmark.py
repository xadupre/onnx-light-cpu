import math
from unittest import TestCase

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
        "family": None,
        "opset": 24,
        "layout": "rank3",
        "geometry": "gqa",
        "q_length": 8,
        "kv_length": 1024,
        "head_dim": 128,
        "q_heads": None,
        "kv_heads": None,
        "mask": "causal",
        "cache": "nonpad",
        "dtype": "bfloat16",
    }


def test_backend_case_name_decodes_named_model_families_and_head_counts():
    case = parse_case_name(
        "test_cpu_attention_llm_qwen3_8b_opset24_rank4_gqa_q1_kv4096_hd128"
        "_qh32_kvh8_causal_internal_cache_float16_benchmark"
    )

    assert case == {
        "family": "llm_qwen3_8b",
        "opset": 24,
        "layout": "rank4",
        "geometry": "gqa",
        "q_length": 1,
        "kv_length": 4096,
        "head_dim": 128,
        "q_heads": 32,
        "kv_heads": 8,
        "mask": "causal",
        "cache": "internal_cache",
        "dtype": "float16",
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


def test_summary_rejects_a_priority_regression():
    for field, value in (
        ("speedup", 0.89),
        ("tail_speedup", 0.89),
        ("memory_gate_passed", False),
    ):
        results = [
            {"dtype": dtype, "speedup": 1.0, "memory_gate_passed": True}
            for dtype in ("float32", "float16", "bfloat16")
        ]
        results[0][field] = value

        assert not summarize(results)["passed"]


def test_benchmark_defaults_to_identical_single_thread_execution():
    args = parse_args(["--enforce"])
    assert args.threads == 1
    assert args.enforce


def test_cpu_ranges_are_validated():
    assert list(_expand_cpu_part("2-4")) == [2, 3, 4]
    with TestCase().assertRaises(ValueError):
        _expand_cpu_part("4-2")
