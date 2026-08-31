import math
from unittest import TestCase

from tools.benchmark_bias_gelu_parity import (
    CASES,
    _parse_cpu_list,
    measure_alternating,
    parse_args,
    percentile,
    summarize,
)


def test_matrix_covers_vector_boundaries_and_parallel_threshold():
    cases = dict(CASES)

    assert cases["avx2_before"][-1] == 7
    assert cases["avx2_aligned"][-1] == 8
    assert cases["avx2_after"][-1] == 9
    assert cases["avx512_before"][-1] == 15
    assert cases["avx512_aligned"][-1] == 16
    assert cases["avx512_after"][-1] == 17
    assert math.prod(cases["threshold_before"]) * 4 < 256 * 1024
    assert math.prod(cases["threshold_at"]) * 4 == 256 * 1024
    assert math.prod(cases["threshold_after"]) * 4 > 256 * 1024


def test_candidate_order_samples_and_tail_are_retained():
    order = []

    def cpu():
        order.append("cpu")

    def ort():
        order.append("ort")

    samples, candidate_order = measure_alternating(
        (cpu, ort), repeat=3, warmup=0, max_repeat_time=1.0
    )

    assert order == ["cpu", "ort", "ort", "cpu", "cpu", "ort"]
    assert tuple(map(len, samples)) == (3, 3)
    assert candidate_order[1] == ["onnxruntime", "onnx-light-cpu"]
    assert math.isclose(percentile([1.0, 2.0, 3.0], 0.9), 2.8)
    with TestCase().assertRaises(ValueError):
        measure_alternating((cpu,), repeat=1, warmup=0, max_repeat_time=1.0)


def test_summary_requires_median_and_per_case_tail_parity():
    passing = [
        {"speedup": 0.9, "tail_speedup": 0.9},
        {"speedup": 1.1, "tail_speedup": 1.1},
    ]

    assert summarize(passing)["passed"]
    for field in ("speedup", "tail_speedup"):
        failing = [dict(row) for row in passing]
        failing[0][field] = 0.89
        assert not summarize(failing)["passed"]


def test_defaults_match_single_thread_reproducibility_contract():
    args = parse_args(["--enforce"])

    assert args.threads == 1
    assert args.warmup > 0
    assert args.enforce


def test_cpu_affinity_ranges_are_validated():
    assert _parse_cpu_list("0-2,4") == {0, 1, 2, 4}
    with TestCase().assertRaises(ValueError):
        _parse_cpu_list("3-1")
