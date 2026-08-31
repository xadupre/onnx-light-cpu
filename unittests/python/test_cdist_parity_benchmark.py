import math
from unittest import TestCase

from tools.benchmark_cdist_parity import (
    CASES,
    DTYPES,
    METRICS,
    measure_alternating,
    numerical_tolerance,
    parse_args,
    summarize,
)


def test_matrix_covers_contract_boundaries_and_latency_threshold():
    cases = {case.name: case for case in CASES}

    assert set(DTYPES) == {"float32", "float64"}
    assert set(METRICS) == {"sqeuclidean", "euclidean"}
    assert cases["ort_float_vector"].input_kind == "ort_float"
    assert cases["ort_double_vector"].input_kind == "ort_double"
    assert cases["empty_a"].m == 0
    assert cases["empty_b"].k == 0
    assert cases["singleton"].work == 1
    assert cases["vector_before"].n == 15
    assert cases["vector_aligned"].n == 16
    assert cases["vector_after"].n == 17
    for dtype_size, threshold in ((4, 128 * 1024), (8, 256 * 1024)):
        assert cases["threshold_before"].m * cases["threshold_before"].n * dtype_size < threshold
        assert cases["threshold_at"].m * cases["threshold_at"].n * dtype_size == threshold
        assert cases["threshold_after"].m * cases["threshold_after"].n * dtype_size > threshold


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
    with TestCase().assertRaises(ValueError):
        measure_alternating((cpu,), repeat=1, warmup=0, max_repeat_time=1.0)


def test_numerical_bound_accounts_for_expanded_formula_cancellation():
    assert math.isclose(numerical_tolerance("float32", "sqeuclidean", 1, 1), 2.0**-20)
    assert math.isclose(numerical_tolerance("float32", "euclidean", 1, 1), math.sqrt(2.0**-20))
    assert numerical_tolerance("float32", "sqeuclidean", 65, 1024) > 1
    assert numerical_tolerance("float64", "sqeuclidean", 65, 1024) < 1e-6


def test_summary_requires_full_dtype_matrix_and_per_case_tail_parity():
    passing = [
        {"dtype": dtype, "speedup": speedup, "tail_speedup": speedup}
        for dtype in DTYPES
        for speedup in (0.9, 1.1)
    ]

    assert summarize(passing)["passed"]
    failing = [dict(row) for row in passing]
    failing[0]["tail_speedup"] = 0.89
    assert not summarize(failing)["passed"]
    assert not summarize([row for row in passing if row["dtype"] == "float32"])["passed"]


def test_defaults_match_single_thread_reproducibility_contract():
    args = parse_args(["--enforce"])

    assert args.threads == 1
    assert args.warmup > 0
    assert args.enforce
