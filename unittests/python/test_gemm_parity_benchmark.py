from tools.benchmark_gemm_parity import GemmCase, repeat_count, summarize


def test_repeat_count_is_bounded():
    assert repeat_count(GemmCase("tiny", 1, 1, 1), 7, 31) == 31
    assert repeat_count(GemmCase("large", 1024, 1024, 1024), 7, 31) == 7


def test_summary_applies_gate_per_dtype():
    report = summarize(
        [
            {"dtype": "float32", "speedup": 1.2},
            {"dtype": "float32", "speedup": 0.9},
            {"dtype": "float64", "speedup": 1.1},
            {"dtype": "float64", "speedup": 1.0},
        ]
    )

    assert report["passed"]
    assert report["by_dtype"]["float32"]["median_speedup"] == 1.05
    assert report["by_dtype"]["float32"]["minimum_speedup"] == 0.9


def test_summary_rejects_a_slow_priority_case():
    report = summarize(
        [
            {"dtype": "float32", "speedup": 1.5},
            {"dtype": "float32", "speedup": 0.89},
        ]
    )

    assert not report["passed"]
