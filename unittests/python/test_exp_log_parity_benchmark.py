import math

from tools.benchmark_exp_log_parity import _parse_cpu_list, measure_alternating, summarize


def test_cpu_list_parses_ranges_and_individual_processors():
    assert _parse_cpu_list("0-2,4") == {0, 1, 2, 4}


def test_measurement_order_alternates_candidates():
    order = []

    def first():
        order.append("cpu")

    def second():
        order.append("ort")

    samples = measure_alternating((first, second), repeat=3, warmup=0)

    assert order == ["cpu", "ort", "ort", "cpu", "cpu", "ort"]
    assert tuple(map(len, samples)) == (3, 3)


def test_summary_applies_complete_parity_gate():
    results = [
        {"operator": "Abs", "size": 100, "speedup": 1.0},
        {"operator": "Abs", "size": 4_194_304, "speedup": 1.0},
        {"operator": "Exp", "size": 100, "speedup": 0.9},
        {"operator": "Exp", "size": 4_194_304, "speedup": 0.95},
        {"operator": "Log", "size": 100, "speedup": 1.1},
        {"operator": "Log", "size": 4_194_304, "speedup": 1.2},
    ]

    summary = summarize(results)

    assert summary["passed"]
    assert math.isclose(summary["median_speedup"], 1.0)
    assert summary["minimum_speedup"] == 0.9


def test_summary_rejects_large_model_below_threshold():
    results = [
        {"operator": "Abs", "size": 100, "speedup": 1.2},
        {"operator": "Abs", "size": 4_194_304, "speedup": 1.2},
        {"operator": "Exp", "size": 100, "speedup": 1.2},
        {"operator": "Exp", "size": 4_194_304, "speedup": 0.94},
        {"operator": "Log", "size": 100, "speedup": 1.2},
        {"operator": "Log", "size": 4_194_304, "speedup": 1.2},
    ]

    assert not summarize(results)["passed"]
