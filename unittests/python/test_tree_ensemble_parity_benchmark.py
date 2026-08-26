from unittest.mock import patch

from tools.benchmark_tree_ensemble_parity import (
    BENCHMARK_GRID_CASES,
    PRIORITY_CASES,
    TREE_ENSEMBLE_GRID_BATCHES,
    TREE_ENSEMBLE_GRID_FEATURES,
    TREE_ENSEMBLE_GRID_TREES,
    TreeEnsembleCase,
    measure_one,
    render_comparison_table,
    repeat_count,
    summarize,
)


def test_benchmark_grid_crosses_trees_features_and_batches():
    dimensions = {(case.trees, case.features, case.rows) for case in BENCHMARK_GRID_CASES}

    assert dimensions == {
        (trees, features, batch)
        for trees in TREE_ENSEMBLE_GRID_TREES
        for features in TREE_ENSEMBLE_GRID_FEATURES
        for batch in TREE_ENSEMBLE_GRID_BATCHES
    }
    assert 1 in TREE_ENSEMBLE_GRID_BATCHES
    assert max(TREE_ENSEMBLE_GRID_FEATURES) >= 4096
    assert max(TREE_ENSEMBLE_GRID_TREES) >= 10000
    assert len(TREE_ENSEMBLE_GRID_FEATURES) > 1
    assert len(TREE_ENSEMBLE_GRID_BATCHES) > 1
    assert len(TREE_ENSEMBLE_GRID_TREES) > 1
    assert {(case.depth, case.dtype, case.outputs) for case in BENCHMARK_GRID_CASES} == {
        (4, "float32", 1)
    }


def _result(name, task, dtype, speedup, rows=32):
    return {
        "name": name,
        "task": task,
        "dtype": dtype,
        "rows": rows,
        "speedup": speedup,
        "cpu_median_seconds": 1.0,
        "correct": True,
        "preparation_passed": True,
        "workspace_passed": True,
    }


def test_priority_corpus_covers_final_gate_dimensions():
    assert {case.task for case in PRIORITY_CASES} == {"regression", "classification"}
    assert {case.dtype for case in PRIORITY_CASES} == {"float32", "float64"}
    assert {case.rows == 1 for case in PRIORITY_CASES} == {False, True}
    assert min(case.trees for case in PRIORITY_CASES) == 1
    assert max(case.trees for case in PRIORITY_CASES) >= 1024
    assert min(case.depth for case in PRIORITY_CASES) == 1
    assert max(case.depth for case in PRIORITY_CASES) >= 10
    assert any(case.branch_distribution == "mixed" for case in PRIORITY_CASES)
    assert any(case.membership for case in PRIORITY_CASES)
    assert any(case.outputs > 2 for case in PRIORITY_CASES)
    assert {case.labels for case in PRIORITY_CASES} >= {"zero_based", "int64", "string"}
    assert {case.transform for case in PRIORITY_CASES} >= {
        "NONE",
        "LOGISTIC",
        "SOFTMAX",
        "SOFTMAX_ZERO",
    }


def test_repeat_count_is_bounded():
    case = TreeEnsembleCase("case", "regression", "float32", 1, 1, 1, 1, 1)
    assert repeat_count(case, 7, 31) == 31
    large = TreeEnsembleCase("large", "regression", "float32", 1024, 1024, 10, 1, 1)
    assert repeat_count(large, 7, 31) == 7


def test_measure_one_bounds_warmup_and_measurement_separately():
    calls = []

    def measured():
        calls.append(None)

    clock = iter((0, 1_100_000_000, 2_000_000_000, 3_100_000_000))
    with patch(
        "tools.benchmark_tree_ensemble_parity.time.perf_counter_ns",
        side_effect=clock,
    ):
        samples = measure_one(measured, repeat=5, warmup=1, max_duration=1.0)

    assert samples == [1.1]
    assert len(calls) == 2
    assert len(calls) == 2


def test_summary_enforces_both_tasks_dtypes_and_single_row_baseline():
    results = [
        _result("reg32", "regression", "float32", 1.1, 1),
        _result("reg64", "regression", "float64", 1.0, 1),
        _result("cls32", "classification", "float32", 1.2, 1),
        _result("cls64", "classification", "float64", 1.0, 1),
    ]
    baseline = {row["name"]: 1.0 for row in results}

    report = summarize(results, baseline)

    assert report["passed"]
    assert report["single_row"]["passed"]


def test_summary_rejects_a_slow_case_even_when_median_passes():
    results = [
        _result("reg32_fast", "regression", "float32", 1.2),
        _result("reg32", "regression", "float32", 0.89),
        _result("reg64", "regression", "float64", 1.0),
        _result("cls32", "classification", "float32", 1.0),
        _result("cls64", "classification", "float64", 1.0),
    ]
    baseline = {row["name"]: 1.0 for row in results}

    report = summarize(results, baseline)

    assert report["groups"]["regression_float32"]["median_speedup"] > 1.0
    assert not report["groups"]["regression_float32"]["passed"]
    assert not report["passed"]


def test_comparison_table_reports_both_engines():
    table = render_comparison_table(
        [
            {
                **_result("forest", "regression", "float32", 2.0),
                "features": 32,
                "trees": 81,
                "depth": 4,
                "cpu_median_seconds": 0.001,
                "ort_median_seconds": 0.002,
            }
        ]
    )

    assert "onnx-light-cpu (us)" in table
    assert "onnxruntime (us)" in table
    assert "features" in table
    assert "forest" in table
    assert "2.000x" in table
