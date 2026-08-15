from tools.benchmark_gemm_parity import (
    GemmCase,
    render_comparison_table,
    repeat_count,
    summarize,
)


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


def test_comparison_table_reports_both_engines_side_by_side():
    table = render_comparison_table(
        [
            {
                "name": "square_1024",
                "dtype": "float32",
                "m": 1024,
                "n": 1024,
                "k": 1024,
                "cpu_median_seconds": 2 * 1024**3 / 1e9 / 100.0,
                "ort_median_seconds": 2 * 1024**3 / 1e9 / 200.0,
                "speedup": 0.5,
            }
        ]
    )

    lines = table.splitlines()
    assert "onnx-light-cpu (GFLOP/s)" in lines[0]
    assert "onnxruntime (GFLOP/s)" in lines[0]
    data = lines[2]
    assert "square_1024" in data
    assert "100.00" in data
    assert "200.00" in data
    assert "0.500x" in data
