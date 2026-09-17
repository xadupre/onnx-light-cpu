from tools.benchmark_gemm_parity import (
    MATMUL_PRIORITY_CASES,
    PARITY_DTYPES,
    PRIORITY_CASES,
    GemmCase,
    _build_case,
    _output_shape,
    parse_args,
    render_comparison_table,
    repeat_count,
    summarize,
)


def test_default_thread_policy_is_not_overridden():
    assert parse_args([]).threads is None
    assert parse_args(["--threads", "4"]).threads == 4


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


def test_float16_is_a_parity_dtype():
    import numpy as np

    model, feeds = _build_case(GemmCase("float16", 2, 3, 4), "float16", np.random.default_rng(0))

    assert "float16" in PARITY_DTYPES
    assert model
    assert feeds["A"].dtype == np.float16
    assert feeds["B"].dtype == np.float16


def test_matmul_corpus_covers_batches_broadcasting_and_vectors():
    cases = {case.name: case for case in MATMUL_PRIORITY_CASES}
    names = set(cases)
    assert {"batched_direct", "broadcast_b", "vector_matrix", "matrix_vector"} <= names
    assert all(case.operator == "MatMul" for case in MATMUL_PRIORITY_CASES)
    assert any(case.batch_shape for case in MATMUL_PRIORITY_CASES)
    assert any(case.b_batch_shape for case in MATMUL_PRIORITY_CASES)
    assert any(case.vector_a or case.vector_b for case in MATMUL_PRIORITY_CASES)
    assert (cases["large_k"].m, cases["large_k"].n, cases["large_k"].k) == (32, 32, 8192)


def test_matmul_output_shapes_follow_vector_and_batch_broadcast_semantics():
    assert _output_shape(
        GemmCase("vector_matrix", 1, 128, 128, operator="MatMul", vector_a=True)
    ) == (128,)
    assert _output_shape(
        GemmCase("matrix_vector", 128, 1, 128, operator="MatMul", vector_b=True)
    ) == (128,)
    assert _output_shape(
        GemmCase(
            "broadcast",
            8,
            16,
            4,
            operator="MatMul",
            batch_shape=(3, 1),
            b_batch_shape=(2,),
        )
    ) == (3, 2, 8, 16)


def test_gemm_corpus_covers_large_k_1024_benchmark():
    cases = {case.name: case for case in PRIORITY_CASES}

    assert (cases["large_k_1024"].m, cases["large_k_1024"].n, cases["large_k_1024"].k) == (
        32,
        32,
        1024,
    )


def test_transformer_projection_covers_dynamic_and_constant_b():
    cases = {case.name: case for case in PRIORITY_CASES}

    assert not cases["transformer_projection_dynamic"].constant_b
    assert cases["transformer_projection"].constant_b
