from tools.benchmark_integer_gemm_parity import (
    PRIORITY_CASES,
    IntegerGemmCase,
    parse_args,
    repeat_count,
    summarize,
    _build_case,
)


def test_priority_corpus_covers_integer_shapes():
    """Checks the required integer performance shapes."""
    names = {case.name for case in PRIORITY_CASES}
    assert {"tiny", "direct", "square_512", "skinny_m", "skinny_n", "large_k"} <= names
    assert {"skinny_m_4096", "skinny_m_mr", "skinny_m_tail"} <= names


def test_integer_plan_diagnostics():
    """Diagnostics use the production planner rather than inferred ISA names."""
    from onnx_light_cpu import has_backend_test_cases

    if not has_backend_test_cases():
        return
    from onnx_light_cpu.onnx_py._cpuregister import integer_matmul_plan

    assert integer_matmul_plan(3, 4096, 4096) == "packed"
    assert integer_matmul_plan(2, 31, 128) == "packed"
    assert integer_matmul_plan(1, 4096, 4096) in {
        "packed",
        "no-pack-avx2",
        "no-pack-vnni",
    }
    assert integer_matmul_plan(1, 4096, 4096) == integer_matmul_plan(2, 4096, 4096)


def test_public_skinny_case_omits_zero_points():
    """Keep the acceptance benchmark comparable to the public backend case."""
    import numpy
    from onnx_light.onnx import ModelProto

    case = next(case for case in PRIORITY_CASES if case.name == "skinny_m_4096")
    assert (case.m, case.n, case.k) == (1, 4096, 4096)
    assert case.a_zero_point is None and case.b_zero_point is None
    model_bytes, feeds = _build_case(
        IntegerGemmCase("small", 1, 8, 8, None, None), numpy.random.default_rng(0)
    )
    model = ModelProto()
    model.ParseFromString(model_bytes)
    assert list(model.graph.node[0].input) == ["A", "B"]
    assert set(feeds) == {"A", "B"}


def test_repeat_count_stays_within_bounds():
    """Checks operation-scaled repeat bounds."""
    assert repeat_count(IntegerGemmCase("tiny", 1, 1, 1), 7, 31) == 31
    assert repeat_count(IntegerGemmCase("large", 1024, 1024, 1024), 7, 31) == 7


def test_summary_enforces_median_and_minimum():
    """Checks the integer parity thresholds."""
    passing = summarize([{"speedup": 0.95}, {"speedup": 1.05}, {"speedup": 1.2}])
    assert passing["passed"]
    assert passing["median_speedup"] == 1.05
    assert passing["minimum_speedup"] == 0.95
    assert not summarize([{"speedup": 0.89}, {"speedup": 1.1}, {"speedup": 1.2}])["passed"]


def test_parse_args_selects_cases_and_output():
    """Checks command-line selections."""
    args = parse_args(["--threads", "4", "--case", "direct", "--output", "result.json"])
    assert args.threads == 4
    assert args.case == ["direct"]
    assert str(args.output) == "result.json"
