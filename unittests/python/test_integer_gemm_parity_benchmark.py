from tools.benchmark_integer_gemm_parity import (
    PRIORITY_CASES,
    IntegerGemmCase,
    parse_args,
    repeat_count,
    summarize,
)


def test_priority_corpus_covers_integer_shapes():
    """Checks the required integer performance shapes."""
    names = {case.name for case in PRIORITY_CASES}
    assert {"tiny", "direct", "square_512", "skinny_m", "skinny_n", "large_k"} <= names


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
