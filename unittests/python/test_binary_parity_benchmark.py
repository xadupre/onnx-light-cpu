import math

from tools.benchmark_binary_parity import (
    ELEMENT_COUNTS,
    ORT_UNSUPPORTED_SIGNATURES,
    PRIORITY_SIGNATURES,
    SHAPE_FAMILIES,
    THREAD_POLICIES,
    _parse_cpu_list,
    measure_alternating,
    parse_args,
    parse_case_name,
    summarize,
)


def test_binary_case_name_retains_gate_identity():
    case = parse_case_name(
        "test_cpu_mod_v13_general_float32xfloat32_to_float32_fmod1_n65536_benchmark"
    )
    assert case == {
        "operator": "Mod",
        "opset": 13,
        "shape_family": "general",
        "left_type": "float32",
        "right_type": "float32",
        "output_type": "float32",
        "attributes": "fmod1",
        "element_count": 65_536,
    }


def test_fixed_priority_matrix_is_complete():
    assert ELEMENT_COUNTS == (4_096, 65_536, 1_048_576, 4_194_304)
    assert SHAPE_FAMILIES == (
        "contiguous",
        "left_scalar",
        "right_scalar",
        "row",
        "per_channel",
        "outer",
        "general",
    )
    assert ("PRelu", "bfloat16") in PRIORITY_SIGNATURES
    assert ("And", "bool") in PRIORITY_SIGNATURES
    assert THREAD_POLICIES == ("1", "physical")


def test_raw_samples_and_alternating_order_are_retained():
    calls = []

    def cpu():
        calls.append("cpu")

    def ort():
        calls.append("ort")

    samples, order = measure_alternating((cpu, ort), repeat=3, warmup=0)
    assert calls == ["cpu", "ort", "ort", "cpu", "cpu", "ort"]
    assert tuple(map(len, samples)) == (3, 3)
    assert order[1] == ["onnxruntime", "onnx-light-cpu"]


def test_summary_enforces_global_median_and_minimum():
    results = [
        {
            "operator": operator,
            "left_type": dtype,
            "shape_family": family,
            "element_count": count,
            "thread_policy": policy,
            "speedup": 1.0,
            "cpu_p90_seconds": 1.0,
        }
        for operator, dtype in PRIORITY_SIGNATURES
        for family in SHAPE_FAMILIES
        for count in ELEMENT_COUNTS
        for policy in THREAD_POLICIES
    ]
    results[0]["speedup"] = 0.9
    results[1]["speedup"] = 1.1
    summary = summarize(results)
    assert summary["passed"]
    assert math.isclose(summary["median_speedup"], 1.0)
    assert summary["groups_passed"]
    assert summary["small_p90_passed"]

    results[0]["speedup"] = 0.89
    assert not summarize(results)["passed"]
    assert not summarize(results[1:])["matrix_complete"]


def test_summary_enforces_every_group_and_small_p90():
    results = [
        {
            "operator": operator,
            "left_type": dtype,
            "shape_family": family,
            "element_count": count,
            "thread_policy": policy,
            "speedup": 1.1,
            "cpu_p90_seconds": 1.0,
        }
        for operator, dtype in PRIORITY_SIGNATURES
        for family in SHAPE_FAMILIES
        for count in ELEMENT_COUNTS
        for policy in THREAD_POLICIES
    ]
    group = (
        results[0]["operator"],
        results[0]["left_type"],
        results[0]["shape_family"],
    )
    for result in results:
        if tuple(result[field] for field in ("operator", "left_type", "shape_family")) == group:
            result["speedup"] = 0.99
    summary = summarize(results)
    assert not summary["passed"]
    assert not summary["groups_passed"]

    for result in results:
        result["speedup"] = 1.1
        if result["thread_policy"] == "physical" and result["element_count"] == 4096:
            result["cpu_p90_seconds"] = 1.03
    summary = summarize(results)
    assert not summary["passed"]
    assert not summary["small_p90_passed"]
    assert summary["small_p90_regressions"]


def test_summary_only_accepts_known_ort_unsupported_signatures():
    results = [
        {
            "operator": operator,
            "left_type": dtype,
            "shape_family": family,
            "element_count": count,
            "thread_policy": policy,
            "speedup": 1.1,
            "cpu_p90_seconds": 1.0,
        }
        for operator, dtype in PRIORITY_SIGNATURES
        for family in SHAPE_FAMILIES
        for count in ELEMENT_COUNTS
        for policy in THREAD_POLICIES
    ]
    unsupported = [
        result
        for result in results
        if (result["operator"], result["left_type"]) in ORT_UNSUPPORTED_SIGNATURES
    ]
    results = [result for result in results if result not in unsupported]
    summary = summarize(results, unsupported)
    assert summary["passed"]
    assert not summary["unexpected_unsupported_cases"]

    unexpected = next(
        result
        for result in results
        if (result["operator"], result["left_type"]) not in ORT_UNSUPPORTED_SIGNATURES
    )
    results.remove(unexpected)
    unsupported.append(unexpected)
    summary = summarize(results, unsupported)
    assert not summary["passed"]
    assert summary["unexpected_unsupported_cases"]


def test_cli_and_cpu_affinity_parsing():
    assert parse_args([]).threads == THREAD_POLICIES
    assert parse_args([]).calibrate
    assert parse_args(["--threads", "1"]).threads == ("1",)
    assert not parse_args(["--no-calibrate"]).calibrate
    assert _parse_cpu_list("0-2,4") == {0, 1, 2, 4}
