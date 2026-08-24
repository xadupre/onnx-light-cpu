import math

from tools.benchmark_binary_parity import (
    ELEMENT_COUNTS,
    PRIORITY_SIGNATURES,
    SHAPE_FAMILIES,
    THREAD_POLICIES,
    _parse_cpu_list,
    _task_metrics,
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

    results[0]["speedup"] = 0.89
    assert not summarize(results)["passed"]
    assert not summarize(results[1:])["matrix_complete"]


def test_task_accounting_respects_serial_and_worker_caps():
    case = parse_case_name(
        "test_cpu_add_v14_contiguous_float32xfloat32_to_float32_n4194304_benchmark"
    )
    assert _task_metrics(case, 4_194_304, 4_194_304, 1) == (1, 1)
    tasks, workers = _task_metrics(case, 4_194_304, 4_194_304, 32)
    assert tasks == workers == 4


def test_cli_and_cpu_affinity_parsing():
    assert parse_args([]).threads == THREAD_POLICIES
    assert parse_args(["--threads", "1"]).threads == ("1",)
    assert _parse_cpu_list("0-2,4") == {0, 1, 2, 4}
