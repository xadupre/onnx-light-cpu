import math

from tools.benchmark_avx2_parity import (
    AVX2_CORPUS,
    AVX2_DTYPES,
    AVX2_TESTS,
    SPEEDUP_GROUPS,
    THREAD_POLICIES,
    build_report,
    parse_args,
    render_markdown,
    speedup_group,
)


def test_avx2_corpus_covers_priority_families():
    selection = " ".join(AVX2_TESTS)
    assert "gemm|matmul" in selection
    assert "attention|group_query_attention" in selection
    assert "layernormalization" in selection
    assert "abs|exp|log" in selection
    assert "add|sub|mul|div" in selection
    assert AVX2_DTYPES == ("float32", "float16")
    assert AVX2_CORPUS[0][1] == ("float32", "float16")
    assert all(dtypes == ("float32",) for _, dtypes in AVX2_CORPUS[2:])
    assert THREAD_POLICIES == ("1", "physical")


def test_speedup_groups_have_exact_boundaries():
    assert SPEEDUP_GROUPS == ("<0.5x", "0.5x-0.9x", "0.9x-1.0x", ">=1.0x")
    assert speedup_group(0.49) == "<0.5x"
    assert speedup_group(0.5) == "0.5x-0.9x"
    assert speedup_group(0.9) == "0.9x-1.0x"
    assert speedup_group(1.0) == ">=1.0x"


def test_report_ranks_gaps_retains_samples_and_marks_shared_results():
    cases = [
        ("test_cpu_matmul_llm_qwen3_8b_qkv_m1_k4096_n6144_float32_benchmark", 2.0, 1.0),
        (
            (
                "test_cpu_group_query_attention_model_qwen3_8b_int4_mb_prefill_"
                "b1_s128_qh32_kvh8_hd128_pastlen0_causal_float32_benchmark"
            ),
            1.0,
            1.2,
        ),
    ]
    raw = [
        {
            "case": case,
            "threads": 1,
            "thread_policy": "1",
            "runtime": runtime,
            "duration_s": duration,
        }
        for case, cpu, ort in cases
        for runtime, duration in (("onnx-light-cpu", cpu), ("onnxruntime", ort))
    ]
    aggregated = [
        {
            "case": case,
            "operator": "MatMul" if "matmul" in case else "GroupQueryAttention",
            "dtype": "float32",
            "thread_policy": "1",
            "threads": 1,
            "input_shapes": '[{"x": [1, 2]}]',
            "runtime_order": "onnx-light-cpu,onnxruntime",
        }
        for case, _, _ in cases
    ]
    report = build_report(
        raw,
        aggregated,
        environment="shared",
        command="benchmark",
        simd_level="AVX2",
    )

    first, second = report["results"]
    assert first["workload"] == "Qwen decode"
    assert first["gap_rank"] == 1
    assert first["onnx_light_cpu_samples_s"] == [2.0]
    assert first["onnxruntime_samples_s"] == [1.0]
    assert math.isclose(first["speedup"], 0.5)
    assert second["workload"] == "Qwen prefill"
    assert report["follow_up_candidates"][0]["case"] == first["case"]
    assert report["complete"]
    assert not report["metadata"]["final_parity_decision"]
    markdown = render_markdown(report)
    assert "Diagnostic only" in markdown
    assert "## 0.5x-0.9x onnx-light-cpu / ONNX Runtime" in markdown
    assert "## >=1.0x onnx-light-cpu / ONNX Runtime" in markdown
    assert "Qwen decode" in markdown
    assert "Qwen prefill" in markdown


def test_report_retains_unsupported_onnxruntime_case():
    case = "test_cpu_abs_n1024_float32_benchmark"
    raw = [
        {
            "case": case,
            "threads": 1,
            "thread_policy": "1",
            "runtime": "onnx-light-cpu",
            "duration_s": 1.0,
        }
    ]
    aggregated = [
        {
            "case": case,
            "operator": "Abs",
            "dtype": "float32",
            "thread_policy": "1",
            "threads": 1,
            "input_shapes": '[{"x": [1024]}]',
            "runtime_order": "onnx-light-cpu",
            "onnxruntime_error": "unsupported",
        }
    ]
    report = build_report(
        raw,
        aggregated,
        environment="shared",
        command="benchmark",
        simd_level="AVX2",
    )
    assert not report["complete"]
    assert report["results"] == []
    assert report["unsupported"][0]["onnxruntime_error"] == "unsupported"
    assert report["unsupported"][0]["onnx_light_cpu_samples_s"] == [1.0]


def test_cli_defaults_to_shared_one_and_physical_core_runs():
    args = parse_args([])
    assert args.dtypes == AVX2_DTYPES
    assert args.thread_policies == THREAD_POLICIES
    assert args.environment == "shared"
