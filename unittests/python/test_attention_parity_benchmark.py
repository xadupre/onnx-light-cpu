import math
from types import SimpleNamespace
from unittest import TestCase

from tools.benchmark_attention_parity import (
    _expand_cpu_part,
    estimate_temporary_memory,
    measure_alternating,
    parse_args,
    parse_case_name,
    summarize,
)


def test_backend_case_name_is_traceable_and_complete():
    case = parse_case_name(
        "test_cpu_attention_opset24_rank3_gqa_q8_kv1024_hd128_causal_nonpad_bfloat16_benchmark"
    )

    assert case == {
        "family": None,
        "opset": 24,
        "layout": "rank3",
        "geometry": "gqa",
        "q_length": 8,
        "kv_length": 1024,
        "head_dim": 128,
        "q_heads": None,
        "kv_heads": None,
        "mask": "causal",
        "cache": "nonpad",
        "dtype": "bfloat16",
    }


def test_backend_case_name_decodes_named_model_families_and_head_counts():
    case = parse_case_name(
        "test_cpu_attention_llm_qwen3_8b_opset24_rank4_gqa_q1_kv4096_hd128"
        "_qh32_kvh8_causal_internal_cache_float16_benchmark"
    )

    assert case == {
        "family": "llm_qwen3_8b",
        "opset": 24,
        "layout": "rank4",
        "geometry": "gqa",
        "q_length": 1,
        "kv_length": 4096,
        "head_dim": 128,
        "q_heads": 32,
        "kv_heads": 8,
        "mask": "causal",
        "cache": "internal_cache",
        "dtype": "float16",
    }

    qwen36_case = parse_case_name(
        "test_cpu_attention_llm_qwen3_6_35b_a3b_opset23_rank3_gqa_q128_kv128_hd256"
        "_qh16_kvh2_causal_stateless_float16_benchmark"
    )
    assert qwen36_case["family"] == "llm_qwen3_6_35b_a3b"
    assert qwen36_case["q_heads"] == 16
    assert qwen36_case["kv_heads"] == 2
    assert qwen36_case["head_dim"] == 256


def test_temporary_memory_matches_single_key_and_streaming_dispatch():
    single_key = parse_case_name(
        "test_cpu_attention_opset23_rank4_mha_q128_kv1_hd64_none_stateless_float16_benchmark"
    )
    feeds = {
        "Q": SimpleNamespace(shape=(1, 12, 128, 64), size=12 * 128 * 64),
        "K": SimpleNamespace(shape=(1, 12, 1, 64), size=12 * 64),
        "V": SimpleNamespace(shape=(1, 12, 1, 64), size=12 * 64),
    }
    assert estimate_temporary_memory(single_key, feeds, workers=4) == {
        "peak_temporary_bytes": 0,
        "peak_score_tile_bytes": 0,
        "score_block": {"Br": 0, "Bc": 0},
        "memory_gate_passed": True,
    }

    streaming = parse_case_name(
        "test_cpu_attention_opset23_rank4_mha_q1_kv128_hd64_none_stateless_bfloat16_benchmark"
    )
    memory = estimate_temporary_memory(streaming, feeds, workers=4)
    assert memory["peak_score_tile_bytes"] == 4 * 128 * 4
    assert memory["peak_temporary_bytes"] == 4 * (128 + 2 * 64) * 4
    assert memory["score_block"] == {"Br": 1, "Bc": 128}


def test_candidate_order_and_raw_samples_are_retained():
    order = []

    def cpu():
        order.append("cpu")

    def ort():
        order.append("ort")

    samples, candidate_order = measure_alternating((cpu, ort), repeat=3, warmup=0)

    assert order == ["cpu", "ort", "ort", "cpu", "cpu", "ort"]
    assert tuple(map(len, samples)) == (3, 3)
    assert candidate_order == [
        ["onnx-light-cpu", "onnxruntime"],
        ["onnxruntime", "onnx-light-cpu"],
        ["onnx-light-cpu", "onnxruntime"],
    ]


def test_rank3_packing_memory_is_tile_bounded_for_all_query_lengths():
    for q_length in (64, 128, 129):
        case = parse_case_name(
            f"test_cpu_attention_opset23_rank3_gqa_q{q_length}_kv128_hd64"
            "_none_stateless_float32_benchmark"
        )
        feeds = {
            "Q": SimpleNamespace(shape=(2, q_length, 8 * 64), size=2 * q_length * 8 * 64),
            "K": SimpleNamespace(shape=(2, 128, 2 * 64), size=2 * 128 * 2 * 64),
            "V": SimpleNamespace(shape=(2, 128, 2 * 64), size=2 * 128 * 2 * 64),
        }
        for workers in (1, 4):
            query_block = min(q_length, 128)
            tile_bytes = workers * (
                query_block * 128 * 4 + query_block * 64 * 8 + query_block * 9
            )
            packing_bytes = workers * (query_block + 128) * 64 * 8
            memory = estimate_temporary_memory(case, feeds, workers)
            assert memory["peak_temporary_bytes"] == tile_bytes + packing_bytes
            assert memory["memory_gate_passed"]


def test_half_conversion_memory_is_independent_of_batch_and_head_counts():
    for layout in ("rank3", "rank4"):
        for q_length in (1, 8, 128, 129, 1024):
            case = parse_case_name(
                f"test_cpu_attention_opset23_{layout}_gqa_q{q_length}_kv257_hd63"
                "_none_stateless_float16_benchmark"
            )
            memories = []
            for batch, q_heads, kv_heads in ((1, 4, 2), (3, 16, 4)):
                shapes = {
                    "Q": (batch, q_heads, q_length, 63),
                    "K": (batch, kv_heads, 257, 63),
                    "V": (batch, kv_heads, 257, 31),
                }
                feeds = {
                    name: SimpleNamespace(
                        shape=(
                            (b, heads, length, dimension)
                            if layout == "rank4"
                            else (b, length, heads * dimension)
                        ),
                        size=b * heads * length * dimension,
                    )
                    for name, (b, heads, length, dimension) in shapes.items()
                }
                memories.append(estimate_temporary_memory(case, feeds, workers=2))
            assert memories[0] == memories[1]
            if q_length < 16:
                conversion_block = 256 if q_length == 1 else 257
                assert (
                    memories[0]["peak_temporary_bytes"]
                    == 2 * (256 + 63 + 31 + conversion_block * (63 + 31)) * 4
                )
            assert memories[0]["memory_gate_passed"]


def test_streaming_conversion_reuse_stops_at_four_score_blocks():
    for dimension, block in ((64, 256), (256, 128)):
        for kv_length in (block * 4, block * 4 + 1):
            case = parse_case_name(
                f"test_cpu_attention_opset23_rank4_mha_q8_kv{kv_length}_hd{dimension}"
                "_none_stateless_float16_benchmark"
            )
            feeds = {
                name: SimpleNamespace(shape=(1, 12, length, dimension))
                for name, length in (("Q", 8), ("K", kv_length), ("V", kv_length))
            }
            conversion_block = kv_length if kv_length <= 4 * block else block
            memory = estimate_temporary_memory(case, feeds, workers=1)
            assert (
                memory["peak_temporary_bytes"]
                == (block + 2 * dimension + conversion_block * 2 * dimension) * 4
            )


def test_summary_applies_per_type_parity_and_memory_gates():
    results = []
    for dtype in ("float32", "float16", "bfloat16"):
        results.extend(
            [
                {"dtype": dtype, "speedup": 0.9, "memory_gate_passed": True},
                {"dtype": dtype, "speedup": 1.1, "memory_gate_passed": True},
            ]
        )

    summary = summarize(results)

    assert summary["passed"]
    assert all(
        math.isclose(dtype_summary["median_speedup"], 1.0)
        for dtype_summary in summary["by_dtype"].values()
    )


def test_summary_rejects_a_priority_regression():
    for field, value in (
        ("speedup", 0.89),
        ("tail_speedup", 0.89),
        ("memory_gate_passed", False),
    ):
        results = [
            {"dtype": dtype, "speedup": 1.0, "memory_gate_passed": True}
            for dtype in ("float32", "float16", "bfloat16")
        ]
        results[0][field] = value

        assert not summarize(results)["passed"]


def test_benchmark_defaults_to_identical_single_thread_execution():
    args = parse_args(["--enforce"])
    assert args.threads == 1
    assert args.enforce


def test_cpu_ranges_are_validated():
    assert list(_expand_cpu_part("2-4")) == [2, 3, 4]
    with TestCase().assertRaises(ValueError):
        _expand_cpu_part("4-2")


def test_registered_attention_reports_actual_execution_paths():
    import ml_dtypes
    import numpy as np
    from onnx_light.onnx import TensorProto, helper
    from onnx_light.onnx.reference import ReferenceEvaluator

    from onnx_light_cpu import (
        clear_used_kernel_names,
        register_kernels,
        set_kernel_usage_recording,
        used_kernel_names,
        used_kernel_paths,
    )

    register_kernels()
    for dtype, tensor_type in (
        (np.float32, TensorProto.FLOAT),
        (np.float16, TensorProto.FLOAT16),
        (ml_dtypes.bfloat16, TensorProto.BFLOAT16),
    ):
        for rank3, q_length, kv_length, materialized, expected_path in (
            (False, 16, 17, False, "tiled"),
            (True, 16, 17, False, "tiled"),
            (False, 1, 17, False, "streaming"),
            (True, 1, 17, False, "streaming"),
            (False, 16, 1, False, "single_key"),
            (False, 16, 17, True, "materialized"),
        ):
            if tensor_type == TensorProto.BFLOAT16 and expected_path == "tiled":
                expected_path = "streaming"
            shapes = {
                name: ([1, length, 2 * 7] if rank3 else [1, 2, length, 7])
                for name, length in (("Q", q_length), ("K", kv_length), ("V", kv_length))
            }
            attrs = {"q_num_heads": 2, "kv_num_heads": 2} if rank3 else {}
            if materialized:
                attrs["softmax_precision"] = TensorProto.DOUBLE
            model = helper.make_model(
                helper.make_graph(
                    [helper.make_node("Attention", ["Q", "K", "V"], ["Y"], **attrs)],
                    "attention_path",
                    [
                        helper.make_tensor_value_info(name, tensor_type, shape)
                        for name, shape in shapes.items()
                    ],
                    [helper.make_tensor_value_info("Y", tensor_type, shapes["Q"])],
                ),
                opset_imports=[helper.make_opsetid("", 23)],
            )
            session = ReferenceEvaluator(model, cpu_execution={"num_threads": 1})
            feeds = {name: np.full(shape, 0.25, dtype=dtype) for name, shape in shapes.items()}
            without_recording = session.run(None, feeds)
            assert used_kernel_paths(session) == []
            set_kernel_usage_recording(session, True)
            with_recording = session.run(None, feeds)
            paths = used_kernel_paths(session)
            assert f"Attention.{expected_path}" in paths, paths
            conversion = (
                "none"
                if dtype == np.float32 or expected_path == "single_key"
                else ("materialized" if materialized else "tile")
            )
            if tensor_type == TensorProto.BFLOAT16 and expected_path == "streaming":
                conversion = "element"
            packing = "tile" if rank3 and expected_path == "tiled" else "none"
            assert f"Attention.conversion.{conversion}" in paths, paths
            assert f"Attention.packing.{packing}" in paths, paths
            assert used_kernel_names(session) == ["onnx_light_cpu::Attention"]
            np.testing.assert_array_equal(without_recording[0], with_recording[0])
            clear_used_kernel_names(session)
            assert used_kernel_paths(session) == []
