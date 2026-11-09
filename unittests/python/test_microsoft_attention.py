# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Microsoft fused Attention and offline split Whisper generation regressions."""

import importlib.util
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto, helper, load, numpy_helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light_cpu import (
    clear_used_kernel_names,
    register_kernel_for_session,
    register_kernels,
    registered_kernels,
    set_kernel_usage_recording,
    used_kernel_names,
)


def _run_upstream(model, feeds):
    """Keep bias-free CPU coverage while avoiding ORT 1.31's null bias dereference."""
    ort_model = type(model)()
    ort_model.CopyFrom(model)
    node = ort_model.graph.node[0]
    inputs = list(node.input)
    if not inputs[2]:
        inputs[2] = "ort_zero_bias"
        node.input.clear()
        node.input.extend(inputs)
        weights = feeds[inputs[1]]
        ort_model.graph.initializer.append(
            numpy_helper.from_array(np.zeros(weights.shape[1], dtype=weights.dtype), inputs[2])
        )
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        ort_model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    ).run(None, feeds)


class TestMicrosoftAttention(ExtTestCase):
    def test_finite_causal_mask_matches_upstream(self):
        for length, mask_rank, with_bias in (
            (2, 1, False),
            (2, 2, False),
            (2, 3, False),
            (2, 2, True),
            (32, 2, False),
            (2, 0, False),
            (32, 0, False),
        ):
            with self.subTest(length=length, rank=mask_rank, bias=with_bias):
                maximum = np.finfo(np.float32).max
                feeds = {
                    "input": np.zeros((1, length, 2), dtype=np.float32),
                    "weights": np.array([[1, 0, 1], [0, maximum, 2]], dtype=np.float32),
                }
                feeds["input"][0, :2] = np.eye(2, dtype=np.float32)
                inputs = ["input", "weights", "", "mask" if mask_rank else "", ""]
                if mask_rank:
                    feeds["weights"][...] = [[0, 0, 1], [0, 0, 2]]
                    shape = {1: (1,), 2: (1, length), 3: (1, length, length)}[mask_rank]
                    feeds["mask"] = np.zeros(shape, dtype=np.int32)
                if with_bias:
                    inputs.append("attention_bias")
                    feeds["attention_bias"] = np.full(
                        (1, 1, length, length), maximum, dtype=np.float32
                    )
                model = helper.make_model(
                    helper.make_graph(
                        [
                            helper.make_node(
                                "Attention",
                                inputs,
                                ["output"],
                                domain="com.microsoft",
                                num_heads=1,
                                unidirectional=1,
                                mask_filter_value=float(-maximum),
                            )
                        ],
                        "microsoft-attention-finite-causal",
                        [
                            helper.make_tensor_value_info(
                                name,
                                TensorProto.INT32 if name == "mask" else TensorProto.FLOAT,
                                value.shape,
                            )
                            for name, value in feeds.items()
                        ],
                        [
                            helper.make_tensor_value_info(
                                "output", TensorProto.FLOAT, [1, length, 1]
                            )
                        ],
                    ),
                    opset_imports=[helper.make_opsetid("com.microsoft", 1)],
                    ir_version=10,
                )
                expected = _run_upstream(model, feeds)
                session = ReferenceEvaluator(model)
                register_kernel_for_session(session, "com.microsoft", "Attention")
                actual = session.run(None, feeds)
                np.testing.assert_allclose(actual[0], expected[0], rtol=3e-5, atol=3e-5)

    def test_causal_attention_without_mask_matches_upstream(self):
        feeds = {
            "input": np.array([[[1, 0], [0, 1], [1, 1]]], dtype=np.float32),
            "weights": np.array([[1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1]], dtype=np.float32),
        }
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "Attention",
                        ["input", "weights", "", "", ""],
                        ["output"],
                        domain="com.microsoft",
                        num_heads=1,
                        unidirectional=1,
                    )
                ],
                "microsoft-attention-causal",
                [
                    helper.make_tensor_value_info(name, TensorProto.FLOAT, value.shape)
                    for name, value in feeds.items()
                ],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2])],
            ),
            opset_imports=[helper.make_opsetid("com.microsoft", 1)],
            ir_version=10,
        )
        expected = _run_upstream(model, feeds)
        session = ReferenceEvaluator(model)
        register_kernel_for_session(session, "com.microsoft", "Attention")
        actual = session.run(None, feeds)
        np.testing.assert_allclose(actual[0], expected[0], rtol=3e-5, atol=3e-5)

    def test_projection_and_present_match_upstream(self):
        rng = np.random.default_rng(42)
        for with_bias, with_past, with_present, mask_kind, with_attention_bias in (
            (False, False, False, "", False),
            (True, False, True, "", False),
            (True, True, True, "", False),
            (True, False, True, "length", False),
            (True, False, True, "start", False),
            (True, True, True, "padding", False),
            (True, False, True, "query", False),
            (True, False, True, "", True),
            (True, False, True, "padding", True),
            (False, False, False, "zero_scale", False),
            (True, True, True, "zero_filter", False),
        ):
            with self.subTest(
                bias=with_bias,
                past=with_past,
                present=with_present,
                mask=mask_kind,
                attention_bias=with_attention_bias,
            ):
                attributes = {}
                if mask_kind == "zero_scale":
                    attributes["scale"] = 0.0
                    mask_kind = ""
                elif mask_kind == "zero_filter":
                    attributes["mask_filter_value"] = 0.0
                    mask_kind = ""
                feeds = {
                    "input": rng.standard_normal((2, 3, 4)).astype(np.float32),
                    "weights": rng.standard_normal((4, 12)).astype(np.float32),
                }
                inputs = ["input", "weights", "bias" if with_bias else ""]
                if with_bias:
                    feeds["bias"] = rng.standard_normal(12).astype(np.float32)
                total = 5 if with_past else 3
                inputs.extend(["mask" if mask_kind else "", "past" if with_past else ""])
                if with_past:
                    feeds["past"] = rng.standard_normal((2, 2, 2, 2, 2)).astype(np.float32)
                if mask_kind == "length":
                    feeds["mask"] = np.array([3, 2], dtype=np.int32)
                elif mask_kind == "start":
                    feeds["mask"] = np.array([3, 3, 0, 1], dtype=np.int32)
                elif mask_kind:
                    shape = {"padding": (2, total), "query": (2, 3, total)}[mask_kind]
                    feeds["mask"] = np.ones(shape, dtype=np.int32)
                    feeds["mask"][..., total - 1] = 0
                if with_attention_bias:
                    inputs.append("attention_bias")
                    feeds["attention_bias"] = rng.standard_normal((1, 2, 3, total)).astype(
                        np.float32
                    )
                outputs = ["output", "present"] if with_present else ["output"]
                model = helper.make_model(
                    helper.make_graph(
                        [
                            helper.make_node(
                                "Attention",
                                inputs,
                                outputs,
                                domain="com.microsoft",
                                num_heads=2,
                                unidirectional=int(with_past),
                                **attributes,
                            )
                        ],
                        "microsoft-attention",
                        [
                            helper.make_tensor_value_info(
                                name,
                                TensorProto.INT32 if name == "mask" else TensorProto.FLOAT,
                                value.shape,
                            )
                            for name, value in feeds.items()
                        ],
                        [
                            helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, 4]),
                            *(
                                [
                                    helper.make_tensor_value_info(
                                        "present",
                                        TensorProto.FLOAT,
                                        [2, 2, 2, 5 if with_past else 3, 2],
                                    )
                                ]
                                if with_present
                                else []
                            ),
                        ],
                    ),
                    opset_imports=[helper.make_opsetid("com.microsoft", 1)],
                    ir_version=10,
                )
                expected = _run_upstream(model, feeds)
                session = ReferenceEvaluator(model)
                register_kernel_for_session(session, "com.microsoft", "Attention")
                actual = session.run(None, feeds)
                self.assertEqual(len(expected), len(actual))
                for reference, result in zip(expected, actual, strict=True):
                    np.testing.assert_allclose(result, reference, rtol=3e-5, atol=3e-5)

    @unittest.skipUnless(
        os.environ.get("ONNX_LIGHT_CPU_REQUIRE_WHISPER_EXPORT") == "1"
        or all(
            importlib.util.find_spec(name) for name in ("torch", "transformers", "modelbuilder")
        ),
        "Offline Whisper export requires torch, transformers and the mbext modelbuilder source.",
    )
    def test_native_whisper_encoder_and_five_greedy_tokens(self):
        import torch
        from modelbuilder.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            GenerationConfig,
            PreTrainedTokenizerFast,
            WhisperConfig,
            WhisperForConditionalGeneration,
        )
        from transformers.generation import GenerationMixin

        config = WhisperConfig(
            architectures=["WhisperForConditionalGeneration"],
            vocab_size=128,
            d_model=32,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            num_mel_bins=8,
            max_source_positions=1500,
            max_target_positions=32,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            decoder_start_token_id=1,
            suppress_tokens=None,
            begin_suppress_tokens=None,
        )
        torch.manual_seed(42)
        model = WhisperForConditionalGeneration(config).eval()
        audio = np.random.default_rng(42).standard_normal((1, 8, 3000)).astype(np.float32)
        prompt = np.array([[1, 3, 4]], dtype=np.int64)
        with torch.no_grad():
            encoder_output = model.model.encoder(torch.from_numpy(audio))
            decoder_output = model(
                encoder_outputs=encoder_output,
                decoder_input_ids=torch.from_numpy(prompt),
            )
            reference = GenerationMixin.generate(
                model,
                encoder_outputs=encoder_output,
                decoder_input_ids=torch.from_numpy(prompt),
                generation_config=GenerationConfig(
                    decoder_start_token_id=1,
                    bos_token_id=1,
                    pad_token_id=0,
                    eos_token_id=None,
                    max_new_tokens=5,
                    do_sample=False,
                ),
            ).numpy()
            expected_cross = {
                role: projection(encoder_output.last_hidden_state)
                .reshape(1, 1500, 2, 16)
                .transpose(1, 2)
                .contiguous()
                .numpy()
                for role, projection in (
                    ("key", model.model.decoder.layers[0].encoder_attn.k_proj),
                    ("value", model.model.decoder.layers[0].encoder_attn.v_proj),
                )
            }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir, output_dir, cache_dir = (
                root / name for name in ("model", "export", "cache")
            )
            model.save_pretrained(model_dir)
            PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(
                    WordLevel(vocab={"<unk>": 0, "<s>": 1, "</s>": 2}, unk_token="<unk>")
                ),
                bos_token="<s>",
                eos_token="</s>",
                unk_token="<unk>",
            ).save_pretrained(model_dir)
            create_model(
                model_name="openai/whisper-tiny",
                input_path=str(model_dir),
                output_dir=str(output_dir),
                precision="fp32",
                execution_provider="cpu",
                cache_dir=str(cache_dir),
            )
            encoder = load(str(output_dir / "encoder.onnx"), load_external_data=True)
            self.assertTrue(
                any(
                    node.domain == "com.microsoft" and node.op_type == "Attention"
                    for node in encoder.graph.node
                )
            )
            register_kernels()
            encoder_session = ReferenceEvaluator(encoder)
            set_kernel_usage_recording(encoder_session, True)
            clear_used_kernel_names(encoder_session)
            try:
                values = encoder_session.run(None, {"audio_features": audio})
                native_encoder = dict(
                    zip((out.name for out in encoder.graph.output), values, strict=True)
                )
                np.testing.assert_allclose(
                    native_encoder["hidden_states"],
                    encoder_output.last_hidden_state.numpy(),
                    rtol=1e-4,
                    atol=1e-4,
                )
                decoder = load(str(output_dir / "decoder.onnx"), load_external_data=True)
                feeds = {"input_ids": prompt.astype(np.int32)}
                for role in ("key", "value"):
                    cross = native_encoder[f"present_{role}_cross_0"]
                    np.testing.assert_allclose(cross, expected_cross[role], rtol=1e-4, atol=1e-4)
                    feeds[f"past_{role}_cross_0"] = cross
                    input_name = f"past_{role}_self_0"
                    feeds[input_name] = np.zeros((1, 2, 0, 16), dtype=np.float32)
                    binding = decoder.graph.persistent_bindings.add()
                    binding.input_name = input_name
                    binding.output_name = f"present_{role}_self_0"
                prefill = ReferenceEvaluator(decoder).run(None, feeds)
                logits = dict(
                    zip((out.name for out in decoder.graph.output), prefill, strict=True)
                )["logits"]
                np.testing.assert_allclose(
                    logits, decoder_output.logits.numpy(), rtol=1e-3, atol=1e-4
                )
                for value in decoder.graph.input:
                    if value.name == "input_ids":
                        value.name = "decoder_input_ids"
                for node in decoder.graph.node:
                    inputs = [
                        "decoder_input_ids" if name == "input_ids" else name
                        for name in node.input
                    ]
                    node.input.clear()
                    node.input.extend(inputs)
                token_cast = helper.make_model(
                    helper.make_graph(
                        [
                            helper.make_node(
                                "Cast", ["input_ids"], ["decoder_input_ids"], to=TensorProto.INT32
                            )
                        ],
                        "whisper-token-cast",
                        [
                            helper.make_tensor_value_info(
                                "input_ids", TensorProto.INT64, [None, None]
                            )
                        ],
                        [
                            helper.make_tensor_value_info(
                                "decoder_input_ids", TensorProto.INT32, [None, None]
                            )
                        ],
                    ),
                    opset_imports=[helper.make_opsetid("", 18)],
                )
                feeds["input_ids"] = prompt
                tokens = ReferenceEvaluator(decoder).generate(
                    feeds,
                    max_new_tokens=5,
                    temperature=0.0,
                    embedding=ReferenceEvaluator(token_cast),
                )
                np.testing.assert_array_equal(tokens, reference)
                self.assertEqual(tokens.shape, (1, prompt.shape[1] + 5))
                attention = next(
                    kernel
                    for kernel in registered_kernels()
                    if kernel.domain == "com.microsoft" and kernel.op_type == "Attention"
                )
                self.assertIn(attention.kernel_name, used_kernel_names(encoder_session))
            finally:
                set_kernel_usage_recording(encoder_session, False)


if __name__ == "__main__":
    unittest.main()
