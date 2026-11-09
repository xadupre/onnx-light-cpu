#!/usr/bin/env python3
"""Separate generation-only logits benchmark, not the all-token TinyLLM ASV case."""

import argparse
import json
import statistics
import time

import numpy as np
from onnx_light.onnx import ModelProto, TensorProto, helper, load, numpy_helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_py._onnxpykernels import runtime

from onnx_light_cpu import register_kernels_for_session
from onnx_light_cpu.generation import optimize_generation_logits


def make_model(batch=1, sequence=128, hidden=512, vocab=32000):
    """Synthetic prefill proxy: Relu plus a TinyLLM-sized FP32 LM head."""
    weight = np.random.default_rng(42).standard_normal((hidden, vocab)).astype(np.float32)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Relu", ["input"], ["hidden"]),
                helper.make_node("MatMul", ["hidden", "weight"], ["full"]),
                helper.make_node("GatherND", ["full", "positions"], ["logits"], batch_dims=1),
            ],
            "generation_last_token_logits",
            [
                helper.make_tensor_value_info("input", TensorProto.FLOAT, ["B", "S", hidden]),
                helper.make_tensor_value_info("positions", TensorProto.INT64, ["B", 1, 1]),
            ],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["B", 1, vocab])],
            [numpy_helper.from_array(weight, "weight")],
            value_info=[
                helper.make_tensor_value_info("hidden", TensorProto.FLOAT, ["B", "S", hidden]),
                helper.make_tensor_value_info("full", TensorProto.FLOAT, ["B", "S", vocab]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    feeds = {
        "input": np.random.default_rng(43)
        .standard_normal((batch, sequence, hidden))
        .astype(np.float32),
        "positions": np.full((batch, 1, 1), sequence - 1, dtype=np.int64),
    }
    return model, feeds


def benchmark(
    model,
    feeds,
    *,
    logits="logits",
    threads=1,
    repeat=10,
    warmup=2,
    backends=("onnx-light-cpu", "onnxruntime"),
):
    """Time full and selected prefill and isolated projections on both backends.

    An external model must already end in the supported generation selection.
    All other outputs (for example KV caches) remain in the prefill graph.
    """
    import onnxruntime as ort

    reduced = optimize_generation_logits(model)
    if model.SerializeToString() == reduced.SerializeToString():
        raise ValueError("No eligible generation logits selection found.")
    if not any(
        node.op_type == "MatMul" and list(node.output) == [logits] for node in reduced.graph.node
    ):
        raise ValueError("The requested logits output is not an eligible terminal selection.")
    selection = next(node for node in model.graph.node if list(node.output) == [logits])
    projection = next(
        node for node in model.graph.node if list(node.output) == [selection.input[0]]
    )
    hidden, weight_name = map(str, projection.input)
    positions = str(selection.input[1])
    weight = next(value for value in model.graph.initializer if value.name == weight_name)
    infos = {value.name: value for value in (*model.graph.input, *model.graph.value_info)}

    full = ModelProto()
    full.CopyFrom(model)
    full.graph.node.clear()
    full.graph.node.extend(node for node in model.graph.node if list(node.output) != [logits])
    for output in full.graph.output:
        if output.name == logits:
            output.CopyFrom(infos[selection.input[0]])
    capture = ModelProto()
    capture.CopyFrom(full)
    capture.graph.output.append(infos[hidden])
    hidden_values = ReferenceEvaluator(capture).run([hidden], feeds)[0]
    projection_model = helper.make_model(
        helper.make_graph(
            [projection, selection],
            "generation_projection",
            [infos[hidden], infos[positions]],
            [next(value for value in model.graph.output if value.name == logits)],
            [weight],
            value_info=[infos[selection.input[0]]],
        ),
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
    )
    full_projection = ModelProto()
    full_projection.CopyFrom(projection_model)
    full_projection.graph.node.clear()
    full_projection.graph.node.append(projection)
    full_projection.graph.output[0].CopyFrom(infos[selection.input[0]])
    head_feeds = {hidden: hidden_values, positions: feeds[positions]}
    cases = (
        ("prefill", "all_tokens", full, feeds),
        ("prefill", "selected_after_projection", model, feeds),
        ("prefill", "selected_before_projection", reduced, feeds),
        ("projection", "all_tokens", full_projection, head_feeds),
        ("projection", "selected_after_projection", projection_model, head_feeds),
        (
            "projection",
            "selected_before_projection",
            optimize_generation_logits(projection_model),
            head_feeds,
        ),
    )
    rows = []
    expected = {}
    full_sizes = {}
    # Finish CPU timings before constructing any ORT spinning thread pools.
    for backend in ("onnx-light-cpu", "onnxruntime"):
        if backend not in backends:
            continue
        for phase, contract, candidate, inputs in cases:
            serialized = candidate.SerializeToString()
            if backend == "onnx-light-cpu":
                session = ReferenceEvaluator(
                    candidate, cpu_execution={"num_threads": threads, "affinity_policy": "none"}
                )
                register_kernels_for_session(session)
                timed_feeds = {
                    name: runtime.tensor_from_proto(numpy_helper.from_array(value, name))
                    for name, value in inputs.items()
                }
            else:
                options = ort.SessionOptions()
                options.intra_op_num_threads = threads
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(
                    serialized, options, providers=["CPUExecutionProvider"]
                )
                timed_feeds = inputs
            output_name = str(selection.input[0]) if contract == "all_tokens" else logits
            actual = session.run([output_name], timed_feeds)[0]
            array = np.asarray(actual)
            if contract == "all_tokens":
                full_sizes[backend, phase] = array.nbytes
                selected = np.stack(
                    [array[b, int(p), :] for b, p in enumerate(feeds[positions][:, 0, 0])]
                )[:, None, :]
                expected[backend, phase] = selected
            else:
                np.testing.assert_allclose(array, expected[backend, phase], rtol=2e-3, atol=2e-3)
            for _ in range(warmup):
                session.run(None, timed_feeds)
            durations = []
            for _ in range(repeat):
                start = time.perf_counter()
                session.run(None, timed_feeds)
                durations.append(time.perf_counter() - start)
            rows.append(
                {
                    "scenario": "generation_last_token_logits",
                    "backend": backend,
                    "phase": phase,
                    "contract": contract,
                    "median_s": statistics.median(durations),
                    "logits_bytes": array.nbytes,
                    "logits_bytes_saved": full_sizes[backend, phase] - array.nbytes,
                    "threads": threads,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", help="Exported model with terminal GatherND generation selection"
    )
    parser.add_argument(
        "--feeds", help="NPZ feeds for --model (including physical token positions)"
    )
    parser.add_argument("--logits", default="logits")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--backend",
        nargs="+",
        choices=("onnx-light-cpu", "onnxruntime"),
        default=["onnx-light-cpu", "onnxruntime"],
    )
    args = parser.parse_args()
    if (
        any(
            value <= 0
            for value in (
                args.batch,
                args.sequence,
                args.hidden,
                args.vocab,
                args.threads,
                args.repeat,
            )
        )
        or args.warmup < 0
    ):
        parser.error(
            "dimensions, threads and repeat must be positive; warmup must be nonnegative"
        )
    if bool(args.model) != bool(args.feeds):
        parser.error("--model and --feeds must be supplied together")
    if args.model:
        model = load(args.model, load_external_data=True)
        with np.load(args.feeds, allow_pickle=False) as archive:
            feeds = dict(archive)
    else:
        model, feeds = make_model(args.batch, args.sequence, args.hidden, args.vocab)
    rows = benchmark(
        model,
        feeds,
        logits=args.logits,
        threads=args.threads,
        repeat=args.repeat,
        warmup=args.warmup,
        backends=args.backend,
    )
    print(json.dumps({"synthetic": not bool(args.model), "results": rows}, indent=2))


if __name__ == "__main__":
    main()
