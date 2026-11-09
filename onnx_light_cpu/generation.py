# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Explicit, output-contract-preserving generation model preparation."""

from onnx_light.onnx import AttributeProto, ModelProto, TensorProto


def optimize_generation_logits(model: ModelProto) -> ModelProto:
    """Return a copy with eligible per-batch logits selections before MatMul.

    Only ``GatherND(MatMul(hidden, weight), positions, batch_dims=1)`` is
    supported: hidden must have rank three, weight rank two, and INT64
    positions shape ``[batch, 1, 1]``. The caller supplies physical token
    positions, including padding-aware positions; no token is inferred here.
    Full logits must not be a graph output or have any other consumer.
    Missing shape information and graphs with subgraphs are left unchanged.

    This is opt-in model preparation, not a registered runtime optimization.
    Neither the original model nor its initializer ownership is changed.
    """
    result = ModelProto()
    result.CopyFrom(model)
    graph = result.graph
    if any(
        attr.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS)
        for node in graph.node
        for attr in node.attribute
    ):
        return result
    if any(
        function.domain in ("", "ai.onnx") and function.name in ("MatMul", "GatherND")
        for function in result.functions
    ):
        return result
    opset = next(
        (entry.version for entry in result.opset_import if entry.domain in ("", "ai.onnx")),
        0,
    )
    if opset < 12:
        return result

    infos = {value.name: value for value in (*graph.input, *graph.value_info)}
    weights = {value.name: value for value in graph.initializer}
    outputs = {value.name for value in graph.output}
    consumers: dict[str, list[int]] = {}
    producers = {}
    nodes = list(graph.node)
    for index, node in enumerate(nodes):
        for name in node.input:
            consumers.setdefault(name, []).append(index)
        for name in node.output:
            producers[name] = index

    def shape(name):
        if name in weights:
            return list(weights[name].dims)
        value = infos.get(name)
        if value is None or not value.type.tensor_type.HasField("shape"):
            return None
        return [
            dim.dim_value if dim.HasField("dim_value") else None
            for dim in value.type.tensor_type.shape.dim
        ]

    replacements = {}
    removed = set()
    for index, selection in enumerate(nodes):
        if (
            selection.domain not in ("", "ai.onnx")
            or selection.op_type != "GatherND"
            or len(selection.input) != 2
            or len(selection.output) != 1
            or len(selection.attribute) != 1
            or selection.attribute[0].name != "batch_dims"
            or selection.attribute[0].i != 1
        ):
            continue
        full_name, positions = selection.input
        projection_index = producers.get(full_name)
        if projection_index is None:
            continue
        projection = nodes[projection_index]
        if (
            projection.domain not in ("", "ai.onnx")
            or projection.op_type != "MatMul"
            or len(projection.input) != 2
            or len(projection.output) != 1
            or projection.attribute
            or full_name in outputs
            or consumers.get(full_name) != [index]
        ):
            continue
        hidden, weight = projection.input
        hshape, wshape, pshape = shape(hidden), shape(weight), shape(positions)
        ptype = None
        if positions in weights:
            ptype = weights[positions].data_type
        elif positions in infos:
            ptype = infos[positions].type.tensor_type.elem_type
        if (
            hshape is None
            or len(hshape) != 3
            or wshape is None
            or len(wshape) != 2
            or not isinstance(hshape[2], int)
            or hshape[2] <= 0
            or hshape[2] != wshape[0]
            or pshape is None
            or len(pshape) != 3
            or pshape[1:] != [1, 1]
            or ptype != TensorProto.INT64
        ):
            continue
        # Reuse the now-private full-logits name for selected hidden states.
        # Insert at the selection site so dynamically computed indices are ready.
        gather = type(selection)()
        gather.CopyFrom(selection)
        gather.input.clear()
        gather.input.extend((hidden, positions))
        gather.output.clear()
        gather.output.append(full_name)
        matmul = type(projection)()
        matmul.CopyFrom(projection)
        matmul.input.clear()
        matmul.input.extend((full_name, weight))
        matmul.output.clear()
        matmul.output.append(selection.output[0])
        replacements[index] = (gather, matmul)
        removed.add(projection_index)

    if removed:
        graph.node.clear()
        for index, node in enumerate(nodes):
            if index not in removed:
                graph.node.extend(replacements.get(index, (node,)))
        stale = {nodes[index].output[0] for index in removed}
        retained = [value for value in graph.value_info if value.name not in stale]
        graph.value_info.clear()
        graph.value_info.extend(retained)
    return result
