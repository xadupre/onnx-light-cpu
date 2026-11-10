// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/graph_builder.h"

namespace onnx_light_cpu {

/**
 * Explicit, lossy CPU preparation of constant weights and biases in QDQ graphs.
 *
 * Run before QDQ fusions. Uses scalar symmetric INT8 weights and zero-point-zero
 * INT32 biases; per-axis/blocked quantization is not supported. Constant DQ
 * boundaries are retained. Subsequent optimization must likewise exclude
 * DequantizeLinear from constant folding until its QDQ fusions have run.
 * This stage is deliberately absent from the default pattern registry.
 */
ONNX_LIGHT_NAMESPACE::RewritingProto
PrepareWeightBiasQuantization(ONNX_LIGHT_NAMESPACE::core::builder::GraphBuilder &graph);

} // namespace onnx_light_cpu
