// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/shapes/shapes_context.h"

#include <cstdint>
#include <vector>

namespace onnx_light_cpu {

/// Infers experimental SimplifiedLayerNormalization outputs using runtime-compatible shapes.
void ComputeShapeSimplifiedLayerNormalization(
    ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node);

/// Returns CPU scratch memory, excluding input and output tensors.
int64_t ComputePeakMemorySimplifiedLayerNormalization(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);

/// Registers experimental ai.onnx shape-inference and CPU peak-memory functions.
void RegisterExperimentalShapeAndMemoryFunctions();

} // namespace onnx_light_cpu
