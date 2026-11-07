// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/shapes/shapes_context.h"

#include <cstdint>
#include <vector>

namespace onnx_light_cpu {

namespace detail {

void RegisterBuiltinPeakMemoryFunctions();

} // namespace detail

/// CPU MatMul product scratch, excluding declared Y and input/prepared-weight storage.
/// Without dtype/executor metadata this covers all supported dtypes and executor sizes.
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate ComputePeakMemoryMatMul(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);

/// Conservative CPU Attention peak memory bound, excluding declared outputs.
/// Unresolved dimensions remain unknown.
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate ComputePeakMemoryAttention(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);

/// Infers experimental SimplifiedLayerNormalization outputs using runtime-compatible shapes.
void ComputeShapeSimplifiedLayerNormalization(
    ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node);

void ComputeShapeRotaryEmbedding(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                                 const ONNX_LIGHT_NAMESPACE::NodeProto &node);

ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate ComputePeakMemoryRotaryEmbedding(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);

/// Returns CPU scratch memory, excluding input and output tensors.
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate
ComputePeakMemorySimplifiedLayerNormalization(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);

/// Registers experimental ai.onnx shape-inference and CPU peak-memory functions.
void RegisterExperimentalShapeAndMemoryFunctions();

/// Installs the CPU MatMul estimator after its global kernel is selected.
void RegisterMatMulMemoryFunction();

} // namespace onnx_light_cpu
