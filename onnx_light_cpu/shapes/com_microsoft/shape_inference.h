// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/shapes/shapes_context.h"

#include <cstdint>
#include <string>
#include <vector>

namespace onnx_light_cpu {

/// Describes the additional runtime support supplied for one custom operator.
struct OperatorSupportRegistration {
  std::string domain;
  std::string op_type;
  std::string shape_inference_function;
  std::string peak_memory_function;
  std::vector<std::string> fusion_patterns;
  bool has_gradient = false;
};

/// Returns the custom operator support available from onnx-light-cpu.
std::vector<OperatorSupportRegistration> CollectOperatorSupport();

/// Infers the output shape for ``com.microsoft::CDist``.
void ComputeShapeCDist(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                       const ONNX_LIGHT_NAMESPACE::NodeProto &node);
/// Infers the output shape for ``com.microsoft::BiasGelu``.
void ComputeShapeBiasGelu(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                          const ONNX_LIGHT_NAMESPACE::NodeProto &node);
/// Infers the output shape for ``com.microsoft::GroupQueryAttention``.
void ComputeShapeGroupQueryAttention(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                                     const ONNX_LIGHT_NAMESPACE::NodeProto &node);
/// Infers output and state shapes for ``com.microsoft::LinearAttention``.
void ComputeShapeLinearAttention(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                                 const ONNX_LIGHT_NAMESPACE::NodeProto &node);

/// Returns the scratch-memory requirement for ``com.microsoft::CDist``.
int64_t ComputePeakMemoryCDist(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);
/// Returns the scratch-memory requirement for ``com.microsoft::BiasGelu``.
int64_t ComputePeakMemoryBiasGelu(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);
/// Returns the scratch-memory requirement for ``com.microsoft::GroupQueryAttention``.
int64_t ComputePeakMemoryGroupQueryAttention(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);
/// Returns the scratch-memory requirement for ``com.microsoft::LinearAttention``.
int64_t ComputePeakMemoryLinearAttention(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);

/// Registers the custom shape-inference and peak-memory functions.
void RegisterMicrosoftShapeAndMemoryFunctions();

} // namespace onnx_light_cpu
