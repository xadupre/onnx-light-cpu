// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/shapes/dispatch_table.h"
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
/// Infers output and unwindowed carry state for ``com.microsoft::CausalConvWithState``.
void ComputeShapeCausalConvWithState(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                                     const ONNX_LIGHT_NAMESPACE::NodeProto &node);
/// Infers the output shape for ``com.microsoft::BiasGelu``.
void ComputeShapeBiasGelu(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                          const ONNX_LIGHT_NAMESPACE::NodeProto &node);
/// Infers the output, optional statistics, and residual shapes for
/// ``com.microsoft::SkipLayerNormalization``.
void ComputeShapeSkipLayerNormalization(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                                        const ONNX_LIGHT_NAMESPACE::NodeProto &node);
/// Infers the output, optional statistics, and residual shapes for
/// ``com.microsoft::SkipSimplifiedLayerNormalization``.
void ComputeShapeSkipSimplifiedLayerNormalization(
    ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node);
/// Infers the output shape for ``com.microsoft::GroupQueryAttention``.
void ComputeShapeGroupQueryAttention(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                                     const ONNX_LIGHT_NAMESPACE::NodeProto &node);
void ComputeShapeMultiHeadAttention(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                                    const ONNX_LIGHT_NAMESPACE::NodeProto &node);
/// Infers output and state shapes for ``com.microsoft::LinearAttention``.
void ComputeShapeLinearAttention(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                                 const ONNX_LIGHT_NAMESPACE::NodeProto &node);
/// Infers the packed ``com.microsoft::GatherBlockQuantized`` output shape.
void ComputeShapeGatherBlockQuantized(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                                      const ONNX_LIGHT_NAMESPACE::NodeProto &node);
/// Infers the restricted ``com.microsoft::MatMulNBits`` output shape.
void ComputeShapeMatMulNBits(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                             const ONNX_LIGHT_NAMESPACE::NodeProto &node);
/// Infers the output shape for the supported ``com.microsoft::MoE`` subset.
void ComputeShapeMoE(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                     const ONNX_LIGHT_NAMESPACE::NodeProto &node);
/// Infers the output shape for CPU quantized expert projections.
void ComputeShapeQMoE(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                      const ONNX_LIGHT_NAMESPACE::NodeProto &node);

/// Returns the scratch-memory requirement for ``com.microsoft::CDist``.
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate ComputePeakMemoryCDist(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);
/// Returns zero scratch memory for the streaming causal convolution.
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate ComputePeakMemoryCausalConvWithState(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);
/// Returns the scratch-memory requirement for ``com.microsoft::BiasGelu``.
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate ComputePeakMemoryBiasGelu(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);
/// Returns the zero scratch-memory requirement for
/// ``com.microsoft::SkipLayerNormalization``.
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate ComputePeakMemorySkipLayerNormalization(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);
/// Returns the zero scratch-memory requirement for
/// ``com.microsoft::SkipSimplifiedLayerNormalization``.
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate
ComputePeakMemorySkipSimplifiedLayerNormalization(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);
/// Returns the scratch-memory requirement for ``com.microsoft::GroupQueryAttention``.
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate ComputePeakMemoryGroupQueryAttention(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate ComputePeakMemoryMultiHeadAttention(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);
/// Returns the scratch-memory requirement for ``com.microsoft::LinearAttention``.
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate ComputePeakMemoryLinearAttention(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);
/// Returns zero scratch memory for the streaming quantized gather.
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate ComputePeakMemoryGatherBlockQuantized(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);
/// Returns zero heap scratch; bounded INT4 panels use 6144 bytes of stack per callback.
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate ComputePeakMemoryMatMulNBits(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);
/// Returns a conservative whole-operator routing and expert scratch bound for
/// ``com.microsoft::MoE``.
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate ComputePeakMemoryMoE(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);
/// Returns the conservative quantized expert and routing scratch bound.
ONNX_LIGHT_NAMESPACE::core::shapes::PeakMemoryEstimate ComputePeakMemoryQMoE(
    ONNX_LIGHT_NAMESPACE::core::symbolic::Device device,
    const ONNX_LIGHT_NAMESPACE::NodeProto &node,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::symbolic::SymShape> &input_shapes);

/// Registers the custom shape-inference and peak-memory functions.
void RegisterMicrosoftShapeAndMemoryFunctions();

} // namespace onnx_light_cpu
