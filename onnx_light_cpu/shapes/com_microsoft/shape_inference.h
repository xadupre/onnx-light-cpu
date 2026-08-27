// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/shapes/shapes_context.h"

namespace onnx_light_cpu {

void ComputeShapeCDist(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                       const ONNX_LIGHT_NAMESPACE::NodeProto &node);
void ComputeShapeBiasGelu(ONNX_LIGHT_NAMESPACE::core::shapes::ShapesContext &ctx,
                          const ONNX_LIGHT_NAMESPACE::NodeProto &node);

void RegisterMicrosoftShapeAndMemoryFunctions();

} // namespace onnx_light_cpu
