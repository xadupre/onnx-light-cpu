// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ml_cpu_compute/attention/attention_plan.h"

namespace ml_cpu_compute {

/// Fused FP32 projection followed by the existing tiled ONNX attention engine.
/// The plan uses rank-3 Q/K/V and rank-4 past, with equal Q/K widths.
void ComputeMicrosoftAttentionFloat32(const AttentionPlan &plan, std::size_t input_hidden,
                                      const float *input, const float *weights, const float *bias,
                                      const float *mask, const float *past, float *output,
                                      float *present);

/// Scratch only; excludes inputs and the optional packed present output.
std::size_t MicrosoftAttentionPeakMemoryBytes(const AttentionPlan &plan, std::size_t input_hidden);

} // namespace ml_cpu_compute
