// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/gradient/grad_dispatcher.h"

namespace onnx_light_cpu {

/**
 * Registers standard-ONNX gradient graphs for `com.microsoft` operators.
 *
 * BiasGelu gradient:
 * \verbatim
 * A -----\ [Add] --> z --> [GELU derivative] --\
 * B -----/                                      [Mul] --> dz --+--> dA
 * dC ----------------------------------------------------------+
 *                                                             \--> [ReduceSum(leading axes)] --> dB
 * \endverbatim
 *
 * CDist gradient:
 * \verbatim
 * A --> [U1] --\
 *              [Sub] --> diff --\
 * B --> [U0] --/                +--> [Mul] --> weighted --+--> [ReduceSum(1)] --> dA
 * [metric gradient] --> [U2] --/                           \--> [ReduceSum(0)] --> [Neg] --> dB
 * A -----\ [CDist] --> C --\
 * B -----/                  +--> [metric gradient]
 * dC (output gradient) -----/
 * \endverbatim
 */
void RegisterCustomOperatorGradients(ONNX_LIGHT_NAMESPACE::core::gradient::GradRegistry &registry);

} // namespace onnx_light_cpu
