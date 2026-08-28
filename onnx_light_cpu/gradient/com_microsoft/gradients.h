// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/gradient/grad_dispatcher.h"

namespace onnx_light_cpu {

/**
 * Registers standard-ONNX gradient graphs for ``com.microsoft`` operators.
 *
 * BiasGelu gradient:
 * \verbatim
 * A -----\
 *         Add --> z --> GELU derivative --\
 * B -----/                                 Mul --> dA
 * dC --------------------------------------/
 *                                           |
 *                                           +--> ReduceSum(leading axes) --> dB
 * \endverbatim
 *
 * CDist gradient:
 * \verbatim
 * A --> Unsq(1) --\
 *                  Sub --> difference --\
 * B --> Unsq(0) --/                    Mul --> weighted --+--> ReduceSum(1) --> dA
 *                                        ^               |
 * dC --> metric scaling --> Unsq(2) ----/               +--> ReduceSum(0) --> Neg --> dB
 * \endverbatim
 */
void RegisterCustomOperatorGradients(ONNX_LIGHT_NAMESPACE::core::gradient::GradRegistry &registry);

} // namespace onnx_light_cpu
