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
 * @code
 *              +-----+
 *   A, B ----> | Add | ----> z
 *              +-----+
 *                 |
 *                 |    +----------+
 *                 +--> | CDF term | --> cdf --------------------------+
 *                 |    +----------+                                   |
 *                 |                                                   |
 *                 |    +------------------+                           |
 *                 +--> | Gaussian density | --> density               |
 *                 |    +------------------+       |                   |
 *                 |                               v                   v
 *                 +---------------------------> +-----+            +-----+
 *                                               | Mul | ---------> | Add | --> derivative
 *                                               +-----+            +-----+
 *                                                                      |
 *                                                                      v
 *                                                                  +-----+
 *   dC ------------------------------------------------------------>| Mul | --> dz --> dA
 *                                                                  +-----+
 *                                                                      |
 *                                                                      v
 *                                                       +------------------------+
 *                                                       | ReduceSum leading axes | --> dB
 *                                                       +------------------------+
 * @endcode
 *
 * The CDF term is ``0.5 * (1 + erf(z / sqrt(2)))`` and the Gaussian density is
 * ``exp(-0.5 * z^2) / sqrt(2 * pi)``. Their combination gives
 * ``derivative = CDF term + z * density``.
 *
 * CDist gradient:
 * @code
 *            +-------+
 *   A, B --> | CDist | ----> C
 *            +-------+         |
 *                              v
 *                         +-------------------+
 *   dC -----------------> | Metric derivative | ----> pair gradient
 *                         +-------------------+
 *                                                       |
 *                                                       v
 *                                                +--------------+
 *                                                | Unsqueeze(2) |
 *                                                +--------------+
 *                                                       |
 *          +--------------+                             |
 *   A ---> | Unsqueeze(1) | ---+                        |
 *          +--------------+    |                        |
 *                              v                        |
 *                           +-----+                      |
 *                           | Sub | ----> difference     |
 *                           +-----+              |       |
 *                              ^                 |       |
 *          +--------------+    |                 v       |
 *   B ---> | Unsqueeze(0) | ---+              +-----+    |
 *          +--------------+                   | Mul | <---+
 *                                             +-----+
 *                                                |
 *                                                +-------------------------+
 *                                                |                         |
 *                                                v                         v
 *                                      +--------------+             +--------------+
 *                                      | ReduceSum(1) | --> dA      | ReduceSum(0) |
 *                                      +--------------+             +--------------+
 *                                                                         |
 *                                                                         v
 *                                                                      +-----+
 *                                                                      | Neg | --> dB
 *                                                                      +-----+
 * @endcode
 *
 * The metric derivative is ``dC / C`` for Euclidean distance (zero where
 * ``C == 0``), and ``2 * dC`` for squared Euclidean distance.
 */
void RegisterCustomOperatorGradients(ONNX_LIGHT_NAMESPACE::core::gradient::GradRegistry &registry);

} // namespace onnx_light_cpu
