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
 *              ┌─────┐
 *   A, B ─────→│ Add │────→ z
 *              └─────┘
 *                 │
 *                 │    ┌──────────┐
 *                 ├───→│ CDF term │────→ cdf ──────────────────────────┐
 *                 │    └──────────┘                                    │
 *                 │                                                    │
 *                 │    ┌──────────────────┐                            │
 *                 ├───→│ Gaussian density │────→ density               │
 *                 │    └──────────────────┘       │                    │
 *                 │                               ↓                    ↓
 *                 └────────────────────────────→┌─────┐            ┌─────┐
 *                                              │ Mul │────────────→│ Add │───→ derivative
 *                                              └─────┘            └─────┘
 *                                                                      │
 *                                                                      ↓
 *                                                                  ┌─────┐
 *   dC ────────────────────────────────────────────────────────────→│ Mul │───→ dz ───→ dA
 *                                                                  └─────┘
 *                                                                      │
 *                                                                      ↓
 *                                                       ┌────────────────────────┐
 *                                                       │ ReduceSum leading axes │───→ dB
 *                                                       └────────────────────────┘
 * @endcode
 *
 * The CDF term is ``0.5 * (1 + erf(z / sqrt(2)))`` and the Gaussian density is
 * ``exp(-0.5 * z^2) / sqrt(2 * pi)``. Their combination gives
 * ``derivative = CDF term + z * density``.
 *
 * CDist gradient:
 * @code
 *            ┌───────┐
 *   A, B ───→│ CDist │────→ C
 *            └───────┘         │
 *                              ↓
 *                         ┌───────────────────┐
 *   dC ─────────────────→│ Metric derivative │────→ pair gradient
 *                         └───────────────────┘
 *                                                       │
 *                                                       ↓
 *                                                ┌──────────────┐
 *                                                │ Unsqueeze(2) │───────────┐
 *                                                └──────────────┘           │
 *                                                                          │
 *          ┌──────────────┐                                                │
 *   A ────→│ Unsqueeze(1) │───┐                                            │
 *          └──────────────┘   │                                            │
 *                             │                                            │
 *          ┌──────────────┐   │   ┌─────┐                                  │
 *   B ────→│ Unsqueeze(0) │───┴──→│ Sub │────→ difference ───────────────┐ │
 *          └──────────────┘       └─────┘                               │ │
 *                                                                        ↓ ↓
 *                                                                      ┌─────┐
 *                                                                      │ Mul │
 *                                                                      └─────┘
 *                                                                         │
 *                                               ┌─────────────────────────┤
 *                                               │                         │
 *                                               ↓                         ↓
 *                                      ┌──────────────┐             ┌──────────────┐
 *                                      │ ReduceSum(1) │───→ dA      │ ReduceSum(0) │
 *                                      └──────────────┘             └──────────────┘
 *                                                                         │
 *                                                                         ↓
 *                                                                      ┌─────┐
 *                                                                      │ Neg │───→ dB
 *                                                                      └─────┘
 * @endcode
 *
 * The metric derivative is ``dC / C`` for Euclidean distance (zero where
 * ``C == 0``), and ``2 * dC`` for squared Euclidean distance.
 *
 * GroupQueryAttention gradient:
 * @code
 *   Q, K, V ────→┌────────────┐────→┌────────────────────┐────→ probabilities
 *                │ Cast FLOAT │     │ Attention (mode 3) │
 *                └────────────┘     └────────────────────┘
 *
 *   dY ─────────→┌────────────┐────→ float dY
 *                │ Cast FLOAT │
 *                └────────────┘
 *
 *   probabilities, float dY, expanded V
 *                │
 *                ↓
 *          ┌──────────────────┐
 *          │ Softmax backward │
 *          └──────────────────┘
 *                │
 *                ↓
 *          ┌───────────────┐
 *          │ scale dScores │
 *          └───────────────┘
 *             │         │
 *             ↓         ↓
 *       ┌───────────┐ ┌───────────┐
 *       │ MatMul K  │ │ MatMul Q  │
 *       └───────────┘ └───────────┘
 *             │         │
 *             ↓         ↓
 *       ┌────────────┐ ┌───────────────┐
 *       │ reshape dQ │ │ reduce groups │
 *       └────────────┘ └───────────────┘
 *             │          │          │
 *             ↓          ↓          ↓
 *       ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
 *       │ CastLike(Q) │ │ CastLike(K) │ │ CastLike(V) │
 *       └─────────────┘ └─────────────┘ └─────────────┘
 *             │              │               │
 *             ↓              ↓               ↓
 *            dQ             dK              dV
 * @endcode
 *
 * The score scale is replayed from the forward attribute. When omitted, the
 * graph computes ``1 / sqrt(Shape(Q)[2] / num_heads)`` dynamically. Grouped
 * K/V head gradients are summed back to their original head count.
 */
void RegisterCustomOperatorGradients(ONNX_LIGHT_NAMESPACE::core::gradient::GradRegistry &registry);

} // namespace onnx_light_cpu
