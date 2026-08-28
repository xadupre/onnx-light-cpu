// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace onnx_light_cpu {

/**
 * Fuses an exact Gelu applied to an Add with a rank-one bias.
 *
 * @code
 * Before:
 *                 +-----+
 *   A, bias ----> | Add |
 *                 +-----+
 *                    |
 *                    v
 *                    z
 *                    |
 *                    v
 *                 +------+
 *                 | Gelu | ----> Y
 *                 +------+
 *
 * After:
 *                 +----------+
 *   A, bias ----> | BiasGelu | ----> Y
 *                 +----------+
 * @endcode
 *
 * The bias must broadcast over the last dimension of a floating-point input,
 * and the Add output must be consumed exclusively by the exact Gelu.
 */
class BiasGeluFusionPattern final
    : public ONNX_LIGHT_NAMESPACE::core::builder::PatternOptimization {
public:
  BiasGeluFusionPattern() : PatternOptimization(0, "MicrosoftBiasGelu") {}

  std::set<std::string> FastOpType() const override;
  ONNX_LIGHT_NAMESPACE::core::builder::MatchResult
  Match(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const ONNX_LIGHT_NAMESPACE::NodeProto &candidate) const override;
  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<ONNX_LIGHT_NAMESPACE::NodeProto>
  Apply(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const std::vector<const ONNX_LIGHT_NAMESPACE::NodeProto *> &nodes) const override;
};

/**
 * Fuses the squared-Euclidean distance expansion into CDist.
 *
 * @code
 * Before:
 *          +--------------+
 *   A ---> | Unsqueeze(1) | ---+
 *          +--------------+    |
 *                              v
 *                           +-----+
 *                           | Sub |
 *                           +-----+
 *                              ^
 *          +--------------+    |
 *   B ---> | Unsqueeze(0) | ---+
 *          +--------------+
 *                              |
 *                              v
 *                         difference
 *                              |
 *                              v
 *                           +----------+
 *                           | Mul self |
 *                           +----------+
 *                              |
 *                              v
 *                     squared difference
 *                              |
 *                              v
 *                       +---------------+
 *                       | ReduceSum(-1) | ----> Y
 *                       +---------------+
 *
 * After:
 *              +--------------------------+
 *   A, B ----> | CDist metric=sqeuclidean | ----> Y
 *              +--------------------------+
 * @endcode
 *
 * Both inputs must have rank two. The difference must be squared by an
 * exclusively consumed self-Mul and reduced over the last axis without keeping
 * the reduced dimension.
 */
class CDistFusionPattern final : public ONNX_LIGHT_NAMESPACE::core::builder::PatternOptimization {
public:
  CDistFusionPattern() : PatternOptimization(0, "MicrosoftCDist") {}

  std::set<std::string> FastOpType() const override;
  ONNX_LIGHT_NAMESPACE::core::builder::MatchResult
  Match(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const ONNX_LIGHT_NAMESPACE::NodeProto &candidate) const override;
  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<ONNX_LIGHT_NAMESPACE::NodeProto>
  Apply(ONNX_LIGHT_NAMESPACE::core::builder::GraphGraph &graph,
        const std::vector<const ONNX_LIGHT_NAMESPACE::NodeProto *> &nodes) const override;
};

void RegisterCustomOperatorPatterns();

} // namespace onnx_light_cpu
