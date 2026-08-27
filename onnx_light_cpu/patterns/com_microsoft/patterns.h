// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace onnx_light_cpu {

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
