// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace onnx_light_cpu {

enum class ArmSimdLevel {
  kNone,
  kNeon,
  kSve,
  kSve2,
};

ArmSimdLevel DetectArmSimdLevel();

} // namespace onnx_light_cpu
