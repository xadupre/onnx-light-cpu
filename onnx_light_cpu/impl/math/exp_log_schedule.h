// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/execution.h"

namespace onnx_light_cpu {

inline constexpr ExecutionSchedule kExpExecutionSchedule{32768, 16384, 2};
inline constexpr ExecutionSchedule kLogExecutionSchedule{131072, 65536, 4};

} // namespace onnx_light_cpu
