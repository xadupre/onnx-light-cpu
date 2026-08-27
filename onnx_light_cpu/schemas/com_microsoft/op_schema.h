// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/light_op_schema/light_op_schema.h"

#include <string>
#include <vector>

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

inline constexpr const char *kMicrosoftDomain = "com.microsoft";

std::vector<ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema>
GetMicrosoftOpSchemasWithHistory(const std::string &op_type = "", bool init_doc = true);

} // namespace onnx_light_cpu
