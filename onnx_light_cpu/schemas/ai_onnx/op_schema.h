// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/light_op_schema/light_op_schema.h"

#include <string>
#include <vector>

namespace onnx_light_cpu {

/// Returns experimental ai.onnx compatibility schemas, separate from standard ONNX schemas.
std::vector<ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema>
GetExperimentalOpSchemasWithHistory(const std::string &op_type = "", bool init_doc = true);

} // namespace onnx_light_cpu
