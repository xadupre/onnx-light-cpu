// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/backend_test/test_case_registry.h"

#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {

using ONNX_LIGHT_NAMESPACE::core::backend_test::TestCase;
using ONNX_LIGHT_NAMESPACE::core::backend_test::TestMode;

void RegisterCpuBiasGeluCases(std::vector<TestCase> &registry, TestMode mode);
void RegisterCpuCDistCases(std::vector<TestCase> &registry, TestMode mode);
void RegisterCpuGroupQueryAttentionCases(std::vector<TestCase> &registry, TestMode mode);

void CollectCpuMicrosoftTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                                  TestMode mode = TestMode::TEST);

} // namespace onnx_light_cpu::backend_test
