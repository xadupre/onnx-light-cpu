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

/// Registers the ``ai.onnx::Attention`` (v23/v24) backend test corpus:
/// stateless FP32 MHA/GQA/MQA, rank-3/rank-4 layouts, no/causal/boolean/
/// additive masks. Names encode opset, layout, geometry, query/KV lengths,
/// head dimension, cache/mask mode and element type, e.g.
/// ``test_cpu_attention_opset23_rank4_mha_q128_kv128_hd64_none_float32``.
void RegisterCpuAttentionCases(std::vector<TestCase> &registry, TestMode mode);

void CollectCpuAttentionTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                                  TestMode mode = TestMode::TEST);

} // namespace onnx_light_cpu::backend_test
