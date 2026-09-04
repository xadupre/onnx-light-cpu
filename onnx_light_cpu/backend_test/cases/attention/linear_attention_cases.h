// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/kernel_context.h"

#include <cstdint>
#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {

struct LinearAttentionCase {
  const char *name;
  std::int64_t batch;
  std::int64_t sequence;
  std::int64_t query_heads;
  std::int64_t key_heads;
  std::int64_t key_value_heads;
  std::int64_t key_head_size;
  std::int64_t value_head_size;
  const char *rule;
  bool with_past;
  bool decay_per_dimension;
  bool beta_shared;
  ONNX_LIGHT_NAMESPACE::core::runtime::DataType data_type;
  std::uint64_t seed;
};

enum class LinearAttentionCaseContract {
  kOnnx,
  kMicrosoft,
};

void RegisterLinearAttentionCase(
    std::vector<ONNX_LIGHT_NAMESPACE::core::backend_test::TestCase> &registry,
    const LinearAttentionCase &test_case, LinearAttentionCaseContract contract, bool benchmark);

} // namespace onnx_light_cpu::backend_test
