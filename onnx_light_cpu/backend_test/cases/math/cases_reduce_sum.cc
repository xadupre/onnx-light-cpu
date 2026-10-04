// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::Expect;
using bt_ns::IoData;
using bt_ns::TestCase;
using bt_ns::TestMode;
using rt_ns::DefaultOpset;
using rt_ns::OpsetId;
using rt_ns::Tensor;

} // namespace

void RegisterCpuReduceSumCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    return;
  }
  const OpsetId opset = DefaultOpset(13);
  Expect(registry, ONNX_LIGHT_NAMESPACE::MakeNode("ReduceSum", {"data", "axes"}, {"reduced"}),
         "test_cpu_reducesum_int64", {opset}, []() -> IoData {
           Tensor data = Tensor::FromInt64("", {2, 3}, {1, 2, 3, 4, 5, 6});
           Tensor axes = Tensor::FromInt64("", {1}, {1});
           Tensor reduced = Tensor::FromInt64("", {2, 1}, {6, 15});
           return IoData{{data, axes}, {reduced}};
         });
}

} // namespace onnx_light_cpu::backend_test
