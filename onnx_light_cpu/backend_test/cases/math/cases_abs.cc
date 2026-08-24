// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_light_cpu/kernels/math/abs_kernel.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::Expect;
using bt_ns::IoData;
using bt_ns::TestCase;
using bt_ns::TestMode;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Tensor;

// Builds a single-input / single-output unary ``Abs`` NodeProto.
NodeProto MakeAbsNode() {
  NodeProto node;
  node.set_op_type("Abs");
  node.add_input("x");
  node.add_output("y");
  return node;
}

// Encodes ``values`` as a BFLOAT16 ``Tensor`` (raw 16-bit bit patterns),
// mirroring how onnx-light's own backend test cases build bfloat16 inputs.
Tensor MakeBfloat16Tensor(const std::vector<int64_t> &shape, const std::vector<float> &values) {
  std::vector<std::uint8_t> raw(values.size() * sizeof(std::uint16_t));
  auto *dst = reinterpret_cast<std::uint16_t *>(raw.data());
  for (std::size_t i = 0; i < values.size(); ++i) {
    dst[i] = rt_ns::FloatToBfloat16Bits(values[i]);
  }
  return Tensor("", static_cast<std::int32_t>(DataType::BFLOAT16), shape, std::move(raw));
}

} // namespace

// Abs — covers every element type ``AbsKernel`` implements:
// float32, float64, int8, int16, int32, int64, float16, bfloat16.
void RegisterCpuAbsCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const onnx_light_cpu::AbsKernel abs_kernel{KernelContext{opset}};
  const std::vector<int64_t> shape = {2, 3};
  const std::vector<float> f = {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f};

  if (mode == TestMode::BENCHMARK) {
    for (const int64_t size : {1024, 32768, 65535, 65536, 131072, 1048576, 4194304}) {
      for (const DataType data_type :
           {DataType::FLOAT, DataType::DOUBLE, DataType::INT8, DataType::INT16, DataType::INT32,
            DataType::INT64, DataType::FLOAT16, DataType::BFLOAT16}) {
        RegisterUnaryBenchmark(registry, "Abs", abs_kernel, opset, data_type, size);
      }
    }
    return;
  }

  Expect(registry, MakeAbsNode(), "test_cpu_abs_float32", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromFloat("", shape, f);
    return IoData{{x}, {abs_kernel(x)}};
  });
  Expect(registry, MakeAbsNode(), "test_cpu_abs_float64", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromDouble("", shape, {-1.0, 0.0, 1.5, -2.25, 3.5, -4.75});
    return IoData{{x}, {abs_kernel(x)}};
  });
  Expect(registry, MakeAbsNode(), "test_cpu_abs_int8", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromInt8("", shape, {-1, 0, 2, -127, 3, -5});
    return IoData{{x}, {abs_kernel(x)}};
  });
  Expect(registry, MakeAbsNode(), "test_cpu_abs_int16", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromInt16("", shape, {-1, 0, 2, -1000, 3, -5});
    return IoData{{x}, {abs_kernel(x)}};
  });
  Expect(registry, MakeAbsNode(), "test_cpu_abs_int32", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromInt32("", shape, {-1, 0, 2, -100000, 3, -5});
    return IoData{{x}, {abs_kernel(x)}};
  });
  Expect(registry, MakeAbsNode(), "test_cpu_abs_int64", {opset}, [=]() -> IoData {
    Tensor x = Tensor::FromInt64("", shape, {-1, 0, 2, -1000000000000LL, 3, -5});
    return IoData{{x}, {abs_kernel(x)}};
  });
  Expect(registry, MakeAbsNode(), "test_cpu_abs_float16", {opset}, [=]() -> IoData {
    Tensor x = rt_ns::MakeFloat16Tensor("", shape, f);
    return IoData{{x}, {abs_kernel(x)}};
  });
  Expect(registry, MakeAbsNode(), "test_cpu_abs_bfloat16", {opset}, [=]() -> IoData {
    Tensor x = MakeBfloat16Tensor(shape, f);
    return IoData{{x}, {abs_kernel(x)}};
  });
}

} // namespace onnx_light_cpu::backend_test
