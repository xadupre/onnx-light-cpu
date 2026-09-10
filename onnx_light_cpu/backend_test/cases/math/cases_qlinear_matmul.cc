// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/backend_test/cases/math/include_math_cases.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::Expect;
using bt_ns::IoData;
using bt_ns::TestCase;
using bt_ns::TestMode;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::OpsetId;
using rt_ns::Tensor;

template <typename T> T Requantize(std::int64_t accumulator, float scale, std::int32_t zero_point) {
  const double rounded =
      std::nearbyint(static_cast<double>(accumulator) * static_cast<double>(scale));
  const double shifted = rounded + static_cast<double>(zero_point);
  const double clamped = std::clamp(shifted, static_cast<double>(std::numeric_limits<T>::min()),
                                    static_cast<double>(std::numeric_limits<T>::max()));
  return static_cast<T>(clamped);
}

template <typename T> std::vector<T> CastValues(const std::vector<std::int64_t> &values) {
  std::vector<T> casted(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    casted[i] = static_cast<T>(values[i]);
  }
  return casted;
}

template <typename T>
Tensor MakeIntegerTensor(const rt_ns::Shape &shape, const std::vector<std::int64_t> &values) {
  return Tensor::From<T>("", shape, CastValues<T>(values));
}

rt_ns::Shape OutputShape(const rt_ns::Shape &a_shape, const rt_ns::Shape &b_shape) {
  const size_t rank = std::max(a_shape.size(), b_shape.size());
  std::vector<std::int64_t> a_norm(rank, 1);
  std::vector<std::int64_t> b_norm(rank, 1);
  for (size_t i = 0; i < a_shape.size(); ++i) {
    a_norm[rank - a_shape.size() + i] = a_shape[i];
  }
  for (size_t i = 0; i < b_shape.size(); ++i) {
    b_norm[rank - b_shape.size() + i] = b_shape[i];
  }
  std::vector<std::int64_t> y_shape(rank, 1);
  for (size_t axis = 0; axis + 2 < rank; ++axis) {
    y_shape[axis] = std::max(a_norm[axis], b_norm[axis]);
  }
  y_shape[rank - 2] = a_norm[rank - 2];
  y_shape[rank - 1] = b_norm[rank - 1];
  return rt_ns::Shape(std::move(y_shape));
}

template <typename T>
std::vector<T> ComputeQLinearReference(const std::vector<T> &a, const std::vector<T> &b,
                                       const rt_ns::Shape &a_shape, const rt_ns::Shape &b_shape,
                                       std::int32_t a_zero_point, std::int32_t b_zero_point,
                                       float combined_scale, std::int32_t y_zero_point) {
  if (a_shape.size() < 2 || b_shape.size() < 2) {
    throw std::invalid_argument("QLinearMatMul backend references require rank >= 2.");
  }
  const size_t rank = std::max(a_shape.size(), b_shape.size());
  std::vector<std::int64_t> a_norm(rank, 1);
  std::vector<std::int64_t> b_norm(rank, 1);
  for (size_t i = 0; i < a_shape.size(); ++i) {
    a_norm[rank - a_shape.size() + i] = a_shape[i];
  }
  for (size_t i = 0; i < b_shape.size(); ++i) {
    b_norm[rank - b_shape.size() + i] = b_shape[i];
  }
  const std::int64_t m = a_norm[rank - 2];
  const std::int64_t k = a_norm[rank - 1];
  if (b_norm[rank - 2] != k) {
    throw std::invalid_argument("QLinearMatMul backend references expect matching K dimensions.");
  }
  const std::int64_t n = b_norm[rank - 1];
  std::vector<std::int64_t> y_shape(rank, 1);
  y_shape[rank - 2] = m;
  y_shape[rank - 1] = n;
  std::int64_t batch_count = 1;
  for (size_t axis = 0; axis + 2 < rank; ++axis) {
    if (a_norm[axis] != b_norm[axis] && a_norm[axis] != 1 && b_norm[axis] != 1) {
      throw std::invalid_argument("QLinearMatMul backend references found incompatible batches.");
    }
    y_shape[axis] = std::max(a_norm[axis], b_norm[axis]);
    batch_count *= y_shape[axis];
  }
  std::vector<T> output(static_cast<size_t>(batch_count * m * n));
  std::vector<std::int64_t> y_batch_dims(y_shape.begin(), y_shape.end() - 2);
  std::vector<std::int64_t> y_batch_strides(y_batch_dims.size(), 1);
  for (std::ptrdiff_t axis = static_cast<std::ptrdiff_t>(y_batch_dims.size()) - 2; axis >= 0;
       --axis) {
    y_batch_strides[static_cast<size_t>(axis)] = y_batch_strides[static_cast<size_t>(axis + 1)] *
                                                 y_batch_dims[static_cast<size_t>(axis + 1)];
  }
  std::vector<std::int64_t> a_strides(rank, 1);
  std::vector<std::int64_t> b_strides(rank, 1);
  for (std::ptrdiff_t axis = static_cast<std::ptrdiff_t>(rank) - 2; axis >= 0; --axis) {
    a_strides[static_cast<size_t>(axis)] =
        a_strides[static_cast<size_t>(axis + 1)] * a_norm[static_cast<size_t>(axis + 1)];
    b_strides[static_cast<size_t>(axis)] =
        b_strides[static_cast<size_t>(axis + 1)] * b_norm[static_cast<size_t>(axis + 1)];
  }
  for (std::int64_t batch = 0; batch < batch_count; ++batch) {
    std::int64_t a_base = 0;
    std::int64_t b_base = 0;
    std::int64_t residue = batch;
    for (size_t axis = 0; axis < y_batch_dims.size(); ++axis) {
      const std::int64_t coord = y_batch_dims.empty() ? 0 : residue / y_batch_strides[axis];
      residue = y_batch_dims.empty() ? 0 : residue % y_batch_strides[axis];
      if (a_norm[axis] != 1) {
        a_base += coord * a_strides[axis];
      }
      if (b_norm[axis] != 1) {
        b_base += coord * b_strides[axis];
      }
    }
    for (std::int64_t row = 0; row < m; ++row) {
      for (std::int64_t col = 0; col < n; ++col) {
        std::int64_t acc = 0;
        for (std::int64_t depth = 0; depth < k; ++depth) {
          const std::int64_t a_index =
              a_base + row * a_strides[rank - 2] + depth * a_strides.back();
          const std::int64_t b_index =
              b_base + depth * b_strides[rank - 2] + col * b_strides.back();
          const auto av = static_cast<std::int32_t>(a[static_cast<size_t>(a_index)]);
          const auto bv = static_cast<std::int32_t>(b[static_cast<size_t>(b_index)]);
          acc += static_cast<std::int64_t>(av - a_zero_point) *
                 static_cast<std::int64_t>(bv - b_zero_point);
        }
        const std::int64_t out_index = batch * (m * n) + row * n + col;
        output[static_cast<size_t>(out_index)] = Requantize<T>(acc, combined_scale, y_zero_point);
      }
    }
  }
  return output;
}

template <typename T>
void RegisterQLinearCase(std::vector<TestCase> &registry, const OpsetId &opset,
                         const std::string &name_suffix, const rt_ns::Shape &a_shape,
                         const rt_ns::Shape &b_shape, const std::vector<std::int64_t> &a_values,
                         const std::vector<std::int64_t> &b_values, float a_scale, float b_scale,
                         float y_scale, std::int32_t a_zero_point, std::int32_t b_zero_point,
                         std::int32_t y_zero_point) {
  const std::string type = std::is_same_v<T, std::int8_t> ? "int8" : "uint8";
  const std::string name = "test_cpu_qlinearmatmul_" + name_suffix + "_" + type;

  Expect(registry,
         MakeNode("QLinearMatMul",
                  {"a", "a_scale", "a_zero", "b", "b_scale", "b_zero", "y_scale", "y_zero"}, {"y"}),
         name, {opset}, [=]() -> IoData {
           const Tensor a = MakeIntegerTensor<T>(a_shape, a_values);
           const Tensor b = MakeIntegerTensor<T>(b_shape, b_values);
           const Tensor a_s = Tensor::FromFloat("", {}, {a_scale});
           const Tensor b_s = Tensor::FromFloat("", {}, {b_scale});
           const Tensor y_s = Tensor::FromFloat("", {}, {y_scale});
           const Tensor a_z = Tensor::From<T>("", {}, {static_cast<T>(a_zero_point)});
           const Tensor b_z = Tensor::From<T>("", {}, {static_cast<T>(b_zero_point)});
           const Tensor y_z = Tensor::From<T>("", {}, {static_cast<T>(y_zero_point)});
           const float combined_scale = (a_scale * b_scale) / y_scale;
           const std::vector<T> expected_values = ComputeQLinearReference<T>(
               CastValues<T>(a_values), CastValues<T>(b_values), a_shape, b_shape, a_zero_point,
               b_zero_point, combined_scale, y_zero_point);
           const Tensor y = Tensor::From<T>("", OutputShape(a_shape, b_shape), expected_values);
           return IoData{{a, a_s, a_z, b, b_s, b_z, y_s, y_z}, {y}};
         });
}

void RegisterQLinearBenchmark(std::vector<TestCase> &registry, const OpsetId &opset,
                              const std::string &name_suffix, DataType data_type, std::int64_t m,
                              std::int64_t n, std::int64_t k) {
  const std::string name =
      "test_cpu_qlinearmatmul_" + name_suffix + "_" + DataTypeSuffix(data_type) + "_benchmark";
  const std::int64_t a_count = m * k;
  const std::int64_t b_count = k * n;
  const std::int64_t y_count = m * n;
  Expect(registry,
         MakeNode("QLinearMatMul",
                  {"a", "a_scale", "a_zero", "b", "b_scale", "b_zero", "y_scale", "y_zero"}, {"y"}),
         name, {opset}, {a_count, 1, 1, b_count, 1, 1, 1, 1}, {y_count},
         [=](bool generate_expected_outputs) -> IoData {
           Tensor a = MakeBenchmarkTensor(data_type, {m, k}, 9876501);
           Tensor b = MakeBenchmarkTensor(data_type, {k, n}, 9876502);
           Tensor a_s = Tensor::FromFloat("", {}, {0.5f});
           Tensor b_s = Tensor::FromFloat("", {}, {0.25f});
           Tensor y_s = Tensor::FromFloat("", {}, {0.2f});
           Tensor a_z = data_type == DataType::INT8 ? Tensor::FromInt8("", {}, {2})
                                                    : Tensor::FromUint8("", {}, {2});
           Tensor b_z = data_type == DataType::INT8 ? Tensor::FromInt8("", {}, {1})
                                                    : Tensor::FromUint8("", {}, {1});
           Tensor y_z = data_type == DataType::INT8 ? Tensor::FromInt8("", {}, {0})
                                                    : Tensor::FromUint8("", {}, {0});
           if (!generate_expected_outputs) {
             return IoData{{std::move(a), std::move(a_s), std::move(a_z), std::move(b),
                            std::move(b_s), std::move(b_z), std::move(y_s), std::move(y_z)},
                           {},
                           {},
                           false};
           }
           const float combined_scale = (0.5f * 0.25f) / 0.2f;
           if (data_type == DataType::INT8) {
             const std::vector<std::int8_t> expected = ComputeQLinearReference<std::int8_t>(
                 std::vector<std::int8_t>(a.AsInt8(), a.AsInt8() + static_cast<size_t>(a_count)),
                 std::vector<std::int8_t>(b.AsInt8(), b.AsInt8() + static_cast<size_t>(b_count)),
                 {m, k}, {k, n}, 2, 1, combined_scale, 0);
             Tensor y = Tensor::FromInt8("", {m, n}, expected);
             return IoData{{std::move(a), std::move(a_s), std::move(a_z), std::move(b),
                            std::move(b_s), std::move(b_z), std::move(y_s), std::move(y_z)},
                           {std::move(y)}};
           }
           const std::vector<std::uint8_t> expected = ComputeQLinearReference<std::uint8_t>(
               std::vector<std::uint8_t>(a.AsUint8(), a.AsUint8() + static_cast<size_t>(a_count)),
               std::vector<std::uint8_t>(b.AsUint8(), b.AsUint8() + static_cast<size_t>(b_count)),
               {m, k}, {k, n}, 2, 1, combined_scale, 0);
           Tensor y = Tensor::FromUint8("", {m, n}, expected);
           return IoData{{std::move(a), std::move(a_s), std::move(a_z), std::move(b),
                          std::move(b_s), std::move(b_z), std::move(y_s), std::move(y_z)},
                         {std::move(y)}};
         },
         "backend-test", bt_ns::TestCaseTag::NONE,
         {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(data_type), {m, n})});
}

} // namespace

void RegisterCpuQLinearMatMulCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset10 = DefaultOpset(10);
  const OpsetId opset13 = DefaultOpset(13);
  if (mode == TestMode::BENCHMARK) {
    RegisterQLinearBenchmark(registry, opset10, "square_64", DataType::INT8, 64, 64, 64);
    RegisterQLinearBenchmark(registry, opset10, "square_64", DataType::UINT8, 64, 64, 64);
    return;
  }

  RegisterQLinearCase<std::int8_t>(registry, opset10, "scalar_requantized_opset10", {2, 3}, {3, 2},
                                   {3, -4, 9, -2, 1, 5}, {2, -3, 4, 6, -5, 1}, 0.5f, 0.25f, 0.2f, 2,
                                   -1, 3);

  RegisterQLinearCase<std::uint8_t>(registry, opset13, "unsigned_asymmetric_opset13", {1, 4},
                                    {4, 3}, {3, 4, 8, 9}, {6, 1, 4, 8, 2, 5, 3, 7, 9, 4, 1, 2},
                                    0.125f, 0.25f, 0.1f, 3, 4, 5);
  RegisterQLinearCase<std::int8_t>(registry, opset13, "batch_broadcast_opset13", {2, 1, 3},
                                   {1, 3, 2}, {1, -2, 3, 4, -5, 6}, {2, -3, 4, -1, 5, -6}, 0.2f,
                                   0.3f, 0.25f, -1, 2, 3);
}

} // namespace onnx_light_cpu::backend_test
