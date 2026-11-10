// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/gemm_kernel.h"

#include "ml_cpu_compute/execution.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/tensor_compare.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext MakeCtx(int64_t version = 18) {
  return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), version));
}

void ConfigureTuning(onnx_light_cpu::GemmKernel &kernel, rt_ns::DataType data_type) {
  onnx_light_cpu::GemmKernel::RegisterTuningSchemas();
  const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(
      kernel.TuningKey(static_cast<int32_t>(data_type)));
  ASSERT_NE(schema, nullptr);
  auto parameters = schema->portable_defaults();
  for (const char *name :
       {"blocking.mc", "blocking.nc", "blocking.kc", "compact_blocking.mc", "compact_blocking.nc",
        "compact_blocking.kc", "parallel.maximum_threads"}) {
    parameters.values[name] = int64_t{1};
  }
  kernel.Configure(parameters);
}

void ConfigureMaximumParticipants(onnx_light_cpu::GemmKernel &kernel, rt_ns::DataType data_type,
                                  std::int64_t maximum) {
  onnx_light_cpu::GemmKernel::RegisterTuningSchemas();
  const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(
      kernel.TuningKey(static_cast<int32_t>(data_type)));
  ASSERT_NE(schema, nullptr);
  auto parameters = schema->portable_defaults();
  parameters.values["parallel.maximum_threads"] = maximum;
  kernel.Configure(parameters);
}

// Straightforward reference Gemm: Y = alpha * op(A) @ op(B) + beta * C, with C
// broadcast (here only the full M x N case is exercised).
std::vector<float> ReferenceGemm(bool trans_a, bool trans_b, std::size_t M, std::size_t N,
                                 std::size_t K, float alpha, const std::vector<float> &A,
                                 const std::vector<float> &B, float beta,
                                 const std::vector<float> *C) {
  std::vector<float> Y(M * N, 0.0f);
  for (std::size_t m = 0; m < M; ++m) {
    for (std::size_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (std::size_t k = 0; k < K; ++k) {
        const float a = trans_a ? A[k * M + m] : A[m * K + k];
        const float b = trans_b ? B[n * K + k] : B[k * N + n];
        acc += a * b;
      }
      float value = alpha * acc;
      if (C != nullptr && beta != 0.0f) {
        value += beta * (*C)[m * N + n];
      }
      Y[m * N + n] = value;
    }
  }
  return Y;
}

TEST(OnnxLightGemmKernel, RegistersAndValidatesTuningSchemas) {
  EXPECT_NO_THROW(onnx_light_cpu::GemmKernel::RegisterTuningSchemas());
  EXPECT_NO_THROW(onnx_light_cpu::GemmKernel::RegisterTuningSchemas());

  for (rt_ns::DataType data_type : {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE,
                                    rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    onnx_light_cpu::GemmKernel kernel(MakeCtx());
    const auto key = kernel.TuningKey(static_cast<int32_t>(data_type));
    EXPECT_EQ(key.library, "onnx_light_cpu");
    EXPECT_EQ(key.kernel, "Gemm");
    EXPECT_EQ(key.implementation, "simd_dispatch");
    EXPECT_EQ(key.tuning_abi, onnx_light_cpu::GemmKernel::kTuningAbi);

    const auto schema = rt_ns::GetKernelTuningRegistry().FindSchema(key);
    ASSERT_NE(schema, nullptr);
    auto parameters = schema->portable_defaults();
    for (const char *name :
         {"blocking.mc", "blocking.nc", "blocking.kc", "compact_blocking.mc", "compact_blocking.nc",
          "compact_blocking.kc", "parallel.maximum_threads"}) {
      EXPECT_EQ(parameters.Get<int64_t>(name), 0);
      parameters.values[name] = int64_t{7};
    }
    EXPECT_NO_THROW(schema->Validate(parameters));
    EXPECT_NO_THROW(kernel.Configure(parameters));

    parameters.values["blocking.mc"] = int64_t{-1};
    EXPECT_THROW(schema->Validate(parameters), std::invalid_argument);
    EXPECT_THROW(kernel.Configure(parameters), std::invalid_argument);
  }

  onnx_light_cpu::GemmKernel kernel(MakeCtx());
  EXPECT_EQ(kernel.TuningKey(static_cast<int32_t>(rt_ns::DataType::INT32)).device,
            ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kUndefined);
}

TEST(OnnxLightGemmKernel, ConfigurationChangeRebuildsCachedPlan) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Gemm");
  node.add_input("A");
  node.add_input("B");
  node.add_output("Y");
  rt_ns::RuntimeContext runtime(MakeCtx());
  runtime.Set("A", rt_ns::Tensor::FromFloat("A", {2, 2}, {1, 2, 3, 4}));
  runtime.Set("B", rt_ns::Tensor::FromFloat("B", {2, 2}, {1, 0, 0, 1}));

  onnx_light_cpu::GemmKernel kernel(MakeCtx());
  kernel.set_node(node);
  kernel.Run(runtime);
  EXPECT_FLOAT_EQ(runtime.Get("Y").AsFloat()[3], 4.0f);

  ConfigureTuning(kernel, rt_ns::DataType::FLOAT);
  kernel.Run(runtime);
  EXPECT_FLOAT_EQ(runtime.Get("Y").AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(runtime.Get("Y").AsFloat()[3], 4.0f);
}

TEST(OnnxLightGemmKernel, Float32Matmul) {
  onnx_light_cpu::GemmKernel kernel(MakeCtx());
  ConfigureTuning(kernel, rt_ns::DataType::FLOAT);
  const std::size_t M = 2, N = 3, K = 4;
  const std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<float> b = {1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0};
  const rt_ns::Tensor A = rt_ns::Tensor::FromFloat("A", {2, 4}, a);
  const rt_ns::Tensor B = rt_ns::Tensor::FromFloat("B", {4, 3}, b);
  const auto expected = ReferenceGemm(false, false, M, N, K, 1.0f, a, b, 0.0f, nullptr);
  const rt_ns::Tensor y = kernel(A, B, 1.0f, false, false);
  ASSERT_EQ(y.element_count(), static_cast<int64_t>(M * N));
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(OnnxLightGemmKernel, Float32AlphaBetaBias) {
  onnx_light_cpu::GemmKernel kernel(MakeCtx());
  const std::size_t M = 2, N = 2, K = 3;
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  const std::vector<float> b = {1, 2, 3, 4, 5, 6};
  const std::vector<float> c = {10, 20, 30, 40};
  const rt_ns::Tensor A = rt_ns::Tensor::FromFloat("A", {2, 3}, a);
  const rt_ns::Tensor B = rt_ns::Tensor::FromFloat("B", {3, 2}, b);
  const rt_ns::Tensor C = rt_ns::Tensor::FromFloat("C", {2, 2}, c);
  const float alpha = 0.5f, beta = 2.0f;
  const auto expected = ReferenceGemm(false, false, M, N, K, alpha, a, b, beta, &c);
  const rt_ns::Tensor y = kernel(A, B, C, alpha, beta, false, false);
  ASSERT_EQ(y.element_count(), static_cast<int64_t>(M * N));
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

// C is a 1-D vector of length N and must be broadcast across the M rows.
TEST(OnnxLightGemmKernel, Float32BiasBroadcastVector) {
  onnx_light_cpu::GemmKernel kernel(MakeCtx());
  const std::size_t M = 2, N = 2, K = 2;
  const std::vector<float> a = {1, 2, 3, 4};
  const std::vector<float> b = {1, 0, 0, 1};
  const std::vector<float> c = {5, 7}; // shape (2,)
  std::vector<float> c_full = {5, 7, 5, 7};
  const rt_ns::Tensor A = rt_ns::Tensor::FromFloat("A", {2, 2}, a);
  const rt_ns::Tensor B = rt_ns::Tensor::FromFloat("B", {2, 2}, b);
  const rt_ns::Tensor C = rt_ns::Tensor::FromFloat("C", {2}, c);
  const auto expected = ReferenceGemm(false, false, M, N, K, 1.0f, a, b, 1.0f, &c_full);
  const rt_ns::Tensor y = kernel(A, B, C, 1.0f, 1.0f, false, false);
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(OnnxLightGemmKernel, Float32BiasBroadcastScalar) {
  onnx_light_cpu::GemmKernel kernel(MakeCtx());
  const std::vector<float> a = {1, 2, 3, 4};
  const std::vector<float> b = {1, 0, 0, 1};
  const std::vector<float> c = {5};
  const std::vector<float> c_full(4, 5);
  const rt_ns::Tensor A = rt_ns::Tensor::FromFloat("A", {2, 2}, a);
  const rt_ns::Tensor B = rt_ns::Tensor::FromFloat("B", {2, 2}, b);
  const rt_ns::Tensor C = rt_ns::Tensor::FromFloat("C", {}, c);
  const auto expected = ReferenceGemm(false, false, 2, 2, 2, 1.0f, a, b, 2.0f, &c_full);

  const rt_ns::Tensor y = kernel(A, B, C, 1.0f, 2.0f, false, false);

  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(OnnxLightGemmKernel, Float32BiasBroadcastColumn) {
  onnx_light_cpu::GemmKernel kernel(MakeCtx());
  const std::vector<float> a = {1, 2, 3, 4};
  const std::vector<float> b = {1, 0, 0, 1};
  const std::vector<float> c = {5, 7};
  const std::vector<float> c_full = {5, 5, 7, 7};
  const rt_ns::Tensor A = rt_ns::Tensor::FromFloat("A", {2, 2}, a);
  const rt_ns::Tensor B = rt_ns::Tensor::FromFloat("B", {2, 2}, b);
  const rt_ns::Tensor C = rt_ns::Tensor::FromFloat("C", {2, 1}, c);
  const auto expected = ReferenceGemm(false, false, 2, 2, 2, 0.5f, a, b, 3.0f, &c_full);

  const rt_ns::Tensor y = kernel(A, B, C, 0.5f, 3.0f, false, false);

  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(OnnxLightGemmKernel, Float32TransposeVariants) {
  onnx_light_cpu::GemmKernel kernel(MakeCtx());
  const std::size_t M = 2, N = 2, K = 3;
  // A stored transposed as K x M, B stored transposed as N x K.
  const std::vector<float> a_t = {1, 4, 2, 5, 3, 6}; // K x M
  const std::vector<float> b_t = {1, 2, 3, 4, 5, 6}; // N x K
  const rt_ns::Tensor A = rt_ns::Tensor::FromFloat("A", {3, 2}, a_t);
  const rt_ns::Tensor B = rt_ns::Tensor::FromFloat("B", {2, 3}, b_t);
  const auto expected = ReferenceGemm(true, true, M, N, K, 1.0f, a_t, b_t, 0.0f, nullptr);
  const rt_ns::Tensor y = kernel(A, B, 1.0f, true, true);
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

// FLOAT16 support: A, B and the bias C are rounded to fp16 before the call
// (as they would be when decoded from a FLOAT16 tensor), so the reduction
// happening in float32 internally reproduces the reference bit-exactly once
// rounded back to fp16.
TEST(OnnxLightGemmKernel, Float16AlphaBetaBias) {
  onnx_light_cpu::GemmKernel kernel(MakeCtx());
  ConfigureTuning(kernel, rt_ns::DataType::FLOAT16);
  const std::size_t M = 2, N = 2, K = 3;
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  const std::vector<float> b = {1, 2, 3, 4, 5, 6};
  const std::vector<float> c = {10, 20, 30, 40};
  const float alpha = 0.5f, beta = 2.0f;

  const rt_ns::Tensor A = rt_ns::MakeFloat16Tensor("A", {2, 3}, a);
  const rt_ns::Tensor B = rt_ns::MakeFloat16Tensor("B", {3, 2}, b);
  const rt_ns::Tensor C = rt_ns::MakeFloat16Tensor("C", {2, 2}, c);
  const rt_ns::Tensor y = kernel(A, B, C, alpha, beta, false, false);
  ASSERT_EQ(y.element_count(), static_cast<int64_t>(M * N));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(rt_ns::DataType::FLOAT16));

  const auto expected = ReferenceGemm(false, false, M, N, K, alpha, a, b, beta, &c);
  const auto *py = reinterpret_cast<const std::uint16_t *>(y.bytes());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(rt_ns::Float16BitsToFloat(py[i]), expected[i], 5e-2f) << "i=" << i;
  }
}

TEST(OnnxLightGemmKernel, Float16BroadcastRowBias) {
  onnx_light_cpu::GemmKernel kernel(MakeCtx());
  const std::vector<float> a = {1, 2, 3, 4};
  const std::vector<float> b = {1, 0, 0, 1};
  const std::vector<float> c = {5, 7};
  const std::vector<float> c_full = {5, 7, 5, 7};
  const rt_ns::Tensor A = rt_ns::MakeFloat16Tensor("A", {2, 2}, a);
  const rt_ns::Tensor B = rt_ns::MakeFloat16Tensor("B", {2, 2}, b);
  const rt_ns::Tensor C = rt_ns::MakeFloat16Tensor("C", {2}, c);
  const auto expected = ReferenceGemm(false, false, 2, 2, 2, 1.0f, a, b, 2.0f, &c_full);

  const rt_ns::Tensor y = kernel(A, B, C, 1.0f, 2.0f, false, false);

  const auto *py = reinterpret_cast<const std::uint16_t *>(y.bytes());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(rt_ns::Float16BitsToFloat(py[i]), expected[i], 5e-2f) << "i=" << i;
  }
}

// BFLOAT16 support, mirroring the FLOAT16 test above.
TEST(OnnxLightGemmKernel, Bfloat16AlphaBetaBias) {
  onnx_light_cpu::GemmKernel kernel(MakeCtx());
  ConfigureTuning(kernel, rt_ns::DataType::BFLOAT16);
  const std::size_t M = 2, N = 2, K = 3;
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  const std::vector<float> b = {1, 2, 3, 4, 5, 6};
  const std::vector<float> c = {10, 20, 30, 40};
  const float alpha = 0.5f, beta = 2.0f;

  const rt_ns::Tensor A = rt_ns::MakeBfloat16Tensor("A", {2, 3}, a);
  const rt_ns::Tensor B = rt_ns::MakeBfloat16Tensor("B", {3, 2}, b);
  const rt_ns::Tensor C = rt_ns::MakeBfloat16Tensor("C", {2, 2}, c);
  const rt_ns::Tensor y = kernel(A, B, C, alpha, beta, false, false);
  ASSERT_EQ(y.element_count(), static_cast<int64_t>(M * N));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(rt_ns::DataType::BFLOAT16));

  const auto expected = ReferenceGemm(false, false, M, N, K, alpha, a, b, beta, &c);
  const auto *py = reinterpret_cast<const std::uint16_t *>(y.bytes());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(rt_ns::Bfloat16BitsToFloat(py[i]), expected[i], 5e-1f) << "i=" << i;
  }
}

// FLOAT16 without a bias, exercising the no-bias operator() overload.
TEST(OnnxLightGemmKernel, Float16NoBias) {
  onnx_light_cpu::GemmKernel kernel(MakeCtx());
  const std::size_t M = 2, N = 3, K = 4;
  const std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<float> b = {1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0};
  const rt_ns::Tensor A = rt_ns::MakeFloat16Tensor("A", {2, 4}, a);
  const rt_ns::Tensor B = rt_ns::MakeFloat16Tensor("B", {4, 3}, b);
  const rt_ns::Tensor y = kernel(A, B, 1.0f, false, false);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(rt_ns::DataType::FLOAT16));

  const auto expected = ReferenceGemm(false, false, M, N, K, 1.0f, a, b, 0.0f, nullptr);
  const auto *py = reinterpret_cast<const std::uint16_t *>(y.bytes());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(rt_ns::Float16BitsToFloat(py[i]), expected[i], 5e-2f) << "i=" << i;
  }
}

TEST(OnnxLightGemmKernel, Float64Matmul) {
  onnx_light_cpu::GemmKernel kernel(MakeCtx());
  ConfigureTuning(kernel, rt_ns::DataType::DOUBLE);
  const std::vector<double> a = {1, 2, 3, 4};
  const std::vector<double> b = {5, 6, 7, 8};
  const rt_ns::Tensor A = rt_ns::Tensor::FromDouble("A", {2, 2}, a);
  const rt_ns::Tensor B = rt_ns::Tensor::FromDouble("B", {2, 2}, b);
  const rt_ns::Tensor y = kernel(A, B, 1.0f, false, false);
  ASSERT_EQ(y.element_count(), 4);
  const double *py = y.AsDouble();
  // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
  EXPECT_DOUBLE_EQ(py[0], 19.0);
  EXPECT_DOUBLE_EQ(py[1], 22.0);
  EXPECT_DOUBLE_EQ(py[2], 43.0);
  EXPECT_DOUBLE_EQ(py[3], 50.0);
}

rt_ns::Tensor MakeGemmInput(rt_ns::DataType type, const rt_ns::Shape &shape, int seed,
                            bool nan = false) {
  const auto count = static_cast<std::size_t>(shape.product());
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = nan ? std::numeric_limits<float>::quiet_NaN()
                    : static_cast<float>(static_cast<int>((i * 13 + seed) % 17) - 8) / 4.0f;
  }
  switch (type) {
  case rt_ns::DataType::FLOAT:
    return rt_ns::Tensor::FromFloat("", shape, values);
  case rt_ns::DataType::DOUBLE:
    return rt_ns::Tensor::FromDouble("", shape, std::vector<double>(values.begin(), values.end()));
  case rt_ns::DataType::FLOAT16:
    return rt_ns::MakeFloat16Tensor("", shape, values);
  case rt_ns::DataType::BFLOAT16:
    return rt_ns::MakeBfloat16Tensor("", shape, values);
  default:
    throw std::invalid_argument("Unsupported Gemm test dtype.");
  }
}

double GemmValue(const rt_ns::Tensor &tensor, std::size_t index) {
  switch (static_cast<rt_ns::DataType>(tensor.data_type)) {
  case rt_ns::DataType::FLOAT:
    return tensor.AsFloat()[index];
  case rt_ns::DataType::DOUBLE:
    return tensor.AsDouble()[index];
  case rt_ns::DataType::FLOAT16:
    return rt_ns::Float16BitsToFloat(
        reinterpret_cast<const std::uint16_t *>(tensor.bytes())[index]);
  case rt_ns::DataType::BFLOAT16:
    return rt_ns::Bfloat16BitsToFloat(
        reinterpret_cast<const std::uint16_t *>(tensor.bytes())[index]);
  default:
    throw std::invalid_argument("Unsupported Gemm test dtype.");
  }
}

struct GemmExecutorRecorder {
  std::int64_t maximum_blocks = 0;

  static void Run(void *context, std::int64_t blocks, void *task_context,
                  ml_cpu_compute::ExecutionBlockFn task) {
    auto &executor = *static_cast<GemmExecutorRecorder *>(context);
    executor.maximum_blocks = std::max(executor.maximum_blocks, blocks);
    for (std::int64_t block = 0; block < blocks; ++block) {
      task(task_context, block);
    }
  }
};

void ExpectGemmMatches(const rt_ns::Tensor &actual, const rt_ns::Tensor &expected) {
  ASSERT_EQ(actual.data_type, expected.data_type);
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.element_count(), expected.element_count());
  const double tolerance = actual.data_type == rt_ns::DataType::BFLOAT16  ? 0.02
                           : actual.data_type == rt_ns::DataType::FLOAT16 ? 0.002
                           : actual.data_type == rt_ns::DataType::FLOAT   ? 1e-5
                                                                          : 1e-10;
  for (std::int64_t i = 0; i < actual.element_count(); ++i) {
    const double reference = GemmValue(expected, static_cast<std::size_t>(i));
    const double value = GemmValue(actual, static_cast<std::size_t>(i));
    if (std::isnan(reference)) {
      EXPECT_TRUE(std::isnan(value)) << "element " << i;
    } else if (std::isinf(reference)) {
      EXPECT_EQ(value, reference) << "element " << i;
    } else {
      EXPECT_NEAR(value, reference, tolerance * std::max(1.0, std::abs(reference)))
          << "element " << i;
    }
  }
}

TEST(OnnxLightGemmKernel, Gemm29GeneralizedShapesMatchBuiltin) {
  struct Case {
    const char *name;
    rt_ns::Shape a;
    rt_ns::Shape b;
    rt_ns::Shape c;
    bool has_bias;
    bool trans_a;
    bool trans_b;
    float beta;
    bool nan_bias;
  };
  const std::vector<Case> cases = {
      {"vector_transpose", {4}, {3, 4}, {3}, true, true, true, -0.75f, false},
      {"batch_row_bias", {2, 3, 4}, {4, 5}, {2, 1, 5}, true, false, false, -0.75f, false},
      {"broadcast_column", {2, 3, 4, 5}, {6, 5}, {3, 4, 1}, true, false, true, -0.75f, false},
      {"transpose_batches", {2, 3, 5, 4}, {6, 5}, {2, 1, 1, 6}, true, true, true, -0.75f, false},
      {"broadcast_matrices", {2, 3, 4, 5}, {5, 6}, {2, 1, 4, 6}, true, false, false, -0.75f, false},
      {"batch_scalars", {2, 3, 4, 5}, {5, 6}, {1, 3, 1, 1}, true, false, false, -0.75f, false},
      {"rank5_bias", {2, 3, 2, 4, 5}, {5, 6}, {3, 1, 4, 6}, true, false, false, -0.75f, false},
      {"no_bias", {2, 3, 4}, {4, 5}, {}, false, false, false, 0.0f, false},
      {"empty_batch", {0, 3, 4}, {4, 5}, {1, 1, 5}, true, false, false, -0.75f, false},
      {"empty_rows", {2, 0, 4}, {4, 5}, {2, 0, 1}, true, false, false, -0.75f, false},
      {"empty_features", {2, 3, 0}, {0, 5}, {2, 1, 5}, true, false, false, -0.75f, false},
      {"empty_output", {2, 3, 4}, {4, 0}, {1, 0}, true, false, false, -0.75f, false},
      {"zero_beta", {2, 3, 4}, {4, 5}, {2, 1, 5}, true, false, false, 0.0f, true},
  };
  for (const auto &test : cases) {
    for (const auto type : {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE,
                            rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
      SCOPED_TRACE(test.name);
      SCOPED_TRACE(static_cast<int>(type));
      const auto a = MakeGemmInput(type, test.a, 1);
      const auto b = MakeGemmInput(type, test.b, 2);
      const auto c = MakeGemmInput(type, test.c, 3, test.nan_bias);
      const auto *bias = test.has_bias ? &c : nullptr;
      const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Gemm builtin(MakeCtx(29));
      const auto expected = builtin(a, b, bias, 0.5f, test.beta, test.trans_a, test.trans_b);
      const onnx_light_cpu::GemmKernel kernel(MakeCtx(29));
      const auto actual = test.has_bias
                              ? kernel(a, b, c, 0.5f, test.beta, test.trans_a, test.trans_b)
                              : kernel(a, b, 0.5f, test.trans_a, test.trans_b);
      ExpectGemmMatches(actual, expected);
    }
  }
}

TEST(OnnxLightGemmKernel, Gemm29ZeroDepthPreservesAlphaSemantics) {
  for (const auto type : {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
                          rt_ns::DataType::BFLOAT16}) {
    for (const float alpha :
         {-1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
      for (const bool has_bias : {false, true}) {
        SCOPED_TRACE(static_cast<int>(type));
        SCOPED_TRACE(alpha);
        SCOPED_TRACE(has_bias);
        const auto a = MakeGemmInput(type, {2, 3, 0}, 1);
        const auto b = MakeGemmInput(type, {0, 5}, 2);
        const auto c = MakeGemmInput(type, {5}, 3);
        const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Gemm builtin(MakeCtx(29));
        const auto expected = builtin(a, b, has_bias ? &c : nullptr, alpha, 0.75f, 0, 0);
        const onnx_light_cpu::GemmKernel kernel(MakeCtx(29));
        const auto actual = has_bias ? kernel(a, b, c, alpha, 0.75f, false, false)
                                     : kernel(a, b, alpha, false, false);
        ExpectGemmMatches(actual, expected);
        if (!has_bias && alpha == -1.0f) {
          EXPECT_TRUE(std::signbit(GemmValue(actual, 0)));
        }
      }
    }
  }
}

TEST(OnnxLightGemmKernel, GemmParticipantLimitBoundsBatchedEpilogues) {
  for (const auto type : {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE}) {
    SCOPED_TRACE(static_cast<int>(type));
    GemmExecutorRecorder recorder;
    ml_cpu_compute::ExecutionExecutorView executor{&recorder, 4, &GemmExecutorRecorder::Run};
    ml_cpu_compute::ExecutionExecutorScope executor_scope(&executor);
    onnx_light_cpu::GemmKernel kernel(MakeCtx(29));
    ConfigureMaximumParticipants(kernel, type, 1);

    const auto zero_depth_a = MakeGemmInput(type, {2, 513, 0}, 1);
    const auto zero_depth_b = MakeGemmInput(type, {0, 512}, 2);
    const auto zero_depth_output = kernel(zero_depth_a, zero_depth_b, 0.75f, false, false);
    EXPECT_EQ(zero_depth_output.shape, (rt_ns::Shape{2, 513, 512}));
    EXPECT_LE(recorder.maximum_blocks, 1);

    const auto a = MakeGemmInput(type, {2, 513, 8}, 1);
    const auto b = MakeGemmInput(type, {8, 512}, 2);
    const auto c = MakeGemmInput(type, {2, 1, 512}, 3);
    const auto output = kernel(a, b, c, 0.75f, 0.5f, false, false);
    EXPECT_EQ(output.shape, (rt_ns::Shape{2, 513, 512}));
    EXPECT_LE(recorder.maximum_blocks, 1);
    EXPECT_TRUE(std::isfinite(GemmValue(output, 0)));
  }
}

TEST(OnnxLightGemmKernel, Gemm29RejectsInvalidInputsAndKeepsLegacyRankRules) {
  const onnx_light_cpu::GemmKernel legacy(MakeCtx(28));
  const onnx_light_cpu::GemmKernel modern(MakeCtx(29));
  const auto vector = rt_ns::Tensor::FromFloat("A", {2}, {1, 2});
  const auto matrix = rt_ns::Tensor::FromFloat("B", {2, 2}, {1, 0, 0, 1});
  EXPECT_THROW(legacy(vector, matrix, 1.0f, false, false), std::invalid_argument);

  const auto scalar = rt_ns::Tensor::FromFloat("A", {}, {1});
  EXPECT_THROW(modern(scalar, matrix, 1.0f, false, false), std::invalid_argument);
  const auto a = MakeGemmInput(rt_ns::DataType::FLOAT, {2, 3, 4}, 1);
  const auto b = MakeGemmInput(rt_ns::DataType::FLOAT, {4, 5}, 2);
  const auto invalid_bias = MakeGemmInput(rt_ns::DataType::FLOAT, {2, 2}, 3);
  EXPECT_THROW(modern(a, b, invalid_bias, 1.0f, 0.0f, false, false), std::invalid_argument);
  const auto wrong_b = MakeGemmInput(rt_ns::DataType::FLOAT, {1, 4, 5}, 2);
  EXPECT_THROW(modern(a, wrong_b, 1.0f, false, false), std::invalid_argument);

  auto short_a = rt_ns::Tensor::FromFloat("A", {1}, {1});
  short_a.shape = {2, 3, 4};
  EXPECT_THROW(modern(short_a, b, 1.0f, false, false), std::invalid_argument);
  auto short_b = rt_ns::Tensor::FromFloat("B", {1}, {1});
  short_b.shape = {4, 5};
  EXPECT_THROW(modern(a, short_b, 1.0f, false, false), std::invalid_argument);
  auto short_c = rt_ns::Tensor::FromFloat("C", {1}, {1});
  short_c.shape = {2, 1, 5};
  EXPECT_THROW(modern(a, b, short_c, 1.0f, 1.0f, false, false), std::invalid_argument);
  auto negative_dimension = rt_ns::Tensor::FromFloat("A", {1}, {1});
  negative_dimension.shape = {2, -1, 4};
  EXPECT_THROW(modern(negative_dimension, b, 1.0f, false, false), std::invalid_argument);
}

TEST(OnnxLightGemmKernel, Gemm29RuntimeCacheHandlesChangingShapes) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Gemm");
  node.add_input("A");
  node.add_input("B");
  node.add_output("Y");
  onnx_light_cpu::GemmKernel kernel(MakeCtx(29));
  kernel.set_node(node);
  rt_ns::RuntimeContext runtime(MakeCtx(29));
  const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Gemm builtin(MakeCtx(29));
  for (const auto &[a_shape, b_shape] : {std::pair{rt_ns::Shape{2, 2, 3}, rt_ns::Shape{3, 4}},
                                         std::pair{rt_ns::Shape{3, 2, 3}, rt_ns::Shape{3, 4}},
                                         std::pair{rt_ns::Shape{3}, rt_ns::Shape{3, 4}},
                                         std::pair{rt_ns::Shape{3, 2, 2}, rt_ns::Shape{2, 5}}}) {
    const auto a = MakeGemmInput(rt_ns::DataType::FLOAT, a_shape, 1);
    const auto b = MakeGemmInput(rt_ns::DataType::FLOAT, b_shape, 2);
    const auto expected = builtin(a, b, nullptr, 1.0f, 1.0f, 0, 0);
    runtime.Put("A", a);
    runtime.Put("B", b);
    kernel.Run(runtime);
    ExpectGemmMatches(runtime.Get("Y"), expected);
  }
}

rt_ns::Tensor GemmTensor(rt_ns::DataType dtype, const rt_ns::Shape &shape,
                         const std::vector<float> &values) {
  switch (dtype) {
  case rt_ns::DataType::DOUBLE:
    return rt_ns::Tensor::FromDouble("", shape, std::vector<double>(values.begin(), values.end()));
  case rt_ns::DataType::FLOAT16:
    return rt_ns::MakeFloat16Tensor("", shape, values);
  case rt_ns::DataType::BFLOAT16:
    return rt_ns::MakeBfloat16Tensor("", shape, values);
  default:
    return rt_ns::Tensor::FromFloat("", shape, values);
  }
}

void CheckGemm29(const rt_ns::Tensor &a, const rt_ns::Tensor &b, const rt_ns::Tensor *c,
                 bool trans_a, bool trans_b, float beta = 0.5f) {
  const rt_ns::KernelContext ctx(rt_ns::DefaultOpset(29));
  const ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Gemm reference(ctx);
  onnx_light_cpu::GemmKernel kernel(ctx);
  const auto expected = reference(a, b, c, 0.25f, beta, trans_a, trans_b);
  const auto actual = c != nullptr ? kernel(a, b, *c, 0.25f, beta, trans_a, trans_b)
                                   : kernel(a, b, 0.25f, trans_a, trans_b);
  EXPECT_EQ(actual.shape, expected.shape);
  const auto comparison = rt_ns::CompareTensors(actual, expected, 0, 0, true);
  EXPECT_TRUE(comparison.close) << comparison.message;

  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Gemm");
  node.add_input("A");
  node.add_input("B");
  if (c != nullptr) {
    node.add_input("C");
  }
  node.add_output("Y");
  for (const auto &[name, value] : {std::pair{"transA", trans_a}, std::pair{"transB", trans_b}}) {
    auto *attribute = node.add_attribute();
    attribute->set_name(name);
    attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
    attribute->set_i(value);
  }
  for (const auto &[name, value] : {std::pair{"alpha", 0.25f}, std::pair{"beta", beta}}) {
    auto *attribute = node.add_attribute();
    attribute->set_name(name);
    attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::FLOAT);
    attribute->set_f(value);
  }
  rt_ns::RuntimeContext runtime(ctx);
  runtime.Set("A", a);
  runtime.Set("B", b);
  if (c != nullptr) {
    runtime.Set("C", *c);
  }
  kernel.set_node(node);
  for (int run = 0; run < 2; ++run) {
    kernel.Run(runtime);
    EXPECT_EQ(runtime.Get("Y").shape, expected.shape);
    const auto cached = rt_ns::CompareTensors(runtime.Get("Y"), expected, 0, 0, true);
    EXPECT_TRUE(cached.close) << cached.message;
  }
}

TEST(OnnxLightGemmKernel, Opset29VectorAndHighRankBroadcastTranspose) {
  for (const auto dtype : {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE,
                           rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    for (const bool trans_a : {false, true}) {
      for (const bool trans_b : {false, true}) {
        const auto b =
            GemmTensor(dtype, {2, 2},
                       trans_b ? std::vector<float>{2, 1, 3, 4} : std::vector<float>{2, 3, 1, 4});
        const auto vector = GemmTensor(dtype, {2}, {1, 2});
        const auto vector_bias = GemmTensor(dtype, {2}, {2, 4});
        CheckGemm29(vector, b, nullptr, trans_a, trans_b);
        CheckGemm29(vector, b, &vector_bias, trans_a, trans_b);
        for (const bool rank4 : {false, true}) {
          const rt_ns::Shape shape =
              rank4 ? (trans_a ? rt_ns::Shape{2, 2, 2, 3} : rt_ns::Shape{2, 2, 3, 2})
                    : (trans_a ? rt_ns::Shape{2, 2, 3} : rt_ns::Shape{2, 3, 2});
          std::vector<float> input_values(shape.product());
          for (std::size_t i = 0; i < input_values.size(); ++i) {
            input_values[i] = static_cast<float>(i + 1);
          }
          const auto a = GemmTensor(dtype, shape, input_values);
          CheckGemm29(a, b, nullptr, trans_a, trans_b);
          for (const auto &bias_shape :
               {rt_ns::Shape{}, rt_ns::Shape{2}, rt_ns::Shape{3, 1}, rt_ns::Shape{1, 3, 2},
                rank4 ? rt_ns::Shape{2, 1, 1, 2} : rt_ns::Shape{2, 1, 2},
                rank4 ? rt_ns::Shape{1, 2, 1, 2} : rt_ns::Shape{2, 3, 1},
                rank4 ? rt_ns::Shape{2, 2, 3, 2} : rt_ns::Shape{2, 3, 2}}) {
            std::vector<float> values(bias_shape.product());
            for (std::size_t i = 0; i < values.size(); ++i) {
              values[i] = static_cast<float>(2 * i);
            }
            const auto c = GemmTensor(dtype, bias_shape, values);
            CheckGemm29(a, b, &c, trans_a, trans_b);
          }
        }
      }
    }
  }
}

TEST(OnnxLightGemmKernel, Opset29EmptyDimensions) {
  for (const auto dtype : {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE,
                           rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    for (const bool trans_a : {false, true}) {
      for (const bool trans_b : {false, true}) {
        for (const auto &shape : {rt_ns::Shape{0}, rt_ns::Shape{2, 3, 0}, rt_ns::Shape{0, 3, 2},
                                  rt_ns::Shape{2, 0, 2}, rt_ns::Shape{2, 3, 2}}) {
          auto stored_shape = shape;
          if (trans_a && shape.size() > 1) {
            std::swap(stored_shape[shape.size() - 2], stored_shape.back());
          }
          const int64_t k = shape.back();
          const auto input =
              GemmTensor(dtype, stored_shape, std::vector<float>(shape.product(), 1));
          for (const int64_t n : {0, 2}) {
            const auto b = GemmTensor(dtype, trans_b ? rt_ns::Shape{n, k} : rt_ns::Shape{k, n},
                                      std::vector<float>(k * n, 2));
            const auto c = GemmTensor(dtype, {n}, std::vector<float>(n, 2));
            CheckGemm29(input, b, nullptr, trans_a, trans_b);
            CheckGemm29(input, b, &c, trans_a, trans_b);
          }
        }
      }
    }
  }
}

TEST(OnnxLightGemmKernel, Opset29ValidationAndZeroBeta) {
  const rt_ns::KernelContext ctx(rt_ns::DefaultOpset(29));
  onnx_light_cpu::GemmKernel kernel(ctx);
  const auto a = rt_ns::Tensor::FromFloat("", {2}, {1, 2});
  const auto b = rt_ns::Tensor::FromFloat("", {2, 2}, {2, 3, 1, 4});
  const auto c = rt_ns::Tensor::FromFloat(
      "", {2}, {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()});
  CheckGemm29(a, b, &c, true, false, 0);
  for (int64_t opset : {13, 18, 28}) {
    onnx_light_cpu::GemmKernel old_kernel(rt_ns::KernelContext{rt_ns::DefaultOpset(opset)});
    EXPECT_THROW(old_kernel(a, b, 1, false, false), std::invalid_argument);
  }
  const auto scalar = rt_ns::Tensor::FromFloat("", {}, {1});
  const auto batched_b = rt_ns::Tensor::FromFloat("", {1, 2, 2}, {2, 3, 1, 4});
  const auto bad_k = rt_ns::Tensor::FromFloat("", {3, 2}, std::vector<float>(6));
  EXPECT_THROW(kernel(scalar, b, 1, false, false), std::invalid_argument);
  EXPECT_THROW(kernel(a, batched_b, 1, false, false), std::invalid_argument);
  EXPECT_THROW(kernel(a, bad_k, 1, false, false), std::invalid_argument);
  for (const auto &shape : {rt_ns::Shape{1, 2}, rt_ns::Shape{3}}) {
    const auto bad_bias = rt_ns::Tensor::FromFloat("", shape, std::vector<float>(shape.product()));
    EXPECT_THROW(kernel(a, b, bad_bias, 1, 0, false, false), std::invalid_argument);
  }
  auto short_input = a;
  short_input.shape = {2, 3, 2};
  EXPECT_THROW(kernel(short_input, b, 1, false, false), std::invalid_argument);
  auto negative = a;
  negative.shape = {-1, 2};
  EXPECT_THROW(kernel(negative, b, 1, false, false), std::invalid_argument);
  auto overflowing = a;
  overflowing.shape = {std::numeric_limits<int64_t>::max(), 2, 2};
  EXPECT_THROW(kernel(overflowing, b, 1, false, false), std::invalid_argument);

  const auto matrix = rt_ns::Tensor::FromFloat("", {1, 2}, {1, 2});
  auto short_matrix = rt_ns::Tensor::FromFloat("", {1}, {1});
  short_matrix.shape = {1, 2};
  EXPECT_THROW(kernel(short_matrix, b, 1, false, false), std::invalid_argument);
  auto short_b = b;
  short_b.shape = {2, 3};
  EXPECT_THROW(kernel(matrix, short_b, 1, false, false), std::invalid_argument);
  auto short_c = rt_ns::Tensor::FromFloat("", {1}, {1});
  short_c.shape = {1, 2};
  EXPECT_THROW(kernel(matrix, b, short_c, 1, 0, false, false), std::invalid_argument);
}

} // namespace
