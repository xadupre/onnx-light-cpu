// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/gemm_kernel.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/tensor_compare.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

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
}

} // namespace
