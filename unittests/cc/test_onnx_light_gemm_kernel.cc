// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/gemm_kernel.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

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

TEST(OnnxLightGemmKernel, Float32Matmul) {
  onnx_light_cpu::GemmKernel kernel(MakeCtx());
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

} // namespace
