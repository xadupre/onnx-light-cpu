// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <random>
#include <vector>

namespace {

// Straightforward reference Gemm: Y = alpha * op(A) @ op(B) + beta * C.
template <typename T>
std::vector<T> ReferenceGemm(bool trans_a, bool trans_b, std::size_t M, std::size_t N,
                             std::size_t K, T alpha, const std::vector<T> &A,
                             const std::vector<T> &B, T beta, const std::vector<T> *C) {
  std::vector<T> Y(M * N, T(0));
  for (std::size_t m = 0; m < M; ++m) {
    for (std::size_t n = 0; n < N; ++n) {
      T acc = T(0);
      for (std::size_t k = 0; k < K; ++k) {
        const T a = trans_a ? A[k * M + m] : A[m * K + k];
        const T b = trans_b ? B[n * K + k] : B[k * N + n];
        acc += a * b;
      }
      T value = alpha * acc;
      if (C != nullptr && beta != T(0)) {
        value += beta * (*C)[m * N + n];
      }
      Y[m * N + n] = value;
    }
  }
  return Y;
}

std::vector<float> RandomVector(std::size_t size, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> v(size);
  for (auto &x : v) {
    x = dist(gen);
  }
  return v;
}

std::vector<double> RandomVectorD(std::size_t size, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  std::vector<double> v(size);
  for (auto &x : v) {
    x = dist(gen);
  }
  return v;
}

} // namespace

TEST(GemmFloat32, MatmulMatchesReference) {
  const std::size_t M = 5, N = 7, K = 3;
  const auto A = RandomVector(M * K, 1);
  const auto B = RandomVector(K * N, 2);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 0.0f, nullptr);

  std::vector<float> Y(M * N, -123.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr,
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-4f) << "i=" << i;
  }
}

TEST(GemmFloat32, AlphaBetaAndBias) {
  const std::size_t M = 4, N = 6, K = 8;
  const auto A = RandomVector(M * K, 3);
  const auto B = RandomVector(K * N, 4);
  const auto C = RandomVector(M * N, 5);
  const float alpha = 0.5f, beta = 2.0f;
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, alpha, A, B, beta, &C);

  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, alpha, A.data(), B.data(), beta, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-3f) << "i=" << i;
  }
}

TEST(GemmFloat32, TransposeVariants) {
  const std::size_t M = 6, N = 5, K = 4;
  for (int ta = 0; ta < 2; ++ta) {
    for (int tb = 0; tb < 2; ++tb) {
      const bool trans_a = ta != 0;
      const bool trans_b = tb != 0;
      const auto A = RandomVector((trans_a ? K * M : M * K), 10 + ta * 2 + tb);
      const auto B = RandomVector((trans_b ? N * K : K * N), 20 + ta * 2 + tb);
      const auto expected =
          ReferenceGemm<float>(trans_a, trans_b, M, N, K, 1.0f, A, B, 0.0f, nullptr);
      std::vector<float> Y(M * N, 0.0f);
      onnx_light_cpu::GemmFloat32(trans_a, trans_b, M, N, K, 1.0f, A.data(), B.data(), 0.0f,
                                  nullptr, Y.data());
      for (std::size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(Y[i], expected[i], 1e-4f)
            << "trans_a=" << trans_a << " trans_b=" << trans_b << " i=" << i;
      }
    }
  }
}

TEST(GemmFloat32, BetaZeroIgnoresBias) {
  const std::size_t M = 3, N = 3, K = 3;
  const auto A = RandomVector(M * K, 6);
  const auto B = RandomVector(K * N, 7);
  // C is deliberately garbage; beta == 0 must ignore it entirely.
  std::vector<float> C(M * N, std::numeric_limits<float>::quiet_NaN());
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 0.0f, nullptr);
  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-4f) << "i=" << i;
  }
}

TEST(GemmFloat32, LargeMatrixParallel) {
  const std::size_t M = 128, N = 96, K = 64;
  const auto A = RandomVector(M * K, 11);
  const auto B = RandomVector(K * N, 12);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 0.0f, nullptr);
  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr,
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-2f) << "i=" << i;
  }
}

// Dimensions larger than the internal cache tiles (kGemmTileN=256, kGemmTileM=64)
// so the output is split across several column panels and row blocks. This is a
// regression test for a bug where the micro-kernels ignored the column-panel
// offset ``n0`` when writing ``Y``, corrupting every panel past the first.
TEST(GemmFloat32, MultiPanelExceedsTiles) {
  const std::size_t M = 130, N = 300, K = 40;
  const auto A = RandomVector(M * K, 21);
  const auto B = RandomVector(K * N, 22);
  const auto C = RandomVector(M * N, 23);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 0.75f, A, B, 1.5f, &C);
  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 0.75f, A.data(), B.data(), 1.5f, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-2f) << "i=" << i;
  }
}

TEST(GemmFloat32, EmptyKGivesBiasOnly) {
  const std::size_t M = 3, N = 4, K = 0;
  const auto C = RandomVector(M * N, 8);
  std::vector<float> Y(M * N, 7.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, nullptr, nullptr, 3.0f, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], 3.0f * C[i], 1e-5f) << "i=" << i;
  }
}

TEST(GemmFloat64, MatmulMatchesReference) {
  const std::size_t M = 5, N = 7, K = 9;
  const auto A = RandomVectorD(M * K, 21);
  const auto B = RandomVectorD(K * N, 22);
  const auto C = RandomVectorD(M * N, 23);
  const double alpha = 1.5, beta = -0.5;
  const auto expected = ReferenceGemm<double>(true, false, M, N, K, alpha, A, B, beta, &C);
  std::vector<double> Y(M * N, 0.0);
  // trans_a: A stored as K x M.
  onnx_light_cpu::GemmFloat64(true, false, M, N, K, alpha, A.data(), B.data(), beta, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-9) << "i=" << i;
  }
}
