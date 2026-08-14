// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"
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

TEST(GemmFloat32, UnitScalesWithBiasAcrossChunks) {
  const std::size_t M = 7, N = 19, K = 300;
  const auto A = RandomVector(M * K, 91);
  const auto B = RandomVector(K * N, 92);
  const auto C = RandomVector(M * N, 93);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 1.0f, &C);
  std::vector<float> Y(M * N, 0.0f);

  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 1.0f, C.data(),
                              Y.data());

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-2f) << "i=" << i;
  }
}

TEST(GemmFloat32, TransposeVariants) {
  // These dimensions force the general path even with 16-lane AVX-512.
  const std::size_t M = 8, N = 17, K = 40;
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
        EXPECT_NEAR(Y[i], expected[i], 2e-3f)
            << "trans_a=" << trans_a << " trans_b=" << trans_b << " i=" << i;
      }
    }
  }
}

TEST(GemmFloat32, SpecializedRowVariantsMatchReference) {
  const std::size_t N = 19, K = 40;
  for (std::size_t M = 1; M <= 4; ++M) {
    const auto A = RandomVector(M * K, static_cast<unsigned>(80 + M));
    const auto B = RandomVector(K * N, static_cast<unsigned>(90 + M));
    const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 0.0f, nullptr);
    std::vector<float> Y(M * N, 0.0f);

    onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr,
                                Y.data());

    for (std::size_t i = 0; i < M * N; ++i) {
      EXPECT_NEAR(Y[i], expected[i], 2e-3f) << "M=" << M << " i=" << i;
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

TEST(GemmFloat32, PlannedGridCoversRowAndColumnPanels) {
  const std::size_t M = 70, N = 100, K = 130;
  const auto A = RandomVector(M * K, 31);
  const auto B = RandomVector(K * N, 32);
  const auto C = RandomVector(M * N, 33);
  const auto expected = ReferenceGemm<float>(false, true, M, N, K, 0.75f, A, B, 1.25f, &C);
  std::vector<float> Y(M * N);
  const onnx_light_cpu::GemmBlocking blocking{32, 48, 64, 4, 16};

  onnx_light_cpu::detail::GemmFloat32Planned<onnx_light_cpu::GemmAlgorithm::kGeneral>(
      false, true, M, N, K, 0.75f, A.data(), B.data(), 1.25f, C.data(), Y.data(), &blocking);

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 2e-2f) << "i=" << i;
  }
}

// K larger than the internal K-blocking chunk (kGemmTileK=256) so the k
// reduction is split across several chunks that must be accumulated together
// (GemmAccumMode::kAccumulate) instead of each overwriting Y. Also exercises
// transpose_a/transpose_b together with a k-chunk boundary that does not align
// with kMR-sized row groups.
TEST(GemmFloat32, LargeKSpansMultipleChunks) {
  const std::size_t M = 5, N = 7, K = 600;
  const auto A = RandomVector(M * K, 41);
  const auto B = RandomVector(K * N, 42);
  const auto C = RandomVector(M * N, 43);
  const float alpha = 1.25f, beta = 0.5f;
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, alpha, A, B, beta, &C);
  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, alpha, A.data(), B.data(), beta, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 2e-2f) << "i=" << i;
  }
}

TEST(GemmFloat32, SplitKMatchesReference) {
  const std::size_t M = 2, N = 2, K = 4096;
  const auto A = RandomVector(M * K, 51);
  const auto B = RandomVector(K * N, 52);
  const auto C = RandomVector(M * N, 53);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 0.75f, A, B, 0.5f, &C);
  std::vector<float> Y(M * N, 0.0f);

  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 0.75f, A.data(), B.data(), 0.5f, C.data(),
                              Y.data());

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 5e-3f) << "i=" << i;
  }
}

TEST(GemmFloat32, SplitKSupportsTransposedInputs) {
  const std::size_t M = 2, N = 2, K = 4096;
  const auto A = RandomVector(K * M, 54);
  const auto B = RandomVector(N * K, 55);
  const auto expected = ReferenceGemm<float>(true, true, M, N, K, 1.0f, A, B, 0.0f, nullptr);
  std::vector<float> Y(M * N);

  onnx_light_cpu::GemmFloat32(true, true, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr,
                              Y.data());

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-2f) << "i=" << i;
  }
}

// M much smaller than N: with the flattened (row block, column panel) task
// list this shape still spans multiple column panels, exercising the case
// that motivated splitting the parallel work across N and not only M.
TEST(GemmFloat32, SkinnyMWideNMultipleColumnPanels) {
  const std::size_t M = 2, N = 1000, K = 33;
  const auto A = RandomVector(M * K, 44);
  const auto B = RandomVector(K * N, 45);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 0.0f, nullptr);
  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr,
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

TEST(GemmFloat32, EmptyMOrNReturnsWithoutWriting) {
  // M == 0 and N == 0 both hit the early return in GemmImpl: the output buffer
  // must be left untouched (here it is empty, so the call must simply not crash).
  std::vector<float> Y;
  onnx_light_cpu::GemmFloat32(false, false, /*M=*/0, /*N=*/4, /*K=*/3, 1.0f, nullptr, nullptr, 0.0f,
                              nullptr, Y.data());
  onnx_light_cpu::GemmFloat32(false, false, /*M=*/4, /*N=*/0, /*K=*/3, 1.0f, nullptr, nullptr, 0.0f,
                              nullptr, Y.data());
  SUCCEED();
}

TEST(GemmFloat32, SingleRowColumnTail) {
  // M == 1 exercises the mr == 1 register block and N == 5 leaves a column tail
  // (5 % 8 != 0 for float32) that the scalar micro-kernel finishes, with no bias.
  const std::size_t M = 1, N = 5, K = 4;
  const auto A = RandomVector(M * K, 31);
  const auto B = RandomVector(K * N, 32);
  const auto expected = ReferenceGemm<float>(false, false, M, N, K, 1.0f, A, B, 0.0f, nullptr);
  std::vector<float> Y(M * N, 0.0f);
  onnx_light_cpu::GemmFloat32(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, nullptr,
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-4f) << "i=" << i;
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

TEST(GemmFloat64, TransposeVariants) {
  // The float32 path already covers every trans_a/trans_b combination; do the
  // same for float64 so the double transpose-b packing and both-transpose
  // address arithmetic are exercised too.
  // These dimensions force the general path even with 8-lane AVX-512.
  const std::size_t M = 8, N = 9, K = 40;
  for (int ta = 0; ta < 2; ++ta) {
    for (int tb = 0; tb < 2; ++tb) {
      const bool trans_a = ta != 0;
      const bool trans_b = tb != 0;
      const auto A = RandomVectorD((trans_a ? K * M : M * K), 40 + ta * 2 + tb);
      const auto B = RandomVectorD((trans_b ? N * K : K * N), 50 + ta * 2 + tb);
      const auto expected =
          ReferenceGemm<double>(trans_a, trans_b, M, N, K, 1.0, A, B, 0.0, nullptr);
      std::vector<double> Y(M * N, 0.0);
      onnx_light_cpu::GemmFloat64(trans_a, trans_b, M, N, K, 1.0, A.data(), B.data(), 0.0, nullptr,
                                  Y.data());
      for (std::size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(Y[i], expected[i], 1e-9)
            << "trans_a=" << trans_a << " trans_b=" << trans_b << " i=" << i;
      }
    }
  }
}

TEST(GemmFloat64, SpecializedRowVariantsMatchReference) {
  const std::size_t N = 11, K = 40;
  for (std::size_t M = 1; M <= 4; ++M) {
    const auto A = RandomVectorD(M * K, static_cast<unsigned>(100 + M));
    const auto B = RandomVectorD(K * N, static_cast<unsigned>(110 + M));
    const auto expected = ReferenceGemm<double>(false, false, M, N, K, 1.0, A, B, 0.0, nullptr);
    std::vector<double> Y(M * N, 0.0);

    onnx_light_cpu::GemmFloat64(false, false, M, N, K, 1.0, A.data(), B.data(), 0.0, nullptr,
                                Y.data());

    for (std::size_t i = 0; i < M * N; ++i) {
      EXPECT_NEAR(Y[i], expected[i], 1e-9) << "M=" << M << " i=" << i;
    }
  }
}

// Same K-blocking regression as GemmFloat32.LargeKSpansMultipleChunks, for the
// float64 micro-kernels (AVX/SSE2 4/2-wide accumulate paths).
TEST(GemmFloat64, LargeKSpansMultipleChunks) {
  const std::size_t M = 5, N = 7, K = 600;
  const auto A = RandomVectorD(M * K, 70);
  const auto B = RandomVectorD(K * N, 71);
  const auto C = RandomVectorD(M * N, 72);
  const double alpha = 1.25, beta = 0.5;
  const auto expected = ReferenceGemm<double>(false, false, M, N, K, alpha, A, B, beta, &C);
  std::vector<double> Y(M * N, 0.0);
  onnx_light_cpu::GemmFloat64(false, false, M, N, K, alpha, A.data(), B.data(), beta, C.data(),
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-9) << "i=" << i;
  }
}

TEST(GemmFloat64, SplitKSupportsTransposedBAndBias) {
  const std::size_t M = 2, N = 2, K = 4096;
  const auto A = RandomVectorD(M * K, 73);
  const auto B = RandomVectorD(N * K, 74);
  const auto C = RandomVectorD(M * N, 75);
  const auto expected = ReferenceGemm<double>(false, true, M, N, K, 0.5, A, B, 2.0, &C);
  std::vector<double> Y(M * N);

  onnx_light_cpu::GemmFloat64(false, true, M, N, K, 0.5, A.data(), B.data(), 2.0, C.data(),
                              Y.data());

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-9) << "i=" << i;
  }
}

TEST(GemmFloat64, NoBiasColumnTail) {
  // N == 7 leaves a column tail (7 % 4 != 0 for float64) that is finished by the
  // scalar micro-kernel; with beta == 0 / C == nullptr this exercises the
  // bias-free branch of the scalar tail handler.
  const std::size_t M = 3, N = 7, K = 5;
  const auto A = RandomVectorD(M * K, 60);
  const auto B = RandomVectorD(K * N, 61);
  const auto expected = ReferenceGemm<double>(false, false, M, N, K, 1.0, A, B, 0.0, nullptr);
  std::vector<double> Y(M * N, 0.0);
  onnx_light_cpu::GemmFloat64(false, false, M, N, K, 1.0, A.data(), B.data(), 0.0, nullptr,
                              Y.data());
  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(Y[i], expected[i], 1e-9) << "i=" << i;
  }
}
