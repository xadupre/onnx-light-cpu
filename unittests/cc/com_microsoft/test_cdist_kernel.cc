// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/cdist.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using onnx_light_cpu::CDistExecutionTuning;
using onnx_light_cpu::CDistFloat32;
using onnx_light_cpu::CDistFloat32WithTuning;
using onnx_light_cpu::CDistFloat64;
using onnx_light_cpu::CDistMetric;

// Direct (non-expanded) reference implementation used to check the scalar
// kernel's numerics without sharing any code path with it.
template <typename T>
std::vector<T> Reference(const std::vector<T> &a, const std::vector<T> &b, std::size_t m,
                         std::size_t k, std::size_t n, CDistMetric metric) {
  std::vector<T> expected(m * k);
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t col = 0; col < k; ++col) {
      T sum_squares = T(0);
      for (std::size_t feature = 0; feature < n; ++feature) {
        const T difference = a[row * n + feature] - b[col * n + feature];
        sum_squares += difference * difference;
      }
      expected[row * k + col] =
          metric == CDistMetric::kEuclidean ? std::sqrt(sum_squares) : sum_squares;
    }
  }
  return expected;
}

TEST(CDist, Float32SqEuclideanMatchesDirectReference) {
  constexpr std::size_t m = 3;
  constexpr std::size_t k = 2;
  constexpr std::size_t n = 4;
  const std::vector<float> a = {1.0f, 2.0f,  3.0f, 4.0f, -1.0f, 0.5f,
                                2.5f, -3.0f, 0.0f, 0.0f, 0.0f,  0.0f};
  const std::vector<float> b = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> output(m * k);
  CDistFloat32(a.data(), b.data(), output.data(), m, k, n, CDistMetric::kSqEuclidean);
  const std::vector<float> expected = Reference(a, b, m, k, n, CDistMetric::kSqEuclidean);
  for (std::size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-5f) << i;
  }
}

TEST(CDist, Float32EuclideanTakesSquareRootOfSqEuclidean) {
  constexpr std::size_t m = 2;
  constexpr std::size_t k = 3;
  constexpr std::size_t n = 3;
  const std::vector<float> a = {0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f};
  const std::vector<float> b = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 3.0f, 3.0f, 3.0f};
  std::vector<float> sq(m * k);
  std::vector<float> eu(m * k);
  CDistFloat32(a.data(), b.data(), sq.data(), m, k, n, CDistMetric::kSqEuclidean);
  CDistFloat32(a.data(), b.data(), eu.data(), m, k, n, CDistMetric::kEuclidean);
  for (std::size_t i = 0; i < sq.size(); ++i) {
    EXPECT_NEAR(eu[i], std::sqrt(sq[i]), 1e-5f) << i;
  }
}

TEST(CDist, Float64MatchesDirectReferenceForBothMetrics) {
  constexpr std::size_t m = 4;
  constexpr std::size_t k = 3;
  constexpr std::size_t n = 5;
  std::vector<double> a(m * n);
  std::vector<double> b(k * n);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<double>(static_cast<int>(i % 13) - 6) * 0.5;
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<double>(static_cast<int>(i % 7) - 3) * 0.25;
  }
  for (CDistMetric metric : {CDistMetric::kSqEuclidean, CDistMetric::kEuclidean}) {
    std::vector<double> output(m * k);
    CDistFloat64(a.data(), b.data(), output.data(), m, k, n, metric);
    const std::vector<double> expected = Reference(a, b, m, k, n, metric);
    for (std::size_t i = 0; i < output.size(); ++i) {
      EXPECT_NEAR(output[i], expected[i], 1e-9) << i;
    }
  }
}

TEST(CDist, VectorizedFeatureTailMatchesDirectReference) {
  constexpr std::size_t m = 3;
  constexpr std::size_t k = 5;
  constexpr std::size_t n = 131;
  std::vector<float> a(m * n);
  std::vector<float> b(k * n);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.0625f;
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.03125f;
  }
  for (CDistMetric metric : {CDistMetric::kSqEuclidean, CDistMetric::kEuclidean}) {
    std::vector<float> output(m * k);
    CDistFloat32(a.data(), b.data(), output.data(), m, k, n, metric);
    const std::vector<float> expected = Reference(a, b, m, k, n, metric);
    for (std::size_t i = 0; i < output.size(); ++i) {
      EXPECT_NEAR(output[i], expected[i], 1e-4f) << i;
    }
  }
}

TEST(CDist, ZeroRowsOrColumnsProduceNoWrites) {
  std::vector<float> sentinel(1, -42.0f);
  CDistFloat32(nullptr, nullptr, sentinel.data(), 0, 0, 3, CDistMetric::kSqEuclidean);
  EXPECT_EQ(sentinel[0], -42.0f);
}

TEST(CDist, TunedDispatchWithParallelThresholdMatchesUntuned) {
  constexpr std::size_t m = 64;
  constexpr std::size_t k = 8;
  constexpr std::size_t n = 16;
  std::vector<float> a(m * n);
  std::vector<float> b(k * n);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.125f;
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25f;
  }
  std::vector<float> serial(m * k);
  std::vector<float> parallel(m * k);
  CDistFloat32(a.data(), b.data(), serial.data(), m, k, n, CDistMetric::kEuclidean);

  CDistExecutionTuning tuning;
  tuning.parallel_threshold_bytes = 1;
  tuning.target_block_bytes = 1;
  tuning.use_cost_model = false;
  CDistFloat32WithTuning(a.data(), b.data(), parallel.data(), m, k, n, CDistMetric::kEuclidean,
                         tuning);
  for (std::size_t i = 0; i < serial.size(); ++i) {
    EXPECT_NEAR(serial[i], parallel[i], 1e-5f) << i;
  }
}

} // namespace
