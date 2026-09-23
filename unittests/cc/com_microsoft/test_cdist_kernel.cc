// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/cdist.h"
#include "onnx_light_cpu/impl/execution.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
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

TEST(CDist, ZeroFeatureDimensionProducesPositiveZero) {
  std::vector<double> output(6, -1.0);
  CDistFloat64(nullptr, nullptr, output.data(), 2, 3, 0, CDistMetric::kEuclidean);
  for (double value : output) {
    EXPECT_EQ(value, 0.0);
    EXPECT_FALSE(std::signbit(value));
  }
}

TEST(CDist, NearIdenticalLargePointsRemainFiniteAndNonNegative) {
  constexpr std::size_t n = 65;
  std::vector<float> a(n, 1024.0f);
  std::vector<float> b = a;
  b.back() = std::nextafter(b.back(), std::numeric_limits<float>::infinity());
  float squared = -1.0f;
  float euclidean = -1.0f;
  CDistFloat32(a.data(), b.data(), &squared, 1, 1, n, CDistMetric::kSqEuclidean);
  CDistFloat32(a.data(), b.data(), &euclidean, 1, 1, n, CDistMetric::kEuclidean);
  EXPECT_TRUE(std::isfinite(squared));
  EXPECT_GE(squared, 0.0f);
  EXPECT_FLOAT_EQ(euclidean, std::sqrt(squared));
}

TEST(CDist, NonFiniteAndSignedZeroBehavior) {
  const float infinity = std::numeric_limits<float>::infinity();
  const std::vector<float> a = {infinity, 0.0f, -0.0f};
  const std::vector<float> b = {infinity, 0.0f, 1.0f, 0.0f, -0.0f, 0.0f};
  std::vector<float> output(2);
  CDistFloat32(a.data(), b.data(), output.data(), 1, 2, 3, CDistMetric::kSqEuclidean);
  EXPECT_TRUE(std::isnan(output[0]));
  EXPECT_TRUE(std::isinf(output[1]));

  const std::vector<float> zeros = {0.0f, -0.0f};
  float zero_distance = -1.0f;
  CDistFloat32(zeros.data(), zeros.data(), &zero_distance, 1, 1, 2, CDistMetric::kSqEuclidean);
  EXPECT_EQ(zero_distance, 0.0f);
  EXPECT_FALSE(std::signbit(zero_distance));
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

template <typename T>
void RunCDist(const T *a, const T *b, T *c, std::size_t m, std::size_t k, std::size_t n,
              CDistMetric metric, const CDistExecutionTuning &tuning = {}) {
  if constexpr (std::is_same_v<T, float>) {
    CDistFloat32WithTuning(a, b, c, m, k, n, metric, tuning);
  } else {
    onnx_light_cpu::CDistFloat64WithTuning(a, b, c, m, k, n, metric, tuning);
  }
}

template <typename T> class CDistPacked : public ::testing::Test {};
using CDistTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CDistPacked, CDistTypes);

TYPED_TEST(CDistPacked, TileBoundariesAndAllSimdTails) {
  using T = TypeParam;
  std::vector<std::size_t> features;
  for (std::size_t n = 1; n <= 33; ++n) {
    features.push_back(n);
  }
  features.insert(features.end(), {63, 64, 65, 255, 256, 257});
  std::vector<std::size_t> columns;
  for (std::size_t k = 1; k <= 33; ++k) {
    columns.push_back(k);
  }
  for (std::size_t k = 64; k <= 81; ++k) {
    columns.push_back(k);
  }
  for (std::size_t m : {7, 31, 32, 33, 63, 64, 65}) {
    for (std::size_t k : columns) {
      for (std::size_t n : features) {
        SCOPED_TRACE(::testing::Message() << "m=" << m << " k=" << k << " n=" << n);
        // Offset every buffer to exercise unaligned loads/stores. The final B
        // row ends at the allocation boundary, including partial column tiles.
        std::vector<T> a(m * n + 1), b(k * n + 1), output(m * k + 2, T(-42));
        for (std::size_t i = 1; i < a.size(); ++i) {
          a[i] = T(static_cast<int>(i % 29) - 14) / T(19);
        }
        for (std::size_t i = 1; i < b.size(); ++i) {
          b[i] = T(static_cast<int>(i % 23) - 11) / T(31);
        }
        const std::vector<T> a_values(a.begin() + 1, a.end());
        const std::vector<T> b_values(b.begin() + 1, b.end());
        for (auto metric : {CDistMetric::kSqEuclidean, CDistMetric::kEuclidean}) {
          RunCDist(a.data() + 1, b.data() + 1, output.data() + 1, m, k, n, metric);
          const auto expected = Reference(a_values, b_values, m, k, n, metric);
          for (std::size_t i = 0; i < expected.size(); ++i) {
            const T tolerance =
                std::numeric_limits<T>::epsilon() * T(n) * std::max(T(1), expected[i]);
            ASSERT_NEAR(output[i + 1], expected[i], tolerance) << i;
          }
          EXPECT_EQ(output.front(), T(-42));
          EXPECT_EQ(output.back(), T(-42));
        }
      }
    }
  }
}

TYPED_TEST(CDistPacked, StableDistancesAndSpecialValues) {
  using T = TypeParam;
  constexpr std::size_t m = 67, k = 65;
  const T infinity = std::numeric_limits<T>::infinity();
  const T nan = std::numeric_limits<T>::quiet_NaN();
  for (std::size_t n : {3, 17, 65, 256}) {
    std::vector<T> a(m * n, T(1024)), b(k * n, T(1024)), output(m * k);
    b[n - 1] = std::nextafter(T(1024), infinity);
    a[n] = infinity;
    b[n] = infinity;
    a[2 * n] = nan;
    b[2 * n] = nan;
    b[3 * n] = std::numeric_limits<T>::max();
    b[4 * n] = -infinity;
    // Exercise both the four-row body and the remaining rows, and the partial
    // output column tile. Identical signed zeros must produce positive zero.
    std::fill(a.begin() + 60 * n, a.end(), T(-0.0));
    std::fill(b.begin() + (k - 1) * n, b.end(), T(0));
    for (auto metric : {CDistMetric::kSqEuclidean, CDistMetric::kEuclidean}) {
      RunCDist(a.data(), b.data(), output.data(), m, k, n, metric);
      const auto expected = Reference(a, b, m, k, n, metric);
      for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::isnan(expected[i])) {
          EXPECT_TRUE(std::isnan(output[i])) << i;
        } else {
          EXPECT_EQ(output[i], expected[i]) << i;
          if (expected[i] == T(0)) {
            EXPECT_FALSE(std::signbit(output[i])) << i;
          }
        }
      }
      EXPECT_GT(output[0], T(0));
    }
  }
}

struct CDistExecutor {
  int calls = 0;
  std::int64_t blocks = 0;

  static void Run(void *context, std::int64_t count, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<CDistExecutor *>(context);
    ++self.calls;
    self.blocks = count;
    // Reverse order catches accidental dependencies between row slices.
    for (std::int64_t block = count; block-- > 0;) {
      task(task_context, block);
    }
  }
};

TYPED_TEST(CDistPacked, ExecutorSlicesAndNestedFallback) {
  using T = TypeParam;
  for (std::size_t m : {9, 35, 131, 259}) {
    constexpr std::size_t k = 65, n = 17;
    std::vector<T> a(m * n), b(k * n), output(m * k, T(-42));
    for (std::size_t i = 0; i < a.size(); ++i) {
      a[i] = T(i % 17) / T(4);
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
      b[i] = T(i % 13) / T(8);
    }
    CDistExecutor executor;
    const onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &CDistExecutor::Run};
    const onnx_light_cpu::ExecutionExecutorScope scope(&view);
    CDistExecutionTuning tuning;
    tuning.parallel_threshold_bytes = 1;
    tuning.target_block_bytes = 1;
    tuning.use_cost_model = false;
    for (auto metric : {CDistMetric::kSqEuclidean, CDistMetric::kEuclidean}) {
      executor.calls = 0;
      RunCDist(a.data(), b.data(), output.data(), m, k, n, metric, tuning);
      EXPECT_EQ(executor.calls, 1);
      EXPECT_GT(executor.blocks, 1);
      EXPECT_EQ(output, Reference(a, b, m, k, n, metric));
      // No nested executor submissions, even when the enclosing range is inline.
      onnx_light_cpu::ExecuteRanges(
          1, onnx_light_cpu::ExecutionSchedule{1, 1, 4}, [&](std::int64_t, std::int64_t) {
            RunCDist(a.data(), b.data(), output.data(), m, k, n, metric, tuning);
          });
      EXPECT_EQ(executor.calls, 1);
      EXPECT_EQ(output, Reference(a, b, m, k, n, metric));
      tuning.parallel_threshold_bytes = 0;
      RunCDist(a.data(), b.data(), output.data(), m, k, n, metric, tuning);
      EXPECT_EQ(executor.calls, 1);
      tuning.parallel_threshold_bytes = 1;
    }
  }
}

} // namespace
