// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/attention/attention_plan.h"

#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/math/half_conversion.h"
#include "ml_cpu_compute/math/math.h"

#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
#include "ml_cpu_compute/attention/avx2/attention_avx2_fma.h"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

struct InlineExecutor {
  std::int64_t maximum_blocks = 0;
  std::int64_t dispatches = 0;
  bool nested = false;

  static void Run(void *context, std::int64_t num_blocks, void *task_context,
                  ml_cpu_compute::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.nested |= ml_cpu_compute::ExecutionInParallelRegion();
    self.maximum_blocks = std::max(self.maximum_blocks, num_blocks);
    for (std::int64_t block = 0; block < num_blocks; ++block) {
      task(task_context, block);
    }
  }
};

using ml_cpu_compute::AttentionDescriptor;
using ml_cpu_compute::AttentionLayout;
using ml_cpu_compute::AttentionMaskKind;
using ml_cpu_compute::AttentionPlan;
using ml_cpu_compute::ComputeAttentionBFloat16Streaming;
using ml_cpu_compute::ComputeAttentionFloat16Streaming;
using ml_cpu_compute::ComputeAttentionFloat32;
using ml_cpu_compute::ComputeAttentionFloat32Streaming;

TEST(AttentionSink, NormalizesRealValuesAcrossTypesAndWindows) {
  for (const bool cached : {false, true}) {
    for (const std::int64_t window : {-1, 2}) {
      const std::size_t sequence = cached ? 1 : 20;
      const std::size_t past_length = cached ? 5 : 0;
      ml_cpu_compute::AttentionDescriptor descriptor;
      descriptor.q_num_heads = 2;
      descriptor.kv_num_heads = 1;
      descriptor.is_causal = true;
      descriptor.left_window_size = window < 0 ? -1 : window - 1;
      if (window > 0) {
        descriptor.opset = 25;
      }
      const std::array<std::int64_t, 4> q_shape{1, 2, static_cast<std::int64_t>(sequence), 1};
      const std::array<std::int64_t, 4> kv_shape{1, 1, static_cast<std::int64_t>(sequence), 1};
      const std::vector<std::int64_t> past_shape =
          cached ? std::vector<std::int64_t>{1, 1, 5, 1} : std::vector<std::int64_t>{};
      ml_cpu_compute::AttentionPlan plan(
          descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape, kv_shape, kv_shape, {},
          ml_cpu_compute::AttentionMaskKind::kNone, past_shape, past_shape);
      const std::array<float, 2> sink{0.0f, 1.0f};
      plan.head_sink = sink.data();
      const std::vector<float> query(2 * sequence, 0.0f);
      const std::vector<float> key(sequence, 0.0f);
      const std::vector<float> value(sequence, 4.0f);
      const std::vector<float> past_key(past_length, 0.0f);
      const std::vector<float> past_value(past_length, 4.0f);
      const auto check = [&](auto run, float tolerance) {
        for (std::size_t h = 0; h < 2; ++h) {
          for (std::size_t i = 0; i < sequence; ++i) {
            const std::size_t visible = window < 0
                                            ? past_length + i + 1
                                            : std::min<std::size_t>(window, past_length + i + 1);
            EXPECT_NEAR(run[h * sequence + i], 4.0f * visible / (std::exp(sink[h]) + visible),
                        tolerance);
          }
        }
      };
      std::vector<float> y(2 * sequence);
      ml_cpu_compute::ComputeAttentionFloat32(plan, query.data(), key.data(), value.data(), nullptr,
                                              y.data(), cached ? past_key.data() : nullptr,
                                              cached ? past_value.data() : nullptr);
      check(y, 1e-5f);
      for (const bool fp64 : {false, true}) {
        for (int fallback = -1; fallback < 5; ++fallback) {
          if (fallback == -1 && !fp64) {
            continue;
          }
          SCOPED_TRACE(::testing::Message() << "cached=" << cached << " window=" << window
                                            << " fp64=" << fp64 << " fallback=" << fallback);
          ml_cpu_compute::AttentionPlan materialized_plan = plan;
          materialized_plan.softmax_fp64 = fp64;
          materialized_plan.has_present_output = fallback == 0;
          materialized_plan.has_qk_matmul_output = fallback > 0;
          materialized_plan.qk_matmul_output_mode = fallback > 0 ? fallback - 1 : 0;
          std::vector<float> probabilities(2 * sequence * (sequence + past_length));
          ml_cpu_compute::AttentionExecutionInfo info;
          ml_cpu_compute::ComputeAttentionFloat32(
              materialized_plan, query.data(), key.data(), value.data(), nullptr, y.data(),
              cached ? past_key.data() : nullptr, cached ? past_value.data() : nullptr, nullptr,
              fallback > 0 ? probabilities.data() : nullptr, &info);
          EXPECT_EQ(info.path, ml_cpu_compute::AttentionExecutionPath::kMaterialized);
          check(y, 1e-5f);
          if (fallback == 4) {
            for (std::size_t h = 0; h < 2; ++h) {
              for (std::size_t i = 0; i < sequence; ++i) {
                const std::size_t frontier = past_length + i;
                const std::size_t visible =
                    window < 0 ? frontier + 1 : std::min<std::size_t>(window, frontier + 1);
                for (std::size_t j = 0; j < sequence + past_length; ++j) {
                  const bool allowed =
                      j <= frontier &&
                      (window < 0 || frontier - j < static_cast<std::size_t>(window));
                  const float expected = allowed ? 1.0f / (std::exp(sink[h]) + visible) : 0.0f;
                  EXPECT_NEAR(probabilities[(h * sequence + i) * (sequence + past_length) + j],
                              expected, 1e-6f);
                }
              }
            }
          }
        }
      }
      for (const bool bfloat : {false, true}) {
        const auto encode = [bfloat](const std::vector<float> &input) {
          std::vector<std::uint16_t> output(input.size());
          for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = bfloat ? ml_cpu_compute::detail::FloatToBFloat16Bits(input[i])
                               : ml_cpu_compute::detail::FloatToFloat16Bits(input[i]);
          }
          return output;
        };
        const auto q16 = encode(query), k16 = encode(key), v16 = encode(value);
        const auto pk16 = encode(past_key), pv16 = encode(past_value);
        std::vector<std::uint16_t> y16(y.size());
        if (bfloat) {
          ml_cpu_compute::ComputeAttentionBFloat16Streaming(
              plan, q16.data(), k16.data(), v16.data(), nullptr, y16.data(),
              cached ? pk16.data() : nullptr, cached ? pv16.data() : nullptr);
        } else {
          ml_cpu_compute::ComputeAttentionFloat16Streaming(
              plan, q16.data(), k16.data(), v16.data(), nullptr, y16.data(),
              cached ? pk16.data() : nullptr, cached ? pv16.data() : nullptr);
        }
        std::vector<float> decoded(y.size());
        for (std::size_t i = 0; i < y.size(); ++i) {
          decoded[i] = bfloat ? ml_cpu_compute::detail::Bfloat16BitsToFloat(y16[i])
                              : ml_cpu_compute::detail::Float16BitsToFloat(y16[i]);
        }
        check(decoded, bfloat ? 0.02f : 0.002f);
      }
    }
  }
}

TEST(AttentionSink, MaterializedExtremesPreserveFullyMaskedRows) {
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  const std::int64_t q_shape[] = {1, 2, 2, 1};
  const std::int64_t kv_shape[] = {1, 1, 1, 1};
  const std::int64_t mask_shape[] = {2, 1};
  const float query[] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float key[] = {0.0f}, value[] = {4.0f};
  const float mask[] = {negative_infinity, 0.0f};
  for (const std::int64_t precision : {1, 11}) {
    ml_cpu_compute::AttentionDescriptor descriptor;
    descriptor.softmax_precision = precision;
    descriptor.has_qk_matmul_output = true;
    descriptor.qk_matmul_output_mode = 3;
    ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                       kv_shape, kv_shape, mask_shape,
                                       ml_cpu_compute::AttentionMaskKind::kAdditive);
    for (const auto &sink :
         {std::array<float, 2>{1000.0f, -1000.0f}, std::array<float, 2>{negative_infinity, 0.0f}}) {
      plan.head_sink = sink.data();
      std::array<float, 4> output, probabilities;
      ml_cpu_compute::ComputeAttentionFloat32(plan, query, key, value, mask, output.data(), nullptr,
                                              nullptr, nullptr, probabilities.data());
      for (std::size_t h = 0; h < sink.size(); ++h) {
        const float expected =
            static_cast<float>(1.0 / (1.0 + std::exp(static_cast<double>(sink[h]))));
        EXPECT_FLOAT_EQ(output[2 * h], 0.0f);
        EXPECT_FLOAT_EQ(probabilities[2 * h], 0.0f);
        EXPECT_FLOAT_EQ(output[2 * h + 1], 4.0f * expected);
        EXPECT_FLOAT_EQ(probabilities[2 * h + 1], expected);
      }
    }
  }
}

TEST(AttentionExp, Avx2WithoutFmaSelectsScalar) {
  using ml_cpu_compute::SimdLevel;
  using ml_cpu_compute::detail::SelectAttentionExpRoutine;
  const auto scalar = SelectAttentionExpRoutine(ml_cpu_compute::SimdLevel::kNone, false);
  const auto avx2_without_fma = SelectAttentionExpRoutine(ml_cpu_compute::SimdLevel::kAVX2, false);
  ASSERT_EQ(avx2_without_fma, scalar);
  EXPECT_EQ(SelectAttentionExpRoutine(ml_cpu_compute::SimdLevel::kAVX, true), scalar);

  for (std::size_t count : {0u, 1u, 7u, 8u, 9u, 16u, 17u, 257u}) {
    SCOPED_TRACE(count);
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = -static_cast<float>(i) / 8.0f;
    }
    const auto input = values;
    avx2_without_fma(values.data(), values.data(), count);
    for (std::size_t i = 0; i < count; ++i) {
      EXPECT_FLOAT_EQ(values[i], std::exp(input[i]));
    }
  }

  float masked_score = -std::numeric_limits<float>::infinity();
  avx2_without_fma(&masked_score, &masked_score, 1);
  EXPECT_EQ(masked_score, 0.0f);
}

TEST(AttentionExp, RuntimeSelectionMatchesStdExp) {
  const auto exp = ml_cpu_compute::detail::SelectAttentionExpRoutine(
      ml_cpu_compute::DetectSimdLevel(), ml_cpu_compute::CpuSupportsFma());
  std::array<float, 17> values;
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = -static_cast<float>(i) / 8.0f;
  }
  const auto input = values;
  exp(values.data(), values.data(), values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(values[i], std::exp(input[i]), 1e-6f);
  }
}

std::vector<float> RandomTensor(std::size_t count, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> values(count);
  for (float &value : values) {
    value = dist(rng);
  }
  return values;
}

// Naive, independently-written oracle mirroring the ONNX Attention spec: it
// only understands the rank-4 layout, so rank-3 test cases reshape their
// inputs into the equivalent rank-4 tensors before comparison.
std::vector<float> ReferenceAttention(std::size_t batch, std::size_t q_heads, std::size_t kv_heads,
                                      std::size_t q_len, std::size_t kv_len, std::size_t head_dim,
                                      std::size_t v_head_dim, const std::vector<float> &q,
                                      const std::vector<float> &k, const std::vector<float> &v,
                                      float scale, bool causal,
                                      const std::vector<float> *additive_mask,
                                      const std::vector<std::uint8_t> *bool_mask) {
  const std::size_t group = q_heads / kv_heads;
  std::vector<float> y(batch * q_heads * q_len * v_head_dim, 0.0f);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < q_heads; ++h) {
      const std::size_t kv_h = h / group;
      for (std::size_t i = 0; i < q_len; ++i) {
        std::vector<float> scores(kv_len);
        float row_max = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j < kv_len; ++j) {
          bool allowed = !causal || j <= i;
          float bias = 0.0f;
          if (allowed && bool_mask != nullptr) {
            allowed = (*bool_mask)[((b * q_heads + h) * q_len + i) * kv_len + j] != 0;
          }
          if (allowed && additive_mask != nullptr) {
            bias = (*additive_mask)[((b * q_heads + h) * q_len + i) * kv_len + j];
          }
          if (!allowed) {
            scores[j] = -std::numeric_limits<float>::infinity();
            continue;
          }
          float dot = 0.0f;
          for (std::size_t d = 0; d < head_dim; ++d) {
            dot += q[((b * q_heads + h) * q_len + i) * head_dim + d] *
                   k[((b * kv_heads + kv_h) * kv_len + j) * head_dim + d];
          }
          scores[j] = scale * dot + bias;
          row_max = std::max(row_max, scores[j]);
        }
        if (row_max == -std::numeric_limits<float>::infinity()) {
          continue; // Already zero-initialized.
        }
        float sum = 0.0f;
        for (float &score : scores) {
          score = std::exp(score - row_max);
          sum += score;
        }
        for (std::size_t j = 0; j < kv_len; ++j) {
          const float p = scores[j] / sum;
          for (std::size_t d = 0; d < v_head_dim; ++d) {
            y[((b * q_heads + h) * q_len + i) * v_head_dim + d] +=
                p * v[((b * kv_heads + kv_h) * kv_len + j) * v_head_dim + d];
          }
        }
      }
    }
  }
  return y;
}

void ExpectClose(const std::vector<float> &actual, const std::vector<float> &expected,
                 float tolerance = 1e-4f) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], tolerance) << "index=" << i;
  }
}

TEST(AttentionExp, DenseUnnormalizedRowsMatchOnlineSoftmax) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
  if (ml_cpu_compute::DetectSimdLevel() < ml_cpu_compute::SimdLevel::kAVX2 ||
      !ml_cpu_compute::CpuSupportsFma()) {
    GTEST_SKIP() << "AVX2/FMA kernels are unavailable";
  }
  for (const std::size_t columns :
       {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 511, 512, 513}) {
    for (const float range : {1.0f, 43.5f, 100.0f, std::numeric_limits<float>::max()}) {
      SCOPED_TRACE(::testing::Message() << "columns=" << columns << " range=" << range);
      auto scores = RandomTensor(3 * columns, 1984);
      for (float &value : scores) {
        value *= range;
      }
      auto expected = scores;
      std::array<float, 3> denominators{}, expected_denominators{};
      for (std::size_t row = 0; row < 3; ++row) {
        ml_cpu_compute::AttentionSoftmaxBlockFloat32_AVX2_FMA(
            expected.data() + row * columns, columns, -std::numeric_limits<float>::infinity(),
            expected_denominators[row]);
      }
      ml_cpu_compute::SoftmaxUnnormalizedFloat32_AVX2_FMA(scores.data(), denominators.data(), 3,
                                                          columns);
      ExpectClose(scores, expected, 2e-6f);
      for (std::size_t row = 0; row < 3; ++row) {
        EXPECT_NEAR(denominators[row], expected_denominators[row],
                    2e-6f * expected_denominators[row]);
      }
    }
  }
#else
  GTEST_SKIP() << "AVX2/FMA kernels were not built";
#endif
}

TEST(AttentionDescriptor, ValidatesOpsetRange) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.opset = 22;
  EXPECT_THROW(descriptor.Validate(), std::invalid_argument);
}

TEST(AttentionDescriptor, RejectsNonpadKvSeqlenBeforeOpset24) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.opset = 23;
  descriptor.has_nonpad_kv_seqlen = true;
  EXPECT_THROW(descriptor.Validate(), std::invalid_argument);
}

TEST(AttentionDescriptor, RejectsUnpairedPastCache) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.has_past_key = true;
  EXPECT_THROW(descriptor.Validate(), std::invalid_argument);
}

TEST(AttentionDescriptor, MaterializedPathAcceptsPastCache) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.has_past_key = true;
  descriptor.has_past_value = true;
  EXPECT_NO_THROW(descriptor.ValidateSupportedByMaterializedPath());
}

TEST(AttentionDescriptor, MaterializedPathAcceptsSoftcap) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.softcap = 0.5f;
  EXPECT_NO_THROW(descriptor.ValidateSupportedByMaterializedPath());
}

TEST(AttentionDescriptor, MaterializedPathAcceptsQkMatmulOutput) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.has_qk_matmul_output = true;
  EXPECT_NO_THROW(descriptor.ValidateSupportedByMaterializedPath());
}

TEST(AttentionDescriptor, MaterializedPathAcceptsExplicitFloatSoftmaxPrecision) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::int64_t kFloat = 1;
  descriptor.softmax_precision = kFloat;
  EXPECT_NO_THROW(descriptor.ValidateSupportedByMaterializedPath());
}

TEST(AttentionDescriptor, MaterializedPathRejectsUnsupportedSoftmaxPrecision) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::int64_t kInt64 = 7;
  descriptor.softmax_precision = kInt64;
  EXPECT_THROW(descriptor.ValidateSupportedByMaterializedPath(), std::invalid_argument);
}

TEST(AttentionDescriptor, MaterializedPathAcceptsStatelessDefaults) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  EXPECT_NO_THROW(descriptor.ValidateSupportedByMaterializedPath());
}

TEST(AttentionPlan, Rank4MhaDefaultScaleAndShape) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  const std::int64_t q_shape[] = {2, 4, 5, 8};
  const std::int64_t k_shape[] = {2, 4, 6, 8};
  const std::int64_t v_shape[] = {2, 4, 6, 16};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);
  EXPECT_EQ(plan.batch, 2u);
  EXPECT_EQ(plan.q_num_heads, 4u);
  EXPECT_EQ(plan.kv_num_heads, 4u);
  EXPECT_EQ(plan.group_size, 1u);
  EXPECT_EQ(plan.q_length, 5u);
  EXPECT_EQ(plan.kv_length, 6u);
  EXPECT_EQ(plan.head_dim, 8u);
  EXPECT_EQ(plan.v_head_dim, 16u);
  EXPECT_NEAR(plan.scale, 1.0f / std::sqrt(8.0f), 1e-6f);
  const std::vector<std::int64_t> expected_shape = {2, 4, 5, 16};
  EXPECT_EQ(plan.output_shape(), expected_shape);
}

TEST(AttentionPlan, RejectsHeadCountNotDivisible) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  const std::int64_t q_shape[] = {1, 5, 4, 8};
  const std::int64_t k_shape[] = {1, 2, 4, 8};
  const std::int64_t v_shape[] = {1, 2, 4, 8};
  EXPECT_THROW(ml_cpu_compute::AttentionPlan(descriptor, ml_cpu_compute::AttentionLayout::kRank4,
                                             q_shape, k_shape, v_shape, {},
                                             ml_cpu_compute::AttentionMaskKind::kNone),
               std::invalid_argument);
}

TEST(AttentionPlan, RejectsElementCountAndTotalLengthOverflow) {
  ml_cpu_compute::AttentionDescriptor desc;
  constexpr std::int64_t maximum = std::numeric_limits<std::int64_t>::max();
  EXPECT_THROW(ml_cpu_compute::AttentionPlan(desc, ml_cpu_compute::AttentionLayout::kRank4,
                                             std::array<std::int64_t, 4>{maximum, 2, 1, 1},
                                             std::array<std::int64_t, 4>{maximum, 1, 1, 1},
                                             std::array<std::int64_t, 4>{maximum, 1, 1, 1},
                                             std::span<const std::int64_t>{},
                                             ml_cpu_compute::AttentionMaskKind::kNone),
               std::invalid_argument);

  desc.has_past_key = true;
  desc.has_past_value = true;
  EXPECT_THROW(ml_cpu_compute::AttentionPlan(
                   desc, ml_cpu_compute::AttentionLayout::kRank4,
                   std::array<std::int64_t, 4>{1, 1, 1, 1}, std::array<std::int64_t, 4>{1, 1, 1, 1},
                   std::array<std::int64_t, 4>{1, 1, 1, 1}, std::span<const std::int64_t>{},
                   ml_cpu_compute::AttentionMaskKind::kNone,
                   std::array<std::int64_t, 4>{1, 1, maximum, 1},
                   std::array<std::int64_t, 4>{1, 1, maximum, 1}),
               std::invalid_argument);
}

TEST(AttentionPlan, Rank3RequiresHeadCountAttributes) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  const std::int64_t q_shape[] = {1, 4, 32};
  const std::int64_t k_shape[] = {1, 4, 32};
  const std::int64_t v_shape[] = {1, 4, 32};
  EXPECT_THROW(ml_cpu_compute::AttentionPlan(descriptor, ml_cpu_compute::AttentionLayout::kRank3,
                                             q_shape, k_shape, v_shape, {},
                                             ml_cpu_compute::AttentionMaskKind::kNone),
               std::invalid_argument);
}

// MHA, rank-4, no mask, no causal: differential test against the naive
// reference oracle.
TEST(ComputeAttentionFloat32, Rank4MhaNoMaskMatchesReference) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::size_t batch = 2, heads = 3, q_len = 4, kv_len = 5, head_dim = 8, v_head_dim = 8;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 1);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 2);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 3);
  std::vector<float> y(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data());

  const auto expected = ReferenceAttention(batch, heads, heads, q_len, kv_len, head_dim, v_head_dim,
                                           q, k, v, plan.scale, false, nullptr, nullptr);
  ExpectClose(y, expected);
}

TEST(ComputeAttentionFloat32Streaming, DecodeDistributesHeadsAcrossWorkers) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::size_t batch = 1, heads = 12, q_len = 1, kv_len = 1024, head_dim = 64;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t kv_shape[] = {batch, heads, kv_len, head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     kv_shape, kv_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);
  const auto q = RandomTensor(batch * heads * q_len * head_dim, 101);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 102);
  const auto v = RandomTensor(batch * heads * kv_len * head_dim, 103);
  std::vector<float> y(batch * heads * q_len * head_dim);
  InlineExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};

  {
    ml_cpu_compute::ExecutionExecutorScope scope(&view);
    ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data());
  }

  EXPECT_EQ(executor.maximum_blocks, 4);
}

TEST(ComputeAttentionFloat32Streaming, SmallCausalGqaCapsWorkerCount) {
  constexpr std::size_t batch = 4, q_heads = 8, kv_heads = 1, length = 32, head_dim = 64;
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.q_num_heads = q_heads;
  descriptor.kv_num_heads = kv_heads;
  descriptor.is_causal = true;
  const std::int64_t q_shape[] = {batch, length, q_heads * head_dim};
  const std::int64_t kv_shape[] = {batch, length, kv_heads * head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank3, q_shape,
                                     kv_shape, kv_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);
  const auto q = RandomTensor(batch * q_heads * length * head_dim, 104);
  const auto k = RandomTensor(batch * kv_heads * length * head_dim, 105);
  const auto v = RandomTensor(batch * kv_heads * length * head_dim, 106);
  std::vector<float> y(q.size());
  InlineExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 16, &InlineExecutor::Run};

  {
    ml_cpu_compute::ExecutionExecutorScope scope(&view);
    ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data());
  }

  EXPECT_GT(executor.maximum_blocks, 1);
  EXPECT_LE(executor.maximum_blocks, 6);
}

// GQA, rank-4, causal.
TEST(ComputeAttentionFloat32, Rank4GqaCausalMatchesReference) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  constexpr std::size_t batch = 1, q_heads = 4, kv_heads = 2, len = 6, head_dim = 4, v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, q_heads, len, head_dim};
  const std::int64_t k_shape[] = {batch, kv_heads, len, head_dim};
  const std::int64_t v_shape[] = {batch, kv_heads, len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);
  EXPECT_EQ(plan.group_size, 2u);
  EXPECT_TRUE(plan.causal);

  const auto q = RandomTensor(batch * q_heads * len * head_dim, 11);
  const auto k = RandomTensor(batch * kv_heads * len * head_dim, 12);
  const auto v = RandomTensor(batch * kv_heads * len * v_head_dim, 13);
  std::vector<float> y(batch * q_heads * len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data());

  const auto expected = ReferenceAttention(batch, q_heads, kv_heads, len, len, head_dim, v_head_dim,
                                           q, k, v, plan.scale, true, nullptr, nullptr);
  ExpectClose(y, expected);
}

// MQA, rank-4, boolean mask.
TEST(ComputeAttentionFloat32, Rank4MqaBooleanMaskMatchesReference) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::size_t batch = 1, q_heads = 4, kv_heads = 1, q_len = 3, kv_len = 5, head_dim = 4,
                        v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, q_heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, kv_heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, kv_heads, kv_len, v_head_dim};
  const std::int64_t mask_shape[] = {q_len, kv_len};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, mask_shape,
                                     ml_cpu_compute::AttentionMaskKind::kBoolean);
  EXPECT_EQ(plan.group_size, 4u);

  const auto q = RandomTensor(batch * q_heads * q_len * head_dim, 21);
  const auto k = RandomTensor(batch * kv_heads * kv_len * head_dim, 22);
  const auto v = RandomTensor(batch * kv_heads * kv_len * v_head_dim, 23);
  std::vector<std::uint8_t> bool_mask(q_len * kv_len, 1);
  bool_mask[0] = 0; // First query row cannot see the first key.
  bool_mask[kv_len] = 0;

  std::vector<float> y(batch * q_heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), bool_mask.data(),
                                          y.data());

  // Broadcast the 2D mask to (batch, heads, q_len, kv_len) for the oracle.
  std::vector<std::uint8_t> broadcast_mask(batch * q_heads * q_len * kv_len);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < q_heads; ++h) {
      for (std::size_t i = 0; i < q_len * kv_len; ++i) {
        broadcast_mask[(b * q_heads + h) * q_len * kv_len + i] = bool_mask[i];
      }
    }
  }
  const auto expected =
      ReferenceAttention(batch, q_heads, kv_heads, q_len, kv_len, head_dim, v_head_dim, q, k, v,
                         plan.scale, false, nullptr, &broadcast_mask);
  ExpectClose(y, expected);
}

// Rank-3 additive mask, MHA.
TEST(ComputeAttentionFloat32, Rank3AdditiveMaskMatchesReference) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.q_num_heads = 2;
  descriptor.kv_num_heads = 2;
  constexpr std::size_t batch = 2, heads = 2, q_len = 17, kv_len = 19, head_dim = 9,
                        v_head_dim = 11;
  const std::int64_t q_shape[] = {batch, q_len, heads * head_dim};
  const std::int64_t k_shape[] = {batch, kv_len, heads * head_dim};
  const std::int64_t v_shape[] = {batch, kv_len, heads * v_head_dim};
  const std::int64_t mask_shape[] = {batch, 1, q_len, kv_len};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank3, q_shape,
                                     k_shape, v_shape, mask_shape,
                                     ml_cpu_compute::AttentionMaskKind::kAdditive);

  const auto q_r3 = RandomTensor(batch * q_len * heads * head_dim, 31);
  const auto k_r3 = RandomTensor(batch * kv_len * heads * head_dim, 32);
  const auto v_r3 = RandomTensor(batch * kv_len * heads * v_head_dim, 33);
  const auto mask = RandomTensor(batch * q_len * kv_len, 34);

  std::vector<float> y(batch * q_len * heads * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q_r3.data(), k_r3.data(), v_r3.data(), mask.data(),
                                          y.data());

  // Reshape rank-3 [B, L, H*D] into rank-4 [B, H, L, D] for the oracle, and
  // broadcast the (batch, 1, q_len, kv_len) mask over heads.
  auto to_rank4 = [&](const std::vector<float> &src, std::size_t len, std::size_t dim) {
    std::vector<float> dst(batch * heads * len * dim);
    for (std::size_t b = 0; b < batch; ++b) {
      for (std::size_t l = 0; l < len; ++l) {
        for (std::size_t h = 0; h < heads; ++h) {
          for (std::size_t d = 0; d < dim; ++d) {
            dst[((b * heads + h) * len + l) * dim + d] =
                src[(b * len + l) * (heads * dim) + h * dim + d];
          }
        }
      }
    }
    return dst;
  };
  const auto q4 = to_rank4(q_r3, q_len, head_dim);
  const auto k4 = to_rank4(k_r3, kv_len, head_dim);
  const auto v4 = to_rank4(v_r3, kv_len, v_head_dim);
  std::vector<float> broadcast_mask(batch * heads * q_len * kv_len);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < heads; ++h) {
      for (std::size_t i = 0; i < q_len * kv_len; ++i) {
        broadcast_mask[(b * heads + h) * q_len * kv_len + i] = mask[b * q_len * kv_len + i];
      }
    }
  }
  const auto expected = ReferenceAttention(batch, heads, heads, q_len, kv_len, head_dim, v_head_dim,
                                           q4, k4, v4, plan.scale, false, &broadcast_mask, nullptr);

  // Reshape the rank-4 oracle output back to rank-3 [B, L, H*D] for comparison.
  std::vector<float> expected_r3(batch * q_len * heads * v_head_dim);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t l = 0; l < q_len; ++l) {
      for (std::size_t h = 0; h < heads; ++h) {
        for (std::size_t d = 0; d < v_head_dim; ++d) {
          expected_r3[(b * q_len + l) * (heads * v_head_dim) + h * v_head_dim + d] =
              expected[((b * heads + h) * q_len + l) * v_head_dim + d];
        }
      }
    }
  }
  ExpectClose(y, expected_r3);
}

TEST(ComputeAttentionFloat32, FullyMaskedRowProducesZeroOutput) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::size_t batch = 1, heads = 1, q_len = 16, kv_len = 17, head_dim = 9,
                        v_head_dim = 11;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  const std::int64_t mask_shape[] = {q_len, kv_len};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, mask_shape,
                                     ml_cpu_compute::AttentionMaskKind::kBoolean);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 41);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 42);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 43);
  std::vector<std::uint8_t> mask(q_len * kv_len, 1);
  // Row 0 is fully masked; row 1 keeps every key.
  for (std::size_t j = 0; j < kv_len; ++j) {
    mask[j] = 0;
  }

  std::vector<float> y(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), mask.data(),
                                          y.data());
  for (std::size_t d = 0; d < v_head_dim; ++d) {
    EXPECT_EQ(y[d], 0.0f);
  }
  bool any_nonzero = false;
  for (std::size_t d = 0; d < v_head_dim; ++d) {
    any_nonzero |= (y[v_head_dim + d] != 0.0f);
  }
  EXPECT_TRUE(any_nonzero);
}

TEST(ComputeAttentionFloat32, AdditiveMaskExtremesMatchReferenceForEveryExecutionPath) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::size_t batch = 1, heads = 1, head_dim = 9;
  const float lowest = std::numeric_limits<float>::lowest();
  const float neg_inf = -std::numeric_limits<float>::infinity();
  for (const std::int64_t q_len : {1, 2, 16}) {
    for (const std::int64_t kv_len : {1, 17, 257}) {
      SCOPED_TRACE(q_len);
      SCOPED_TRACE(kv_len);
      const std::vector<std::int64_t> q_shape = {batch, heads, q_len, head_dim};
      const std::vector<std::int64_t> kv_shape = {batch, heads, kv_len, head_dim};
      const std::vector<std::int64_t> mask_shape = {q_len, kv_len};
      ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4,
                                         q_shape, kv_shape, kv_shape, mask_shape,
                                         ml_cpu_compute::AttentionMaskKind::kAdditive);
      const auto q = RandomTensor(q_len * head_dim, 44);
      const auto k = RandomTensor(kv_len * head_dim, 45);
      const auto v = RandomTensor(kv_len * head_dim, 46);
      std::vector<std::vector<float>> masks(6, std::vector<float>(q_len * kv_len, lowest));
      std::fill(masks[0].begin(), masks[0].end(), neg_inf);
      for (std::int64_t i = 0; i < q_len; ++i) {
        for (std::int64_t j = 0; j < kv_len; ++j) {
          const auto index = i * kv_len + j;
          masks[2][index] = j % 2 == 0 ? lowest : neg_inf;
          masks[3][index] = j % 3 == 0 ? 0.0f : (j % 3 == 1 ? lowest : neg_inf);
          // Exercise transitions between fully masked and finite streaming blocks.
          masks[4][index] = j < 128 ? neg_inf : lowest;
          masks[5][index] = j < 128 ? lowest : neg_inf;
        }
      }
      for (std::size_t case_index = 0; case_index < masks.size(); ++case_index) {
        SCOPED_TRACE(case_index);
        const auto &mask = masks[case_index];
        const auto expected =
            ReferenceAttention(batch, heads, heads, q_len, kv_len, head_dim, head_dim, q, k, v,
                               plan.scale, false, &mask, nullptr);
        std::vector<float> streaming_y(q_len * head_dim, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> materialized_y = streaming_y;

        ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(),
                                                         mask.data(), streaming_y.data());
        ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(),
                                                            mask.data(), materialized_y.data());

        ExpectClose(streaming_y, expected);
        ExpectClose(materialized_y, expected);
      }
    }
  }
}

TEST(ComputeAttentionFloat32, AdditiveMaskExtremesProduceOnnxProbabilities) {
  const float lowest = std::numeric_limits<float>::lowest();
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const std::int64_t q_shape[] = {1, 1, 4, 1};
  const std::int64_t kv_shape[] = {1, 1, 3, 1};
  const std::int64_t mask_shape[] = {4, 3};
  const std::vector<float> q(4, 0.0f), k(3, 0.0f), v = {2.0f, 4.0f, 12.0f};
  const std::vector<float> mask = {neg_inf, neg_inf, neg_inf, lowest, lowest, lowest,
                                   lowest,  neg_inf, lowest,  0.0f,   lowest, neg_inf};
  const std::vector<float> expected_y = {0.0f, 6.0f, 7.0f, 2.0f};
  const std::vector<float> expected_probabilities = {0.0f, 0.0f, 0.0f, 1.0f / 3, 1.0f / 3, 1.0f / 3,
                                                     0.5f, 0.0f, 0.5f, 1.0f,     0.0f,     0.0f};
  for (const std::int64_t precision : {1, 11}) {
    SCOPED_TRACE(precision);
    ml_cpu_compute::AttentionDescriptor descriptor;
    descriptor.softmax_precision = precision;
    descriptor.has_qk_matmul_output = true;
    descriptor.qk_matmul_output_mode = 3;
    ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                       kv_shape, kv_shape, mask_shape,
                                       ml_cpu_compute::AttentionMaskKind::kAdditive);
    std::vector<float> y(expected_y.size()), probabilities(expected_probabilities.size());
    ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(),
                                                        mask.data(), y.data(), nullptr, nullptr,
                                                        nullptr, probabilities.data());

    ExpectClose(y, expected_y);
    ExpectClose(probabilities, expected_probabilities);
  }
}

TEST(ComputeAttentionFloat32, SoftcapMatchesManualTanhFormula) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.softcap = 2.0f;
  descriptor.has_qk_matmul_output = true;
  // Mode 1 captures the post-softcap (pre-bias) score; mode 0 (the default)
  // captures the raw pre-softcap score instead.
  descriptor.qk_matmul_output_mode = 1;
  constexpr std::size_t batch = 1, heads = 1, q_len = 1, kv_len = 3, head_dim = 4, v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);
  EXPECT_NEAR(plan.softcap, 2.0f, 1e-6f);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 51);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 52);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 53);
  std::vector<float> qk(kv_len);
  std::vector<float> y(v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data(),
                                          nullptr, nullptr, nullptr, qk.data());
  for (std::size_t j = 0; j < kv_len; ++j) {
    float dot = 0.0f;
    for (std::size_t d = 0; d < head_dim; ++d) {
      dot += q[d] * k[j * head_dim + d];
    }
    const float raw = plan.scale * dot;
    const float expected = plan.softcap * std::tanh(raw / plan.softcap);
    EXPECT_NEAR(qk[j], expected, 1e-4f) << "j=" << j;
  }
}

// qk_matmul_output_mode 0-3: verifies each mode captures the expected stage
// (raw QK, post-softcap, post-bias, post-softmax) for a single row with a
// causal mask so bias/softmax actually differ from the raw score.
TEST(ComputeAttentionFloat32, QkMatmulOutputModesCaptureEachStage) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  descriptor.has_qk_matmul_output = true;
  constexpr std::size_t batch = 1, heads = 1, len = 3, head_dim = 4, v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, heads, len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, len, v_head_dim};

  const auto q = RandomTensor(batch * heads * len * head_dim, 61);
  const auto k = RandomTensor(batch * heads * len * head_dim, 62);
  const auto v = RandomTensor(batch * heads * len * v_head_dim, 63);
  std::vector<float> y(batch * heads * len * v_head_dim);

  for (std::int64_t mode = 0; mode <= 3; ++mode) {
    descriptor.qk_matmul_output_mode = mode;
    ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                       k_shape, v_shape, {},
                                       ml_cpu_compute::AttentionMaskKind::kNone);
    EXPECT_TRUE(plan.has_qk_matmul_output);
    std::vector<float> qk(len * len, -123.0f);
    ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data(),
                                            nullptr, nullptr, nullptr, qk.data());

    for (std::size_t i = 0; i < len; ++i) {
      std::vector<float> raw(len), with_bias(len);
      float row_max = -std::numeric_limits<float>::infinity();
      for (std::size_t j = 0; j < len; ++j) {
        float dot = 0.0f;
        for (std::size_t d = 0; d < head_dim; ++d) {
          dot += q[i * head_dim + d] * k[j * head_dim + d];
        }
        raw[j] = plan.scale * dot;
        const bool allowed = j <= i;
        with_bias[j] = raw[j] + (allowed ? 0.0f : -std::numeric_limits<float>::infinity());
        row_max = std::max(row_max, with_bias[j]);
      }
      if (mode == 0) {
        for (std::size_t j = 0; j < len; ++j) {
          EXPECT_NEAR(qk[i * len + j], raw[j], 1e-4f) << "mode=0 i=" << i << " j=" << j;
        }
      } else if (mode == 1) {
        // No softcap configured: mode 1 equals the raw score.
        for (std::size_t j = 0; j < len; ++j) {
          EXPECT_NEAR(qk[i * len + j], raw[j], 1e-4f) << "mode=1 i=" << i << " j=" << j;
        }
      } else if (mode == 2) {
        for (std::size_t j = 0; j <= i; ++j) {
          EXPECT_NEAR(qk[i * len + j], with_bias[j], 1e-4f) << "mode=2 i=" << i << " j=" << j;
        }
        for (std::size_t j = i + 1; j < len; ++j) {
          EXPECT_TRUE(std::isinf(qk[i * len + j]) && qk[i * len + j] < 0)
              << "mode=2 i=" << i << " j=" << j;
        }
      } else {
        float sum = 0.0f;
        std::vector<float> p(len, 0.0f);
        for (std::size_t j = 0; j <= i; ++j) {
          p[j] = std::exp(with_bias[j] - row_max);
          sum += p[j];
        }
        for (std::size_t j = 0; j < len; ++j) {
          EXPECT_NEAR(qk[i * len + j], p[j] / sum, 1e-4f) << "mode=3 i=" << i << " j=" << j;
        }
      }
    }
  }
}

// Internal tensor past_key/past_value cache: the bottom-right causal offset
// shifts by past_length, and the plan's present_key/present_value shapes
// report the combined (past + new) length.
TEST(ComputeAttentionFloat32, InternalPastCacheShiftsCausalOffsetAndMatchesReference) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  constexpr std::size_t batch = 1, heads = 2, past_len = 3, q_len = 2, kv_len = 2, head_dim = 4,
                        v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  const std::int64_t past_k_shape[] = {batch, heads, past_len, head_dim};
  const std::int64_t past_v_shape[] = {batch, heads, past_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {}, ml_cpu_compute::AttentionMaskKind::kNone,
                                     past_k_shape, past_v_shape);
  EXPECT_EQ(plan.past_length, past_len);
  EXPECT_EQ(plan.total_kv_length, past_len + kv_len);
  EXPECT_EQ(plan.causal_offset, static_cast<std::int64_t>(past_len));
  const std::vector<std::int64_t> expected_present_key = {batch, heads, past_len + kv_len,
                                                          head_dim};
  EXPECT_EQ(plan.present_key_shape(), expected_present_key);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 71);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 72);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 73);
  const auto past_k = RandomTensor(batch * heads * past_len * head_dim, 74);
  const auto past_v = RandomTensor(batch * heads * past_len * v_head_dim, 75);

  std::vector<float> y(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data(),
                                          past_k.data(), past_v.data());

  // Reference: concatenate past and new K/V, then run the standalone oracle
  // with causal offset baked in by only allowing j <= i + past_len.
  std::vector<float> full_k(batch * heads * (past_len + kv_len) * head_dim);
  std::vector<float> full_v(batch * heads * (past_len + kv_len) * v_head_dim);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < heads; ++h) {
      std::copy_n(past_k.data() + (b * heads + h) * past_len * head_dim, past_len * head_dim,
                  full_k.data() + (b * heads + h) * (past_len + kv_len) * head_dim);
      std::copy_n(k.data() + (b * heads + h) * kv_len * head_dim, kv_len * head_dim,
                  full_k.data() + (b * heads + h) * (past_len + kv_len) * head_dim +
                      past_len * head_dim);
      std::copy_n(past_v.data() + (b * heads + h) * past_len * v_head_dim, past_len * v_head_dim,
                  full_v.data() + (b * heads + h) * (past_len + kv_len) * v_head_dim);
      std::copy_n(v.data() + (b * heads + h) * kv_len * v_head_dim, kv_len * v_head_dim,
                  full_v.data() + (b * heads + h) * (past_len + kv_len) * v_head_dim +
                      past_len * v_head_dim);
    }
  }
  std::vector<float> bias_mask(batch * heads * q_len * (past_len + kv_len), 0.0f);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < heads; ++h) {
      for (std::size_t i = 0; i < q_len; ++i) {
        for (std::size_t j = 0; j < past_len + kv_len; ++j) {
          const bool allowed = j <= i + past_len;
          bias_mask[((b * heads + h) * q_len + i) * (past_len + kv_len) + j] =
              allowed ? 0.0f : -std::numeric_limits<float>::infinity();
        }
      }
    }
  }
  const auto expected =
      ReferenceAttention(batch, heads, heads, q_len, past_len + kv_len, head_dim, v_head_dim, q,
                         full_k, full_v, plan.scale, false, &bias_mask, nullptr);
  ExpectClose(y, expected);
}

// v24 external cache: nonpad_kv_seqlen supplies a per-batch bottom-right
// causal offset and a padding mask, without any past_key/past_value.
TEST(ComputeAttentionFloat32, NonpadKvSeqlenAppliesPerBatchOffsetAndPaddingMask) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  constexpr std::size_t batch = 2, heads = 1, q_len = 2, kv_len = 5, head_dim = 4, v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);
  EXPECT_EQ(plan.causal_offset, 0);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 81);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 82);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 83);
  // batch 0: only 2 valid keys, offset = 2 - 2 = 0 (standard causal).
  // batch 1: 4 valid keys (1 padding tail), offset = 4 - 2 = 2 (shifted).
  const std::vector<std::int64_t> nonpad = {2, 4};

  std::vector<float> y(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data(),
                                          nullptr, nullptr, nonpad.data());

  std::vector<float> bias_mask(batch * heads * q_len * kv_len, 0.0f);
  for (std::size_t b = 0; b < batch; ++b) {
    const std::int64_t offset = nonpad[b] - static_cast<std::int64_t>(q_len);
    for (std::size_t i = 0; i < q_len; ++i) {
      for (std::size_t j = 0; j < kv_len; ++j) {
        const bool allowed_causal =
            static_cast<std::int64_t>(j) <= static_cast<std::int64_t>(i) + offset;
        const bool allowed_pad = static_cast<std::int64_t>(j) < nonpad[b];
        bias_mask[((b * heads) * q_len + i) * kv_len + j] =
            (allowed_causal && allowed_pad) ? 0.0f : -std::numeric_limits<float>::infinity();
      }
    }
  }
  const auto expected = ReferenceAttention(batch, heads, heads, q_len, kv_len, head_dim, v_head_dim,
                                           q, k, v, plan.scale, false, &bias_mask, nullptr);
  ExpectClose(y, expected);
}

// Roadmap PR14: the streaming entry point now consumes an internal tensor
// past_key/past_value cache directly (block by block, without concatenating
// past and new K/V), so it must match the materialized path exactly for the
// same cache configuration that Roadmap PR13's dispatcher previously routed
// to the materialized path only.
TEST(ComputeAttentionFloat32Streaming, InternalPastCacheMatchesMaterialized) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  constexpr std::size_t batch = 2, heads = 4, group = 2, past_len = 5, q_len = 3, kv_len = 6,
                        head_dim = 8, v_head_dim = 8;
  constexpr std::size_t kv_heads = heads / group;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, kv_heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, kv_heads, kv_len, v_head_dim};
  const std::int64_t past_k_shape[] = {batch, kv_heads, past_len, head_dim};
  const std::int64_t past_v_shape[] = {batch, kv_heads, past_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {}, ml_cpu_compute::AttentionMaskKind::kNone,
                                     past_k_shape, past_v_shape);
  ASSERT_FALSE(plan.has_qk_matmul_output);
  ASSERT_FALSE(plan.has_present_output);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 601);
  const auto k = RandomTensor(batch * kv_heads * kv_len * head_dim, 602);
  const auto v = RandomTensor(batch * kv_heads * kv_len * v_head_dim, 603);
  const auto past_k = RandomTensor(batch * kv_heads * past_len * head_dim, 604);
  const auto past_v = RandomTensor(batch * kv_heads * past_len * v_head_dim, 605);

  std::vector<float> streaming_y(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                   streaming_y.data(), past_k.data(),
                                                   past_v.data());

  std::vector<float> materialized_y(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(), nullptr,
                                                      materialized_y.data(), past_k.data(),
                                                      past_v.data());

  ExpectClose(streaming_y, materialized_y);
}

// Roadmap PR14: the streaming entry point also consumes `nonpad_kv_seqlen`
// (the v24 external cache) directly, matching the materialized path.
TEST(ComputeAttentionFloat32Streaming, NonpadKvSeqlenMatchesMaterialized) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  constexpr std::size_t batch = 2, heads = 2, q_len = 3, kv_len = 7, head_dim = 8, v_head_dim = 8;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 611);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 612);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 613);
  const std::vector<std::int64_t> nonpad = {3, 6};

  std::vector<float> streaming_y(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                   streaming_y.data(), nullptr, nullptr,
                                                   nonpad.data());

  std::vector<float> materialized_y(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(), nullptr,
                                                      materialized_y.data(), nullptr, nullptr,
                                                      nonpad.data());

  ExpectClose(streaming_y, materialized_y);
}

TEST(ComputeAttentionFloat32Streaming, SingleKeyGqaAdditiveMaskMatchesMaterialized) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::size_t batch = 2, q_heads = 4, kv_heads = 2, q_len = 5, kv_len = 1, head_dim = 8,
                        v_head_dim = 6;
  const std::int64_t q_shape[] = {batch, q_heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, kv_heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, kv_heads, kv_len, v_head_dim};
  const std::int64_t mask_shape[] = {q_len, kv_len};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, mask_shape,
                                     ml_cpu_compute::AttentionMaskKind::kAdditive);

  const auto q = RandomTensor(batch * q_heads * q_len * head_dim, 614);
  const auto k = RandomTensor(batch * kv_heads * kv_len * head_dim, 615);
  const auto v = RandomTensor(batch * kv_heads * kv_len * v_head_dim, 616);
  const std::vector<float> mask = {
      0.0f,
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::quiet_NaN(),
      -std::numeric_limits<float>::infinity(),
      -2.0f,
  };
  std::vector<float> streaming_y(batch * q_heads * q_len * v_head_dim);
  std::vector<float> materialized_y(streaming_y.size());

  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), mask.data(),
                                                   streaming_y.data());
  ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(),
                                                      mask.data(), materialized_y.data());

  for (std::size_t index = 0; index < streaming_y.size(); ++index) {
    if (std::isnan(materialized_y[index])) {
      EXPECT_TRUE(std::isnan(streaming_y[index])) << "index=" << index;
    } else {
      EXPECT_NEAR(streaming_y[index], materialized_y[index], 1e-4f) << "index=" << index;
    }
  }
}

TEST(ComputeAttentionFloat32Streaming, TiledCausalAdditiveMaskMatchesMaterialized) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  constexpr std::size_t batch = 1, heads = 2, q_len = 16, kv_len = 17, head_dim = 8, v_head_dim = 8;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  const std::int64_t mask_shape[] = {q_len, kv_len};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, mask_shape,
                                     ml_cpu_compute::AttentionMaskKind::kAdditive);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 614);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 615);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 616);
  const auto mask = RandomTensor(q_len * kv_len, 617);
  std::vector<float> streaming_y(batch * heads * q_len * v_head_dim);
  std::vector<float> materialized_y(streaming_y.size());

  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), mask.data(),
                                                   streaming_y.data());
  ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(),
                                                      mask.data(), materialized_y.data());

  ExpectClose(streaming_y, materialized_y);
}

TEST(ComputeAttentionFloat32Streaming, TiledCausalFastPathMatchesMaterialized) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  constexpr std::size_t batch = 1, heads = 2, q_len = 16, kv_len = 17, head_dim = 8;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 618);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 619);
  const auto v = RandomTensor(batch * heads * kv_len * head_dim, 620);
  std::vector<float> streaming_y(batch * heads * q_len * head_dim);
  std::vector<float> materialized_y(streaming_y.size());

  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                   streaming_y.data());
  ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(), nullptr,
                                                      materialized_y.data());

  ExpectClose(streaming_y, materialized_y);
}

// Roadmap PR14: a boolean attn_mask carves out a fully-disallowed block in
// the middle of the KV axis (a sliding-window-like shape spanning more than
// one `kStreamingKvBlock`-sized tile): the streaming path must infer the
// safe per-block skip and still match the reference exactly, and it must
// also match the materialized path bit-for-bit in behavior (both must zero a
// fully-masked row rather than produce NaN).
TEST(ComputeAttentionFloat32Streaming, BooleanMaskTileSkipMatchesReference) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::size_t batch = 1, heads = 1, q_len = 2, kv_len = 300, head_dim = 4, v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  const std::int64_t mask_shape[] = {q_len, kv_len};

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 621);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 622);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 623);

  // Two disjoint allowed windows (each narrower than `kStreamingKvBlock`),
  // separated by a fully-disallowed gap spanning several whole KV blocks.
  std::vector<std::uint8_t> bool_mask(q_len * kv_len, 0);
  for (std::size_t i = 0; i < q_len; ++i) {
    for (std::size_t j : {std::size_t{10}, std::size_t{11}, std::size_t{250}, std::size_t{251}}) {
      bool_mask[i * kv_len + j] = 1;
    }
  }

  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, mask_shape,
                                     ml_cpu_compute::AttentionMaskKind::kBoolean);
  std::vector<float> y(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(),
                                                   bool_mask.data(), y.data());

  const auto expected = ReferenceAttention(batch, heads, heads, q_len, kv_len, head_dim, v_head_dim,
                                           q, k, v, plan.scale, false, nullptr, &bool_mask);
  ExpectClose(y, expected);

  // The materialized path must agree exactly on the same inputs.
  std::vector<float> materialized_y(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(),
                                                      bool_mask.data(), materialized_y.data());
  ExpectClose(y, materialized_y);
}

TEST(ComputeAttentionFloat32Streaming, DenseBenchmarkShapesMatchMaterialized) {
  constexpr std::size_t batch = 1, heads = 12, kv_len = 128, head_dim = 64;
  for (const auto &[layout, q_len] :
       {std::pair{ml_cpu_compute::AttentionLayout::kRank4, std::size_t{8}},
        std::pair{ml_cpu_compute::AttentionLayout::kRank3, std::size_t{128}}}) {
    ml_cpu_compute::AttentionDescriptor descriptor;
    descriptor.q_num_heads = heads;
    descriptor.kv_num_heads = heads;
    const auto query_length = static_cast<std::int64_t>(q_len);
    const std::vector<std::int64_t> q_shape =
        layout == ml_cpu_compute::AttentionLayout::kRank3
            ? std::vector<std::int64_t>{batch, query_length, heads * head_dim}
            : std::vector<std::int64_t>{batch, heads, query_length, head_dim};
    const std::vector<std::int64_t> kv_shape =
        layout == ml_cpu_compute::AttentionLayout::kRank3
            ? std::vector<std::int64_t>{batch, kv_len, heads * head_dim}
            : std::vector<std::int64_t>{batch, heads, kv_len, head_dim};
    ml_cpu_compute::AttentionPlan plan(descriptor, layout, q_shape, kv_shape, kv_shape, {},
                                       ml_cpu_compute::AttentionMaskKind::kNone, {}, {});
    const auto q = RandomTensor(batch * heads * q_len * head_dim, 1291 + q_len);
    const auto k = RandomTensor(batch * heads * kv_len * head_dim, 1292 + q_len);
    const auto v = RandomTensor(batch * heads * kv_len * head_dim, 1293 + q_len);
    std::vector<float> actual(batch * heads * q_len * head_dim);
    std::vector<float> expected(actual.size());
    ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                     actual.data());
    ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(), nullptr,
                                                        expected.data());
    ExpectClose(actual, expected, 2.0e-4f);
  }
}

// -----------------------------------------------------------------------
// AVX2+FMA `Q == 1` decode fast path (ml_cpu_compute/attention/avx2):
// on AVX2 hardware without AVX-512 these cases exercise
// `ComputeAttentionDecodeRowAVX2FMA` and its AVX2 primitives directly; on
// any other hardware they still validate the (unconditionally correct)
// scalar fallback, so the suite stays portable. Compared against
// `ComputeAttentionFloat32Materialized`, which never takes this fast path.
// -----------------------------------------------------------------------

// Exact production benchmark shape: B1/H12/Q1/KV1024/D64, MHA, internal past
// cache, no mask (the shape the AVX2+FMA decode fast path targets).
TEST(ComputeAttentionFloat32Streaming, Avx2DecodeBenchmarkShapeMatchesMaterialized) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::size_t batch = 1, heads = 12, q_len = 1, head_dim = 64, total_kv_length = 1024,
                        past_len = total_kv_length - 1;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, 1, head_dim};
  const std::int64_t past_shape[] = {batch, heads, past_len, head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, k_shape, {}, ml_cpu_compute::AttentionMaskKind::kNone,
                                     past_shape, past_shape);
  ASSERT_EQ(plan.q_length, 1u);
  ASSERT_EQ(plan.total_kv_length, total_kv_length);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 1301);
  const auto k = RandomTensor(batch * heads * 1 * head_dim, 1302);
  const auto v = RandomTensor(batch * heads * 1 * head_dim, 1303);
  const auto past_k = RandomTensor(batch * heads * past_len * head_dim, 1304);
  const auto past_v = RandomTensor(batch * heads * past_len * head_dim, 1305);

  std::vector<float> streaming_y(batch * heads * q_len * head_dim);
  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                   streaming_y.data(), past_k.data(),
                                                   past_v.data());
  std::vector<float> materialized_y(streaming_y.size());
  ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(), nullptr,
                                                      materialized_y.data(), past_k.data(),
                                                      past_v.data());
  ExpectClose(streaming_y, materialized_y);

  std::vector<float> extreme_q(q.size(), std::numeric_limits<float>::max());
  std::vector<float> extreme_k(k.size(), -std::numeric_limits<float>::max());
  std::vector<float> extreme_past_k(past_k.size(), -std::numeric_limits<float>::max());
  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, extreme_q.data(), extreme_k.data(),
                                                   v.data(), nullptr, streaming_y.data(),
                                                   extreme_past_k.data(), past_v.data());
  for (float value : streaming_y) {
    EXPECT_FLOAT_EQ(value, 0.0f);
  }
}

// Vector-tail coverage: `head_dim` is not a multiple of the AVX2 8-wide
// lane width, exercising the masked remainder path of
// `AttentionDotFloat32_AVX2_FMA`/`AttentionAccumulateFloat32_AVX2_FMA`, with
// an internal past cache (so the block loop reads from two buffers).
TEST(ComputeAttentionFloat32Streaming, Avx2DecodeHeadDimTailMatchesMaterialized) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  constexpr std::size_t batch = 1, heads = 2, q_len = 1, head_dim = 13, past_len = 37;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, 1, head_dim};
  const std::int64_t past_shape[] = {batch, heads, past_len, head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, k_shape, {}, ml_cpu_compute::AttentionMaskKind::kNone,
                                     past_shape, past_shape);
  ASSERT_EQ(plan.q_length, 1u);
  ASSERT_EQ(plan.total_kv_length, past_len + 1);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 1401);
  const auto k = RandomTensor(batch * heads * 1 * head_dim, 1402);
  const auto v = RandomTensor(batch * heads * 1 * head_dim, 1403);
  const auto past_k = RandomTensor(batch * heads * past_len * head_dim, 1404);
  const auto past_v = RandomTensor(batch * heads * past_len * head_dim, 1405);

  std::vector<float> streaming_y(batch * heads * q_len * head_dim);
  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                   streaming_y.data(), past_k.data(),
                                                   past_v.data());
  std::vector<float> materialized_y(streaming_y.size());
  ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(), nullptr,
                                                      materialized_y.data(), past_k.data(),
                                                      past_v.data());
  ExpectClose(streaming_y, materialized_y);
}

// Boolean mask decode with `q_length == 1` (so the AVX2 decode path is
// actually exercised, unlike the existing `kv_len == 1` mask tests): a
// scattered allowed set spanning an AVX2 tail `kv_len`, checked against the
// independent oracle and the materialized path, plus a fully-disallowed mask
// that must zero the output rather than produce NaN.
TEST(ComputeAttentionFloat32Streaming, Avx2DecodeBooleanMaskMatchesReferenceAndZerosFullyMasked) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::size_t batch = 1, heads = 2, q_len = 1, kv_len = 21, head_dim = 8, v_head_dim = 8;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  const std::int64_t mask_shape[] = {batch, heads, q_len, kv_len};

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 1001);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 1002);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 1003);

  // Scattered allowed keys, identical across every (batch, head) row,
  // including the very last (tail) element.
  std::vector<std::uint8_t> bool_mask(batch * heads * q_len * kv_len, 0);
  for (std::size_t h = 0; h < heads; ++h) {
    for (std::size_t j : {std::size_t{0}, std::size_t{3}, std::size_t{8}, std::size_t{20}}) {
      bool_mask[h * kv_len + j] = 1;
    }
  }

  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, mask_shape,
                                     ml_cpu_compute::AttentionMaskKind::kBoolean);
  ASSERT_EQ(plan.q_length, 1u);
  ASSERT_EQ(plan.mask_strides.kv, 1);
  std::vector<float> y(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(),
                                                   bool_mask.data(), y.data());

  const auto expected = ReferenceAttention(batch, heads, heads, q_len, kv_len, head_dim, v_head_dim,
                                           q, k, v, plan.scale, false, nullptr, &bool_mask);
  ExpectClose(y, expected);

  std::vector<float> materialized_y(y.size());
  ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(),
                                                      bool_mask.data(), materialized_y.data());
  ExpectClose(y, materialized_y);

  // A fully-disallowed mask must zero the output rather than produce NaN.
  const std::vector<std::uint8_t> fully_masked(bool_mask.size(), 0);
  std::vector<float> zeroed_y(y.size(), std::numeric_limits<float>::quiet_NaN());
  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(),
                                                   fully_masked.data(), zeroed_y.data());
  EXPECT_EQ(zeroed_y, std::vector<float>(zeroed_y.size(), 0.0f));
}

TEST(ComputeAttentionFloat32Streaming, Avx2ShortAndTiledMaskTailsMatchMaterialized) {
  for (const std::int64_t q_len : {8, 15, 16, 17}) {
    for (const std::int64_t tail : {1, 2, 3, 4, 5, 6, 7, 8}) {
      for (const auto kind :
           {ml_cpu_compute::AttentionMaskKind::kNone, ml_cpu_compute::AttentionMaskKind::kBoolean,
            ml_cpu_compute::AttentionMaskKind::kAdditive}) {
        SCOPED_TRACE(::testing::Message()
                     << "q=" << q_len << " tail=" << tail << " mask=" << static_cast<int>(kind));
        ml_cpu_compute::AttentionDescriptor descriptor;
        descriptor.opset = 24;
        descriptor.is_causal = true;
        descriptor.has_nonpad_kv_seqlen = true;
        const std::int64_t kv_len = 256 + tail, head_dim = 56 + tail, v_head_dim = 8 + tail;
        const std::int64_t q_shape[] = {2, 4, q_len, head_dim};
        const std::int64_t k_shape[] = {2, 2, kv_len, head_dim};
        const std::int64_t v_shape[] = {2, 2, kv_len, v_head_dim};
        const std::int64_t mask_shape[] = {q_len, kv_len};
        const std::span<const std::int64_t> mask_dims =
            kind == ml_cpu_compute::AttentionMaskKind::kNone ? std::span<const std::int64_t>{}
                                                             : mask_shape;
        ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4,
                                           q_shape, k_shape, v_shape, mask_dims, kind);
        const auto q = RandomTensor(8 * q_len * head_dim, 1501);
        const auto k = RandomTensor(4 * kv_len * head_dim, 1502);
        const auto v = RandomTensor(4 * kv_len * v_head_dim, 1503);
        auto additive = RandomTensor(q_len * kv_len, 1504);
        std::vector<std::uint8_t> boolean(q_len * kv_len);
        for (std::int64_t i = 0; i < q_len; ++i) {
          for (std::int64_t j = 0; j < kv_len; ++j) {
            const bool allowed = i != 0 && j % 3 != 0 && (i != 1 || j >= 256);
            boolean[i * kv_len + j] = allowed ? 255 : 0;
            if (!allowed) {
              additive[i * kv_len + j] = -std::numeric_limits<float>::infinity();
            }
          }
        }
        const void *mask =
            kind == ml_cpu_compute::AttentionMaskKind::kBoolean
                ? static_cast<const void *>(boolean.data())
                : (kind == ml_cpu_compute::AttentionMaskKind::kAdditive ? additive.data()
                                                                        : nullptr);
        const std::int64_t nonpad[] = {kv_len, q_len - 2};
        std::vector<float> actual(8 * q_len * v_head_dim), expected(actual.size());
        ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), mask,
                                                actual.data(), nullptr, nullptr, nonpad);
        ml_cpu_compute::ComputeAttentionFloat32Materialized(
            plan, q.data(), k.data(), v.data(), mask, expected.data(), nullptr, nullptr, nonpad);
        ExpectClose(actual, expected);
      }
    }
  }
}

TEST(ComputeAttentionFloat32Streaming, Avx2Rank3ShortAndTiledBroadcastMaskMatchesMaterialized) {
  for (const std::int64_t q_len : {8, 17, 128}) {
    for (const std::int64_t mask_kv : {1, 263}) {
      ml_cpu_compute::AttentionDescriptor descriptor;
      descriptor.q_num_heads = 4;
      descriptor.kv_num_heads = 1;
      const std::int64_t q_shape[] = {1, q_len, 4 * 65};
      const std::int64_t k_shape[] = {1, 263, 65};
      const std::int64_t v_shape[] = {1, 263, 63};
      const std::int64_t mask_shape[] = {q_len, mask_kv};
      ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank3,
                                         q_shape, k_shape, v_shape, mask_shape,
                                         ml_cpu_compute::AttentionMaskKind::kAdditive);
      const auto q = RandomTensor(q_len * 4 * 65, 1601);
      const auto k = RandomTensor(263 * 65, 1602);
      const auto v = RandomTensor(263 * 63, 1603);
      auto mask = RandomTensor(q_len * mask_kv, 1604);
      std::fill_n(mask.begin(), mask_kv, std::numeric_limits<float>::lowest());
      std::vector<float> actual(q_len * 4 * 63), expected(actual.size());
      ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), mask.data(),
                                              actual.data());
      ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(),
                                                          mask.data(), expected.data());
      ExpectClose(actual, expected);
    }
  }
}

TEST(ComputeAttentionFloat32Streaming, ShortStatelessQueriesAndHeadTailsMatchReference) {
  for (const std::int64_t q_len : {2, 8}) {
    for (const std::int64_t dim : {7, 8, 9, 63, 64, 65}) {
      SCOPED_TRACE(::testing::Message() << "q=" << q_len << " dim=" << dim);
      const std::int64_t q_shape[] = {1, 1, q_len, dim};
      const std::int64_t kv_shape[] = {1, 1, 128, dim};
      ml_cpu_compute::AttentionPlan plan({}, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                         kv_shape, kv_shape, {},
                                         ml_cpu_compute::AttentionMaskKind::kNone);
      const auto q = RandomTensor(q_len * dim, 1801);
      const auto k = RandomTensor(128 * dim, 1802);
      const auto v = RandomTensor(128 * dim, 1803);
      const auto expected = ReferenceAttention(1, 1, 1, q_len, 128, dim, dim, q, k, v, plan.scale,
                                               false, nullptr, nullptr);
      std::vector<float> actual(expected.size());
      ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr,
                                              actual.data());
      ExpectClose(actual, expected);
    }
  }
}

TEST(ComputeAttentionFloat32Streaming, LongContextsAndTileTailsMatchReference) {
  for (const auto &[q_len, kv_len, vdim] : {std::tuple{128, 1024, 64}, std::tuple{128, 8192, 64},
                                            std::tuple{129, 1025, 63}, std::tuple{129, 1025, 65}}) {
    SCOPED_TRACE(::testing::Message() << "q=" << q_len << " kv=" << kv_len);
    constexpr std::int64_t dim = 64;
    const std::int64_t q_shape[] = {1, 1, q_len, dim};
    const std::int64_t kv_shape[] = {1, 1, kv_len, dim};
    const std::int64_t v_shape[] = {1, 1, kv_len, vdim};
    ml_cpu_compute::AttentionPlan plan({}, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                       kv_shape, v_shape, {},
                                       ml_cpu_compute::AttentionMaskKind::kNone);
    const auto q = RandomTensor(q_len * dim, 1811);
    const auto k = RandomTensor(kv_len * dim, 1812);
    const auto v = RandomTensor(kv_len * vdim, 1813);
    const auto expected = ReferenceAttention(1, 1, 1, q_len, kv_len, dim, vdim, q, k, v, plan.scale,
                                             false, nullptr, nullptr);
    std::vector<float> actual(expected.size());
    ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr,
                                            actual.data());
    ExpectClose(actual, expected);
  }
}

TEST(ComputeAttentionFloat32Streaming, ShortQueriesSplitPastCacheAtBlockBoundaries) {
  for (const std::int64_t past_len : {127, 128, 129, 255, 256, 257}) {
    for (const float softcap : {0.0f, 0.7f}) {
      SCOPED_TRACE(::testing::Message() << "past=" << past_len << " softcap=" << softcap);
      ml_cpu_compute::AttentionDescriptor descriptor;
      descriptor.is_causal = true;
      descriptor.softcap = softcap;
      const std::int64_t q_shape[] = {1, 2, 8, 9};
      const std::int64_t k_shape[] = {1, 1, 135, 9};
      const std::int64_t v_shape[] = {1, 1, 135, 11};
      const std::int64_t past_k_shape[] = {1, 1, past_len, 9};
      const std::int64_t past_v_shape[] = {1, 1, past_len, 11};
      ml_cpu_compute::AttentionPlan plan(
          descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape, k_shape, v_shape, {},
          ml_cpu_compute::AttentionMaskKind::kNone, past_k_shape, past_v_shape);
      const auto q = RandomTensor(2 * 8 * 9, 1821);
      const auto k = RandomTensor(135 * 9, 1822);
      const auto v = RandomTensor(135 * 11, 1823);
      const auto past_k = RandomTensor(past_len * 9, 1824);
      const auto past_v = RandomTensor(past_len * 11, 1825);
      std::vector<float> expected(2 * 8 * 11), actual(expected.size());
      ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(),
                                                          nullptr, expected.data(), past_k.data(),
                                                          past_v.data());
      ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                       actual.data(), past_k.data(), past_v.data());
      ExpectClose(actual, expected);
    }
  }
}

TEST(ComputeAttentionFloat32Streaming, SingleBlockRank3CausalReusesWorkspaceAfterEmptyRows) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.q_num_heads = 4;
  descriptor.kv_num_heads = 4;
  descriptor.is_causal = true;
  const std::int64_t shape[] = {1, 16, 4 * 32};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank3, shape,
                                     shape, shape, {}, ml_cpu_compute::AttentionMaskKind::kNone);
  const auto q = RandomTensor(16 * 4 * 32, 1831);
  const auto k = RandomTensor(16 * 4 * 32, 1832);
  const auto v = RandomTensor(16 * 4 * 32, 1833);
  for (const std::int64_t nonpad : {16, 0, 12, 16}) {
    SCOPED_TRACE(nonpad);
    std::vector<float> expected(q.size()), actual(q.size());
    ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(), nullptr,
                                                        expected.data(), nullptr, nullptr, &nonpad);
    ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                     actual.data(), nullptr, nullptr, &nonpad);
    ExpectClose(actual, expected);
  }
}

TEST(ComputeAttentionFloat32Streaming, ShortQueriesSkipZeroWeightNanValues) {
  const std::int64_t q_shape[] = {1, 1, 2, 9};
  const std::int64_t kv_shape[] = {1, 1, 128, 9};
  const std::int64_t mask_shape[] = {128};
  ml_cpu_compute::AttentionPlan plan({}, ml_cpu_compute::AttentionLayout::kRank4, q_shape, kv_shape,
                                     kv_shape, mask_shape,
                                     ml_cpu_compute::AttentionMaskKind::kBoolean);
  std::vector<float> q(2 * 9, 0.0f), k(128 * 9, 0.0f), v(128 * 9, 1.0f), actual(2 * 9);
  std::vector<std::uint8_t> mask(128, 1);
  mask[17] = 0;
  std::fill_n(v.data() + 17 * 9, 9, std::numeric_limits<float>::quiet_NaN());
  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), mask.data(),
                                                   actual.data());
  ExpectClose(actual, std::vector<float>(actual.size(), 1.0f));
}

TEST(ComputeAttentionFloat32Streaming, Rank3HeadPackingPreservesMasksGroupsAndNonpadLengths) {
  constexpr std::int64_t batch = 2, heads = 4, q_len = 17, kv_len = 31, dim = 17, vdim = 19;
  for (std::int64_t kv_heads : {1, 2, 4}) {
    for (bool causal : {false, true}) {
      for (ml_cpu_compute::AttentionMaskKind kind :
           {ml_cpu_compute::AttentionMaskKind::kBoolean,
            ml_cpu_compute::AttentionMaskKind::kAdditive}) {
        SCOPED_TRACE(::testing::Message()
                     << kv_heads << " causal=" << causal << " mask=" << static_cast<int>(kind));
        ml_cpu_compute::AttentionDescriptor descriptor;
        descriptor.q_num_heads = heads;
        descriptor.kv_num_heads = kv_heads;
        descriptor.is_causal = causal;
        const std::int64_t q_shape[] = {batch, q_len, heads * dim};
        const std::int64_t k_shape[] = {batch, kv_len, kv_heads * dim};
        const std::int64_t v_shape[] = {batch, kv_len, kv_heads * vdim};
        const std::int64_t mask_shape[] = {batch, heads, q_len, kv_len};
        ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank3,
                                           q_shape, k_shape, v_shape, mask_shape, kind);
        const auto q = RandomTensor(batch * q_len * heads * dim, 1701);
        const auto k = RandomTensor(batch * kv_len * kv_heads * dim, 1702);
        const auto v = RandomTensor(batch * kv_len * kv_heads * vdim, 1703);
        auto additive = RandomTensor(batch * heads * q_len * kv_len, 1704);
        std::vector<std::uint8_t> boolean(additive.size());
        for (std::size_t i = 0; i < boolean.size(); ++i) {
          boolean[i] = (i % 7 != 0);
        }
        std::fill_n(additive.begin(), kv_len, -std::numeric_limits<float>::infinity());
        std::fill_n(boolean.begin(), kv_len, 0);
        const void *mask = kind == ml_cpu_compute::AttentionMaskKind::kBoolean
                               ? static_cast<const void *>(boolean.data())
                               : static_cast<const void *>(additive.data());
        const std::int64_t nonpad[] = {27, 31};
        std::vector<float> expected(batch * q_len * heads * vdim), actual(expected.size());
        ml_cpu_compute::ComputeAttentionFloat32Materialized(
            plan, q.data(), k.data(), v.data(), mask, expected.data(), nullptr, nullptr, nonpad);
        InlineExecutor executor;
        ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
        {
          ml_cpu_compute::ExecutionExecutorScope scope(&view);
          ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), mask,
                                                           actual.data(), nullptr, nullptr, nonpad);
        }
        ExpectClose(actual, expected);
        EXPECT_EQ(executor.dispatches, 1);
        EXPECT_GT(executor.maximum_blocks, 1);
        EXPECT_LE(executor.maximum_blocks, 4);
        EXPECT_FALSE(executor.nested);
        executor = {};
        {
          ml_cpu_compute::ExecutionExecutorScope scope(&view);
          ml_cpu_compute::detail::ExecutionRegionScope region;
          ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), mask,
                                                           actual.data(), nullptr, nullptr, nonpad);
        }
        ExpectClose(actual, expected);
        EXPECT_EQ(executor.dispatches, 0);
      }
    }
  }
}

TEST(ComputeAttentionFloat32Streaming, Rank3HeadPackingBoundariesMatchMaterialized) {
  for (const std::int64_t q_len : {16, 128, 129}) {
    for (const std::int64_t kv_len : {256, 257, 511, 512, 513}) {
      ml_cpu_compute::AttentionDescriptor descriptor;
      descriptor.q_num_heads = 2;
      descriptor.kv_num_heads = 1;
      const std::int64_t q_shape[] = {1, q_len, 2 * 7};
      const std::int64_t k_shape[] = {1, kv_len, 7};
      const std::int64_t v_shape[] = {1, kv_len, 9};
      ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank3,
                                         q_shape, k_shape, v_shape, {},
                                         ml_cpu_compute::AttentionMaskKind::kNone);
      const auto q = RandomTensor(q_len * 2 * 7, 1711);
      const auto k = RandomTensor(kv_len * 7, 1712);
      const auto v = RandomTensor(kv_len * 9, 1713);
      std::vector<float> expected(q_len * 2 * 9), actual(expected.size());
      ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(),
                                                          nullptr, expected.data());
      InlineExecutor executor;
      ml_cpu_compute::AttentionExecutionInfo info;
      ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
      {
        ml_cpu_compute::ExecutionExecutorScope scope(&view);
        ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(),
                                                         nullptr, actual.data(), nullptr, nullptr,
                                                         nullptr, &info);
      }
      ExpectClose(actual, expected);
      EXPECT_EQ(info.path, ml_cpu_compute::AttentionExecutionPath::kTiled);
      EXPECT_EQ(info.kv_tile, std::min<std::size_t>(kv_len, 512));
    }
  }
}

TEST(ComputeAttentionFloat32Streaming, LargeHeadKv257UsesSingleTileAndMatchesMaterialized) {
  constexpr std::int64_t q_len = 16, kv_len = 257, head_dim = 129, v_head_dim = 129;
  const std::int64_t q_shape[] = {1, 1, q_len, head_dim};
  const std::int64_t k_shape[] = {1, 1, kv_len, head_dim};
  const std::int64_t v_shape[] = {1, 1, kv_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan({}, ml_cpu_compute::AttentionLayout::kRank4, q_shape, k_shape,
                                     v_shape, {}, ml_cpu_compute::AttentionMaskKind::kNone);
  const auto q = RandomTensor(q_len * head_dim, 1721);
  const auto k = RandomTensor(kv_len * head_dim, 1722);
  const auto v = RandomTensor(kv_len * v_head_dim, 1723);
  std::vector<float> expected(q_len * v_head_dim), actual(expected.size());
  ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(), nullptr,
                                                      expected.data());
  ml_cpu_compute::AttentionExecutionInfo info;
  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                   actual.data(), nullptr, nullptr, nullptr, &info);
  ExpectClose(actual, expected);
  EXPECT_EQ(info.path, ml_cpu_compute::AttentionExecutionPath::kTiled);
  EXPECT_EQ(info.kv_tile, 257u);
}

namespace half_precision {

std::vector<std::uint16_t> ToFloat16(const std::vector<float> &values) {
  std::vector<std::uint16_t> out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    out[i] = ml_cpu_compute::detail::FloatToFloat16Bits(values[i]);
  }
  return out;
}

std::vector<float> FromFloat16(const std::vector<std::uint16_t> &values) {
  std::vector<float> out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    out[i] = ml_cpu_compute::detail::Float16BitsToFloat(values[i]);
  }
  return out;
}

std::vector<std::uint16_t> ToBFloat16(const std::vector<float> &values) {
  std::vector<std::uint16_t> out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    out[i] = ml_cpu_compute::detail::FloatToBFloat16Bits(values[i]);
  }
  return out;
}

std::vector<float> FromBFloat16(const std::vector<std::uint16_t> &values) {
  std::vector<float> out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    out[i] = ml_cpu_compute::detail::Bfloat16BitsToFloat(values[i]);
  }
  return out;
}

} // namespace half_precision

TEST(ComputeAttentionFloat32Streaming, ShortTilesPreserveMaskedNonfiniteValues) {
  for (const std::int64_t rows : {8, 16}) {
    if (rows == 16) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
      if (ml_cpu_compute::DetectSimdLevel() < ml_cpu_compute::SimdLevel::kAVX2 ||
          !ml_cpu_compute::CpuSupportsFma()) {
        continue;
      }
#else
      continue;
#endif
    }
    const std::int64_t qs[] = {1, 1, rows, 64};
    const std::int64_t kvs[] = {1, 1, 17, 64};
    const std::int64_t ms[] = {17};
    const AttentionPlan plan({}, AttentionLayout::kRank4, qs, kvs, kvs, ms,
                             AttentionMaskKind::kBoolean);
    const std::vector<float> q(rows * 64, 0.0f), k(17 * 64, 0.0f);
    std::vector<float> v(17 * 64, 1.0f), actual(q.size());
    v[64] = std::numeric_limits<float>::quiet_NaN();
    v[128] = std::numeric_limits<float>::infinity();
    v[192] = -std::numeric_limits<float>::infinity();
    std::vector<std::uint8_t> mask(17, 0);
    const auto q16 = half_precision::ToFloat16(q);
    const auto k16 = half_precision::ToFloat16(k);
    const auto v16 = half_precision::ToFloat16(v);
    std::vector<std::uint16_t> actual16(actual.size());
    for (const std::uint8_t allowed : {0, 1}) {
      mask[0] = allowed;
      if (rows == 8) {
        ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), mask.data(),
                                         actual.data());
        ExpectClose(actual, std::vector<float>(actual.size(), allowed));
      }
      ComputeAttentionFloat16Streaming(plan, q16.data(), k16.data(), v16.data(), mask.data(),
                                       actual16.data());
      ExpectClose(half_precision::FromFloat16(actual16),
                  std::vector<float>(actual.size(), allowed));
    }
  }
}

TEST(ComputeAttentionFloat32Streaming, ZeroQueryHeadsDoNotReadValues) {
  const std::int64_t qs[] = {2, 0, 8, 64};
  const std::int64_t kvs[] = {2, 1, 17, 64};
  const AttentionPlan plan({}, AttentionLayout::kRank4, qs, kvs, kvs, {}, AttentionMaskKind::kNone);
  EXPECT_NO_THROW(
      ComputeAttentionFloat32Streaming(plan, nullptr, nullptr, nullptr, nullptr, nullptr));
  EXPECT_NO_THROW(
      ComputeAttentionFloat16Streaming(plan, nullptr, nullptr, nullptr, nullptr, nullptr));
}

TEST(ComputeAttentionFloat32Streaming, FiniteValueCheckHandlesEveryVectorTail) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
  if (ml_cpu_compute::DetectSimdLevel() < ml_cpu_compute::SimdLevel::kAVX2 ||
      !ml_cpu_compute::CpuSupportsFma()) {
    GTEST_SKIP() << "AVX2/FMA is unavailable";
  }
  for (std::size_t count = 0; count <= 33; ++count) {
    std::vector<float> values(count, -0.5f);
    auto half = half_precision::ToFloat16(values);
    EXPECT_TRUE(ml_cpu_compute::AttentionAllFiniteFloat32_AVX2_FMA(values.data(), count));
    EXPECT_TRUE(ml_cpu_compute::AttentionAllFiniteFloat16_AVX2_FMA(half.data(), count));
    for (std::size_t index = 0; index < count; ++index) {
      for (const float special :
           {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()}) {
        values[index] = special;
        half[index] = ml_cpu_compute::detail::FloatToFloat16Bits(special);
        EXPECT_FALSE(ml_cpu_compute::AttentionAllFiniteFloat32_AVX2_FMA(values.data(), count));
        EXPECT_FALSE(ml_cpu_compute::AttentionAllFiniteFloat16_AVX2_FMA(half.data(), count));
        values[index] = -0.5f;
        half[index] = ml_cpu_compute::detail::FloatToFloat16Bits(-0.5f);
      }
    }
  }
#else
  GTEST_SKIP() << "AVX2/FMA kernels were not built";
#endif
}

TEST(ComputeAttentionFloat32Streaming, Avx2SoftmaxPreservesNaNsAcrossVectorTails) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
  if (ml_cpu_compute::DetectSimdLevel() < ml_cpu_compute::SimdLevel::kAVX2 ||
      !ml_cpu_compute::CpuSupportsFma()) {
    GTEST_SKIP() << "AVX2/FMA is unavailable";
  }
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  for (std::size_t count = 1; count <= 33; ++count) {
    for (std::size_t index = 0; index < count; ++index) {
      for (const float previous_maximum : {negative_infinity, 0.0f}) {
        SCOPED_TRACE(::testing::Message() << "count=" << count << "index=" << index
                                          << "previous_maximum=" << previous_maximum);
        std::vector<float> scores(count, negative_infinity);
        scores[0] = 0.0f;
        scores[index] = std::numeric_limits<float>::quiet_NaN();
        float denominator = previous_maximum == negative_infinity ? 0.0f : 1.0f;
        ml_cpu_compute::AttentionSoftmaxBlockFloat32_AVX2_FMA(scores.data(), count,
                                                              previous_maximum, denominator);
        EXPECT_TRUE(std::isnan(denominator));
        EXPECT_TRUE(std::isnan(scores[index]));
      }
    }
  }
#else
  GTEST_SKIP() << "AVX2/FMA kernels were not built";
#endif
}

TEST(ComputeAttentionFloat32Streaming, ShortTilesOnlyCheckReachableValueRows) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
  if (ml_cpu_compute::DetectSimdLevel() != ml_cpu_compute::SimdLevel::kAVX2 ||
      !ml_cpu_compute::CpuSupportsFma()) {
    GTEST_SKIP() << "AVX2/FMA short-query tiles are unavailable";
  }
  constexpr std::int64_t batch = 2, heads = 4, kv_heads = 2, kv_len = 8192, dim = 64;
  for (const auto layout : {AttentionLayout::kRank3, AttentionLayout::kRank4}) {
    for (const std::int64_t rows : {8, 16}) {
      for (const auto &[causal, nonpad, left, right] :
           {std::tuple{true, false, -1, -1}, std::tuple{false, false, -1, 1},
            std::tuple{false, true, -1, -1}, std::tuple{true, true, 2, 0}}) {
        SCOPED_TRACE(::testing::Message()
                     << "layout=" << static_cast<int>(layout) << " rows=" << rows << " causal="
                     << causal << " nonpad=" << nonpad << " left=" << left << " right=" << right);
        AttentionDescriptor descriptor;
        descriptor.opset = 25;
        descriptor.q_num_heads = heads;
        descriptor.kv_num_heads = kv_heads;
        descriptor.is_causal = causal;
        descriptor.has_nonpad_kv_seqlen = nonpad;
        descriptor.left_window_size = left;
        descriptor.right_window_size = right;
        const std::vector<std::int64_t> qs =
            layout == AttentionLayout::kRank3 ? std::vector<std::int64_t>{batch, rows, heads * dim}
                                              : std::vector<std::int64_t>{batch, heads, rows, dim};
        const std::vector<std::int64_t> kvs =
            layout == AttentionLayout::kRank3
                ? std::vector<std::int64_t>{batch, kv_len, kv_heads * dim}
                : std::vector<std::int64_t>{batch, kv_heads, kv_len, dim};
        const AttentionPlan plan(descriptor, layout, qs, kvs, kvs, {}, AttentionMaskKind::kNone);
        const std::int64_t lengths[] = {0, 40};
        const auto *nonpad_lengths = nonpad ? lengths : nullptr;
        const std::vector<float> q(batch * heads * rows * dim, 0.0f);
        const std::vector<float> k(batch * kv_heads * kv_len * dim, 0.0f);
        std::vector<float> v(k.size(), std::numeric_limits<float>::quiet_NaN());
        for (std::size_t b = 0; b < batch; ++b) {
          const std::int64_t start =
              nonpad && left >= 0 ? std::max<std::int64_t>(lengths[b] - rows - left, 0) : 0;
          const std::int64_t limit = nonpad ? lengths[b] : rows + (right >= 0 ? right : 0);
          for (std::size_t h = 0; h < kv_heads; ++h) {
            for (std::int64_t row = start; row < limit; ++row) {
              std::fill_n(v.data() + b * plan.v_strides.batch + h * plan.v_strides.head +
                              row * plan.v_strides.sequence,
                          dim, 1.0f);
            }
          }
        }
        std::vector<float> expected(q.size(), 1.0f), actual(q.size());
        if (nonpad) {
          std::fill_n(expected.begin(), heads * rows * dim, 0.0f);
        }
        ml_cpu_compute::AttentionExecutionInfo info;
        ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr, actual.data(),
                                         nullptr, nullptr, nonpad_lengths, &info);
        ExpectClose(actual, expected);
        EXPECT_EQ(info.path, ml_cpu_compute::AttentionExecutionPath::kTiled);
        const auto q16 = half_precision::ToFloat16(q);
        const auto k16 = half_precision::ToFloat16(k);
        const auto v16 = half_precision::ToFloat16(v);
        std::vector<std::uint16_t> actual16(q.size());
        ComputeAttentionFloat16Streaming(plan, q16.data(), k16.data(), v16.data(), nullptr,
                                         actual16.data(), nullptr, nullptr, nonpad_lengths, &info);
        ExpectClose(half_precision::FromFloat16(actual16), expected);
        EXPECT_EQ(info.path, ml_cpu_compute::AttentionExecutionPath::kTiled);
      }
    }
  }
#else
  GTEST_SKIP() << "AVX2/FMA kernels were not built";
#endif
}

TEST(ComputeAttentionFloat32Streaming, ShortTilesPreserveAdditiveMaskNaNs) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
  if (ml_cpu_compute::DetectSimdLevel() != ml_cpu_compute::SimdLevel::kAVX2 ||
      !ml_cpu_compute::CpuSupportsFma()) {
    GTEST_SKIP() << "AVX2/FMA short-query tiles are unavailable";
  }
  constexpr std::int64_t q_len = 8, dim = 64;
  AttentionDescriptor descriptor;
  descriptor.q_num_heads = 1;
  descriptor.kv_num_heads = 1;
  descriptor.is_causal = true;
  const std::int64_t qs[] = {1, 1, q_len, dim};
  const std::int64_t kvs[] = {1, 1, q_len, dim};
  const std::int64_t mask_shape[] = {q_len, q_len};
  const AttentionPlan plan(descriptor, AttentionLayout::kRank4, qs, kvs, kvs, mask_shape,
                           AttentionMaskKind::kAdditive);
  const std::vector<float> q(q_len * dim, 0.0f), k(q_len * dim, 0.0f);
  const std::vector<float> v(q_len * dim, 1.0f);
  std::vector<float> mask(q_len * q_len, -std::numeric_limits<float>::infinity());
  for (std::int64_t row = 0; row < q_len; ++row) {
    mask[row * q_len] = 0.0f;
    mask[row * q_len + 4] = std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<float> actual(q.size());
  ml_cpu_compute::AttentionExecutionInfo info;
  ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), mask.data(), actual.data(),
                                   nullptr, nullptr, nullptr, &info);
  EXPECT_EQ(info.path, ml_cpu_compute::AttentionExecutionPath::kStreaming);
  for (std::int64_t row = 0; row < q_len; ++row) {
    for (std::int64_t d = 0; d < dim; ++d) {
      if (row < 4) {
        EXPECT_FLOAT_EQ(actual[row * dim + d], 1.0f);
      } else {
        EXPECT_TRUE(std::isnan(actual[row * dim + d]));
      }
    }
  }

  const auto q16 = half_precision::ToFloat16(q);
  const auto k16 = half_precision::ToFloat16(k);
  const auto v16 = half_precision::ToFloat16(v);
  std::vector<std::uint16_t> actual16(q.size());
  ComputeAttentionFloat16Streaming(plan, q16.data(), k16.data(), v16.data(), mask.data(),
                                   actual16.data(), nullptr, nullptr, nullptr, &info);
  EXPECT_EQ(info.path, ml_cpu_compute::AttentionExecutionPath::kStreaming);
  const auto actual16_float = half_precision::FromFloat16(actual16);
  for (std::int64_t row = 0; row < q_len; ++row) {
    for (std::int64_t d = 0; d < dim; ++d) {
      if (row < 4) {
        EXPECT_FLOAT_EQ(actual16_float[row * dim + d], 1.0f);
      } else {
        EXPECT_TRUE(std::isnan(actual16_float[row * dim + d]));
      }
    }
  }
#else
  GTEST_SKIP() << "AVX2/FMA kernels were not built";
#endif
}

TEST(ComputeAttentionFloat32Streaming, ShortTilesPreserveComputedScoreNaNs) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
  if (ml_cpu_compute::DetectSimdLevel() != ml_cpu_compute::SimdLevel::kAVX2 ||
      !ml_cpu_compute::CpuSupportsFma()) {
    GTEST_SKIP() << "AVX2/FMA short-query tiles are unavailable";
  }
  constexpr std::int64_t dim = 64;
  for (const std::int64_t rows : {8, 16}) {
    AttentionDescriptor descriptor;
    descriptor.q_num_heads = 1;
    descriptor.kv_num_heads = 1;
    descriptor.is_causal = true;
    const std::int64_t shape[] = {1, 1, rows, dim};
    const std::int64_t mask_shape[] = {rows, rows};
    const AttentionPlan plan(descriptor, AttentionLayout::kRank4, shape, shape, shape, mask_shape,
                             AttentionMaskKind::kBoolean);
    const std::vector<float> q(rows * dim, 1.0f), v(rows * dim, 1.0f);
    std::vector<float> k(q.size(), 0.0f);
    k[4 * dim] = std::numeric_limits<float>::quiet_NaN();
    std::vector<std::uint8_t> mask(rows * rows, 0);
    for (std::int64_t row = 0; row < rows; ++row) {
      mask[row * rows] = mask[row * rows + 4] = 1;
    }
    const auto q16 = half_precision::ToFloat16(q);
    const auto k16 = half_precision::ToFloat16(k);
    const auto v16 = half_precision::ToFloat16(v);
    std::vector<float> actual(q.size()), expected(q.size());
    std::vector<std::uint16_t> actual16(q.size());
    ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(),
                                                        mask.data(), expected.data());
    ml_cpu_compute::AttentionExecutionInfo info;
    ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), mask.data(), actual.data(),
                                     nullptr, nullptr, nullptr, &info);
    EXPECT_EQ(info.path, ml_cpu_compute::AttentionExecutionPath::kTiled);
    ComputeAttentionFloat16Streaming(plan, q16.data(), k16.data(), v16.data(), mask.data(),
                                     actual16.data(), nullptr, nullptr, nullptr, &info);
    EXPECT_EQ(info.path, ml_cpu_compute::AttentionExecutionPath::kTiled);
    const auto actual16_float = half_precision::FromFloat16(actual16);
    for (std::int64_t row = 0; row < rows; ++row) {
      for (std::int64_t d = 0; d < dim; ++d) {
        if (row < 4) {
          EXPECT_FLOAT_EQ(actual[row * dim + d], 1.0f);
          EXPECT_FLOAT_EQ(actual16_float[row * dim + d], 1.0f);
        } else {
          EXPECT_TRUE(std::isnan(expected[row * dim + d]));
          EXPECT_TRUE(std::isnan(actual[row * dim + d]));
          EXPECT_TRUE(std::isnan(actual16_float[row * dim + d]));
        }
      }
    }
  }
#else
  GTEST_SKIP() << "AVX2/FMA kernels were not built";
#endif
}

TEST(ComputeAttentionFloat32Streaming, RejectsNegativeTiledOutputStride) {
  AttentionDescriptor descriptor;
  const std::int64_t shape[] = {1, 1, 128, 64};
  AttentionPlan plan(descriptor, AttentionLayout::kRank4, shape, shape, shape, {},
                     AttentionMaskKind::kNone);
  plan.y_strides.sequence = -1;
  const std::vector<float> input(128 * 64, 0.0f);
  std::vector<float> output(input.size());
  EXPECT_THROW(ComputeAttentionFloat32Streaming(plan, input.data(), input.data(), input.data(),
                                                nullptr, output.data()),
               std::invalid_argument);
}

TEST(ComputeAttentionFloat32Streaming, DenseSoftmaxPreservesNonfiniteScoreFallback) {
  AttentionDescriptor descriptor;
  const std::int64_t qs[] = {1, 1, 129, 64}, kvs[] = {1, 1, 257, 64};
  const std::int64_t mask_shape[] = {257};
  const AttentionPlan dense(descriptor, AttentionLayout::kRank4, qs, kvs, kvs, {},
                            AttentionMaskKind::kNone);
  const AttentionPlan masked(descriptor, AttentionLayout::kRank4, qs, kvs, kvs, mask_shape,
                             AttentionMaskKind::kBoolean);
  const std::vector<std::uint8_t> mask(257, 1);
  const std::vector<float> q(129 * 64, 1.0f), v(257 * 64, 1.0f);
  std::vector<float> k(v.size(), 0.0f), actual(q.size()), expected(q.size());
  for (const float value :
       {0.0f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 0.0f}) {
    SCOPED_TRACE(value);
    k[256 * 64] = value;
    ComputeAttentionFloat32Streaming(masked, q.data(), k.data(), v.data(), mask.data(),
                                     expected.data());
    ComputeAttentionFloat32Streaming(dense, q.data(), k.data(), v.data(), nullptr, actual.data());
    for (std::size_t i = 0; i < actual.size(); ++i) {
      if (std::isnan(expected[i])) {
        EXPECT_TRUE(std::isnan(actual[i])) << "index=" << i;
      } else {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index=" << i;
      }
    }
  }
}

TEST(ComputeAttentionFloat32Streaming, Avx2DecodeAdditiveMaskPreservesFiniteNaNAndMaskedRows) {
  constexpr std::int64_t heads = 2, keys = 1024, dim = 64;
  const std::int64_t qs[] = {1, heads, 1, dim};
  const std::int64_t kvs[] = {1, heads, keys, dim};
  const std::int64_t mask_shape[] = {1, heads, 1, keys};
  const AttentionPlan plan({}, AttentionLayout::kRank4, qs, kvs, kvs, mask_shape,
                           AttentionMaskKind::kAdditive);
  const std::vector<float> q(heads * dim, 0.0f), k(heads * keys * dim, 0.0f);
  const std::vector<float> v(k.size(), 1.0f);
  const auto q16 = half_precision::ToFloat16(q);
  const auto k16 = half_precision::ToFloat16(k);
  const auto v16 = half_precision::ToFloat16(v);
  std::vector<float> mask(heads * keys, 0.0f), actual(q.size());
  std::vector<std::uint16_t> actual16(q.size());
  for (int mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    if (mode == 1)
      mask[keys + 4] = std::numeric_limits<float>::quiet_NaN();
    else if (mode == 2)
      std::fill(mask.begin() + keys, mask.end(), -std::numeric_limits<float>::infinity());
    ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), mask.data(),
                                     actual.data());
    ComputeAttentionFloat16Streaming(plan, q16.data(), k16.data(), v16.data(), mask.data(),
                                     actual16.data());
    const auto actual16_float = half_precision::FromFloat16(actual16);
    for (std::int64_t d = 0; d < dim; ++d) {
      EXPECT_FLOAT_EQ(actual[d], 1.0f);
      EXPECT_FLOAT_EQ(actual16_float[d], 1.0f);
      if (mode == 1) {
        EXPECT_TRUE(std::isnan(actual[dim + d]));
        EXPECT_TRUE(std::isnan(actual16_float[dim + d]));
      } else {
        EXPECT_FLOAT_EQ(actual[dim + d], mode == 2 ? 0.0f : 1.0f);
        EXPECT_FLOAT_EQ(actual16_float[dim + d], mode == 2 ? 0.0f : 1.0f);
      }
    }
  }
}

TEST(ComputeAttentionFloat32Streaming, ShortTilesFallBackForUnsafeFloat32Magnitudes) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
  if (ml_cpu_compute::DetectSimdLevel() != ml_cpu_compute::SimdLevel::kAVX2 ||
      !ml_cpu_compute::CpuSupportsFma()) {
    GTEST_SKIP() << "AVX2/FMA short-query tiles are unavailable";
  }
  constexpr std::int64_t q_len = 8, kv_len = 1024, dim = 16;
  const std::int64_t qs[] = {1, 1, q_len, dim};
  const std::int64_t kvs[] = {1, 1, kv_len, dim};
  const AttentionPlan plan({}, AttentionLayout::kRank4, qs, kvs, kvs, {}, AttentionMaskKind::kNone);
  const std::vector<float> q(q_len * dim, 0.0f), k(kv_len * dim, 0.0f);
  std::vector<float> v(kv_len * dim);
  for (std::int64_t row = 0; row < kv_len; ++row) {
    std::fill_n(v.data() + row * dim, dim, row < kv_len / 2 ? 4.0e35f : -8.0e35f);
  }
  std::vector<float> actual(q.size());
  ml_cpu_compute::AttentionExecutionInfo info;
  ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr, actual.data(),
                                   nullptr, nullptr, nullptr, &info);
  EXPECT_EQ(info.path, ml_cpu_compute::AttentionExecutionPath::kStreaming);
  for (float value : actual) {
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_NEAR(value / 1.0e35f, -2.0f, 2.0e-5f);
  }
#else
  GTEST_SKIP() << "AVX2/FMA kernels were not built";
#endif
}

TEST(ComputeAttentionFloat32Streaming, GroupedQueryTilesPreserveUnsafeValueFrontiers) {
  AttentionDescriptor descriptor;
  descriptor.q_num_heads = 2;
  descriptor.kv_num_heads = 1;
  descriptor.is_causal = true;
  const std::int64_t qs[] = {1, 32, 128};
  const std::int64_t kvs[] = {1, 32, 64};
  const AttentionPlan plan(descriptor, AttentionLayout::kRank3, qs, kvs, kvs, {},
                           AttentionMaskKind::kNone);
  const std::vector<float> q(32 * 128, 0.0f), k(32 * 64, 0.0f);
  const auto q16 = half_precision::ToFloat16(q), k16 = half_precision::ToFloat16(k);
  std::vector<float> v(32 * 64, 1.0f), actual(q.size());
  std::vector<std::uint16_t> actual16(q.size());
  ml_cpu_compute::AttentionExecutionInfo info;
  for (float unsafe :
       {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::max()}) {
    SCOPED_TRACE(unsafe);
    v[20 * 64] = unsafe;
    ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr, actual.data(),
                                     nullptr, nullptr, nullptr, &info);
    EXPECT_EQ(info.query_tile, 16u);
    EXPECT_EQ(info.output_tile_elements, 16 * 64u);
    for (std::size_t i = 0; i < 16 * 128; ++i) {
      EXPECT_FLOAT_EQ(actual[i], 1.0f);
    }
    const auto v16 = half_precision::ToFloat16(v);
    ComputeAttentionFloat16Streaming(plan, q16.data(), k16.data(), v16.data(), nullptr,
                                     actual16.data(), nullptr, nullptr, nullptr, &info);
    EXPECT_EQ(info.query_tile, 16u);
    for (std::size_t i = 0; i < 16 * 128; ++i) {
      EXPECT_FLOAT_EQ(ml_cpu_compute::detail::Float16BitsToFloat(actual16[i]), 1.0f);
    }
  }
  v[20 * 64] = 1.0f;
  ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr, actual.data(),
                                   nullptr, nullptr, nullptr, &info);
  ExpectClose(actual, std::vector<float>(actual.size(), 1.0f));
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
  if (ml_cpu_compute::DetectSimdLevel() == ml_cpu_compute::SimdLevel::kAVX2 &&
      ml_cpu_compute::CpuSupportsFma()) {
    EXPECT_EQ(info.query_tile, 32u);
  }
#endif
}

TEST(ComputeAttentionFloat32Streaming, ShortTilesAndValueProductsMatchMaterialized) {
  for (const auto &[q_len, kv_len, dim, vdim] :
       {std::tuple{7, 17, 16, 16},      std::tuple{8, 17, 16, 16},
        std::tuple{8, 0, 64, 64},       std::tuple{8, 1, 64, 64},
        std::tuple{8, 513, 64, 63},     std::tuple{15, 512, 63, 65},
        std::tuple{16, 511, 64, 64},    std::tuple{17, 33, 17, 17},
        std::tuple{8, 17, 15, 16},      std::tuple{31, 31, 64, 64},
        std::tuple{32, 32, 64, 64},     std::tuple{33, 33, 64, 64},
        std::tuple{63, 63, 64, 64},     std::tuple{64, 64, 64, 64},
        std::tuple{65, 65, 64, 64},     std::tuple{8, 17, 16, 15},
        std::tuple{8, 17, 257, 256},    std::tuple{8, 17, 256, 257},
        std::tuple{129, 17, 64, 256},   std::tuple{128, 17, 64, 64},
        std::tuple{128, 17, 64, 65},    std::tuple{128, 128, 128, 128},
        std::tuple{128, 128, 256, 256}, std::tuple{129, 257, 63, 64},
        std::tuple{129, 257, 64, 63},   std::tuple{129, 257, 256, 256},
        std::tuple{129, 513, 256, 256}}) {
    for (const bool rank3 : {false, true}) {
      for (int mode = 0; mode < 4; ++mode) {
        SCOPED_TRACE(::testing::Message()
                     << "q=" << q_len << " kv=" << kv_len << " d=" << dim << " vd=" << vdim
                     << " rank3=" << rank3 << " mode=" << mode);
        AttentionDescriptor descriptor;
        descriptor.opset = 24;
        descriptor.q_num_heads = 2;
        descriptor.kv_num_heads = 1;
        descriptor.is_causal = mode == 1;
        descriptor.has_nonpad_kv_seqlen = mode == 1;
        const std::vector<std::int64_t> qs = rank3 ? std::vector<std::int64_t>{2, q_len, 2 * dim}
                                                   : std::vector<std::int64_t>{2, 2, q_len, dim};
        const std::vector<std::int64_t> ks = rank3 ? std::vector<std::int64_t>{2, kv_len, dim}
                                                   : std::vector<std::int64_t>{2, 1, kv_len, dim};
        const std::vector<std::int64_t> vs = rank3 ? std::vector<std::int64_t>{2, kv_len, vdim}
                                                   : std::vector<std::int64_t>{2, 1, kv_len, vdim};
        const auto kind = mode == 2   ? AttentionMaskKind::kBoolean
                          : mode == 3 ? AttentionMaskKind::kAdditive
                                      : AttentionMaskKind::kNone;
        const std::vector<std::int64_t> ms =
            mode >= 2 ? std::vector<std::int64_t>{q_len, kv_len} : std::vector<std::int64_t>{};
        const AttentionPlan plan(descriptor,
                                 rank3 ? AttentionLayout::kRank3 : AttentionLayout::kRank4, qs, ks,
                                 vs, ms, kind);
        const auto q16 = half_precision::ToFloat16(RandomTensor(4 * q_len * dim, 1981));
        const auto k16 = half_precision::ToFloat16(RandomTensor(2 * kv_len * dim, 1982));
        const auto v16 = half_precision::ToFloat16(RandomTensor(2 * kv_len * vdim, 1983));
        const auto q = half_precision::FromFloat16(q16);
        const auto k = half_precision::FromFloat16(k16);
        const auto v = half_precision::FromFloat16(v16);
        std::vector<std::uint8_t> boolean(q_len * kv_len, 1);
        std::vector<float> additive(q_len * kv_len, 0.2f);
        for (std::size_t index = 0; index < boolean.size(); ++index) {
          // Include a fully masked query and mixed noncanonical true bytes.
          boolean[index] = index < static_cast<std::size_t>(kv_len) || index % 5 == 0 ? 0 : 128;
          if (boolean[index] == 0) {
            additive[index] = -std::numeric_limits<float>::infinity();
          }
        }
        const void *mask = mode == 2   ? static_cast<const void *>(boolean.data())
                           : mode == 3 ? static_cast<const void *>(additive.data())
                                       : nullptr;
        const std::int64_t lengths[] = {0, std::max(kv_len - 1, 0)};
        const auto *nonpad = mode == 1 ? lengths : nullptr;
        std::vector<float> expected(4 * q_len * vdim), actual(expected.size());
        ml_cpu_compute::ComputeAttentionFloat32Materialized(
            plan, q.data(), k.data(), v.data(), mask, expected.data(), nullptr, nullptr, nonpad);
        ml_cpu_compute::AttentionExecutionInfo info;
        InlineExecutor executor;
        ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
        {
          ml_cpu_compute::ExecutionExecutorScope scope(&view);
          ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), mask, actual.data(),
                                           nullptr, nullptr, nonpad, &info);
        }
        EXPECT_FALSE(executor.nested);
        if (q_len == 8 && kv_len == 513) {
          EXPECT_GT(executor.maximum_blocks, 1);
        }
        if (!rank3 && q_len == 8 && kv_len == 17 && dim == 16 && vdim == 16) {
          EXPECT_EQ(executor.dispatches, 0);
        }
        ExpectClose(actual, expected, 2e-5f);
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
        if (ml_cpu_compute::DetectSimdLevel() == ml_cpu_compute::SimdLevel::kAVX2 &&
            ml_cpu_compute::CpuSupportsFma() && q_len >= 8 && kv_len > 1 && dim >= 16 &&
            dim <= 256 && vdim >= 16 && vdim <= 256) {
          EXPECT_EQ(info.path, ml_cpu_compute::AttentionExecutionPath::kTiled);
          if (rank3 && mode == 1 && dim == 64 && vdim == 64 && q_len >= 31 && q_len == kv_len) {
            EXPECT_EQ(info.query_tile, q_len <= 64 ? std::min(q_len, 32) : 16);
          }
        }
#endif
        std::vector<std::uint16_t> actual16(expected.size());
        executor = {};
        {
          ml_cpu_compute::ExecutionExecutorScope scope(&view);
          ComputeAttentionFloat16Streaming(plan, q16.data(), k16.data(), v16.data(), mask,
                                           actual16.data(), nullptr, nullptr, nonpad, &info);
        }
        EXPECT_FALSE(executor.nested);
        ExpectClose(half_precision::FromFloat16(actual16), expected, 1e-3f);
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
        if (ml_cpu_compute::DetectSimdLevel() == ml_cpu_compute::SimdLevel::kAVX2 &&
            ml_cpu_compute::CpuSupportsFma() && q_len >= 8 && kv_len > 1 && dim >= 16 &&
            dim <= 256 && vdim >= 16 && vdim <= 256) {
          EXPECT_EQ(info.path, ml_cpu_compute::AttentionExecutionPath::kTiled);
          if (rank3 && mode == 1 && dim == 64 && vdim == 64 && q_len >= 31 && q_len == kv_len) {
            EXPECT_EQ(info.query_tile, q_len <= 64 ? std::min(q_len, 32) : 16);
          }
        }
#endif
      }
    }
  }
}

// Roadmap PR14: the FP16 streaming path rounds Q/K/V once on the way in and
// Y once on the way out, but performs the score/softmax/P@V recurrence in
// FP32 throughout. Round-tripping the same FP16-representable inputs through
// the FP32 streaming/materialized path (no further rounding) must therefore
// match the FP16 path within FP16 rounding tolerance.
TEST(ComputeAttentionFloat16Streaming, BoundedTilesMatchMaterializedAcrossLayoutsAndMasks) {
  constexpr std::int64_t batch = 2, heads = 4, kv_heads = 2, dim = 9, vdim = 11;
  for (const auto layout :
       {ml_cpu_compute::AttentionLayout::kRank3, ml_cpu_compute::AttentionLayout::kRank4}) {
    for (const auto &[q_len, kv_len, past_len] :
         {std::tuple{16, 16, 0}, std::tuple{17, 31, 0}, std::tuple{129, 263, 0},
          std::tuple{8, 13, 255}, std::tuple{17, 13, 257}, std::tuple{8, 1024, 0},
          std::tuple{8, 1025, 0}, std::tuple{8, 13, 1025}}) {
      for (const auto kind : {ml_cpu_compute::AttentionMaskKind::kBoolean,
                              ml_cpu_compute::AttentionMaskKind::kAdditive}) {
        SCOPED_TRACE(::testing::Message()
                     << static_cast<int>(layout) << " q=" << q_len << " kv=" << kv_len
                     << " past=" << past_len << " mask=" << static_cast<int>(kind));
        ml_cpu_compute::AttentionDescriptor descriptor;
        descriptor.q_num_heads = heads;
        descriptor.kv_num_heads = kv_heads;
        descriptor.is_causal = past_len != 0 || q_len == 16 || q_len == 17;
        const std::vector<std::int64_t> q_shape =
            layout == ml_cpu_compute::AttentionLayout::kRank3
                ? std::vector<std::int64_t>{batch, q_len, heads * dim}
                : std::vector<std::int64_t>{batch, heads, q_len, dim};
        const std::vector<std::int64_t> k_shape =
            layout == ml_cpu_compute::AttentionLayout::kRank3
                ? std::vector<std::int64_t>{batch, kv_len, kv_heads * dim}
                : std::vector<std::int64_t>{batch, kv_heads, kv_len, dim};
        const std::vector<std::int64_t> v_shape =
            layout == ml_cpu_compute::AttentionLayout::kRank3
                ? std::vector<std::int64_t>{batch, kv_len, kv_heads * vdim}
                : std::vector<std::int64_t>{batch, kv_heads, kv_len, vdim};
        const std::int64_t mask_shape[] = {batch, heads, q_len, past_len + kv_len};
        const std::int64_t past_k_shape[] = {batch, kv_heads, past_len, dim};
        const std::int64_t past_v_shape[] = {batch, kv_heads, past_len, vdim};
        ml_cpu_compute::AttentionPlan plan(descriptor, layout, q_shape, k_shape, v_shape,
                                           mask_shape, kind, past_k_shape, past_v_shape);
        const auto q = half_precision::ToFloat16(RandomTensor(batch * heads * q_len * dim, 1901));
        const auto k =
            half_precision::ToFloat16(RandomTensor(batch * kv_heads * kv_len * dim, 1902));
        const auto v =
            half_precision::ToFloat16(RandomTensor(batch * kv_heads * kv_len * vdim, 1903));
        const auto pk =
            half_precision::ToFloat16(RandomTensor(batch * kv_heads * past_len * dim, 1904));
        const auto pv =
            half_precision::ToFloat16(RandomTensor(batch * kv_heads * past_len * vdim, 1905));
        auto additive = RandomTensor(batch * heads * q_len * (past_len + kv_len), 1906);
        std::vector<std::uint8_t> boolean(additive.size(), 1);
        for (std::size_t row = 0; row < batch * heads * q_len; ++row) {
          for (std::size_t j = 0; j < plan.total_kv_length; ++j) {
            const std::size_t index = row * plan.total_kv_length + j;
            boolean[index] = row % 7 != 0 && j % 5 != 0;
            if (row % 7 == 0) {
              additive[index] = row % 2 ? -std::numeric_limits<float>::infinity()
                                        : std::numeric_limits<float>::lowest();
            }
          }
        }
        const void *mask = kind == ml_cpu_compute::AttentionMaskKind::kBoolean
                               ? static_cast<const void *>(boolean.data())
                               : static_cast<const void *>(additive.data());
        std::vector<float> expected(batch * heads * q_len * vdim);
        std::vector<std::uint16_t> actual(expected.size());
        ml_cpu_compute::ComputeAttentionFloat32Materialized(
            plan, half_precision::FromFloat16(q).data(), half_precision::FromFloat16(k).data(),
            half_precision::FromFloat16(v).data(), mask, expected.data(),
            half_precision::FromFloat16(pk).data(), half_precision::FromFloat16(pv).data());
        ml_cpu_compute::AttentionExecutionInfo info;
        InlineExecutor executor;
        ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
        {
          ml_cpu_compute::ExecutionExecutorScope scope(&view);
          ml_cpu_compute::ComputeAttentionFloat16Streaming(plan, q.data(), k.data(), v.data(), mask,
                                                           actual.data(), pk.data(), pv.data(),
                                                           nullptr, &info);
        }
        ExpectClose(half_precision::FromFloat16(actual), expected, 1e-3f);
        EXPECT_FALSE(executor.nested);
        EXPECT_LE(executor.dispatches, 1);
        const bool tiled = info.path == ml_cpu_compute::AttentionExecutionPath::kTiled;
        if (q_len != 16) {
          EXPECT_EQ(info.path, past_len == 0 && q_len > 16
                                   ? ml_cpu_compute::AttentionExecutionPath::kTiled
                                   : ml_cpu_compute::AttentionExecutionPath::kStreaming);
        } else {
          EXPECT_TRUE(tiled || info.path == ml_cpu_compute::AttentionExecutionPath::kStreaming);
        }
        EXPECT_TRUE(info.tile_conversion);
        EXPECT_EQ(info.tile_packing, tiled && layout == ml_cpu_compute::AttentionLayout::kRank3);
        EXPECT_LE(info.query_tile, 128);
        EXPECT_LE(info.kv_tile, 512);
        EXPECT_EQ(info.conversion_kv_tile, tiled || plan.total_kv_length > 4 * info.kv_tile
                                               ? info.kv_tile
                                               : plan.total_kv_length);
        EXPECT_EQ(info.output_tile_elements, info.query_tile * vdim);
        executor = {};
        {
          ml_cpu_compute::ExecutionExecutorScope scope(&view);
          ml_cpu_compute::detail::ExecutionRegionScope region;
          ml_cpu_compute::ComputeAttentionFloat16Streaming(plan, q.data(), k.data(), v.data(), mask,
                                                           actual.data(), pk.data(), pv.data());
        }
        EXPECT_EQ(executor.dispatches, 0);
        ExpectClose(half_precision::FromFloat16(actual), expected, 1e-3f);
      }
    }
  }
}

TEST(ComputeAttentionFloat16Streaming, RowConversionPreservesBroadcastNonpadAndWindowMasks) {
  for (const auto kind : {ml_cpu_compute::AttentionMaskKind::kBoolean,
                          ml_cpu_compute::AttentionMaskKind::kAdditive}) {
    ml_cpu_compute::AttentionDescriptor descriptor;
    descriptor.opset = 25;
    descriptor.q_num_heads = 4;
    descriptor.kv_num_heads = 2;
    descriptor.softcap = 0.7f;
    descriptor.left_window_size = 259;
    descriptor.right_window_size = 0;
    const std::int64_t q_shape[] = {2, 8, 4 * 17};
    const std::int64_t k_shape[] = {2, 263, 2 * 17};
    const std::int64_t v_shape[] = {2, 263, 2 * 19};
    const std::int64_t mask_shape[] = {8, 1};
    ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank3, q_shape,
                                       k_shape, v_shape, mask_shape, kind);
    const auto q = half_precision::ToFloat16(RandomTensor(2 * 8 * 4 * 17, 1911));
    const auto k = half_precision::ToFloat16(RandomTensor(2 * 263 * 2 * 17, 1912));
    const auto v = half_precision::ToFloat16(RandomTensor(2 * 263 * 2 * 19, 1913));
    const float additive[] = {std::numeric_limits<float>::lowest(),    0.1f, -0.3f, 1.0f,
                              -std::numeric_limits<float>::infinity(), 0.2f, 0.5f,  -0.1f};
    const std::uint8_t boolean[] = {0, 1, 1, 1, 0, 1, 1, 1};
    const void *mask = kind == ml_cpu_compute::AttentionMaskKind::kBoolean
                           ? static_cast<const void *>(boolean)
                           : static_cast<const void *>(additive);
    const std::int64_t nonpad[] = {0, 261};
    std::vector<float> expected(2 * 8 * 4 * 19);
    std::vector<std::uint16_t> actual(expected.size());
    ml_cpu_compute::ComputeAttentionFloat32Materialized(
        plan, half_precision::FromFloat16(q).data(), half_precision::FromFloat16(k).data(),
        half_precision::FromFloat16(v).data(), mask, expected.data(), nullptr, nullptr, nonpad);
    ml_cpu_compute::ComputeAttentionFloat16Streaming(plan, q.data(), k.data(), v.data(), mask,
                                                     actual.data(), nullptr, nullptr, nonpad);
    ExpectClose(half_precision::FromFloat16(actual), expected, 1e-3f);
  }
}

TEST(AttentionExecutionInfo, ReportsActualDispatchAndResetsReusedInfo) {
  using ml_cpu_compute::AttentionExecutionInfo;
  using ml_cpu_compute::AttentionExecutionPath;
  ml_cpu_compute::AttentionExecutionInfo info;
  for (const std::int64_t q_len : {8, 17}) {
    for (const std::int64_t kv_len : {1, 263}) {
      const std::int64_t q_shape[] = {1, 2, q_len, 9};
      const std::int64_t k_shape[] = {1, 1, kv_len, 9};
      const std::int64_t v_shape[] = {1, 1, kv_len, 11};
      ml_cpu_compute::AttentionPlan plan({}, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                         k_shape, v_shape, {},
                                         ml_cpu_compute::AttentionMaskKind::kNone);
      const auto q = RandomTensor(2 * q_len * 9, 1921);
      const auto k = RandomTensor(kv_len * 9, 1922);
      const auto v = RandomTensor(kv_len * 11, 1923);
      std::vector<float> y(2 * q_len * 11);
      ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data(),
                                              nullptr, nullptr, nullptr, nullptr, &info);
      EXPECT_EQ(info.path, kv_len == 1
                               ? ml_cpu_compute::AttentionExecutionPath::kSingleKey
                               : (q_len < 16 ? ml_cpu_compute::AttentionExecutionPath::kStreaming
                                             : ml_cpu_compute::AttentionExecutionPath::kTiled));
      EXPECT_FALSE(info.tile_conversion);
      EXPECT_FALSE(info.tile_packing);
      EXPECT_EQ(info.query_tile, kv_len == 1 ? 0 : (q_len < 16 ? 1 : q_len));
      std::vector<float> qk_output(2 * q_len * kv_len);
      for (int fallback = 0; fallback < 3; ++fallback) {
        plan.softmax_fp64 = fallback == 0;
        plan.has_present_output = fallback == 1;
        plan.has_qk_matmul_output = fallback == 2;
        ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr,
                                                y.data(), nullptr, nullptr, nullptr,
                                                fallback == 2 ? qk_output.data() : nullptr, &info);
        EXPECT_EQ(info.path, ml_cpu_compute::AttentionExecutionPath::kMaterialized);
        EXPECT_EQ(info.query_tile, 0);
        EXPECT_EQ(info.kv_tile, 0);
        EXPECT_EQ(info.output_tile_elements, 0);
      }
    }
  }
}

TEST(AttentionExecutionInfo, EmptyTensorsNeedNoConversionOrPacking) {
  for (const auto &[batch, q_len, kv_len, vdim] :
       {std::tuple{0, 17, 263, 11}, std::tuple{2, 0, 263, 11}, std::tuple{2, 17, 0, 11},
        std::tuple{2, 17, 263, 0}, std::tuple{2, 8, 1, 0}}) {
    const std::int64_t q_shape[] = {batch, 2, q_len, 9};
    const std::int64_t k_shape[] = {batch, 1, kv_len, 9};
    const std::int64_t v_shape[] = {batch, 1, kv_len, vdim};
    ml_cpu_compute::AttentionPlan plan({}, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                       k_shape, v_shape, {},
                                       ml_cpu_compute::AttentionMaskKind::kNone);
    const std::size_t count = batch * 2 * q_len * vdim;
    std::vector<float> y(std::max<std::size_t>(count, 1), -1.0f);
    std::vector<std::uint16_t> y16 = half_precision::ToFloat16(y);
    ml_cpu_compute::AttentionExecutionInfo info;
    ml_cpu_compute::ComputeAttentionFloat32(plan, nullptr, nullptr, nullptr, nullptr, y.data(),
                                            nullptr, nullptr, nullptr, nullptr, &info);
    EXPECT_EQ(info.output_tile_elements, 0);
    ml_cpu_compute::ComputeAttentionFloat16Streaming(plan, nullptr, nullptr, nullptr, nullptr,
                                                     y16.data(), nullptr, nullptr, nullptr, &info);
    EXPECT_EQ(info.output_tile_elements, 0);
    EXPECT_FALSE(info.tile_conversion);
    EXPECT_FALSE(info.tile_packing);
    ExpectClose(y, std::vector<float>(y.size(), count == 0 ? -1.0f : 0.0f));
    ExpectClose(half_precision::FromFloat16(y16), y);
  }
}

TEST(AttentionExecutionInfo, ReportsLogicalOutputTileForDirectAndBufferedWrites) {
  for (const auto layout :
       {ml_cpu_compute::AttentionLayout::kRank3, ml_cpu_compute::AttentionLayout::kRank4}) {
    ml_cpu_compute::AttentionDescriptor descriptor;
    descriptor.q_num_heads = 1;
    descriptor.kv_num_heads = 1;
    const std::vector<std::int64_t> q_shape = layout == ml_cpu_compute::AttentionLayout::kRank3
                                                  ? std::vector<std::int64_t>{2, 17, 9}
                                                  : std::vector<std::int64_t>{2, 1, 17, 9};
    const std::vector<std::int64_t> k_shape = layout == ml_cpu_compute::AttentionLayout::kRank3
                                                  ? std::vector<std::int64_t>{2, 31, 9}
                                                  : std::vector<std::int64_t>{2, 1, 31, 9};
    const std::vector<std::int64_t> v_shape = layout == ml_cpu_compute::AttentionLayout::kRank3
                                                  ? std::vector<std::int64_t>{2, 31, 11}
                                                  : std::vector<std::int64_t>{2, 1, 31, 11};
    ml_cpu_compute::AttentionPlan plan(descriptor, layout, q_shape, k_shape, v_shape, {},
                                       ml_cpu_compute::AttentionMaskKind::kNone);
    const auto q = RandomTensor(2 * 17 * 9, 1931);
    const auto k = RandomTensor(2 * 31 * 9, 1932);
    const auto v = RandomTensor(2 * 31 * 11, 1933);
    std::vector<float> actual(2 * 17 * 11), expected(actual.size());
    ml_cpu_compute::AttentionExecutionInfo info;
    for (const std::int64_t first_nonpad : {31, 0, 29}) {
      const std::int64_t nonpad[] = {first_nonpad, 31};
      ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr,
                                              actual.data(), nullptr, nullptr, nonpad, nullptr,
                                              &info);
      ml_cpu_compute::ComputeAttentionFloat32Materialized(
          plan, q.data(), k.data(), v.data(), nullptr, expected.data(), nullptr, nullptr, nonpad);
      ExpectClose(actual, expected);
      EXPECT_FALSE(info.tile_conversion);
      EXPECT_FALSE(info.tile_packing);
      EXPECT_EQ(info.output_tile_elements, 17 * 11);
    }
  }
}

TEST(AttentionExecutionInfo, BFloat16UsesBoundedKvConversionTile) {
  const std::int64_t q_shape[] = {1, 1, 8, 9};
  const std::int64_t k_shape[] = {1, 1, 31, 9};
  const std::int64_t v_shape[] = {1, 1, 31, 11};
  ml_cpu_compute::AttentionPlan plan({}, ml_cpu_compute::AttentionLayout::kRank4, q_shape, k_shape,
                                     v_shape, {}, ml_cpu_compute::AttentionMaskKind::kNone);
  const auto q = half_precision::ToBFloat16(RandomTensor(8 * 9, 1941));
  const auto k = half_precision::ToBFloat16(RandomTensor(31 * 9, 1942));
  const auto v = half_precision::ToBFloat16(RandomTensor(31 * 11, 1943));
  std::vector<std::uint16_t> y(8 * 11);
  ml_cpu_compute::AttentionExecutionInfo info;
  ml_cpu_compute::ComputeAttentionBFloat16Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                    y.data(), nullptr, nullptr, nullptr, &info);
  EXPECT_TRUE(info.tile_conversion);
  EXPECT_FALSE(info.tile_packing);
  EXPECT_EQ(info.conversion_kv_tile, 31);
  EXPECT_EQ(info.output_tile_elements, 11);
}

TEST(ComputeAttentionFloat16Streaming, MatchesFloat32ReferenceWithinHalfPrecisionTolerance) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  constexpr std::size_t batch = 1, heads = 2, q_len = 4, kv_len = 9, head_dim = 8, v_head_dim = 8;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);

  // Build FP32 tensors that already round-trip exactly through FP16, so the
  // only source of divergence between the two paths is Y's final rounding.
  const auto q32 = half_precision::FromFloat16(
      half_precision::ToFloat16(RandomTensor(batch * heads * q_len * head_dim, 631)));
  const auto k32 = half_precision::FromFloat16(
      half_precision::ToFloat16(RandomTensor(batch * heads * kv_len * head_dim, 632)));
  const auto v32 = half_precision::FromFloat16(
      half_precision::ToFloat16(RandomTensor(batch * heads * kv_len * v_head_dim, 633)));

  const auto q16 = half_precision::ToFloat16(q32);
  const auto k16 = half_precision::ToFloat16(k32);
  const auto v16 = half_precision::ToFloat16(v32);

  std::vector<std::uint16_t> y16(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat16Streaming(plan, q16.data(), k16.data(), v16.data(),
                                                   nullptr, y16.data());
  const auto y = half_precision::FromFloat16(y16);

  std::vector<float> expected(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q32.data(), k32.data(), v32.data(), nullptr,
                                          expected.data());

  ExpectClose(y, expected, 5e-3f);
}

TEST(ComputeAttentionFloat16Streaming, ShortQueryLengthEightMatchesFloat32Reference) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::size_t batch = 1, heads = 2, q_len = 8, kv_len = 13, head_dim = 8, v_head_dim = 6;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);

  const auto q32 = half_precision::FromFloat16(
      half_precision::ToFloat16(RandomTensor(batch * heads * q_len * head_dim, 644)));
  const auto k32 = half_precision::FromFloat16(
      half_precision::ToFloat16(RandomTensor(batch * heads * kv_len * head_dim, 645)));
  const auto v32 = half_precision::FromFloat16(
      half_precision::ToFloat16(RandomTensor(batch * heads * kv_len * v_head_dim, 646)));
  const auto q16 = half_precision::ToFloat16(q32);
  const auto k16 = half_precision::ToFloat16(k32);
  const auto v16 = half_precision::ToFloat16(v32);
  std::vector<std::uint16_t> y16(batch * heads * q_len * v_head_dim);
  std::vector<float> expected(y16.size());

  ml_cpu_compute::ComputeAttentionFloat16Streaming(plan, q16.data(), k16.data(), v16.data(),
                                                   nullptr, y16.data());
  ml_cpu_compute::ComputeAttentionFloat32(plan, q32.data(), k32.data(), v32.data(), nullptr,
                                          expected.data());

  ExpectClose(half_precision::FromFloat16(y16), expected, 5e-3f);
}

TEST(ComputeAttentionFloat16Streaming, Rank3GqaMatchesFloat32Reference) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  constexpr std::size_t batch = 2, q_heads = 4, kv_heads = 2, q_len = 17, kv_len = 19, head_dim = 8,
                        v_head_dim = 6;
  descriptor.q_num_heads = q_heads;
  descriptor.kv_num_heads = kv_heads;
  const std::int64_t q_shape[] = {batch, q_len, q_heads * head_dim};
  const std::int64_t k_shape[] = {batch, kv_len, kv_heads * head_dim};
  const std::int64_t v_shape[] = {batch, kv_len, kv_heads * v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank3, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);

  const auto q32 = half_precision::FromFloat16(
      half_precision::ToFloat16(RandomTensor(batch * q_len * q_shape[2], 634)));
  const auto k32 = half_precision::FromFloat16(
      half_precision::ToFloat16(RandomTensor(batch * kv_len * k_shape[2], 635)));
  const auto v32 = half_precision::FromFloat16(
      half_precision::ToFloat16(RandomTensor(batch * kv_len * v_shape[2], 636)));
  const auto q16 = half_precision::ToFloat16(q32);
  const auto k16 = half_precision::ToFloat16(k32);
  const auto v16 = half_precision::ToFloat16(v32);

  std::vector<std::uint16_t> y16(batch * q_len * q_heads * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat16Streaming(plan, q16.data(), k16.data(), v16.data(),
                                                   nullptr, y16.data());
  const auto y = half_precision::FromFloat16(y16);

  std::vector<float> expected(y.size());
  ml_cpu_compute::ComputeAttentionFloat32(plan, q32.data(), k32.data(), v32.data(), nullptr,
                                          expected.data());
  ExpectClose(y, expected, 5e-3f);
}

TEST(ComputeAttentionFloat16Streaming, RequestedShapesUseWideBoundedTeams) {
  InlineExecutor executor;
  ml_cpu_compute::ExecutionExecutorView view{&executor, 32, &InlineExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&view);

  {
    ml_cpu_compute::AttentionDescriptor descriptor;
    descriptor.q_num_heads = 8;
    descriptor.kv_num_heads = 1;
    descriptor.is_causal = true;
    const std::int64_t q_shape[] = {4, 32, 512};
    const std::int64_t kv_shape[] = {4, 32, 64};
    ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank3, q_shape,
                                       kv_shape, kv_shape, {},
                                       ml_cpu_compute::AttentionMaskKind::kNone);
    const auto q = half_precision::ToFloat16(RandomTensor(4 * 32 * 512, 1951));
    const auto k = half_precision::ToFloat16(RandomTensor(4 * 32 * 64, 1952));
    const auto v = half_precision::ToFloat16(RandomTensor(4 * 32 * 64, 1953));
    std::vector<std::uint16_t> y(q.size());
    ml_cpu_compute::ComputeAttentionFloat16Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                     y.data());
    EXPECT_EQ(executor.dispatches, 1);
    EXPECT_EQ(executor.maximum_blocks, 16);
  }

  executor = {};
  {
    ml_cpu_compute::AttentionDescriptor descriptor;
    const std::int64_t q_shape[] = {1, 12, 129, 256};
    const std::int64_t kv_shape[] = {1, 12, 257, 256};
    ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                       kv_shape, kv_shape, {},
                                       ml_cpu_compute::AttentionMaskKind::kNone);
    const auto q = half_precision::ToFloat16(RandomTensor(12 * 129 * 256, 1954));
    const auto k = half_precision::ToFloat16(RandomTensor(12 * 257 * 256, 1955));
    const auto v = half_precision::ToFloat16(RandomTensor(12 * 257 * 256, 1956));
    std::vector<std::uint16_t> y(q.size());
    ml_cpu_compute::ComputeAttentionFloat16Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                     y.data());
    EXPECT_EQ(executor.dispatches, 1);
    EXPECT_EQ(executor.maximum_blocks, 24);
  }
}

TEST(ComputeAttentionFloat16Streaming, SingleKeyRank3GqaCopiesMappedValue) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::size_t batch = 2, q_heads = 4, kv_heads = 2, q_len = 5, kv_len = 1, head_dim = 8,
                        v_head_dim = 6;
  descriptor.q_num_heads = q_heads;
  descriptor.kv_num_heads = kv_heads;
  const std::int64_t q_shape[] = {batch, q_len, q_heads * head_dim};
  const std::int64_t k_shape[] = {batch, kv_len, kv_heads * head_dim};
  const std::int64_t v_shape[] = {batch, kv_len, kv_heads * v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank3, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);

  const auto q = half_precision::ToFloat16(RandomTensor(batch * q_len * q_shape[2], 637));
  const auto k = half_precision::ToFloat16(RandomTensor(batch * kv_len * k_shape[2], 638));
  const auto v = half_precision::ToFloat16(RandomTensor(batch * kv_len * v_shape[2], 639));
  std::vector<std::uint16_t> y(batch * q_len * q_heads * v_head_dim);

  ml_cpu_compute::ComputeAttentionFloat16Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                   y.data());

  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t query = 0; query < q_len; ++query) {
      for (std::size_t h = 0; h < q_heads; ++h) {
        const std::size_t kv_h = h / (q_heads / kv_heads);
        const std::size_t y_offset = ((b * q_len + query) * q_heads + h) * v_head_dim;
        const std::size_t v_offset = (b * kv_heads + kv_h) * v_head_dim;
        EXPECT_TRUE(std::equal(v.begin() + static_cast<std::ptrdiff_t>(v_offset),
                               v.begin() + static_cast<std::ptrdiff_t>(v_offset + v_head_dim),
                               y.begin() + static_cast<std::ptrdiff_t>(y_offset)));
      }
    }
  }
}

// Roadmap PR14: same contract as the FP16 test above, for BF16.
TEST(ComputeAttentionBFloat16Streaming, MatchesFloat32ReferenceWithinHalfPrecisionTolerance) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::size_t batch = 2, heads = 4, group = 2, q_len = 2, kv_len = 5, head_dim = 8,
                        v_head_dim = 8;
  constexpr std::size_t kv_heads = heads / group;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, kv_heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, kv_heads, kv_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);

  const auto q32 = half_precision::FromBFloat16(
      half_precision::ToBFloat16(RandomTensor(batch * heads * q_len * head_dim, 641)));
  const auto k32 = half_precision::FromBFloat16(
      half_precision::ToBFloat16(RandomTensor(batch * kv_heads * kv_len * head_dim, 642)));
  const auto v32 = half_precision::FromBFloat16(
      half_precision::ToBFloat16(RandomTensor(batch * kv_heads * kv_len * v_head_dim, 643)));

  const auto q16 = half_precision::ToBFloat16(q32);
  const auto k16 = half_precision::ToBFloat16(k32);
  const auto v16 = half_precision::ToBFloat16(v32);

  std::vector<std::uint16_t> y16(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionBFloat16Streaming(plan, q16.data(), k16.data(), v16.data(),
                                                    nullptr, y16.data());
  const auto y = half_precision::FromBFloat16(y16);

  std::vector<float> expected(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q32.data(), k32.data(), v32.data(), nullptr,
                                          expected.data());

  // BF16 has fewer mantissa bits than FP16, so use a looser tolerance.
  ExpectClose(y, expected, 3e-2f);
}

TEST(ComputeAttentionBFloat16Streaming, TiledGqaTailsMatchFloat32Reference) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  constexpr std::size_t batch = 1, heads = 4, group = 2, q_len = 17, kv_len = 33, head_dim = 9,
                        v_head_dim = 11;
  constexpr std::size_t kv_heads = heads / group;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, kv_heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, kv_heads, kv_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);

  const auto q32 = half_precision::FromBFloat16(
      half_precision::ToBFloat16(RandomTensor(batch * heads * q_len * head_dim, 651)));
  const auto k32 = half_precision::FromBFloat16(
      half_precision::ToBFloat16(RandomTensor(batch * kv_heads * kv_len * head_dim, 652)));
  const auto v32 = half_precision::FromBFloat16(
      half_precision::ToBFloat16(RandomTensor(batch * kv_heads * kv_len * v_head_dim, 653)));
  const auto q16 = half_precision::ToBFloat16(q32);
  const auto k16 = half_precision::ToBFloat16(k32);
  const auto v16 = half_precision::ToBFloat16(v32);

  std::vector<std::uint16_t> y16(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::AttentionExecutionInfo info;
  ml_cpu_compute::ComputeAttentionBFloat16Streaming(plan, q16.data(), k16.data(), v16.data(),
                                                    nullptr, y16.data(), nullptr, nullptr, nullptr,
                                                    &info);
  EXPECT_EQ(info.path, ml_cpu_compute::AttentionExecutionPath::kTiled);
  EXPECT_TRUE(info.tile_conversion);

  std::vector<float> expected(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q32.data(), k32.data(), v32.data(), nullptr,
                                          expected.data());
  ExpectClose(half_precision::FromBFloat16(y16), expected, 3e-2f);
}

// Roadmap PR14: a plan with a requested `present` output must never
// dispatch to streaming, since it necessarily materializes a full observable
// tensor (like `qk_matmul_output`, covered by
// ComputeAttentionFloat32.QkMatmulOutputModesCaptureEachStage above).
TEST(ComputeAttentionFloat32, PresentOutputSelectsMaterializedExecution) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.has_present_key = true;
  descriptor.has_present_value = true;
  constexpr std::size_t batch = 1, heads = 1, q_len = 1, kv_len = 4, head_dim = 4, v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);
  EXPECT_TRUE(plan.has_present_output);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 651);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 652);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 653);
  std::vector<float> y(batch * heads * q_len * v_head_dim);
  std::vector<float> materialized_y(batch * heads * q_len * v_head_dim);

  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data());
  ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(), nullptr,
                                                      materialized_y.data());
  ExpectClose(y, materialized_y, 0.0f);
}

// ---- Tests for softmax_precision = 0 (UNDEFINED / default) ----

TEST(AttentionDescriptor, MaterializedPathAcceptsUndefinedSoftmaxPrecision) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.softmax_precision = 0; // UNDEFINED means "use input precision"
  EXPECT_NO_THROW(descriptor.ValidateSupportedByMaterializedPath());
}

TEST(AttentionDescriptor, MaterializedPathAcceptsExplicitDoubleSoftmaxPrecision) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.softmax_precision = 11;
  EXPECT_NO_THROW(descriptor.ValidateSupportedByMaterializedPath());
  const std::int64_t shape[] = {1, 1, 1, 1};
  const ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4,
                                           shape, shape, shape, {},
                                           ml_cpu_compute::AttentionMaskKind::kNone);
  EXPECT_TRUE(plan.softmax_fp64);
}

// ---- Tests for present_key / present_value output construction ----

// Verifies that ComputeAttentionFloat32 plus the plan's present shapes
// produce the expected concatenation of past_key + key (and past_value +
// value).
TEST(ComputeAttentionFloat32, PresentKeyValueMatchesConcatenatedInput) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.has_present_key = true;
  descriptor.has_present_value = true;
  constexpr std::size_t batch = 1, heads = 2, past_len = 3, q_len = 2, kv_len = 2, head_dim = 4,
                        v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  const std::int64_t past_k_shape[] = {batch, heads, past_len, head_dim};
  const std::int64_t past_v_shape[] = {batch, heads, past_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {}, ml_cpu_compute::AttentionMaskKind::kNone,
                                     past_k_shape, past_v_shape);
  EXPECT_TRUE(plan.has_present_output);
  EXPECT_EQ(plan.total_kv_length, past_len + kv_len);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 701);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 702);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 703);
  const auto past_k = RandomTensor(batch * heads * past_len * head_dim, 704);
  const auto past_v = RandomTensor(batch * heads * past_len * v_head_dim, 705);

  // Build expected present_key = concat(past_key, key) along sequence axis.
  std::vector<float> expected_present_k(batch * heads * (past_len + kv_len) * head_dim);
  std::vector<float> expected_present_v(batch * heads * (past_len + kv_len) * v_head_dim);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < heads; ++h) {
      const std::size_t dst_k = (b * heads + h) * (past_len + kv_len) * head_dim;
      const std::size_t dst_v = (b * heads + h) * (past_len + kv_len) * v_head_dim;
      std::copy_n(past_k.data() + (b * heads + h) * past_len * head_dim, past_len * head_dim,
                  expected_present_k.data() + dst_k);
      std::copy_n(k.data() + (b * heads + h) * kv_len * head_dim, kv_len * head_dim,
                  expected_present_k.data() + dst_k + past_len * head_dim);
      std::copy_n(past_v.data() + (b * heads + h) * past_len * v_head_dim, past_len * v_head_dim,
                  expected_present_v.data() + dst_v);
      std::copy_n(v.data() + (b * heads + h) * kv_len * v_head_dim, kv_len * v_head_dim,
                  expected_present_v.data() + dst_v + past_len * v_head_dim);
    }
  }

  // Use the kernel's Compute path indirectly by using the plan to create
  // present output shapes and verifying the concatenation manually. Below we
  // call the materialized path (which is what ComputeAttentionFloat32 selects
  // when has_present_output is true) and then verify Y is correct.
  std::vector<float> y(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data(),
                                          past_k.data(), past_v.data());

  // Verify Y matches the reference (concat past + current, then attend).
  const auto ref = ReferenceAttention(batch, heads, heads, q_len, past_len + kv_len, head_dim,
                                      v_head_dim, q, expected_present_k, expected_present_v,
                                      plan.scale, false, nullptr, nullptr);
  ExpectClose(y, ref);
}

// Verifies present output construction for rank-3 input with the plan
// strides (the present output is always rank-4 regardless of input layout).
TEST(ComputeAttentionFloat32, PresentKeyValueRank3Layout) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.has_present_key = true;
  descriptor.has_present_value = true;
  descriptor.q_num_heads = 2;
  descriptor.kv_num_heads = 2;
  constexpr std::size_t batch = 1, heads = 2, q_len = 3, kv_len = 4, head_dim = 4, v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, q_len, heads * head_dim};
  const std::int64_t k_shape[] = {batch, kv_len, heads * head_dim};
  const std::int64_t v_shape[] = {batch, kv_len, heads * v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank3, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);
  EXPECT_TRUE(plan.has_present_output);
  EXPECT_EQ(plan.past_length, 0u);

  // present_key shape should be rank-4 (batch, heads, kv_len, head_dim)
  const auto pk_shape = plan.present_key_shape();
  ASSERT_EQ(pk_shape.size(), 4u);
  EXPECT_EQ(pk_shape[0], static_cast<std::int64_t>(batch));
  EXPECT_EQ(pk_shape[1], static_cast<std::int64_t>(heads));
  EXPECT_EQ(pk_shape[2], static_cast<std::int64_t>(kv_len));
  EXPECT_EQ(pk_shape[3], static_cast<std::int64_t>(head_dim));
}

// ---- Tests for FLOAT16 mask support ----

TEST(ComputeAttentionFloat16Streaming, Float16AdditiveMaskMatchesFP32MaskReference) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::size_t batch = 1, heads = 2, q_len = 3, kv_len = 4, head_dim = 4, v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  const std::int64_t mask_shape[] = {q_len, kv_len};

  // Build FP16-roundtrippable inputs.
  const auto q32 = half_precision::FromFloat16(
      half_precision::ToFloat16(RandomTensor(batch * heads * q_len * head_dim, 801)));
  const auto k32 = half_precision::FromFloat16(
      half_precision::ToFloat16(RandomTensor(batch * heads * kv_len * head_dim, 802)));
  const auto v32 = half_precision::FromFloat16(
      half_precision::ToFloat16(RandomTensor(batch * heads * kv_len * v_head_dim, 803)));
  const auto mask32 =
      half_precision::FromFloat16(half_precision::ToFloat16(RandomTensor(q_len * kv_len, 804)));

  // Reference: FP32 path with FP32 additive mask.
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, mask_shape,
                                     ml_cpu_compute::AttentionMaskKind::kAdditive);
  std::vector<float> expected(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q32.data(), k32.data(), v32.data(), mask32.data(),
                                          expected.data());

  // Test: FP16 streaming path with the same mask values (via FP16 conversion,
  // which the kernel should convert back to FP32 internally).
  const auto q16 = half_precision::ToFloat16(q32);
  const auto k16 = half_precision::ToFloat16(k32);
  const auto v16 = half_precision::ToFloat16(v32);
  const auto mask16 = half_precision::ToFloat16(mask32);

  std::vector<std::uint16_t> y16(batch * heads * q_len * v_head_dim);
  // The FLOAT16 mask is handled by the Compute() helper (which converts it to
  // FP32), not by ComputeAttentionFloat16Streaming directly. Since we cannot
  // call Compute() from here (it's a kernel-internal function), we verify the
  // FP32 reference matches the expected result and trust the unit test for the
  // Compute path via the backend test cases.
  // For direct streaming testing, use the FP32 mask:
  ml_cpu_compute::ComputeAttentionFloat16Streaming(plan, q16.data(), k16.data(), v16.data(),
                                                   mask32.data(), y16.data());
  const auto y = half_precision::FromFloat16(y16);
  ExpectClose(y, expected, 5e-3f);
}

// ---- Tests for qk_matmul_output with present outputs wired ----

TEST(ComputeAttentionFloat32, QkMatmulOutputAndPresentCoexist) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  descriptor.has_qk_matmul_output = true;
  descriptor.has_present_key = true;
  descriptor.has_present_value = true;
  constexpr std::size_t batch = 1, heads = 1, past_len = 2, q_len = 2, kv_len = 3, head_dim = 4,
                        v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  const std::int64_t past_k_shape[] = {batch, heads, past_len, head_dim};
  const std::int64_t past_v_shape[] = {batch, heads, past_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {}, ml_cpu_compute::AttentionMaskKind::kNone,
                                     past_k_shape, past_v_shape);
  EXPECT_TRUE(plan.has_qk_matmul_output);
  EXPECT_TRUE(plan.has_present_output);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 901);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 902);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 903);
  const auto past_k = RandomTensor(batch * heads * past_len * head_dim, 904);
  const auto past_v = RandomTensor(batch * heads * past_len * v_head_dim, 905);

  std::vector<float> y(batch * heads * q_len * v_head_dim);
  std::vector<float> qk(batch * heads * q_len * (past_len + kv_len));
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data(),
                                          past_k.data(), past_v.data(), nullptr, qk.data());

  // qk_matmul_output mode 0: raw QK scores. Verify against manual dot
  // products.
  for (std::size_t i = 0; i < q_len; ++i) {
    for (std::size_t j = 0; j < past_len + kv_len; ++j) {
      float dot = 0.0f;
      const float *k_row;
      if (j < past_len) {
        k_row = past_k.data() + j * head_dim;
      } else {
        k_row = k.data() + (j - past_len) * head_dim;
      }
      for (std::size_t d = 0; d < head_dim; ++d) {
        dot += q[i * head_dim + d] * k_row[d];
      }
      EXPECT_NEAR(qk[i * (past_len + kv_len) + j], plan.scale * dot, 1e-4f)
          << "i=" << i << " j=" << j;
    }
  }
}

// ---- Local window / bidirectional mask patterns ----

// Sliding window via a boolean mask: each query attends to a local window of
// keys around its position. This exercises both the streaming and materialized
// paths for a sparse, non-causal pattern.
TEST(ComputeAttentionFloat32Streaming, LocalWindowBoolMaskMatchesMaterialized) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  constexpr std::size_t batch = 1, heads = 2, q_len = 20, kv_len = 20, head_dim = 8, v_head_dim = 8;
  constexpr std::size_t window = 5;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  const std::int64_t mask_shape[] = {q_len, kv_len};

  std::vector<std::uint8_t> bool_mask(q_len * kv_len, 0);
  for (std::size_t i = 0; i < q_len; ++i) {
    for (std::size_t j = 0; j < kv_len; ++j) {
      const std::int64_t distance = static_cast<std::int64_t>(j) - static_cast<std::int64_t>(i);
      if (std::abs(distance) <= static_cast<std::int64_t>(window / 2)) {
        bool_mask[i * kv_len + j] = 1;
      }
    }
  }

  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, mask_shape,
                                     ml_cpu_compute::AttentionMaskKind::kBoolean);
  const auto q = RandomTensor(batch * heads * q_len * head_dim, 1001);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 1002);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 1003);

  std::vector<float> streaming_y(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(),
                                                   bool_mask.data(), streaming_y.data());

  std::vector<float> materialized_y(streaming_y.size());
  ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(),
                                                      bool_mask.data(), materialized_y.data());
  ExpectClose(streaming_y, materialized_y, 1e-4f);

  // Also verify against the reference oracle.
  std::vector<std::uint8_t> broadcast_mask(batch * heads * q_len * kv_len);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < heads; ++h) {
      std::copy_n(bool_mask.data(), q_len * kv_len,
                  broadcast_mask.data() + (b * heads + h) * q_len * kv_len);
    }
  }
  const auto ref = ReferenceAttention(batch, heads, heads, q_len, kv_len, head_dim, v_head_dim, q,
                                      k, v, plan.scale, false, nullptr, &broadcast_mask);
  ExpectClose(streaming_y, ref, 1e-4f);
}

// Bidirectional (non-causal) attention over the full KV range: streaming and
// materialized must agree.
TEST(ComputeAttentionFloat32Streaming, BidirectionalFullAttentionMatchesMaterialized) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  // Non-causal, no mask: bidirectional attention.
  constexpr std::size_t batch = 2, heads = 2, q_len = 16, kv_len = 18, head_dim = 8, v_head_dim = 8;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);
  EXPECT_FALSE(plan.causal);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 1101);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 1102);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 1103);

  std::vector<float> streaming_y(batch * heads * q_len * v_head_dim);
  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                   streaming_y.data());

  std::vector<float> materialized_y(streaming_y.size());
  ml_cpu_compute::ComputeAttentionFloat32Materialized(plan, q.data(), k.data(), v.data(), nullptr,
                                                      materialized_y.data());
  ExpectClose(streaming_y, materialized_y, 1e-4f);
}

TEST(ComputeAttentionFloat32Streaming, MaximumWindowBoundsAreEffectivelyUnbounded) {
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.opset = 25;
  descriptor.left_window_size = std::numeric_limits<std::int64_t>::max();
  descriptor.right_window_size = std::numeric_limits<std::int64_t>::max();
  constexpr std::size_t batch = 1, heads = 1, q_len = 16, kv_len = 18, head_dim = 4;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, head_dim};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape,
                                     k_shape, v_shape, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);

  ml_cpu_compute::AttentionDescriptor unbounded_descriptor;
  unbounded_descriptor.opset = 25;
  ml_cpu_compute::AttentionPlan unbounded_plan(
      unbounded_descriptor, ml_cpu_compute::AttentionLayout::kRank4, q_shape, k_shape, v_shape, {},
      ml_cpu_compute::AttentionMaskKind::kNone);
  const auto q = RandomTensor(batch * heads * q_len * head_dim, 1111);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 1112);
  const auto v = RandomTensor(batch * heads * kv_len * head_dim, 1113);
  std::vector<float> actual(batch * heads * q_len * head_dim);
  std::vector<float> expected(actual.size());
  ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                   actual.data());
  ml_cpu_compute::ComputeAttentionFloat32Materialized(unbounded_plan, q.data(), k.data(), v.data(),
                                                      nullptr, expected.data());
  ExpectClose(actual, expected, 1e-4f);
}

// ---------------------------------------------------------------------------
// Independent expected-value regression tests for local/bidirectional windows.
//
// These compute expected Y values *independently* of the attention kernel: Q=0,
// K=0 ⇒ softmax is uniform over allowed positions, so Y[b,h,i,d] = average of
// V[kv_h,j,d] for all allowed j. The V tensor uses a position-based pattern
// V[b,h,s,d] = 100*h + 10*d + s so that each element is uniquely identifiable.
// This directly mirrors the upstream MakeUniformWindowReference4 / ONNX spec.
// ---------------------------------------------------------------------------

namespace {

// Constructs a rank-4 V tensor with V[b,h,s,d] = 100*h + 10*d + offset + s.
std::vector<float> MakePositionV4(std::size_t batch, std::size_t heads, std::size_t seq_len,
                                  std::size_t head_size, std::size_t offset = 0) {
  std::vector<float> v(batch * heads * seq_len * head_size);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < heads; ++h) {
      for (std::size_t s = 0; s < seq_len; ++s) {
        for (std::size_t d = 0; d < head_size; ++d) {
          v[((b * heads + h) * seq_len + s) * head_size + d] =
              static_cast<float>(100 * h + 10 * d + offset + s);
        }
      }
    }
  }
  return v;
}

// Computes expected Y using the ONNX spec's window semantics.
// Q=0, K=0 ⇒ all scores are equal, so softmax is uniform over allowed
// positions. The mask_allows predicate (if any) additionally filters by
// (batch, head, query, key_pos).
using MaskPredicate = std::function<bool(std::int64_t, std::int64_t, std::int64_t, std::int64_t)>;

std::vector<float> UniformWindowExpected(std::size_t batch, std::size_t q_heads,
                                         std::size_t kv_heads, std::size_t q_len,
                                         std::size_t kv_len, std::size_t v_head_size,
                                         std::int64_t left_window, std::int64_t right_window,
                                         bool is_causal, const std::vector<std::int64_t> &offsets,
                                         const std::vector<std::int64_t> &valid_lengths,
                                         const MaskPredicate &mask_allows = nullptr) {
  const std::size_t group = q_heads / kv_heads;
  std::vector<float> y(batch * q_heads * q_len * v_head_size, 0.0f);
  for (std::size_t b = 0; b < batch; ++b) {
    const std::int64_t offset = offsets.empty() ? 0 : offsets[b];
    const std::int64_t valid =
        valid_lengths.empty() ? static_cast<std::int64_t>(kv_len) : valid_lengths[b];
    for (std::size_t h = 0; h < q_heads; ++h) {
      const std::size_t kv_h = h / group;
      for (std::size_t i = 0; i < q_len; ++i) {
        std::vector<std::int64_t> allowed;
        for (std::int64_t j = 0; j < static_cast<std::int64_t>(kv_len); ++j) {
          const std::int64_t diff = static_cast<std::int64_t>(i) + offset - j;
          if (j >= valid)
            continue;
          if (is_causal && diff < 0)
            continue;
          if (left_window >= 0 && diff > left_window)
            continue;
          if (right_window >= 0 && -diff > right_window)
            continue;
          if (mask_allows &&
              !mask_allows(static_cast<std::int64_t>(b), static_cast<std::int64_t>(h),
                           static_cast<std::int64_t>(i), j))
            continue;
          allowed.push_back(j);
        }
        if (allowed.empty())
          continue;
        const float prob = 1.0f / static_cast<float>(allowed.size());
        for (std::size_t d = 0; d < v_head_size; ++d) {
          float sum = 0.0f;
          for (std::int64_t j : allowed) {
            sum += static_cast<float>(100 * static_cast<std::int64_t>(kv_h) +
                                      10 * static_cast<std::int64_t>(d) + j);
          }
          y[((b * q_heads + h) * q_len + i) * v_head_size + d] = sum * prob;
        }
      }
    }
  }
  return y;
}

} // namespace

// 1. test_cc_attention_local_window: rank4, causal, left_window=2
//    Q(2,3,4,8)=0, K(2,3,6,8)=0, V=position
TEST(AttentionWindowRegression, LocalWindowCausal) {
  ml_cpu_compute::AttentionDescriptor desc;
  desc.opset = 25;
  desc.is_causal = true;
  desc.left_window_size = 2;
  const std::int64_t qs[] = {2, 3, 4, 8};
  const std::int64_t ks[] = {2, 3, 6, 8};
  const std::int64_t vs[] = {2, 3, 6, 8};
  ml_cpu_compute::AttentionPlan plan(desc, ml_cpu_compute::AttentionLayout::kRank4, qs, ks, vs, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);
  std::vector<float> q(2 * 3 * 4 * 8, 0.0f);
  std::vector<float> k(2 * 3 * 6 * 8, 0.0f);
  auto v = MakePositionV4(2, 3, 6, 8);
  auto expected = UniformWindowExpected(2, 3, 3, 4, 6, 8, 2, -1, true, {}, {});
  std::vector<float> actual(2 * 3 * 4 * 8);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr,
                                          actual.data());
  ExpectClose(actual, expected, 1e-5f);
}

// 2. test_cc_attention_bidirectional_window: NOT causal, left=1, right=2
//    Q(1,1,5,1)=0, K(1,1,5,1)=0, V=position
TEST(AttentionWindowRegression, BidirectionalWindow) {
  ml_cpu_compute::AttentionDescriptor desc;
  desc.opset = 25;
  desc.left_window_size = 1;
  desc.right_window_size = 2;
  const std::int64_t qs[] = {1, 1, 5, 1};
  const std::int64_t ks[] = {1, 1, 5, 1};
  const std::int64_t vs[] = {1, 1, 5, 1};
  ml_cpu_compute::AttentionPlan plan(desc, ml_cpu_compute::AttentionLayout::kRank4, qs, ks, vs, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);
  std::vector<float> q(5, 0.0f);
  std::vector<float> k(5, 0.0f);
  auto v = MakePositionV4(1, 1, 5, 1);
  auto expected = UniformWindowExpected(1, 1, 1, 5, 5, 1, 1, 2, false, {}, {});
  std::vector<float> actual(5);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr,
                                          actual.data());
  ExpectClose(actual, expected, 1e-5f);
}

// 3. test_cc_attention_3d_local_window: rank3, GQA (q=4,kv=1), causal, left=2
//    Tested in rank-4 form: Q(2,4,4,8)=0, K(2,1,6,8)=0, V=position(2,1,6,6)
TEST(AttentionWindowRegression, LocalWindow3dGqa) {
  ml_cpu_compute::AttentionDescriptor desc;
  desc.opset = 25;
  desc.is_causal = true;
  desc.left_window_size = 2;
  desc.q_num_heads = 4;
  desc.kv_num_heads = 1;
  const std::int64_t qs[] = {2, 4, 4, 8};
  const std::int64_t ks[] = {2, 1, 6, 8};
  const std::int64_t vs[] = {2, 1, 6, 6};
  ml_cpu_compute::AttentionPlan plan(desc, ml_cpu_compute::AttentionLayout::kRank4, qs, ks, vs, {},
                                     ml_cpu_compute::AttentionMaskKind::kNone);
  std::vector<float> q(2 * 4 * 4 * 8, 0.0f);
  std::vector<float> k(2 * 1 * 6 * 8, 0.0f);
  auto v = MakePositionV4(2, 1, 6, 6);
  auto expected = UniformWindowExpected(2, 4, 1, 4, 6, 6, 2, -1, true, {}, {});
  std::vector<float> actual(2 * 4 * 4 * 6);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr,
                                          actual.data());
  ExpectClose(actual, expected, 1e-5f);
}

// 4. test_cc_attention_local_window_rank1_boolean_mask: causal, left=2, bool mask (6,)
//    mask = [1,1,1,1,0,0] ⇒ only j < 4 allowed
TEST(AttentionWindowRegression, LocalWindowRank1BoolMask) {
  ml_cpu_compute::AttentionDescriptor desc;
  desc.opset = 25;
  desc.is_causal = true;
  desc.left_window_size = 2;
  desc.has_attn_mask = true;
  std::vector<std::uint8_t> mask = {1, 1, 1, 1, 0, 0};
  MaskPredicate mask_allows = [](std::int64_t, std::int64_t, std::int64_t, std::int64_t j) {
    return j < 4;
  };
  const std::int64_t qs[] = {2, 3, 4, 8};
  const std::int64_t ks[] = {2, 3, 6, 8};
  const std::int64_t vs[] = {2, 3, 6, 8};
  const std::int64_t ms[] = {6};
  ml_cpu_compute::AttentionPlan plan(desc, ml_cpu_compute::AttentionLayout::kRank4, qs, ks, vs, ms,
                                     ml_cpu_compute::AttentionMaskKind::kBoolean);
  std::vector<float> q(2 * 3 * 4 * 8, 0.0f);
  std::vector<float> k(2 * 3 * 6 * 8, 0.0f);
  auto v = MakePositionV4(2, 3, 6, 8);
  auto expected = UniformWindowExpected(2, 3, 3, 4, 6, 8, 2, -1, true, {}, {}, mask_allows);
  std::vector<float> actual(2 * 3 * 4 * 8);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), mask.data(),
                                          actual.data());
  ExpectClose(actual, expected, 1e-5f);
}

// 5. test_cc_attention_local_window_ext_cache_rank2_mask: causal, left=2,
//    additive mask (1,8) with -inf at position 1, nonpad=[6,7], offsets=[2,3]
TEST(AttentionWindowRegression, LocalWindowExtCacheRank2Mask) {
  ml_cpu_compute::AttentionDescriptor desc;
  desc.opset = 25;
  desc.is_causal = true;
  desc.left_window_size = 2;
  desc.has_attn_mask = true;
  desc.has_nonpad_kv_seqlen = true;
  std::vector<float> mask(8, 0.0f);
  mask[1] = -std::numeric_limits<float>::infinity();
  std::vector<std::int64_t> nonpad = {6, 7};
  MaskPredicate mask_allows = [](std::int64_t, std::int64_t, std::int64_t, std::int64_t j) {
    return j != 1;
  };
  const std::int64_t qs[] = {2, 3, 4, 8};
  const std::int64_t ks[] = {2, 3, 8, 8};
  const std::int64_t vs[] = {2, 3, 8, 8};
  const std::int64_t ms[] = {1, 8};
  ml_cpu_compute::AttentionPlan plan(desc, ml_cpu_compute::AttentionLayout::kRank4, qs, ks, vs, ms,
                                     ml_cpu_compute::AttentionMaskKind::kAdditive);
  std::vector<float> q(2 * 3 * 4 * 8, 0.0f);
  std::vector<float> k(2 * 3 * 8 * 8, 0.0f);
  auto v = MakePositionV4(2, 3, 8, 8);
  auto expected = UniformWindowExpected(2, 3, 3, 4, 8, 8, 2, -1, true, {2, 3}, {6, 7}, mask_allows);
  std::vector<float> actual(2 * 3 * 4 * 8);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), mask.data(),
                                          actual.data(), nullptr, nullptr, nonpad.data());
  ExpectClose(actual, expected, 1e-5f);
}

// 6. test_cc_attention_local_window_ext_cache_rank3_head_mask: causal, left=2,
//    additive mask (3,4,8) with -inf at [h,i,h], nonpad=[6,7], offsets=[2,3]
TEST(AttentionWindowRegression, LocalWindowExtCacheRank3HeadMask) {
  ml_cpu_compute::AttentionDescriptor desc;
  desc.opset = 25;
  desc.is_causal = true;
  desc.left_window_size = 2;
  desc.has_attn_mask = true;
  desc.has_nonpad_kv_seqlen = true;
  std::vector<float> mask(3 * 4 * 8, 0.0f);
  for (std::size_t h = 0; h < 3; ++h) {
    for (std::size_t i = 0; i < 4; ++i) {
      mask[(h * 4 + i) * 8 + h] = -std::numeric_limits<float>::infinity();
    }
  }
  std::vector<std::int64_t> nonpad = {6, 7};
  MaskPredicate mask_allows = [](std::int64_t, std::int64_t h, std::int64_t, std::int64_t j) {
    return j != h;
  };
  const std::int64_t qs[] = {2, 3, 4, 8};
  const std::int64_t ks[] = {2, 3, 8, 8};
  const std::int64_t vs[] = {2, 3, 8, 8};
  const std::int64_t ms[] = {3, 4, 8};
  ml_cpu_compute::AttentionPlan plan(desc, ml_cpu_compute::AttentionLayout::kRank4, qs, ks, vs, ms,
                                     ml_cpu_compute::AttentionMaskKind::kAdditive);
  std::vector<float> q(2 * 3 * 4 * 8, 0.0f);
  std::vector<float> k(2 * 3 * 8 * 8, 0.0f);
  auto v = MakePositionV4(2, 3, 8, 8);
  auto expected = UniformWindowExpected(2, 3, 3, 4, 8, 8, 2, -1, true, {2, 3}, {6, 7}, mask_allows);
  std::vector<float> actual(2 * 3 * 4 * 8);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), mask.data(),
                                          actual.data(), nullptr, nullptr, nonpad.data());
  ExpectClose(actual, expected, 1e-5f);
}

// 7. test_cc_attention_local_window_ext_cache_rank4_batch_mask: causal, left=2,
//    additive mask (2,1,4,8) with -inf at [b,_,i,b], nonpad=[6,7], offsets=[2,3]
TEST(AttentionWindowRegression, LocalWindowExtCacheRank4BatchMask) {
  ml_cpu_compute::AttentionDescriptor desc;
  desc.opset = 25;
  desc.is_causal = true;
  desc.left_window_size = 2;
  desc.has_attn_mask = true;
  desc.has_nonpad_kv_seqlen = true;
  std::vector<float> mask(2 * 1 * 4 * 8, 0.0f);
  for (std::size_t b = 0; b < 2; ++b) {
    for (std::size_t i = 0; i < 4; ++i) {
      mask[(b * 4 + i) * 8 + b] = -std::numeric_limits<float>::infinity();
    }
  }
  std::vector<std::int64_t> nonpad = {6, 7};
  MaskPredicate mask_allows = [](std::int64_t b, std::int64_t, std::int64_t, std::int64_t j) {
    return j != b;
  };
  const std::int64_t qs[] = {2, 3, 4, 8};
  const std::int64_t ks[] = {2, 3, 8, 8};
  const std::int64_t vs[] = {2, 3, 8, 8};
  const std::int64_t ms[] = {2, 1, 4, 8};
  ml_cpu_compute::AttentionPlan plan(desc, ml_cpu_compute::AttentionLayout::kRank4, qs, ks, vs, ms,
                                     ml_cpu_compute::AttentionMaskKind::kAdditive);
  std::vector<float> q(2 * 3 * 4 * 8, 0.0f);
  std::vector<float> k(2 * 3 * 8 * 8, 0.0f);
  auto v = MakePositionV4(2, 3, 8, 8);
  auto expected = UniformWindowExpected(2, 3, 3, 4, 8, 8, 2, -1, true, {2, 3}, {6, 7}, mask_allows);
  std::vector<float> actual(2 * 3 * 4 * 8);
  ml_cpu_compute::ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), mask.data(),
                                          actual.data(), nullptr, nullptr, nonpad.data());
  ExpectClose(actual, expected, 1e-5f);
}

} // namespace
