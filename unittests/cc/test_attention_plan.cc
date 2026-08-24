// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/attention/attention_plan.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using onnx_light_cpu::AttentionDescriptor;
using onnx_light_cpu::AttentionLayout;
using onnx_light_cpu::AttentionMaskKind;
using onnx_light_cpu::AttentionPlan;
using onnx_light_cpu::ComputeAttentionFloat32;

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

TEST(AttentionDescriptor, ValidatesOpsetRange) {
  AttentionDescriptor descriptor;
  descriptor.opset = 22;
  EXPECT_THROW(descriptor.Validate(), std::invalid_argument);
}

TEST(AttentionDescriptor, RejectsNonpadKvSeqlenBeforeOpset24) {
  AttentionDescriptor descriptor;
  descriptor.opset = 23;
  descriptor.has_nonpad_kv_seqlen = true;
  EXPECT_THROW(descriptor.Validate(), std::invalid_argument);
}

TEST(AttentionDescriptor, RejectsUnpairedPastCache) {
  AttentionDescriptor descriptor;
  descriptor.has_past_key = true;
  EXPECT_THROW(descriptor.Validate(), std::invalid_argument);
}

TEST(AttentionDescriptor, MaterializedPathAcceptsPastCache) {
  AttentionDescriptor descriptor;
  descriptor.has_past_key = true;
  descriptor.has_past_value = true;
  EXPECT_NO_THROW(descriptor.ValidateSupportedByMaterializedPath());
}

TEST(AttentionDescriptor, MaterializedPathAcceptsSoftcap) {
  AttentionDescriptor descriptor;
  descriptor.softcap = 0.5f;
  EXPECT_NO_THROW(descriptor.ValidateSupportedByMaterializedPath());
}

TEST(AttentionDescriptor, MaterializedPathAcceptsQkMatmulOutput) {
  AttentionDescriptor descriptor;
  descriptor.has_qk_matmul_output = true;
  EXPECT_NO_THROW(descriptor.ValidateSupportedByMaterializedPath());
}

TEST(AttentionDescriptor, MaterializedPathAcceptsExplicitFloatSoftmaxPrecision) {
  AttentionDescriptor descriptor;
  constexpr std::int64_t kFloat = 1;
  descriptor.softmax_precision = kFloat;
  EXPECT_NO_THROW(descriptor.ValidateSupportedByMaterializedPath());
}

TEST(AttentionDescriptor, MaterializedPathRejectsNonFloatSoftmaxPrecision) {
  AttentionDescriptor descriptor;
  constexpr std::int64_t kDouble = 11;
  descriptor.softmax_precision = kDouble;
  EXPECT_THROW(descriptor.ValidateSupportedByMaterializedPath(), std::invalid_argument);
}

TEST(AttentionDescriptor, MaterializedPathAcceptsStatelessDefaults) {
  AttentionDescriptor descriptor;
  EXPECT_NO_THROW(descriptor.ValidateSupportedByMaterializedPath());
}

TEST(AttentionPlan, Rank4MhaDefaultScaleAndShape) {
  AttentionDescriptor descriptor;
  const std::int64_t q_shape[] = {2, 4, 5, 8};
  const std::int64_t k_shape[] = {2, 4, 6, 8};
  const std::int64_t v_shape[] = {2, 4, 6, 16};
  AttentionPlan plan(descriptor, AttentionLayout::kRank4, q_shape, k_shape, v_shape, {},
                     AttentionMaskKind::kNone);
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
  AttentionDescriptor descriptor;
  const std::int64_t q_shape[] = {1, 5, 4, 8};
  const std::int64_t k_shape[] = {1, 2, 4, 8};
  const std::int64_t v_shape[] = {1, 2, 4, 8};
  EXPECT_THROW(AttentionPlan(descriptor, AttentionLayout::kRank4, q_shape, k_shape, v_shape, {},
                             AttentionMaskKind::kNone),
               std::invalid_argument);
}

TEST(AttentionPlan, Rank3RequiresHeadCountAttributes) {
  AttentionDescriptor descriptor;
  const std::int64_t q_shape[] = {1, 4, 32};
  const std::int64_t k_shape[] = {1, 4, 32};
  const std::int64_t v_shape[] = {1, 4, 32};
  EXPECT_THROW(AttentionPlan(descriptor, AttentionLayout::kRank3, q_shape, k_shape, v_shape, {},
                             AttentionMaskKind::kNone),
               std::invalid_argument);
}

// MHA, rank-4, no mask, no causal: differential test against the naive
// reference oracle.
TEST(ComputeAttentionFloat32, Rank4MhaNoMaskMatchesReference) {
  AttentionDescriptor descriptor;
  constexpr std::size_t batch = 2, heads = 3, q_len = 4, kv_len = 5, head_dim = 8, v_head_dim = 8;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  AttentionPlan plan(descriptor, AttentionLayout::kRank4, q_shape, k_shape, v_shape, {},
                     AttentionMaskKind::kNone);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 1);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 2);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 3);
  std::vector<float> y(batch * heads * q_len * v_head_dim);
  ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data());

  const auto expected = ReferenceAttention(batch, heads, heads, q_len, kv_len, head_dim, v_head_dim,
                                           q, k, v, plan.scale, false, nullptr, nullptr);
  ExpectClose(y, expected);
}

// GQA, rank-4, causal.
TEST(ComputeAttentionFloat32, Rank4GqaCausalMatchesReference) {
  AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  constexpr std::size_t batch = 1, q_heads = 4, kv_heads = 2, len = 6, head_dim = 4, v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, q_heads, len, head_dim};
  const std::int64_t k_shape[] = {batch, kv_heads, len, head_dim};
  const std::int64_t v_shape[] = {batch, kv_heads, len, v_head_dim};
  AttentionPlan plan(descriptor, AttentionLayout::kRank4, q_shape, k_shape, v_shape, {},
                     AttentionMaskKind::kNone);
  EXPECT_EQ(plan.group_size, 2u);
  EXPECT_TRUE(plan.causal);

  const auto q = RandomTensor(batch * q_heads * len * head_dim, 11);
  const auto k = RandomTensor(batch * kv_heads * len * head_dim, 12);
  const auto v = RandomTensor(batch * kv_heads * len * v_head_dim, 13);
  std::vector<float> y(batch * q_heads * len * v_head_dim);
  ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data());

  const auto expected = ReferenceAttention(batch, q_heads, kv_heads, len, len, head_dim, v_head_dim,
                                           q, k, v, plan.scale, true, nullptr, nullptr);
  ExpectClose(y, expected);
}

// MQA, rank-4, boolean mask.
TEST(ComputeAttentionFloat32, Rank4MqaBooleanMaskMatchesReference) {
  AttentionDescriptor descriptor;
  constexpr std::size_t batch = 1, q_heads = 4, kv_heads = 1, q_len = 3, kv_len = 5, head_dim = 4,
                        v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, q_heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, kv_heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, kv_heads, kv_len, v_head_dim};
  const std::int64_t mask_shape[] = {q_len, kv_len};
  AttentionPlan plan(descriptor, AttentionLayout::kRank4, q_shape, k_shape, v_shape, mask_shape,
                     AttentionMaskKind::kBoolean);
  EXPECT_EQ(plan.group_size, 4u);

  const auto q = RandomTensor(batch * q_heads * q_len * head_dim, 21);
  const auto k = RandomTensor(batch * kv_heads * kv_len * head_dim, 22);
  const auto v = RandomTensor(batch * kv_heads * kv_len * v_head_dim, 23);
  std::vector<std::uint8_t> bool_mask(q_len * kv_len, 1);
  bool_mask[0] = 0; // First query row cannot see the first key.
  bool_mask[kv_len] = 0;

  std::vector<float> y(batch * q_heads * q_len * v_head_dim);
  ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), bool_mask.data(), y.data());

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
  AttentionDescriptor descriptor;
  descriptor.q_num_heads = 2;
  descriptor.kv_num_heads = 2;
  constexpr std::size_t batch = 2, heads = 2, q_len = 3, kv_len = 4, head_dim = 4, v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, q_len, heads * head_dim};
  const std::int64_t k_shape[] = {batch, kv_len, heads * head_dim};
  const std::int64_t v_shape[] = {batch, kv_len, heads * v_head_dim};
  const std::int64_t mask_shape[] = {batch, 1, q_len, kv_len};
  AttentionPlan plan(descriptor, AttentionLayout::kRank3, q_shape, k_shape, v_shape, mask_shape,
                     AttentionMaskKind::kAdditive);

  const auto q_r3 = RandomTensor(batch * q_len * heads * head_dim, 31);
  const auto k_r3 = RandomTensor(batch * kv_len * heads * head_dim, 32);
  const auto v_r3 = RandomTensor(batch * kv_len * heads * v_head_dim, 33);
  const auto mask = RandomTensor(batch * q_len * kv_len, 34);

  std::vector<float> y(batch * q_len * heads * v_head_dim);
  ComputeAttentionFloat32(plan, q_r3.data(), k_r3.data(), v_r3.data(), mask.data(), y.data());

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
  AttentionDescriptor descriptor;
  constexpr std::size_t batch = 1, heads = 1, q_len = 2, kv_len = 3, head_dim = 4, v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  const std::int64_t mask_shape[] = {q_len, kv_len};
  AttentionPlan plan(descriptor, AttentionLayout::kRank4, q_shape, k_shape, v_shape, mask_shape,
                     AttentionMaskKind::kBoolean);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 41);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 42);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 43);
  std::vector<std::uint8_t> mask(q_len * kv_len, 1);
  // Row 0 is fully masked; row 1 keeps every key.
  for (std::size_t j = 0; j < kv_len; ++j) {
    mask[j] = 0;
  }

  std::vector<float> y(batch * heads * q_len * v_head_dim);
  ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), mask.data(), y.data());
  for (std::size_t d = 0; d < v_head_dim; ++d) {
    EXPECT_EQ(y[d], 0.0f);
  }
  bool any_nonzero = false;
  for (std::size_t d = 0; d < v_head_dim; ++d) {
    any_nonzero |= (y[v_head_dim + d] != 0.0f);
  }
  EXPECT_TRUE(any_nonzero);
}

TEST(ComputeAttentionFloat32, SoftcapMatchesManualTanhFormula) {
  AttentionDescriptor descriptor;
  descriptor.softcap = 2.0f;
  descriptor.has_qk_matmul_output = true;
  // Mode 1 captures the post-softcap (pre-bias) score; mode 0 (the default)
  // captures the raw pre-softcap score instead.
  descriptor.qk_matmul_output_mode = 1;
  constexpr std::size_t batch = 1, heads = 1, q_len = 1, kv_len = 3, head_dim = 4, v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  AttentionPlan plan(descriptor, AttentionLayout::kRank4, q_shape, k_shape, v_shape, {},
                     AttentionMaskKind::kNone);
  EXPECT_NEAR(plan.softcap, 2.0f, 1e-6f);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 51);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 52);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 53);
  std::vector<float> qk(kv_len);
  std::vector<float> y(v_head_dim);
  ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data(), nullptr, nullptr,
                          nullptr, qk.data());
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
  AttentionDescriptor descriptor;
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
    AttentionPlan plan(descriptor, AttentionLayout::kRank4, q_shape, k_shape, v_shape, {},
                       AttentionMaskKind::kNone);
    EXPECT_TRUE(plan.has_qk_matmul_output);
    std::vector<float> qk(len * len, -123.0f);
    ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data(), nullptr, nullptr,
                            nullptr, qk.data());

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
  AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  constexpr std::size_t batch = 1, heads = 2, past_len = 3, q_len = 2, kv_len = 2, head_dim = 4,
                        v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  const std::int64_t past_k_shape[] = {batch, heads, past_len, head_dim};
  const std::int64_t past_v_shape[] = {batch, heads, past_len, v_head_dim};
  AttentionPlan plan(descriptor, AttentionLayout::kRank4, q_shape, k_shape, v_shape, {},
                     AttentionMaskKind::kNone, past_k_shape, past_v_shape);
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
  ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data(), past_k.data(),
                          past_v.data());

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
  AttentionDescriptor descriptor;
  descriptor.is_causal = true;
  constexpr std::size_t batch = 2, heads = 1, q_len = 2, kv_len = 5, head_dim = 4, v_head_dim = 4;
  const std::int64_t q_shape[] = {batch, heads, q_len, head_dim};
  const std::int64_t k_shape[] = {batch, heads, kv_len, head_dim};
  const std::int64_t v_shape[] = {batch, heads, kv_len, v_head_dim};
  AttentionPlan plan(descriptor, AttentionLayout::kRank4, q_shape, k_shape, v_shape, {},
                     AttentionMaskKind::kNone);
  EXPECT_EQ(plan.causal_offset, 0);

  const auto q = RandomTensor(batch * heads * q_len * head_dim, 81);
  const auto k = RandomTensor(batch * heads * kv_len * head_dim, 82);
  const auto v = RandomTensor(batch * heads * kv_len * v_head_dim, 83);
  // batch 0: only 2 valid keys, offset = 2 - 2 = 0 (standard causal).
  // batch 1: 4 valid keys (1 padding tail), offset = 4 - 2 = 2 (shifted).
  const std::vector<std::int64_t> nonpad = {2, 4};

  std::vector<float> y(batch * heads * q_len * v_head_dim);
  ComputeAttentionFloat32(plan, q.data(), k.data(), v.data(), nullptr, y.data(), nullptr, nullptr,
                          nonpad.data());

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

} // namespace
