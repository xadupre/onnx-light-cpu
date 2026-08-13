// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/gemm_plan.h"

#include <gtest/gtest.h>

#include <array>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using onnx_light_cpu::GemmAlgorithm;
using onnx_light_cpu::GemmPlan;
using onnx_light_cpu::GemmPlanOptions;
using onnx_light_cpu::GroupedGemm;
using onnx_light_cpu::GroupedGemmProblem;
using onnx_light_cpu::MatMulPlan;
using onnx_light_cpu::StridedBatchedGemm;

void ExpectValues(const std::vector<float> &actual, std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 1e-5f) << "index=" << index;
  }
}

TEST(GemmPlan, ExecutesExistingKernelAndExposesSelection) {
  const GemmPlan<float> plan(GemmPlanOptions<float>{false, false, 2, 2, 3, 0.5f, 2.0f, {}});
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  const std::vector<float> b = {1, 2, 3, 4, 5, 6};
  const std::vector<float> c = {1, 1, 1, 1};
  std::vector<float> y(4);

  plan.Execute(a.data(), b.data(), c.data(), y.data());

  ExpectValues(y, std::array<float, 4>{13, 16, 26.5f, 34});
  EXPECT_EQ(plan.algorithm(), GemmAlgorithm::kGeneral);
  EXPECT_EQ(plan.blocking().mc, 64u);
  EXPECT_EQ(plan.blocking().nc, 256u);
  EXPECT_EQ(plan.blocking().kc, 256u);
  EXPECT_GT(plan.blocking().nr, 0u);
  EXPECT_GE(plan.useful_threads(), 1u);
}

TEST(GemmPlan, OwnsConstantB) {
  std::vector<float> b = {1, 2, 3, 4, 5, 6};
  const GemmPlan<float> plan(GemmPlanOptions<float>{false, false, 2, 2, 3, 1.0f, 0.0f, b});
  b.assign(b.size(), 0.0f);
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  std::vector<float> y(4);

  plan.Execute(a.data(), nullptr, y.data());

  ExpectValues(y, std::array<float, 4>{22, 28, 49, 64});
  EXPECT_TRUE(plan.has_constant_b());
}

TEST(MatMulPlan, ExecutesRankTwoFoundation) {
  const MatMulPlan<double> plan(2, 2, 2);
  const std::array<double, 4> a = {1, 2, 3, 4};
  const std::array<double, 4> b = {5, 6, 7, 8};
  std::array<double, 4> y = {};

  plan.Execute(a.data(), b.data(), y.data());

  EXPECT_DOUBLE_EQ(y[0], 19);
  EXPECT_DOUBLE_EQ(y[1], 22);
  EXPECT_DOUBLE_EQ(y[2], 43);
  EXPECT_DOUBLE_EQ(y[3], 50);
}

TEST(StridedBatchedGemm, ExecutesEveryBatchWithElementStrides) {
  const std::array<float, 4> b = {1, 0, 0, 1};
  const GemmPlan<float> plan(GemmPlanOptions<float>{false, false, 2, 2, 2, 1.0f, 0.0f, b});
  const std::vector<float> a = {1, 2, 3, 4, -1, -1, 5, 6, 7, 8};
  std::vector<float> y(10, -1.0f);

  StridedBatchedGemm<float>(plan, 2, a.data(), 6, nullptr, 0, nullptr, 0, y.data(), 6);

  ExpectValues({y.begin(), y.begin() + 4}, std::array<float, 4>{1, 2, 3, 4});
  ExpectValues({y.begin() + 6, y.end()}, std::array<float, 4>{5, 6, 7, 8});
}

TEST(GroupedGemm, ExecutesHeterogeneousPlans) {
  const GemmPlan<float> first(GemmPlanOptions<float>{false, false, 1, 2, 2});
  const GemmPlan<float> second(GemmPlanOptions<float>{false, false, 2, 1, 2});
  const std::array<float, 2> a1 = {2, 3};
  const std::array<float, 4> b1 = {1, 2, 3, 4};
  const std::array<float, 4> a2 = {1, 2, 3, 4};
  const std::array<float, 2> b2 = {5, 6};
  std::array<float, 2> y1 = {};
  std::array<float, 2> y2 = {};
  const std::array<GroupedGemmProblem<float>, 2> problems = {
      GroupedGemmProblem<float>{&first, a1.data(), b1.data(), nullptr, y1.data()},
      GroupedGemmProblem<float>{&second, a2.data(), b2.data(), nullptr, y2.data()},
  };

  GroupedGemm<float>(problems);

  EXPECT_FLOAT_EQ(y1[0], 11);
  EXPECT_FLOAT_EQ(y1[1], 16);
  EXPECT_FLOAT_EQ(y2[0], 17);
  EXPECT_FLOAT_EQ(y2[1], 39);
}

TEST(GemmPlan, RejectsInvalidConstantAndMissingDynamicB) {
  const std::array<float, 1> wrong_size_b = {1.0f};
  EXPECT_THROW(
      (GemmPlan<float>(GemmPlanOptions<float>{false, false, 2, 2, 3, 1.0f, 0.0f, wrong_size_b})),
      std::invalid_argument);

  const GemmPlan<float> dynamic_plan(GemmPlanOptions<float>{false, false, 1, 1, 1});
  const float a = 1.0f;
  float y = 0.0f;
  EXPECT_THROW(dynamic_plan.Execute(&a, nullptr, nullptr, &y), std::invalid_argument);
  EXPECT_THROW(dynamic_plan.Execute(&a, nullptr, &y), std::logic_error);
}

} // namespace
