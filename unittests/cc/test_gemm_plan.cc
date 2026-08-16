// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/arm/gemm_kernel_arm.h"
#include "onnx_light_cpu/impl/math/gemm/gemm_plan.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using onnx_light_cpu::GemmAlgorithm;
using onnx_light_cpu::GemmHalfPlan;
using onnx_light_cpu::GemmHalfPlanOptions;
using onnx_light_cpu::GemmPlan;
using onnx_light_cpu::GemmPlanOptions;
using onnx_light_cpu::GroupedGemm;
using onnx_light_cpu::GroupedGemmProblem;
using onnx_light_cpu::MatMulPlan;
using onnx_light_cpu::StridedBatchedGemm;

// Encodes an integer-valued float as bfloat16, which is exact because the low 16
// mantissa bits are zero for these small operands.
std::uint16_t EncodeBFloat16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16);
}

float DecodeBFloat16(std::uint16_t bits) {
  const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

void ExpectValues(const std::vector<float> &actual, std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 1e-5f) << "index=" << index;
  }
}

template <std::size_t Size>
void ExpectShape(std::span<const std::size_t> actual,
                 const std::array<std::size_t, Size> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < Size; ++index) {
    EXPECT_EQ(actual[index], expected[index]) << "axis=" << index;
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
  EXPECT_EQ(plan.algorithm(), GemmAlgorithm::kDirect);
  EXPECT_GE(plan.blocking().mc, plan.blocking().mr);
  EXPECT_GE(plan.blocking().nc, plan.blocking().nr);
  EXPECT_GE(plan.blocking().kc, 64u);
  EXPECT_EQ(plan.blocking().mc % plan.blocking().mr, 0u);
  EXPECT_EQ(plan.blocking().nc % plan.blocking().nr, 0u);
  EXPECT_EQ(plan.blocking().kc % 16, 0u);
  EXPECT_GE(plan.useful_threads(), 1u);
}

TEST(GemmPlan, SelectsShapeSpecificAlgorithms) {
  const GemmPlan<float> general(GemmPlanOptions<float>{false, false, 128, 128, 128});
  const GemmPlan<float> skinny_m(GemmPlanOptions<float>{false, false, 1, 1024, 1024});
  const GemmPlan<float> skinny_n(GemmPlanOptions<float>{false, false, 128, 1, 1024});
  const GemmPlan<float> split_k(GemmPlanOptions<float>{false, false, 2, 2, 4096});

  EXPECT_EQ(general.algorithm(), GemmAlgorithm::kGeneral);
  EXPECT_EQ(skinny_m.algorithm(), GemmAlgorithm::kSkinnyM);
  EXPECT_EQ(skinny_n.algorithm(), GemmAlgorithm::kSkinnyN);
  EXPECT_EQ(split_k.algorithm(), GemmAlgorithm::kSplitK);
}

TEST(GemmPlan, BlockingUsesIsaSpecificRegisterRows) {
  const auto avx2 = onnx_light_cpu::detail::SelectGemmBlocking(sizeof(float), 8, 4);
  const auto avx512 = onnx_light_cpu::detail::SelectGemmBlocking(sizeof(float), 16, 6);
  const auto sve384 = onnx_light_cpu::detail::SelectGemmBlocking(sizeof(float), 12, 4);

  EXPECT_EQ(avx2.mr, 4u);
  EXPECT_EQ(avx512.mr, 6u);
  EXPECT_EQ(avx2.mc % avx2.mr, 0u);
  EXPECT_EQ(avx512.mc % avx512.mr, 0u);
  EXPECT_EQ(sve384.nc % sve384.nr, 0u);
  EXPECT_EQ(onnx_light_cpu::detail::SelectGemmAlgorithm(false, false, 5, 257, 64, 16, 4),
            GemmAlgorithm::kGeneral);
  EXPECT_EQ(onnx_light_cpu::detail::SelectGemmAlgorithm(false, false, 5, 257, 64, 16, 6),
            GemmAlgorithm::kSkinnyM);
}

TEST(GemmPlan, SelectsRuntimeVectorLengthAwareArmProfile) {
  using onnx_light_cpu::ArmGemmKernelKind;
  using onnx_light_cpu::ArmSimdLevel;
  using onnx_light_cpu::SelectArmGemmProfile;

  const auto scalar = SelectArmGemmProfile(ArmSimdLevel::kNone, 0, true);
  const auto neon = SelectArmGemmProfile(ArmSimdLevel::kNeon, 0, true);
  const auto sve128 = SelectArmGemmProfile(ArmSimdLevel::kSve, 16, true);
  const auto sve256 = SelectArmGemmProfile(ArmSimdLevel::kSve, 32, true);
  const auto sve2 = SelectArmGemmProfile(ArmSimdLevel::kSve2, 64, true);
  const auto no_sve_kernel = SelectArmGemmProfile(ArmSimdLevel::kSve2, 64, false);

  EXPECT_EQ(scalar.kind, ArmGemmKernelKind::kScalar);
  EXPECT_EQ(neon.kind, ArmGemmKernelKind::kNeon);
  EXPECT_EQ(neon.vector_bytes, 16u);
  EXPECT_EQ(neon.register_rows, onnx_light_cpu::kGemmNeonMR);
  EXPECT_EQ(sve128.kind, ArmGemmKernelKind::kNeon);
  EXPECT_EQ(sve256.kind, ArmGemmKernelKind::kSve);
  EXPECT_EQ(sve256.vector_bytes, 32u);
  EXPECT_EQ(sve256.register_rows, onnx_light_cpu::kGemmSveMR);
  EXPECT_EQ(sve2.kind, ArmGemmKernelKind::kSve);
  EXPECT_EQ(no_sve_kernel.kind, ArmGemmKernelKind::kNeon);
}

TEST(GemmPlan, SelectsMicroarchitectureRegisterRows) {
  using onnx_light_cpu::GemmMicroarchitecture;
  using onnx_light_cpu::SimdLevel;
  using onnx_light_cpu::detail::SelectGemmRegisterRowsForMicroarchitecture;

  EXPECT_EQ(SelectGemmRegisterRowsForMicroarchitecture(SimdLevel::kAVX2, true,
                                                       GemmMicroarchitecture::kGeneric),
            4u);
  EXPECT_EQ(SelectGemmRegisterRowsForMicroarchitecture(SimdLevel::kAVX2, true,
                                                       GemmMicroarchitecture::kIntelCore),
            5u);
  EXPECT_EQ(SelectGemmRegisterRowsForMicroarchitecture(SimdLevel::kAVX2, true,
                                                       GemmMicroarchitecture::kAmdZen),
            6u);
  EXPECT_EQ(SelectGemmRegisterRowsForMicroarchitecture(SimdLevel::kAVX512, true,
                                                       GemmMicroarchitecture::kIntelCore),
            6u);
  EXPECT_EQ(SelectGemmRegisterRowsForMicroarchitecture(SimdLevel::kAVX512, true,
                                                       GemmMicroarchitecture::kAmdZen),
            6u);
  EXPECT_EQ(SelectGemmRegisterRowsForMicroarchitecture(SimdLevel::kAVX2, false,
                                                       GemmMicroarchitecture::kIntelCore),
            4u);
  EXPECT_EQ(SelectGemmRegisterRowsForMicroarchitecture(SimdLevel::kAVX2, false,
                                                       GemmMicroarchitecture::kAmdZen),
            4u);
}

TEST(GemmPlan, ColumnBlockSlicesPackedPanels) {
  using onnx_light_cpu::detail::SelectGemmColumnBlock;
  const onnx_light_cpu::GemmBlocking avx2_f32{240, 1024, 272, 6, 16};
  const onnx_light_cpu::GemmBlocking avx2_f64{240, 1024, 208, 6, 8};
  const onnx_light_cpu::GemmBlocking narrow{240, 8, 272, 6, 16};

  const std::size_t f32_block = SelectGemmColumnBlock(avx2_f32, sizeof(float));
  const std::size_t f64_block = SelectGemmColumnBlock(avx2_f64, sizeof(double));

  EXPECT_EQ(f32_block % avx2_f32.nr, 0u);
  EXPECT_EQ(f64_block % avx2_f64.nr, 0u);
  EXPECT_GE(f32_block, avx2_f32.nr);
  EXPECT_GE(f64_block, avx2_f64.nr);
  EXPECT_LE(f32_block, avx2_f32.nc);
  EXPECT_LE(f64_block, avx2_f64.nc);
  // A float64 element is twice as wide, so the same byte budget covers half as
  // many columns.
  EXPECT_LE(f64_block, f32_block);
  // A column panel narrower than the micro-panel is never split further.
  EXPECT_EQ(SelectGemmColumnBlock(narrow, sizeof(float)), narrow.nc);
}

TEST(GemmPlan, BlockingExposesTaskGridForPrioritySizes) {
  const onnx_light_cpu::GemmBlocking initial{256, 1024, 448, 4, 16};
  for (const std::size_t size : {256u, 512u, 1024u}) {
    const auto blocking =
        onnx_light_cpu::detail::ConstrainGemmBlockingForTasks(initial, size, size, 6);
    const std::size_t row_tasks = (size + blocking.mc - 1) / blocking.mc;
    const std::size_t column_tasks = (size + blocking.nc - 1) / blocking.nc;

    EXPECT_GE(row_tasks * column_tasks, 6u) << "size=" << size;
    EXPECT_EQ(blocking.mc % blocking.mr, 0u);
    EXPECT_EQ(blocking.nc % blocking.nr, 0u);
    EXPECT_LE(blocking.mc, initial.mc);
    EXPECT_LE(blocking.nc, initial.nc);
  }
}

TEST(GemmPlan, ExecutesEveryPreparedAlgorithm) {
  const auto check = [](std::size_t m, std::size_t n, std::size_t k,
                        GemmAlgorithm expected_algorithm) {
    const GemmPlan<float> plan(GemmPlanOptions<float>{false, false, m, n, k});
    const std::vector<float> a(m * k, 1.0f);
    const std::vector<float> b(k * n, 1.0f);
    std::vector<float> y(m * n);

    plan.Execute(a.data(), b.data(), nullptr, y.data());

    EXPECT_EQ(plan.algorithm(), expected_algorithm);
    for (float value : y) {
      EXPECT_FLOAT_EQ(value, static_cast<float>(k));
    }
  };

  check(2, 3, 16, GemmAlgorithm::kDirect);
  check(2, 257, 64, GemmAlgorithm::kSkinnyM);
  check(16, 1, 64, GemmAlgorithm::kSkinnyN);
  check(32, 32, 64, GemmAlgorithm::kGeneral);
  check(2, 2, 4096, GemmAlgorithm::kSplitK);
}

TEST(GemmPlan, AvoidsSplitKForSingleColumnOutputs) {
  using onnx_light_cpu::detail::SelectGemmAlgorithm;

  // A single output column streams best as a vectorized K reduction that is
  // parallelized over its M rows, so split-K's partition and reduction
  // overhead is never worth paying for it.
  EXPECT_EQ(SelectGemmAlgorithm(false, false, 8, 1, 8192, 8, 4), GemmAlgorithm::kSkinnyN);
  EXPECT_EQ(SelectGemmAlgorithm(false, false, 64, 1, 8192, 8, 4), GemmAlgorithm::kSkinnyN);

  // Two or more columns without enough M/N parallelism still use split-K.
  EXPECT_EQ(SelectGemmAlgorithm(false, false, 8, 2, 8192, 8, 4), GemmAlgorithm::kSplitK);

  // When both dimensions are skinny, the narrower output vectorizes over K,
  // while the wider output keeps vectorizing over its columns.
  EXPECT_EQ(SelectGemmAlgorithm(false, false, 4, 1, 256, 8, 4), GemmAlgorithm::kSkinnyN);
  EXPECT_EQ(SelectGemmAlgorithm(false, false, 1, 1, 256, 8, 4), GemmAlgorithm::kSkinnyN);
  EXPECT_EQ(SelectGemmAlgorithm(false, false, 1, 8, 256, 8, 4), GemmAlgorithm::kSkinnyM);
}

TEST(GemmPlan, VectorizedSkinnyNMatchesReference) {
  const auto reference = [](bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                            float alpha, float beta, const std::vector<float> &a,
                            const std::vector<float> &b, const std::vector<float> &c) {
    std::vector<float> y(M * N, 0.0f);
    for (std::size_t m = 0; m < M; ++m) {
      for (std::size_t n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (std::size_t k = 0; k < K; ++k) {
          const float av = trans_a ? a[k * M + m] : a[m * K + k];
          const float bv = trans_b ? b[n * K + k] : b[k * N + n];
          acc += av * bv;
        }
        y[m * N + n] = alpha * acc + (c.empty() ? 0.0f : beta * c[m * N + n]);
      }
    }
    return y;
  };

  const auto check = [&](bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                         float alpha, float beta) {
    std::vector<float> a(M * K);
    std::vector<float> b(K * N);
    std::vector<float> c(M * N);
    for (std::size_t i = 0; i < a.size(); ++i) {
      a[i] = 0.5f + 0.25f * static_cast<float>(i % 7);
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
      b[i] = -1.0f + 0.5f * static_cast<float>(i % 5);
    }
    for (std::size_t i = 0; i < c.size(); ++i) {
      c[i] = 2.0f - 0.5f * static_cast<float>(i % 3);
    }
    const bool has_bias = beta != 0.0f;
    std::vector<float> y(M * N, 0.0f);
    onnx_light_cpu::detail::GemmFloat32Planned<GemmAlgorithm::kSkinnyN>(
        trans_a, trans_b, M, N, K, alpha, a.data(), b.data(), beta, has_bias ? c.data() : nullptr,
        y.data());
    const std::vector<float> expected = reference(trans_a, trans_b, M, N, K, alpha, beta, a, b,
                                                  has_bias ? c : std::vector<float>{});
    ASSERT_EQ(y.size(), expected.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
      EXPECT_NEAR(y[i], expected[i], 1e-3f) << "index=" << i;
    }
  };

  // K = 7 exercises the exact scalar tail after the 4-wide unrolled body; the
  // larger K = 64 keeps every stride combination on the vectorized body. Each
  // shape is run without bias (alpha == 1, beta == 0) and with a scaled bias
  // matrix (alpha != 1, beta != 0) to cover the epilogue.
  for (const std::size_t k : {std::size_t{7}, std::size_t{64}}) {
    check(false, false, 5, 1, k, 1.0f, 0.0f);
    check(false, true, 5, 3, k, 1.0f, 0.0f);
    check(true, false, 5, 3, k, 1.0f, 0.0f);
    check(true, true, 5, 3, k, 1.0f, 0.0f);
    check(false, false, 5, 1, k, 0.5f, 2.0f);
    check(false, true, 5, 3, k, 0.5f, 2.0f);
    check(true, false, 5, 3, k, 0.5f, 2.0f);
    check(true, true, 5, 3, k, 0.5f, 2.0f);
  }
}

TEST(GemmPlan, VectorizedSkinnyMMatchesReference) {
  const auto reference = [](bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                            float alpha, float beta, const std::vector<float> &a,
                            const std::vector<float> &b, const std::vector<float> &c) {
    std::vector<float> y(M * N, 0.0f);
    for (std::size_t m = 0; m < M; ++m) {
      for (std::size_t n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (std::size_t k = 0; k < K; ++k) {
          const float av = trans_a ? a[k * M + m] : a[m * K + k];
          const float bv = trans_b ? b[n * K + k] : b[k * N + n];
          acc += av * bv;
        }
        y[m * N + n] = alpha * acc + (c.empty() ? 0.0f : beta * c[m * N + n]);
      }
    }
    return y;
  };

  const auto check = [&](bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K,
                         float alpha, float beta) {
    std::vector<float> a(M * K);
    std::vector<float> b(K * N);
    std::vector<float> c(M * N);
    for (std::size_t i = 0; i < a.size(); ++i) {
      a[i] = 0.5f + 0.25f * static_cast<float>(i % 7);
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
      b[i] = -1.0f + 0.5f * static_cast<float>(i % 5);
    }
    for (std::size_t i = 0; i < c.size(); ++i) {
      c[i] = 2.0f - 0.5f * static_cast<float>(i % 3);
    }
    const bool has_bias = beta != 0.0f;
    std::vector<float> y(M * N, 0.0f);
    onnx_light_cpu::detail::GemmFloat32Planned<GemmAlgorithm::kSkinnyM>(
        trans_a, trans_b, M, N, K, alpha, a.data(), b.data(), beta, has_bias ? c.data() : nullptr,
        y.data());
    const std::vector<float> expected = reference(trans_a, trans_b, M, N, K, alpha, beta, a, b,
                                                  has_bias ? c : std::vector<float>{});
    ASSERT_EQ(y.size(), expected.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
      EXPECT_NEAR(y[i], expected[i], 1e-3f) << "index=" << i;
    }
  };

  // N = 300 spans several column panels so the panel loop and its short final
  // panel are exercised. K = 7 forces the strided/tail combinations while
  // K = 64 keeps the unit-stride axpy body busy. Each shape runs without bias
  // (alpha == 1, beta == 0) and with a scaled bias (alpha != 1, beta != 0) to
  // cover the epilogue, across every transpose combination and a single row
  // (GEMV) and a short batch of rows.
  for (const std::size_t k : {std::size_t{7}, std::size_t{64}}) {
    for (const std::size_t m : {std::size_t{1}, std::size_t{3}}) {
      check(false, false, m, 300, k, 1.0f, 0.0f);
      check(false, true, m, 300, k, 1.0f, 0.0f);
      check(true, false, m, 300, k, 1.0f, 0.0f);
      check(true, true, m, 300, k, 1.0f, 0.0f);
      check(false, false, m, 300, k, 0.5f, 2.0f);
      check(false, true, m, 300, k, 0.5f, 2.0f);
      check(true, false, m, 300, k, 0.5f, 2.0f);
      check(true, true, m, 300, k, 0.5f, 2.0f);
    }
  }
}

TEST(GemmPlan, PlannedAlgorithmsFallbackOutsideSelectionContract) {
  const std::array<float, 6> transposed_a = {1, 4, 2, 5, 3, 6};
  const std::array<float, 6> b = {1, 2, 3, 4, 5, 6};
  std::array<float, 4> direct_y = {};
  onnx_light_cpu::detail::GemmFloat32Planned<GemmAlgorithm::kDirect>(
      true, false, 2, 2, 3, 1.0f, transposed_a.data(), b.data(), 0.0f, nullptr, direct_y.data());
  ExpectValues({direct_y.begin(), direct_y.end()}, std::array<float, 4>{22, 28, 49, 64});

  const std::vector<float> a(5 * 3, 1.0f);
  const std::vector<float> skinny_b(3 * 2, 1.0f);
  std::vector<float> skinny_y(5 * 2);
  onnx_light_cpu::detail::GemmFloat32Planned<GemmAlgorithm::kSkinnyM>(
      false, false, 5, 2, 3, 1.0f, a.data(), skinny_b.data(), 0.0f, nullptr, skinny_y.data());
  ExpectValues(skinny_y, std::array<float, 10>{3, 3, 3, 3, 3, 3, 3, 3, 3, 3});
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

TEST(GemmPlan, ExecutesRowBroadcastBiasEpilogue) {
  // alpha applies to the product, the epilogue adds beta * bias broadcast over
  // rows and clamps negatives to zero with a fused ReLU.
  const GemmPlan<float> plan(GemmPlanOptions<float>{false, false, 2, 2, 2, 0.5f, 0.0f, {}});
  const std::vector<float> a = {1, 2, 3, 4};
  const std::vector<float> b = {1, 0, 0, 1};
  const std::vector<float> bias = {-10, 1}; // length N, broadcast over the 2 rows
  std::vector<float> y(4);

  onnx_light_cpu::GemmEpilogue<float> epilogue;
  epilogue.bias = bias.data();
  epilogue.bias_layout = onnx_light_cpu::GemmBroadcast::kRow;
  epilogue.beta = 2.0f;
  epilogue.activation = onnx_light_cpu::GemmActivation::kRelu;

  plan.Execute(a.data(), b.data(), epilogue, y.data());

  // 0.5*[[1,2],[3,4]] + 2*[[-10,1],[-10,1]] = [[-19.5,3],[-18.5,4]] -> ReLU
  ExpectValues(y, std::array<float, 4>{0, 3, 0, 4});
}

TEST(GemmPlan, ExecutesMatrixBiasEpilogueMatchesFullMatrixExecute) {
  // The full-matrix bias epilogue must match the fused (a, b, c, y) overload.
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  const std::vector<float> b = {1, 2, 3, 4, 5, 6};
  const std::vector<float> c = {1, 1, 1, 1};

  const GemmPlan<float> fused(GemmPlanOptions<float>{false, false, 2, 2, 3, 0.5f, 2.0f, {}});
  std::vector<float> expected(4);
  fused.Execute(a.data(), b.data(), c.data(), expected.data());

  const GemmPlan<float> plan(GemmPlanOptions<float>{false, false, 2, 2, 3, 0.5f, 0.0f, {}});
  onnx_light_cpu::GemmEpilogue<float> epilogue;
  epilogue.bias = c.data();
  epilogue.bias_layout = onnx_light_cpu::GemmBroadcast::kMatrix;
  epilogue.beta = 2.0f;
  std::vector<float> y(4);
  plan.Execute(a.data(), b.data(), epilogue, y.data());

  ExpectValues(y, std::span<const float>(expected));
}

TEST(GemmPlan, ExecutesConstantBEpilogue) {
  std::vector<float> b = {1, 2, 3, 4, 5, 6};
  const GemmPlan<float> plan(GemmPlanOptions<float>{false, false, 2, 2, 3, 1.0f, 0.0f, b});
  b.assign(b.size(), 0.0f);
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  const std::vector<float> bias = {5};
  std::vector<float> y(4);

  onnx_light_cpu::GemmEpilogue<float> epilogue;
  epilogue.bias = bias.data();
  epilogue.bias_layout = onnx_light_cpu::GemmBroadcast::kScalar;
  epilogue.beta = 1.0f;

  plan.Execute(a.data(), epilogue, y.data());

  ExpectValues(y, std::array<float, 4>{27, 33, 54, 69});
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

TEST(MatMulPlan, PromotesRankOneInputs) {
  const std::array<std::size_t, 1> vector_shape = {3};
  const MatMulPlan<float> dot_plan(vector_shape, vector_shape);
  const std::array<float, 3> a = {1, 2, 3};
  const std::array<float, 3> b = {4, 5, 6};
  float dot = 0;

  dot_plan.Execute(a.data(), b.data(), &dot);

  EXPECT_TRUE(dot_plan.output_shape().empty());
  EXPECT_FLOAT_EQ(dot, 32);

  const std::array<std::size_t, 2> matrix_shape = {3, 2};
  const MatMulPlan<float> vector_matrix_plan(vector_shape, matrix_shape);
  const std::array<float, 6> matrix = {1, 2, 3, 4, 5, 6};
  std::array<float, 2> vector_matrix = {};
  vector_matrix_plan.Execute(a.data(), matrix.data(), vector_matrix.data());
  ExpectShape(vector_matrix_plan.output_shape(), std::array<std::size_t, 1>{2});
  EXPECT_FLOAT_EQ(vector_matrix[0], 22);
  EXPECT_FLOAT_EQ(vector_matrix[1], 28);

  const std::array<std::size_t, 2> left_shape = {2, 3};
  const MatMulPlan<float> matrix_vector_plan(left_shape, vector_shape);
  std::array<float, 2> matrix_vector = {};
  matrix_vector_plan.Execute(matrix.data(), b.data(), matrix_vector.data());
  ExpectShape(matrix_vector_plan.output_shape(), std::array<std::size_t, 1>{2});
  EXPECT_FLOAT_EQ(matrix_vector[0], 32);
  EXPECT_FLOAT_EQ(matrix_vector[1], 77);
}

TEST(MatMulPlan, BroadcastsBatchesFromEitherInput) {
  const std::array<std::size_t, 4> a_shape = {2, 1, 2, 2};
  const std::array<std::size_t, 4> b_shape = {1, 3, 2, 1};
  const MatMulPlan<float> plan(a_shape, b_shape);
  const std::array<float, 8> a = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<float, 6> b = {1, 1, 2, 1, 1, 3};
  std::array<float, 12> y = {};

  plan.Execute(a.data(), b.data(), y.data());

  EXPECT_EQ(plan.batch_count(), 6u);
  ExpectShape(plan.output_shape(), std::array<std::size_t, 4>{2, 3, 2, 1});
  ExpectValues({y.begin(), y.end()},
               std::array<float, 12>{3, 7, 4, 10, 7, 15, 11, 15, 16, 22, 23, 31});
}

TEST(MatMulPlan, BatchSchedulingTakesPriorityOverSplitK) {
  constexpr std::size_t batch_count = 128;
  constexpr std::size_t m = 2;
  constexpr std::size_t n = 2;
  constexpr std::size_t k = 4096;
  const std::array<std::size_t, 3> a_shape = {batch_count, m, k};
  const std::array<std::size_t, 2> b_shape = {k, n};
  const MatMulPlan<float> plan(a_shape, b_shape);
  std::vector<float> a(batch_count * m * k, 1.0f);
  std::vector<float> b(k * n, 1.0f);
  std::vector<float> y(batch_count * m * n);

  plan.Execute(a.data(), b.data(), y.data());

  EXPECT_EQ(plan.gemm_plan().algorithm(), GemmAlgorithm::kSplitK);
  for (float value : y) {
    EXPECT_FLOAT_EQ(value, static_cast<float>(k));
  }
}

TEST(MatMulPlan, AppliesMatrixTransposeBeforeBatchMultiplication) {
  const std::array<std::size_t, 3> a_shape = {2, 3, 2};
  const std::array<std::size_t, 2> b_shape = {4, 3};
  const MatMulPlan<double> plan(a_shape, b_shape, true, true);
  const std::array<double, 12> a = {1, 4, 2, 5, 3, 6, 1, 0, 0, 1, 1, 1};
  const std::array<double, 12> b = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::array<double, 16> y = {};

  plan.Execute(a.data(), b.data(), y.data());

  ExpectShape(plan.output_shape(), std::array<std::size_t, 3>{2, 2, 4});
  const std::array<double, 16> expected = {14, 32, 50, 68, 32, 77, 122, 167,
                                           4,  10, 16, 22, 5,  11, 17,  23};
  for (std::size_t index = 0; index < y.size(); ++index) {
    EXPECT_DOUBLE_EQ(y[index], expected[index]);
  }
}

TEST(MatMulPlan, OwnsBroadcastConstantB) {
  const std::array<std::size_t, 3> a_shape = {2, 1, 2};
  const std::array<std::size_t, 3> b_shape = {2, 2, 1};
  std::vector<float> b = {2, 3, 4, 5};
  const MatMulPlan<float> plan(a_shape, b_shape, false, false, b);
  b.assign(b.size(), 0);
  const std::array<float, 4> a = {1, 2, 3, 4};
  std::array<float, 2> y = {};

  plan.Execute(a.data(), y.data());

  EXPECT_TRUE(plan.has_constant_b());
  ExpectValues({y.begin(), y.end()}, std::array<float, 2>{8, 32});
}

TEST(MatMulPlan, HandlesEmptyBatchMatrixAndReductionDimensions) {
  const std::array<std::size_t, 3> empty_batch_a = {0, 2, 3};
  const std::array<std::size_t, 3> empty_batch_b = {1, 3, 4};
  const MatMulPlan<float> empty_batch(empty_batch_a, empty_batch_b);
  ExpectShape(empty_batch.output_shape(), std::array<std::size_t, 3>{0, 2, 4});
  EXPECT_NO_THROW(empty_batch.Execute(nullptr, nullptr, nullptr));

  const std::array<std::size_t, 2> empty_rows_a = {0, 3};
  const std::array<std::size_t, 2> regular_b = {3, 2};
  const MatMulPlan<float> empty_rows(empty_rows_a, regular_b);
  EXPECT_NO_THROW(empty_rows.Execute(nullptr, nullptr, nullptr));

  const std::array<std::size_t, 2> regular_a = {2, 3};
  const std::array<std::size_t, 2> empty_columns_b = {3, 0};
  const MatMulPlan<float> empty_columns(regular_a, empty_columns_b);
  EXPECT_NO_THROW(empty_columns.Execute(nullptr, nullptr));

  const std::array<std::size_t, 2> empty_k_a = {2, 0};
  const std::array<std::size_t, 2> empty_k_b = {0, 3};
  const MatMulPlan<float> empty_k(empty_k_a, empty_k_b);
  std::array<float, 6> y = {1, 1, 1, 1, 1, 1};
  empty_k.Execute(nullptr, nullptr, y.data());
  ExpectValues({y.begin(), y.end()}, std::array<float, 6>{0, 0, 0, 0, 0, 0});
}

TEST(MatMulPlan, RejectsInvalidShapes) {
  const std::array<std::size_t, 1> scalar_shape = {0};
  const std::array<std::size_t, 2> matrix_shape = {2, 3};
  EXPECT_THROW((MatMulPlan<float>({}, matrix_shape)), std::invalid_argument);
  EXPECT_THROW((MatMulPlan<float>(matrix_shape, std::array<std::size_t, 2>{4, 2})),
               std::invalid_argument);
  EXPECT_THROW(
      (MatMulPlan<float>(std::array<std::size_t, 3>{2, 2, 3}, std::array<std::size_t, 3>{3, 3, 2})),
      std::invalid_argument);
  EXPECT_THROW((MatMulPlan<float>(scalar_shape, scalar_shape, true)), std::invalid_argument);

  const std::array<float, 1> wrong_constant = {1};
  EXPECT_THROW((MatMulPlan<float>(matrix_shape, std::array<std::size_t, 2>{3, 2}, false, false,
                                  wrong_constant)),
               std::invalid_argument);
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

TEST(GroupedGemm, ValidatesAllProblemsBeforeExecution) {
  const GemmPlan<float> plan(GemmPlanOptions<float>{false, false, 1, 1, 1});
  const float a = 2.0f;
  const float b = 3.0f;
  float y = -1.0f;
  const std::array<GroupedGemmProblem<float>, 2> problems = {
      GroupedGemmProblem<float>{&plan, &a, &b, nullptr, &y},
      GroupedGemmProblem<float>{nullptr, &a, &b, nullptr, &y},
  };

  EXPECT_THROW(GroupedGemm<float>(problems), std::invalid_argument);
  EXPECT_FLOAT_EQ(y, -1.0f);
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

TEST(GemmHalfPlan, ExecutesBFloat16AndExposesSelection) {
  const GemmHalfPlanOptions options{true, false, false, 2, 2, 3, 0.5f};
  const GemmHalfPlan plan(options);

  const std::vector<float> a_f = {1, 2, 3, 4, 5, 6};
  const std::vector<float> b_f = {1, 2, 3, 4, 5, 6};
  std::vector<std::uint16_t> a(a_f.size());
  std::vector<std::uint16_t> b(b_f.size());
  for (std::size_t i = 0; i < a_f.size(); ++i) {
    a[i] = EncodeBFloat16(a_f[i]);
  }
  for (std::size_t i = 0; i < b_f.size(); ++i) {
    b[i] = EncodeBFloat16(b_f[i]);
  }

  std::vector<std::uint16_t> converted(4, 0);
  std::vector<float> y(4, 0.0f);
  onnx_light_cpu::GemmEpilogue<float> epilogue;
  epilogue.output_conversion = onnx_light_cpu::GemmOutputConversion::kBFloat16;
  epilogue.converted_output = converted.data();

  plan.Execute(a.data(), b.data(), epilogue, y.data());

  // 0.5 * ([1 2 3;4 5 6] @ [1 2;3 4;5 6]) = 0.5 * [22 28;49 64].
  const std::array<float, 4> expected = {11, 14, 24.5f, 32};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(DecodeBFloat16(converted[i]), expected[i], 0.25f) << "index=" << i;
  }

  EXPECT_TRUE(plan.is_bfloat16());
  EXPECT_EQ(plan.m(), 2u);
  EXPECT_EQ(plan.n(), 2u);
  EXPECT_EQ(plan.k(), 3u);
  EXPECT_FLOAT_EQ(plan.alpha(), 0.5f);
  EXPECT_GE(plan.blocking().kc, 64u);
  EXPECT_GE(plan.useful_threads(), 1u);
}

} // namespace
