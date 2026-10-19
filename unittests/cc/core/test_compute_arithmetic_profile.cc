// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/compute_arithmetic_profile.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace {

using ml_cpu_compute::ArmSimdLevel;
using ml_cpu_compute::ComputeDotProductOperationAccounting;
using ml_cpu_compute::ComputeFmaOperationAccounting;
using ml_cpu_compute::ComputeImplementation;
using ml_cpu_compute::ComputeImplementationName;
using ml_cpu_compute::ComputeOperationAccounting;
using ml_cpu_compute::ComputeParticipantPolicy;
using ml_cpu_compute::ComputeProfileOptions;
using ml_cpu_compute::ComputeProfileTimerName;
using ml_cpu_compute::ComputeProfileUnavailableReason;
using ml_cpu_compute::ComputeThroughputResult;
using ml_cpu_compute::DataType;
using ml_cpu_compute::MeasureComputeArithmeticThroughput;
using ml_cpu_compute::SimdLevel;
using ml_cpu_compute::detail::ComputeDispatchDecision;
using ml_cpu_compute::detail::ComputeDispatchInputs;
using ml_cpu_compute::detail::SelectComputeImplementation;

// ---------------------------------------------------------------------------
// Operation accounting: pure arithmetic, no timing.
// ---------------------------------------------------------------------------

TEST(ComputeArithmeticProfile, FmaAccountingCountsTwoOpsPerFma) {
  const ml_cpu_compute::ComputeOperationAccounting accounting =
      ml_cpu_compute::ComputeFmaOperationAccounting(8, 64);
  EXPECT_EQ(accounting.fma_count_per_pass, 8u * 64u);
  EXPECT_EQ(accounting.operations_per_pass, 8u * 64u * 2u);
  EXPECT_EQ(accounting.dot_product_length, 0u);
}

TEST(ComputeArithmeticProfile, DotProductAccountingCountsTwoOpsPerElementPair) {
  const ml_cpu_compute::ComputeOperationAccounting accounting =
      ml_cpu_compute::ComputeDotProductOperationAccounting(16, 64, 4);
  const std::uint64_t total_pairs = 16u * 64u * 4u;
  EXPECT_EQ(accounting.fma_count_per_pass, total_pairs);
  EXPECT_EQ(accounting.dot_product_length, 4u);
  EXPECT_EQ(accounting.operations_per_pass, total_pairs * 2u);
}

TEST(ComputeArithmeticProfile, ImplementationNamesAreStable) {
  EXPECT_STREQ(
      ml_cpu_compute::ComputeImplementationName(ml_cpu_compute::ComputeImplementation::kScalar),
      "scalar");
  EXPECT_STREQ(
      ml_cpu_compute::ComputeImplementationName(ml_cpu_compute::ComputeImplementation::kSSE2),
      "SSE2");
  EXPECT_STREQ(
      ml_cpu_compute::ComputeImplementationName(ml_cpu_compute::ComputeImplementation::kAVX2),
      "AVX2");
  EXPECT_STREQ(
      ml_cpu_compute::ComputeImplementationName(ml_cpu_compute::ComputeImplementation::kAVX512),
      "AVX-512");
  EXPECT_STREQ(
      ml_cpu_compute::ComputeImplementationName(ml_cpu_compute::ComputeImplementation::kNeon),
      "NEON");
}

TEST(ComputeArithmeticProfile, TimerNameIsStable) {
  EXPECT_STREQ(ml_cpu_compute::ComputeProfileTimerName(), "std::chrono::steady_clock");
}

// ---------------------------------------------------------------------------
// Dispatch selection: injected inputs, no hardware access, no execution of
// any unsupported instruction. Covers native and fallback selection for
// every element type.
// ---------------------------------------------------------------------------

TEST(ComputeArithmeticProfile, Float32IsAlwaysAvailableAndFallsBackToScalar) {
  ComputeDispatchInputs inputs;
  const ComputeDispatchDecision decision = SelectComputeImplementation(DataType::FLOAT, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ml_cpu_compute::ComputeImplementation::kScalar);
}

TEST(ComputeArithmeticProfile, Float64IsAlwaysAvailableAndFallsBackToScalar) {
  ComputeDispatchInputs inputs;
  const ComputeDispatchDecision decision = SelectComputeImplementation(DataType::DOUBLE, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ml_cpu_compute::ComputeImplementation::kScalar);
}

TEST(ComputeArithmeticProfile, Float32PrefersSse2OverScalarWhenDetected) {
  ComputeDispatchInputs inputs;
  inputs.simd_level = ml_cpu_compute::SimdLevel::kSSE2;
  const ComputeDispatchDecision decision = SelectComputeImplementation(DataType::FLOAT, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ml_cpu_compute::ComputeImplementation::kSSE2);
}

TEST(ComputeArithmeticProfile, Float32PrefersAvx2OverSse2WhenFmaDetected) {
  ComputeDispatchInputs inputs;
  inputs.simd_level = ml_cpu_compute::SimdLevel::kAVX2;
  inputs.has_fma = true;
  const ComputeDispatchDecision decision = SelectComputeImplementation(DataType::FLOAT, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ml_cpu_compute::ComputeImplementation::kAVX2);
}

TEST(ComputeArithmeticProfile, Float32DoesNotSelectAvx2WithoutFma) {
  ComputeDispatchInputs inputs;
  inputs.simd_level = ml_cpu_compute::SimdLevel::kAVX2;
  inputs.has_fma = false;
  const ComputeDispatchDecision decision = SelectComputeImplementation(DataType::FLOAT, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ml_cpu_compute::ComputeImplementation::kSSE2);
}

TEST(ComputeArithmeticProfile, Float32SelectsAvx512OnlyWhenCompiledAndDetected) {
  ComputeDispatchInputs inputs;
  inputs.simd_level = ml_cpu_compute::SimdLevel::kAVX512;
  inputs.has_fma = true;
  inputs.avx512_compiled = false;
  const ComputeDispatchDecision without_compiled =
      SelectComputeImplementation(DataType::FLOAT, inputs);
  EXPECT_TRUE(without_compiled.available);
  EXPECT_EQ(without_compiled.implementation, ml_cpu_compute::ComputeImplementation::kAVX2);

  inputs.avx512_compiled = true;
  const ComputeDispatchDecision with_compiled =
      SelectComputeImplementation(DataType::FLOAT, inputs);
  EXPECT_TRUE(with_compiled.available);
  EXPECT_EQ(with_compiled.implementation, ml_cpu_compute::ComputeImplementation::kAVX512);
}

TEST(ComputeArithmeticProfile, Float32SelectsNeonOnArmRegardlessOfX86Fields) {
  ComputeDispatchInputs inputs;
  inputs.arm_simd_level = ml_cpu_compute::ArmSimdLevel::kNeon;
  const ComputeDispatchDecision decision = SelectComputeImplementation(DataType::FLOAT, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ml_cpu_compute::ComputeImplementation::kNeon);
}

TEST(ComputeArithmeticProfile, Float16IsUnavailableWithoutNativeCompiledAndDetectedPath) {
  ComputeDispatchInputs inputs;
  inputs.avx512fp16_compiled = false;
  inputs.avx512fp16_runtime = true;
  EXPECT_FALSE(SelectComputeImplementation(DataType::FLOAT16, inputs).available);

  inputs.avx512fp16_compiled = true;
  inputs.avx512fp16_runtime = false;
  EXPECT_FALSE(SelectComputeImplementation(DataType::FLOAT16, inputs).available);

  inputs.avx512fp16_compiled = true;
  inputs.avx512fp16_runtime = true;
  const ComputeDispatchDecision decision = SelectComputeImplementation(DataType::FLOAT16, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ml_cpu_compute::ComputeImplementation::kAVX512);
}

TEST(ComputeArithmeticProfile, BFloat16IsUnavailableWithoutNativeCompiledAndDetectedPath) {
  ComputeDispatchInputs inputs;
  inputs.avx512bf16_compiled = false;
  inputs.avx512bf16_runtime = true;
  EXPECT_FALSE(SelectComputeImplementation(DataType::BFLOAT16, inputs).available);

  inputs.avx512bf16_compiled = true;
  inputs.avx512bf16_runtime = false;
  EXPECT_FALSE(SelectComputeImplementation(DataType::BFLOAT16, inputs).available);

  inputs.avx512bf16_compiled = true;
  inputs.avx512bf16_runtime = true;
  const ComputeDispatchDecision decision = SelectComputeImplementation(DataType::BFLOAT16, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ml_cpu_compute::ComputeImplementation::kAVX512);
}

TEST(ComputeArithmeticProfile, Int8PrefersAvx512VnniOverNeonWhenBothCompiledAndDetected) {
  ComputeDispatchInputs inputs;
  inputs.avx512vnni_compiled = true;
  inputs.avx512vnni_runtime = true;
  inputs.neon_dotprod_compiled = true;
  inputs.neon_dotprod_runtime = true;
  const ComputeDispatchDecision decision = SelectComputeImplementation(DataType::INT8, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ml_cpu_compute::ComputeImplementation::kAVX512);
}

TEST(ComputeArithmeticProfile, Int8FallsBackToNeonWhenVnniNotCompiled) {
  ComputeDispatchInputs inputs;
  inputs.avx512vnni_compiled = false;
  inputs.avx512vnni_runtime = true;
  inputs.neon_dotprod_compiled = true;
  inputs.neon_dotprod_runtime = true;
  const ComputeDispatchDecision decision = SelectComputeImplementation(DataType::INT8, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ml_cpu_compute::ComputeImplementation::kNeon);
}

TEST(ComputeArithmeticProfile, Int8IsUnavailableWithoutAnyNativeCompiledAndDetectedPath) {
  ComputeDispatchInputs inputs;
  inputs.avx512vnni_compiled = true;
  inputs.avx512vnni_runtime = false;
  inputs.neon_dotprod_compiled = true;
  inputs.neon_dotprod_runtime = false;
  EXPECT_FALSE(SelectComputeImplementation(DataType::INT8, inputs).available);

  inputs.avx512vnni_compiled = false;
  inputs.neon_dotprod_compiled = false;
  EXPECT_FALSE(SelectComputeImplementation(DataType::INT8, inputs).available);
}

// ---------------------------------------------------------------------------
// Invalid options: fails before any allocation or timing.
// ---------------------------------------------------------------------------

TEST(ComputeArithmeticProfile, ZeroRepeatsIsInvalidOptions) {
  ml_cpu_compute::ComputeProfileOptions options;
  options.repeats = 0;
  const ml_cpu_compute::ComputeThroughputResult result =
      ml_cpu_compute::MeasureComputeArithmeticThroughput(
          DataType::FLOAT, ml_cpu_compute::ComputeParticipantPolicy::kSingle, options);
  EXPECT_FALSE(result.available);
  EXPECT_EQ(result.unavailable_reason,
            ml_cpu_compute::ComputeProfileUnavailableReason::kInvalidOptions);
  EXPECT_FALSE(result.diagnostic.empty());
}

TEST(ComputeArithmeticProfile, NonPositiveMinimumDurationIsInvalidOptions) {
  ml_cpu_compute::ComputeProfileOptions options;
  options.minimum_duration_ms = 0.0;
  const ml_cpu_compute::ComputeThroughputResult result =
      ml_cpu_compute::MeasureComputeArithmeticThroughput(
          DataType::DOUBLE, ml_cpu_compute::ComputeParticipantPolicy::kSingle, options);
  EXPECT_FALSE(result.available);
  EXPECT_EQ(result.unavailable_reason,
            ml_cpu_compute::ComputeProfileUnavailableReason::kInvalidOptions);
}

// ---------------------------------------------------------------------------
// Real measurements: FP32/FP64 must be available on every platform. Kept
// short (few repeats, small minimum duration) so the suite stays fast.
// ---------------------------------------------------------------------------

ml_cpu_compute::ComputeProfileOptions FastOptions() {
  ml_cpu_compute::ComputeProfileOptions options;
  options.repeats = 3;
  options.minimum_duration_ms = 2.0;
  return options;
}

void ExpectFiniteSuccessfulResult(const ml_cpu_compute::ComputeThroughputResult &result) {
  ASSERT_TRUE(result.available);
  EXPECT_EQ(result.unavailable_reason, ml_cpu_compute::ComputeProfileUnavailableReason::kNone);
  EXPECT_FALSE(result.implementation_name.empty());
  EXPECT_FALSE(result.timer_name.empty());
  EXPECT_GT(result.operations_per_pass_per_participant, 0u);
  EXPECT_GT(result.participant_count, 0u);
  ASSERT_FALSE(result.raw_gops_samples.empty());
  for (double sample : result.raw_gops_samples) {
    EXPECT_TRUE(std::isfinite(sample));
    EXPECT_GT(sample, 0.0);
  }
  EXPECT_TRUE(std::isfinite(result.median_gops));
  EXPECT_GT(result.median_gops, 0.0);
  EXPECT_TRUE(std::isfinite(result.dispersion_gops));
  EXPECT_GE(result.dispersion_gops, 0.0);
}

TEST(ComputeArithmeticProfile, Float32SingleParticipantIsFiniteAndPositive) {
  const ml_cpu_compute::ComputeThroughputResult result =
      ml_cpu_compute::MeasureComputeArithmeticThroughput(
          DataType::FLOAT, ml_cpu_compute::ComputeParticipantPolicy::kSingle, FastOptions());
  ExpectFiniteSuccessfulResult(result);
  EXPECT_EQ(result.participant_count, 1u);
  EXPECT_EQ(result.dot_product_length, 0u);
}

TEST(ComputeArithmeticProfile, Float64SingleParticipantIsFiniteAndPositive) {
  const ml_cpu_compute::ComputeThroughputResult result =
      ml_cpu_compute::MeasureComputeArithmeticThroughput(
          DataType::DOUBLE, ml_cpu_compute::ComputeParticipantPolicy::kSingle, FastOptions());
  ExpectFiniteSuccessfulResult(result);
  EXPECT_EQ(result.participant_count, 1u);
}

TEST(ComputeArithmeticProfile, Float32PhysicalParticipantsIsFiniteAndPositive) {
  const ml_cpu_compute::ComputeThroughputResult result =
      ml_cpu_compute::MeasureComputeArithmeticThroughput(
          DataType::FLOAT, ml_cpu_compute::ComputeParticipantPolicy::kPhysical, FastOptions());
  ExpectFiniteSuccessfulResult(result);
}

TEST(ComputeArithmeticProfile, Float64PhysicalParticipantsIsFiniteAndPositive) {
  const ml_cpu_compute::ComputeThroughputResult result =
      ml_cpu_compute::MeasureComputeArithmeticThroughput(
          DataType::DOUBLE, ml_cpu_compute::ComputeParticipantPolicy::kPhysical, FastOptions());
  ExpectFiniteSuccessfulResult(result);
}

// ---------------------------------------------------------------------------
// Low-precision types: only asserted available/finite when this build and
// host actually have a native path; otherwise the measurement must be
// explicitly absent, never emulated.
// ---------------------------------------------------------------------------

TEST(ComputeArithmeticProfile, Float16IsFiniteWhenAvailableOtherwiseExplicitlyAbsent) {
  const ml_cpu_compute::ComputeThroughputResult result =
      ml_cpu_compute::MeasureComputeArithmeticThroughput(
          DataType::FLOAT16, ml_cpu_compute::ComputeParticipantPolicy::kSingle, FastOptions());
  if (result.available) {
    ExpectFiniteSuccessfulResult(result);
  } else {
    EXPECT_EQ(result.unavailable_reason,
              ml_cpu_compute::ComputeProfileUnavailableReason::kUnsupportedNativePath);
    EXPECT_FALSE(result.diagnostic.empty());
  }
}

TEST(ComputeArithmeticProfile, BFloat16IsFiniteWhenAvailableOtherwiseExplicitlyAbsent) {
  const ml_cpu_compute::ComputeThroughputResult result =
      ml_cpu_compute::MeasureComputeArithmeticThroughput(
          DataType::BFLOAT16, ml_cpu_compute::ComputeParticipantPolicy::kSingle, FastOptions());
  if (result.available) {
    ExpectFiniteSuccessfulResult(result);
    EXPECT_GT(result.dot_product_length, 0u);
  } else {
    EXPECT_EQ(result.unavailable_reason,
              ml_cpu_compute::ComputeProfileUnavailableReason::kUnsupportedNativePath);
    EXPECT_FALSE(result.diagnostic.empty());
  }
}

TEST(ComputeArithmeticProfile, Int8IsFiniteWhenAvailableOtherwiseExplicitlyAbsent) {
  const ml_cpu_compute::ComputeThroughputResult result =
      ml_cpu_compute::MeasureComputeArithmeticThroughput(
          DataType::INT8, ml_cpu_compute::ComputeParticipantPolicy::kSingle, FastOptions());
  if (result.available) {
    ExpectFiniteSuccessfulResult(result);
    EXPECT_GT(result.dot_product_length, 0u);
  } else {
    EXPECT_EQ(result.unavailable_reason,
              ml_cpu_compute::ComputeProfileUnavailableReason::kUnsupportedNativePath);
    EXPECT_FALSE(result.diagnostic.empty());
  }
}

} // namespace
