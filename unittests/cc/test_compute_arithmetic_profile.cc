// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/compute_arithmetic_profile.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace {

using onnx_light_cpu::ArmSimdLevel;
using onnx_light_cpu::ComputeDotProductOperationAccounting;
using onnx_light_cpu::ComputeElementType;
using onnx_light_cpu::ComputeFmaOperationAccounting;
using onnx_light_cpu::ComputeImplementation;
using onnx_light_cpu::ComputeImplementationName;
using onnx_light_cpu::ComputeOperationAccounting;
using onnx_light_cpu::ComputeParticipantPolicy;
using onnx_light_cpu::ComputeProfileOptions;
using onnx_light_cpu::ComputeProfileTimerName;
using onnx_light_cpu::ComputeProfileUnavailableReason;
using onnx_light_cpu::ComputeThroughputResult;
using onnx_light_cpu::MeasureComputeArithmeticThroughput;
using onnx_light_cpu::SimdLevel;
using onnx_light_cpu::detail::ComputeDispatchDecision;
using onnx_light_cpu::detail::ComputeDispatchInputs;
using onnx_light_cpu::detail::SelectComputeImplementation;

// ---------------------------------------------------------------------------
// Operation accounting: pure arithmetic, no timing.
// ---------------------------------------------------------------------------

TEST(ComputeArithmeticProfile, FmaAccountingCountsTwoOpsPerFma) {
  const ComputeOperationAccounting accounting = ComputeFmaOperationAccounting(8, 64);
  EXPECT_EQ(accounting.fma_count_per_pass, 8u * 64u);
  EXPECT_EQ(accounting.operations_per_pass, 8u * 64u * 2u);
  EXPECT_EQ(accounting.dot_product_length, 0u);
}

TEST(ComputeArithmeticProfile, DotProductAccountingCountsTwoOpsPerElementPair) {
  const ComputeOperationAccounting accounting = ComputeDotProductOperationAccounting(16, 64, 4);
  const std::uint64_t total_pairs = 16u * 64u * 4u;
  EXPECT_EQ(accounting.fma_count_per_pass, total_pairs);
  EXPECT_EQ(accounting.dot_product_length, 4u);
  EXPECT_EQ(accounting.operations_per_pass, total_pairs * 2u);
}

TEST(ComputeArithmeticProfile, ImplementationNamesAreStable) {
  EXPECT_STREQ(ComputeImplementationName(ComputeImplementation::kScalar), "scalar");
  EXPECT_STREQ(ComputeImplementationName(ComputeImplementation::kSSE2), "SSE2");
  EXPECT_STREQ(ComputeImplementationName(ComputeImplementation::kAVX2), "AVX2");
  EXPECT_STREQ(ComputeImplementationName(ComputeImplementation::kAVX512), "AVX-512");
  EXPECT_STREQ(ComputeImplementationName(ComputeImplementation::kNeon), "NEON");
}

TEST(ComputeArithmeticProfile, TimerNameIsStable) {
  EXPECT_STREQ(ComputeProfileTimerName(), "std::chrono::steady_clock");
}

// ---------------------------------------------------------------------------
// Dispatch selection: injected inputs, no hardware access, no execution of
// any unsupported instruction. Covers native and fallback selection for
// every element type.
// ---------------------------------------------------------------------------

TEST(ComputeArithmeticProfile, Float32IsAlwaysAvailableAndFallsBackToScalar) {
  ComputeDispatchInputs inputs;
  const ComputeDispatchDecision decision =
      SelectComputeImplementation(ComputeElementType::kFloat32, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ComputeImplementation::kScalar);
}

TEST(ComputeArithmeticProfile, Float64IsAlwaysAvailableAndFallsBackToScalar) {
  ComputeDispatchInputs inputs;
  const ComputeDispatchDecision decision =
      SelectComputeImplementation(ComputeElementType::kFloat64, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ComputeImplementation::kScalar);
}

TEST(ComputeArithmeticProfile, Float32PrefersSse2OverScalarWhenDetected) {
  ComputeDispatchInputs inputs;
  inputs.simd_level = SimdLevel::kSSE2;
  const ComputeDispatchDecision decision =
      SelectComputeImplementation(ComputeElementType::kFloat32, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ComputeImplementation::kSSE2);
}

TEST(ComputeArithmeticProfile, Float32PrefersAvx2OverSse2WhenFmaDetected) {
  ComputeDispatchInputs inputs;
  inputs.simd_level = SimdLevel::kAVX2;
  inputs.has_fma = true;
  const ComputeDispatchDecision decision =
      SelectComputeImplementation(ComputeElementType::kFloat32, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ComputeImplementation::kAVX2);
}

TEST(ComputeArithmeticProfile, Float32DoesNotSelectAvx2WithoutFma) {
  ComputeDispatchInputs inputs;
  inputs.simd_level = SimdLevel::kAVX2;
  inputs.has_fma = false;
  const ComputeDispatchDecision decision =
      SelectComputeImplementation(ComputeElementType::kFloat32, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ComputeImplementation::kSSE2);
}

TEST(ComputeArithmeticProfile, Float32SelectsAvx512OnlyWhenCompiledAndDetected) {
  ComputeDispatchInputs inputs;
  inputs.simd_level = SimdLevel::kAVX512;
  inputs.has_fma = true;
  inputs.avx512_compiled = false;
  const ComputeDispatchDecision without_compiled =
      SelectComputeImplementation(ComputeElementType::kFloat32, inputs);
  EXPECT_TRUE(without_compiled.available);
  EXPECT_EQ(without_compiled.implementation, ComputeImplementation::kAVX2);

  inputs.avx512_compiled = true;
  const ComputeDispatchDecision with_compiled =
      SelectComputeImplementation(ComputeElementType::kFloat32, inputs);
  EXPECT_TRUE(with_compiled.available);
  EXPECT_EQ(with_compiled.implementation, ComputeImplementation::kAVX512);
}

TEST(ComputeArithmeticProfile, Float32SelectsNeonOnArmRegardlessOfX86Fields) {
  ComputeDispatchInputs inputs;
  inputs.arm_simd_level = ArmSimdLevel::kNeon;
  const ComputeDispatchDecision decision =
      SelectComputeImplementation(ComputeElementType::kFloat32, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ComputeImplementation::kNeon);
}

TEST(ComputeArithmeticProfile, Float16IsUnavailableWithoutNativeCompiledAndDetectedPath) {
  ComputeDispatchInputs inputs;
  inputs.avx512fp16_compiled = false;
  inputs.avx512fp16_runtime = true;
  EXPECT_FALSE(SelectComputeImplementation(ComputeElementType::kFloat16, inputs).available);

  inputs.avx512fp16_compiled = true;
  inputs.avx512fp16_runtime = false;
  EXPECT_FALSE(SelectComputeImplementation(ComputeElementType::kFloat16, inputs).available);

  inputs.avx512fp16_compiled = true;
  inputs.avx512fp16_runtime = true;
  const ComputeDispatchDecision decision =
      SelectComputeImplementation(ComputeElementType::kFloat16, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ComputeImplementation::kAVX512);
}

TEST(ComputeArithmeticProfile, BFloat16IsUnavailableWithoutNativeCompiledAndDetectedPath) {
  ComputeDispatchInputs inputs;
  inputs.avx512bf16_compiled = false;
  inputs.avx512bf16_runtime = true;
  EXPECT_FALSE(SelectComputeImplementation(ComputeElementType::kBFloat16, inputs).available);

  inputs.avx512bf16_compiled = true;
  inputs.avx512bf16_runtime = false;
  EXPECT_FALSE(SelectComputeImplementation(ComputeElementType::kBFloat16, inputs).available);

  inputs.avx512bf16_compiled = true;
  inputs.avx512bf16_runtime = true;
  const ComputeDispatchDecision decision =
      SelectComputeImplementation(ComputeElementType::kBFloat16, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ComputeImplementation::kAVX512);
}

TEST(ComputeArithmeticProfile, Int8PrefersAvx512VnniOverNeonWhenBothCompiledAndDetected) {
  ComputeDispatchInputs inputs;
  inputs.avx512vnni_compiled = true;
  inputs.avx512vnni_runtime = true;
  inputs.neon_dotprod_compiled = true;
  inputs.neon_dotprod_runtime = true;
  const ComputeDispatchDecision decision =
      SelectComputeImplementation(ComputeElementType::kInt8, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ComputeImplementation::kAVX512);
}

TEST(ComputeArithmeticProfile, Int8FallsBackToNeonWhenVnniNotCompiled) {
  ComputeDispatchInputs inputs;
  inputs.avx512vnni_compiled = false;
  inputs.avx512vnni_runtime = true;
  inputs.neon_dotprod_compiled = true;
  inputs.neon_dotprod_runtime = true;
  const ComputeDispatchDecision decision =
      SelectComputeImplementation(ComputeElementType::kInt8, inputs);
  EXPECT_TRUE(decision.available);
  EXPECT_EQ(decision.implementation, ComputeImplementation::kNeon);
}

TEST(ComputeArithmeticProfile, Int8IsUnavailableWithoutAnyNativeCompiledAndDetectedPath) {
  ComputeDispatchInputs inputs;
  inputs.avx512vnni_compiled = true;
  inputs.avx512vnni_runtime = false;
  inputs.neon_dotprod_compiled = true;
  inputs.neon_dotprod_runtime = false;
  EXPECT_FALSE(SelectComputeImplementation(ComputeElementType::kInt8, inputs).available);

  inputs.avx512vnni_compiled = false;
  inputs.neon_dotprod_compiled = false;
  EXPECT_FALSE(SelectComputeImplementation(ComputeElementType::kInt8, inputs).available);
}

// ---------------------------------------------------------------------------
// Invalid options: fails before any allocation or timing.
// ---------------------------------------------------------------------------

TEST(ComputeArithmeticProfile, ZeroRepeatsIsInvalidOptions) {
  ComputeProfileOptions options;
  options.repeats = 0;
  const ComputeThroughputResult result = MeasureComputeArithmeticThroughput(
      ComputeElementType::kFloat32, ComputeParticipantPolicy::kSingle, options);
  EXPECT_FALSE(result.available);
  EXPECT_EQ(result.unavailable_reason, ComputeProfileUnavailableReason::kInvalidOptions);
  EXPECT_FALSE(result.diagnostic.empty());
}

TEST(ComputeArithmeticProfile, NonPositiveMinimumDurationIsInvalidOptions) {
  ComputeProfileOptions options;
  options.minimum_duration_ms = 0.0;
  const ComputeThroughputResult result = MeasureComputeArithmeticThroughput(
      ComputeElementType::kFloat64, ComputeParticipantPolicy::kSingle, options);
  EXPECT_FALSE(result.available);
  EXPECT_EQ(result.unavailable_reason, ComputeProfileUnavailableReason::kInvalidOptions);
}

// ---------------------------------------------------------------------------
// Real measurements: FP32/FP64 must be available on every platform. Kept
// short (few repeats, small minimum duration) so the suite stays fast.
// ---------------------------------------------------------------------------

ComputeProfileOptions FastOptions() {
  ComputeProfileOptions options;
  options.repeats = 3;
  options.minimum_duration_ms = 2.0;
  return options;
}

void ExpectFiniteSuccessfulResult(const ComputeThroughputResult &result) {
  ASSERT_TRUE(result.available);
  EXPECT_EQ(result.unavailable_reason, ComputeProfileUnavailableReason::kNone);
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
  const ComputeThroughputResult result = MeasureComputeArithmeticThroughput(
      ComputeElementType::kFloat32, ComputeParticipantPolicy::kSingle, FastOptions());
  ExpectFiniteSuccessfulResult(result);
  EXPECT_EQ(result.participant_count, 1u);
  EXPECT_EQ(result.dot_product_length, 0u);
}

TEST(ComputeArithmeticProfile, Float64SingleParticipantIsFiniteAndPositive) {
  const ComputeThroughputResult result = MeasureComputeArithmeticThroughput(
      ComputeElementType::kFloat64, ComputeParticipantPolicy::kSingle, FastOptions());
  ExpectFiniteSuccessfulResult(result);
  EXPECT_EQ(result.participant_count, 1u);
}

TEST(ComputeArithmeticProfile, Float32PhysicalParticipantsIsFiniteAndPositive) {
  const ComputeThroughputResult result = MeasureComputeArithmeticThroughput(
      ComputeElementType::kFloat32, ComputeParticipantPolicy::kPhysical, FastOptions());
  ExpectFiniteSuccessfulResult(result);
}

TEST(ComputeArithmeticProfile, Float64PhysicalParticipantsIsFiniteAndPositive) {
  const ComputeThroughputResult result = MeasureComputeArithmeticThroughput(
      ComputeElementType::kFloat64, ComputeParticipantPolicy::kPhysical, FastOptions());
  ExpectFiniteSuccessfulResult(result);
}

// ---------------------------------------------------------------------------
// Low-precision types: only asserted available/finite when this build and
// host actually have a native path; otherwise the measurement must be
// explicitly absent, never emulated.
// ---------------------------------------------------------------------------

TEST(ComputeArithmeticProfile, Float16IsFiniteWhenAvailableOtherwiseExplicitlyAbsent) {
  const ComputeThroughputResult result = MeasureComputeArithmeticThroughput(
      ComputeElementType::kFloat16, ComputeParticipantPolicy::kSingle, FastOptions());
  if (result.available) {
    ExpectFiniteSuccessfulResult(result);
  } else {
    EXPECT_EQ(result.unavailable_reason, ComputeProfileUnavailableReason::kUnsupportedNativePath);
    EXPECT_FALSE(result.diagnostic.empty());
  }
}

TEST(ComputeArithmeticProfile, BFloat16IsFiniteWhenAvailableOtherwiseExplicitlyAbsent) {
  const ComputeThroughputResult result = MeasureComputeArithmeticThroughput(
      ComputeElementType::kBFloat16, ComputeParticipantPolicy::kSingle, FastOptions());
  if (result.available) {
    ExpectFiniteSuccessfulResult(result);
    EXPECT_GT(result.dot_product_length, 0u);
  } else {
    EXPECT_EQ(result.unavailable_reason, ComputeProfileUnavailableReason::kUnsupportedNativePath);
    EXPECT_FALSE(result.diagnostic.empty());
  }
}

TEST(ComputeArithmeticProfile, Int8IsFiniteWhenAvailableOtherwiseExplicitlyAbsent) {
  const ComputeThroughputResult result = MeasureComputeArithmeticThroughput(
      ComputeElementType::kInt8, ComputeParticipantPolicy::kSingle, FastOptions());
  if (result.available) {
    ExpectFiniteSuccessfulResult(result);
    EXPECT_GT(result.dot_product_length, 0u);
  } else {
    EXPECT_EQ(result.unavailable_reason, ComputeProfileUnavailableReason::kUnsupportedNativePath);
    EXPECT_FALSE(result.diagnostic.empty());
  }
}

} // namespace
