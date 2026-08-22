// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Register-resident sustained arithmetic throughput measurement engine
// (Processor Profile PR03, see
// docs/next_steps/2026/2026_08_processor_performance_profile.rst). Every
// measurement keeps its operands and accumulators in registers: it must never
// become a memory-bandwidth benchmark like memory_traffic_profile.h.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx_light_cpu/impl/arm_simd_level.h"
#include "onnx_light_cpu/impl/simd_level.h"

namespace onnx_light_cpu {

/// Fused multiply-add (or dot-product reduction) iterations executed per
/// accumulator, per pass, by every arithmetic-profile kernel. Shared across
/// every scalar and SIMD implementation so operation accounting matches the
/// exact work each one executes.
constexpr std::size_t kComputeChainLength = 64;

/// Element type measured by one arithmetic throughput kernel.
enum class ComputeElementType {
  kFloat32,
  kFloat64,
  kFloat16,
  kBFloat16,
  kInt8,
};

/// Number and placement of participants used by one measurement.
enum class ComputeParticipantPolicy {
  kSingle,
  kPhysical,
};

/// Actual implementation selected for a measurement. Recorded so a result is
/// never attributed to native low-precision hardware that does not exist on
/// the current build or host.
enum class ComputeImplementation {
  kScalar,
  kSSE2,
  kAVX2,
  kAVX512,
  kNeon,
  kSve,
  kAmx,
  kOther,
};

/// Explicit reason a measurement could not produce a truthful result.
enum class ComputeProfileUnavailableReason {
  kNone,
  /// No compiled and runtime-detected native arithmetic path exists for the
  /// requested element type; the measurement is intentionally absent instead
  /// of emulated and reported as native.
  kUnsupportedNativePath,
  /// The requested repeat count or minimum duration is invalid; the
  /// measurement fails before allocating or timing anything.
  kInvalidOptions,
};

/// Options shared by every arithmetic throughput measurement.
struct ComputeProfileOptions {
  /// Number of recorded samples after warmup/calibration.
  std::size_t repeats = 7;
  /// Minimum wall-clock duration, per recorded sample, used to size the
  /// number of internal passes so timer resolution stays negligible.
  double minimum_duration_ms = 20.0;
};

/// Pure, timing-independent operation accounting for one arithmetic kernel.
struct ComputeOperationAccounting {
  /// Number of fused multiply-add (or multiply-accumulate) pairs executed
  /// per pass, across every independent accumulator.
  std::uint64_t fma_count_per_pass = 0;
  /// Number of scalar element pairs contracted by one dot-product reduction.
  /// Zero for floating-point FMA kernels, whose accounting is FMA-based
  /// rather than dot-product based.
  std::size_t dot_product_length = 0;
  /// Exact operations represented by one pass. One FMA (multiply, then
  /// accumulate) counts as exactly two floating-point operations. One INT8
  /// dot-product element pair counts as exactly two integer operations (one
  /// multiply, one accumulate), the same convention as FMA.
  std::uint64_t operations_per_pass = 0;
};

/// Computes the exact operation accounting for ``accumulator_count``
/// independent accumulators each executing ``fma_count_per_accumulator``
/// fused multiply-adds. Used by FP32/FP64/FP16 kernels.
ComputeOperationAccounting ComputeFmaOperationAccounting(std::size_t accumulator_count,
                                                         std::size_t fma_count_per_accumulator);

/// Computes the exact operation accounting for ``accumulator_count``
/// independent dot-product accumulators, each performing
/// ``reductions_per_accumulator`` reductions of ``dot_product_length``
/// element pairs. Used by the INT8/BF16 dot-product kernels.
ComputeOperationAccounting
ComputeDotProductOperationAccounting(std::size_t accumulator_count,
                                     std::size_t reductions_per_accumulator,
                                     std::size_t dot_product_length);

/// Wall-clock timer identity used by every measurement in this file.
const char *ComputeProfileTimerName();

/// Returns the short, stable label used for ``implementation`` in results and
/// diagnostics (for example ``"AVX2"``, ``"NEON"``, ``"scalar"``).
const char *ComputeImplementationName(ComputeImplementation implementation);

/// Result of one arithmetic throughput measurement.
struct ComputeThroughputResult {
  bool available = false;
  ComputeProfileUnavailableReason unavailable_reason = ComputeProfileUnavailableReason::kNone;
  std::string diagnostic;

  ComputeElementType element_type = ComputeElementType::kFloat32;
  ComputeParticipantPolicy policy = ComputeParticipantPolicy::kSingle;
  ComputeImplementation implementation = ComputeImplementation::kScalar;
  std::string implementation_name;

  std::size_t participant_count = 0;
  bool affinity_pinned = false;
  std::string timer_name;

  /// Exact operations accounted for one pass over one participant.
  std::uint64_t operations_per_pass_per_participant = 0;
  /// Scalar element pairs contracted by one dot-product reduction, or 0 for
  /// FMA-based element types.
  std::size_t dot_product_length = 0;

  /// One aggregate-throughput sample (summed across participants) per
  /// recorded repeat, in billions of operations per second. This is GFLOP/s
  /// for floating-point element types and GOP/s for INT8.
  std::vector<double> raw_gops_samples;
  double median_gops = 0.0;
  /// Interquartile range of ``raw_gops_samples``.
  double dispersion_gops = 0.0;
};

/// Measures register-resident arithmetic throughput for ``element_type``
/// using ``policy`` participants. Participant creation, synchronization, and
/// the final checksum reduction happen outside every timed sample.
ComputeThroughputResult
MeasureComputeArithmeticThroughput(ComputeElementType element_type, ComputeParticipantPolicy policy,
                                   const ComputeProfileOptions &options = {});

namespace detail {

/// Injected feature-detection inputs consumed by ``SelectComputeImplementation``.
/// Kept as plain data so dispatch selection is testable without depending on
/// the host CPU or executing any unsupported instruction.
struct ComputeDispatchInputs {
  SimdLevel simd_level = SimdLevel::kNone;
  bool has_fma = false;

  /// True only when the translation unit implementing the native AVX-512F
  /// FP32/FP64 kernel was compiled into this build.
  bool avx512_compiled = false;
  /// True only when the translation unit implementing the native AVX-512FP16
  /// kernel was compiled into this build.
  bool avx512fp16_compiled = false;
  bool avx512fp16_runtime = false;
  /// True only when the translation unit implementing the native AVX-512BF16
  /// dot-product kernel was compiled into this build.
  bool avx512bf16_compiled = false;
  bool avx512bf16_runtime = false;
  /// True only when the translation unit implementing the native AVX-512VNNI
  /// dot-product kernel was compiled into this build.
  bool avx512vnni_compiled = false;
  bool avx512vnni_runtime = false;

  ArmSimdLevel arm_simd_level = ArmSimdLevel::kNone;
  /// True only when the translation unit implementing the native NEON
  /// dot-product (SDOT/UDOT) kernel was compiled into this build.
  bool neon_dotprod_compiled = false;
  bool neon_dotprod_runtime = false;
};

/// Pure dispatch decision: chooses which implementation would be used for
/// ``element_type`` given ``inputs``, without allocating, timing, or
/// executing any instruction. ``available`` is false exactly when no
/// compiled and runtime-detected path exists for ``element_type`` (this is
/// always false for a low-precision element type when its native
/// translation unit was not compiled into the build, regardless of what the
/// runtime flags claim).
struct ComputeDispatchDecision {
  bool available = false;
  ComputeImplementation implementation = ComputeImplementation::kScalar;
};

ComputeDispatchDecision SelectComputeImplementation(ComputeElementType element_type,
                                                    const ComputeDispatchInputs &inputs);

} // namespace detail

} // namespace onnx_light_cpu
