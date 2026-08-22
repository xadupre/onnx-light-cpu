// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/compute_arithmetic_profile.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <functional>
#include <thread>
#include <vector>

#include "onnx_light_cpu/impl/thread_topology.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ONNX_LIGHT_CPU_COMPUTE_X86 1
#include <immintrin.h>
#else
#define ONNX_LIGHT_CPU_COMPUTE_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define ONNX_LIGHT_CPU_COMPUTE_ARM64 1
#include <arm_neon.h>
#else
#define ONNX_LIGHT_CPU_COMPUTE_ARM64 0
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
#include "onnx_light_cpu/impl/compute/avx512/compute_kernel_avx512.h"
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512FP16
#include "onnx_light_cpu/impl/compute/avx512fp16/compute_kernel_avx512fp16.h"
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512BF16
#include "onnx_light_cpu/impl/compute/avx512bf16/compute_kernel_avx512bf16.h"
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512VNNI
#include "onnx_light_cpu/impl/compute/avx512vnni/compute_kernel_avx512vnni.h"
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD
#include "onnx_light_cpu/impl/compute/arm/compute_kernel_neon_dotprod.h"
#endif

#if defined(_MSC_VER)
#define ONNX_LIGHT_CPU_NOINLINE __declspec(noinline)
#else
#define ONNX_LIGHT_CPU_NOINLINE __attribute__((noinline))
#endif

namespace onnx_light_cpu {

namespace {

using Clock = std::chrono::steady_clock;

// Volatile module-level sink. Every measurement writes its post-timing
// checksum here so the compiler cannot prove the timed arithmetic is dead,
// without adding any consumption inside a timed sample.
volatile double g_compute_profile_sink = 0.0;

double ToSeconds(Clock::duration duration) {
  return std::chrono::duration<double>(duration).count();
}

double Median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t count = values.size();
  return count % 2 == 1 ? values[count / 2] : 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

double Quantile(const std::vector<double> &sorted_values, double q) {
  if (sorted_values.empty()) {
    return 0.0;
  }
  if (sorted_values.size() == 1) {
    return sorted_values.front();
  }
  const double position = q * static_cast<double>(sorted_values.size() - 1);
  const std::size_t low = static_cast<std::size_t>(std::floor(position));
  const std::size_t high = static_cast<std::size_t>(std::ceil(position));
  if (low == high) {
    return sorted_values[low];
  }
  const double fraction = position - static_cast<double>(low);
  return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction;
}

double InterquartileRange(std::vector<double> values) {
  if (values.size() < 2) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  return Quantile(values, 0.75) - Quantile(values, 0.25);
}

constexpr double kComputeProfileGiga = 1.0e9;
constexpr std::size_t kComputeProfileMaxCalibrationPasses = 1u << 20;

// ---------------------------------------------------------------------------
// Baseline (portable scalar, plus x86 SSE2/AVX2 or AArch64 NEON) FP32/FP64
// register-resident FMA kernels. Every accumulator lives in a local array
// small enough to stay in registers; nothing here reads or writes a working
// set, so this never becomes a memory-bandwidth benchmark.
// ---------------------------------------------------------------------------

constexpr std::size_t kComputeScalarAccumulatorCount = 8;

ONNX_LIGHT_CPU_NOINLINE double RunScalarFloat32Round(std::size_t passes, double seed) {
  float acc[kComputeScalarAccumulatorCount];
  float mul[kComputeScalarAccumulatorCount];
  float add[kComputeScalarAccumulatorCount];
  for (std::size_t i = 0; i < kComputeScalarAccumulatorCount; ++i) {
    acc[i] = static_cast<float>(seed) + static_cast<float>(i);
    mul[i] = 1.0000001f + static_cast<float>(i) * 1e-7f;
    add[i] = 1.0e-6f * static_cast<float>(i + 1);
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (std::size_t i = 0; i < kComputeScalarAccumulatorCount; ++i) {
        acc[i] = acc[i] * mul[i] + add[i];
      }
    }
  }
  double sum = 0.0;
  for (float value : acc) {
    sum += static_cast<double>(value);
  }
  return sum;
}

ONNX_LIGHT_CPU_NOINLINE double RunScalarFloat64Round(std::size_t passes, double seed) {
  double acc[kComputeScalarAccumulatorCount];
  double mul[kComputeScalarAccumulatorCount];
  double add[kComputeScalarAccumulatorCount];
  for (std::size_t i = 0; i < kComputeScalarAccumulatorCount; ++i) {
    acc[i] = seed + static_cast<double>(i);
    mul[i] = 1.0000001 + static_cast<double>(i) * 1e-9;
    add[i] = 1.0e-9 * static_cast<double>(i + 1);
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (std::size_t i = 0; i < kComputeScalarAccumulatorCount; ++i) {
        acc[i] = acc[i] * mul[i] + add[i];
      }
    }
  }
  double sum = 0.0;
  for (double value : acc) {
    sum += value;
  }
  return sum;
}

#if ONNX_LIGHT_CPU_COMPUTE_X86

constexpr int kComputeSse2Registers = 4;
constexpr int kComputeAvx2Registers = 4;
constexpr std::size_t kComputeSse2Float32AccumulatorCount = kComputeSse2Registers * 4;
constexpr std::size_t kComputeSse2Float64AccumulatorCount = kComputeSse2Registers * 2;
constexpr std::size_t kComputeAvx2Float32AccumulatorCount = kComputeAvx2Registers * 8;
constexpr std::size_t kComputeAvx2Float64AccumulatorCount = kComputeAvx2Registers * 4;

inline __m256 ComputeMulAdd(__m256 a, __m256 b, __m256 acc) {
#ifdef __FMA__
  return _mm256_fmadd_ps(a, b, acc);
#else
  return _mm256_add_ps(acc, _mm256_mul_ps(a, b));
#endif
}
inline __m256d ComputeMulAdd(__m256d a, __m256d b, __m256d acc) {
#ifdef __FMA__
  return _mm256_fmadd_pd(a, b, acc);
#else
  return _mm256_add_pd(acc, _mm256_mul_pd(a, b));
#endif
}

ONNX_LIGHT_CPU_NOINLINE double RunSse2Float32Round(std::size_t passes, double seed) {
  __m128 acc[kComputeSse2Registers];
  __m128 mul[kComputeSse2Registers];
  __m128 add[kComputeSse2Registers];
  for (int reg = 0; reg < kComputeSse2Registers; ++reg) {
    acc[reg] = _mm_set1_ps(static_cast<float>(seed) + static_cast<float>(reg));
    mul[reg] = _mm_set1_ps(1.0000001f + static_cast<float>(reg) * 1e-7f);
    add[reg] = _mm_set1_ps(1.0e-6f * static_cast<float>(reg + 1));
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (int reg = 0; reg < kComputeSse2Registers; ++reg) {
        acc[reg] = _mm_add_ps(_mm_mul_ps(acc[reg], mul[reg]), add[reg]);
      }
    }
  }
  alignas(16) float lanes[4];
  double sum = 0.0;
  for (int reg = 0; reg < kComputeSse2Registers; ++reg) {
    _mm_store_ps(lanes, acc[reg]);
    for (float lane : lanes) {
      sum += lane;
    }
  }
  return sum;
}

ONNX_LIGHT_CPU_NOINLINE double RunSse2Float64Round(std::size_t passes, double seed) {
  __m128d acc[kComputeSse2Registers];
  __m128d mul[kComputeSse2Registers];
  __m128d add[kComputeSse2Registers];
  for (int reg = 0; reg < kComputeSse2Registers; ++reg) {
    acc[reg] = _mm_set1_pd(seed + static_cast<double>(reg));
    mul[reg] = _mm_set1_pd(1.0000001 + static_cast<double>(reg) * 1e-9);
    add[reg] = _mm_set1_pd(1.0e-9 * static_cast<double>(reg + 1));
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (int reg = 0; reg < kComputeSse2Registers; ++reg) {
        acc[reg] = _mm_add_pd(_mm_mul_pd(acc[reg], mul[reg]), add[reg]);
      }
    }
  }
  alignas(16) double lanes[2];
  double sum = 0.0;
  for (int reg = 0; reg < kComputeSse2Registers; ++reg) {
    _mm_store_pd(lanes, acc[reg]);
    for (double lane : lanes) {
      sum += lane;
    }
  }
  return sum;
}

ONNX_LIGHT_CPU_NOINLINE double RunAvx2Float32Round(std::size_t passes, double seed) {
  __m256 acc[kComputeAvx2Registers];
  __m256 mul[kComputeAvx2Registers];
  __m256 add[kComputeAvx2Registers];
  for (int reg = 0; reg < kComputeAvx2Registers; ++reg) {
    acc[reg] = _mm256_set1_ps(static_cast<float>(seed) + static_cast<float>(reg));
    mul[reg] = _mm256_set1_ps(1.0000001f + static_cast<float>(reg) * 1e-7f);
    add[reg] = _mm256_set1_ps(1.0e-6f * static_cast<float>(reg + 1));
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (int reg = 0; reg < kComputeAvx2Registers; ++reg) {
        acc[reg] = ComputeMulAdd(acc[reg], mul[reg], add[reg]);
      }
    }
  }
  alignas(32) float lanes[8];
  double sum = 0.0;
  for (int reg = 0; reg < kComputeAvx2Registers; ++reg) {
    _mm256_store_ps(lanes, acc[reg]);
    for (float lane : lanes) {
      sum += lane;
    }
  }
  return sum;
}

ONNX_LIGHT_CPU_NOINLINE double RunAvx2Float64Round(std::size_t passes, double seed) {
  __m256d acc[kComputeAvx2Registers];
  __m256d mul[kComputeAvx2Registers];
  __m256d add[kComputeAvx2Registers];
  for (int reg = 0; reg < kComputeAvx2Registers; ++reg) {
    acc[reg] = _mm256_set1_pd(seed + static_cast<double>(reg));
    mul[reg] = _mm256_set1_pd(1.0000001 + static_cast<double>(reg) * 1e-9);
    add[reg] = _mm256_set1_pd(1.0e-9 * static_cast<double>(reg + 1));
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (int reg = 0; reg < kComputeAvx2Registers; ++reg) {
        acc[reg] = ComputeMulAdd(acc[reg], mul[reg], add[reg]);
      }
    }
  }
  alignas(32) double lanes[4];
  double sum = 0.0;
  for (int reg = 0; reg < kComputeAvx2Registers; ++reg) {
    _mm256_store_pd(lanes, acc[reg]);
    for (double lane : lanes) {
      sum += lane;
    }
  }
  return sum;
}

#endif // ONNX_LIGHT_CPU_COMPUTE_X86

#if ONNX_LIGHT_CPU_COMPUTE_ARM64

constexpr int kComputeNeonRegisters = 4;
constexpr std::size_t kComputeNeonFloat32AccumulatorCount = kComputeNeonRegisters * 4;
constexpr std::size_t kComputeNeonFloat64AccumulatorCount = kComputeNeonRegisters * 2;

ONNX_LIGHT_CPU_NOINLINE double RunNeonFloat32Round(std::size_t passes, double seed) {
  float32x4_t acc[kComputeNeonRegisters];
  float32x4_t mul[kComputeNeonRegisters];
  float32x4_t add[kComputeNeonRegisters];
  for (int reg = 0; reg < kComputeNeonRegisters; ++reg) {
    acc[reg] = vdupq_n_f32(static_cast<float>(seed) + static_cast<float>(reg));
    mul[reg] = vdupq_n_f32(1.0000001f + static_cast<float>(reg) * 1e-7f);
    add[reg] = vdupq_n_f32(1.0e-6f * static_cast<float>(reg + 1));
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (int reg = 0; reg < kComputeNeonRegisters; ++reg) {
        acc[reg] = vfmaq_f32(add[reg], acc[reg], mul[reg]);
      }
    }
  }
  double sum = 0.0;
  for (int reg = 0; reg < kComputeNeonRegisters; ++reg) {
    alignas(16) float lanes[4];
    vst1q_f32(lanes, acc[reg]);
    for (float lane : lanes) {
      sum += lane;
    }
  }
  return sum;
}

ONNX_LIGHT_CPU_NOINLINE double RunNeonFloat64Round(std::size_t passes, double seed) {
  float64x2_t acc[kComputeNeonRegisters];
  float64x2_t mul[kComputeNeonRegisters];
  float64x2_t add[kComputeNeonRegisters];
  for (int reg = 0; reg < kComputeNeonRegisters; ++reg) {
    acc[reg] = vdupq_n_f64(seed + static_cast<double>(reg));
    mul[reg] = vdupq_n_f64(1.0000001 + static_cast<double>(reg) * 1e-9);
    add[reg] = vdupq_n_f64(1.0e-9 * static_cast<double>(reg + 1));
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t chain = 0; chain < kComputeChainLength; ++chain) {
      for (int reg = 0; reg < kComputeNeonRegisters; ++reg) {
        acc[reg] = vfmaq_f64(add[reg], acc[reg], mul[reg]);
      }
    }
  }
  double sum = 0.0;
  for (int reg = 0; reg < kComputeNeonRegisters; ++reg) {
    alignas(16) double lanes[2];
    vst1q_f64(lanes, acc[reg]);
    for (double lane : lanes) {
      sum += lane;
    }
  }
  return sum;
}

#endif // ONNX_LIGHT_CPU_COMPUTE_ARM64

// ---------------------------------------------------------------------------
// Participant coordination: mirrors memory_traffic_profile.cc's worker pool
// so participant creation, synchronization, and final checksum reduction all
// stay outside every timed sample.
// ---------------------------------------------------------------------------

using ComputeRoundFn = std::function<double(std::size_t passes, double seed)>;

class ComputeWorkerPool {
public:
  ComputeWorkerPool(ComputeRoundFn kernel, std::size_t participant_count,
                    const std::vector<CpuAffinity> &affinities)
      : kernel_(std::move(kernel)), barrier_(static_cast<std::ptrdiff_t>(participant_count)) {
    for (std::size_t index = 1; index < participant_count; ++index) {
      const CpuAffinity *affinity = index < affinities.size() ? &affinities[index] : nullptr;
      workers_.emplace_back([this, index, affinity]() {
        if (affinity != nullptr) {
          SetCurrentThreadAffinity(*affinity);
        }
        while (true) {
          barrier_.arrive_and_wait();
          const std::size_t passes = passes_.load(std::memory_order_acquire);
          if (passes == 0 && stop_.load(std::memory_order_relaxed)) {
            barrier_.arrive_and_wait();
            return;
          }
          const double checksum = kernel_(passes, static_cast<double>(index) + 1.0);
          barrier_.arrive_and_wait();
          g_compute_profile_sink += checksum;
        }
      });
    }
  }

  ~ComputeWorkerPool() {
    if (!workers_.empty()) {
      passes_.store(0, std::memory_order_release);
      stop_.store(true, std::memory_order_relaxed);
      barrier_.arrive_and_wait();
      barrier_.arrive_and_wait();
      for (std::thread &worker : workers_) {
        worker.join();
      }
    }
  }

  // Runs one synchronized round of ``passes`` iterations across every
  // participant and returns the aggregate wall-clock seconds elapsed.
  double RunRound(std::size_t passes) {
    passes_.store(passes, std::memory_order_release);
    barrier_.arrive_and_wait();
    const Clock::time_point start = Clock::now();
    const double checksum = kernel_(passes, 0.0);
    barrier_.arrive_and_wait();
    const Clock::time_point end = Clock::now();
    g_compute_profile_sink += checksum;
    return ToSeconds(end - start);
  }

private:
  ComputeRoundFn kernel_;
  std::barrier<> barrier_;
  std::vector<std::thread> workers_;
  std::atomic<std::size_t> passes_{0};
  std::atomic<bool> stop_{false};
};

} // namespace

ComputeOperationAccounting ComputeFmaOperationAccounting(std::size_t accumulator_count,
                                                         std::size_t fma_count_per_accumulator) {
  ComputeOperationAccounting accounting;
  accounting.fma_count_per_pass = static_cast<std::uint64_t>(accumulator_count) *
                                  static_cast<std::uint64_t>(fma_count_per_accumulator);
  accounting.operations_per_pass = accounting.fma_count_per_pass * 2;
  return accounting;
}

ComputeOperationAccounting
ComputeDotProductOperationAccounting(std::size_t accumulator_count,
                                     std::size_t reductions_per_accumulator,
                                     std::size_t dot_product_length) {
  ComputeOperationAccounting accounting;
  const std::uint64_t total_pairs = static_cast<std::uint64_t>(accumulator_count) *
                                    static_cast<std::uint64_t>(reductions_per_accumulator) *
                                    static_cast<std::uint64_t>(dot_product_length);
  accounting.fma_count_per_pass = total_pairs;
  accounting.dot_product_length = dot_product_length;
  accounting.operations_per_pass = total_pairs * 2;
  return accounting;
}

const char *ComputeProfileTimerName() { return "std::chrono::steady_clock"; }

const char *ComputeImplementationName(ComputeImplementation implementation) {
  switch (implementation) {
  case ComputeImplementation::kScalar:
    return "scalar";
  case ComputeImplementation::kSSE2:
    return "SSE2";
  case ComputeImplementation::kAVX2:
    return "AVX2";
  case ComputeImplementation::kAVX512:
    return "AVX-512";
  case ComputeImplementation::kNeon:
    return "NEON";
  case ComputeImplementation::kSve:
    return "SVE";
  case ComputeImplementation::kAmx:
    return "AMX";
  case ComputeImplementation::kOther:
    return "other";
  }
  return "other";
}

namespace detail {

ComputeDispatchDecision SelectComputeImplementation(ComputeElementType element_type,
                                                    const ComputeDispatchInputs &inputs) {
  ComputeDispatchDecision decision;
  switch (element_type) {
  case ComputeElementType::kFloat32:
  case ComputeElementType::kFloat64: {
    // Always available: register-resident FP32/FP64 FMA has a portable
    // scalar fallback on every platform.
    decision.available = true;
    if (inputs.avx512_compiled && inputs.simd_level >= SimdLevel::kAVX512) {
      decision.implementation = ComputeImplementation::kAVX512;
    } else if (inputs.arm_simd_level != ArmSimdLevel::kNone) {
      // SVE is not yet measured directly (Profile PR03 scope): NEON is a
      // safe, always-available AArch64 baseline and satisfies "every
      // platform" without fabricating an unmeasured SVE result.
      decision.implementation = ComputeImplementation::kNeon;
    } else if (inputs.simd_level >= SimdLevel::kAVX2 && inputs.has_fma) {
      decision.implementation = ComputeImplementation::kAVX2;
    } else if (inputs.simd_level >= SimdLevel::kSSE2) {
      decision.implementation = ComputeImplementation::kSSE2;
    } else {
      decision.implementation = ComputeImplementation::kScalar;
    }
    return decision;
  }
  case ComputeElementType::kFloat16: {
    if (inputs.avx512fp16_compiled && inputs.avx512fp16_runtime) {
      decision.available = true;
      decision.implementation = ComputeImplementation::kAVX512;
    }
    return decision;
  }
  case ComputeElementType::kBFloat16: {
    if (inputs.avx512bf16_compiled && inputs.avx512bf16_runtime) {
      decision.available = true;
      decision.implementation = ComputeImplementation::kAVX512;
    }
    return decision;
  }
  case ComputeElementType::kInt8: {
    if (inputs.avx512vnni_compiled && inputs.avx512vnni_runtime) {
      decision.available = true;
      decision.implementation = ComputeImplementation::kAVX512;
    } else if (inputs.neon_dotprod_compiled && inputs.neon_dotprod_runtime) {
      decision.available = true;
      decision.implementation = ComputeImplementation::kNeon;
    }
    return decision;
  }
  }
  return decision;
}

} // namespace detail

ComputeThroughputResult MeasureComputeArithmeticThroughput(ComputeElementType element_type,
                                                           ComputeParticipantPolicy policy,
                                                           const ComputeProfileOptions &options) {
  ComputeThroughputResult result;
  result.element_type = element_type;
  result.policy = policy;
  result.timer_name = ComputeProfileTimerName();

  if (options.repeats == 0 || options.minimum_duration_ms <= 0.0) {
    result.unavailable_reason = ComputeProfileUnavailableReason::kInvalidOptions;
    result.diagnostic = "invalid measurement options";
    return result;
  }

  detail::ComputeDispatchInputs inputs;
  inputs.simd_level = DetectSimdLevel();
  inputs.has_fma = CpuSupportsFma();
  inputs.arm_simd_level = DetectArmSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  inputs.avx512_compiled = true;
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512FP16
  inputs.avx512fp16_compiled = true;
  inputs.avx512fp16_runtime = CpuSupportsAvx512Fp16();
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512BF16
  inputs.avx512bf16_compiled = true;
  inputs.avx512bf16_runtime = CpuSupportsAvx512Bf16();
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512VNNI
  inputs.avx512vnni_compiled = true;
  inputs.avx512vnni_runtime = CpuSupportsAvx512Vnni();
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD
  inputs.neon_dotprod_compiled = true;
  inputs.neon_dotprod_runtime = CpuSupportsNeonDotProd();
#endif

  const detail::ComputeDispatchDecision decision =
      detail::SelectComputeImplementation(element_type, inputs);
  if (!decision.available) {
    result.unavailable_reason = ComputeProfileUnavailableReason::kUnsupportedNativePath;
    result.diagnostic =
        "no compiled and runtime-detected native arithmetic path for this element type";
    return result;
  }
  result.implementation = decision.implementation;
  result.implementation_name = ComputeImplementationName(decision.implementation);

  ComputeOperationAccounting accounting;
  ComputeRoundFn kernel;

  switch (element_type) {
  case ComputeElementType::kFloat32: {
    switch (decision.implementation) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
    case ComputeImplementation::kAVX512:
      accounting =
          ComputeFmaOperationAccounting(kComputeAvx512Float32AccumulatorCount, kComputeChainLength);
      kernel = &ComputeArithmeticAvx512Float32Round;
      break;
#endif
#if ONNX_LIGHT_CPU_COMPUTE_ARM64
    case ComputeImplementation::kNeon:
      accounting =
          ComputeFmaOperationAccounting(kComputeNeonFloat32AccumulatorCount, kComputeChainLength);
      kernel = &RunNeonFloat32Round;
      break;
#endif
#if ONNX_LIGHT_CPU_COMPUTE_X86
    case ComputeImplementation::kAVX2:
      accounting =
          ComputeFmaOperationAccounting(kComputeAvx2Float32AccumulatorCount, kComputeChainLength);
      kernel = &RunAvx2Float32Round;
      break;
    case ComputeImplementation::kSSE2:
      accounting =
          ComputeFmaOperationAccounting(kComputeSse2Float32AccumulatorCount, kComputeChainLength);
      kernel = &RunSse2Float32Round;
      break;
#endif
    default:
      accounting =
          ComputeFmaOperationAccounting(kComputeScalarAccumulatorCount, kComputeChainLength);
      kernel = &RunScalarFloat32Round;
      break;
    }
    break;
  }
  case ComputeElementType::kFloat64: {
    switch (decision.implementation) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
    case ComputeImplementation::kAVX512:
      accounting =
          ComputeFmaOperationAccounting(kComputeAvx512Float64AccumulatorCount, kComputeChainLength);
      kernel = &ComputeArithmeticAvx512Float64Round;
      break;
#endif
#if ONNX_LIGHT_CPU_COMPUTE_ARM64
    case ComputeImplementation::kNeon:
      accounting =
          ComputeFmaOperationAccounting(kComputeNeonFloat64AccumulatorCount, kComputeChainLength);
      kernel = &RunNeonFloat64Round;
      break;
#endif
#if ONNX_LIGHT_CPU_COMPUTE_X86
    case ComputeImplementation::kAVX2:
      accounting =
          ComputeFmaOperationAccounting(kComputeAvx2Float64AccumulatorCount, kComputeChainLength);
      kernel = &RunAvx2Float64Round;
      break;
    case ComputeImplementation::kSSE2:
      accounting =
          ComputeFmaOperationAccounting(kComputeSse2Float64AccumulatorCount, kComputeChainLength);
      kernel = &RunSse2Float64Round;
      break;
#endif
    default:
      accounting =
          ComputeFmaOperationAccounting(kComputeScalarAccumulatorCount, kComputeChainLength);
      kernel = &RunScalarFloat64Round;
      break;
    }
    break;
  }
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512FP16
  case ComputeElementType::kFloat16: {
    accounting =
        ComputeFmaOperationAccounting(kComputeAvx512Fp16AccumulatorCount, kComputeChainLength);
    kernel = &ComputeArithmeticAvx512Fp16Round;
    break;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512BF16
  case ComputeElementType::kBFloat16: {
    accounting = ComputeDotProductOperationAccounting(kComputeAvx512Bf16AccumulatorCount,
                                                      kComputeChainLength,
                                                      kComputeAvx512Bf16DotProductLength);
    kernel = &ComputeArithmeticAvx512Bf16Round;
    break;
  }
#endif
  case ComputeElementType::kInt8: {
    switch (decision.implementation) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512VNNI
    case ComputeImplementation::kAVX512:
      accounting = ComputeDotProductOperationAccounting(kComputeAvx512VnniAccumulatorCount,
                                                        kComputeChainLength,
                                                        kComputeAvx512VnniDotProductLength);
      kernel = &ComputeArithmeticAvx512VnniRound;
      break;
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD
    case ComputeImplementation::kNeon:
      accounting = ComputeDotProductOperationAccounting(kComputeNeonDotProdAccumulatorCount,
                                                        kComputeChainLength,
                                                        kComputeNeonDotProdDotProductLength);
      kernel = &ComputeArithmeticNeonDotProdRound;
      break;
#endif
    default:
      break;
    }
    break;
  }
  default:
    break;
  }

  if (!kernel) {
    // The dispatch decision claimed a native path but this build was not
    // compiled with the matching translation unit (should not happen given
    // SelectComputeImplementation only reports a compiled path available).
    result.unavailable_reason = ComputeProfileUnavailableReason::kUnsupportedNativePath;
    result.diagnostic = "selected implementation has no compiled kernel";
    return result;
  }

  result.operations_per_pass_per_participant = accounting.operations_per_pass;
  result.dot_product_length = accounting.dot_product_length;

  std::vector<CpuAffinity> affinities;
  std::size_t participant_count = 1;
  bool affinity_pinned = false;
  if (policy == ComputeParticipantPolicy::kPhysical) {
    const CpuTopology &cpu_topology = GetCpuTopology();
    participant_count = std::max<std::size_t>(cpu_topology.physical_core_count, 1);
    affinities = SelectCpuAffinities(cpu_topology, participant_count);
    if (!affinities.empty()) {
      participant_count = affinities.size();
      affinity_pinned = true;
    } else {
      participant_count = 1;
    }
  }
  result.participant_count = participant_count;
  result.affinity_pinned = affinity_pinned;

  if (!affinity_pinned) {
    affinities.clear();
  }
  ComputeWorkerPool pool(kernel, participant_count, affinities);

  std::size_t passes = 1;
  const double calibration_elapsed = pool.RunRound(1);
  {
    const double target_seconds = options.minimum_duration_ms / 1000.0;
    const double safe_elapsed = calibration_elapsed > 0.0 ? calibration_elapsed : 1e-9;
    const double estimated_passes = target_seconds / safe_elapsed;
    passes = static_cast<std::size_t>(std::ceil(std::max(1.0, estimated_passes)));
    passes = std::min(passes, kComputeProfileMaxCalibrationPasses);
  }

  result.raw_gops_samples.reserve(options.repeats);
  for (std::size_t repeat = 0; repeat < options.repeats; ++repeat) {
    const double elapsed_seconds = pool.RunRound(passes);
    const double total_operations = static_cast<double>(accounting.operations_per_pass) *
                                    static_cast<double>(passes) *
                                    static_cast<double>(participant_count);
    const double seconds = elapsed_seconds > 0.0 ? elapsed_seconds : 1e-12;
    result.raw_gops_samples.push_back(total_operations / seconds / kComputeProfileGiga);
  }

  result.median_gops = Median(result.raw_gops_samples);
  result.dispersion_gops = InterquartileRange(result.raw_gops_samples);
  result.available = true;
  return result;
}

} // namespace onnx_light_cpu
