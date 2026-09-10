// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include "onnx_light_cpu/impl/cpu_cache_topology.h"
#include "onnx_light_cpu/impl/execution.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ONNX_LIGHT_CPU_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#else
#define ONNX_LIGHT_CPU_X86 0
#endif

namespace onnx_light_cpu {

#ifdef ONNX_LIGHT_CPU_HAVE_AVX
void AbsFloat32_AVX(const float *input, float *output, std::size_t count);
void AbsFloat32_AVXStreaming(const float *input, float *output, std::size_t count);
void AbsFloat64_AVX(const double *input, double *output, std::size_t count);
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
void AbsFloat16_AVX2(const uint16_t *input, uint16_t *output, std::size_t count);
void AbsInt8_AVX2(const int8_t *input, int8_t *output, std::size_t count);
void AbsInt16_AVX2(const int16_t *input, int16_t *output, std::size_t count);
void AbsInt32_AVX2(const int32_t *input, int32_t *output, std::size_t count);
void AbsInt64_AVX2(const int64_t *input, int64_t *output, std::size_t count);
#endif

#if defined(ONNX_LIGHT_CPU_HAVE_AVX512) && defined(__AVX512F__)
void AbsFloat64_AVX512(const double *input, double *output, std::size_t count);
void AbsFloat16_AVX512(const uint16_t *input, uint16_t *output, std::size_t count);
void AbsInt32_AVX512(const int32_t *input, int32_t *output, std::size_t count);
void AbsInt64_AVX512(const int64_t *input, int64_t *output, std::size_t count);
#endif
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512) && defined(__AVX512BW__)
void AbsInt8_AVX512(const int8_t *input, int8_t *output, std::size_t count);
void AbsInt16_AVX512(const int16_t *input, int16_t *output, std::size_t count);
#endif

// ---------------------------------------------------------------------------
// AbsFloat32 implementations
// ---------------------------------------------------------------------------

namespace {

void AbsFloat32_Scalar(const float *input, float *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}

#if ONNX_LIGHT_CPU_X86

void AbsFloat32_SSE2(const float *input, float *output, std::size_t count) {
  // Clear sign bit: AND with 0x7FFFFFFF
  const __m128 sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
  std::size_t i = 0;
  const std::size_t stride = 4;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m128 v = _mm_loadu_ps(input + i);
    v = _mm_and_ps(v, sign_mask);
    _mm_storeu_ps(output + i, v);
  }
  for (; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}

#endif // ONNX_LIGHT_CPU_X86

} // namespace

namespace {
using AbsFloat32Fn = void (*)(const float *, float *, std::size_t);

struct AbsFloat32Dispatch {
  AbsFloat32Fn function;
  AbsFloat32Fn streaming_function;
  double compute_cycles;
  UnaryExecutionTuning default_tuning = kDefaultAbsFloat32ExecutionTuning;
  bool dynamic_ranges = false;
};

constexpr double AbsComputeCycles(std::size_t simd_lanes) {
  constexpr double kAvx2ComputeCycles = 1.0;
  constexpr std::size_t kAvx2Lanes = 8;
  return std::max(kAvx2ComputeCycles, kAvx2ComputeCycles * static_cast<double>(kAvx2Lanes) /
                                          static_cast<double>(simd_lanes));
}

const AbsFloat32Dispatch &GetAbsFloat32Dispatch() {
  static const AbsFloat32Dispatch dispatch = [] {
#if ONNX_LIGHT_CPU_X86
    const SimdLevel level = DetectSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
    if (level >= SimdLevel::kAVX512) {
      return AbsFloat32Dispatch{&AbsFloat32_AVX512, &AbsFloat32_AVX512Streaming,
                                AbsComputeCycles(16)};
    }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX
    if (level >= SimdLevel::kAVX) {
      const std::size_t cache_bytes = CpuCacheSizeBytesOrFallback(GetCpuCacheTopology(), 3, 0);
      // Keep cached stores when the input and output fit in the last-level
      // cache. Unknown cache topology retains the opt-in streaming policy.
      const std::size_t streaming_threshold =
          cache_bytes == 0 ? 0 : std::max<std::size_t>(16 * 1024 * 1024, cache_bytes / 2 + 1);
      return AbsFloat32Dispatch{&AbsFloat32_AVX,
                                &AbsFloat32_AVXStreaming,
                                AbsComputeCycles(8),
                                {2 * 1024 * 1024, 1024 * 1024, 32, true, 0, streaming_threshold},
                                true};
    }
#endif
    if (level >= SimdLevel::kSSE2) {
      return AbsFloat32Dispatch{&AbsFloat32_SSE2, nullptr, AbsComputeCycles(4)};
    }
#endif
    return AbsFloat32Dispatch{&AbsFloat32_Scalar, nullptr, AbsComputeCycles(1)};
  }();
  return dispatch;
}
} // namespace

const UnaryExecutionTuning &DefaultAbsFloat32ExecutionTuning() {
  return GetAbsFloat32Dispatch().default_tuning;
}

void AbsFloat32(const float *input, float *output, std::size_t count) {
  AbsFloat32WithTuning(input, output, count, DefaultAbsFloat32ExecutionTuning());
}

void AbsFloat32WithTuning(const float *input, float *output, std::size_t count,
                          const UnaryExecutionTuning &tuning) {
  if (count == 0) {
    return;
  }
  const bool streaming_store =
      tuning.streaming_store_threshold_bytes != 0 &&
      count >= UnaryBytesToElements(tuning.streaming_store_threshold_bytes, sizeof(float));
  const AbsFloat32Dispatch &dispatch = GetAbsFloat32Dispatch();
  const AbsFloat32Fn function = streaming_store && dispatch.streaming_function != nullptr
                                    ? dispatch.streaming_function
                                    : dispatch.function;
  constexpr std::size_t tile = 16 * 1024;
  std::atomic<std::size_t> next{0};
  const bool dynamic = dispatch.dynamic_ranges && streaming_store && count >= 2 * tile &&
                       tuning.max_participants != 1 && tuning.preferred_participants != 1 &&
                       ExecutionThreadCount() > 1 && !ExecutionInParallelRegion();
  auto execute = [input, output, function, &next, dynamic, count, tile](std::int64_t begin,
                                                                        std::int64_t end) {
    if (dynamic) {
      // Workers claim cache-sized tiles so slower workers do not delay fixed ranges.
      for (std::size_t first = next.fetch_add(tile, std::memory_order_relaxed); first < count;
           first = next.fetch_add(tile, std::memory_order_relaxed)) {
        function(input + first, output + first, std::min(tile, count - first));
      }
    } else {
      function(input + begin, output + begin, static_cast<std::size_t>(end - begin));
    }
  };
  if (tuning.use_cost_model) {
    auto bounded = tuning;
    if (dynamic) {
      const std::size_t useful_participants = std::max<std::size_t>(
          1, count / UnaryBytesToElements(tuning.target_block_bytes, sizeof(float)));
      bounded.max_participants = tuning.max_participants == 0
                                     ? useful_participants
                                     : std::min(tuning.max_participants, useful_participants);
    }
    ExecuteCostedUnaryRanges<float>(count, bounded, dispatch.compute_cycles, std::move(execute));
  } else {
    ExecuteUnaryRanges<float>(count, tuning, std::move(execute));
  }
}

// ---------------------------------------------------------------------------
// AbsFloat64 implementations
// ---------------------------------------------------------------------------

namespace {

void AbsFloat64_Scalar(const double *input, double *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}

#if ONNX_LIGHT_CPU_X86

void AbsFloat64_SSE2(const double *input, double *output, std::size_t count) {
  const __m128d sign_mask =
      _mm_castsi128_pd(_mm_set_epi64x(0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF));
  std::size_t i = 0;
  const std::size_t stride = 2;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m128d v = _mm_loadu_pd(input + i);
    v = _mm_and_pd(v, sign_mask);
    _mm_storeu_pd(output + i, v);
  }
  for (; i < count; ++i) {
    output[i] = std::fabs(input[i]);
  }
}

#endif // ONNX_LIGHT_CPU_X86

} // namespace

namespace {
void AbsFloat64_Dispatch(const double *input, double *output, std::size_t count) {
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512) && defined(__AVX512F__)
  if (level >= SimdLevel::kAVX512) {
    AbsFloat64_AVX512(input, output, count);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX
  if (level >= SimdLevel::kAVX) {
    AbsFloat64_AVX(input, output, count);
    return;
  }
#endif
  if (level >= SimdLevel::kSSE2) {
    AbsFloat64_SSE2(input, output, count);
    return;
  }
#endif
  AbsFloat64_Scalar(input, output, count);
}
} // namespace

void AbsFloat64(const double *input, double *output, std::size_t count) {
  AbsFloat64WithTuning(input, output, count, kDefaultAbs64ExecutionTuning);
}

void AbsFloat64WithTuning(const double *input, double *output, std::size_t count,
                          const UnaryExecutionTuning &tuning) {
  ExecuteUnaryRanges<double>(count, tuning, [input, output](std::int64_t begin, std::int64_t end) {
    AbsFloat64_Dispatch(input + begin, output + begin, static_cast<std::size_t>(end - begin));
  });
}

// ---------------------------------------------------------------------------
// AbsFloat16 implementations
// ---------------------------------------------------------------------------
//
// The elementwise absolute value of an IEEE 754 half-precision value is a pure
// bit operation on its 16-bit representation: clearing the sign bit (bit 15).
// This works for every category (normal, subnormal, zero, infinity, NaN) and
// avoids any float16<->float32 conversion, so no F16C support is required.

namespace {

constexpr std::uint16_t kFloat16AbsMask = 0x7FFF;

void AbsFloat16_Scalar(const uint16_t *input, uint16_t *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = static_cast<uint16_t>(input[i] & kFloat16AbsMask);
  }
}

#if ONNX_LIGHT_CPU_X86

void AbsFloat16_SSE2(const uint16_t *input, uint16_t *output, std::size_t count) {
  const __m128i mask = _mm_set1_epi16(static_cast<short>(kFloat16AbsMask));
  std::size_t i = 0;
  const std::size_t stride = 8;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));
    v = _mm_and_si128(v, mask);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = static_cast<uint16_t>(input[i] & kFloat16AbsMask);
  }
}

#endif // ONNX_LIGHT_CPU_X86

} // namespace

namespace {
void AbsFloat16_Dispatch(const uint16_t *input, uint16_t *output, std::size_t count) {
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512) && defined(__AVX512F__)
  if (level >= SimdLevel::kAVX512) {
    AbsFloat16_AVX512(input, output, count);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (level >= SimdLevel::kAVX2) {
    AbsFloat16_AVX2(input, output, count);
    return;
  }
#endif
  if (level >= SimdLevel::kSSE2) {
    AbsFloat16_SSE2(input, output, count);
    return;
  }
#endif
  AbsFloat16_Scalar(input, output, count);
}
} // namespace

void AbsFloat16(const uint16_t *input, uint16_t *output, std::size_t count) {
  AbsFloat16WithTuning(input, output, count, kDefaultAbs16ExecutionTuning);
}

void AbsFloat16WithTuning(const uint16_t *input, uint16_t *output, std::size_t count,
                          const UnaryExecutionTuning &tuning) {
  ExecuteUnaryRanges<std::uint16_t>(
      count, tuning, [input, output](std::int64_t begin, std::int64_t end) {
        AbsFloat16_Dispatch(input + begin, output + begin, static_cast<std::size_t>(end - begin));
      });
}

// ---------------------------------------------------------------------------
// AbsInt8 implementations
// ---------------------------------------------------------------------------

namespace {

void AbsInt8_Scalar(const int8_t *input, int8_t *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    const int v = static_cast<int>(input[i]);
    output[i] = static_cast<int8_t>(v < 0 ? -v : v);
  }
}

#if ONNX_LIGHT_CPU_X86

void AbsInt8_SSE2(const int8_t *input, int8_t *output, std::size_t count) {
  // SSSE3 pabsb is not guaranteed at SSE2; emulate with (x XOR mask) - mask
  // where mask = (x < 0) ? 0xFF : 0x00 via a signed byte compare.
  const __m128i zero = _mm_setzero_si128();
  std::size_t i = 0;
  const std::size_t stride = 16;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));
    __m128i mask = _mm_cmpgt_epi8(zero, v);
    v = _mm_sub_epi8(_mm_xor_si128(v, mask), mask);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), v);
  }
  for (; i < count; ++i) {
    const int v = static_cast<int>(input[i]);
    output[i] = static_cast<int8_t>(v < 0 ? -v : v);
  }
}

#endif // ONNX_LIGHT_CPU_X86

} // namespace

namespace {
void AbsInt8_Dispatch(const int8_t *input, int8_t *output, std::size_t count) {
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512) && defined(__AVX512BW__)
  if (level >= SimdLevel::kAVX512) {
    AbsInt8_AVX512(input, output, count);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (level >= SimdLevel::kAVX2) {
    AbsInt8_AVX2(input, output, count);
    return;
  }
#endif
  if (level >= SimdLevel::kSSE2) {
    AbsInt8_SSE2(input, output, count);
    return;
  }
#endif
  AbsInt8_Scalar(input, output, count);
}
} // namespace

void AbsInt8(const int8_t *input, int8_t *output, std::size_t count) {
  AbsInt8WithTuning(input, output, count, kDefaultAbsInt8ExecutionTuning);
}

void AbsInt8WithTuning(const int8_t *input, int8_t *output, std::size_t count,
                       const UnaryExecutionTuning &tuning) {
  ExecuteUnaryRanges<std::int8_t>(
      count, tuning, [input, output](std::int64_t begin, std::int64_t end) {
        AbsInt8_Dispatch(input + begin, output + begin, static_cast<std::size_t>(end - begin));
      });
}

// ---------------------------------------------------------------------------
// AbsInt16 implementations
// ---------------------------------------------------------------------------

namespace {

void AbsInt16_Scalar(const int16_t *input, int16_t *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    const int32_t value = static_cast<int32_t>(input[i]);
    output[i] = static_cast<int16_t>(value < 0 ? -value : value);
  }
}

#if ONNX_LIGHT_CPU_X86

void AbsInt16_SSE2(const int16_t *input, int16_t *output, std::size_t count) {
  std::size_t i = 0;
  constexpr std::size_t stride = 8;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    const __m128i value = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));
    const __m128i sign = _mm_srai_epi16(value, 15);
    const __m128i result = _mm_sub_epi16(_mm_xor_si128(value, sign), sign);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), result);
  }
  AbsInt16_Scalar(input + i, output + i, count - i);
}

#endif

void AbsInt16_Dispatch(const int16_t *input, int16_t *output, std::size_t count) {
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512) && defined(__AVX512BW__)
  if (level >= SimdLevel::kAVX512) {
    AbsInt16_AVX512(input, output, count);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (level >= SimdLevel::kAVX2) {
    AbsInt16_AVX2(input, output, count);
    return;
  }
#endif
  if (level >= SimdLevel::kSSE2) {
    AbsInt16_SSE2(input, output, count);
    return;
  }
#endif
  AbsInt16_Scalar(input, output, count);
}

} // namespace

void AbsInt16(const int16_t *input, int16_t *output, std::size_t count) {
  AbsInt16WithTuning(input, output, count, kDefaultAbsInt16ExecutionTuning);
}

void AbsInt16WithTuning(const int16_t *input, int16_t *output, std::size_t count,
                        const UnaryExecutionTuning &tuning) {
  ExecuteUnaryRanges<std::int16_t>(
      count, tuning, [input, output](std::int64_t begin, std::int64_t end) {
        AbsInt16_Dispatch(input + begin, output + begin, static_cast<std::size_t>(end - begin));
      });
}

// ---------------------------------------------------------------------------
// AbsInt32 implementations
// ---------------------------------------------------------------------------

namespace {

void AbsInt32_Scalar(const int32_t *input, int32_t *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}

#if ONNX_LIGHT_CPU_X86

void AbsInt32_SSE2(const int32_t *input, int32_t *output, std::size_t count) {
  // SSE2 pabsd is not available; use (x XOR mask) - mask where mask = x >> 31.
  std::size_t i = 0;
  const std::size_t stride = 4;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));
    __m128i mask = _mm_srai_epi32(v, 31);
    v = _mm_sub_epi32(_mm_xor_si128(v, mask), mask);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}

#endif // ONNX_LIGHT_CPU_X86

} // namespace

namespace {
void AbsInt32_Dispatch(const int32_t *input, int32_t *output, std::size_t count) {
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512) && defined(__AVX512F__)
  if (level >= SimdLevel::kAVX512) {
    AbsInt32_AVX512(input, output, count);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (level >= SimdLevel::kAVX2) {
    AbsInt32_AVX2(input, output, count);
    return;
  }
#endif
  if (level >= SimdLevel::kSSE2) {
    AbsInt32_SSE2(input, output, count);
    return;
  }
#endif
  AbsInt32_Scalar(input, output, count);
}
} // namespace

void AbsInt32(const int32_t *input, int32_t *output, std::size_t count) {
  AbsInt32WithTuning(input, output, count, kDefaultAbsInt32ExecutionTuning);
}

void AbsInt32WithTuning(const int32_t *input, int32_t *output, std::size_t count,
                        const UnaryExecutionTuning &tuning) {
  auto execute = [input, output](std::int64_t begin, std::int64_t end) {
    AbsInt32_Dispatch(input + begin, output + begin, static_cast<std::size_t>(end - begin));
  };
  if (tuning.use_cost_model) {
    ExecuteCostedUnaryRanges<std::int32_t>(count, tuning, 1.0, std::move(execute));
  } else {
    ExecuteUnaryRanges<std::int32_t>(count, tuning, std::move(execute));
  }
}

// ---------------------------------------------------------------------------
// AbsInt64 implementations
// ---------------------------------------------------------------------------

namespace {

void AbsInt64_Scalar(const int64_t *input, int64_t *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}

#if ONNX_LIGHT_CPU_X86

void AbsInt64_SSE2(const int64_t *input, int64_t *output, std::size_t count) {
  // No pabsq before AVX512; use conditional negate with arithmetic shift
  // emulation for 64-bit.
  std::size_t i = 0;
  const std::size_t stride = 2;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));
    // Arithmetic right shift by 63: shift hi 32 bits, then shuffle
    __m128i sign = _mm_srai_epi32(v, 31);
    sign = _mm_shuffle_epi32(sign, 0xF5); // replicate bits 63 across both halves
    __m128i abs_v = _mm_sub_epi64(_mm_xor_si128(v, sign), sign);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), abs_v);
  }
  for (; i < count; ++i) {
    output[i] = input[i] < 0 ? -input[i] : input[i];
  }
}

#endif // ONNX_LIGHT_CPU_X86

} // namespace

namespace {
void AbsInt64_Dispatch(const int64_t *input, int64_t *output, std::size_t count) {
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512) && defined(__AVX512F__)
  if (level >= SimdLevel::kAVX512) {
    AbsInt64_AVX512(input, output, count);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (level >= SimdLevel::kAVX2) {
    AbsInt64_AVX2(input, output, count);
    return;
  }
#endif
  if (level >= SimdLevel::kSSE2) {
    AbsInt64_SSE2(input, output, count);
    return;
  }
#endif
  AbsInt64_Scalar(input, output, count);
}
} // namespace

void AbsInt64(const int64_t *input, int64_t *output, std::size_t count) {
  AbsInt64WithTuning(input, output, count, kDefaultAbsInt64ExecutionTuning);
}

void AbsInt64WithTuning(const int64_t *input, int64_t *output, std::size_t count,
                        const UnaryExecutionTuning &tuning) {
  ExecuteUnaryRanges<std::int64_t>(
      count, tuning, [input, output](std::int64_t begin, std::int64_t end) {
        AbsInt64_Dispatch(input + begin, output + begin, static_cast<std::size_t>(end - begin));
      });
}

} // namespace onnx_light_cpu
