// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/logical/logical_kernels.h"

#include "onnx_light_cpu/impl/execution.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ONNX_LIGHT_CPU_X86 1
#include <immintrin.h>
#else
#define ONNX_LIGHT_CPU_X86 0
#endif

namespace onnx_light_cpu {

// ---------------------------------------------------------------------------
// NotBool implementations
// ---------------------------------------------------------------------------
//
// The elementwise logical negation of an ONNX ``bool`` tensor is a pure byte
// operation: every zero byte becomes ``1`` and every non-zero byte becomes
// ``0`` (matching ``numpy.logical_not``). Comparing each byte against zero and
// keeping the low bit produces a canonical ``0``/``1`` result regardless of the
// input byte value.

namespace {

void NotBool_Scalar(const uint8_t *input, uint8_t *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = static_cast<uint8_t>(input[i] == 0 ? 1 : 0);
  }
}

#if ONNX_LIGHT_CPU_X86

void NotBool_SSE2(const uint8_t *input, uint8_t *output, std::size_t count) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i one = _mm_set1_epi8(1);
  std::size_t i = 0;
  const std::size_t stride = 16;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));
    // 0xFF where the byte equals zero, 0x00 otherwise; keep the low bit -> 1/0.
    __m128i eq = _mm_cmpeq_epi8(v, zero);
    v = _mm_and_si128(eq, one);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = static_cast<uint8_t>(input[i] == 0 ? 1 : 0);
  }
}

void NotBool_AVX2(const uint8_t *input, uint8_t *output, std::size_t count) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i one = _mm256_set1_epi8(1);
  std::size_t i = 0;
  const std::size_t stride = 32;
  const std::size_t aligned_count = count - (count % stride);
  for (; i < aligned_count; i += stride) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
    __m256i eq = _mm256_cmpeq_epi8(v, zero);
    v = _mm256_and_si256(eq, one);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), v);
  }
  for (; i < count; ++i) {
    output[i] = static_cast<uint8_t>(input[i] == 0 ? 1 : 0);
  }
}

#endif // ONNX_LIGHT_CPU_X86

} // namespace

namespace {
void NotBool_Dispatch(const uint8_t *input, uint8_t *output, std::size_t count) {
#if ONNX_LIGHT_CPU_X86
  static const SimdLevel level = DetectSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512BW
  static const bool has_avx512bw = CpuSupportsAvx512BW();
  if (level >= SimdLevel::kAVX512 && has_avx512bw) {
    NotBool_AVX512(input, output, count);
    return;
  }
#endif
  if (level >= SimdLevel::kAVX2) {
    NotBool_AVX2(input, output, count);
    return;
  }
  if (level >= SimdLevel::kSSE2) {
    NotBool_SSE2(input, output, count);
    return;
  }
#endif
  NotBool_Scalar(input, output, count);
}
} // namespace

void NotBool(const uint8_t *input, uint8_t *output, std::size_t count) {
  NotBoolWithTuning(input, output, count, NotExecutionTuning{});
}

void NotBoolWithTuning(const uint8_t *input, uint8_t *output, std::size_t count,
                       const NotExecutionTuning &tuning) {
  if (count == 0) {
    return;
  }
  const auto bounded = [](std::size_t value) {
    return static_cast<std::int64_t>(
        std::min(value, static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())));
  };
  const ExecutionSchedule schedule{bounded(tuning.parallel_threshold_bytes),
                                   bounded(std::max<std::size_t>(tuning.target_block_bytes, 1)),
                                   tuning.max_participants == 0
                                       ? std::numeric_limits<std::int64_t>::max()
                                       : bounded(tuning.max_participants),
                                   bounded(tuning.preferred_participants)};
  auto execute = [input, output](std::int64_t begin, std::int64_t end) {
    NotBool_Dispatch(input + begin, output + begin, static_cast<std::size_t>(end - begin));
  };
  if (tuning.use_cost_model) {
    ExecuteCostedRanges(static_cast<std::int64_t>(count), ExecutionWorkCost{1.0, 1.0, 2.0},
                        schedule, ExecutionSimdLanes<std::uint8_t>(), std::move(execute));
  } else {
    ExecuteRanges(static_cast<std::int64_t>(count), schedule, ExecutionSimdLanes<std::uint8_t>(),
                  std::move(execute));
  }
}

} // namespace onnx_light_cpu
