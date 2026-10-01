// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <algorithm>
#include <atomic>
#include <cmath>

namespace onnx_light_cpu {
namespace {

using TanhRange = void (*)(const float *, float *, std::size_t);

constexpr UnaryExecutionTuning kTanhFloat32Tuning{256 * 1024, 128 * 1024, 32, false};
constexpr UnaryExecutionTuning kTanhFloat32SmallTuning{128 * 1024, 64 * 1024, 4, false};
constexpr UnaryExecutionTuning kTanhFloat32Avx512Tuning{128 * 1024, 64 * 1024, 32, false};
constexpr UnaryExecutionTuning kTanhHalfTuning{128 * 1024, 64 * 1024, 32, false};

TanhRange GetTanhRange() {
  static const TanhRange function = []() -> TanhRange {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
    if (DetectSimdLevel() >= SimdLevel::kAVX512) {
      return &TanhFloat32_AVX512;
    }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_FMA
    if (DetectSimdLevel() >= SimdLevel::kAVX2 && CpuSupportsFma()) {
      return &TanhFloat32_AVX2_FMA;
    }
#endif
    return &TanhFloat32_Scalar;
  }();
  return function;
}

template <bool BFloat16>
void TanhHalf(const std::uint16_t *input, std::uint16_t *output, std::size_t count) {
#if defined(ONNX_LIGHT_CPU_HAVE_AVX2_FMA) && defined(ONNX_LIGHT_CPU_HAVE_F16C)
  if constexpr (!BFloat16) {
    static const bool use_avx2_f16c =
        DetectSimdLevel() >= SimdLevel::kAVX2 && CpuSupportsFma() && CpuSupportsF16C();
    if (use_avx2_f16c) {
      ExecuteUnaryRanges<std::uint16_t>(
          count, kTanhHalfTuning, [input, output](std::int64_t begin, std::int64_t end) {
            TanhFloat16_AVX2_FMA(input + begin, output + begin,
                                 static_cast<std::size_t>(end - begin));
          });
      return;
    }
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512BF16
  if constexpr (BFloat16) {
    static const bool use_avx512bf16 =
        DetectSimdLevel() >= SimdLevel::kAVX512 && CpuSupportsAvx512Bf16();
    if (use_avx512bf16 && count >= 256) {
      ExecuteUnaryRanges<std::uint16_t>(
          count, kTanhHalfTuning, [input, output](std::int64_t begin, std::int64_t end) {
            TanhBFloat16_AVX512BF16(input + begin, output + begin,
                                    static_cast<std::size_t>(end - begin));
          });
      return;
    }
  }
#endif
  const auto function = GetTanhRange();
  ExecuteUnaryRanges<std::uint16_t>(
      count, kTanhHalfTuning, [input, output, function](std::int64_t begin, std::int64_t end) {
        constexpr std::size_t kBlockSize = 1024;
        alignas(64) float values[kBlockSize];
        while (begin < end) {
          const auto block =
              static_cast<std::size_t>(std::min<std::int64_t>(end - begin, kBlockSize));
          if constexpr (BFloat16) {
            detail::ConvertBFloat16ToFloat32(input + begin, values, block);
          } else {
            detail::ConvertFloat16ToFloat32(input + begin, values, block);
          }
          function(values, values, block);
          if constexpr (BFloat16) {
            detail::ConvertFloat32ToBFloat16(values, output + begin, block);
          } else {
            detail::ConvertFloat32ToFloat16(values, output + begin, block);
          }
          begin += static_cast<std::int64_t>(block);
        }
      });
}

} // namespace

void TanhFloat32_Scalar(const float *input, float *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = std::tanh(input[i]);
  }
}

void TanhFloat32(const float *input, float *output, std::size_t count) {
  const auto function = GetTanhRange();
  auto tuning = count <= 64 * 1024 ? kTanhFloat32SmallTuning : kTanhFloat32Tuning;
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  static const bool use_avx512 = DetectSimdLevel() >= SimdLevel::kAVX512;
  if (use_avx512) {
    tuning = kTanhFloat32Avx512Tuning;
  }
#endif
  constexpr std::size_t tile = 16 * 1024;
  std::atomic<std::size_t> next{0};
  const bool dynamic =
      count >= 512 * 1024 && ExecutionThreadCount() > 1 && !ExecutionInParallelRegion();
  ExecuteUnaryRanges<float>(
      count, tuning,
      [input, output, function, &next, dynamic, count, tile](std::int64_t begin, std::int64_t end) {
        if (dynamic) {
          for (std::size_t first = next.fetch_add(tile, std::memory_order_relaxed); first < count;
               first = next.fetch_add(tile, std::memory_order_relaxed)) {
            function(input + first, output + first, std::min(tile, count - first));
          }
          return;
        }
        function(input + begin, output + begin, static_cast<std::size_t>(end - begin));
      });
}

void TanhFloat16(const std::uint16_t *input, std::uint16_t *output, std::size_t count) {
  TanhHalf<false>(input, output, count);
}

void TanhBFloat16(const std::uint16_t *input, std::uint16_t *output, std::size_t count) {
  TanhHalf<true>(input, output, count);
}

} // namespace onnx_light_cpu
