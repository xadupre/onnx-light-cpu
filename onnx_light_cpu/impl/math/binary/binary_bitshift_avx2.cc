// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_bitshift.h"

#include <array>
#include <immintrin.h>
#include <limits>

namespace onnx_light_cpu::detail {
namespace {

template <typename T> __m256i LoadShiftValues_AVX2(const T *values) {
  if constexpr (sizeof(T) == 8 || sizeof(T) == 4) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(values));
  } else if constexpr (sizeof(T) == 2) {
    return _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(values)));
  } else {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(values)));
  }
}

template <typename T> __m256i BroadcastShiftValue_AVX2(T value) {
  if constexpr (sizeof(T) == 8) {
    return _mm256_set1_epi64x(static_cast<std::int64_t>(value));
  } else {
    return _mm256_set1_epi32(static_cast<std::int32_t>(value));
  }
}

template <typename T> void StoreShiftValues_AVX2(T *output, __m256i values) {
  if constexpr (sizeof(T) == 8 || sizeof(T) == 4) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output), values);
  } else {
    const __m128i low = _mm256_castsi256_si128(values);
    const __m128i high = _mm256_extracti128_si256(values, 1);
    const __m128i packed16 = _mm_packus_epi32(low, high);
    if constexpr (sizeof(T) == 2) {
      _mm_storeu_si128(reinterpret_cast<__m128i *>(output), packed16);
    } else {
      const __m128i packed8 = _mm_packus_epi16(packed16, _mm_setzero_si128());
      _mm_storel_epi64(reinterpret_cast<__m128i *>(output), packed8);
    }
  }
}

template <typename T, bool Left> __m256i ShiftValues_AVX2(__m256i values, __m256i shifts) {
  __m256i shifted;
  if constexpr (sizeof(T) == 8) {
    shifted = Left ? _mm256_sllv_epi64(values, shifts) : _mm256_srlv_epi64(values, shifts);
  } else {
    shifted = Left ? _mm256_sllv_epi32(values, shifts) : _mm256_srlv_epi32(values, shifts);
  }
  if constexpr (sizeof(T) == 2) {
    return _mm256_and_si256(shifted, _mm256_set1_epi32(0xFFFF));
  } else if constexpr (sizeof(T) == 1) {
    return _mm256_and_si256(shifted, _mm256_set1_epi32(0xFF));
  } else {
    return shifted;
  }
}

} // namespace

template <typename T, bool Left, ShiftInputMode Mode>
void BinaryBitShift_AVX2(const T *left, const T *right, T *out, std::size_t count) {
  constexpr std::size_t kLanes = sizeof(T) == 8 ? 4 : 8;
  const __m256i left_scalar = Mode == ShiftInputMode::kLeftScalar
                                  ? BroadcastShiftValue_AVX2(left[0])
                                  : _mm256_setzero_si256();
  const __m256i right_scalar = Mode == ShiftInputMode::kRightScalar
                                   ? BroadcastShiftValue_AVX2(right[0])
                                   : _mm256_setzero_si256();
  std::size_t i = 0;
  for (; i + kLanes <= count; i += kLanes) {
    const __m256i values =
        Mode == ShiftInputMode::kLeftScalar ? left_scalar : LoadShiftValues_AVX2(left + i);
    const __m256i shifts =
        Mode == ShiftInputMode::kRightScalar ? right_scalar : LoadShiftValues_AVX2(right + i);
    StoreShiftValues_AVX2(out + i, ShiftValues_AVX2<T, Left>(values, shifts));
  }
  BinaryBitShift_Scalar<T, Left, Mode>(left + (Mode == ShiftInputMode::kLeftScalar ? 0 : i),
                                       right + (Mode == ShiftInputMode::kRightScalar ? 0 : i),
                                       out + i, count - i);
}

template <typename T> bool BinaryBitShiftHasInvalidAmount_AVX2(const T *shifts, std::size_t count) {
  constexpr std::size_t kLanes = 32 / sizeof(T);
  __m256i maximum = _mm256_setzero_si256();
  std::size_t i = 0;
  for (; i + kLanes <= count; i += kLanes) {
    const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(shifts + i));
    if constexpr (sizeof(T) == 1) {
      maximum = _mm256_max_epu8(maximum, values);
    } else if constexpr (sizeof(T) == 2) {
      maximum = _mm256_max_epu16(maximum, values);
    } else if constexpr (sizeof(T) == 4) {
      maximum = _mm256_max_epu32(maximum, values);
    } else {
      const __m256i sign = _mm256_set1_epi64x(std::numeric_limits<std::int64_t>::min());
      const __m256i greater =
          _mm256_cmpgt_epi64(_mm256_xor_si256(values, sign), _mm256_xor_si256(maximum, sign));
      maximum = _mm256_blendv_epi8(maximum, values, greater);
    }
  }
  alignas(32) std::array<T, kLanes> lane_maxima;
  _mm256_store_si256(reinterpret_cast<__m256i *>(lane_maxima.data()), maximum);
  const T bit_width = static_cast<T>(sizeof(T) * 8);
  for (T value : lane_maxima) {
    if (value >= bit_width) {
      return true;
    }
  }
  for (; i < count; ++i) {
    if (shifts[i] >= bit_width) {
      return true;
    }
  }
  return false;
}

#define ONNX_LIGHT_CPU_INSTANTIATE_SHIFT_MODE(T, Left, Mode)                                       \
  template void BinaryBitShift_AVX2<T, Left, ShiftInputMode::Mode>(const T *, const T *, T *,      \
                                                                   std::size_t);
#define ONNX_LIGHT_CPU_INSTANTIATE_SHIFT_DIRECTION(T, Left)                                        \
  ONNX_LIGHT_CPU_INSTANTIATE_SHIFT_MODE(T, Left, kContiguous)                                      \
  ONNX_LIGHT_CPU_INSTANTIATE_SHIFT_MODE(T, Left, kLeftScalar)                                      \
  ONNX_LIGHT_CPU_INSTANTIATE_SHIFT_MODE(T, Left, kRightScalar)
#define ONNX_LIGHT_CPU_INSTANTIATE_SHIFT(T)                                                        \
  ONNX_LIGHT_CPU_INSTANTIATE_SHIFT_DIRECTION(T, true)                                              \
  ONNX_LIGHT_CPU_INSTANTIATE_SHIFT_DIRECTION(T, false)                                             \
  template bool BinaryBitShiftHasInvalidAmount_AVX2<T>(const T *, std::size_t);

ONNX_LIGHT_CPU_INSTANTIATE_SHIFT(std::uint8_t)
ONNX_LIGHT_CPU_INSTANTIATE_SHIFT(std::uint16_t)
ONNX_LIGHT_CPU_INSTANTIATE_SHIFT(std::uint32_t)
ONNX_LIGHT_CPU_INSTANTIATE_SHIFT(std::uint64_t)

#undef ONNX_LIGHT_CPU_INSTANTIATE_SHIFT
#undef ONNX_LIGHT_CPU_INSTANTIATE_SHIFT_DIRECTION
#undef ONNX_LIGHT_CPU_INSTANTIATE_SHIFT_MODE

} // namespace onnx_light_cpu::detail
