// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/com_microsoft/moe_matvec.h"

#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/math/half_conversion.h"
#include "ml_cpu_compute/simd_level.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ml_cpu_compute::detail {

#if defined(ML_CPU_COMPUTE_HAVE_AVX512)
void MoEMatVecFloat32Avx512(const float *, const float *, float *, std::size_t, std::size_t,
                            std::size_t);
void MoEMatVecBFloat16Avx512(const std::uint16_t *, const std::uint16_t *, float *, std::size_t,
                             std::size_t, std::size_t);
void MoEMatVecFloat32BFloat16Avx512(const float *, const std::uint16_t *, float *, std::size_t,
                                    std::size_t, std::size_t);
#endif

#if defined(ML_CPU_COMPUTE_HAVE_AVX2_FMA)
void MoEMatVecFloat32Avx2Fma(const float *, const float *, float *, std::size_t, std::size_t,
                             std::size_t);
void MoEMatVecBFloat16Avx2Fma(const std::uint16_t *, const std::uint16_t *, float *, std::size_t,
                              std::size_t, std::size_t);
void MoEMatVecFloat32BFloat16Avx2Fma(const float *, const std::uint16_t *, float *, std::size_t,
                                     std::size_t, std::size_t);
#endif

#if defined(ML_CPU_COMPUTE_HAVE_AVX2_F16C)
void MoEMatVecFloat16Avx2F16c(const std::uint16_t *, const std::uint16_t *, float *, std::size_t,
                              std::size_t, std::size_t);
void MoEMatVecFloat32Float16Avx2F16c(const float *, const std::uint16_t *, float *, std::size_t,
                                     std::size_t, std::size_t);
#endif

namespace {

using FloatMatVecRange = void (*)(const float *, const float *, float *, std::size_t, std::size_t,
                                  std::size_t);
using HalfMatVecRange = void (*)(const std::uint16_t *, const std::uint16_t *, float *, std::size_t,
                                 std::size_t, std::size_t);
using MixedMatVecRange = void (*)(const float *, const std::uint16_t *, float *, std::size_t,
                                  std::size_t, std::size_t);

void MatVecFloat32Scalar(const float *input, const float *weights, float *output, std::size_t begin,
                         std::size_t end, std::size_t columns) {
  for (std::size_t row = begin; row < end; ++row) {
    float value = 0.0f;
    for (std::size_t column = 0; column < columns; ++column) {
      value += input[column] * weights[row * columns + column];
    }
    output[row] = value;
  }
}

template <bool BFloat16>
void MatVecHalfScalar(const std::uint16_t *input, const std::uint16_t *weights, float *output,
                      std::size_t begin, std::size_t end, std::size_t columns) {
  for (std::size_t row = begin; row < end; ++row) {
    float value = 0.0f;
    for (std::size_t column = 0; column < columns; ++column) {
      const float left =
          BFloat16 ? Bfloat16BitsToFloat(input[column]) : Float16BitsToFloat(input[column]);
      const float right = BFloat16 ? Bfloat16BitsToFloat(weights[row * columns + column])
                                   : Float16BitsToFloat(weights[row * columns + column]);
      value += left * right;
    }
    output[row] = value;
  }
}

template <bool BFloat16>
void MatVecMixedScalar(const float *input, const std::uint16_t *weights, float *output,
                       std::size_t begin, std::size_t end, std::size_t columns) {
  for (std::size_t row = begin; row < end; ++row) {
    float value = 0.0f;
    for (std::size_t column = 0; column < columns; ++column) {
      const float right = BFloat16 ? Bfloat16BitsToFloat(weights[row * columns + column])
                                   : Float16BitsToFloat(weights[row * columns + column]);
      value += input[column] * right;
    }
    output[row] = value;
  }
}

FloatMatVecRange SelectFloat32Range() {
#if defined(ML_CPU_COMPUTE_HAVE_AVX512)
  if (DetectSimdLevel() >= SimdLevel::kAVX512) {
    return MoEMatVecFloat32Avx512;
  }
#endif
#if defined(ML_CPU_COMPUTE_HAVE_AVX2_FMA)
  if (DetectSimdLevel() >= SimdLevel::kAVX2 && CpuSupportsFma()) {
    return MoEMatVecFloat32Avx2Fma;
  }
#endif
  return MatVecFloat32Scalar;
}

HalfMatVecRange SelectFloat16Range() {
#if defined(ML_CPU_COMPUTE_HAVE_AVX2_F16C)
  if (DetectSimdLevel() >= SimdLevel::kAVX2 && CpuSupportsF16C()) {
    return MoEMatVecFloat16Avx2F16c;
  }
#endif
  return MatVecHalfScalar<false>;
}

HalfMatVecRange SelectBFloat16Range() {
#if defined(ML_CPU_COMPUTE_HAVE_AVX512)
  if (DetectSimdLevel() >= SimdLevel::kAVX512) {
    return MoEMatVecBFloat16Avx512;
  }
#endif
#if defined(ML_CPU_COMPUTE_HAVE_AVX2_FMA)
  if (DetectSimdLevel() >= SimdLevel::kAVX2 && CpuSupportsFma()) {
    return MoEMatVecBFloat16Avx2Fma;
  }
#endif
  return MatVecHalfScalar<true>;
}

MixedMatVecRange SelectMixedFloat16Range() {
#if defined(ML_CPU_COMPUTE_HAVE_AVX2_F16C)
  if (DetectSimdLevel() >= SimdLevel::kAVX2 && CpuSupportsF16C()) {
    return MoEMatVecFloat32Float16Avx2F16c;
  }
#endif
  return MatVecMixedScalar<false>;
}

MixedMatVecRange SelectMixedBFloat16Range() {
#if defined(ML_CPU_COMPUTE_HAVE_AVX512)
  if (DetectSimdLevel() >= SimdLevel::kAVX512) {
    return MoEMatVecFloat32BFloat16Avx512;
  }
#endif
#if defined(ML_CPU_COMPUTE_HAVE_AVX2_FMA)
  if (DetectSimdLevel() >= SimdLevel::kAVX2 && CpuSupportsFma()) {
    return MoEMatVecFloat32BFloat16Avx2Fma;
  }
#endif
  return MatVecMixedScalar<true>;
}

template <typename Input, typename Weight, typename Range>
void MatVec(const Input *input, const Weight *weights, float *output, std::size_t rows,
            std::size_t columns, Range range) {
  const double operations = static_cast<double>(rows) * static_cast<double>(columns);
  if (operations < 512.0 * 1024.0 || ExecutionInParallelRegion()) {
    range(input, weights, output, 0, rows, columns);
    return;
  }
  const ExecutionWorkCost cost{static_cast<double>(sizeof(Input) + sizeof(Weight)) * columns,
                               sizeof(float), static_cast<double>(columns)};
  const ExecutionSchedule schedule{2, 1, std::numeric_limits<std::int64_t>::max(), 0};
  ExecuteCostedRanges(static_cast<std::int64_t>(rows), cost, schedule, std::int64_t{1},
                      [=](std::int64_t begin, std::int64_t end) {
                        range(input, weights, output, static_cast<std::size_t>(begin),
                              static_cast<std::size_t>(end), columns);
                      });
}

} // namespace

void MoEMatVecFloat32(const float *input, const float *weights, float *output, std::size_t rows,
                      std::size_t columns) {
  static const FloatMatVecRange range = SelectFloat32Range();
  MatVec(input, weights, output, rows, columns, range);
}

void MoEMatVecFloat16(const std::uint16_t *input, const std::uint16_t *weights, float *output,
                      std::size_t rows, std::size_t columns) {
  static const HalfMatVecRange range = SelectFloat16Range();
  MatVec(input, weights, output, rows, columns, range);
}

void MoEMatVecBFloat16(const std::uint16_t *input, const std::uint16_t *weights, float *output,
                       std::size_t rows, std::size_t columns) {
  static const HalfMatVecRange range = SelectBFloat16Range();
  MatVec(input, weights, output, rows, columns, range);
}

void MoEMatVecFloat32Float16(const float *input, const std::uint16_t *weights, float *output,
                             std::size_t rows, std::size_t columns) {
  static const MixedMatVecRange range = SelectMixedFloat16Range();
  MatVec(input, weights, output, rows, columns, range);
}

void MoEMatVecFloat32BFloat16(const float *input, const std::uint16_t *weights, float *output,
                              std::size_t rows, std::size_t columns) {
  static const MixedMatVecRange range = SelectMixedBFloat16Range();
  MatVec(input, weights, output, rows, columns, range);
}

} // namespace ml_cpu_compute::detail
