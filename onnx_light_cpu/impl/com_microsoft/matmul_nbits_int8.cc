// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace onnx_light_cpu {

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
void MatMulNBitsInt8DotAvx2(const std::uint8_t *a, const std::uint8_t *b, std::int32_t *dots);
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVXVNNI
void MatMulNBitsInt8DotAvxVnni(const std::uint8_t *a, const std::uint8_t *b, std::int32_t *dots);
#endif
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512VNNI) && defined(ONNX_LIGHT_CPU_HAVE_AVX512BW)
void MatMulNBitsInt8DotAvx512Vnni(const std::uint8_t *a, const std::uint8_t *b, std::int32_t *dots);
#endif

namespace {

using DotFn = void (*)(const std::uint8_t *, const std::uint8_t *, std::int32_t *);

void DotScalar(const std::uint8_t *a, const std::uint8_t *b, std::int32_t *dots) {
  std::fill_n(dots, 8, 0);
  for (std::size_t quad = 0; quad < 8; ++quad) {
    for (std::size_t lane = 0; lane < 8; ++lane) {
      for (std::size_t p = 0; p < 4; ++p) {
        dots[lane] += static_cast<std::int32_t>(a[quad * 4 + p]) *
                      (static_cast<std::int32_t>(b[quad * 32 + lane * 4 + p]) - 128);
      }
    }
  }
}

DotFn SelectDot() {
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512VNNI) && defined(ONNX_LIGHT_CPU_HAVE_AVX512BW)
  if (CpuSupportsAvx512Vnni() && CpuSupportsAvx512BW()) {
    return MatMulNBitsInt8DotAvx512Vnni;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVXVNNI
  if (CpuSupportsAvxVnni()) {
    return MatMulNBitsInt8DotAvxVnni;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
    return MatMulNBitsInt8DotAvx2;
  }
#endif
  return DotScalar;
}

DotFn SelectedDot() {
  static const DotFn dot = SelectDot();
  return dot;
}

} // namespace

bool MatMulNBitsInt8Float32Available() { return SelectedDot() != DotScalar; }

void PackMatMulNBitsInt8(const std::uint8_t *b, const float *scales, std::uint8_t *weights,
                         std::int32_t *weight_sums, float *block_scales, std::size_t k,
                         std::size_t n) {
  const std::size_t blocks = k / 32;
  const std::size_t groups = n / 8 + (n % 8 != 0);
  for (std::size_t group = 0; group < groups; ++group) {
    for (std::size_t block = 0; block < blocks; ++block) {
      const std::size_t index = (group * blocks + block) * 8;
      for (std::size_t lane = 0; lane < 8; ++lane) {
        const std::size_t column = group * 8 + lane;
        std::int32_t sum = 0;
        block_scales[index + lane] = column < n ? scales[column * blocks + block] : 0.0f;
        for (std::size_t p = 0; p < 32; ++p) {
          const std::uint8_t value = column < n ? b[(column * blocks + block) * 32 + p] : 128;
          weights[index * 32 + (p / 4) * 32 + lane * 4 + p % 4] = value;
          sum += static_cast<std::int32_t>(value) - 128;
        }
        weight_sums[index + lane] = sum;
      }
    }
  }
}

const char *MatMulNBitsInt8Implementation() {
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512VNNI) && defined(ONNX_LIGHT_CPU_HAVE_AVX512BW)
  if (SelectedDot() == MatMulNBitsInt8DotAvx512Vnni) {
    return "int8_avx512_vnni";
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVXVNNI
  if (SelectedDot() == MatMulNBitsInt8DotAvxVnni) {
    return "int8_avx_vnni";
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (SelectedDot() == MatMulNBitsInt8DotAvx2) {
    return "int8_avx2";
  }
#endif
  return "int8_scalar";
}

void MatMulNBitsInt8Float32(const float *a, const std::uint8_t *weights,
                            const std::int32_t *weight_sums, const float *scales, const float *bias,
                            float *y, std::size_t rows, std::size_t k, std::size_t n,
                            std::int64_t max_participants) {
  const std::size_t blocks = k / 32;
  const std::size_t groups = n / 8 + (n % 8 != 0);
  const DotFn dot = SelectedDot();
  ExecuteRanges(
      static_cast<std::int64_t>(rows), ExecutionSchedule{1, 2, max_participants},
      [&](std::int64_t begin, std::int64_t end) {
        std::vector<std::uint8_t> quantized(k);
        std::vector<float> activation_scales(blocks);
        std::vector<float> activation_maxima(blocks);
        for (std::size_t row = static_cast<std::size_t>(begin); row < static_cast<std::size_t>(end);
             ++row) {
          bool finite = true;
          for (std::size_t block = 0; block < blocks; ++block) {
            const float *source = a + row * k + block * 32;
            float maximum = 0.0f;
            for (std::size_t p = 0; p < 32; ++p) {
              if (!std::isfinite(source[p])) {
                finite = false;
                break;
              }
              maximum = std::max(maximum, std::abs(source[p]));
            }
            if (!finite) {
              break;
            }
            activation_maxima[block] = maximum;
            activation_scales[block] = maximum / 63.0f;
            for (std::size_t p = 0; p < 32; ++p) {
              const float scaled = maximum == 0.0f    ? 0.0f
                                   : maximum < 1e-36f ? (source[p] / maximum) * 63.0f
                                                      : source[p] * (63.0f / maximum);
              quantized[block * 32 + p] =
                  static_cast<std::uint8_t>(static_cast<int>(std::nearbyint(scaled)) + 64);
            }
          }
          if (!finite) {
            for (std::size_t column = 0; column < n; ++column) {
              float sum = bias == nullptr ? 0.0f : bias[column];
              const std::size_t group = column / 8;
              const std::size_t lane = column % 8;
              for (std::size_t block = 0; block < blocks; ++block) {
                const std::size_t index = (group * blocks + block) * 8;
                for (std::size_t p = 0; p < 32; ++p) {
                  const auto weight = weights[index * 32 + (p / 4) * 32 + lane * 4 + p % 4];
                  sum +=
                      a[row * k + block * 32 + p] *
                      (static_cast<float>(static_cast<int>(weight) - 128) * scales[index + lane]);
                }
              }
              y[row * n + column] = sum;
            }
            continue;
          }
          for (std::size_t group = 0; group < groups; ++group) {
            float sums[8];
            for (std::size_t lane = 0; lane < 8; ++lane) {
              sums[lane] = bias != nullptr && group * 8 + lane < n ? bias[group * 8 + lane] : 0.0f;
            }
            for (std::size_t block = 0; block < blocks; ++block) {
              const std::size_t index = (group * blocks + block) * 8;
              std::int32_t dots[8];
              dot(quantized.data() + block * 32, weights + index * 32, dots);
              for (std::size_t lane = 0; lane < 8; ++lane) {
                const float factor =
                    activation_scales[block] == 0.0f && activation_maxima[block] != 0.0f
                        ? static_cast<float>(static_cast<double>(activation_maxima[block]) *
                                             static_cast<double>(scales[index + lane]) / 63.0)
                        : activation_scales[block] * scales[index + lane];
                sums[lane] +=
                    static_cast<float>(dots[lane] - 64 * weight_sums[index + lane]) * factor;
              }
            }
            for (std::size_t lane = 0; lane < std::min<std::size_t>(8, n - group * 8); ++lane) {
              y[row * n + group * 8 + lane] = sums[lane];
            }
          }
        }
      });
}

} // namespace onnx_light_cpu
