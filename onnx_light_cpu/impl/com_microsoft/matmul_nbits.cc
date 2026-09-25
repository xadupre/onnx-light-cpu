// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits.h"

#include "onnx_light_cpu/impl/checked_arithmetic.h"
#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits_panel.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace onnx_light_cpu {

#if defined(ONNX_LIGHT_CPU_HAVE_AVX512VNNI) && defined(ONNX_LIGHT_CPU_HAVE_AVX512BW)
void MatMulNBitsAccuracy4Float32Avx512Vnni(const float *a, const std::uint8_t *packed_weights,
                                           const std::int32_t *weight_sums,
                                           const float *block_scales, const float *bias, float *y,
                                           std::size_t rows, std::size_t k, std::size_t n,
                                           std::int64_t max_participants);
#endif

namespace {

struct FloatCodec {
  using Storage = float;
  static float Load(float value) noexcept { return value; }
  static float Store(float value) noexcept { return value; }
};

struct Float16Codec {
  using Storage = std::uint16_t;
  static float Load(std::uint16_t value) noexcept { return detail::Float16BitsToFloat(value); }
  static std::uint16_t Store(float value) noexcept { return detail::FloatToFloat16Bits(value); }
};

struct BFloat16Codec {
  using Storage = std::uint16_t;
  static float Load(std::uint16_t value) noexcept { return detail::Bfloat16BitsToFloat(value); }
  static std::uint16_t Store(float value) noexcept { return detail::FloatToBFloat16Bits(value); }
};

void NBitsPanelScalar(const float *a, const float *b, float *sums, std::size_t rows,
                      std::size_t depth) {
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t p = 0; p < depth; ++p) {
      for (std::size_t column = 0; column < detail::kNBitsColumns; ++column) {
        sums[row * detail::kNBitsColumns + column] +=
            a[row * detail::kNBitsBlock + p] * b[p * detail::kNBitsColumns + column];
      }
    }
  }
}

detail::NBitsPanelFn SelectPanel() {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (DetectSimdLevel() >= SimdLevel::kAVX512) {
    return detail::NBitsPanelAvx512;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
    return detail::NBitsPanelAvx2;
  }
#endif
  return NBitsPanelScalar;
}

detail::NBitsPanelFn PanelImplementation() {
  static const auto panel = SelectPanel();
  return panel;
}

template <typename Codec>
void MatMulInt4Panels(const typename Codec::Storage *a, const std::uint8_t *b,
                      const typename Codec::Storage *scales, const typename Codec::Storage *bias,
                      typename Codec::Storage *y, std::size_t rows, std::size_t k, std::size_t n,
                      std::size_t k_blocks, const ExecutionSchedule &schedule) {
  using namespace detail;
  const NBitsPanelFn accumulate = PanelImplementation();
  static_assert(sizeof(float) * (kNBitsBlock * kNBitsColumns + kNBitsRows * kNBitsBlock +
                                 kNBitsRows * kNBitsColumns) ==
                kMatMulNBitsInt4WorkspaceBytes);
  const std::size_t column_tiles = n / kNBitsColumns + (n % kNBitsColumns != 0);
  const std::size_t row_tiles = rows / kNBitsRows + (rows % kNBitsRows != 0);
  const std::size_t tiles = CheckedMultiply(row_tiles, column_tiles, "MatMulNBits", "panels");
  const auto tile_outputs =
      static_cast<std::int64_t>(std::min(rows, kNBitsRows) * std::min(n, kNBitsColumns));
  const auto to_panels = [tile_outputs](std::int64_t outputs) {
    outputs = std::max<std::int64_t>(outputs, 1);
    return outputs / tile_outputs + (outputs % tile_outputs != 0);
  };
  const bool parallel =
      rows * n >= static_cast<std::size_t>(std::max<std::int64_t>(schedule.min_parallel_size, 1));
  const ExecutionSchedule panel_schedule{1, to_panels(schedule.min_block_size),
                                         parallel ? schedule.max_participants : 1};
  ExecuteRanges(
      static_cast<std::int64_t>(tiles), panel_schedule, [&](std::int64_t begin, std::int64_t end) {
        // Bounded per callback, including decode: never expand or repack all weights.
        float weights[kNBitsBlock * kNBitsColumns]{};
        float activations[kNBitsRows * kNBitsBlock];
        float sums[kNBitsRows * kNBitsColumns];
        for (auto tile = begin; tile < end; ++tile) {
          const std::size_t first_row = (tile / column_tiles) * kNBitsRows;
          const std::size_t first_column = (tile % column_tiles) * kNBitsColumns;
          const auto mr = std::min(kNBitsRows, rows - first_row);
          const auto nr = std::min(kNBitsColumns, n - first_column);
          std::fill_n(sums, kNBitsRows * kNBitsColumns, 0.0f);
          if (bias != nullptr) {
            for (std::size_t r = 0; r < mr; ++r) {
              for (std::size_t c = 0; c < nr; ++c) {
                sums[r * kNBitsColumns + c] = Codec::Load(bias[first_column + c]);
              }
            }
          }
          for (std::size_t block = 0; block < k_blocks; ++block) {
            const auto depth = std::min(kNBitsBlock, k - block * kNBitsBlock);
            for (std::size_t c = 0; c < nr; ++c) {
              const auto index = (first_column + c) * k_blocks + block;
              const auto *packed = b + index * 16;
              const float scale = Codec::Load(scales[index]);
              for (std::size_t p = 0; p < depth; ++p) {
                const int q = (packed[p / 2] >> ((p % 2) * 4)) & 15;
                weights[p * kNBitsColumns + c] = static_cast<float>(q - 8) * scale;
              }
            }
            for (std::size_t r = 0; r < mr; ++r) {
              for (std::size_t p = 0; p < depth; ++p) {
                activations[r * kNBitsBlock + p] =
                    Codec::Load(a[(first_row + r) * k + block * kNBitsBlock + p]);
              }
            }
            accumulate(activations, weights, sums, mr, depth);
          }
          for (std::size_t r = 0; r < mr; ++r) {
            for (std::size_t c = 0; c < nr; ++c) {
              y[(first_row + r) * n + first_column + c] = Codec::Store(sums[r * kNBitsColumns + c]);
            }
          }
        }
      });
}

template <typename Codec>
void MatMulNBitsTyped(const void *a_raw, const std::uint8_t *b, const void *scales_raw,
                      const void *bias_raw, void *y_raw, std::size_t rows, std::size_t k,
                      std::size_t n, std::size_t bits, std::size_t block_size,
                      const MatMulNBitsExecutionTuning &tuning) {
  if (block_size != 32) {
    throw std::invalid_argument("onnx_light_cpu::MatMulNBits: block_size must be 32.");
  }
  if (bits != 2 && bits != 4 && bits != 8) {
    throw std::invalid_argument("onnx_light_cpu::MatMulNBits: bits must be 2, 4, or 8.");
  }
  if (rows == 0 || n == 0) {
    return;
  }
  using Storage = typename Codec::Storage;
  const auto *a = static_cast<const Storage *>(a_raw);
  const auto *scales = static_cast<const Storage *>(scales_raw);
  const auto *bias = static_cast<const Storage *>(bias_raw);
  auto *y = static_cast<Storage *>(y_raw);
  const std::size_t k_blocks =
      CheckedAdd(k, block_size - 1, "MatMulNBits", "K blocks") / block_size;
  const std::size_t values_per_byte = 8 / bits;
  const std::size_t blob_size =
      CheckedMultiply(block_size, bits, "MatMulNBits", "packed block bits") / 8;
  const std::uint8_t mask = static_cast<std::uint8_t>((1U << bits) - 1U);
  const float zero_point = static_cast<float>(1U << (bits - 1));
  const std::size_t packed_column_stride =
      CheckedMultiply(k_blocks, blob_size, "MatMulNBits", "packed column stride");
  CheckedMultiply(rows, k, "MatMulNBits", "activation element count");
  CheckedMultiply(n, packed_column_stride, "MatMulNBits", "packed weight element count");
  CheckedMultiply(n, k_blocks, "MatMulNBits", "scale element count");
  const std::size_t output_size = CheckedMultiply(rows, n, "MatMulNBits", "output element count");
  if (output_size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulNBits: output element count exceeds int64_t.");
  }
  const ExecutionSchedule schedule{
      static_cast<std::int64_t>(std::max<std::size_t>(tuning.parallel_threshold_outputs, 1)),
      static_cast<std::int64_t>(std::max<std::size_t>(tuning.target_block_outputs, 1)),
      tuning.max_participants,
  };
  if (bits == 4) {
    MatMulInt4Panels<Codec>(a, b, scales, bias, y, rows, k, n, k_blocks, schedule);
    return;
  }
  ExecuteRanges(
      static_cast<std::int64_t>(output_size), schedule, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t output_index = begin; output_index < end; ++output_index) {
          const std::size_t index = static_cast<std::size_t>(output_index);
          const std::size_t row = index / n;
          const std::size_t column = index - row * n;
          const Storage *a_row = a + row * k;
          const std::uint8_t *b_column = b + column * packed_column_stride;
          const Storage *column_scales = scales + column * k_blocks;
          float sum = bias == nullptr ? 0.0f : Codec::Load(bias[column]);
          for (std::size_t block = 0; block < k_blocks; ++block) {
            const std::size_t block_begin = block * block_size;
            const std::size_t block_end = std::min(block_begin + block_size, k);
            const std::uint8_t *packed = b_column + block * blob_size;
            const float scale = Codec::Load(column_scales[block]);
            for (std::size_t offset = 0; block_begin + offset < block_end; ++offset) {
              const std::uint8_t byte = packed[offset / values_per_byte];
              const std::size_t shift = (offset % values_per_byte) * bits;
              const std::uint8_t quantized = static_cast<std::uint8_t>((byte >> shift) & mask);
              sum += Codec::Load(a_row[block_begin + offset]) *
                     (static_cast<float>(quantized) - zero_point) * scale;
            }
          }
          y[index] = Codec::Store(sum);
        }
      });
}

} // namespace

bool MatMulNBitsAccuracy4Float32Available() {
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512VNNI) && defined(ONNX_LIGHT_CPU_HAVE_AVX512BW)
  return CpuSupportsAvx512Vnni() && CpuSupportsAvx512BW();
#else
  return false;
#endif
}

void MatMulNBitsAccuracy4Float32(const float *a, const std::uint8_t *packed_weights,
                                 const std::int32_t *weight_sums, const float *block_scales,
                                 const float *bias, float *y, std::size_t rows, std::size_t k,
                                 std::size_t n, std::int64_t max_participants) {
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512VNNI) && defined(ONNX_LIGHT_CPU_HAVE_AVX512BW)
  if (CpuSupportsAvx512Vnni() && CpuSupportsAvx512BW()) {
    MatMulNBitsAccuracy4Float32Avx512Vnni(a, packed_weights, weight_sums, block_scales, bias, y,
                                          rows, k, n, max_participants);
    return;
  }
#endif
  throw std::runtime_error("MatMulNBits accuracy-level-4 kernel is unavailable.");
}

const char *MatMulNBitsInt4Implementation() {
  const auto panel = PanelImplementation();
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (panel == detail::NBitsPanelAvx512) {
    return "int4_panel_avx512";
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (panel == detail::NBitsPanelAvx2) {
    return "int4_panel_avx2";
  }
#endif
  (void)panel;
  return "int4_panel_scalar";
}

void MatMulNBits(const void *a, const std::uint8_t *b, const void *scales, const void *bias,
                 void *y, DataType data_type, std::size_t rows, std::size_t k, std::size_t n,
                 std::size_t bits, std::size_t block_size,
                 const MatMulNBitsExecutionTuning &tuning) {
  switch (data_type) {
  case DataType::FLOAT:
    MatMulNBitsTyped<FloatCodec>(a, b, scales, bias, y, rows, k, n, bits, block_size, tuning);
    return;
  case DataType::FLOAT16:
    MatMulNBitsTyped<Float16Codec>(a, b, scales, bias, y, rows, k, n, bits, block_size, tuning);
    return;
  case DataType::BFLOAT16:
    MatMulNBitsTyped<BFloat16Codec>(a, b, scales, bias, y, rows, k, n, bits, block_size, tuning);
    return;
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulNBits: data type must be FLOAT, FLOAT16, or BFLOAT16.");
  }
}

void MatMulNBitsFloat32(const float *a, const std::uint8_t *b, const float *scales,
                        const float *bias, float *y, std::size_t rows, std::size_t k, std::size_t n,
                        std::size_t block_size, const MatMulNBitsExecutionTuning &tuning) {
  MatMulNBits(a, b, scales, bias, y, DataType::FLOAT, rows, k, n, 4, block_size, tuning);
}

} // namespace onnx_light_cpu
