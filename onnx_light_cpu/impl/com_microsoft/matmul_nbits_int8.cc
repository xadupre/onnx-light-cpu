// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/checked_arithmetic.h"
#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits.h"
#include "onnx_light_cpu/impl/com_microsoft/matmul_nbits_int8_tile.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace onnx_light_cpu {

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
void MatMulNBitsInt8TileAvx2(const MatMulNBitsInt8Tile &tile);
bool QuantizeMatMulNBitsInt8RowAvx2(const float *input, std::uint8_t *quantized, double *scales,
                                    std::size_t k);
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVXVNNI
void MatMulNBitsInt8TileAvxVnni(const MatMulNBitsInt8Tile &tile);
#endif
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512VNNI) && defined(ONNX_LIGHT_CPU_HAVE_AVX512BW)
void MatMulNBitsInt8TileAvx512Vnni(const MatMulNBitsInt8Tile &tile);
#endif

namespace {

using TileFn = void (*)(const MatMulNBitsInt8Tile &);
using QuantizeFn = bool (*)(const float *, std::uint8_t *, double *, std::size_t);

QuantizeFn SelectedQuantizer() {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  static const QuantizeFn quantize =
      DetectSimdLevel() >= SimdLevel::kAVX2 ? QuantizeMatMulNBitsInt8RowAvx2 : nullptr;
  return quantize;
#else
  return nullptr;
#endif
}

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

void TileScalar(const MatMulNBitsInt8Tile &tile) { ComputeMatMulNBitsInt8Tile<DotScalar>(tile); }

TileFn SelectTile() {
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512VNNI) && defined(ONNX_LIGHT_CPU_HAVE_AVX512BW)
  if (CpuSupportsAvx512Vnni() && CpuSupportsAvx512BW()) {
    return MatMulNBitsInt8TileAvx512Vnni;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVXVNNI
  if (CpuSupportsAvxVnni()) {
    return MatMulNBitsInt8TileAvxVnni;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
    return MatMulNBitsInt8TileAvx2;
  }
#endif
  return TileScalar;
}

TileFn SelectedTile() {
  static const TileFn tile = SelectTile();
  return tile;
}

ExecutionSchedule Int8Schedule(std::size_t n, const MatMulNBitsExecutionTuning &tuning) {
  const auto output_rows = [n](std::size_t outputs) {
    return static_cast<std::int64_t>(outputs / n + (outputs % n != 0));
  };
  return {output_rows(tuning.parallel_threshold_outputs), output_rows(tuning.target_block_outputs),
          tuning.max_participants};
}

ExecutionSchedule Int8SmallSchedule(std::size_t rows, std::size_t k, std::size_t n,
                                    const MatMulNBitsExecutionTuning &tuning) {
  const auto minimum_columns =
      tuning.parallel_threshold_outputs / rows + (tuning.parallel_threshold_outputs % rows != 0);
  if (n < minimum_columns) {
    return {1, 1, 1};
  }
  const auto groups_per_worker = static_cast<std::int64_t>(std::max<std::size_t>(1, 16384 / k));
  const auto groups_for_outputs = [rows](std::size_t outputs) {
    const auto columns = rows * 8;
    return static_cast<std::int64_t>(outputs / columns + (outputs % columns != 0));
  };
  return {std::max(groups_per_worker * 2, groups_for_outputs(tuning.parallel_threshold_outputs)),
          std::max(groups_per_worker, groups_for_outputs(tuning.target_block_outputs)),
          tuning.max_participants};
}

void ComputeFloatRow(const float *a, const std::uint8_t *weights, const float *scales,
                     const float *bias, float *y, std::size_t k, std::size_t n,
                     std::size_t first_column = 0, std::size_t last_column = 0) {
  const std::size_t blocks = k / 32;
  if (last_column == 0) {
    last_column = n;
  }
  for (std::size_t column = first_column; column < last_column; ++column) {
    float sum = bias == nullptr ? 0.0f : bias[column];
    const std::size_t group = column / 8;
    const std::size_t lane = column % 8;
    for (std::size_t block = 0; block < blocks; ++block) {
      const std::size_t index = (group * blocks + block) * 8;
      for (std::size_t p = 0; p < 32; ++p) {
        const auto weight = weights[index * 32 + (p / 4) * 32 + lane * 4 + p % 4];
        sum += a[block * 32 + p] *
               (static_cast<float>(static_cast<int>(weight) - 128) * scales[index + lane]);
      }
    }
    y[column] = sum;
  }
}

} // namespace

bool MatMulNBitsInt8Float32Available() { return SelectedTile() != TileScalar; }

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
  if (SelectedTile() == MatMulNBitsInt8TileAvx512Vnni) {
    return "int8_avx512_vnni";
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVXVNNI
  if (SelectedTile() == MatMulNBitsInt8TileAvxVnni) {
    return "int8_avx_vnni";
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
  if (SelectedTile() == MatMulNBitsInt8TileAvx2) {
    return "int8_avx2";
  }
#endif
  return "int8_scalar";
}

std::size_t MatMulNBitsInt8WorkspaceElements(std::size_t rows, std::size_t k, std::size_t n,
                                             const MatMulNBitsExecutionTuning &tuning) {
  if (rows == 0 || n == 0) {
    return 0;
  }
  if (k % 32 != 0) {
    throw std::invalid_argument("MatMulNBits INT8 preparation requires K divisible by 32.");
  }
  if (rows > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    ThrowArithmeticError("MatMulNBits", "INT8 row count", "overflows int64_t");
  }
  if (rows <= 4) {
    const std::size_t blocks = k / 32;
    return CheckedMultiply(rows, k / 8 + blocks, "MatMulNBits", "INT8 workspace");
  }
  const auto participants = static_cast<std::size_t>(
      ExecutionBlockCount(static_cast<std::int64_t>(rows), Int8Schedule(n, tuning)));
  // Four byte rows and four rows of double scales, one scale per 32 activations.
  const std::size_t per_participant = CheckedMultiply(k, 5, "MatMulNBits", "INT8 workspace") / 8;
  return CheckedMultiply(participants, per_participant, "MatMulNBits", "INT8 workspace");
}

void MatMulNBitsInt8Float32(const float *a, const std::uint8_t *weights,
                            const std::int32_t *weight_sums, const float *scales, const float *bias,
                            float *y, std::size_t rows, std::size_t k, std::size_t n,
                            const MatMulNBitsExecutionTuning &tuning, std::span<double> workspace) {
  if (rows == 0 || n == 0) {
    return;
  }
  const std::size_t blocks = k / 32;
  const std::size_t groups = n / 8 + (n % 8 != 0);
  const std::size_t workspace_elements = MatMulNBitsInt8WorkspaceElements(rows, k, n, tuning);
  if (k == 0) {
    const auto count = CheckedMultiply(rows, n, "MatMulNBits", "INT8 output count");
    for (std::size_t index = 0; index < count; ++index) {
      y[index] = bias == nullptr ? 0.0f : bias[index % n];
    }
    return;
  }
  const TileFn tile = SelectedTile();
  const QuantizeFn quantize = SelectedQuantizer();
  const ExecutionSchedule schedule = Int8Schedule(n, tuning);
  const auto participants =
      static_cast<std::size_t>(ExecutionBlockCount(static_cast<std::int64_t>(rows), schedule));
  const std::size_t rows_per_range = rows / participants + (rows % participants != 0);
  std::vector<double> owned_workspace;
  if (workspace.empty()) {
    owned_workspace.resize(workspace_elements);
    workspace = owned_workspace;
  } else if (workspace.size() < workspace_elements) {
    throw std::invalid_argument("MatMulNBits INT8 workspace is too small.");
  }
  const bool finite_scales = std::all_of(scales, scales + groups * blocks * 8,
                                         [](float scale) { return std::isfinite(scale); });
  if (rows <= 4) {
    auto *quantized = reinterpret_cast<std::uint8_t *>(workspace.data());
    double *activation_scales = workspace.data() + rows * k / 8;
    bool finite_rows[4]{};
    for (std::size_t row = 0; row < rows; ++row) {
      finite_rows[row] = finite_scales;
      if (!finite_scales) {
        continue;
      }
      if (quantize != nullptr &&
          quantize(a + row * k, quantized + row * k, activation_scales + row * blocks, k)) {
        continue;
      }
      for (std::size_t block = 0; block < blocks; ++block) {
        const float *source = a + row * k + block * 32;
        float maximum = 0.0f;
        for (std::size_t p = 0; p < 32; ++p) {
          finite_rows[row] &= std::isfinite(source[p]);
          maximum = std::max(maximum, std::abs(source[p]));
        }
        activation_scales[row * blocks + block] =
            finite_rows[row] ? static_cast<double>(maximum) / 127.0 : 0.0;
        for (std::size_t p = 0; p < 32; ++p) {
          const float scaled = !finite_rows[row] || maximum == 0.0f ? 0.0f
                               : maximum < 1e-36f                   ? (source[p] / maximum) * 127.0f
                                                  : source[p] * (127.0f / maximum);
          quantized[row * k + block * 32 + p] =
              static_cast<std::uint8_t>(static_cast<int>(std::nearbyint(scaled)) + 128);
        }
      }
    }
    const auto small_schedule = Int8SmallSchedule(rows, k, n, tuning);
    ExecuteRanges(static_cast<std::int64_t>(groups), small_schedule,
                  [&](std::int64_t begin, std::int64_t end) noexcept {
                    for (auto group = static_cast<std::size_t>(begin);
                         group < static_cast<std::size_t>(end); ++group) {
                      const std::size_t index = group * blocks * 8;
                      const std::size_t column = group * 8;
                      if (finite_scales) {
                        tile({quantized, activation_scales, weights + index * 32,
                              weight_sums + index, scales + index,
                              bias == nullptr ? nullptr : bias + column, y + column, rows, k, n,
                              std::min<std::size_t>(8, n - column)});
                      }
                      for (std::size_t row = 0; row < rows; ++row) {
                        if (!finite_rows[row]) {
                          ComputeFloatRow(a + row * k, weights, scales, bias, y + row * n, k, n,
                                          column, std::min(column + 8, n));
                        }
                      }
                    }
                  });
    return;
  }
  ExecuteRanges(
      static_cast<std::int64_t>(rows), schedule,
      [&](std::int64_t begin, std::int64_t end) noexcept {
        double *scratch = workspace.data() + (static_cast<std::size_t>(begin) / rows_per_range) *
                                                 (workspace_elements / participants);
        auto *quantized = reinterpret_cast<std::uint8_t *>(scratch);
        double *activation_scales = scratch + k / 2;
        for (std::size_t first = static_cast<std::size_t>(begin);
             first < static_cast<std::size_t>(end); first += 4) {
          const std::size_t tile_rows = std::min<std::size_t>(4, end - first);
          bool finite_rows[4]{};
          if (!finite_scales) {
            for (std::size_t row = first; row < first + tile_rows; ++row) {
              ComputeFloatRow(a + row * k, weights, scales, bias, y + row * n, k, n);
            }
            continue;
          }
          for (std::size_t row = 0; row < tile_rows; ++row) {
            finite_rows[row] = true;
            if (quantize != nullptr && quantize(a + (first + row) * k, quantized + row * k,
                                                activation_scales + row * blocks, k)) {
              continue;
            }
            for (std::size_t block = 0; block < blocks; ++block) {
              const float *source = a + (first + row) * k + block * 32;
              float maximum = 0.0f;
              for (std::size_t p = 0; p < 32; ++p) {
                finite_rows[row] &= std::isfinite(source[p]);
                maximum = std::max(maximum, std::abs(source[p]));
              }
              activation_scales[row * blocks + block] =
                  finite_rows[row] ? static_cast<double>(maximum) / 127.0 : 0.0;
              for (std::size_t p = 0; p < 32; ++p) {
                const float scaled = !finite_rows[row] || maximum == 0.0f ? 0.0f
                                     : maximum < 1e-36f ? (source[p] / maximum) * 127.0f
                                                        : source[p] * (127.0f / maximum);
                const std::size_t index = row * k + block * 32 + p;
                quantized[index] =
                    static_cast<std::uint8_t>(static_cast<int>(std::nearbyint(scaled)) + 128);
              }
            }
          }
          for (std::size_t group = 0; group < groups; ++group) {
            const std::size_t index = group * blocks * 8;
            tile({quantized, activation_scales, weights + index * 32, weight_sums + index,
                  scales + index, bias == nullptr ? nullptr : bias + group * 8,
                  y + first * n + group * 8, tile_rows, k, n,
                  std::min<std::size_t>(8, n - group * 8)});
          }
          for (std::size_t row = 0; row < tile_rows; ++row) {
            if (!finite_rows[row]) {
              ComputeFloatRow(a + (first + row) * k, weights, scales, bias, y + (first + row) * n,
                              k, n);
            }
          }
        }
      });
}

} // namespace onnx_light_cpu
