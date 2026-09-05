// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Portable driver and scalar sibling for integer matrix multiplication. This
// translation unit is compiled at the project's baseline SIMD flags; optional
// AVX2, AVX-512 VNNI, AMX, and NEON kernels live in ISA-specific translation
// units and are only dispatched to when the running CPU supports them.
//
// Exactness relies on decomposing the zero-point-shifted accumulation into a
// raw byte dot-product plus row/column sum corrections. With
//
//   uA = trueA + oa   (0..255, oa = 128 when A is signed else 0)
//   sB = trueB - ob   (-128..127, ob = 128 when B is unsigned else 0)
//
// the ``vpdpbusd`` accumulator vp = sum(uA * sB) reconstructs the desired
// output S = sum((trueA - az) * (trueB - bz)) exactly:
//
//   S = vp + saT * (ob - bz) + sbT * (-oa - az) + depth * (oa * ob + az * bz)
//
// where saT / sbT are the true (zero-point-free) row / column sums. Every term
// is evaluated in 64-bit integers and truncated to the low 32 bits, matching
// the modulo-2^32 semantics of ``MatMulInteger``.

#include "onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_vnni.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/simd_level.h"

#ifdef ONNX_LIGHT_CPU_HAVE_AMX_INT8
#include "onnx_light_cpu/impl/math/gemm/amx/gemm_amx_int8.h"
#include "onnx_light_cpu/impl/math/gemm/amx/gemm_amx_tile.h"
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD
#include "onnx_light_cpu/impl/arm_simd_level.h"
#include "onnx_light_cpu/impl/math/gemm/arm/gemm_kernel_arm.h"
#endif

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace onnx_light_cpu {

namespace detail {

std::int32_t IntegerDotU8S8Scalar(const std::uint8_t *ua, const std::int8_t *sb,
                                  std::int64_t depth) {
  std::uint32_t accumulator = 0;
  for (std::int64_t i = 0; i < depth; ++i) {
    accumulator += static_cast<std::uint32_t>(static_cast<std::int32_t>(ua[i]) *
                                              static_cast<std::int32_t>(sb[i]));
  }
  return std::bit_cast<std::int32_t>(accumulator);
}

namespace {

std::int64_t ReadPacked4Bit(const std::uint8_t *data, std::int64_t index, bool is_signed) {
  const std::uint8_t packed = data[static_cast<std::size_t>(index) >> 1];
  const unsigned shift = static_cast<unsigned>(index & 1) * 4u;
  const std::int64_t nibble = (packed >> shift) & 0x0fu;
  return is_signed && nibble >= 8 ? nibble - 16 : nibble;
}

// Unpacks the disjoint byte-pair range ``[pair_begin, pair_end)`` of
// ``packed`` into ``raw`` (each pair maps to exactly two disjoint output
// elements). This is a plain (non-lambda) function so the single-participant
// fast path in ``UnpackPacked4BitBuffer`` below can call it directly with the
// same codegen as a hand-inlined loop, instead of indirecting through
// ``ExecuteRanges``'s generic closure plumbing.
void UnpackPacked4BitPairRange(const std::uint8_t *packed, bool is_signed, std::int8_t *raw,
                               std::int64_t pair_begin, std::int64_t pair_end) {
  for (std::int64_t i = pair_begin; i < pair_end; ++i) {
    const std::uint8_t byte = packed[i];
    int low = byte & 0x0f;
    int high = (byte >> 4) & 0x0f;
    if (is_signed) {
      low -= (low >= 8) ? 16 : 0;
      high -= (high >= 8) ? 16 : 0;
    }
    raw[2 * i] = static_cast<std::int8_t>(low);
    raw[2 * i + 1] = static_cast<std::int8_t>(high);
  }
}

// Unpacks the first ``count`` logical INT4/UINT4 elements of ``packed`` (the
// same layout as ``ReadPacked4Bit``: even index in the low nibble, odd index
// in the high nibble, no padding) into ``raw``, one raw signed value per
// element. Unlike calling ``ReadPacked4Bit`` per element, this walks whole
// bytes and extracts both nibbles per iteration with no per-element index
// shift/mask recomputation, so the compiler can auto-vectorize it; a single
// bulk unpack of the whole operand is far cheaper than unpacking through the
// transposed per-row/per-column access pattern used to build the dot-product
// panels below. A large A or B operand makes this unpack pass itself a
// measurable share of the total time, so full byte pairs are split across
// independent worker ranges when a real executor is installed; the common
// single-thread case bypasses ``ExecuteRanges`` entirely and calls
// ``UnpackPacked4BitPairRange`` once for the whole range, matching the
// original unparallelized loop exactly.
void UnpackPacked4BitBuffer(const std::uint8_t *packed, std::int64_t count, bool is_signed,
                            std::int8_t *raw) {
  const std::int64_t full_pairs = count / 2;
  if (CurrentExecutionExecutor() == nullptr) {
    UnpackPacked4BitPairRange(packed, is_signed, raw, 0, full_pairs);
  } else {
    ExecuteRanges(full_pairs, 2.0, [packed, is_signed, raw](std::int64_t begin, std::int64_t end) {
      UnpackPacked4BitPairRange(packed, is_signed, raw, begin, end);
    });
  }
  if (count & 1) {
    raw[count - 1] = static_cast<std::int8_t>(ReadPacked4Bit(packed, count - 1, is_signed));
  }
}

// Packs the disjoint row range ``[begin, end)`` of A into ``a_panel_data``
// and accumulates the true per-row sums into ``a_row_sum_data``. Plain
// function (not a lambda) so the single-participant fast path below calls it
// directly with the same codegen as the original unparallelized loop.
void PackU8RowRange(const std::uint8_t *a, bool a_signed, std::int64_t oa, std::int64_t depth,
                    std::uint8_t *a_panel_data, std::int64_t *a_row_sum_data, std::int64_t begin,
                    std::int64_t end) {
  for (std::int64_t i = begin; i < end; ++i) {
    std::int64_t row_sum = 0;
    const std::uint8_t *a_row = a + i * depth;
    std::uint8_t *packed = a_panel_data + i * depth;
    for (std::int64_t d = 0; d < depth; ++d) {
      const std::int64_t value =
          a_signed ? static_cast<std::int8_t>(a_row[d]) : static_cast<std::int64_t>(a_row[d]);
      row_sum += value;
      packed[d] = static_cast<std::uint8_t>(value + oa);
    }
    a_row_sum_data[i] = row_sum;
  }
}

// Packs the disjoint column range ``[begin, end)`` of B into a transposed
// contiguous panel and accumulates the true per-column sums.
void PackS8ColRange(const std::uint8_t *b, bool b_signed, std::int64_t ob, std::int64_t depth,
                    std::int64_t cols, std::int8_t *b_panel_data, std::int64_t *b_col_sum_data,
                    std::int64_t begin, std::int64_t end) {
  for (std::int64_t j = begin; j < end; ++j) {
    std::int64_t col_sum = 0;
    std::int8_t *packed = b_panel_data + j * depth;
    for (std::int64_t d = 0; d < depth; ++d) {
      const std::uint8_t raw = b[d * cols + j];
      const std::int64_t value =
          b_signed ? static_cast<std::int8_t>(raw) : static_cast<std::int64_t>(raw);
      col_sum += value;
      packed[d] = static_cast<std::int8_t>(value - ob);
    }
    b_col_sum_data[j] = col_sum;
  }
}

// Applies the zero-point correction (and, when ``has_products`` is false,
// the raw dot product itself) to the disjoint row range ``[row_begin,
// row_end)`` of ``c``.
void CorrectU8S8RowRange(IntegerVnniDotFn dot, std::int32_t *c, std::int64_t cols,
                         std::int64_t depth, std::int64_t oa, std::int64_t ob,
                         const std::int32_t *a_zero_point, std::int64_t a_zero_point_count,
                         const std::int32_t *b_zero_point, std::int64_t b_zero_point_count,
                         bool has_products, const std::uint8_t *a_panel_data,
                         const std::int8_t *b_panel_data, const std::int64_t *a_row_sum_data,
                         const std::int64_t *b_col_sum_data, std::int64_t row_begin,
                         std::int64_t row_end) {
  for (std::int64_t i = row_begin; i < row_end; ++i) {
    const std::int64_t az = a_zero_point_count == 1 ? a_zero_point[0] : a_zero_point[i];
    const std::int64_t sa = a_row_sum_data[i];
    const std::uint8_t *a_row = a_panel_data + i * depth;
    for (std::int64_t j = 0; j < cols; ++j) {
      const std::int64_t bz = b_zero_point_count == 1 ? b_zero_point[0] : b_zero_point[j];
      const std::int64_t sb_sum = b_col_sum_data[j];
      const std::int64_t vp =
          has_products ? c[i * cols + j] : dot(a_row, b_panel_data + j * depth, depth);
      const std::int64_t s =
          vp + sa * (ob - bz) + sb_sum * (-oa - az) + depth * (oa * ob + az * bz);
      c[i * cols + j] = std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(s));
    }
  }
}

} // namespace

void IntegerMatMul2DWithDot(IntegerVnniDotFn dot, const std::uint8_t *a, bool a_signed,
                            const std::uint8_t *b, bool b_signed, std::int32_t *c,
                            std::int64_t rows, std::int64_t cols, std::int64_t depth,
                            const std::int32_t *a_zero_point, std::int64_t a_zero_point_count,
                            const std::int32_t *b_zero_point, std::int64_t b_zero_point_count) {
  // Offsets that map each operand into the ``vpdpbusd`` unsigned x signed domain.
  const std::int64_t oa = a_signed ? 128 : 0;
  const std::int64_t ob = b_signed ? 0 : 128;
  const bool parallel = CurrentExecutionExecutor() != nullptr;

  // Pack A into a contiguous UINT8 panel and accumulate the true row sums.
  // The common single-thread case bypasses ``ExecuteRanges`` entirely and
  // calls ``PackU8RowRange`` once for the whole range, matching the original
  // unparallelized loop exactly; a real executor instead hands out disjoint
  // row ranges, scaling this pass across the physical-core policy the same
  // way the FP16/BF16 ``GemmHalfPlan`` already does.
  std::vector<std::uint8_t> a_panel(static_cast<std::size_t>(rows * depth));
  std::vector<std::int64_t> a_row_sum(static_cast<std::size_t>(rows), 0);
  {
    std::uint8_t *const a_panel_data = a_panel.data();
    std::int64_t *const a_row_sum_data = a_row_sum.data();
    if (!parallel) {
      PackU8RowRange(a, a_signed, oa, depth, a_panel_data, a_row_sum_data, 0, rows);
    } else {
      ExecuteRanges(rows, static_cast<double>(depth),
                    [a, a_signed, oa, depth, a_panel_data, a_row_sum_data](std::int64_t begin,
                                                                           std::int64_t end) {
                      PackU8RowRange(a, a_signed, oa, depth, a_panel_data, a_row_sum_data, begin,
                                     end);
                    });
    }
  }

  // Pack B into a transposed contiguous INT8 panel (one column per row) and
  // accumulate the true column sums. Same single-thread bypass as above.
  std::vector<std::int8_t> b_panel(static_cast<std::size_t>(cols * depth));
  std::vector<std::int64_t> b_col_sum(static_cast<std::size_t>(cols), 0);
  {
    std::int8_t *const b_panel_data = b_panel.data();
    std::int64_t *const b_col_sum_data = b_col_sum.data();
    if (!parallel) {
      PackS8ColRange(b, b_signed, ob, depth, cols, b_panel_data, b_col_sum_data, 0, cols);
    } else {
      ExecuteRanges(cols, static_cast<double>(depth),
                    [b, b_signed, ob, depth, cols, b_panel_data, b_col_sum_data](std::int64_t begin,
                                                                                 std::int64_t end) {
                      PackS8ColRange(b, b_signed, ob, depth, cols, b_panel_data, b_col_sum_data,
                                     begin, end);
                    });
    }
  }

  bool has_products = false;
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER
  if (dot == &IntegerDotU8S8Avx2 && rows >= 2 && cols >= 2 && depth >= 32) {
    IntegerMatMulU8S8Avx2(a_panel.data(), b_panel.data(), c, rows, cols, depth);
    has_products = true;
  }
#endif
  // The correction loop below is O(cols) per row when the bulk dot-product
  // above already filled ``c`` (``has_products``), or O(cols * depth) per row
  // when every (row, column) pair still calls ``dot`` here; scale the
  // per-row cost estimate accordingly so short-``K`` shapes are not starved
  // of participants while tall-``K`` shapes are not over-split. Same
  // single-thread bypass as the packing loops above.
  const double per_row_cost =
      static_cast<double>(cols) * (has_products ? 1.0 : static_cast<double>(depth));
  {
    const std::uint8_t *const a_panel_data = a_panel.data();
    const std::int8_t *const b_panel_data = b_panel.data();
    const std::int64_t *const a_row_sum_data = a_row_sum.data();
    const std::int64_t *const b_col_sum_data = b_col_sum.data();
    if (!parallel) {
      CorrectU8S8RowRange(dot, c, cols, depth, oa, ob, a_zero_point, a_zero_point_count,
                          b_zero_point, b_zero_point_count, has_products, a_panel_data,
                          b_panel_data, a_row_sum_data, b_col_sum_data, 0, rows);
    } else {
      ExecuteRanges(rows, per_row_cost,
                    [dot, c, cols, depth, oa, ob, a_zero_point, a_zero_point_count, b_zero_point,
                     b_zero_point_count, has_products, a_panel_data, b_panel_data, a_row_sum_data,
                     b_col_sum_data](std::int64_t row_begin, std::int64_t row_end) {
                      CorrectU8S8RowRange(dot, c, cols, depth, oa, ob, a_zero_point,
                                          a_zero_point_count, b_zero_point, b_zero_point_count,
                                          has_products, a_panel_data, b_panel_data, a_row_sum_data,
                                          b_col_sum_data, row_begin, row_end);
                    });
    }
  }
}

namespace {

// Unpacks and packs the disjoint row range ``[begin, end)`` of the 4-bit A
// operand into ``a_panel_data`` and its true per-row sums into
// ``a_row_sum_data``. Plain function so the single-participant fast path
// below calls it directly with the same codegen as the original loop.
void Pack4BitU8RowRange(const std::int8_t *a_raw, std::int64_t oa, std::int64_t depth,
                        std::uint8_t *a_panel_data, std::int64_t *a_row_sum_data,
                        std::int64_t begin, std::int64_t end) {
  for (std::int64_t row = begin; row < end; ++row) {
    std::int64_t row_sum = 0;
    const std::int8_t *raw_row = a_raw + row * depth;
    std::uint8_t *packed = a_panel_data + row * depth;
    for (std::int64_t inner = 0; inner < depth; ++inner) {
      row_sum += raw_row[inner];
      packed[inner] = static_cast<std::uint8_t>(raw_row[inner] + oa);
    }
    a_row_sum_data[row] = row_sum;
  }
}

// Packs the disjoint column range ``[begin, end)`` of the unpacked 4-bit B
// operand into a transposed contiguous panel and its true per-column sums.
void Pack4BitS8ColRange(const std::int8_t *b_raw, std::int64_t ob, std::int64_t depth,
                        std::int64_t cols, std::int8_t *b_panel_data, std::int64_t *b_col_sum_data,
                        std::int64_t begin, std::int64_t end) {
  for (std::int64_t column = begin; column < end; ++column) {
    std::int64_t column_sum = 0;
    std::int8_t *packed = b_panel_data + column * depth;
    for (std::int64_t inner = 0; inner < depth; ++inner) {
      const std::int8_t value = b_raw[inner * cols + column];
      column_sum += value;
      packed[inner] = static_cast<std::int8_t>(value - ob);
    }
    b_col_sum_data[column] = column_sum;
  }
}

// Applies the raw dot product and zero-point correction to the disjoint row
// range ``[row_begin, row_end)`` of ``c``. INT4 has no batched microkernel,
// so every (row, column) pair always calls ``dot`` here.
void Correct4BitRowRange(IntegerVnniDotFn dot, std::int32_t *c, std::int64_t cols,
                         std::int64_t depth, std::int64_t oa, std::int64_t ob,
                         const std::int32_t *a_zero_point, std::int64_t a_zero_point_count,
                         const std::int32_t *b_zero_point, std::int64_t b_zero_point_count,
                         const std::uint8_t *a_panel_data, const std::int8_t *b_panel_data,
                         const std::int64_t *a_row_sum_data, const std::int64_t *b_col_sum_data,
                         std::int64_t row_begin, std::int64_t row_end) {
  for (std::int64_t row = row_begin; row < row_end; ++row) {
    const std::int64_t az = a_zero_point_count == 1 ? a_zero_point[0] : a_zero_point[row];
    const std::int64_t sa = a_row_sum_data[row];
    const std::uint8_t *a_values = a_panel_data + row * depth;
    for (std::int64_t column = 0; column < cols; ++column) {
      const std::int64_t bz = b_zero_point_count == 1 ? b_zero_point[0] : b_zero_point[column];
      const std::int64_t sb = b_col_sum_data[column];
      const std::int64_t product = dot(a_values, b_panel_data + column * depth, depth);
      const std::int64_t corrected =
          product + sa * (ob - bz) + sb * (-oa - az) + depth * (oa * ob + az * bz);
      c[row * cols + column] = std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(corrected));
    }
  }
}

} // namespace

void IntegerMatMul4Bit2DWithDot(IntegerVnniDotFn dot, const std::uint8_t *a, bool a_signed,
                                const std::uint8_t *b, bool b_signed, std::int32_t *c,
                                std::int64_t rows, std::int64_t cols, std::int64_t depth,
                                const std::int32_t *a_zero_point, std::int64_t a_zero_point_count,
                                const std::int32_t *b_zero_point, std::int64_t b_zero_point_count) {
  const std::int64_t oa = a_signed ? 8 : 0;
  const std::int64_t ob = b_signed ? 0 : 8;
  const bool parallel = CurrentExecutionExecutor() != nullptr;

  // A's logical (row, inner) index is exactly ``row * depth + inner`` with no
  // padding, so the whole operand can be unpacked in a single contiguous pass
  // before the per-row panel and row-sum are derived from plain byte reads.
  std::vector<std::int8_t> a_raw(static_cast<std::size_t>(rows * depth));
  UnpackPacked4BitBuffer(a, rows * depth, a_signed, a_raw.data());
  std::vector<std::uint8_t> a_panel(static_cast<std::size_t>(rows * depth));
  std::vector<std::int64_t> a_row_sum(static_cast<std::size_t>(rows), 0);
  {
    const std::int8_t *const a_raw_data = a_raw.data();
    std::uint8_t *const a_panel_data = a_panel.data();
    std::int64_t *const a_row_sum_data = a_row_sum.data();
    if (!parallel) {
      Pack4BitU8RowRange(a_raw_data, oa, depth, a_panel_data, a_row_sum_data, 0, rows);
    } else {
      ExecuteRanges(rows, static_cast<double>(depth),
                    [a_raw_data, oa, depth, a_panel_data, a_row_sum_data](std::int64_t begin,
                                                                          std::int64_t end) {
                      Pack4BitU8RowRange(a_raw_data, oa, depth, a_panel_data, a_row_sum_data, begin,
                                         end);
                    });
    }
  }

  // B's logical (inner, column) index is ``inner * cols + column``: a single
  // bulk unpack in that same row-major order is contiguous, unlike the
  // transposed per-column panel built from it below.
  std::vector<std::int8_t> b_raw(static_cast<std::size_t>(cols * depth));
  UnpackPacked4BitBuffer(b, cols * depth, b_signed, b_raw.data());
  std::vector<std::int8_t> b_panel(static_cast<std::size_t>(cols * depth));
  std::vector<std::int64_t> b_col_sum(static_cast<std::size_t>(cols), 0);
  {
    const std::int8_t *const b_raw_data = b_raw.data();
    std::int8_t *const b_panel_data = b_panel.data();
    std::int64_t *const b_col_sum_data = b_col_sum.data();
    if (!parallel) {
      Pack4BitS8ColRange(b_raw_data, ob, depth, cols, b_panel_data, b_col_sum_data, 0, cols);
    } else {
      ExecuteRanges(cols, static_cast<double>(depth),
                    [b_raw_data, ob, depth, cols, b_panel_data, b_col_sum_data](std::int64_t begin,
                                                                                std::int64_t end) {
                      Pack4BitS8ColRange(b_raw_data, ob, depth, cols, b_panel_data, b_col_sum_data,
                                         begin, end);
                    });
    }
  }

  // Unlike ``IntegerMatMul2DWithDot``, INT4 has no batched AVX2 microkernel
  // (the unpacked panels are transient and not worth widening back into
  // whole bytes for a blocked kernel), so every (row, column) pair always
  // calls ``dot`` here; the per-row cost is O(cols * depth) regardless of
  // shape.
  {
    const std::uint8_t *const a_panel_data = a_panel.data();
    const std::int8_t *const b_panel_data = b_panel.data();
    const std::int64_t *const a_row_sum_data = a_row_sum.data();
    const std::int64_t *const b_col_sum_data = b_col_sum.data();
    if (!parallel) {
      Correct4BitRowRange(dot, c, cols, depth, oa, ob, a_zero_point, a_zero_point_count,
                          b_zero_point, b_zero_point_count, a_panel_data, b_panel_data,
                          a_row_sum_data, b_col_sum_data, 0, rows);
    } else {
      ExecuteRanges(rows, static_cast<double>(cols) * static_cast<double>(depth),
                    [dot, c, cols, depth, oa, ob, a_zero_point, a_zero_point_count, b_zero_point,
                     b_zero_point_count, a_panel_data, b_panel_data, a_row_sum_data,
                     b_col_sum_data](std::int64_t row_begin, std::int64_t row_end) {
                      Correct4BitRowRange(dot, c, cols, depth, oa, ob, a_zero_point,
                                          a_zero_point_count, b_zero_point, b_zero_point_count,
                                          a_panel_data, b_panel_data, a_row_sum_data,
                                          b_col_sum_data, row_begin, row_end);
                    });
    }
  }
}

} // namespace detail

bool IntegerMatMul2DUsesVnni() {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512VNNI
  return CpuSupportsAvx512Vnni();
#else
  return false;
#endif
}

void IntegerMatMul2D(const std::uint8_t *a, bool a_signed, const std::uint8_t *b, bool b_signed,
                     std::int32_t *c, std::int64_t rows, std::int64_t cols, std::int64_t depth,
                     const std::int32_t *a_zero_point, std::int64_t a_zero_point_count,
                     const std::int32_t *b_zero_point, std::int64_t b_zero_point_count) {
#if defined(ONNX_LIGHT_CPU_HAVE_AVX512VNNI) && defined(ONNX_LIGHT_CPU_HAVE_AVX512BW)
  if (rows == 1 && CpuSupportsAvx512Vnni() && CpuSupportsAvx512BW()) {
    detail::IntegerMatMulSkinnyMAvx512(a, a_signed, b, b_signed, c, cols, depth, a_zero_point[0],
                                       b_zero_point, b_zero_point_count);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER
  if (rows == 1 && DetectSimdLevel() >= SimdLevel::kAVX2) {
    detail::IntegerMatMulSkinnyMAvx2(a, a_signed, b, b_signed, c, cols, depth, a_zero_point[0],
                                     b_zero_point, b_zero_point_count);
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AMX_INT8
  if (AmxTileStateAvailable() && CpuSupportsAmxInt8()) {
    GemmMatMulIntegerAmxInt8(a, a_signed, b, b_signed, c, static_cast<std::size_t>(rows),
                             static_cast<std::size_t>(cols), static_cast<std::size_t>(depth),
                             a_zero_point, static_cast<std::size_t>(a_zero_point_count),
                             b_zero_point, static_cast<std::size_t>(b_zero_point_count));
    return;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD
  // Roadmap PR09.3: prefer the native ARM NEON dot-product kernel when the CPU
  // supports it. It reproduces the same modulo-2^32 accumulation as the scalar
  // and VNNI paths below.
  if (CpuSupportsNeonDotProd()) {
    GemmMatMulIntegerNeonDotProd(a, a_signed, b, b_signed, c, static_cast<std::size_t>(rows),
                                 static_cast<std::size_t>(cols), static_cast<std::size_t>(depth),
                                 a_zero_point, static_cast<std::size_t>(a_zero_point_count),
                                 b_zero_point, static_cast<std::size_t>(b_zero_point_count));
    return;
  }
#endif
  detail::IntegerVnniDotFn dot = &detail::IntegerDotU8S8Scalar;
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER
  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
    // ``ShortTail`` only pays off (and is only selected) when ``depth`` is
    // not an exact multiple of 32, so every square/transformer/large-K/
    // skinny shape in the compact GEMM baseline -- all exact multiples of
    // 32 -- keeps calling the plain ``IntegerDotU8S8Avx2`` and remains
    // eligible for the batched ``IntegerMatMulU8S8Avx2`` kernel below.
    dot = (depth % 32 != 0) ? &detail::IntegerDotU8S8Avx2ShortTail : &detail::IntegerDotU8S8Avx2;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512VNNI
  if (CpuSupportsAvx512Vnni()) {
    dot = &detail::IntegerDotU8S8Avx512Vnni;
  }
#endif
  detail::IntegerMatMul2DWithDot(dot, a, a_signed, b, b_signed, c, rows, cols, depth, a_zero_point,
                                 a_zero_point_count, b_zero_point, b_zero_point_count);
}

void IntegerMatMul4Bit2D(const std::uint8_t *a, bool a_signed, const std::uint8_t *b, bool b_signed,
                         std::int32_t *c, std::int64_t rows, std::int64_t cols, std::int64_t depth,
                         const std::int32_t *a_zero_point, std::int64_t a_zero_point_count,
                         const std::int32_t *b_zero_point, std::int64_t b_zero_point_count) {
  detail::IntegerVnniDotFn dot = &detail::IntegerDotU8S8Scalar;
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER
  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
    // INT4 has no batched microkernel, so every (row, column) pair always
    // calls ``dot`` directly (see ``IntegerMatMul4Bit2DWithDot``); picking
    // ``ShortTail`` only for non-multiple-of-32 depths keeps the tested
    // multiple-of-32 shapes on the plain function with zero added overhead.
    dot = (depth % 32 != 0) ? &detail::IntegerDotU8S8Avx2ShortTail : &detail::IntegerDotU8S8Avx2;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD
  if (CpuSupportsNeonDotProd()) {
    dot = &detail::IntegerDot4BitU8S8NeonDotProd;
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512VNNI
  if (CpuSupportsAvx512Vnni()) {
    dot = &detail::IntegerDotU8S8Avx512Vnni;
  }
#endif
  detail::IntegerMatMul4Bit2DWithDot(dot, a, a_signed, b, b_signed, c, rows, cols, depth,
                                     a_zero_point, a_zero_point_count, b_zero_point,
                                     b_zero_point_count);
}

} // namespace onnx_light_cpu
