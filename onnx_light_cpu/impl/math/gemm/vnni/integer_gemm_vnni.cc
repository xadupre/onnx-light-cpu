// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Portable driver and scalar sibling for the x86 VNNI INT8 matrix
// multiplication (Roadmap PR09.2). This translation unit is compiled at the
// project's baseline SIMD flags; the native ``vpdpbusd`` dot-product lives in
// integer_gemm_avx512vnni.cc (compiled with -mavx512vnni) and is only linked in
// and dispatched to when the CPU reports AVX-512 VNNI at runtime.
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

#include "onnx_light_cpu/impl/simd_level.h"

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

void IntegerMatMul2DWithDot(IntegerVnniDotFn dot, const std::uint8_t *a, bool a_signed,
                            const std::uint8_t *b, bool b_signed, std::int32_t *c,
                            std::int64_t rows, std::int64_t cols, std::int64_t depth,
                            const std::int32_t *a_zero_point, std::int64_t a_zero_point_count,
                            const std::int32_t *b_zero_point, std::int64_t b_zero_point_count) {
  // Offsets that map each operand into the ``vpdpbusd`` unsigned x signed domain.
  const std::int64_t oa = a_signed ? 128 : 0;
  const std::int64_t ob = b_signed ? 0 : 128;

  // Pack A into a contiguous UINT8 panel and accumulate the true row sums.
  std::vector<std::uint8_t> a_panel(static_cast<std::size_t>(rows * depth));
  std::vector<std::int64_t> a_row_sum(static_cast<std::size_t>(rows), 0);
  for (std::int64_t i = 0; i < rows; ++i) {
    std::int64_t row_sum = 0;
    const std::uint8_t *a_row = a + i * depth;
    std::uint8_t *packed = a_panel.data() + i * depth;
    for (std::int64_t d = 0; d < depth; ++d) {
      const std::int64_t value =
          a_signed ? static_cast<std::int8_t>(a_row[d]) : static_cast<std::int64_t>(a_row[d]);
      row_sum += value;
      packed[d] = static_cast<std::uint8_t>(value + oa);
    }
    a_row_sum[static_cast<std::size_t>(i)] = row_sum;
  }

  // Pack B into a transposed contiguous INT8 panel (one column per row) and
  // accumulate the true column sums.
  std::vector<std::int8_t> b_panel(static_cast<std::size_t>(cols * depth));
  std::vector<std::int64_t> b_col_sum(static_cast<std::size_t>(cols), 0);
  for (std::int64_t j = 0; j < cols; ++j) {
    std::int64_t col_sum = 0;
    std::int8_t *packed = b_panel.data() + j * depth;
    for (std::int64_t d = 0; d < depth; ++d) {
      const std::uint8_t raw = b[d * cols + j];
      const std::int64_t value =
          b_signed ? static_cast<std::int8_t>(raw) : static_cast<std::int64_t>(raw);
      col_sum += value;
      packed[d] = static_cast<std::int8_t>(value - ob);
    }
    b_col_sum[static_cast<std::size_t>(j)] = col_sum;
  }

  for (std::int64_t i = 0; i < rows; ++i) {
    const std::int64_t az =
        a_zero_point_count == 1 ? a_zero_point[0] : a_zero_point[static_cast<std::size_t>(i)];
    const std::int64_t sa = a_row_sum[static_cast<std::size_t>(i)];
    const std::uint8_t *a_row = a_panel.data() + i * depth;
    for (std::int64_t j = 0; j < cols; ++j) {
      const std::int64_t bz =
          b_zero_point_count == 1 ? b_zero_point[0] : b_zero_point[static_cast<std::size_t>(j)];
      const std::int64_t sb_sum = b_col_sum[static_cast<std::size_t>(j)];
      const std::int64_t vp = dot(a_row, b_panel.data() + j * depth, depth);
      const std::int64_t s =
          vp + sa * (ob - bz) + sb_sum * (-oa - az) + depth * (oa * ob + az * bz);
      c[i * cols + j] = std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(s));
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
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512VNNI
  if (CpuSupportsAvx512Vnni()) {
    dot = &detail::IntegerDotU8S8Avx512Vnni;
  }
#endif
  detail::IntegerMatMul2DWithDot(dot, a, a_signed, b, b_signed, c, rows, cols, depth, a_zero_point,
                                 a_zero_point_count, b_zero_point, b_zero_point_count);
}

} // namespace onnx_light_cpu
