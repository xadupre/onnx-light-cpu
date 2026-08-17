// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Native AArch64 NEON dot-product INT8 GEMM kernel (Roadmap PR09.3). It backs
// the contiguous 2D ``MatMulInteger`` fast path with the ``UDOT`` instruction
// (ARMv8.2-A ``+dotprod``), which reduces four unsigned byte products into each
// of four INT32 lanes per issue.
//
// Every signedness combination is served by a single unsigned dot product. A
// signed operand ``v`` in ``[-128, 127]`` maps to the unsigned byte ``v + 128``
// by flipping the sign bit, and the ``+128`` bias is absorbed into that
// operand's effective zero point (``a_signed`` adds 128 to ``a_zero_point``).
// After that transform every value is an unsigned byte ``u`` with an unsigned
// zero point ``zu`` and
//   ``C[m,n] = sum_k (ua[m,k] - azu[m]) * (ub[n,k] - bzu[n])``
//           ``= Puu[m,n] - azu[m]*BSum[n] - bzu[n]*ASum[m] + K*azu[m]*bzu[n]``
// where ``Puu = sum_k ua*ub`` is the raw unsigned dot product and ``ASum`` /
// ``BSum`` are the per-row / per-column unsigned byte sums. The identity holds
// over the integers, so evaluating every term modulo 2^32 (unsigned wraparound)
// reproduces the portable scalar accumulator bit for bit.
//
// ``A`` is already contiguous along ``K``; ``B`` is packed once into a
// column-major ``N x K`` buffer so both dot-product operands are contiguous.
// The sign flip is folded into the pack, and the row / column sums are computed
// while packing.

#include "onnx_light_cpu/impl/math/gemm/arm/gemm_kernel_arm.h"

#if defined(_MSC_VER)
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif

#include <bit>
#include <cstdint>
#include <vector>

namespace onnx_light_cpu {

namespace {

// Unsigned byte dot product ``sum_k a[k]*b[k]`` accumulated modulo 2^32. The
// 16-byte ``UDOT`` body reduces four byte products into each INT32 lane; the
// ``K`` remainder that does not fill a vector is summed by an exact scalar tail.
inline std::uint32_t DotProductU8(const std::uint8_t *a, const std::uint8_t *b, std::size_t k) {
  uint32x4_t acc = vdupq_n_u32(0);
  std::size_t p = 0;
  for (; p + 16 <= k; p += 16) {
    acc = vdotq_u32(acc, vld1q_u8(a + p), vld1q_u8(b + p));
  }
  std::uint32_t sum = vaddvq_u32(acc);
  for (; p < k; ++p) {
    sum += static_cast<std::uint32_t>(a[p]) * static_cast<std::uint32_t>(b[p]);
  }
  return sum;
}

} // namespace

void GemmMatMulIntegerNeonDotProd(const std::uint8_t *a, bool a_signed, const std::uint8_t *b,
                                  bool b_signed, std::int32_t *c, std::size_t m, std::size_t n,
                                  std::size_t k, const std::int32_t *a_zero_points,
                                  std::size_t a_zero_point_count, const std::int32_t *b_zero_points,
                                  std::size_t b_zero_point_count) {
  const std::uint8_t a_flip = a_signed ? 0x80 : 0x00;
  const std::uint8_t b_flip = b_signed ? 0x80 : 0x00;
  const std::int32_t a_bias = a_signed ? 128 : 0;
  const std::int32_t b_bias = b_signed ? 128 : 0;

  // Pack ``A`` into the unsigned domain (contiguous along ``K``) and accumulate
  // each row's byte sum.
  std::vector<std::uint8_t> a_packed(m * k);
  std::vector<std::uint32_t> a_sum(m, 0);
  for (std::size_t row = 0; row < m; ++row) {
    const std::uint8_t *src = a + row * k;
    std::uint8_t *dst = a_packed.data() + row * k;
    std::uint32_t sum = 0;
    for (std::size_t depth = 0; depth < k; ++depth) {
      const std::uint8_t value = static_cast<std::uint8_t>(src[depth] ^ a_flip);
      dst[depth] = value;
      sum += value;
    }
    a_sum[row] = sum;
  }

  // Pack ``B`` transposed into a column-major ``N x K`` unsigned buffer so each
  // column is contiguous along ``K``, accumulating each column's byte sum.
  std::vector<std::uint8_t> b_packed(n * k);
  std::vector<std::uint32_t> b_sum(n, 0);
  for (std::size_t col = 0; col < n; ++col) {
    std::uint8_t *dst = b_packed.data() + col * k;
    std::uint32_t sum = 0;
    for (std::size_t depth = 0; depth < k; ++depth) {
      const std::uint8_t value = static_cast<std::uint8_t>(b[depth * n + col] ^ b_flip);
      dst[depth] = value;
      sum += value;
    }
    b_sum[col] = sum;
  }

  const std::uint32_t k_wrapped = static_cast<std::uint32_t>(k);
  for (std::size_t row = 0; row < m; ++row) {
    const std::uint32_t azu = static_cast<std::uint32_t>(
        (a_zero_point_count == 1 ? a_zero_points[0] : a_zero_points[row]) + a_bias);
    const std::uint8_t *a_row = a_packed.data() + row * k;
    std::int32_t *c_row = c + row * n;
    for (std::size_t col = 0; col < n; ++col) {
      const std::uint32_t bzu = static_cast<std::uint32_t>(
          (b_zero_point_count == 1 ? b_zero_points[0] : b_zero_points[col]) + b_bias);
      const std::uint32_t puu = DotProductU8(a_row, b_packed.data() + col * k, k);
      std::uint32_t value = puu;
      value -= azu * b_sum[col];
      value -= bzu * a_sum[row];
      value += k_wrapped * azu * bzu;
      c_row[col] = std::bit_cast<std::int32_t>(value);
    }
  }
}

} // namespace onnx_light_cpu
