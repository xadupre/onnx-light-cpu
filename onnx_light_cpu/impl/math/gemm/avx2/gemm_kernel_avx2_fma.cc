// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/avx2/gemm_kernel_avx2_fma.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <algorithm>
#include <immintrin.h>

namespace onnx_light_cpu {

namespace {

constexpr int kGemmPrefetchDistanceK = 4;

template <typename T> inline void PrefetchT0(const T *ptr) {
  _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0);
}

template <bool Bfloat16> inline float ReadHalf(const std::uint16_t *value) {
  if constexpr (Bfloat16) {
    return detail::Bfloat16BitsToFloat(*value);
  } else {
#ifdef ONNX_LIGHT_CPU_HAVE_F16C
    const __m128i half = _mm_cvtsi32_si128(static_cast<int>(*value));
    return _mm_cvtss_f32(_mm_cvtph_ps(half));
#else
    return detail::Float16BitsToFloat(*value);
#endif
  }
}

template <bool Bfloat16> inline __m256 WidenHalf8(const std::uint16_t *values) {
  const __m128i halves = _mm_loadu_si128(reinterpret_cast<const __m128i *>(values));
  if constexpr (Bfloat16) {
    const __m256i widened = _mm256_slli_epi32(_mm256_cvtepu16_epi32(halves), 16);
    return _mm256_castsi256_ps(widened);
  } else {
    return _mm256_cvtph_ps(halves);
  }
}

inline void Transpose8x8U16(const std::uint16_t *src, std::size_t src_stride, std::uint16_t *dst) {
  const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
  const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + src_stride));
  const __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 2 * src_stride));
  const __m128i r3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 3 * src_stride));
  const __m128i r4 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 4 * src_stride));
  const __m128i r5 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 5 * src_stride));
  const __m128i r6 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 6 * src_stride));
  const __m128i r7 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 7 * src_stride));
  const __m128i a0 = _mm_unpacklo_epi16(r0, r1);
  const __m128i a1 = _mm_unpackhi_epi16(r0, r1);
  const __m128i a2 = _mm_unpacklo_epi16(r2, r3);
  const __m128i a3 = _mm_unpackhi_epi16(r2, r3);
  const __m128i a4 = _mm_unpacklo_epi16(r4, r5);
  const __m128i a5 = _mm_unpackhi_epi16(r4, r5);
  const __m128i a6 = _mm_unpacklo_epi16(r6, r7);
  const __m128i a7 = _mm_unpackhi_epi16(r6, r7);
  const __m128i b0 = _mm_unpacklo_epi32(a0, a2);
  const __m128i b1 = _mm_unpackhi_epi32(a0, a2);
  const __m128i b2 = _mm_unpacklo_epi32(a1, a3);
  const __m128i b3 = _mm_unpackhi_epi32(a1, a3);
  const __m128i b4 = _mm_unpacklo_epi32(a4, a6);
  const __m128i b5 = _mm_unpackhi_epi32(a4, a6);
  const __m128i b6 = _mm_unpacklo_epi32(a5, a7);
  const __m128i b7 = _mm_unpackhi_epi32(a5, a7);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), _mm_unpacklo_epi64(b0, b4));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 8), _mm_unpackhi_epi64(b0, b4));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 16), _mm_unpacklo_epi64(b1, b5));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 24), _mm_unpackhi_epi64(b1, b5));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 32), _mm_unpacklo_epi64(b2, b6));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 40), _mm_unpackhi_epi64(b2, b6));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 48), _mm_unpacklo_epi64(b3, b7));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 56), _mm_unpackhi_epi64(b3, b7));
}

inline void Transpose4x4Float(const float *src, std::size_t src_stride, float *dst,
                              std::size_t dst_stride) {
  __m128 row0 = _mm_loadu_ps(src);
  __m128 row1 = _mm_loadu_ps(src + src_stride);
  __m128 row2 = _mm_loadu_ps(src + 2 * src_stride);
  __m128 row3 = _mm_loadu_ps(src + 3 * src_stride);
  _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
  _mm_storeu_ps(dst, row0);
  _mm_storeu_ps(dst + dst_stride, row1);
  _mm_storeu_ps(dst + 2 * dst_stride, row2);
  _mm_storeu_ps(dst + 3 * dst_stride, row3);
}

inline void Transpose4x4Double(const double *src, std::size_t src_stride, double *dst,
                               std::size_t dst_stride) {
  const __m256d row0 = _mm256_loadu_pd(src);
  const __m256d row1 = _mm256_loadu_pd(src + src_stride);
  const __m256d row2 = _mm256_loadu_pd(src + 2 * src_stride);
  const __m256d row3 = _mm256_loadu_pd(src + 3 * src_stride);
  const __m256d low01 = _mm256_unpacklo_pd(row0, row1);
  const __m256d high01 = _mm256_unpackhi_pd(row0, row1);
  const __m256d low23 = _mm256_unpacklo_pd(row2, row3);
  const __m256d high23 = _mm256_unpackhi_pd(row2, row3);
  _mm256_storeu_pd(dst, _mm256_permute2f128_pd(low01, low23, 0x20));
  _mm256_storeu_pd(dst + dst_stride, _mm256_permute2f128_pd(high01, high23, 0x20));
  _mm256_storeu_pd(dst + 2 * dst_stride, _mm256_permute2f128_pd(low01, low23, 0x31));
  _mm256_storeu_pd(dst + 3 * dst_stride, _mm256_permute2f128_pd(high01, high23, 0x31));
}

template <bool Bfloat16>
void PackTransposeHalfToFloat32(const std::uint16_t *src, std::size_t src_stride, float *dst,
                                std::size_t dst_rows, std::size_t dst_cols) {
  alignas(16) std::uint16_t tile[64];
  std::size_t row = 0;
  for (; row + 8 <= dst_rows; row += 8) {
    std::size_t column = 0;
    for (; column + 8 <= dst_cols; column += 8) {
      Transpose8x8U16(src + column * src_stride + row, src_stride, tile);
      for (std::size_t tile_row = 0; tile_row < 8; ++tile_row) {
        _mm256_storeu_ps(dst + (row + tile_row) * dst_cols + column,
                         WidenHalf8<Bfloat16>(tile + tile_row * 8));
      }
    }
    for (; column < dst_cols; ++column) {
      for (std::size_t tile_row = 0; tile_row < 8; ++tile_row) {
        dst[(row + tile_row) * dst_cols + column] =
            ReadHalf<Bfloat16>(src + column * src_stride + row + tile_row);
      }
    }
  }
  for (; row < dst_rows; ++row) {
    for (std::size_t column = 0; column < dst_cols; ++column) {
      dst[row * dst_cols + column] = ReadHalf<Bfloat16>(src + column * src_stride + row);
    }
  }
}

inline float HorizontalSum(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

template <bool Bfloat16>
void GemmHalfSkinnyM(bool trans_a, std::size_t M, std::size_t N, std::size_t K, float alpha,
                     const std::uint16_t *A, const std::uint16_t *B, float *Y) {
  constexpr std::size_t kColumns = 16;
  const std::size_t panels = (N + kColumns - 1) / kColumns;
  const double cost = static_cast<double>(M) * kColumns * K / 32768.0;
  ExecuteRanges(static_cast<std::int64_t>(panels), cost, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t panel = begin; panel < end; ++panel) {
      const std::size_t n0 = static_cast<std::size_t>(panel) * kColumns;
      const std::size_t columns = std::min(kColumns, N - n0);
      if (columns == kColumns) {
        __m256 accumulators0[kGemmAVX2MR];
        __m256 accumulators1[kGemmAVX2MR];
        for (std::size_t row = 0; row < M; ++row) {
          accumulators0[row] = _mm256_setzero_ps();
          accumulators1[row] = _mm256_setzero_ps();
        }
        for (std::size_t depth = 0; depth < K; ++depth) {
          const __m256 vb0 = WidenHalf8<Bfloat16>(B + depth * N + n0);
          const __m256 vb1 = WidenHalf8<Bfloat16>(B + depth * N + n0 + 8);
          for (std::size_t row = 0; row < M; ++row) {
            const float a = ReadHalf<Bfloat16>(trans_a ? A + depth * M + row : A + row * K + depth);
            const __m256 va = _mm256_set1_ps(a);
            accumulators0[row] = _mm256_fmadd_ps(va, vb0, accumulators0[row]);
            accumulators1[row] = _mm256_fmadd_ps(va, vb1, accumulators1[row]);
          }
        }
        const __m256 valpha = _mm256_set1_ps(alpha);
        for (std::size_t row = 0; row < M; ++row) {
          _mm256_storeu_ps(Y + row * N + n0, _mm256_mul_ps(valpha, accumulators0[row]));
          _mm256_storeu_ps(Y + row * N + n0 + 8, _mm256_mul_ps(valpha, accumulators1[row]));
        }
      } else if constexpr (!Bfloat16) {
        if (columns == 2 && !trans_a) {
          const __m256i duplicate_pairs = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
          for (std::size_t row = 0; row < M; ++row) {
            __m256 accumulator = _mm256_setzero_ps();
            const std::uint16_t *arow = A + row * K;
            std::size_t depth = 0;
            for (; depth + 4 <= K; depth += 4) {
              const __m128i ah = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(arow + depth));
              const __m128 a4 = _mm_cvtph_ps(ah);
              const __m256 a =
                  _mm256_permutevar8x32_ps(_mm256_castps128_ps256(a4), duplicate_pairs);
              const __m256 b = WidenHalf8<false>(B + depth * N + n0);
              accumulator = _mm256_fmadd_ps(a, b, accumulator);
            }
            alignas(32) float lanes[8];
            _mm256_store_ps(lanes, accumulator);
            float sum0 = lanes[0] + lanes[2] + lanes[4] + lanes[6];
            float sum1 = lanes[1] + lanes[3] + lanes[5] + lanes[7];
            for (; depth < K; ++depth) {
              const float a = ReadHalf<false>(arow + depth);
              sum0 += a * ReadHalf<false>(B + depth * N + n0);
              sum1 += a * ReadHalf<false>(B + depth * N + n0 + 1);
            }
            Y[row * N + n0] = alpha * sum0;
            Y[row * N + n0 + 1] = alpha * sum1;
          }
          continue;
        }
        float accumulators[kGemmAVX2MR][kColumns] = {};
        for (std::size_t depth = 0; depth < K; ++depth) {
          float bvalues[kColumns];
          for (std::size_t column = 0; column < columns; ++column) {
            bvalues[column] = ReadHalf<Bfloat16>(B + depth * N + n0 + column);
          }
          for (std::size_t row = 0; row < M; ++row) {
            const float a = ReadHalf<Bfloat16>(trans_a ? A + depth * M + row : A + row * K + depth);
            for (std::size_t column = 0; column < columns; ++column) {
              accumulators[row][column] += a * bvalues[column];
            }
          }
        }
        for (std::size_t row = 0; row < M; ++row) {
          for (std::size_t column = 0; column < columns; ++column) {
            Y[row * N + n0 + column] = alpha * accumulators[row][column];
          }
        }
      } else {
        float accumulators[kGemmAVX2MR][kColumns] = {};
        for (std::size_t depth = 0; depth < K; ++depth) {
          float bvalues[kColumns];
          for (std::size_t column = 0; column < columns; ++column) {
            bvalues[column] = ReadHalf<Bfloat16>(B + depth * N + n0 + column);
          }
          for (std::size_t row = 0; row < M; ++row) {
            const float a = ReadHalf<Bfloat16>(trans_a ? A + depth * M + row : A + row * K + depth);
            for (std::size_t column = 0; column < columns; ++column) {
              accumulators[row][column] += a * bvalues[column];
            }
          }
        }
        for (std::size_t row = 0; row < M; ++row) {
          for (std::size_t column = 0; column < columns; ++column) {
            Y[row * N + n0 + column] = alpha * accumulators[row][column];
          }
        }
      }
    }
  });
}

template <bool Bfloat16>
void GemmHalfSkinnyN(std::size_t M, std::size_t K, float alpha, const std::uint16_t *A,
                     const std::uint16_t *B, float *Y) {
  const double cost = static_cast<double>(K) / 32768.0;
  ExecuteRanges(static_cast<std::int64_t>(M), cost, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t row = begin; row < end; ++row) {
      const std::uint16_t *a = A + static_cast<std::size_t>(row) * K;
      __m256 accumulator = _mm256_setzero_ps();
      std::size_t depth = 0;
      for (; depth + 8 <= K; depth += 8) {
        accumulator = _mm256_fmadd_ps(WidenHalf8<Bfloat16>(a + depth),
                                      WidenHalf8<Bfloat16>(B + depth), accumulator);
      }
      float sum = HorizontalSum(accumulator);
      for (; depth < K; ++depth) {
        sum += ReadHalf<Bfloat16>(a + depth) * ReadHalf<Bfloat16>(B + depth);
      }
      Y[row] = alpha * sum;
    }
  });
}

template <bool Bfloat16, std::size_t MR>
void GemmMicroKernel_AVX2HalfImpl(std::size_t nb, std::size_t K, float alpha, float beta,
                                  const std::uint16_t *Bmat, std::size_t N, const float *Crow_base,
                                  std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                                  std::size_t n0, GemmAccumMode mode, const std::uint16_t *Apack) {
  static_assert(MR >= 1 && MR <= kGemmAVX2MR);
  const __m256 valpha = _mm256_set1_ps(alpha);
  const __m256 vbeta = _mm256_set1_ps(beta);
  const bool alpha_is_one = alpha == 1.0f;
  const bool beta_is_one = beta == 1.0f;
  std::size_t n = 0;
  for (; n + 16 <= nb; n += 16) {
    __m256 acc0[MR];
    __m256 acc1[MR];
    for (std::size_t row = 0; row < MR; ++row) {
      acc0[row] = _mm256_setzero_ps();
      acc1[row] = _mm256_setzero_ps();
    }
    for (std::size_t depth = 0; depth < K; ++depth) {
      const std::uint16_t *b = Bmat + depth * N + n0 + n;
      const __m256 vb0 = WidenHalf8<Bfloat16>(b);
      const __m256 vb1 = WidenHalf8<Bfloat16>(b + 8);
      if (depth + kGemmPrefetchDistanceK < K) {
        PrefetchT0(b + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t row = 0; row < MR; ++row) {
        const __m256 va = _mm256_set1_ps(ReadHalf<Bfloat16>(Apack + row * K + depth));
        acc0[row] = _mm256_fmadd_ps(va, vb0, acc0[row]);
        acc1[row] = _mm256_fmadd_ps(va, vb1, acc1[row]);
      }
    }
    for (std::size_t row = 0; row < MR; ++row) {
      float *y = Yrow_base + row * Ystride + n0 + n;
      __m256 result0 = alpha_is_one ? acc0[row] : _mm256_mul_ps(valpha, acc0[row]);
      __m256 result1 = alpha_is_one ? acc1[row] : _mm256_mul_ps(valpha, acc1[row]);
      if (mode == GemmAccumMode::kInitBias) {
        const float *c = Crow_base + row * Cstride + n0 + n;
        const __m256 c0 = _mm256_loadu_ps(c);
        const __m256 c1 = _mm256_loadu_ps(c + 8);
        result0 = _mm256_add_ps(result0, beta_is_one ? c0 : _mm256_mul_ps(vbeta, c0));
        result1 = _mm256_add_ps(result1, beta_is_one ? c1 : _mm256_mul_ps(vbeta, c1));
      } else if (mode == GemmAccumMode::kAccumulate) {
        result0 = _mm256_add_ps(result0, _mm256_loadu_ps(y));
        result1 = _mm256_add_ps(result1, _mm256_loadu_ps(y + 8));
      }
      _mm256_storeu_ps(y, result0);
      _mm256_storeu_ps(y + 8, result1);
    }
  }
  if (n < nb) {
    if constexpr (Bfloat16) {
      GemmMicroKernel_ScalarBf16(MR, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                                 Ystride, n0 + n, mode, Apack);
    } else {
      GemmMicroKernel_ScalarFp16(MR, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                                 Ystride, n0 + n, mode, Apack);
    }
  }
}

} // namespace

template <std::size_t MR>
void GemmMicroKernel_AVX2FMA_F32Impl(std::size_t nb, std::size_t K, float alpha, float beta,
                                     const float *Bmat, std::size_t N, const float *Crow_base,
                                     std::size_t Cstride, float *Yrow_base, std::size_t Ystride,
                                     std::size_t n0, GemmAccumMode mode, const float *Apack) {
  static_assert(MR >= 1 && MR <= kGemmAVX2MR);
  const __m256 valpha = _mm256_set1_ps(alpha);
  const __m256 vbeta = _mm256_set1_ps(beta);
  const bool alpha_is_one = alpha == 1.0f;
  const bool beta_is_one = beta == 1.0f;
  std::size_t n = 0;
  for (; n + 16 <= nb; n += 16) {
    __m256 acc0[MR];
    __m256 acc1[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc0[r] = _mm256_setzero_ps();
      acc1[r] = _mm256_setzero_ps();
    }
    const auto accumulate_k = [&](std::size_t k) {
      const float *Brow = Bmat + k * N + n0 + n;
      const __m256 vb0 = _mm256_loadu_ps(Brow);
      const __m256 vb1 = _mm256_loadu_ps(Brow + 8);
      if (k + kGemmPrefetchDistanceK < K) {
        PrefetchT0(Brow + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < MR; ++r) {
        const __m256 va = _mm256_set1_ps(Apack[r * K + k]);
        acc0[r] = _mm256_fmadd_ps(va, vb0, acc0[r]);
        acc1[r] = _mm256_fmadd_ps(va, vb1, acc1[r]);
      }
    };
    std::size_t k = 0;
    for (; k + 4 <= K; k += 4) {
      accumulate_k(k);
      accumulate_k(k + 1);
      accumulate_k(k + 2);
      accumulate_k(k + 3);
    }
    for (; k < K; ++k) {
      accumulate_k(k);
    }
    for (std::size_t r = 0; r < MR; ++r) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m256 res0 = alpha_is_one ? acc0[r] : _mm256_mul_ps(valpha, acc0[r]);
      __m256 res1 = alpha_is_one ? acc1[r] : _mm256_mul_ps(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const float *Crow = Crow_base + r * Cstride + n0 + n;
        const __m256 vc0 = _mm256_loadu_ps(Crow);
        const __m256 vc1 = _mm256_loadu_ps(Crow + 8);
        res0 = _mm256_add_ps(res0, beta_is_one ? vc0 : _mm256_mul_ps(vbeta, vc0));
        res1 = _mm256_add_ps(res1, beta_is_one ? vc1 : _mm256_mul_ps(vbeta, vc1));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res0 = _mm256_add_ps(res0, _mm256_loadu_ps(Yrow));
        res1 = _mm256_add_ps(res1, _mm256_loadu_ps(Yrow + 8));
      }
      _mm256_storeu_ps(Yrow, res0);
      _mm256_storeu_ps(Yrow + 8, res1);
    }
  }
  for (; n + 8 <= nb; n += 8) {
    __m256 acc[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc[r] = _mm256_setzero_ps();
    }
    const auto accumulate_k = [&](std::size_t k) {
      const __m256 vb = _mm256_loadu_ps(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < MR; ++r) {
        const __m256 va = _mm256_set1_ps(Apack[r * K + k]);
        acc[r] = _mm256_fmadd_ps(va, vb, acc[r]);
      }
    };
    std::size_t k = 0;
    for (; k + 4 <= K; k += 4) {
      accumulate_k(k);
      accumulate_k(k + 1);
      accumulate_k(k + 2);
      accumulate_k(k + 3);
    }
    for (; k < K; ++k) {
      accumulate_k(k);
    }
    for (std::size_t r = 0; r < MR; ++r) {
      float *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m256 res = alpha_is_one ? acc[r] : _mm256_mul_ps(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const __m256 vc = _mm256_loadu_ps(Crow_base + r * Cstride + n0 + n);
        res = _mm256_add_ps(res, beta_is_one ? vc : _mm256_mul_ps(vbeta, vc));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res = _mm256_add_ps(res, _mm256_loadu_ps(Yrow));
      }
      _mm256_storeu_ps(Yrow, res);
    }
  }
  if (n < nb) {
    GemmMicroKernel_Scalar_F32(MR, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0 + n, mode, Apack);
  }
}

template <std::size_t MR>
void GemmMicroKernel_AVX2FMA_F64Impl(std::size_t nb, std::size_t K, double alpha, double beta,
                                     const double *Bmat, std::size_t N, const double *Crow_base,
                                     std::size_t Cstride, double *Yrow_base, std::size_t Ystride,
                                     std::size_t n0, GemmAccumMode mode, const double *Apack) {
  static_assert(MR >= 1 && MR <= kGemmAVX2MR);
  const __m256d valpha = _mm256_set1_pd(alpha);
  const __m256d vbeta = _mm256_set1_pd(beta);
  const bool alpha_is_one = alpha == 1.0;
  const bool beta_is_one = beta == 1.0;
  std::size_t n = 0;
  for (; n + 8 <= nb; n += 8) {
    __m256d acc0[MR];
    __m256d acc1[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc0[r] = _mm256_setzero_pd();
      acc1[r] = _mm256_setzero_pd();
    }
    const auto accumulate_k = [&](std::size_t k) {
      const double *Brow = Bmat + k * N + n0 + n;
      const __m256d vb0 = _mm256_loadu_pd(Brow);
      const __m256d vb1 = _mm256_loadu_pd(Brow + 4);
      if (k + kGemmPrefetchDistanceK < K) {
        PrefetchT0(Brow + kGemmPrefetchDistanceK * N);
      }
      for (std::size_t r = 0; r < MR; ++r) {
        const __m256d va = _mm256_set1_pd(Apack[r * K + k]);
        acc0[r] = _mm256_fmadd_pd(va, vb0, acc0[r]);
        acc1[r] = _mm256_fmadd_pd(va, vb1, acc1[r]);
      }
    };
    std::size_t k = 0;
    for (; k + 4 <= K; k += 4) {
      accumulate_k(k);
      accumulate_k(k + 1);
      accumulate_k(k + 2);
      accumulate_k(k + 3);
    }
    for (; k < K; ++k) {
      accumulate_k(k);
    }
    for (std::size_t r = 0; r < MR; ++r) {
      double *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m256d res0 = alpha_is_one ? acc0[r] : _mm256_mul_pd(valpha, acc0[r]);
      __m256d res1 = alpha_is_one ? acc1[r] : _mm256_mul_pd(valpha, acc1[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const double *Crow = Crow_base + r * Cstride + n0 + n;
        const __m256d vc0 = _mm256_loadu_pd(Crow);
        const __m256d vc1 = _mm256_loadu_pd(Crow + 4);
        res0 = _mm256_add_pd(res0, beta_is_one ? vc0 : _mm256_mul_pd(vbeta, vc0));
        res1 = _mm256_add_pd(res1, beta_is_one ? vc1 : _mm256_mul_pd(vbeta, vc1));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res0 = _mm256_add_pd(res0, _mm256_loadu_pd(Yrow));
        res1 = _mm256_add_pd(res1, _mm256_loadu_pd(Yrow + 4));
      }
      _mm256_storeu_pd(Yrow, res0);
      _mm256_storeu_pd(Yrow + 4, res1);
    }
  }
  for (; n + 4 <= nb; n += 4) {
    __m256d acc[MR];
    for (std::size_t r = 0; r < MR; ++r) {
      acc[r] = _mm256_setzero_pd();
    }
    const auto accumulate_k = [&](std::size_t k) {
      const __m256d vb = _mm256_loadu_pd(Bmat + k * N + n0 + n);
      for (std::size_t r = 0; r < MR; ++r) {
        const __m256d va = _mm256_set1_pd(Apack[r * K + k]);
        acc[r] = _mm256_fmadd_pd(va, vb, acc[r]);
      }
    };
    std::size_t k = 0;
    for (; k + 4 <= K; k += 4) {
      accumulate_k(k);
      accumulate_k(k + 1);
      accumulate_k(k + 2);
      accumulate_k(k + 3);
    }
    for (; k < K; ++k) {
      accumulate_k(k);
    }
    for (std::size_t r = 0; r < MR; ++r) {
      double *Yrow = Yrow_base + r * Ystride + n0 + n;
      __m256d res = alpha_is_one ? acc[r] : _mm256_mul_pd(valpha, acc[r]);
      if (mode == GemmAccumMode::kInitBias) {
        const __m256d vc = _mm256_loadu_pd(Crow_base + r * Cstride + n0 + n);
        res = _mm256_add_pd(res, beta_is_one ? vc : _mm256_mul_pd(vbeta, vc));
      } else if (mode == GemmAccumMode::kAccumulate) {
        res = _mm256_add_pd(res, _mm256_loadu_pd(Yrow));
      }
      _mm256_storeu_pd(Yrow, res);
    }
  }
  if (n < nb) {
    GemmMicroKernel_Scalar_F64(MR, nb - n, K, alpha, beta, Bmat, N, Crow_base, Cstride, Yrow_base,
                               Ystride, n0 + n, mode, Apack);
  }
}

void GemmMicroKernel_AVX2FMA_F32(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                                 float beta, const float *Bmat, std::size_t N,
                                 const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                                 std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                 const float *Apack) {
  switch (mr) {
  case 1:
    return GemmMicroKernel_AVX2FMA_F32Impl<1>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 2:
    return GemmMicroKernel_AVX2FMA_F32Impl<2>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 3:
    return GemmMicroKernel_AVX2FMA_F32Impl<3>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 4:
    return GemmMicroKernel_AVX2FMA_F32Impl<4>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 5:
    return GemmMicroKernel_AVX2FMA_F32Impl<5>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 6:
    return GemmMicroKernel_AVX2FMA_F32Impl<6>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  default:
    return GemmMicroKernel_Scalar_F32(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                      Yrow_base, Ystride, n0, mode, Apack);
  }
}

void GemmMicroKernel_AVX2FMA_F64(std::size_t mr, std::size_t nb, std::size_t K, double alpha,
                                 double beta, const double *Bmat, std::size_t N,
                                 const double *Crow_base, std::size_t Cstride, double *Yrow_base,
                                 std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                                 const double *Apack) {
  switch (mr) {
  case 1:
    return GemmMicroKernel_AVX2FMA_F64Impl<1>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 2:
    return GemmMicroKernel_AVX2FMA_F64Impl<2>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 3:
    return GemmMicroKernel_AVX2FMA_F64Impl<3>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 4:
    return GemmMicroKernel_AVX2FMA_F64Impl<4>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 5:
    return GemmMicroKernel_AVX2FMA_F64Impl<5>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  case 6:
    return GemmMicroKernel_AVX2FMA_F64Impl<6>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                              Yrow_base, Ystride, n0, mode, Apack);
  default:
    return GemmMicroKernel_Scalar_F64(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                      Yrow_base, Ystride, n0, mode, Apack);
  }
}

void GemmConvertBFloat16ToFloat32_AVX2(const std::uint16_t *src, float *dst, std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m128i halves = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
    // BF16 -> FP32 is a zero-extend to 32 bits followed by a 16-bit left shift.
    const __m256i widened = _mm256_slli_epi32(_mm256_cvtepu16_epi32(halves), 16);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(widened));
  }
  for (; i < n; ++i) {
    dst[i] = detail::Bfloat16BitsToFloat(src[i]);
  }
}

void GemmPackTransposeFloat32_AVX2(const float *src, std::size_t src_stride, float *dst,
                                   std::size_t dst_rows, std::size_t dst_cols) {
  std::size_t row = 0;
  for (; row + 4 <= dst_rows; row += 4) {
    std::size_t column = 0;
    for (; column + 4 <= dst_cols; column += 4) {
      Transpose4x4Float(src + column * src_stride + row, src_stride, dst + row * dst_cols + column,
                        dst_cols);
    }
    for (; column < dst_cols; ++column) {
      for (std::size_t tile_row = 0; tile_row < 4; ++tile_row) {
        dst[(row + tile_row) * dst_cols + column] = src[column * src_stride + row + tile_row];
      }
    }
  }
  for (; row < dst_rows; ++row) {
    for (std::size_t column = 0; column < dst_cols; ++column) {
      dst[row * dst_cols + column] = src[column * src_stride + row];
    }
  }
}

void GemmPackTransposeFloat64_AVX2(const double *src, std::size_t src_stride, double *dst,
                                   std::size_t dst_rows, std::size_t dst_cols) {
  std::size_t row = 0;
  for (; row + 4 <= dst_rows; row += 4) {
    std::size_t column = 0;
    for (; column + 4 <= dst_cols; column += 4) {
      Transpose4x4Double(src + column * src_stride + row, src_stride, dst + row * dst_cols + column,
                         dst_cols);
    }
    for (; column < dst_cols; ++column) {
      for (std::size_t tile_row = 0; tile_row < 4; ++tile_row) {
        dst[(row + tile_row) * dst_cols + column] = src[column * src_stride + row + tile_row];
      }
    }
  }
  for (; row < dst_rows; ++row) {
    for (std::size_t column = 0; column < dst_cols; ++column) {
      dst[row * dst_cols + column] = src[column * src_stride + row];
    }
  }
}

void GemmPackTransposeBFloat16ToFloat32_AVX2(const std::uint16_t *src, std::size_t src_stride,
                                             float *dst, std::size_t dst_rows,
                                             std::size_t dst_cols) {
  PackTransposeHalfToFloat32<true>(src, src_stride, dst, dst_rows, dst_cols);
}

void GemmConvertFloat32ToBFloat16_AVX2(const float *src, std::uint16_t *dst, std::size_t n) {
  const __m256i absolute_mask = _mm256_set1_epi32(0x7fffffff);
  const __m256i infinity = _mm256_set1_epi32(0x7f800000);
  const __m256i rounding_base = _mm256_set1_epi32(0x7fff);
  const __m256i one = _mm256_set1_epi32(1);
  const __m256i quiet_nan = _mm256_set1_epi32(0x40);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256i bits = _mm256_castps_si256(_mm256_loadu_ps(src + i));
    const __m256i upper = _mm256_srli_epi32(bits, 16);
    const __m256i rounding = _mm256_add_epi32(rounding_base, _mm256_and_si256(upper, one));
    const __m256i rounded = _mm256_srli_epi32(_mm256_add_epi32(bits, rounding), 16);
    const __m256i is_nan = _mm256_cmpgt_epi32(_mm256_and_si256(bits, absolute_mask), infinity);
    const __m256i canonical_nan = _mm256_or_si256(upper, quiet_nan);
    const __m256i converted = _mm256_blendv_epi8(rounded, canonical_nan, is_nan);
    const __m128i packed =
        _mm_packus_epi32(_mm256_castsi256_si128(converted), _mm256_extracti128_si256(converted, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i), packed);
  }
  for (; i < n; ++i) {
    dst[i] = detail::FloatToBFloat16Bits(src[i]);
  }
}

void GemmDecodeFloat8ToFloat32_AVX2(const float *table, const std::uint8_t *src, float *dst,
                                    std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    // Load eight Float8 bytes, zero-extend the indices to 32 bits, and gather
    // the exact decoded float32 values from the per-format table.
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src + i));
    const __m256i indices = _mm256_cvtepu8_epi32(bytes);
    _mm256_storeu_ps(dst + i, _mm256_i32gather_ps(table, indices, 4));
  }
  for (; i < n; ++i) {
    dst[i] = table[src[i]];
  }
}

void GemmMicroKernel_AVX2BF16(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                              float beta, const std::uint16_t *Bmat, std::size_t N,
                              const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                              std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                              const std::uint16_t *Apack) {
  switch (mr) {
  case 1:
    return GemmMicroKernel_AVX2HalfImpl<true, 1>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                 Yrow_base, Ystride, n0, mode, Apack);
  case 2:
    return GemmMicroKernel_AVX2HalfImpl<true, 2>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                 Yrow_base, Ystride, n0, mode, Apack);
  case 3:
    return GemmMicroKernel_AVX2HalfImpl<true, 3>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                 Yrow_base, Ystride, n0, mode, Apack);
  case 4:
    return GemmMicroKernel_AVX2HalfImpl<true, 4>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                 Yrow_base, Ystride, n0, mode, Apack);
  case 5:
    return GemmMicroKernel_AVX2HalfImpl<true, 5>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                 Yrow_base, Ystride, n0, mode, Apack);
  case 6:
    return GemmMicroKernel_AVX2HalfImpl<true, 6>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                 Yrow_base, Ystride, n0, mode, Apack);
  default:
    return GemmMicroKernel_ScalarBf16(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                      Yrow_base, Ystride, n0, mode, Apack);
  }
}

#ifdef ONNX_LIGHT_CPU_HAVE_F16C
void GemmConvertFloat16ToFloat32_F16C(const std::uint16_t *src, float *dst, std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m128i halves = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
    _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(halves));
  }
  for (; i < n; ++i) {
    dst[i] = detail::Float16BitsToFloat(src[i]);
  }
}

void GemmPackTransposeFloat16ToFloat32_F16C(const std::uint16_t *src, std::size_t src_stride,
                                            float *dst, std::size_t dst_rows,
                                            std::size_t dst_cols) {
  PackTransposeHalfToFloat32<false>(src, src_stride, dst, dst_rows, dst_cols);
}

void GemmConvertFloat32ToFloat16_F16C(const float *src, std::uint16_t *dst, std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 values = _mm256_loadu_ps(src + i);
    const __m128i halves = _mm256_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i), halves);
    const int nan_mask = _mm256_movemask_ps(_mm256_cmp_ps(values, values, _CMP_UNORD_Q));
    if (nan_mask != 0) {
      for (int lane = 0; lane < 8; ++lane) {
        if ((nan_mask & (1 << lane)) != 0) {
          dst[i + static_cast<std::size_t>(lane)] =
              detail::FloatToFloat16Bits(src[i + static_cast<std::size_t>(lane)]);
        }
      }
    }
  }
  for (; i < n; ++i) {
    dst[i] = detail::FloatToFloat16Bits(src[i]);
  }
}

void GemmMicroKernel_AVX2F16C(std::size_t mr, std::size_t nb, std::size_t K, float alpha,
                              float beta, const std::uint16_t *Bmat, std::size_t N,
                              const float *Crow_base, std::size_t Cstride, float *Yrow_base,
                              std::size_t Ystride, std::size_t n0, GemmAccumMode mode,
                              const std::uint16_t *Apack) {
  switch (mr) {
  case 1:
    return GemmMicroKernel_AVX2HalfImpl<false, 1>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                  Yrow_base, Ystride, n0, mode, Apack);
  case 2:
    return GemmMicroKernel_AVX2HalfImpl<false, 2>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                  Yrow_base, Ystride, n0, mode, Apack);
  case 3:
    return GemmMicroKernel_AVX2HalfImpl<false, 3>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                  Yrow_base, Ystride, n0, mode, Apack);
  case 4:
    return GemmMicroKernel_AVX2HalfImpl<false, 4>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                  Yrow_base, Ystride, n0, mode, Apack);
  case 5:
    return GemmMicroKernel_AVX2HalfImpl<false, 5>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                  Yrow_base, Ystride, n0, mode, Apack);
  case 6:
    return GemmMicroKernel_AVX2HalfImpl<false, 6>(nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                                  Yrow_base, Ystride, n0, mode, Apack);
  default:
    return GemmMicroKernel_ScalarFp16(mr, nb, K, alpha, beta, Bmat, N, Crow_base, Cstride,
                                      Yrow_base, Ystride, n0, mode, Apack);
  }
}

void GemmFloat16SkinnyM_F16C(bool trans_a, std::size_t M, std::size_t N, std::size_t K, float alpha,
                             const std::uint16_t *A, const std::uint16_t *B, float *Y) {
  GemmHalfSkinnyM<false>(trans_a, M, N, K, alpha, A, B, Y);
}

void GemmFloat16SkinnyN_F16C(std::size_t M, std::size_t K, float alpha, const std::uint16_t *A,
                             const std::uint16_t *B, float *Y) {
  GemmHalfSkinnyN<false>(M, K, alpha, A, B, Y);
}
#endif

} // namespace onnx_light_cpu
