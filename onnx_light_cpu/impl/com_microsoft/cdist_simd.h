// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/com_microsoft/cdist.h"

#include <algorithm>
#include <cstddef>

namespace onnx_light_cpu {

inline constexpr std::size_t kCDistPackedMaxFeatures = 256;

template <std::size_t Lanes> bool CDistUsePackedRows(std::size_t m, std::size_t k, std::size_t n) {
  return n > 0 && n <= kCDistPackedMaxFeatures && k >= Lanes &&
         (k % Lanes == 0 || k >= 4 * Lanes) && m >= (n <= 128 ? 32u : 64u);
}

// Keep packing bounded and amortize it over several A rows. Each SIMD lane
// accumulates one distance, avoiding horizontal reductions and feature tails.
// Unlike a GEMM expansion this still computes (a - b)^2 directly.
template <typename T, typename Vector, std::size_t Lanes, typename Load, typename Store,
          typename Broadcast, typename Subtract, typename MultiplyAdd, typename Sqrt>
bool CDistPackedRows(const T *a, const T *b, T *c, std::size_t k, std::size_t n, CDistMetric metric,
                     std::size_t row_begin, std::size_t row_end, Vector zero, Load load,
                     Store store, Broadcast broadcast, Subtract subtract, MultiplyAdd multiply_add,
                     Sqrt square_root) {
  if (!CDistUsePackedRows<Lanes>(row_end - row_begin, k, n)) {
    return false;
  }
  alignas(64) T packed[kCDistPackedMaxFeatures * Lanes];
  for (std::size_t col = 0; col < k; col += Lanes) {
    const std::size_t columns = std::min(Lanes, k - col);
    for (std::size_t feature = 0; feature < n; ++feature) {
      for (std::size_t lane = 0; lane < columns; ++lane) {
        packed[feature * Lanes + lane] = b[(col + lane) * n + feature];
      }
      std::fill(packed + feature * Lanes + columns, packed + (feature + 1) * Lanes, T(0));
    }
    const auto finish = [&](Vector sum, std::size_t row) {
      if (metric == CDistMetric::kEuclidean) {
        sum = square_root(sum);
      }
      T *output = c + row * k + col;
      if (columns == Lanes) {
        store(output, sum);
      } else {
        T tail[Lanes];
        store(tail, sum);
        std::copy_n(tail, columns, output);
      }
    };
    std::size_t row = row_begin;
    for (; row + 4 <= row_end; row += 4) {
      Vector sums[4] = {zero, zero, zero, zero};
      for (std::size_t feature = 0; feature < n; ++feature) {
        const Vector values = load(packed + feature * Lanes);
        for (std::size_t r = 0; r < 4; ++r) {
          const Vector difference = subtract(broadcast(a[(row + r) * n + feature]), values);
          sums[r] = multiply_add(difference, difference, sums[r]);
        }
      }
      for (std::size_t r = 0; r < 4; ++r) {
        finish(sums[r], row + r);
      }
    }
    for (; row < row_end; ++row) {
      Vector sum = zero;
      for (std::size_t feature = 0; feature < n; ++feature) {
        const Vector difference =
            subtract(broadcast(a[row * n + feature]), load(packed + feature * Lanes));
        sum = multiply_add(difference, difference, sum);
      }
      finish(sum, row);
    }
  }
  return true;
}

} // namespace onnx_light_cpu
