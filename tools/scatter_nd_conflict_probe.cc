// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Standalone AVX-512CD experiment; not linked into the runtime.
// Build only on an AVX-512CD host with -mavx512f -mavx512cd -mavx2.
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using Clock = std::chrono::steady_clock;

__attribute__((noinline)) void Ordered(uint32_t *out, const uint32_t *updates,
                                       const uint64_t *offsets, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    std::memcpy(reinterpret_cast<char *>(out) + offsets[i], updates + i, 4);
  }
}

__attribute__((noinline)) void Conflict(uint32_t *out, const uint32_t *updates,
                                        const uint64_t *offsets, size_t count) {
  const __m512i reverse = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m512i locations = _mm512_loadu_si512(offsets + i);
    const __m512i reversed = _mm512_permutexvar_epi64(reverse, locations);
    const __mmask8 reverse_winners =
        _mm512_cmpeq_epi64_mask(_mm512_conflict_epi64(reversed), _mm512_setzero_si512());
    unsigned winners = 0;
    for (int j = 0; j < 8; ++j) {
      winners |= ((reverse_winners >> j) & 1u) << (7 - j);
    }
    const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(updates + i));
    _mm512_mask_i64scatter_epi32(out, static_cast<__mmask8>(winners), locations, values, 1);
  }
  Ordered(out, updates + i, offsets + i, count - i);
}

int main() {
  for (size_t count : {size_t{7}, size_t{8}, size_t{9}, size_t{17}}) {
    for (int mode = 0; mode < 3; ++mode) {
      std::vector<uint32_t> updates(count), scalar(32), simd(32);
      std::vector<uint64_t> offsets(count);
      for (size_t i = 0; i < count; ++i) {
        updates[i] = static_cast<uint32_t>(i + 1);
        offsets[i] = 4 * (mode == 0 ? i : mode == 1 ? (i * 7) % 16 : 3);
      }
      Ordered(scalar.data(), updates.data(), offsets.data(), count);
      Conflict(simd.data(), updates.data(), offsets.data(), count);
      if (scalar != simd) {
        return 1;
      }
    }
  }
  constexpr size_t rows = 65536;
  for (size_t count : {size_t{8192}, size_t{65536}}) {
    for (int mode = 0; mode != 3; ++mode) {
      std::vector<uint32_t> data(rows), updates(count), scalar(rows), simd(rows);
      std::vector<uint64_t> offsets(count);
      for (size_t i = 0; i < rows; ++i) {
        data[i] = static_cast<uint32_t>(i + 37);
      }
      for (size_t i = 0; i < count; ++i) {
        updates[i] = static_cast<uint32_t>(i * 13 + 7);
        offsets[i] = 4 * (mode == 0 ? i % rows : mode == 1 ? (i * 7) % 16 : 3);
      }
      std::memcpy(scalar.data(), data.data(), rows * 4);
      std::memcpy(simd.data(), data.data(), rows * 4);
      Ordered(scalar.data(), updates.data(), offsets.data(), count);
      Conflict(simd.data(), updates.data(), offsets.data(), count);
      if (scalar != simd) {
        return 1;
      }
      double times[2][9]{};
      for (int run = 0; run < 9; ++run) {
        for (int slot = 0; slot < 2; ++slot) {
          const int path = (run + slot) % 2;
          const auto begin = Clock::now();
          for (int trial = 0; trial < 100; ++trial) {
            auto &output = path ? simd : scalar;
            std::memcpy(output.data(), data.data(), rows * 4);
            if (path) {
              Conflict(output.data(), updates.data(), offsets.data(), count);
            } else {
              Ordered(output.data(), updates.data(), offsets.data(), count);
            }
          }
          times[path][run] = std::chrono::duration<double>(Clock::now() - begin).count() / 100;
        }
      }
      for (auto &path : times) {
        std::sort(path, path + 9);
      }
      std::printf("%zu,%s,%.9f,%.9f,%.3f\n", count,
                  mode == 0   ? "unique"
                  : mode == 1 ? "clustered"
                              : "duplicate",
                  times[0][4], times[1][4], times[1][4] / times[0][4]);
    }
  }
}
