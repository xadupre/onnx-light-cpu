// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/normalization_kernel.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

void Benchmark(std::size_t count, bool shifted) {
  std::vector<float> input(count);
  for (std::size_t i = 0; i < count; ++i) {
    const float variation = static_cast<float>(static_cast<std::int64_t>(i % 17) - 8) * 0.125F;
    input[i] = (shifted ? 65536.0F : 0.0F) + variation;
  }
  const auto expected = onnx_light_cpu::ComputeNormalizationMomentsFloat32(input.data(), count);
  const std::size_t calls_per_sample = std::max<std::size_t>(1, (1U << 20) / count);
  std::vector<double> timings;
  timings.reserve(101);
  float checksum = 0.0F;
  for (int iteration = -20; iteration < 101; ++iteration) {
    const auto begin = std::chrono::steady_clock::now();
    onnx_light_cpu::Float32NormalizationMoments moments{};
    for (std::size_t call = 0; call < calls_per_sample; ++call) {
      moments = onnx_light_cpu::ComputeNormalizationMomentsFloat32(input.data(), input.size());
      checksum += moments.mean + moments.variance;
    }
    const auto end = std::chrono::steady_clock::now();
    if (iteration >= 0) {
      timings.push_back(std::chrono::duration<double>(end - begin).count() /
                        static_cast<double>(calls_per_sample));
    }
    if (moments.mean != expected.mean || moments.variance != expected.variance) {
      std::fprintf(stderr, "normalization moments changed between repetitions\n");
      std::abort();
    }
  }
  std::sort(timings.begin(), timings.end());
  std::printf("%zu,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", count, shifted ? "shifted" : "centered",
              timings[50], timings[25], timings[75], static_cast<double>(expected.mean),
              static_cast<double>(expected.variance), static_cast<double>(checksum));
}

} // namespace

int main() {
  std::fprintf(stderr, "detected_simd_level=%d\n",
               static_cast<int>(onnx_light_cpu::DetectSimdLevel()));
  std::puts("count,distribution,median_s,p25_s,p75_s,mean,variance,checksum");
  for (const std::size_t count : {31, 32, 33, 127, 256, 1024, 4096, 16384, 65536}) {
    Benchmark(count, false);
    Benchmark(count, true);
  }
}
