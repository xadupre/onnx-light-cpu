// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

template <typename Function>
double Measure(Function function, const std::vector<float> &input, std::vector<float> &output,
               std::size_t samples) {
  for (std::size_t warmup = 0; warmup < 10; ++warmup) {
    function(input.data(), output.data(), input.size());
  }
  std::vector<double> timings;
  timings.reserve(samples);
  for (std::size_t sample = 0; sample < samples; ++sample) {
    const auto begin = std::chrono::steady_clock::now();
    function(input.data(), output.data(), input.size());
    const auto end = std::chrono::steady_clock::now();
    timings.push_back(std::chrono::duration<double>(end - begin).count());
  }
  std::sort(timings.begin(), timings.end());
  return timings[timings.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  const std::size_t count =
      argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : 4'194'304;
  const std::size_t samples =
      argc > 2 ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10)) : 101;
  if (count == 0 || samples == 0) {
    return 2;
  }
  std::vector<float> input(count, -1.0f);
  std::vector<float> output(count);
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  const double cached = Measure(onnx_light_cpu::AbsFloat32_AVX512, input, output, samples);
  const double streaming =
      Measure(onnx_light_cpu::AbsFloat32_AVX512Streaming, input, output, samples);
  std::printf("size=%zu cached_us=%.2f streaming_us=%.2f speedup=%.3f\n", count, cached * 1e6,
              streaming * 1e6, cached / streaming);
#else
  std::fprintf(stderr, "AVX-512 is unavailable in this build\n");
  return 77;
#endif
}
