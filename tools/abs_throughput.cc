// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/math/math.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

void Measure(const std::vector<float> &input, std::vector<float> &output, std::size_t samples,
             const char *mode) {
  auto tuning = ml_cpu_compute::DefaultAbsFloat32ExecutionTuning();
  if (std::strcmp(mode, "cached") == 0) {
    tuning.streaming_store_threshold_bytes = 0;
  } else if (std::strcmp(mode, "streaming") == 0) {
    tuning.streaming_store_threshold_bytes = 1;
  }
  const auto run = [&] {
    ml_cpu_compute::AbsFloat32WithTuning(input.data(), output.data(), input.size(), tuning);
  };
  for (std::size_t warmup = 0; warmup < 10; ++warmup) {
    run();
  }
  std::vector<double> timings;
  timings.reserve(samples);
  for (std::size_t sample = 0; sample < samples; ++sample) {
    const auto begin = std::chrono::steady_clock::now();
    run();
    const auto end = std::chrono::steady_clock::now();
    timings.push_back(std::chrono::duration<double>(end - begin).count());
  }
  std::sort(timings.begin(), timings.end());
  std::printf("simd=%d threads=1 size=%zu mode=%s median_us=%.2f p10_us=%.2f p90_us=%.2f\n",
              static_cast<int>(ml_cpu_compute::DetectSimdLevel()), input.size(), mode,
              timings[samples / 2] * 1e6, timings[samples / 10] * 1e6,
              timings[samples - 1 - samples / 10] * 1e6);
}

} // namespace

int main(int argc, char **argv) {
  const std::size_t count =
      argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : 4'194'304;
  const std::size_t samples =
      argc > 2 ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10)) : 101;
  const char *mode = argc > 3 ? argv[3] : "default";
  if (count == 0 || samples == 0 ||
      (std::strcmp(mode, "default") != 0 && std::strcmp(mode, "cached") != 0 &&
       std::strcmp(mode, "streaming") != 0)) {
    return 2;
  }
  std::vector<float> input(count, -1.0f);
  std::vector<float> output(count);
  Measure(input, output, samples, mode);
  return std::all_of(output.begin(), output.end(), [](float value) { return value == 1.0f; }) ? 0
                                                                                              : 1;
}
