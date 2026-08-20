// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Isolated float32 Exp/Log throughput driver.  This deliberately measures
// preallocated buffers and reports the environment separately from the
// end-to-end parity runner.

#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

const char *IsaName(onnx_light_cpu::SimdLevel level) {
  switch (level) {
  case onnx_light_cpu::SimdLevel::kNone:
    return "scalar";
  case onnx_light_cpu::SimdLevel::kSSE2:
    return "SSE2";
  case onnx_light_cpu::SimdLevel::kAVX:
    return "AVX";
  case onnx_light_cpu::SimdLevel::kAVX2:
    return "AVX2";
  case onnx_light_cpu::SimdLevel::kAVX512:
    return "AVX-512";
  }
  return "unknown";
}

std::string CpuModel() {
  // Linux topology files are used when available; other platforms report unknown.
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.rfind("model name", 0) == 0) {
      return line.substr(line.find(':') + 2);
    }
  }
  return "unknown";
}

std::size_t PhysicalCores() {
  std::unordered_set<std::string> cores;
  for (std::size_t cpu = 0; cpu < std::thread::hardware_concurrency(); ++cpu) {
    std::ifstream core("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/core_id");
    std::string id;
    if (core >> id) {
      cores.insert(id);
    }
  }
  return cores.empty() ? std::thread::hardware_concurrency() : cores.size();
}

std::vector<std::size_t> Sizes() {
  return {100, 1'000, 10'000, 100'000, 1'000'000, 4'194'304, 10'000'000, 100'000'000};
}

template <typename Function>
std::vector<double> Measure(Function function, std::size_t count, std::size_t samples) {
  std::vector<float> input(count);
  std::vector<float> output(count);
  for (std::size_t i = 0; i < count; ++i) {
    input[i] = -8.0f + 16.0f * static_cast<float>(i) / static_cast<float>(count);
  }
  for (std::size_t warmup = 0; warmup < 3; ++warmup) {
    function(input.data(), output.data(), count);
  }
  std::vector<double> timings;
  timings.reserve(samples);
  for (std::size_t sample = 0; sample < samples; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    function(input.data(), output.data(), count);
    const auto stop = std::chrono::steady_clock::now();
    timings.push_back(std::chrono::duration<double>(stop - start).count());
  }
  return timings;
}

std::pair<double, double> Summary(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  const double median =
      values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
  return {median, values[(values.size() * 3) / 4] - values[values.size() / 4]};
}

} // namespace

int main(int argc, char **argv) {
  const std::size_t samples =
      argc > 1 ? static_cast<std::size_t>(std::strtoul(argv[1], nullptr, 10)) : 9;
  if (samples == 0) {
    std::fprintf(stderr, "sample count must be positive\n");
    return 2;
  }
  std::printf("timestamp=steady  cpu_model=%s  logical_cpus=%u  physical_cores=%u\n",
              CpuModel().c_str(), std::thread::hardware_concurrency(),
              static_cast<unsigned>(PhysicalCores()));
  std::printf("compiler=%s  cxx=%ld  isa=%s  samples=%zu  standalone_threads=1\n",
#if defined(__clang__)
              "clang",
#elif defined(__GNUC__)
              "gcc",
#elif defined(_MSC_VER)
              "msvc",
#else
              "unknown",
#endif
              static_cast<long>(__cplusplus), IsaName(onnx_light_cpu::DetectSimdLevel()), samples);
  std::printf("operator,size,median_seconds,iqr_seconds,cycles_per_element\n");
  for (const std::size_t size : Sizes()) {
    const auto exp = Measure(onnx_light_cpu::ExpFloat32, size, samples);
    const auto log = Measure(onnx_light_cpu::LogFloat32, size, samples);
    for (const auto &entry :
         {std::pair<const char *, std::vector<double>>{"Exp", exp}, {"Log", log}}) {
      const auto [median, iqr] = Summary(entry.second);
      std::printf("%s,%zu,%.9g,%.9g,nan\n", entry.first, size, median, iqr);
    }
  }
  return 0;
}
