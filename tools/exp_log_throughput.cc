// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Isolated float32 Exp/Log throughput driver (pass "tanh", "sigmoid", or "softmax" as the second
// argument for a direct scalar/AVX2/AVX-512 comparison). This deliberately measures preallocated
// buffers and reports the environment separately from the end-to-end parity runner.

#include "ml_cpu_compute/math/math.h"
#include "ml_cpu_compute/simd_level.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

const char *IsaName(ml_cpu_compute::SimdLevel level) {
  switch (level) {
  case ml_cpu_compute::SimdLevel::kNone:
    return "scalar";
  case ml_cpu_compute::SimdLevel::kSSE2:
    return "SSE2";
  case ml_cpu_compute::SimdLevel::kAVX:
    return "AVX";
  case ml_cpu_compute::SimdLevel::kAVX2:
    return "AVX2";
  case ml_cpu_compute::SimdLevel::kAVX512:
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

int MeasureActivation(std::size_t samples, bool sigmoid) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
  if (ml_cpu_compute::DetectSimdLevel() < ml_cpu_compute::SimdLevel::kAVX2 ||
      !ml_cpu_compute::CpuSupportsFma()) {
    std::fprintf(stderr, "direct activation comparison requires AVX2 and FMA\n");
    return 2;
  }
  std::printf("operator,size,offset_floats,median_seconds,iqr_seconds,speedup\n");
  const std::vector<std::size_t> sizes =
      sigmoid ? std::vector<std::size_t>{8,    15,    16,    17,    31,     32,      33,
                                         1024, 32768, 65535, 65536, 131072, 1048576, 4194304}
              : std::vector<std::size_t>{8,    15,    16,     17,     31,     32,      33,      100,
                                         1000, 10000, 100000, 202048, 202055, 3232768, 25862144};
  for (const std::size_t count : sizes) {
    for (const std::size_t offset : {0, 1}) {
      std::vector<float> input_storage(count + 32), output_storage(count + 32);
      const auto aligned = [offset](std::vector<float> &storage) {
        const auto address = reinterpret_cast<std::uintptr_t>(storage.data());
        return reinterpret_cast<float *>((address + 63u) & ~std::uintptr_t{63u}) + offset;
      };
      float *input = aligned(input_storage);
      float *output = aligned(output_storage);
      for (std::size_t i = 0; i < count; ++i) {
        input[i] = -8.0f + 16.0f * static_cast<float>(i) / static_cast<float>(count);
      }
      using Kernel = void (*)(const float *, float *, std::size_t);
      const Kernel scalar_sigmoid = [](const float *x, float *y, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
          const float exponent = std::exp(-std::abs(x[i]));
          y[i] = (std::signbit(x[i]) ? exponent : 1.0f) / (1.0f + exponent);
        }
      };
      std::vector<Kernel> kernels{sigmoid ? scalar_sigmoid : ml_cpu_compute::TanhFloat32_Scalar,
                                  sigmoid ? ml_cpu_compute::SigmoidFloat32_AVX2_FMA
                                          : ml_cpu_compute::TanhFloat32_AVX2_FMA};
      std::vector<const char *> names =
          sigmoid ? std::vector<const char *>{"SigmoidScalar", "SigmoidAVX2FMA"}
                  : std::vector<const char *>{"TanhScalar", "TanhAVX2FMA"};
#ifdef ML_CPU_COMPUTE_HAVE_AVX512
      if (ml_cpu_compute::DetectSimdLevel() >= ml_cpu_compute::SimdLevel::kAVX512) {
        kernels.push_back(sigmoid ? ml_cpu_compute::SigmoidFloat32_AVX512
                                  : ml_cpu_compute::TanhFloat32_AVX512);
        names.push_back(sigmoid ? "SigmoidAVX512" : "TanhAVX512");
      }
#endif
      for (int warmup = 0; warmup < 3; ++warmup) {
        for (const auto kernel : kernels) {
          kernel(input, output, count);
        }
      }
      std::vector<std::vector<double>> timings(kernels.size());
      for (auto &values : timings) {
        values.reserve(samples);
      }
      const auto iterations = std::max<std::size_t>(1, 100000 / count);
      for (std::size_t sample = 0; sample < samples; ++sample) {
        for (std::size_t order = 0; order < kernels.size(); ++order) {
          const auto index = (sample + order) % kernels.size();
          const auto start = std::chrono::steady_clock::now();
          for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            kernels[index](input, output, count);
          }
          const auto stop = std::chrono::steady_clock::now();
          timings[index].push_back(std::chrono::duration<double>(stop - start).count() /
                                   static_cast<double>(iterations));
        }
      }
      const auto scalar_median = Summary(timings[0]).first;
      for (std::size_t index = 0; index < kernels.size(); ++index) {
        const auto [median, iqr] = Summary(timings[index]);
        std::printf("%s,%zu,%zu,%.9g,%.9g,%.6g\n", names[index], count, offset, median, iqr,
                    scalar_median / median);
      }
    }
  }
  return 0;
#else
  (void)samples;
  (void)sigmoid;
  std::fprintf(stderr, "direct activation comparison requires compiled AVX2/FMA support\n");
  return 2;
#endif
}

int MeasureSoftmax(std::size_t samples) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_FMA
  if (ml_cpu_compute::DetectSimdLevel() < ml_cpu_compute::SimdLevel::kAVX2 ||
      !ml_cpu_compute::CpuSupportsFma()) {
    std::fprintf(stderr, "direct Softmax comparison requires AVX2 and FMA\n");
    return 2;
  }
  std::printf("operator,rows,columns,offset_floats,median_seconds,iqr_seconds\n");
  for (const std::size_t rows : {1, 32, 1024}) {
    for (const std::size_t columns : {1023, 1024, 1025}) {
      for (const std::size_t offset : {0, 1}) {
        std::vector<float> input(rows * columns + 32), output(input.size());
        const auto aligned = [offset](std::vector<float> &storage) {
          const auto address = reinterpret_cast<std::uintptr_t>(storage.data());
          return reinterpret_cast<float *>((address + 63u) & ~std::uintptr_t{63u}) + offset;
        };
        float *x = aligned(input);
        float *y = aligned(output);
        for (std::size_t i = 0; i < rows * columns; ++i) {
          x[i] = static_cast<float>(static_cast<int>(i % 257) - 128) / 32.0f;
        }
        for (int warmup = 0; warmup < 10; ++warmup) {
          ml_cpu_compute::SoftmaxFloat32_AVX2_FMA(x, y, rows, columns);
        }
        std::vector<double> timings;
        const auto iterations = std::max<std::size_t>(1, 100000 / (rows * columns));
        for (std::size_t sample = 0; sample < samples; ++sample) {
          const auto start = std::chrono::steady_clock::now();
          for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            ml_cpu_compute::SoftmaxFloat32_AVX2_FMA(x, y, rows, columns);
          }
          timings.push_back(
              std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() /
              static_cast<double>(iterations));
        }
        const auto [median, iqr] = Summary(timings);
        std::printf("SoftmaxAVX2FMA,%zu,%zu,%zu,%.9g,%.9g\n", rows, columns, offset, median, iqr);
      }
    }
  }
  return 0;
#else
  (void)samples;
  std::fprintf(stderr, "direct Softmax comparison requires compiled AVX2/FMA support\n");
  return 2;
#endif
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
              static_cast<long>(__cplusplus), IsaName(ml_cpu_compute::DetectSimdLevel()), samples);
  if (argc > 2 && (std::string(argv[2]) == "tanh" || std::string(argv[2]) == "sigmoid")) {
    return MeasureActivation(samples, std::string(argv[2]) == "sigmoid");
  }
  if (argc > 2 && std::string(argv[2]) == "softmax") {
    return MeasureSoftmax(samples);
  }
  std::printf("operator,size,median_seconds,iqr_seconds,cycles_per_element\n");
  for (const std::size_t size : Sizes()) {
    const auto exp = Measure(ml_cpu_compute::ExpFloat32, size, samples);
    const auto log = Measure(ml_cpu_compute::LogFloat32, size, samples);
    for (const auto &entry :
         {std::pair<const char *, std::vector<double>>{"Exp", exp}, {"Log", log}}) {
      const auto [median, iqr] = Summary(entry.second);
      std::printf("%s,%zu,%.9g,%.9g,nan\n", entry.first, size, median, iqr);
    }
  }
  return 0;
}
