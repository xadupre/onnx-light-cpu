// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/simd_level.h"
#include "onnx_light_cpu/kernels/math/normalization_kernel.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;

template <typename T> rt::Tensor MakeTensor(const rt::Shape &shape, const std::vector<T> &values) {
  if constexpr (std::is_same_v<T, float>) {
    return rt::Tensor::FromFloat("", shape, values);
  } else {
    return rt::Tensor::FromDouble("", shape, values);
  }
}

template <typename T>
void Benchmark(std::int64_t batch, std::int64_t channels, std::int64_t spatial, bool training,
               std::ofstream &outputs, std::ofstream &samples) {
  const onnx_light_cpu::BatchNormalizationKernel kernel(
      rt::KernelContext(rt::OpsetId(std::string(), 15)));
  std::vector<T> values(static_cast<std::size_t>(batch * channels * spatial));
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<T>(static_cast<int>(i % 29) - 14) * static_cast<T>(0.125);
  }
  const auto x = MakeTensor<T>({batch, channels, spatial}, values);
  const auto scale = MakeTensor<T>({channels}, std::vector<T>(channels, static_cast<T>(0.75)));
  const auto bias = MakeTensor<T>({channels}, std::vector<T>(channels, static_cast<T>(0.25)));
  const auto mean = MakeTensor<T>({channels}, std::vector<T>(channels, static_cast<T>(0.5)));
  const auto variance = MakeTensor<T>({channels}, std::vector<T>(channels, static_cast<T>(1.25)));
  const auto run = [&] { return kernel.Compute(x, scale, bias, mean, variance, training); };
  const auto expected = run();
  if (outputs.is_open()) {
    outputs.write(reinterpret_cast<const char *>(expected.y.bytes()),
                  static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (training) {
      for (const auto *stat : {&*expected.running_mean, &*expected.running_variance}) {
        outputs.write(reinterpret_cast<const char *>(stat->bytes()),
                      static_cast<std::streamsize>(channels * sizeof(T)));
      }
    }
  }
  std::vector<double> timings;
  for (int iteration = -20; iteration < 101; ++iteration) {
    const auto begin = std::chrono::steady_clock::now();
    const auto actual = run();
    const auto end = std::chrono::steady_clock::now();
    if (iteration >= 0) {
      timings.push_back(std::chrono::duration<double>(end - begin).count());
      if (samples.is_open()) {
        samples << (std::is_same_v<T, float> ? "float32" : "float64") << ',' << batch << ','
                << channels << ',' << spatial << ',' << (training ? "training" : "inference") << ','
                << iteration << ',' << timings.back() << '\n';
      }
    }
    if (std::memcmp(actual.y.bytes(), expected.y.bytes(), values.size() * sizeof(T)) != 0) {
      throw std::runtime_error("BatchNormalization output changed between repetitions.");
    }
    if (training &&
        (std::memcmp(actual.running_mean->bytes(), expected.running_mean->bytes(),
                     static_cast<std::size_t>(channels) * sizeof(T)) != 0 ||
         std::memcmp(actual.running_variance->bytes(), expected.running_variance->bytes(),
                     static_cast<std::size_t>(channels) * sizeof(T)) != 0)) {
      throw std::runtime_error("BatchNormalization statistics changed between repetitions.");
    }
  }
  std::sort(timings.begin(), timings.end());
  std::printf("%s,%lld,%lld,%lld,%s,%.9g,%.9g,%.9g\n",
              std::is_same_v<T, float> ? "float32" : "float64", static_cast<long long>(batch),
              static_cast<long long>(channels), static_cast<long long>(spatial),
              training ? "training" : "inference", timings[50], timings[25], timings[75]);
}

} // namespace

int main(int argc, char **argv) {
  std::ofstream outputs, samples;
  if (argc > 1) {
    outputs.exceptions(std::ios::failbit | std::ios::badbit);
    outputs.open(argv[1], std::ios::binary);
    samples.exceptions(std::ios::failbit | std::ios::badbit);
    samples.open(std::string(argv[1]) + ".samples.csv");
    samples.precision(17);
    samples << "dtype,batch,channels,spatial,mode,sample,seconds\n";
  }
  std::fprintf(stderr, "detected_simd_level=%d effective_threads=%lld\n",
               static_cast<int>(onnx_light_cpu::DetectSimdLevel()),
               static_cast<long long>(onnx_light_cpu::ExecutionThreadCount()));
  std::puts("dtype,batch,channels,spatial,mode,median_s,p25_s,p75_s");
  for (const auto &[batch, channels, spatial] :
       {std::tuple<std::int64_t, std::int64_t, std::int64_t>{1, 1, 1},
        {5, 3, 1},
        {17, 15, 1},
        {17, 16, 1},
        {17, 17, 1},
        {17, 64, 1},
        {2000, 64, 1},
        {2000, 65, 1},
        {2000, 256, 1},
        {1, 65537, 1},
        {128, 64, 4},
        {32, 64, 64}}) {
    for (bool training : {false, true}) {
      Benchmark<float>(batch, channels, spatial, training, outputs, samples);
      Benchmark<double>(batch, channels, spatial, training, outputs, samples);
    }
  }
  if (outputs.is_open()) {
    outputs.close();
    samples.close();
  }
}
