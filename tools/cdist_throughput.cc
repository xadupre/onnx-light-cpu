// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Isolated, single-thread CDist benchmark. Build with
// -DONNX_LIGHT_CPU_BUILD_BENCHMARKS=ON; run cdist_throughput [case substring].
// All per-call packing and dispatch costs are included in the median seconds.
// Use tools/benchmark_cdist_parity.py for end-to-end ONNX Runtime comparisons.

#include "onnx_light_cpu/impl/com_microsoft/cdist.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

namespace {

struct Case {
  const char *name;
  std::size_t m, k, n;
};

constexpr Case kCases[] = {
    {"pack_small", 8, 32, 64},
    {"pack_before", 31, 32, 64},
    {"pack_at", 32, 32, 64},
    {"pack_after", 33, 32, 64},
    {"pack_limit_before", 63, 32, 256},
    {"pack_limit_at", 64, 32, 256},
    {"pack_column_tail", 64, 17, 64},
    {"pack_limit_tail", 64, 17, 256},
    {"tiny", 3, 5, 16},
    {"small", 64, 64, 64},
    {"representative", 256, 128, 64},
    {"large", 512, 512, 128},
    {"short", 256, 128, 3},
    {"feature_before", 256, 128, 15},
    {"feature_after", 256, 128, 17},
    {"tail", 255, 127, 65},
    {"large_tail", 511, 513, 129},
    {"long_features", 4, 5, 4097},
    {"single_row", 1, 128, 64},
    {"single_column", 256, 1, 64},
};

template <typename T> void Measure(const Case &shape, onnx_light_cpu::CDistMetric metric) {
  std::vector<T> a(shape.m * shape.n), b(shape.k * shape.n), c(shape.m * shape.k);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = T(static_cast<int>(i % 29) - 14) / T(16);
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = T(static_cast<int>(i % 23) - 11) / T(32);
  }
  const auto run = [&] {
    if constexpr (std::is_same_v<T, float>) {
      onnx_light_cpu::CDistFloat32(a.data(), b.data(), c.data(), shape.m, shape.k, shape.n, metric);
    } else {
      onnx_light_cpu::CDistFloat64(a.data(), b.data(), c.data(), shape.m, shape.k, shape.n, metric);
    }
  };
  for (int warmup = 0; warmup < 10; ++warmup) {
    run();
  }
  const std::size_t iterations = std::max<std::size_t>(1, 1000000 / (shape.m * shape.k * shape.n));
  std::vector<double> samples;
  for (int sample = 0; sample < 51; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
      run();
    }
    samples.push_back(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() /
        static_cast<double>(iterations));
  }
  std::sort(samples.begin(), samples.end());
  std::printf("%s,%s,%s,%zu,%zu,%zu,%.9g,%.9g\n", shape.name,
              std::is_same_v<T, float> ? "float32" : "float64",
              metric == onnx_light_cpu::CDistMetric::kEuclidean ? "euclidean" : "sqeuclidean",
              shape.m, shape.k, shape.n, samples[samples.size() / 2],
              static_cast<double>(c.front()));
}

} // namespace

int main(int argc, char **argv) {
  std::printf("# SIMD level=%d (0=scalar,1=SSE2,2=AVX,3=AVX2,4=AVX512), threads=1\n",
              static_cast<int>(onnx_light_cpu::DetectSimdLevel()));
  std::printf("case,dtype,metric,m,k,n,median_seconds,first_output\n");
  for (const Case &shape : kCases) {
    if (argc > 1 && std::string(shape.name).find(argv[1]) == std::string::npos) {
      continue;
    }
    for (const auto metric :
         {onnx_light_cpu::CDistMetric::kSqEuclidean, onnx_light_cpu::CDistMetric::kEuclidean}) {
      Measure<float>(shape, metric);
      Measure<double>(shape, metric);
    }
  }
}
