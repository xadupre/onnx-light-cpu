// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/math/gemm/vnni/integer_gemm_vnni.h"
#include "ml_cpu_compute/simd_level.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

template <typename Fn> double MedianSeconds(Fn run) {
  for (int i = 0; i < 5; ++i) {
    run();
  }
  std::vector<double> samples;
  for (int i = 0; i < 31; ++i) {
    const auto start = std::chrono::steady_clock::now();
    run();
    samples.push_back(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

template <typename A, typename B> std::array<double, 2> CompareSeconds(A avx2, B avxvnni) {
  for (int i = 0; i < 5; ++i) {
    avx2();
    avxvnni();
  }
  std::array<std::vector<double>, 2> samples;
  const auto measure = [&](auto run, int index) {
    const auto start = std::chrono::steady_clock::now();
    run();
    samples[index].push_back(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
  };
  for (int i = 0; i < 31; ++i) {
    if (i & 1) {
      measure(avxvnni, 1);
      measure(avx2, 0);
    } else {
      measure(avx2, 0);
      measure(avxvnni, 1);
    }
  }
  for (auto &sample : samples) {
    std::sort(sample.begin(), sample.end());
  }
  return {samples[0][15], samples[1][15]};
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 1 && argc != 4 && argc != 6) {
    std::fprintf(stderr, "Usage: integer_gemm_throughput [M N K [AZ BZ]]\n");
    return 1;
  }
  const std::int64_t m = argc >= 4 ? std::atoll(argv[1]) : 1;
  const std::int64_t n = argc >= 4 ? std::atoll(argv[2]) : 4096;
  const std::int64_t k = argc >= 4 ? std::atoll(argv[3]) : 4096;
  if (m < 1 || n < 1 || k < 1 || m > 8192 || n > 8192 || k > 8192) {
    std::fprintf(stderr, "Dimensions must be in [1, 8192].\n");
    return 1;
  }
  using namespace ml_cpu_compute;
  ml_cpu_compute::detail::IntegerVnniDotFn dot = &ml_cpu_compute::detail::IntegerDotU8S8Scalar;
  const char *isa = "scalar";
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_INTEGER
  if (ml_cpu_compute::DetectSimdLevel() >= ml_cpu_compute::SimdLevel::kAVX2) {
    dot = &ml_cpu_compute::detail::IntegerDotU8S8Avx2;
    isa = "avx2";
  }
#endif
#ifdef ML_CPU_COMPUTE_HAVE_AVXVNNI
  if (ml_cpu_compute::IntegerMatMul2DUsesAvxVnni()) {
    dot = &ml_cpu_compute::detail::IntegerDotU8S8AvxVnni;
    isa = "avxvnni";
  }
#endif
#ifdef ML_CPU_COMPUTE_HAVE_AVX512VNNI
  if (ml_cpu_compute::IntegerMatMul2DUsesVnni()) {
    dot = &ml_cpu_compute::detail::IntegerDotU8S8Avx512Vnni;
    isa = "avx512vnni";
  }
#endif
  std::mt19937 rng(725);
  std::vector<std::uint8_t> a(m * k), b(k * n);
  std::generate(a.begin(), a.end(), [&] { return static_cast<std::uint8_t>(rng()); });
  std::generate(b.begin(), b.end(), [&] { return static_cast<std::uint8_t>(rng()); });
  std::vector<std::int32_t> c(m * n), expected(m * n);
  std::vector<std::int8_t> packed_b(k * n);
  std::vector<std::int64_t> sums(n);
  const int az = argc == 6 ? std::atoi(argv[4]) : 128;
  const int bz = argc == 6 ? std::atoi(argv[5]) : 0;
  if (az < 0 || az > 255 || bz < -128 || bz > 127) {
    std::fprintf(stderr, "Zero points must fit UINT8 (A) and INT8 (B).\n");
    return 1;
  }
  const auto pack = [&] {
    ml_cpu_compute::detail::PackS8ColRange(b.data(), true, 0, k, n, packed_b.data(), sums.data(), 0,
                                           n);
  };
  const auto packed = [&] {
    ml_cpu_compute::detail::IntegerMatMul2DWithDot(dot, a.data(), false, b.data(), true,
                                                   expected.data(), m, n, k, &az, 1, &bz, 1);
  };
  const auto dispatched = [&] {
    ml_cpu_compute::IntegerMatMul2D(a.data(), false, b.data(), true, c.data(), m, n, k, &az, 1, &bz,
                                    1);
  };
  packed();
  dispatched();
  if (c != expected) {
    std::fprintf(stderr, "Dispatched output differs from packed output.\n");
    return 1;
  }
#if defined(ML_CPU_COMPUTE_HAVE_AVX2_INTEGER) && defined(ML_CPU_COMPUTE_HAVE_AVXVNNI)
  if (ml_cpu_compute::DetectSimdLevel() >= ml_cpu_compute::SimdLevel::kAVX2 &&
      ml_cpu_compute::IntegerMatMul2DUsesAvxVnni()) {
    const ml_cpu_compute::detail::IntegerVnniDotFn avx2_dot =
        k % 32 == 0 ? &ml_cpu_compute::detail::IntegerDotU8S8Avx2
                    : &ml_cpu_compute::detail::IntegerDotU8S8Avx2ShortTail;
    const auto avx2 = [&] {
      ml_cpu_compute::detail::IntegerMatMul2DWithDot(avx2_dot, a.data(), false, b.data(), true,
                                                     c.data(), m, n, k, &az, 1, &bz, 1);
    };
    const auto avxvnni = [&] {
      ml_cpu_compute::detail::IntegerMatMul2DWithDot(&ml_cpu_compute::detail::IntegerDotU8S8AvxVnni,
                                                     a.data(), false, b.data(), true, c.data(), m,
                                                     n, k, &az, 1, &bz, 1);
    };
    avx2();
    if (c != expected) {
      std::fprintf(stderr, "AVX2 output differs from packed output.\n");
      return 1;
    }
    avxvnni();
    if (c != expected) {
      std::fprintf(stderr, "AVX-VNNI output differs from packed output.\n");
      return 1;
    }
    const auto times = CompareSeconds(avx2, avxvnni);
    std::printf("avx2_packed_seconds=%.9f avxvnni_packed_seconds=%.9f speedup=%.3fx\n", times[0],
                times[1], times[0] / times[1]);
  }
#endif
  const double pack_seconds = MedianSeconds(pack);
  const double dot_seconds = MedianSeconds([&] {
    for (std::int64_t row = 0; row < m; ++row) {
      for (std::int64_t col = 0; col < n; ++col) {
        c[row * n + col] = dot(a.data() + row * k, packed_b.data() + col * k, k);
      }
    }
  });
  const double packed_seconds = MedianSeconds(packed);
  const double dispatched_seconds = MedianSeconds(dispatched);
  std::printf("M=%lld N=%lld K=%lld dtype=u8s8 az=%d bz=%d threads=1 dot_isa=%s\n",
              static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k), az,
              bz, isa);
  std::printf(
      "plan=%s packed_b_bytes=%lld\n",
      ml_cpu_compute::IntegerMatMulPlanName(ml_cpu_compute::SelectIntegerMatMulPlan(m, n, k)),
      static_cast<long long>(ml_cpu_compute::SelectIntegerMatMulPlan(m, n, k) ==
                                     ml_cpu_compute::IntegerMatMulPlan::kPacked
                                 ? n * k
                                 : 0));
  std::printf("pack_b_seconds=%.9f dot_seconds=%.9f packed_total_seconds=%.9f "
              "dispatched_seconds=%.9f\n",
              pack_seconds, dot_seconds, packed_seconds, dispatched_seconds);
  return 0;
}
