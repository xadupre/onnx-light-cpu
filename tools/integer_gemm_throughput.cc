// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_vnni.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <algorithm>
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
  using namespace onnx_light_cpu;
  detail::IntegerVnniDotFn dot = &detail::IntegerDotU8S8Scalar;
  const char *isa = "scalar";
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER
  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
    dot = &detail::IntegerDotU8S8Avx2;
    isa = "avx2";
  }
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512VNNI
  if (IntegerMatMul2DUsesVnni()) {
    dot = &detail::IntegerDotU8S8Avx512Vnni;
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
    detail::PackS8ColRange(b.data(), true, 0, k, n, packed_b.data(), sums.data(), 0, n);
  };
  const auto packed = [&] {
    detail::IntegerMatMul2DWithDot(dot, a.data(), false, b.data(), true, expected.data(), m, n, k,
                                   &az, 1, &bz, 1);
  };
  const auto dispatched = [&] {
    IntegerMatMul2D(a.data(), false, b.data(), true, c.data(), m, n, k, &az, 1, &bz, 1);
  };
  packed();
  dispatched();
  if (c != expected) {
    std::fprintf(stderr, "Dispatched output differs from packed output.\n");
    return 1;
  }
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
  std::printf("plan=%s packed_b_bytes=%lld\n",
              IntegerMatMulPlanName(SelectIntegerMatMulPlan(m, n, k)),
              static_cast<long long>(
                  SelectIntegerMatMulPlan(m, n, k) == IntegerMatMulPlan::kPacked ? n * k : 0));
  std::printf("pack_b_seconds=%.9f dot_seconds=%.9f packed_total_seconds=%.9f "
              "dispatched_seconds=%.9f\n",
              pack_seconds, dot_seconds, packed_seconds, dispatched_seconds);
  return 0;
}
