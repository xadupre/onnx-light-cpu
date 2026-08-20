// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Isolated FP16/BF16/INT8/INT4/Float8 GEMM throughput driver.

#include "onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_vnni.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

struct GemmCase {
  const char *name;
  std::int64_t m;
  std::int64_t n;
  std::int64_t k;
};

constexpr GemmCase kCases[] = {
    {"tiny", 1, 64, 64},           {"direct", 32, 128, 16},         {"square_128", 128, 128, 128},
    {"square_512", 512, 512, 512}, {"skinny_m", 1, 1024, 1024},     {"skinny_n", 1024, 1, 1024},
    {"large_k", 32, 32, 4096},     {"transformer", 128, 3072, 768},
};

std::size_t RepeatCount(const GemmCase &shape) {
  const std::uint64_t operations = std::uint64_t{2} * static_cast<std::uint64_t>(shape.m) *
                                   static_cast<std::uint64_t>(shape.n) *
                                   static_cast<std::uint64_t>(shape.k);
  return static_cast<std::size_t>(
      std::max<std::uint64_t>(5, std::min<std::uint64_t>(101, 200'000'000ull / operations)));
}

template <typename Fn> double MeasureGops(const GemmCase &shape, Fn run) {
  for (int warmup = 0; warmup < 3; ++warmup) {
    run();
  }
  std::vector<double> seconds;
  const std::size_t repeat = RepeatCount(shape);
  seconds.reserve(repeat);
  for (std::size_t iteration = 0; iteration < repeat; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    run();
    const auto stop = std::chrono::steady_clock::now();
    seconds.push_back(std::chrono::duration<double>(stop - start).count());
  }
  std::sort(seconds.begin(), seconds.end());
  const double operations = 2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.n) *
                            static_cast<double>(shape.k);
  return operations / seconds[seconds.size() / 2] / 1e9;
}

double MeasureInt8(const GemmCase &shape) {
  std::mt19937 rng(0x104u);
  std::uniform_int_distribution<int> byte(0, 255);
  std::vector<std::uint8_t> a(static_cast<std::size_t>(shape.m * shape.k));
  std::vector<std::uint8_t> b(static_cast<std::size_t>(shape.k * shape.n));
  std::vector<std::int32_t> c(static_cast<std::size_t>(shape.m * shape.n));
  std::generate(a.begin(), a.end(), [&]() { return static_cast<std::uint8_t>(byte(rng)); });
  std::generate(b.begin(), b.end(), [&]() { return static_cast<std::uint8_t>(byte(rng)); });
  const std::int32_t a_zero_point = 128;
  const std::int32_t b_zero_point = 0;
  return MeasureGops(shape, [&]() {
    onnx_light_cpu::IntegerMatMul2D(a.data(), false, b.data(), true, c.data(), shape.m, shape.n,
                                    shape.k, &a_zero_point, 1, &b_zero_point, 1);
  });
}

double MeasureInt4(const GemmCase &shape) {
  std::mt19937 rng(0x404u);
  std::uniform_int_distribution<int> byte(0, 255);
  const std::size_t a_count = static_cast<std::size_t>(shape.m * shape.k);
  const std::size_t b_count = static_cast<std::size_t>(shape.k * shape.n);
  std::vector<std::uint8_t> a((a_count + 1) / 2);
  std::vector<std::uint8_t> b((b_count + 1) / 2);
  std::vector<std::int32_t> c(static_cast<std::size_t>(shape.m * shape.n));
  std::generate(a.begin(), a.end(), [&]() { return static_cast<std::uint8_t>(byte(rng)); });
  std::generate(b.begin(), b.end(), [&]() { return static_cast<std::uint8_t>(byte(rng)); });
  const std::int32_t zero_point = 0;
  return MeasureGops(shape, [&]() {
    onnx_light_cpu::IntegerMatMul4Bit2D(a.data(), true, b.data(), true, c.data(), shape.m, shape.n,
                                        shape.k, &zero_point, 1, &zero_point, 1);
  });
}

double MeasureHalf(const GemmCase &shape, bool is_bfloat16) {
  std::mt19937 rng(is_bfloat16 ? 0xbf16u : 0xf16u);
  std::uniform_real_distribution<float> value(-1.0f, 1.0f);
  std::vector<std::uint16_t> a(static_cast<std::size_t>(shape.m * shape.k));
  std::vector<std::uint16_t> b(static_cast<std::size_t>(shape.k * shape.n));
  std::vector<float> y(static_cast<std::size_t>(shape.m * shape.n));
  const auto narrow = [&](std::uint16_t &bits) {
    const float input = value(rng);
    bits = is_bfloat16 ? onnx_light_cpu::detail::FloatToBFloat16Bits(input)
                       : onnx_light_cpu::detail::FloatToFloat16Bits(input);
  };
  std::for_each(a.begin(), a.end(), narrow);
  std::for_each(b.begin(), b.end(), narrow);
  onnx_light_cpu::GemmEpilogue<float> epilogue;
  return MeasureGops(shape, [&]() {
    onnx_light_cpu::GemmHalfWithEpilogue(
        is_bfloat16, false, false, static_cast<std::size_t>(shape.m),
        static_cast<std::size_t>(shape.n), static_cast<std::size_t>(shape.k), 1.0f, a.data(),
        b.data(), epilogue, y.data());
  });
}

double MeasureFloat8(const GemmCase &shape, onnx_light_cpu::GemmFloat8Format format) {
  std::mt19937 rng(0xf8u);
  std::uniform_int_distribution<int> finite_e4m3(0, 0x7e);
  std::vector<std::uint8_t> a(static_cast<std::size_t>(shape.m * shape.k));
  std::vector<std::uint8_t> b(static_cast<std::size_t>(shape.k * shape.n));
  std::vector<float> y(static_cast<std::size_t>(shape.m * shape.n));
  std::generate(a.begin(), a.end(), [&]() { return static_cast<std::uint8_t>(finite_e4m3(rng)); });
  std::generate(b.begin(), b.end(), [&]() { return static_cast<std::uint8_t>(finite_e4m3(rng)); });
  onnx_light_cpu::GemmEpilogue<float> epilogue;
  return MeasureGops(shape, [&]() {
    onnx_light_cpu::GemmFloat8WithEpilogue(
        format, false, false, static_cast<std::size_t>(shape.m), static_cast<std::size_t>(shape.n),
        static_cast<std::size_t>(shape.k), 1.0f, a.data(), b.data(), epilogue, y.data());
  });
}

} // namespace

int main() {
  std::printf("%-14s %6s %6s %6s %13s %13s %13s %13s %13s %13s\n", "case", "M", "N", "K",
              "FP16 GFLOPS", "BF16 GFLOPS", "INT8 GOPS", "INT4 GOPS", "E4M3 GOPS", "E5M2 GOPS");
  for (const GemmCase &shape : kCases) {
    const double fp16 = MeasureHalf(shape, false);
    const double bf16 = MeasureHalf(shape, true);
    const double int8 = MeasureInt8(shape);
    const double int4 = MeasureInt4(shape);
    const double e4m3 = MeasureFloat8(shape, onnx_light_cpu::GemmFloat8Format::kE4M3FN);
    const double e5m2 = MeasureFloat8(shape, onnx_light_cpu::GemmFloat8Format::kE5M2);
    std::printf("%-14s %6lld %6lld %6lld %13.2f %13.2f %13.2f %13.2f %13.2f %13.2f\n", shape.name,
                static_cast<long long>(shape.m), static_cast<long long>(shape.n),
                static_cast<long long>(shape.k), fp16, bf16, int8, int4, e4m3, e5m2);
  }
  return 0;
}
