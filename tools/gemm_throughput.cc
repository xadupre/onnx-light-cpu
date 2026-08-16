// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Isolated FP32/FP64/FP16/BF16 GEMM throughput driver.
//
// This tool measures the throughput of the prepared :cpp:class:`GemmPlan`
// engine in isolation, without ONNX Runtime or onnx-light. It answers the
// "isolated C++ driver" question of the Gemm/MatMul roadmap benchmark contract:
// how many GFLOP/s each priority shape sustains on the current machine,
// separately from the operator dispatch and reference-evaluator overhead that
// the Python parity runner (tools/benchmark_gemm_parity.py) also includes.
//
// It is intentionally not a correctness gate and is not run by ctest. Build it
// with ``-DONNX_LIGHT_CPU_BUILD_BENCHMARKS=ON`` and run the resulting
// ``gemm_throughput`` executable. The thread count honours the same
// ``ONNX_LIGHT_CPU_NUM_THREADS`` environment variable as the kernels.

#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"
#include "onnx_light_cpu/impl/math/gemm/gemm_plan.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using onnx_light_cpu::GemmPlan;
using onnx_light_cpu::GemmPlanOptions;

struct GemmCase {
  const char *name;
  std::size_t m;
  std::size_t n;
  std::size_t k;
  bool trans_a = false;
  bool trans_b = false;
  bool constant_b = false;
};

// Mirrors the priority shapes covered by tools/benchmark_gemm_parity.py, with a
// few extra square sizes so the large-matrix sustain from 512 to 2048 is
// visible at a glance.
constexpr GemmCase kCases[] = {
    {"square_128", 128, 128, 128},
    {"square_256", 256, 256, 256},
    {"square_512", 512, 512, 512},
    {"square_1024", 1024, 1024, 1024},
    {"square_2048", 2048, 2048, 2048},
    {"skinny_m_gemv", 1, 1024, 1024},
    {"skinny_n", 1024, 1, 1024},
    {"large_k", 32, 32, 4096},
    {"trans_a_128", 128, 128, 128, true, false, false},
    {"trans_b_128", 128, 128, 128, false, true, false},
    {"transformer_proj", 128, 3072, 768, false, false, true},
};

std::size_t RepeatCount(const GemmCase &c) {
  const std::uint64_t operations = std::uint64_t{2} * c.m * c.n * c.k;
  const std::uint64_t target = 400'000'000ull;
  return static_cast<std::size_t>(std::max<std::uint64_t>(
      5, std::min<std::uint64_t>(51, target / std::max<std::uint64_t>(1, operations))));
}

template <typename T> double MeasureGflops(const GemmCase &c, std::size_t *useful_threads) {
  std::mt19937 rng(0x5eed5eedu);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<T> a(c.m * c.k);
  std::vector<T> b(c.k * c.n);
  std::vector<T> y(c.m * c.n);
  for (T &value : a) {
    value = static_cast<T>(dist(rng));
  }
  for (T &value : b) {
    value = static_cast<T>(dist(rng));
  }

  GemmPlanOptions<T> options;
  options.trans_a = c.trans_a;
  options.trans_b = c.trans_b;
  options.m = c.m;
  options.n = c.n;
  options.k = c.k;
  options.alpha = T(1);
  options.beta = T(0);
  if (c.constant_b) {
    options.constant_b = std::span<const T>(b.data(), b.size());
  }
  const GemmPlan<T> plan(options);
  *useful_threads = plan.useful_threads();

  const auto run = [&]() {
    if (c.constant_b) {
      plan.Execute(a.data(), nullptr, y.data());
    } else {
      plan.Execute(a.data(), b.data(), nullptr, y.data());
    }
  };

  for (int warmup = 0; warmup < 3; ++warmup) {
    run();
  }

  const std::size_t repeat = RepeatCount(c);
  std::vector<double> seconds;
  seconds.reserve(repeat);
  for (std::size_t iteration = 0; iteration < repeat; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    run();
    const auto stop = std::chrono::steady_clock::now();
    seconds.push_back(std::chrono::duration<double>(stop - start).count());
  }
  std::sort(seconds.begin(), seconds.end());
  const double median = seconds[seconds.size() / 2];
  const double operations =
      2.0 * static_cast<double>(c.m) * static_cast<double>(c.n) * static_cast<double>(c.k);
  return operations / median / 1e9;
}

// Isolated throughput of the FP16/BF16 convert-while-packing path
// (:cpp:func:`GemmHalfWithEpilogue`). Inputs are raw half-precision bit
// patterns derived from random float32 values; the reduction accumulates in
// float32 and the epilogue narrows the result back to the half format, so the
// figure includes the per-element input conversion done during packing.
template <bool Bfloat16> double MeasureHalfGflops(const GemmCase &c) {
  std::mt19937 rng(0x5eed5eedu);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const auto narrow = [](float value) {
    return Bfloat16 ? onnx_light_cpu::detail::FloatToBFloat16Bits(value)
                    : onnx_light_cpu::detail::FloatToFloat16Bits(value);
  };
  std::vector<std::uint16_t> a(c.m * c.k);
  std::vector<std::uint16_t> b(c.k * c.n);
  std::vector<std::uint16_t> out(c.m * c.n);
  std::vector<float> workspace(c.m * c.n);
  for (std::uint16_t &value : a) {
    value = narrow(dist(rng));
  }
  for (std::uint16_t &value : b) {
    value = narrow(dist(rng));
  }

  onnx_light_cpu::GemmEpilogue<float> epilogue;
  epilogue.output_conversion = Bfloat16 ? onnx_light_cpu::GemmOutputConversion::kBFloat16
                                        : onnx_light_cpu::GemmOutputConversion::kFloat16;
  epilogue.converted_output = out.data();

  const auto run = [&]() {
    onnx_light_cpu::GemmHalfWithEpilogue(Bfloat16, c.trans_a, c.trans_b, c.m, c.n, c.k, 1.0f,
                                         a.data(), b.data(), epilogue, workspace.data());
  };

  for (int warmup = 0; warmup < 3; ++warmup) {
    run();
  }

  const std::size_t repeat = RepeatCount(c);
  std::vector<double> seconds;
  seconds.reserve(repeat);
  for (std::size_t iteration = 0; iteration < repeat; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    run();
    const auto stop = std::chrono::steady_clock::now();
    seconds.push_back(std::chrono::duration<double>(stop - start).count());
  }
  std::sort(seconds.begin(), seconds.end());
  const double median = seconds[seconds.size() / 2];
  const double operations =
      2.0 * static_cast<double>(c.m) * static_cast<double>(c.n) * static_cast<double>(c.k);
  return operations / median / 1e9;
}

const char *SimdName(onnx_light_cpu::SimdLevel level) {
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

const char *MicroarchitectureName(onnx_light_cpu::GemmMicroarchitecture microarchitecture) {
  switch (microarchitecture) {
  case onnx_light_cpu::GemmMicroarchitecture::kGeneric:
    return "generic";
  case onnx_light_cpu::GemmMicroarchitecture::kIntelCore:
    return "intel-core";
  case onnx_light_cpu::GemmMicroarchitecture::kAmdZen:
    return "amd-zen";
  }
  return "unknown";
}

} // namespace

int main() {
  const onnx_light_cpu::SimdLevel level = onnx_light_cpu::DetectSimdLevel();
  const bool has_fma = onnx_light_cpu::CpuSupportsFma();
  const onnx_light_cpu::GemmMicroarchitecture microarchitecture =
      onnx_light_cpu::detail::DetectGemmMicroarchitecture();
  const std::size_t register_rows =
      onnx_light_cpu::detail::SelectGemmRegisterRowsForMicroarchitecture(level, has_fma,
                                                                         microarchitecture);

  std::printf("SIMD level: %s  FMA: %s  microarchitecture: %s  register rows: %zu\n",
              SimdName(level), has_fma ? "yes" : "no", MicroarchitectureName(microarchitecture),
              register_rows);
  std::printf("%-18s %6s %6s %6s %8s %14s %8s %14s %14s %14s\n", "case", "M", "N", "K", "fp32 thr",
              "fp32 GFLOP/s", "fp64 thr", "fp64 GFLOP/s", "fp16 GFLOP/s", "bf16 GFLOP/s");

  for (const GemmCase &c : kCases) {
    std::size_t fp32_threads = 0;
    std::size_t fp64_threads = 0;
    const double fp32 = MeasureGflops<float>(c, &fp32_threads);
    const double fp64 = MeasureGflops<double>(c, &fp64_threads);
    const double fp16 = MeasureHalfGflops<false>(c);
    const double bf16 = MeasureHalfGflops<true>(c);
    std::printf("%-18s %6zu %6zu %6zu %8zu %14.2f %8zu %14.2f %14.2f %14.2f\n", c.name, c.m, c.n,
                c.k, fp32_threads, fp32, fp64_threads, fp64, fp16, bf16);
  }
  return 0;
}
