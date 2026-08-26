// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Isolated FP16/BF16/INT8/INT4/Float8 GEMM throughput driver.
//
// FP16/BF16 measurements go through the tunable ``GemmHalfPlan`` so a
// dedicated-machine sweep can select ``MC``/``NC``/``KC`` and the maximum
// executor participant count through the same tuning knobs exposed to the
// ``Gemm``/``MatMul`` kernel (Roadmap PR07/#332), instead of hard-coding
// machine-specific thresholds. Pass ``--json <path>`` to publish every raw
// sample alongside hardware, affinity, compiler, and ISA metadata.

#include "onnx_light_cpu/impl/math/gemm/gemm_plan.h"
#include "onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_vnni.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/impl/simd_level.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <sys/utsname.h>
#include <thread>
#endif

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

/// User-selected FP16/BF16 tuning, mirroring the ``GemmKernel`` tuning
/// schema (``blocking.*``, ``compact_blocking.*``, ``parallel.maximum_threads``
/// from #332). Every field defaults to ``0``, meaning automatic and matching
/// the untuned behavior.
struct HalfTuning {
  onnx_light_cpu::GemmBlocking blocking;
  onnx_light_cpu::GemmBlocking compact_blocking;
  std::size_t maximum_participants = 0;
};

std::size_t RepeatCount(const GemmCase &shape) {
  const std::uint64_t operations = std::uint64_t{2} * static_cast<std::uint64_t>(shape.m) *
                                   static_cast<std::uint64_t>(shape.n) *
                                   static_cast<std::uint64_t>(shape.k);
  return static_cast<std::size_t>(
      std::max<std::uint64_t>(5, std::min<std::uint64_t>(101, 200'000'000ull / operations)));
}

template <typename Fn> std::vector<double> MeasureSeconds(const GemmCase &shape, Fn run) {
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
  return seconds;
}

double GopsFromMedian(const GemmCase &shape, const std::vector<double> &seconds) {
  const double operations = 2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.n) *
                            static_cast<double>(shape.k);
  return operations / seconds[seconds.size() / 2] / 1e9;
}

template <typename Fn> double MeasureGops(const GemmCase &shape, Fn run) {
  return GopsFromMedian(shape, MeasureSeconds(shape, run));
}

std::vector<double> MeasureInt8Seconds(const GemmCase &shape) {
  std::mt19937 rng(0x104u);
  std::uniform_int_distribution<int> byte(0, 255);
  std::vector<std::uint8_t> a(static_cast<std::size_t>(shape.m * shape.k));
  std::vector<std::uint8_t> b(static_cast<std::size_t>(shape.k * shape.n));
  std::vector<std::int32_t> c(static_cast<std::size_t>(shape.m * shape.n));
  std::generate(a.begin(), a.end(), [&]() { return static_cast<std::uint8_t>(byte(rng)); });
  std::generate(b.begin(), b.end(), [&]() { return static_cast<std::uint8_t>(byte(rng)); });
  const std::int32_t a_zero_point = 128;
  const std::int32_t b_zero_point = 0;
  return MeasureSeconds(shape, [&]() {
    onnx_light_cpu::IntegerMatMul2D(a.data(), false, b.data(), true, c.data(), shape.m, shape.n,
                                    shape.k, &a_zero_point, 1, &b_zero_point, 1);
  });
}

std::vector<double> MeasureInt4Seconds(const GemmCase &shape) {
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
  return MeasureSeconds(shape, [&]() {
    onnx_light_cpu::IntegerMatMul4Bit2D(a.data(), true, b.data(), true, c.data(), shape.m, shape.n,
                                        shape.k, &zero_point, 1, &zero_point, 1);
  });
}

std::vector<double> MeasureHalfSeconds(const GemmCase &shape, bool is_bfloat16,
                                       const HalfTuning &tuning) {
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

  onnx_light_cpu::GemmHalfPlanOptions options;
  options.is_bfloat16 = is_bfloat16;
  options.m = static_cast<std::size_t>(shape.m);
  options.n = static_cast<std::size_t>(shape.n);
  options.k = static_cast<std::size_t>(shape.k);
  options.alpha = 1.0f;
  options.blocking = tuning.blocking;
  options.compact_blocking = tuning.compact_blocking;
  options.maximum_participants = tuning.maximum_participants;
  const onnx_light_cpu::GemmHalfPlan plan(options);

  return MeasureSeconds(shape, [&]() { plan.Execute(a.data(), b.data(), epilogue, y.data()); });
}

std::vector<double> MeasureFloat8Seconds(const GemmCase &shape,
                                         onnx_light_cpu::GemmFloat8Format format) {
  std::mt19937 rng(0xf8u);
  std::uniform_int_distribution<int> finite_e4m3(0, 0x7e);
  std::vector<std::uint8_t> a(static_cast<std::size_t>(shape.m * shape.k));
  std::vector<std::uint8_t> b(static_cast<std::size_t>(shape.k * shape.n));
  std::vector<float> y(static_cast<std::size_t>(shape.m * shape.n));
  std::generate(a.begin(), a.end(), [&]() { return static_cast<std::uint8_t>(finite_e4m3(rng)); });
  std::generate(b.begin(), b.end(), [&]() { return static_cast<std::uint8_t>(finite_e4m3(rng)); });
  onnx_light_cpu::GemmEpilogue<float> epilogue;
  return MeasureSeconds(shape, [&]() {
    onnx_light_cpu::GemmFloat8WithEpilogue(
        format, false, false, static_cast<std::size_t>(shape.m), static_cast<std::size_t>(shape.n),
        static_cast<std::size_t>(shape.k), 1.0f, a.data(), b.data(), epilogue, y.data());
  });
}

std::string SimdLevelName(onnx_light_cpu::SimdLevel level) {
  switch (level) {
  case onnx_light_cpu::SimdLevel::kNone:
    return "none";
  case onnx_light_cpu::SimdLevel::kSSE2:
    return "sse2";
  case onnx_light_cpu::SimdLevel::kAVX:
    return "avx";
  case onnx_light_cpu::SimdLevel::kAVX2:
    return "avx2";
  case onnx_light_cpu::SimdLevel::kAVX512:
    return "avx512";
  }
  return "unknown";
}

std::string CpuModel() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (cpuinfo && std::getline(cpuinfo, line)) {
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    std::string key = line.substr(0, colon);
    while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) {
      key.pop_back();
    }
    if (key == "model name" || key == "Model") {
      std::string value = line.substr(colon + 1);
      std::size_t begin = value.find_first_not_of(" \t");
      return begin == std::string::npos ? "unknown" : value.substr(begin);
    }
  }
  return "unknown";
}

std::string AffinityList() {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) {
    return "unknown";
  }
  std::ostringstream out;
  bool first = true;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &set)) {
      if (!first) {
        out << ",";
      }
      out << cpu;
      first = false;
    }
  }
  return out.str();
#else
  return "unknown";
#endif
}

std::string Compiler() {
#if defined(__clang__)
  return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
  return std::string("gcc ") + __VERSION__;
#else
  return "unknown";
#endif
}

std::string Platform() {
#if defined(__linux__)
  struct utsname name{};
  if (uname(&name) == 0) {
    return std::string(name.sysname) + " " + name.release + " " + name.machine;
  }
#endif
  return "unknown";
}

std::string TimestampUtc() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm utc_tm{};
#if defined(__linux__)
  gmtime_r(&now_time, &utc_tm);
#else
  utc_tm = *std::gmtime(&now_time);
#endif
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc_tm);
  return buffer;
}

unsigned int LogicalCpus() {
#if defined(__linux__)
  return std::thread::hardware_concurrency();
#else
  return 0;
#endif
}

/// Escapes ``value`` for embedding in a double-quoted JSON string literal.
std::string JsonEscape(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char c : value) {
    if (c == '"' || c == '\\') {
      escaped.push_back('\\');
    }
    escaped.push_back(c);
  }
  return escaped;
}

std::string JsonSecondsArray(const std::vector<double> &seconds) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < seconds.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << seconds[i];
  }
  out << ']';
  return out.str();
}

std::string JsonBlocking(const onnx_light_cpu::GemmBlocking &blocking) {
  std::ostringstream out;
  out << "{\"mc\": " << blocking.mc << ", \"nc\": " << blocking.nc << ", \"kc\": " << blocking.kc
      << "}";
  return out.str();
}

/// Command-line configuration for the isolated FP16/BF16 sweep. Every
/// tunable defaults to ``0`` (automatic), matching the ``GemmKernel`` tuning
/// schema from #332 so a dedicated-machine run only overrides what it
/// measures a benefit from, per #333.
struct Options {
  HalfTuning tuning;
  std::string json_path;
};

Options ParseArgs(int argc, char **argv) {
  Options options;
  const auto require_value = [&](int &index) -> std::size_t {
    if (index + 1 >= argc) {
      std::fprintf(stderr, "Missing value for argument %s\n", argv[index]);
      std::exit(1);
    }
    const char *token = argv[++index];
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(token, &end, 10);
    if (end == token || *end != '\0') {
      std::fprintf(stderr, "Invalid numeric value %s for argument %s\n", token, argv[index - 1]);
      std::exit(1);
    }
    return static_cast<std::size_t>(parsed);
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--mc") {
      options.tuning.blocking.mc = require_value(i);
    } else if (arg == "--nc") {
      options.tuning.blocking.nc = require_value(i);
    } else if (arg == "--kc") {
      options.tuning.blocking.kc = require_value(i);
    } else if (arg == "--compact-mc") {
      options.tuning.compact_blocking.mc = require_value(i);
    } else if (arg == "--compact-nc") {
      options.tuning.compact_blocking.nc = require_value(i);
    } else if (arg == "--compact-kc") {
      options.tuning.compact_blocking.kc = require_value(i);
    } else if (arg == "--participants") {
      options.tuning.maximum_participants = require_value(i);
    } else if (arg == "--json") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for argument --json\n");
        std::exit(1);
      }
      options.json_path = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::printf("Usage: compact_gemm_throughput [--mc N] [--nc N] [--kc N] "
                  "[--compact-mc N] [--compact-nc N] [--compact-kc N] "
                  "[--participants N] [--json PATH]\n"
                  "Every tuning value defaults to 0 (automatic).\n");
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown argument %s\n", argv[i]);
      std::exit(1);
    }
  }
  return options;
}

} // namespace

int main(int argc, char **argv) {
  const Options options = ParseArgs(argc, argv);

  std::printf("%-14s %6s %6s %6s %13s %13s %13s %13s %13s %13s\n", "case", "M", "N", "K",
              "FP16 GFLOPS", "BF16 GFLOPS", "INT8 GOPS", "INT4 GOPS", "E4M3 GOPS", "E5M2 GOPS");

  std::vector<std::string> json_cases;
  for (const GemmCase &shape : kCases) {
    const std::vector<double> fp16_seconds = MeasureHalfSeconds(shape, false, options.tuning);
    const std::vector<double> bf16_seconds = MeasureHalfSeconds(shape, true, options.tuning);
    const double fp16 = GopsFromMedian(shape, fp16_seconds);
    const double bf16 = GopsFromMedian(shape, bf16_seconds);
    const std::vector<double> int8_seconds = MeasureInt8Seconds(shape);
    const std::vector<double> int4_seconds = MeasureInt4Seconds(shape);
    const std::vector<double> e4m3_seconds =
        MeasureFloat8Seconds(shape, onnx_light_cpu::GemmFloat8Format::kE4M3FN);
    const std::vector<double> e5m2_seconds =
        MeasureFloat8Seconds(shape, onnx_light_cpu::GemmFloat8Format::kE5M2);
    const double int8 = GopsFromMedian(shape, int8_seconds);
    const double int4 = GopsFromMedian(shape, int4_seconds);
    const double e4m3 = GopsFromMedian(shape, e4m3_seconds);
    const double e5m2 = GopsFromMedian(shape, e5m2_seconds);
    std::printf("%-14s %6lld %6lld %6lld %13.2f %13.2f %13.2f %13.2f %13.2f %13.2f\n", shape.name,
                static_cast<long long>(shape.m), static_cast<long long>(shape.n),
                static_cast<long long>(shape.k), fp16, bf16, int8, int4, e4m3, e5m2);

    if (!options.json_path.empty()) {
      std::ostringstream entry;
      entry << "    {\n"
            << "      \"name\": \"" << JsonEscape(shape.name) << "\",\n"
            << "      \"m\": " << shape.m << ",\n"
            << "      \"n\": " << shape.n << ",\n"
            << "      \"k\": " << shape.k << ",\n"
            << "      \"fp16_gflops_median\": " << fp16 << ",\n"
            << "      \"bf16_gflops_median\": " << bf16 << ",\n"
            << "      \"int8_gops_median\": " << int8 << ",\n"
            << "      \"int4_gops_median\": " << int4 << ",\n"
            << "      \"e4m3_gops_median\": " << e4m3 << ",\n"
            << "      \"e5m2_gops_median\": " << e5m2 << ",\n"
            << "      \"fp16_seconds_samples\": " << JsonSecondsArray(fp16_seconds) << ",\n"
            << "      \"bf16_seconds_samples\": " << JsonSecondsArray(bf16_seconds) << ",\n"
            << "      \"int8_seconds_samples\": " << JsonSecondsArray(int8_seconds) << ",\n"
            << "      \"int4_seconds_samples\": " << JsonSecondsArray(int4_seconds) << ",\n"
            << "      \"e4m3_seconds_samples\": " << JsonSecondsArray(e4m3_seconds) << ",\n"
            << "      \"e5m2_seconds_samples\": " << JsonSecondsArray(e5m2_seconds) << "\n"
            << "    }";
      json_cases.push_back(entry.str());
    }
  }

  if (!options.json_path.empty()) {
    std::ofstream out(options.json_path);
    if (!out) {
      std::fprintf(stderr, "Failed to open %s for writing.\n", options.json_path.c_str());
      return 1;
    }
    out << "{\n"
        << "  \"metadata\": {\n"
        << "    \"timestamp_utc\": \"" << TimestampUtc() << "\",\n"
        << "    \"platform\": \"" << JsonEscape(Platform()) << "\",\n"
        << "    \"cpu_model\": \"" << JsonEscape(CpuModel()) << "\",\n"
        << "    \"logical_cpus\": " << LogicalCpus() << ",\n"
        << "    \"affinity\": \"" << JsonEscape(AffinityList()) << "\",\n"
        << "    \"compiler\": \"" << JsonEscape(Compiler()) << "\",\n"
        << "    \"simd_level\": \"" << SimdLevelName(onnx_light_cpu::DetectSimdLevel()) << "\",\n"
        << "    \"requested_participants\": " << options.tuning.maximum_participants << ",\n"
        << "    \"blocking\": " << JsonBlocking(options.tuning.blocking) << ",\n"
        << "    \"compact_blocking\": " << JsonBlocking(options.tuning.compact_blocking) << "\n"
        << "  },\n"
        << "  \"cases\": [\n";
    for (std::size_t i = 0; i < json_cases.size(); ++i) {
      out << json_cases[i] << (i + 1 < json_cases.size() ? ",\n" : "\n");
    }
    out << "  ]\n"
        << "}\n";
    if (!out.good()) {
      std::fprintf(stderr, "Failed to write %s.\n", options.json_path.c_str());
      return 1;
    }
    std::printf("raw results: %s\n", options.json_path.c_str());
  }
  return 0;
}
