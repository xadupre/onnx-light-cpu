// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace onnx_light_cpu {

// Direct references make missing ISA-specific implementations a link error,
// even when the tests run on a machine without AVX-512.
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
void AbsFloat64_AVX512(const double *input, double *output, std::size_t count);
void AbsFloat16_AVX512(const uint16_t *input, uint16_t *output, std::size_t count);
void AbsInt32_AVX512(const int32_t *input, int32_t *output, std::size_t count);
void AbsInt64_AVX512(const int64_t *input, int64_t *output, std::size_t count);
#endif
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512BW
void AbsInt8_AVX512(const int8_t *input, int8_t *output, std::size_t count);
void AbsInt16_AVX512(const int16_t *input, int16_t *output, std::size_t count);
#endif

} // namespace onnx_light_cpu

namespace {

using namespace onnx_light_cpu;

template <typename T> using AbsFn = void (*)(const T *, T *, std::size_t);

template <typename T>
void CheckAbs(AbsFn<T> dispatch, AbsFn<T> direct, const std::vector<T> &samples,
              const std::vector<T> &expected_samples) {
  ASSERT_EQ(samples.size(), expected_samples.size());
  constexpr std::size_t lanes = 64 / sizeof(T);
  for (std::size_t count = 0; count <= 2 * lanes + 1; ++count) {
    SCOPED_TRACE(count);
    for (std::size_t offset : {0, 1}) {
      SCOPED_TRACE(offset);
      const T sentinel = static_cast<T>(42);
      std::vector<T> input(count + offset + 1, sentinel);
      std::vector<T> expected(input.size(), sentinel);
      for (std::size_t i = 0; i < count; ++i) {
        input[offset + i] = samples[i % samples.size()];
        expected[offset + i] = expected_samples[i % samples.size()];
      }
      const auto original = input;
      for (AbsFn<T> kernel : {dispatch, direct}) {
        if (kernel == nullptr) {
          continue;
        }
        std::vector<T> output(input.size(), sentinel);
        kernel(input.data() + offset, output.data() + offset, count);
        EXPECT_EQ(std::memcmp(output.data(), expected.data(), output.size() * sizeof(T)), 0);
        EXPECT_EQ(std::memcmp(input.data(), original.data(), input.size() * sizeof(T)), 0);

        auto inplace = input;
        kernel(inplace.data() + offset, inplace.data() + offset, count);
        EXPECT_EQ(std::memcmp(inplace.data(), expected.data(), inplace.size() * sizeof(T)), 0);
      }
    }
  }
}

template <typename T> void CheckFloatingAbs(AbsFn<T> dispatch, AbsFn<T> direct) {
  const std::vector<T> samples = {-T{0},
                                  T{0},
                                  -T{1.5},
                                  T{2.5},
                                  -std::numeric_limits<T>::infinity(),
                                  std::numeric_limits<T>::infinity(),
                                  -std::numeric_limits<T>::quiet_NaN(),
                                  -std::numeric_limits<T>::denorm_min(),
                                  -std::numeric_limits<T>::max()};
  auto expected = samples;
  for (auto &value : expected) {
    value = std::fabs(value);
  }
  CheckAbs(dispatch, direct, samples, expected);
}

template <typename T> void CheckIntegerAbs(AbsFn<T> dispatch, AbsFn<T> direct) {
  constexpr T largest = std::numeric_limits<T>::max();
  CheckAbs<T>(dispatch, direct, {-largest, -1, 0, 1, largest}, {largest, 1, 0, 1, largest});
}

struct AbsCase {
  const char *name;
  void (*check)(bool direct);
  bool compiled;
  bool requires_bw;
};

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
#define ABS_AVX512_FUNCTION(name) &name
constexpr bool kAvx512Compiled = true;
#else
#define ABS_AVX512_FUNCTION(name) nullptr
constexpr bool kAvx512Compiled = false;
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512BW
#define ABS_AVX512BW_FUNCTION(name) &name
constexpr bool kAvx512BWCompiled = true;
#else
#define ABS_AVX512BW_FUNCTION(name) nullptr
constexpr bool kAvx512BWCompiled = false;
#endif

const AbsCase kCases[] = {
    {"Float32",
     [](bool direct) {
       CheckFloatingAbs<float>(AbsFloat32,
                               direct ? ABS_AVX512_FUNCTION(AbsFloat32_AVX512) : nullptr);
     },
     kAvx512Compiled, false},
    {"Float32Streaming",
     [](bool direct) {
       CheckFloatingAbs<float>(
           [](const float *input, float *output, std::size_t count) {
             auto tuning = DefaultAbsFloat32ExecutionTuning();
             tuning.streaming_store_threshold_bytes = 1;
             AbsFloat32WithTuning(input, output, count, tuning);
           },
           direct ? ABS_AVX512_FUNCTION(AbsFloat32_AVX512Streaming) : nullptr);
     },
     kAvx512Compiled, false},
    {"Float64",
     [](bool direct) {
       CheckFloatingAbs<double>(AbsFloat64,
                                direct ? ABS_AVX512_FUNCTION(AbsFloat64_AVX512) : nullptr);
     },
     kAvx512Compiled, false},
    {"Float16",
     [](bool direct) {
       CheckAbs<uint16_t>(AbsFloat16, direct ? ABS_AVX512_FUNCTION(AbsFloat16_AVX512) : nullptr,
                          {0x8000, 0x0000, 0xbc00, 0x3c00, 0xfc00, 0x7c00, 0xfe01, 0x8001, 0xfbff},
                          {0x0000, 0x0000, 0x3c00, 0x3c00, 0x7c00, 0x7c00, 0x7e01, 0x0001, 0x7bff});
     },
     kAvx512Compiled, false},
    {"BFloat16",
     [](bool direct) {
       CheckAbs<uint16_t>(AbsFloat16, direct ? ABS_AVX512_FUNCTION(AbsFloat16_AVX512) : nullptr,
                          {0x8000, 0x0000, 0xbf80, 0x3f80, 0xff80, 0x7f80, 0xffc1, 0x8001, 0xff7f},
                          {0x0000, 0x0000, 0x3f80, 0x3f80, 0x7f80, 0x7f80, 0x7fc1, 0x0001, 0x7f7f});
     },
     kAvx512Compiled, false},
    {"Int8",
     [](bool direct) {
       CheckIntegerAbs<int8_t>(AbsInt8, direct ? ABS_AVX512BW_FUNCTION(AbsInt8_AVX512) : nullptr);
     },
     kAvx512BWCompiled, true},
    {"Int16",
     [](bool direct) {
       CheckIntegerAbs<int16_t>(AbsInt16,
                                direct ? ABS_AVX512BW_FUNCTION(AbsInt16_AVX512) : nullptr);
     },
     kAvx512BWCompiled, true},
    {"Int32",
     [](bool direct) {
       CheckIntegerAbs<int32_t>(AbsInt32, direct ? ABS_AVX512_FUNCTION(AbsInt32_AVX512) : nullptr);
     },
     kAvx512Compiled, false},
    {"Int64",
     [](bool direct) {
       CheckIntegerAbs<int64_t>(AbsInt64, direct ? ABS_AVX512_FUNCTION(AbsInt64_AVX512) : nullptr);
     },
     kAvx512Compiled, false},
};

#undef ABS_AVX512_FUNCTION
#undef ABS_AVX512BW_FUNCTION

class AbsAvx512 : public testing::TestWithParam<AbsCase> {};

TEST_P(AbsAvx512, DispatchNumerics) { GetParam().check(false); }

TEST_P(AbsAvx512, DirectAndDispatchNumerics) {
  const auto &test = GetParam();
  if (!test.compiled) {
    GTEST_SKIP() << "AVX-512 implementation not compiled";
  }
  if (DetectSimdLevel() < SimdLevel::kAVX512 || (test.requires_bw && !CpuSupportsAvx512BW())) {
    GTEST_SKIP() << "Required AVX-512 ISA not supported at runtime";
  }
  test.check(true);
}

INSTANTIATE_TEST_SUITE_P(AllTypes, AbsAvx512, testing::ValuesIn(kCases),
                         [](const testing::TestParamInfo<AbsCase> &info) {
                           return info.param.name;
                         });

} // namespace
