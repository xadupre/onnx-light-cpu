// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/attention/rotary_embedding.h"
#include "ml_cpu_compute/math/half_conversion.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

template <typename T> void CheckSplitHalfPrefix() {
  constexpr bool half_type = std::is_same_v<T, std::uint16_t>;
  const auto rotate = ml_cpu_compute::SelectRotaryEmbeddingSplitHalf(
      half_type ? ml_cpu_compute::DataType::FLOAT16 : ml_cpu_compute::DataType::FLOAT);
  if (rotate == nullptr) {
    GTEST_SKIP() << "No SIMD implementation for this type on this CPU/build.";
  }
  const auto encode = [](float value) -> T {
    if constexpr (half_type) {
      return ml_cpu_compute::detail::FloatToFloat16Bits(value);
    } else {
      return value;
    }
  };
  const auto decode = [](T value) -> float {
    if constexpr (half_type) {
      return ml_cpu_compute::detail::Float16BitsToFloat(value);
    } else {
      return value;
    }
  };
  const std::vector<float> values{0.0f,
                                  -0.0f,
                                  1.0f,
                                  -1.0f,
                                  0.333251953125f,
                                  65504.0f,
                                  0x1p-24f,
                                  -0x1p-14f,
                                  std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::quiet_NaN()};
  for (std::size_t half = 0; half <= 65; ++half) {
    SCOPED_TRACE(half);
    // Offset every buffer by one element to exercise unaligned vector accesses.
    std::vector<T> x(2 * half + 1), c(half + 1), s(half + 1);
    const T sentinel = encode(42.0f);
    std::vector<T> y(2 * half + 2, sentinel);
    for (std::size_t i = 0; i < 2 * half; ++i) {
      x[i + 1] = encode(values[i % values.size()]);
    }
    for (std::size_t i = 0; i < half; ++i) {
      c[i + 1] = encode(values[(i + 3) % values.size()]);
      s[i + 1] = encode(values[(i + 5) % values.size()]);
    }
    const auto processed = rotate(x.data() + 1, c.data() + 1, s.data() + 1, y.data() + 1, half);
    ASSERT_EQ(processed, half / 8 * 8);
    for (std::size_t i = 0; i < half; ++i) {
      for (const bool second : {false, true}) {
        const auto index = i + (second ? half : 0);
        if (i >= processed) {
          EXPECT_EQ(y[index + 1], sentinel);
          continue;
        }
        const float a = decode(x[i + 1]), b = decode(x[half + i + 1]);
        const float cv = decode(c[i + 1]), sv = decode(s[i + 1]);
        const T expected = encode(second ? sv * a + cv * b : cv * a - sv * b);
        if (std::isnan(decode(expected))) {
          EXPECT_TRUE(std::isnan(decode(y[index + 1])));
        } else if constexpr (half_type) {
          EXPECT_EQ(y[index + 1], expected);
        } else if (expected == 0.0f) {
          EXPECT_EQ(y[index + 1], expected);
          EXPECT_EQ(std::signbit(y[index + 1]), std::signbit(expected));
        } else if (std::isinf(expected)) {
          EXPECT_EQ(y[index + 1], expected);
        } else {
          EXPECT_FLOAT_EQ(y[index + 1], expected);
        }
      }
    }
    EXPECT_EQ(y.front(), sentinel);
    EXPECT_EQ(y.back(), sentinel);
  }
}

TEST(RotaryEmbedding, FloatVectorPrefixAndSpecialValues) { CheckSplitHalfPrefix<float>(); }

TEST(RotaryEmbedding, Float16VectorPrefixAndSpecialValues) {
  CheckSplitHalfPrefix<std::uint16_t>();
}

TEST(RotaryEmbedding, UnsupportedStorageTypesUseScalarFallback) {
  for (const auto type : {ml_cpu_compute::DataType::BFLOAT16, ml_cpu_compute::DataType::DOUBLE,
                          ml_cpu_compute::DataType::INT32, ml_cpu_compute::DataType::UNDEFINED}) {
    EXPECT_EQ(ml_cpu_compute::SelectRotaryEmbeddingSplitHalf(type), nullptr);
  }
}

} // namespace
