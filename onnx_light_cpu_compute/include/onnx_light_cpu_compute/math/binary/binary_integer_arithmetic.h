// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace onnx_light_cpu::detail {

inline std::uint64_t MultiplyHigh64Portable(std::uint64_t a, std::uint64_t b) {
  const std::uint64_t a0 = static_cast<std::uint32_t>(a), a1 = a >> 32;
  const std::uint64_t b0 = static_cast<std::uint32_t>(b), b1 = b >> 32;
  const std::uint64_t low = a0 * b0;
  const std::uint64_t middle = a1 * b0 + (low >> 32);
  const std::uint64_t carry = (middle & UINT64_C(0xffffffff)) + a0 * b1;
  return a1 * b1 + (middle >> 32) + (carry >> 32);
}

inline std::uint64_t MultiplyHigh64(std::uint64_t a, std::uint64_t b) {
#if defined(__SIZEOF_INT128__)
  return static_cast<std::uint64_t>((static_cast<__uint128_t>(a) * b) >> 64);
#elif defined(_MSC_VER) && defined(_M_X64)
  std::uint64_t high;
  _umul128(a, b, &high);
  return high;
#else
  return MultiplyHigh64Portable(a, b);
#endif
}

template <typename T> std::make_unsigned_t<T> IntegerMagnitude(T value) {
  using U = std::make_unsigned_t<T>;
  const U bits = static_cast<U>(value);
  if constexpr (std::is_signed_v<T>) {
    return value < 0 ? static_cast<U>(U{0} - bits) : bits;
  }
  return bits;
}

template <typename T> bool MultiplyOverflowPortable(T a, T b, T &out) {
  using U = std::make_unsigned_t<T>;
  const U ua = IntegerMagnitude(a), ub = IntegerMagnitude(b);
  bool negative = false;
  if constexpr (std::is_signed_v<T>) {
    negative = (a < 0) != (b < 0);
  }
  const U limit = static_cast<U>(std::numeric_limits<T>::max()) + static_cast<U>(negative);
  const std::uint64_t product = static_cast<std::uint64_t>(ua) * ub;
  if ((sizeof(T) == 8 && MultiplyHigh64Portable(ua, ub) != 0) || product > limit) {
    return true;
  }
  const U bits =
      negative ? static_cast<U>(U{0} - static_cast<U>(product)) : static_cast<U>(product);
  out = std::bit_cast<T>(bits);
  return false;
}

template <typename T> T CheckedIntegerMultiply(T a, T b, const char *op_name) {
  T result;
#if defined(__GNUC__) || defined(__clang__)
  const bool overflow = __builtin_mul_overflow(a, b, &result);
#else
  const bool overflow = MultiplyOverflowPortable(a, b, result);
#endif
  if (overflow) {
    throw std::invalid_argument(std::string("onnx_light_cpu::") + op_name +
                                ": integer Pow overflow is unsupported.");
  }
  return result;
}

template <typename TBase, typename TExp>
TBase CheckedIntegerPow(TBase base, TExp exponent, const char *op_name) {
  if constexpr (std::is_floating_point_v<TExp>) {
    const double value = static_cast<double>(exponent);
    if (!std::isfinite(value) || std::trunc(value) != value) {
      throw std::invalid_argument(std::string("onnx_light_cpu::") + op_name +
                                  ": integer Pow requires integral exponents.");
    }
    constexpr double int64_exclusive_upper = 9223372036854775808.0;
    if (value < -int64_exclusive_upper || value >= int64_exclusive_upper) {
      throw std::invalid_argument(std::string("onnx_light_cpu::") + op_name +
                                  ": integer Pow exponent is out of range.");
    }
    return CheckedIntegerPow(base, static_cast<std::int64_t>(value), op_name);
  } else {
    if constexpr (std::is_signed_v<TExp>) {
      if (exponent < 0) {
        throw std::invalid_argument(std::string("onnx_light_cpu::") + op_name +
                                    ": integer Pow requires non-negative exponents.");
      }
    }
    std::uint64_t remaining = static_cast<std::uint64_t>(exponent);
    if (remaining == 0 || base == 1) {
      return TBase{1};
    }
    if (remaining == 1 || base == 0) {
      return base;
    }
    if (remaining == 2) {
      return CheckedIntegerMultiply(base, base, op_name);
    }
    if constexpr (std::is_signed_v<TBase>) {
      if (base == -1) {
        return (remaining & 1U) != 0 ? TBase{-1} : TBase{1};
      }
    }
    TBase result = 1;
    while (remaining != 0) {
      if ((remaining & 1U) != 0) {
        result = CheckedIntegerMultiply(result, base, op_name);
      }
      remaining >>= 1;
      if (remaining != 0) {
        base = CheckedIntegerMultiply(base, base, op_name);
      }
    }
    return result;
  }
}

// For non-powers of two, m = floor(2^width / d). The high product gives a
// quotient at most one below floor(n / d), so one remainder comparison suffices.
// Only construction divides; evaluation uses exact unsigned multiply/shift.
template <typename U> class IntegerDivider {
  static_assert(std::is_unsigned_v<U>);
  using Word = std::conditional_t<(sizeof(U) <= 4), std::uint32_t, std::uint64_t>;

public:
  explicit IntegerDivider(U divisor) : divisor_(divisor) {
    if (divisor == 0) {
      throw std::invalid_argument("onnx_light_cpu::IntegerDivider: zero divisor.");
    }
    if (std::has_single_bit(divisor_)) {
      shift_ = std::countr_zero(divisor_);
    } else {
      const Word maximum = std::numeric_limits<Word>::max();
      multiplier_ = maximum / divisor_;
      multiplier_ += maximum % divisor_ == divisor_ - 1;
    }
  }

  U Divide(U numerator) const {
    if (shift_ >= 0) {
      return static_cast<U>(numerator >> shift_);
    }
    Word quotient;
    if constexpr (sizeof(Word) == 4) {
      quotient = static_cast<Word>((std::uint64_t{numerator} * multiplier_) >> 32);
    } else {
      quotient = MultiplyHigh64(numerator, multiplier_);
    }
    quotient += static_cast<Word>(numerator) - quotient * divisor_ >= divisor_;
    return static_cast<U>(quotient);
  }

private:
  Word divisor_;
  Word multiplier_ = 0;
  int shift_ = -1;
};

} // namespace onnx_light_cpu::detail
