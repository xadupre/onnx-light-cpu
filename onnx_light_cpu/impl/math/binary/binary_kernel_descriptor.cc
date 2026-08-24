// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_kernel_descriptor.h"

#include "onnx_light_cpu/impl/math/binary/binary_arithmetic_kernel.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace onnx_light_cpu {
namespace {

using DT = BinaryDataType;
using Attrs = BinaryKernelDescriptor::Attributes;

std::size_t ElementSize(DT type) {
  switch (type) {
  case DT::BOOL:
  case DT::INT8:
  case DT::UINT8:
    return 1;
  case DT::INT16:
  case DT::UINT16:
  case DT::FLOAT16:
  case DT::BFLOAT16:
    return 2;
  case DT::INT32:
  case DT::UINT32:
  case DT::FLOAT:
    return 4;
  case DT::INT64:
  case DT::UINT64:
  case DT::DOUBLE:
    return 8;
  default:
    throw std::invalid_argument("onnx_light_cpu::BinaryKernelDescriptor: unsupported data type.");
  }
}

template <typename T> T WrapAdd(T left, T right) {
  using U = std::make_unsigned_t<T>;
  const U wrapped = static_cast<U>(static_cast<U>(left) + static_cast<U>(right));
  return std::bit_cast<T>(wrapped);
}

template <typename T> T WrapSub(T left, T right) {
  using U = std::make_unsigned_t<T>;
  const U wrapped = static_cast<U>(static_cast<U>(left) - static_cast<U>(right));
  return std::bit_cast<T>(wrapped);
}

template <typename T> T WrapMul(T left, T right) {
  using U = std::make_unsigned_t<T>;
  const U wrapped = static_cast<U>(static_cast<U>(left) * static_cast<U>(right));
  return std::bit_cast<T>(wrapped);
}

template <typename T> void ValidateIntegerDivisor(T divisor, const char *op_name) {
  if (divisor == 0) {
    throw std::invalid_argument(std::string("onnx_light_cpu::") + op_name +
                                ": zero integer divisor is unsupported.");
  }
}

template <typename T> void ValidateSignedDivisionOverflow(T left, T right, const char *op_name) {
  if constexpr (std::is_signed_v<T>) {
    if (left == std::numeric_limits<T>::min() && right == T(-1)) {
      throw std::invalid_argument(std::string("onnx_light_cpu::") + op_name +
                                  ": INT_MIN / -1 is unsupported.");
    }
  }
}

template <typename TBase, typename TExp>
TBase CheckedIntegerPow(TBase base, TExp exponent, const char *op_name) {
  if constexpr (std::is_floating_point_v<TExp>) {
    const double rounded = std::round(static_cast<double>(exponent));
    if (!std::isfinite(static_cast<double>(exponent)) || rounded != static_cast<double>(exponent)) {
      throw std::invalid_argument(std::string("onnx_light_cpu::") + op_name +
                                  ": integer Pow requires integral exponents.");
    }
    return CheckedIntegerPow<TBase, std::int64_t>(base, static_cast<std::int64_t>(rounded),
                                                  op_name);
  } else {
    if constexpr (std::is_signed_v<TExp>) {
      if (exponent < 0) {
        throw std::invalid_argument(std::string("onnx_light_cpu::") + op_name +
                                    ": integer Pow requires non-negative exponents.");
      }
    }
    using Wide = long double;
    Wide result = 1;
    Wide factor = static_cast<Wide>(base);
    std::uint64_t remaining = static_cast<std::uint64_t>(exponent);
    while (remaining != 0) {
      if ((remaining & 1U) != 0U) {
        result *= factor;
      }
      remaining >>= 1U;
      if (remaining != 0) {
        factor *= factor;
      }
      if (result > static_cast<Wide>(std::numeric_limits<TBase>::max()) ||
          result < static_cast<Wide>(std::numeric_limits<TBase>::min()) || !std::isfinite(result)) {
        throw std::invalid_argument(std::string("onnx_light_cpu::") + op_name +
                                    ": integer Pow overflow is unsupported.");
      }
    }
    return static_cast<TBase>(result);
  }
}

template <typename T> void WriteTyped(void *out, T value) { *reinterpret_cast<T *>(out) = value; }

template <typename T> T ReadTyped(const void *in) { return *reinterpret_cast<const T *>(in); }

template <typename T> void ComputeAdd(const void *left, const void *right, void *out) {
  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, std::uint8_t> &&
                !std::is_same_v<T, bool>) {
    WriteTyped<T>(out, WrapAdd(ReadTyped<T>(left), ReadTyped<T>(right)));
  } else {
    WriteTyped<T>(out, static_cast<T>(ReadTyped<T>(left) + ReadTyped<T>(right)));
  }
}

template <typename T> void ComputeSub(const void *left, const void *right, void *out) {
  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, std::uint8_t> &&
                !std::is_same_v<T, bool>) {
    WriteTyped<T>(out, WrapSub(ReadTyped<T>(left), ReadTyped<T>(right)));
  } else {
    WriteTyped<T>(out, static_cast<T>(ReadTyped<T>(left) - ReadTyped<T>(right)));
  }
}

template <typename T> void ComputeMul(const void *left, const void *right, void *out) {
  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, std::uint8_t> &&
                !std::is_same_v<T, bool>) {
    WriteTyped<T>(out, WrapMul(ReadTyped<T>(left), ReadTyped<T>(right)));
  } else {
    WriteTyped<T>(out, static_cast<T>(ReadTyped<T>(left) * ReadTyped<T>(right)));
  }
}

template <typename T> void ComputeDiv(const void *left, const void *right, void *out) {
  const T lhs = ReadTyped<T>(left);
  const T rhs = ReadTyped<T>(right);
  if constexpr (std::is_integral_v<T>) {
    ValidateIntegerDivisor(rhs, "Div");
    ValidateSignedDivisionOverflow(lhs, rhs, "Div");
  }
  WriteTyped<T>(out, static_cast<T>(lhs / rhs));
}

template <typename T>
void ValidateIntegerDivision(const void *left, const void *right, const char *op_name) {
  const T lhs = ReadTyped<T>(left);
  const T rhs = ReadTyped<T>(right);
  ValidateIntegerDivisor(rhs, op_name);
  ValidateSignedDivisionOverflow(lhs, rhs, op_name);
}

template <typename T> void ValidateDiv(const void *left, const void *right) {
  ValidateIntegerDivision<T>(left, right, "Div");
}

template <typename T> void ValidateMod(const void *left, const void *right) {
  ValidateIntegerDivision<T>(left, right, "Mod");
}

// Binary PR02: type-erased wrappers around the SIMD-dispatched bulk kernels
// declared in binary_arithmetic_kernel.h, adapting their typed signatures to
// the ``void *``-based function pointers stored on ``Adapter``.
template <typename T, void (*Fn)(const T *, const T *, T *, std::size_t)>
void BulkContiguousWrapper(const void *left, const void *right, void *out, std::size_t count) {
  Fn(static_cast<const T *>(left), static_cast<const T *>(right), static_cast<T *>(out), count);
}

template <typename T, void (*Fn)(T, const T *, T *, std::size_t)>
void BulkLeftScalarWrapper(const void *left, const void *right, void *out, std::size_t count) {
  Fn(*static_cast<const T *>(left), static_cast<const T *>(right), static_cast<T *>(out), count);
}

template <typename T, void (*Fn)(const T *, T, T *, std::size_t)>
void BulkRightScalarWrapper(const void *left, const void *right, void *out, std::size_t count) {
  Fn(static_cast<const T *>(left), *static_cast<const T *>(right), static_cast<T *>(out), count);
}

void SelectBulk(BinaryOperator op, DT left, BinaryKernelDescriptor::Adapter &adapter) {
#define ONNX_LIGHT_CPU_BIND_BULK(STEM, T)                                                          \
  adapter.bulk_contiguous = &BulkContiguousWrapper<T, &STEM##Contiguous>;                          \
  adapter.bulk_left_scalar = &BulkLeftScalarWrapper<T, &STEM##LeftScalar>;                         \
  adapter.bulk_right_scalar = &BulkRightScalarWrapper<T, &STEM##RightScalar>;
  switch (op) {
  case BinaryOperator::kAdd:
    if (left == DT::FLOAT) {
      ONNX_LIGHT_CPU_BIND_BULK(BinaryAddFloat32, float)
    } else if (left == DT::DOUBLE) {
      ONNX_LIGHT_CPU_BIND_BULK(BinaryAddFloat64, double)
    }
    break;
  case BinaryOperator::kSub:
    if (left == DT::FLOAT) {
      ONNX_LIGHT_CPU_BIND_BULK(BinarySubFloat32, float)
    } else if (left == DT::DOUBLE) {
      ONNX_LIGHT_CPU_BIND_BULK(BinarySubFloat64, double)
    }
    break;
  case BinaryOperator::kMul:
    if (left == DT::FLOAT) {
      ONNX_LIGHT_CPU_BIND_BULK(BinaryMulFloat32, float)
    } else if (left == DT::DOUBLE) {
      ONNX_LIGHT_CPU_BIND_BULK(BinaryMulFloat64, double)
    }
    break;
  case BinaryOperator::kDiv:
    if (left == DT::FLOAT) {
      ONNX_LIGHT_CPU_BIND_BULK(BinaryDivFloat32, float)
    } else if (left == DT::DOUBLE) {
      ONNX_LIGHT_CPU_BIND_BULK(BinaryDivFloat64, double)
    }
    break;
  default:
    break;
  }
#undef ONNX_LIGHT_CPU_BIND_BULK
}

template <typename T> T PythonMod(T left, T right) {
  T value = static_cast<T>(left % right);
  if constexpr (std::is_signed_v<T>) {
    if (value != 0 && ((value < 0) != (right < 0))) {
      value = static_cast<T>(value + right);
    }
  }
  return value;
}

template <typename T>
void ComputeModInt(const void *left, const void *right, void *out, std::int64_t fmod) {
  const T lhs = ReadTyped<T>(left);
  const T rhs = ReadTyped<T>(right);
  ValidateIntegerDivisor(rhs, "Mod");
  ValidateSignedDivisionOverflow(lhs, rhs, "Mod");
  WriteTyped<T>(out, fmod == 0 ? PythonMod(lhs, rhs) : static_cast<T>(lhs % rhs));
}

template <typename T> void ComputeModFloat(const void *left, const void *right, void *out) {
  WriteTyped<T>(out, static_cast<T>(std::fmod(ReadTyped<T>(left), ReadTyped<T>(right))));
}

template <typename TBase, typename TExp>
void ComputePow(const void *left, const void *right, void *out) {
  const TBase lhs = ReadTyped<TBase>(left);
  const TExp rhs = ReadTyped<TExp>(right);
  if constexpr (std::is_integral_v<TBase>) {
    WriteTyped<TBase>(out, CheckedIntegerPow(lhs, rhs, "Pow"));
  } else {
    WriteTyped<TBase>(out, static_cast<TBase>(std::pow(static_cast<long double>(lhs),
                                                       static_cast<long double>(rhs))));
  }
}

template <typename T> void ComputeEqual(const void *left, const void *right, void *out) {
  WriteTyped<std::uint8_t>(out, ReadTyped<T>(left) == ReadTyped<T>(right) ? 1U : 0U);
}

template <typename T> void ComputeGreater(const void *left, const void *right, void *out) {
  WriteTyped<std::uint8_t>(out, ReadTyped<T>(left) > ReadTyped<T>(right) ? 1U : 0U);
}

template <typename T> void ComputeGreaterOrEqual(const void *left, const void *right, void *out) {
  WriteTyped<std::uint8_t>(out, ReadTyped<T>(left) >= ReadTyped<T>(right) ? 1U : 0U);
}

template <typename T> void ComputeLess(const void *left, const void *right, void *out) {
  WriteTyped<std::uint8_t>(out, ReadTyped<T>(left) < ReadTyped<T>(right) ? 1U : 0U);
}

template <typename T> void ComputeLessOrEqual(const void *left, const void *right, void *out) {
  WriteTyped<std::uint8_t>(out, ReadTyped<T>(left) <= ReadTyped<T>(right) ? 1U : 0U);
}

void ComputeAnd(const void *left, const void *right, void *out) {
  WriteTyped<std::uint8_t>(
      out, (ReadTyped<std::uint8_t>(left) != 0 && ReadTyped<std::uint8_t>(right) != 0) ? 1U : 0U);
}

void ComputeOr(const void *left, const void *right, void *out) {
  WriteTyped<std::uint8_t>(
      out, (ReadTyped<std::uint8_t>(left) != 0 || ReadTyped<std::uint8_t>(right) != 0) ? 1U : 0U);
}

void ComputeXor(const void *left, const void *right, void *out) {
  WriteTyped<std::uint8_t>(
      out,
      ((ReadTyped<std::uint8_t>(left) != 0) != (ReadTyped<std::uint8_t>(right) != 0)) ? 1U : 0U);
}

template <typename T> void ComputeBitwiseAnd(const void *left, const void *right, void *out) {
  WriteTyped<T>(out, static_cast<T>(ReadTyped<T>(left) & ReadTyped<T>(right)));
}

template <typename T> void ComputeBitwiseOr(const void *left, const void *right, void *out) {
  WriteTyped<T>(out, static_cast<T>(ReadTyped<T>(left) | ReadTyped<T>(right)));
}

template <typename T> void ComputeBitwiseXor(const void *left, const void *right, void *out) {
  WriteTyped<T>(out, static_cast<T>(ReadTyped<T>(left) ^ ReadTyped<T>(right)));
}

template <typename T> void ComputeBitShiftLeft(const void *left, const void *right, void *out) {
  const T shift = ReadTyped<T>(right);
  if (shift >= static_cast<T>(sizeof(T) * 8)) {
    throw std::invalid_argument("onnx_light_cpu::BitShift: shift amount is out of range.");
  }
  WriteTyped<T>(out, static_cast<T>(ReadTyped<T>(left) << static_cast<unsigned>(shift)));
}

template <typename T> void ComputeBitShiftRight(const void *left, const void *right, void *out) {
  const T shift = ReadTyped<T>(right);
  if (shift >= static_cast<T>(sizeof(T) * 8)) {
    throw std::invalid_argument("onnx_light_cpu::BitShift: shift amount is out of range.");
  }
  WriteTyped<T>(out, static_cast<T>(ReadTyped<T>(left) >> static_cast<unsigned>(shift)));
}

template <typename T> void ValidateBitShift(const void *, const void *right) {
  if (ReadTyped<T>(right) >= static_cast<T>(sizeof(T) * 8)) {
    throw std::invalid_argument("onnx_light_cpu::BitShift: shift amount is out of range.");
  }
}

template <typename T> void ComputePRelu(const void *left, const void *right, void *out) {
  const T x = ReadTyped<T>(left);
  const T slope = ReadTyped<T>(right);
  WriteTyped<T>(out, x < static_cast<T>(0) ? static_cast<T>(x * slope) : x);
}

void ComputeFloat16Add(const void *left, const void *right, void *out) {
  WriteTyped<std::uint16_t>(
      out, detail::FloatToFloat16Bits(detail::Float16BitsToFloat(ReadTyped<std::uint16_t>(left)) +
                                      detail::Float16BitsToFloat(ReadTyped<std::uint16_t>(right))));
}
void ComputeBfloat16Add(const void *left, const void *right, void *out) {
  WriteTyped<std::uint16_t>(out, detail::FloatToBFloat16Bits(
                                     detail::Bfloat16BitsToFloat(ReadTyped<std::uint16_t>(left)) +
                                     detail::Bfloat16BitsToFloat(ReadTyped<std::uint16_t>(right))));
}
#define ONNX_LIGHT_CPU_HALF_BINARY(NAME, DECODE, ENCODE, EXPR)                                     \
  void NAME(const void *left, const void *right, void *out) {                                      \
    const float a = DECODE(ReadTyped<std::uint16_t>(left));                                        \
    const float b = DECODE(ReadTyped<std::uint16_t>(right));                                       \
    WriteTyped<std::uint16_t>(out, ENCODE(EXPR));                                                  \
  }

ONNX_LIGHT_CPU_HALF_BINARY(ComputeFloat16Sub, detail::Float16BitsToFloat,
                           detail::FloatToFloat16Bits, a - b)
ONNX_LIGHT_CPU_HALF_BINARY(ComputeBfloat16Sub, detail::Bfloat16BitsToFloat,
                           detail::FloatToBFloat16Bits, a - b)
ONNX_LIGHT_CPU_HALF_BINARY(ComputeFloat16Mul, detail::Float16BitsToFloat,
                           detail::FloatToFloat16Bits, a *b)
ONNX_LIGHT_CPU_HALF_BINARY(ComputeBfloat16Mul, detail::Bfloat16BitsToFloat,
                           detail::FloatToBFloat16Bits, a *b)
ONNX_LIGHT_CPU_HALF_BINARY(ComputeFloat16Div, detail::Float16BitsToFloat,
                           detail::FloatToFloat16Bits, a / b)
ONNX_LIGHT_CPU_HALF_BINARY(ComputeBfloat16Div, detail::Bfloat16BitsToFloat,
                           detail::FloatToBFloat16Bits, a / b)
ONNX_LIGHT_CPU_HALF_BINARY(ComputeFloat16Mod, detail::Float16BitsToFloat,
                           detail::FloatToFloat16Bits, std::fmod(a, b))
ONNX_LIGHT_CPU_HALF_BINARY(ComputeBfloat16Mod, detail::Bfloat16BitsToFloat,
                           detail::FloatToBFloat16Bits, std::fmod(a, b))
ONNX_LIGHT_CPU_HALF_BINARY(ComputeFloat16Equal, detail::Float16BitsToFloat,
                           detail::FloatToFloat16Bits, a == b ? 1.0f : 0.0f)
#undef ONNX_LIGHT_CPU_HALF_BINARY

void ComputeFloat16EqualBool(const void *left, const void *right, void *out) {
  WriteTyped<std::uint8_t>(out, detail::Float16BitsToFloat(ReadTyped<std::uint16_t>(left)) ==
                                        detail::Float16BitsToFloat(ReadTyped<std::uint16_t>(right))
                                    ? 1U
                                    : 0U);
}
void ComputeBfloat16EqualBool(const void *left, const void *right, void *out) {
  WriteTyped<std::uint8_t>(out, detail::Bfloat16BitsToFloat(ReadTyped<std::uint16_t>(left)) ==
                                        detail::Bfloat16BitsToFloat(ReadTyped<std::uint16_t>(right))
                                    ? 1U
                                    : 0U);
}
#define ONNX_LIGHT_CPU_HALF_COMPARE(NAME, DECODE, CMP)                                             \
  void NAME(const void *left, const void *right, void *out) {                                      \
    const float a = DECODE(ReadTyped<std::uint16_t>(left));                                        \
    const float b = DECODE(ReadTyped<std::uint16_t>(right));                                       \
    WriteTyped<std::uint8_t>(out, (CMP) ? 1U : 0U);                                                \
  }
ONNX_LIGHT_CPU_HALF_COMPARE(ComputeFloat16Greater, detail::Float16BitsToFloat, a > b)
ONNX_LIGHT_CPU_HALF_COMPARE(ComputeBfloat16Greater, detail::Bfloat16BitsToFloat, a > b)
ONNX_LIGHT_CPU_HALF_COMPARE(ComputeFloat16GreaterOrEqual, detail::Float16BitsToFloat, a >= b)
ONNX_LIGHT_CPU_HALF_COMPARE(ComputeBfloat16GreaterOrEqual, detail::Bfloat16BitsToFloat, a >= b)
ONNX_LIGHT_CPU_HALF_COMPARE(ComputeFloat16Less, detail::Float16BitsToFloat, a < b)
ONNX_LIGHT_CPU_HALF_COMPARE(ComputeBfloat16Less, detail::Bfloat16BitsToFloat, a < b)
ONNX_LIGHT_CPU_HALF_COMPARE(ComputeFloat16LessOrEqual, detail::Float16BitsToFloat, a <= b)
ONNX_LIGHT_CPU_HALF_COMPARE(ComputeBfloat16LessOrEqual, detail::Bfloat16BitsToFloat, a <= b)
#undef ONNX_LIGHT_CPU_HALF_COMPARE

void ComputeFloat16PRelu(const void *left, const void *right, void *out) {
  const float a = detail::Float16BitsToFloat(ReadTyped<std::uint16_t>(left));
  const float b = detail::Float16BitsToFloat(ReadTyped<std::uint16_t>(right));
  WriteTyped<std::uint16_t>(out, detail::FloatToFloat16Bits(a < 0.0f ? a * b : a));
}
void ComputeBfloat16PRelu(const void *left, const void *right, void *out) {
  const float a = detail::Bfloat16BitsToFloat(ReadTyped<std::uint16_t>(left));
  const float b = detail::Bfloat16BitsToFloat(ReadTyped<std::uint16_t>(right));
  WriteTyped<std::uint16_t>(out, detail::FloatToBFloat16Bits(a < 0.0f ? a * b : a));
}

template <typename TBase, typename TExp, auto Decode, auto Encode>
void ComputeHalfPow(const void *left, const void *right, void *out) {
  const float base = Decode(ReadTyped<std::uint16_t>(left));
  const TExp exponent = ReadTyped<TExp>(right);
  WriteTyped<std::uint16_t>(
      out, Encode(static_cast<float>(std::pow(base, static_cast<float>(exponent)))));
}

template <typename TBase, auto Decode>
void ComputeHalfCompareEq(const void *left, const void *right, void *out) {
  WriteTyped<std::uint8_t>(
      out,
      Decode(ReadTyped<std::uint16_t>(left)) == Decode(ReadTyped<std::uint16_t>(right)) ? 1U : 0U);
}

BinaryKernelDescriptor::Adapter::ScalarFn SelectScalar(BinaryOperator op, DT left, DT right,
                                                       const Attrs &attributes) {
  switch (op) {
  case BinaryOperator::kAdd:
    switch (left) {
    case DT::FLOAT:
      return &ComputeAdd<float>;
    case DT::DOUBLE:
      return &ComputeAdd<double>;
    case DT::FLOAT16:
      return &ComputeFloat16Add;
    case DT::BFLOAT16:
      return &ComputeBfloat16Add;
    case DT::INT8:
      return &ComputeAdd<std::int8_t>;
    case DT::INT16:
      return &ComputeAdd<std::int16_t>;
    case DT::INT32:
      return &ComputeAdd<std::int32_t>;
    case DT::INT64:
      return &ComputeAdd<std::int64_t>;
    case DT::UINT8:
      return &ComputeAdd<std::uint8_t>;
    case DT::UINT16:
      return &ComputeAdd<std::uint16_t>;
    case DT::UINT32:
      return &ComputeAdd<std::uint32_t>;
    case DT::UINT64:
      return &ComputeAdd<std::uint64_t>;
    default:
      break;
    }
    break;
  case BinaryOperator::kSub:
    switch (left) {
    case DT::FLOAT:
      return &ComputeSub<float>;
    case DT::DOUBLE:
      return &ComputeSub<double>;
    case DT::FLOAT16:
      return &ComputeFloat16Sub;
    case DT::BFLOAT16:
      return &ComputeBfloat16Sub;
    case DT::INT8:
      return &ComputeSub<std::int8_t>;
    case DT::INT16:
      return &ComputeSub<std::int16_t>;
    case DT::INT32:
      return &ComputeSub<std::int32_t>;
    case DT::INT64:
      return &ComputeSub<std::int64_t>;
    case DT::UINT8:
      return &ComputeSub<std::uint8_t>;
    case DT::UINT16:
      return &ComputeSub<std::uint16_t>;
    case DT::UINT32:
      return &ComputeSub<std::uint32_t>;
    case DT::UINT64:
      return &ComputeSub<std::uint64_t>;
    default:
      break;
    }
    break;
  case BinaryOperator::kMul:
    switch (left) {
    case DT::FLOAT:
      return &ComputeMul<float>;
    case DT::DOUBLE:
      return &ComputeMul<double>;
    case DT::FLOAT16:
      return &ComputeFloat16Mul;
    case DT::BFLOAT16:
      return &ComputeBfloat16Mul;
    case DT::INT8:
      return &ComputeMul<std::int8_t>;
    case DT::INT16:
      return &ComputeMul<std::int16_t>;
    case DT::INT32:
      return &ComputeMul<std::int32_t>;
    case DT::INT64:
      return &ComputeMul<std::int64_t>;
    case DT::UINT8:
      return &ComputeMul<std::uint8_t>;
    case DT::UINT16:
      return &ComputeMul<std::uint16_t>;
    case DT::UINT32:
      return &ComputeMul<std::uint32_t>;
    case DT::UINT64:
      return &ComputeMul<std::uint64_t>;
    default:
      break;
    }
    break;
  case BinaryOperator::kDiv:
    switch (left) {
    case DT::FLOAT:
      return &ComputeDiv<float>;
    case DT::DOUBLE:
      return &ComputeDiv<double>;
    case DT::FLOAT16:
      return &ComputeFloat16Div;
    case DT::BFLOAT16:
      return &ComputeBfloat16Div;
    case DT::INT8:
      return &ComputeDiv<std::int8_t>;
    case DT::INT16:
      return &ComputeDiv<std::int16_t>;
    case DT::INT32:
      return &ComputeDiv<std::int32_t>;
    case DT::INT64:
      return &ComputeDiv<std::int64_t>;
    case DT::UINT8:
      return &ComputeDiv<std::uint8_t>;
    case DT::UINT16:
      return &ComputeDiv<std::uint16_t>;
    case DT::UINT32:
      return &ComputeDiv<std::uint32_t>;
    case DT::UINT64:
      return &ComputeDiv<std::uint64_t>;
    default:
      break;
    }
    break;
  case BinaryOperator::kMod:
    switch (left) {
    case DT::FLOAT:
      return &ComputeModFloat<float>;
    case DT::DOUBLE:
      return &ComputeModFloat<double>;
    case DT::FLOAT16:
      return &ComputeFloat16Mod;
    case DT::BFLOAT16:
      return &ComputeBfloat16Mod;
    case DT::INT8:
      return attributes.mod_fmod == 0 ? [](const void *l, const void *r,
                                           void *o) { ComputeModInt<std::int8_t>(l, r, o, 0); }
                                      : [](const void *l, const void *r, void *o) {
                                          ComputeModInt<std::int8_t>(l, r, o, 1);
                                        };
    case DT::INT16:
      return attributes.mod_fmod == 0 ? [](const void *l, const void *r,
                                           void *o) { ComputeModInt<std::int16_t>(l, r, o, 0); }
                                      : [](const void *l, const void *r, void *o) {
                                          ComputeModInt<std::int16_t>(l, r, o, 1);
                                        };
    case DT::INT32:
      return attributes.mod_fmod == 0 ? [](const void *l, const void *r,
                                           void *o) { ComputeModInt<std::int32_t>(l, r, o, 0); }
                                      : [](const void *l, const void *r, void *o) {
                                          ComputeModInt<std::int32_t>(l, r, o, 1);
                                        };
    case DT::INT64:
      return attributes.mod_fmod == 0 ? [](const void *l, const void *r,
                                           void *o) { ComputeModInt<std::int64_t>(l, r, o, 0); }
                                      : [](const void *l, const void *r, void *o) {
                                          ComputeModInt<std::int64_t>(l, r, o, 1);
                                        };
    case DT::UINT8:
      return attributes.mod_fmod == 0 ? [](const void *l, const void *r,
                                           void *o) { ComputeModInt<std::uint8_t>(l, r, o, 0); }
                                      : [](const void *l, const void *r, void *o) {
                                          ComputeModInt<std::uint8_t>(l, r, o, 1);
                                        };
    case DT::UINT16:
      return attributes.mod_fmod == 0 ? [](const void *l, const void *r,
                                           void *o) { ComputeModInt<std::uint16_t>(l, r, o, 0); }
                                      : [](const void *l, const void *r, void *o) {
                                          ComputeModInt<std::uint16_t>(l, r, o, 1);
                                        };
    case DT::UINT32:
      return attributes.mod_fmod == 0 ? [](const void *l, const void *r,
                                           void *o) { ComputeModInt<std::uint32_t>(l, r, o, 0); }
                                      : [](const void *l, const void *r, void *o) {
                                          ComputeModInt<std::uint32_t>(l, r, o, 1);
                                        };
    case DT::UINT64:
      return attributes.mod_fmod == 0 ? [](const void *l, const void *r,
                                           void *o) { ComputeModInt<std::uint64_t>(l, r, o, 0); }
                                      : [](const void *l, const void *r, void *o) {
                                          ComputeModInt<std::uint64_t>(l, r, o, 1);
                                        };
    default:
      break;
    }
    break;
  case BinaryOperator::kPow:
    if (left == DT::FLOAT) {
      switch (right) {
      case DT::FLOAT:
        return &ComputePow<float, float>;
      case DT::INT32:
        return &ComputePow<float, std::int32_t>;
      case DT::INT64:
        return &ComputePow<float, std::int64_t>;
      case DT::UINT32:
        return &ComputePow<float, std::uint32_t>;
      case DT::UINT64:
        return &ComputePow<float, std::uint64_t>;
      default:
        break;
      }
    } else if (left == DT::FLOAT16) {
      switch (right) {
      case DT::FLOAT:
        return &ComputeHalfPow<std::uint16_t, float, detail::Float16BitsToFloat,
                               detail::FloatToFloat16Bits>;
      case DT::FLOAT16:
        return &ComputeHalfPow<std::uint16_t, std::uint16_t, detail::Float16BitsToFloat,
                               detail::FloatToFloat16Bits>;
      case DT::BFLOAT16:
        return &ComputeHalfPow<std::uint16_t, std::uint16_t, detail::Float16BitsToFloat,
                               detail::FloatToFloat16Bits>;
      case DT::INT32:
        return &ComputeHalfPow<std::uint16_t, std::int32_t, detail::Float16BitsToFloat,
                               detail::FloatToFloat16Bits>;
      case DT::INT64:
        return &ComputeHalfPow<std::uint16_t, std::int64_t, detail::Float16BitsToFloat,
                               detail::FloatToFloat16Bits>;
      case DT::UINT32:
        return &ComputeHalfPow<std::uint16_t, std::uint32_t, detail::Float16BitsToFloat,
                               detail::FloatToFloat16Bits>;
      case DT::UINT64:
        return &ComputeHalfPow<std::uint16_t, std::uint64_t, detail::Float16BitsToFloat,
                               detail::FloatToFloat16Bits>;
      default:
        break;
      }
    } else if (left == DT::BFLOAT16) {
      switch (right) {
      case DT::FLOAT:
        return &ComputeHalfPow<std::uint16_t, float, detail::Bfloat16BitsToFloat,
                               detail::FloatToBFloat16Bits>;
      case DT::FLOAT16:
        return &ComputeHalfPow<std::uint16_t, std::uint16_t, detail::Bfloat16BitsToFloat,
                               detail::FloatToBFloat16Bits>;
      case DT::BFLOAT16:
        return &ComputeHalfPow<std::uint16_t, std::uint16_t, detail::Bfloat16BitsToFloat,
                               detail::FloatToBFloat16Bits>;
      case DT::INT32:
        return &ComputeHalfPow<std::uint16_t, std::int32_t, detail::Bfloat16BitsToFloat,
                               detail::FloatToBFloat16Bits>;
      case DT::INT64:
        return &ComputeHalfPow<std::uint16_t, std::int64_t, detail::Bfloat16BitsToFloat,
                               detail::FloatToBFloat16Bits>;
      case DT::UINT32:
        return &ComputeHalfPow<std::uint16_t, std::uint32_t, detail::Bfloat16BitsToFloat,
                               detail::FloatToBFloat16Bits>;
      case DT::UINT64:
        return &ComputeHalfPow<std::uint16_t, std::uint64_t, detail::Bfloat16BitsToFloat,
                               detail::FloatToBFloat16Bits>;
      default:
        break;
      }
    } else if (left == DT::INT32) {
      switch (right) {
      case DT::FLOAT:
        return &ComputePow<std::int32_t, float>;
      case DT::INT32:
        return &ComputePow<std::int32_t, std::int32_t>;
      case DT::INT64:
        return &ComputePow<std::int32_t, std::int64_t>;
      case DT::UINT32:
        return &ComputePow<std::int32_t, std::uint32_t>;
      case DT::UINT64:
        return &ComputePow<std::int32_t, std::uint64_t>;
      default:
        break;
      }
    } else if (left == DT::INT64) {
      switch (right) {
      case DT::FLOAT:
        return &ComputePow<std::int64_t, float>;
      case DT::INT32:
        return &ComputePow<std::int64_t, std::int32_t>;
      case DT::INT64:
        return &ComputePow<std::int64_t, std::int64_t>;
      case DT::UINT32:
        return &ComputePow<std::int64_t, std::uint32_t>;
      case DT::UINT64:
        return &ComputePow<std::int64_t, std::uint64_t>;
      default:
        break;
      }
    }
    break;
  case BinaryOperator::kEqual:
    switch (left) {
    case DT::BOOL:
      return &ComputeEqual<std::uint8_t>;
    case DT::FLOAT:
      return &ComputeEqual<float>;
    case DT::DOUBLE:
      return &ComputeEqual<double>;
    case DT::FLOAT16:
      return &ComputeFloat16EqualBool;
    case DT::BFLOAT16:
      return &ComputeBfloat16EqualBool;
    case DT::INT8:
      return &ComputeEqual<std::int8_t>;
    case DT::INT16:
      return &ComputeEqual<std::int16_t>;
    case DT::INT32:
      return &ComputeEqual<std::int32_t>;
    case DT::INT64:
      return &ComputeEqual<std::int64_t>;
    case DT::UINT8:
      return &ComputeEqual<std::uint8_t>;
    case DT::UINT16:
      return &ComputeEqual<std::uint16_t>;
    case DT::UINT32:
      return &ComputeEqual<std::uint32_t>;
    case DT::UINT64:
      return &ComputeEqual<std::uint64_t>;
    default:
      break;
    }
    break;
  case BinaryOperator::kGreater:
    switch (left) {
    case DT::FLOAT:
      return &ComputeGreater<float>;
    case DT::FLOAT16:
      return &ComputeFloat16Greater;
    case DT::BFLOAT16:
      return &ComputeBfloat16Greater;
    case DT::INT8:
      return &ComputeGreater<std::int8_t>;
    case DT::INT16:
      return &ComputeGreater<std::int16_t>;
    case DT::INT32:
      return &ComputeGreater<std::int32_t>;
    case DT::INT64:
      return &ComputeGreater<std::int64_t>;
    case DT::UINT8:
      return &ComputeGreater<std::uint8_t>;
    case DT::UINT16:
      return &ComputeGreater<std::uint16_t>;
    case DT::UINT32:
      return &ComputeGreater<std::uint32_t>;
    case DT::UINT64:
      return &ComputeGreater<std::uint64_t>;
    default:
      break;
    }
    break;
  case BinaryOperator::kGreaterOrEqual:
    switch (left) {
    case DT::FLOAT:
      return &ComputeGreaterOrEqual<float>;
    case DT::FLOAT16:
      return &ComputeFloat16GreaterOrEqual;
    case DT::BFLOAT16:
      return &ComputeBfloat16GreaterOrEqual;
    case DT::INT8:
      return &ComputeGreaterOrEqual<std::int8_t>;
    case DT::INT16:
      return &ComputeGreaterOrEqual<std::int16_t>;
    case DT::INT32:
      return &ComputeGreaterOrEqual<std::int32_t>;
    case DT::INT64:
      return &ComputeGreaterOrEqual<std::int64_t>;
    case DT::UINT8:
      return &ComputeGreaterOrEqual<std::uint8_t>;
    case DT::UINT16:
      return &ComputeGreaterOrEqual<std::uint16_t>;
    case DT::UINT32:
      return &ComputeGreaterOrEqual<std::uint32_t>;
    case DT::UINT64:
      return &ComputeGreaterOrEqual<std::uint64_t>;
    default:
      break;
    }
    break;
  case BinaryOperator::kLess:
    switch (left) {
    case DT::FLOAT:
      return &ComputeLess<float>;
    case DT::FLOAT16:
      return &ComputeFloat16Less;
    case DT::BFLOAT16:
      return &ComputeBfloat16Less;
    case DT::INT8:
      return &ComputeLess<std::int8_t>;
    case DT::INT16:
      return &ComputeLess<std::int16_t>;
    case DT::INT32:
      return &ComputeLess<std::int32_t>;
    case DT::INT64:
      return &ComputeLess<std::int64_t>;
    case DT::UINT8:
      return &ComputeLess<std::uint8_t>;
    case DT::UINT16:
      return &ComputeLess<std::uint16_t>;
    case DT::UINT32:
      return &ComputeLess<std::uint32_t>;
    case DT::UINT64:
      return &ComputeLess<std::uint64_t>;
    default:
      break;
    }
    break;
  case BinaryOperator::kLessOrEqual:
    switch (left) {
    case DT::FLOAT:
      return &ComputeLessOrEqual<float>;
    case DT::FLOAT16:
      return &ComputeFloat16LessOrEqual;
    case DT::BFLOAT16:
      return &ComputeBfloat16LessOrEqual;
    case DT::INT8:
      return &ComputeLessOrEqual<std::int8_t>;
    case DT::INT16:
      return &ComputeLessOrEqual<std::int16_t>;
    case DT::INT32:
      return &ComputeLessOrEqual<std::int32_t>;
    case DT::INT64:
      return &ComputeLessOrEqual<std::int64_t>;
    case DT::UINT8:
      return &ComputeLessOrEqual<std::uint8_t>;
    case DT::UINT16:
      return &ComputeLessOrEqual<std::uint16_t>;
    case DT::UINT32:
      return &ComputeLessOrEqual<std::uint32_t>;
    case DT::UINT64:
      return &ComputeLessOrEqual<std::uint64_t>;
    default:
      break;
    }
    break;
  case BinaryOperator::kAnd:
    return &ComputeAnd;
  case BinaryOperator::kOr:
    return &ComputeOr;
  case BinaryOperator::kXor:
    return &ComputeXor;
  case BinaryOperator::kBitwiseAnd:
    switch (left) {
    case DT::INT8:
      return &ComputeBitwiseAnd<std::int8_t>;
    case DT::INT16:
      return &ComputeBitwiseAnd<std::int16_t>;
    case DT::INT32:
      return &ComputeBitwiseAnd<std::int32_t>;
    case DT::INT64:
      return &ComputeBitwiseAnd<std::int64_t>;
    case DT::UINT8:
      return &ComputeBitwiseAnd<std::uint8_t>;
    case DT::UINT16:
      return &ComputeBitwiseAnd<std::uint16_t>;
    case DT::UINT32:
      return &ComputeBitwiseAnd<std::uint32_t>;
    case DT::UINT64:
      return &ComputeBitwiseAnd<std::uint64_t>;
    default:
      break;
    }
    break;
  case BinaryOperator::kBitwiseOr:
    switch (left) {
    case DT::INT8:
      return &ComputeBitwiseOr<std::int8_t>;
    case DT::INT16:
      return &ComputeBitwiseOr<std::int16_t>;
    case DT::INT32:
      return &ComputeBitwiseOr<std::int32_t>;
    case DT::INT64:
      return &ComputeBitwiseOr<std::int64_t>;
    case DT::UINT8:
      return &ComputeBitwiseOr<std::uint8_t>;
    case DT::UINT16:
      return &ComputeBitwiseOr<std::uint16_t>;
    case DT::UINT32:
      return &ComputeBitwiseOr<std::uint32_t>;
    case DT::UINT64:
      return &ComputeBitwiseOr<std::uint64_t>;
    default:
      break;
    }
    break;
  case BinaryOperator::kBitwiseXor:
    switch (left) {
    case DT::INT8:
      return &ComputeBitwiseXor<std::int8_t>;
    case DT::INT16:
      return &ComputeBitwiseXor<std::int16_t>;
    case DT::INT32:
      return &ComputeBitwiseXor<std::int32_t>;
    case DT::INT64:
      return &ComputeBitwiseXor<std::int64_t>;
    case DT::UINT8:
      return &ComputeBitwiseXor<std::uint8_t>;
    case DT::UINT16:
      return &ComputeBitwiseXor<std::uint16_t>;
    case DT::UINT32:
      return &ComputeBitwiseXor<std::uint32_t>;
    case DT::UINT64:
      return &ComputeBitwiseXor<std::uint64_t>;
    default:
      break;
    }
    break;
  case BinaryOperator::kBitShift:
    if (attributes.bitshift_direction == Attrs::BitShiftDirection::kLeft) {
      switch (left) {
      case DT::UINT8:
        return &ComputeBitShiftLeft<std::uint8_t>;
      case DT::UINT16:
        return &ComputeBitShiftLeft<std::uint16_t>;
      case DT::UINT32:
        return &ComputeBitShiftLeft<std::uint32_t>;
      case DT::UINT64:
        return &ComputeBitShiftLeft<std::uint64_t>;
      default:
        break;
      }
    } else {
      switch (left) {
      case DT::UINT8:
        return &ComputeBitShiftRight<std::uint8_t>;
      case DT::UINT16:
        return &ComputeBitShiftRight<std::uint16_t>;
      case DT::UINT32:
        return &ComputeBitShiftRight<std::uint32_t>;
      case DT::UINT64:
        return &ComputeBitShiftRight<std::uint64_t>;
      default:
        break;
      }
    }
    break;
  case BinaryOperator::kPRelu:
    switch (left) {
    case DT::FLOAT:
      return &ComputePRelu<float>;
    case DT::DOUBLE:
      return &ComputePRelu<double>;
    case DT::FLOAT16:
      return &ComputeFloat16PRelu;
    case DT::BFLOAT16:
      return &ComputeBfloat16PRelu;
    case DT::INT32:
      return &ComputePRelu<std::int32_t>;
    case DT::INT64:
      return &ComputePRelu<std::int64_t>;
    case DT::UINT32:
      return &ComputePRelu<std::uint32_t>;
    case DT::UINT64:
      return &ComputePRelu<std::uint64_t>;
    default:
      break;
    }
    break;
  }
  throw std::invalid_argument(
      "onnx_light_cpu::BinaryKernelDescriptor: unsupported type signature.");
}

BinaryKernelDescriptor::Adapter::ValidateFn SelectValidator(BinaryOperator op, DT left) {
  if (op != BinaryOperator::kDiv && op != BinaryOperator::kMod && op != BinaryOperator::kBitShift) {
    return nullptr;
  }
#define ONNX_LIGHT_CPU_INTEGER_VALIDATOR(TYPE, CPP_TYPE)                                           \
  case DT::TYPE:                                                                                   \
    if (op == BinaryOperator::kDiv)                                                                \
      return &ValidateDiv<CPP_TYPE>;                                                               \
    if (op == BinaryOperator::kMod)                                                                \
      return &ValidateMod<CPP_TYPE>;                                                               \
    return &ValidateBitShift<CPP_TYPE>;
  switch (left) {
    ONNX_LIGHT_CPU_INTEGER_VALIDATOR(INT8, std::int8_t)
    ONNX_LIGHT_CPU_INTEGER_VALIDATOR(INT16, std::int16_t)
    ONNX_LIGHT_CPU_INTEGER_VALIDATOR(INT32, std::int32_t)
    ONNX_LIGHT_CPU_INTEGER_VALIDATOR(INT64, std::int64_t)
    ONNX_LIGHT_CPU_INTEGER_VALIDATOR(UINT8, std::uint8_t)
    ONNX_LIGHT_CPU_INTEGER_VALIDATOR(UINT16, std::uint16_t)
    ONNX_LIGHT_CPU_INTEGER_VALIDATOR(UINT32, std::uint32_t)
    ONNX_LIGHT_CPU_INTEGER_VALIDATOR(UINT64, std::uint64_t)
  default:
    return nullptr;
  }
#undef ONNX_LIGHT_CPU_INTEGER_VALIDATOR
}

} // namespace

BinaryKernelDescriptor::BinaryKernelDescriptor(std::string op_type, std::int64_t opset_version,
                                               const Attributes &attributes)
    : manifest_(GetBinaryManifestEntry(op_type)), opset_version_(opset_version),
      attributes_(attributes), cache_identity_(NextCacheIdentity()) {
  if (opset_version_ < manifest_.since_version) {
    throw std::invalid_argument("onnx_light_cpu::BinaryKernelDescriptor: opset " +
                                std::to_string(opset_version_) + " is too old for operator '" +
                                std::string(manifest_.op_type) + "'.");
  }
  if ((manifest_.op == BinaryOperator::kMod) && attributes_.mod_fmod != 0 &&
      attributes_.mod_fmod != 1) {
    throw std::invalid_argument(
        "onnx_light_cpu::BinaryKernelDescriptor: Mod attribute fmod must be 0 or 1.");
  }
  adapters_.reserve(manifest_.signatures.size());
  for (const BinaryTypeSignature &signature : manifest_.signatures) {
    if ((manifest_.op == BinaryOperator::kMod) && attributes_.mod_fmod == 0 &&
        (signature.left == DT::FLOAT || signature.left == DT::DOUBLE ||
         signature.left == DT::FLOAT16 || signature.left == DT::BFLOAT16)) {
      continue;
    }
    Adapter adapter;
    adapter.signature = signature;
    adapter.scalar = SelectScalar(manifest_.op, signature.left, signature.right, attributes_);
    adapter.validate = SelectValidator(manifest_.op, signature.left);
    adapter.left_size = ElementSize(signature.left);
    adapter.right_size = ElementSize(signature.right);
    adapter.output_size = ElementSize(signature.output);
    if (signature.left == signature.right && signature.left == signature.output) {
      SelectBulk(manifest_.op, signature.left, adapter);
    }
    adapters_.push_back(adapter);
  }
}

std::uint64_t BinaryKernelDescriptor::NextCacheIdentity() {
  static std::atomic<std::uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

BinaryDataType BinaryKernelDescriptor::ResolveOutputType(BinaryDataType left,
                                                         BinaryDataType right) const {
  for (const Adapter &adapter : adapters_) {
    if (adapter.signature.left == left && adapter.signature.right == right) {
      return adapter.signature.output;
    }
  }
  throw std::invalid_argument(
      "onnx_light_cpu::BinaryKernelDescriptor: unsupported input types for operator '" +
      std::string(manifest_.op_type) + "'.");
}

const BinaryKernelDescriptor::Adapter &
BinaryKernelDescriptor::ResolveAdapter(BinaryDataType left, BinaryDataType right,
                                       BinaryDataType output) const {
  for (const Adapter &adapter : adapters_) {
    if (adapter.signature.left == left && adapter.signature.right == right &&
        adapter.signature.output == output) {
      return adapter;
    }
  }
  throw std::invalid_argument(
      "onnx_light_cpu::BinaryKernelDescriptor: unsupported input/output types for operator '" +
      std::string(manifest_.op_type) + "'.");
}

} // namespace onnx_light_cpu
