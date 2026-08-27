// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_kernel_descriptor.h"

#include "onnx_light_cpu/impl/math/binary/binary_arithmetic_kernel.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"

#include <algorithm>
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

using DT = DataType;
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

template <typename TBase, typename TExp> TBase PowIntegerExponent(TBase base, TExp exponent) {
  using UnsignedExponent = std::make_unsigned_t<TExp>;
  bool negative = false;
  if constexpr (std::is_signed_v<TExp>) {
    negative = exponent < 0;
  }
  UnsignedExponent remaining = static_cast<UnsignedExponent>(exponent);
  if (negative) {
    remaining = UnsignedExponent{0} - remaining;
  }
  TBase result = static_cast<TBase>(1);
  TBase factor = base;
  while (remaining != 0) {
    if ((remaining & UnsignedExponent{1}) != 0) {
      result *= factor;
    }
    remaining >>= 1;
    if (remaining != 0) {
      factor *= factor;
    }
  }
  return negative ? static_cast<TBase>(1) / result : result;
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

template <typename T> void ComputeDivUnchecked(const void *left, const void *right, void *out) {
  WriteTyped<T>(out, static_cast<T>(ReadTyped<T>(left) / ReadTyped<T>(right)));
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

template <typename TLeft, typename TRight, typename TOut,
          void (*Compute)(const void *, const void *, void *)>
void BulkComputeContiguous(const void *left, const void *right, void *out, std::size_t count) {
  const auto *typed_left = static_cast<const TLeft *>(left);
  const auto *typed_right = static_cast<const TRight *>(right);
  auto *typed_out = static_cast<TOut *>(out);
  for (std::size_t i = 0; i < count; ++i) {
    Compute(typed_left + i, typed_right + i, typed_out + i);
  }
}

template <typename TLeft, typename TRight, typename TOut,
          void (*Compute)(const void *, const void *, void *)>
void BulkComputeLeftScalar(const void *left, const void *right, void *out, std::size_t count) {
  const auto *typed_right = static_cast<const TRight *>(right);
  auto *typed_out = static_cast<TOut *>(out);
  for (std::size_t i = 0; i < count; ++i) {
    Compute(left, typed_right + i, typed_out + i);
  }
}

template <typename TLeft, typename TRight, typename TOut,
          void (*Compute)(const void *, const void *, void *)>
void BulkComputeRightScalar(const void *left, const void *right, void *out, std::size_t count) {
  const auto *typed_left = static_cast<const TLeft *>(left);
  auto *typed_out = static_cast<TOut *>(out);
  for (std::size_t i = 0; i < count; ++i) {
    Compute(typed_left + i, right, typed_out + i);
  }
}

template <char Operation>
void BulkLogicalContiguous(const void *left, const void *right, void *out, std::size_t count) {
  const auto *typed_left = static_cast<const std::uint8_t *>(left);
  const auto *typed_right = static_cast<const std::uint8_t *>(right);
  auto *typed_out = static_cast<std::uint8_t *>(out);
  for (std::size_t i = 0; i < count; ++i) {
    const bool lhs = typed_left[i] != 0;
    const bool rhs = typed_right[i] != 0;
    if constexpr (Operation == '&') {
      typed_out[i] = static_cast<std::uint8_t>(lhs & rhs);
    } else if constexpr (Operation == '|') {
      typed_out[i] = static_cast<std::uint8_t>(lhs | rhs);
    } else {
      typed_out[i] = static_cast<std::uint8_t>(lhs != rhs);
    }
  }
}

template <char Operation>
void BulkLogicalLeftScalar(const void *left, const void *right, void *out, std::size_t count) {
  const bool lhs = *static_cast<const std::uint8_t *>(left) != 0;
  const auto *typed_right = static_cast<const std::uint8_t *>(right);
  auto *typed_out = static_cast<std::uint8_t *>(out);
  for (std::size_t i = 0; i < count; ++i) {
    const bool rhs = typed_right[i] != 0;
    if constexpr (Operation == '&') {
      typed_out[i] = static_cast<std::uint8_t>(lhs & rhs);
    } else if constexpr (Operation == '|') {
      typed_out[i] = static_cast<std::uint8_t>(lhs | rhs);
    } else {
      typed_out[i] = static_cast<std::uint8_t>(lhs != rhs);
    }
  }
}

template <char Operation>
void BulkLogicalRightScalar(const void *left, const void *right, void *out, std::size_t count) {
  const auto *typed_left = static_cast<const std::uint8_t *>(left);
  const bool rhs = *static_cast<const std::uint8_t *>(right) != 0;
  auto *typed_out = static_cast<std::uint8_t *>(out);
  for (std::size_t i = 0; i < count; ++i) {
    const bool lhs = typed_left[i] != 0;
    if constexpr (Operation == '&') {
      typed_out[i] = static_cast<std::uint8_t>(lhs & rhs);
    } else if constexpr (Operation == '|') {
      typed_out[i] = static_cast<std::uint8_t>(lhs | rhs);
    } else {
      typed_out[i] = static_cast<std::uint8_t>(lhs != rhs);
    }
  }
}

template <int Exponent> float FastFloatIntegerPower(float value) {
  float result;
  if constexpr (Exponent == 2) {
    result = value * value;
  } else if constexpr (Exponent == 3) {
    result = value * value * value;
  } else if constexpr (Exponent == 4) {
    const float squared = value * value;
    result = squared * squared;
  } else {
    const float squared = value * value;
    result = squared * squared * value;
  }
  if (std::isfinite(value) && std::isfinite(result) && (result != 0.0f || value == 0.0f)) {
    return result;
  }
  return std::pow(value, static_cast<float>(Exponent));
}

float FastFloatPower(float base, float exponent) {
  if (exponent == 0.0f) {
    return 1.0f;
  }
  if (exponent == 1.0f) {
    return base;
  }
  if (exponent == 2.0f) {
    return FastFloatIntegerPower<2>(base);
  }
  if (exponent == 3.0f) {
    return FastFloatIntegerPower<3>(base);
  }
  if (exponent == 4.0f) {
    return FastFloatIntegerPower<4>(base);
  }
  if (exponent == 5.0f) {
    return FastFloatIntegerPower<5>(base);
  }
  return std::pow(base, exponent);
}

void EvaluateFloatPowBlock(const float *base, const float *exponent, float *output,
                           std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  static const bool use_avx512 = DetectSimdLevel() >= SimdLevel::kAVX512;
  if (count >= 16 && use_avx512) {
    PowFloat32_AVX512(base, exponent, output, count);
    return;
  }
#endif
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = FastFloatPower(base[i], exponent[i]);
  }
}

void EvaluateFloatPowLeftScalarBlock(float base, const float *exponent, float *output,
                                     std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  static const bool use_avx512 = DetectSimdLevel() >= SimdLevel::kAVX512;
  if (count >= 16 && use_avx512) {
    PowFloat32LeftScalar_AVX512(base, exponent, output, count);
    return;
  }
#endif
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = FastFloatPower(base, exponent[i]);
  }
}

void EvaluateFloatPowRightScalarBlock(const float *base, float exponent, float *output,
                                      std::size_t count) {
  if (exponent == 0.0f) {
    std::fill_n(output, count, 1.0f);
    return;
  }
  if (exponent == 1.0f) {
    std::copy_n(base, count, output);
    return;
  }
  if (exponent == 2.0f) {
    for (std::size_t i = 0; i < count; ++i) {
      output[i] = FastFloatIntegerPower<2>(base[i]);
    }
    return;
  }
  if (exponent == 3.0f) {
    for (std::size_t i = 0; i < count; ++i) {
      output[i] = FastFloatIntegerPower<3>(base[i]);
    }
    return;
  }
  if (exponent == 4.0f) {
    for (std::size_t i = 0; i < count; ++i) {
      output[i] = FastFloatIntegerPower<4>(base[i]);
    }
    return;
  }
  if (exponent == 5.0f) {
    for (std::size_t i = 0; i < count; ++i) {
      output[i] = FastFloatIntegerPower<5>(base[i]);
    }
    return;
  }
  constexpr std::size_t kBlockSize = 1024;
  alignas(32) float exponent_f32[kBlockSize];
  for (std::size_t offset = 0; offset < count; offset += kBlockSize) {
    const std::size_t block = std::min(kBlockSize, count - offset);
    std::fill_n(exponent_f32, block, exponent);
    EvaluateFloatPowBlock(base + offset, exponent_f32, output + offset, block);
  }
}

void BulkFloatPowRightScalar(const void *left, const void *right, void *out, std::size_t count) {
  const auto *typed_left = static_cast<const float *>(left);
  const float exponent = *static_cast<const float *>(right);
  auto *typed_out = static_cast<float *>(out);
  if (exponent == 2.0f) {
    for (std::size_t i = 0; i < count; ++i) {
      typed_out[i] = FastFloatIntegerPower<2>(typed_left[i]);
    }
    return;
  }
  if (exponent == 3.0f) {
    for (std::size_t i = 0; i < count; ++i) {
      typed_out[i] = FastFloatIntegerPower<3>(typed_left[i]);
    }
    return;
  }
  if (exponent == 4.0f) {
    for (std::size_t i = 0; i < count; ++i) {
      typed_out[i] = FastFloatIntegerPower<4>(typed_left[i]);
    }
    return;
  }
  if (exponent == 5.0f) {
    for (std::size_t i = 0; i < count; ++i) {
      typed_out[i] = FastFloatIntegerPower<5>(typed_left[i]);
    }
    return;
  }
  if (exponent == 1.0f) {
    std::copy_n(typed_left, count, typed_out);
    return;
  }
  if (exponent == 0.0f) {
    std::fill_n(typed_out, count, 1.0f);
    return;
  }
  for (std::size_t i = 0; i < count; ++i) {
    typed_out[i] = std::pow(typed_left[i], exponent);
  }
}

void BulkFloatPow(const void *left, const void *right, void *out, std::size_t count) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX512
  if (count >= 1024 && DetectSimdLevel() >= SimdLevel::kAVX512) {
    PowFloat32_AVX512(static_cast<const float *>(left), static_cast<const float *>(right),
                      static_cast<float *>(out), count);
    return;
  }
#endif
  const auto *typed_left = static_cast<const float *>(left);
  const auto *typed_right = static_cast<const float *>(right);
  auto *typed_out = static_cast<float *>(out);
  for (std::size_t i = 0; i < count; ++i) {
    typed_out[i] = FastFloatPower(typed_left[i], typed_right[i]);
  }
}

void BulkFloatPowLeftScalar(const void *left, const void *right, void *out, std::size_t count) {
  const float base = *static_cast<const float *>(left);
  const auto *typed_right = static_cast<const float *>(right);
  auto *typed_out = static_cast<float *>(out);
  EvaluateFloatPowLeftScalarBlock(base, typed_right, typed_out, count);
}

void ComputeFloat16Add(const void *, const void *, void *);
void ComputeBfloat16Add(const void *, const void *, void *);
void ComputeFloat16Sub(const void *, const void *, void *);
void ComputeBfloat16Sub(const void *, const void *, void *);
void ComputeFloat16Mul(const void *, const void *, void *);
void ComputeBfloat16Mul(const void *, const void *, void *);
#define ONNX_LIGHT_CPU_DECLARE_HALF_BULK(NAME)                                                     \
  void BulkFloat16##NAME(const void *, const void *, void *, std::size_t);                         \
  void BulkFloat16##NAME##Left(const void *, const void *, void *, std::size_t);                   \
  void BulkFloat16##NAME##Right(const void *, const void *, void *, std::size_t);                  \
  void BulkBfloat16##NAME(const void *, const void *, void *, std::size_t);                        \
  void BulkBfloat16##NAME##Left(const void *, const void *, void *, std::size_t);                  \
  void BulkBfloat16##NAME##Right(const void *, const void *, void *, std::size_t);
ONNX_LIGHT_CPU_DECLARE_HALF_BULK(Add)
ONNX_LIGHT_CPU_DECLARE_HALF_BULK(Sub)
ONNX_LIGHT_CPU_DECLARE_HALF_BULK(Mul)
ONNX_LIGHT_CPU_DECLARE_HALF_BULK(Div)
ONNX_LIGHT_CPU_DECLARE_HALF_BULK(PRelu)
#undef ONNX_LIGHT_CPU_DECLARE_HALF_BULK
void BulkFloat16Mod(const void *, const void *, void *, std::size_t);
void BulkBfloat16Mod(const void *, const void *, void *, std::size_t);
void BulkFloat16Pow(const void *, const void *, void *, std::size_t);
void BulkFloat16PowLeft(const void *, const void *, void *, std::size_t);
void BulkFloat16PowRight(const void *, const void *, void *, std::size_t);
void BulkBfloat16Pow(const void *, const void *, void *, std::size_t);
void BulkBfloat16PowLeft(const void *, const void *, void *, std::size_t);
void BulkBfloat16PowRight(const void *, const void *, void *, std::size_t);
void BulkFloat16PRelu(const void *, const void *, void *, std::size_t);
void BulkBfloat16PRelu(const void *, const void *, void *, std::size_t);

void SelectBulk(BinaryOperator op, DT left, BinaryKernelDescriptor::Adapter &adapter) {
#define ONNX_LIGHT_CPU_BIND_BULK(STEM, T)                                                          \
  adapter.bulk_contiguous = &BulkContiguousWrapper<T, &STEM##Contiguous>;                          \
  adapter.bulk_left_scalar = &BulkLeftScalarWrapper<T, &STEM##LeftScalar>;                         \
  adapter.bulk_right_scalar = &BulkRightScalarWrapper<T, &STEM##RightScalar>;
#define ONNX_LIGHT_CPU_BIND_COMPUTE_BULK(T, COMPUTE)                                               \
  adapter.bulk_contiguous = &BulkComputeContiguous<T, T, T, &COMPUTE>;                             \
  adapter.bulk_left_scalar = &BulkComputeLeftScalar<T, T, T, &COMPUTE>;                            \
  adapter.bulk_right_scalar = &BulkComputeRightScalar<T, T, T, &COMPUTE>;
#define ONNX_LIGHT_CPU_BIND_HALF_BULK(PREFIX, NAME)                                                \
  adapter.bulk_contiguous = &Bulk##PREFIX##NAME;                                                   \
  adapter.bulk_left_scalar = &Bulk##PREFIX##NAME##Left;                                            \
  adapter.bulk_right_scalar = &Bulk##PREFIX##NAME##Right;
#define ONNX_LIGHT_CPU_BIND_INTEGER_BULK(COMPUTE)                                                  \
  switch (left) {                                                                                  \
  case DT::INT8:                                                                                   \
    ONNX_LIGHT_CPU_BIND_COMPUTE_BULK(std::int8_t, COMPUTE<std::int8_t>)                            \
    break;                                                                                         \
  case DT::INT16:                                                                                  \
    ONNX_LIGHT_CPU_BIND_COMPUTE_BULK(std::int16_t, COMPUTE<std::int16_t>)                          \
    break;                                                                                         \
  case DT::INT32:                                                                                  \
    ONNX_LIGHT_CPU_BIND_COMPUTE_BULK(std::int32_t, COMPUTE<std::int32_t>)                          \
    break;                                                                                         \
  case DT::INT64:                                                                                  \
    ONNX_LIGHT_CPU_BIND_COMPUTE_BULK(std::int64_t, COMPUTE<std::int64_t>)                          \
    break;                                                                                         \
  case DT::UINT8:                                                                                  \
    ONNX_LIGHT_CPU_BIND_COMPUTE_BULK(std::uint8_t, COMPUTE<std::uint8_t>)                          \
    break;                                                                                         \
  case DT::UINT16:                                                                                 \
    ONNX_LIGHT_CPU_BIND_COMPUTE_BULK(std::uint16_t, COMPUTE<std::uint16_t>)                        \
    break;                                                                                         \
  case DT::UINT32:                                                                                 \
    ONNX_LIGHT_CPU_BIND_COMPUTE_BULK(std::uint32_t, COMPUTE<std::uint32_t>)                        \
    break;                                                                                         \
  case DT::UINT64:                                                                                 \
    ONNX_LIGHT_CPU_BIND_COMPUTE_BULK(std::uint64_t, COMPUTE<std::uint64_t>)                        \
    break;                                                                                         \
  default:                                                                                         \
    break;                                                                                         \
  }
  switch (op) {
  case BinaryOperator::kAdd:
    if (left == DT::FLOAT) {
      ONNX_LIGHT_CPU_BIND_BULK(BinaryAddFloat32, float)
    } else if (left == DT::DOUBLE) {
      ONNX_LIGHT_CPU_BIND_BULK(BinaryAddFloat64, double)
    } else if (left == DT::FLOAT16) {
      ONNX_LIGHT_CPU_BIND_HALF_BULK(Float16, Add)
    } else if (left == DT::BFLOAT16) {
      ONNX_LIGHT_CPU_BIND_HALF_BULK(Bfloat16, Add)
    } else {
      ONNX_LIGHT_CPU_BIND_INTEGER_BULK(ComputeAdd)
    }
    break;
  case BinaryOperator::kSub:
    if (left == DT::FLOAT) {
      ONNX_LIGHT_CPU_BIND_BULK(BinarySubFloat32, float)
    } else if (left == DT::DOUBLE) {
      ONNX_LIGHT_CPU_BIND_BULK(BinarySubFloat64, double)
    } else if (left == DT::FLOAT16) {
      ONNX_LIGHT_CPU_BIND_HALF_BULK(Float16, Sub)
    } else if (left == DT::BFLOAT16) {
      ONNX_LIGHT_CPU_BIND_HALF_BULK(Bfloat16, Sub)
    } else {
      ONNX_LIGHT_CPU_BIND_INTEGER_BULK(ComputeSub)
    }
    break;
  case BinaryOperator::kMul:
    if (left == DT::FLOAT) {
      ONNX_LIGHT_CPU_BIND_BULK(BinaryMulFloat32, float)
    } else if (left == DT::DOUBLE) {
      ONNX_LIGHT_CPU_BIND_BULK(BinaryMulFloat64, double)
    } else if (left == DT::FLOAT16) {
      ONNX_LIGHT_CPU_BIND_HALF_BULK(Float16, Mul)
    } else if (left == DT::BFLOAT16) {
      ONNX_LIGHT_CPU_BIND_HALF_BULK(Bfloat16, Mul)
    } else {
      ONNX_LIGHT_CPU_BIND_INTEGER_BULK(ComputeMul)
    }
    break;
  case BinaryOperator::kDiv:
    if (left == DT::FLOAT) {
      ONNX_LIGHT_CPU_BIND_BULK(BinaryDivFloat32, float)
    } else if (left == DT::DOUBLE) {
      ONNX_LIGHT_CPU_BIND_BULK(BinaryDivFloat64, double)
    } else if (left == DT::FLOAT16) {
      ONNX_LIGHT_CPU_BIND_HALF_BULK(Float16, Div)
    } else if (left == DT::BFLOAT16) {
      ONNX_LIGHT_CPU_BIND_HALF_BULK(Bfloat16, Div)
    }
    break;
  case BinaryOperator::kMod:
    adapter.bulk_contiguous =
        left == DT::FLOAT16 ? &BulkFloat16Mod : (left == DT::BFLOAT16 ? &BulkBfloat16Mod : nullptr);
    break;
  case BinaryOperator::kPow:
    adapter.bulk_contiguous =
        left == DT::FLOAT16 ? &BulkFloat16Pow : (left == DT::BFLOAT16 ? &BulkBfloat16Pow : nullptr);
    break;
  case BinaryOperator::kPRelu:
    if (left == DT::FLOAT16) {
      ONNX_LIGHT_CPU_BIND_HALF_BULK(Float16, PRelu)
    } else if (left == DT::BFLOAT16) {
      ONNX_LIGHT_CPU_BIND_HALF_BULK(Bfloat16, PRelu)
    }
    break;
  default:
    break;
  }
#undef ONNX_LIGHT_CPU_BIND_INTEGER_BULK
#undef ONNX_LIGHT_CPU_BIND_COMPUTE_BULK
#undef ONNX_LIGHT_CPU_BIND_HALF_BULK
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

template <typename T, bool Fmod>
void ComputeModIntUnchecked(const void *left, const void *right, void *out) {
  const T lhs = ReadTyped<T>(left);
  const T rhs = ReadTyped<T>(right);
  WriteTyped<T>(out, Fmod ? static_cast<T>(lhs % rhs) : PythonMod(lhs, rhs));
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
  } else if constexpr (std::is_integral_v<TExp>) {
    WriteTyped<TBase>(out, PowIntegerExponent(lhs, rhs));
  } else {
    WriteTyped<TBase>(out, std::pow(lhs, static_cast<TBase>(rhs)));
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

template <typename T, bool Left>
void ComputeBitShiftUnchecked(const void *left, const void *right, void *out) {
  const T value = ReadTyped<T>(left);
  const unsigned shift = static_cast<unsigned>(ReadTyped<T>(right));
  WriteTyped<T>(out, Left ? static_cast<T>(value << shift) : static_cast<T>(value >> shift));
}

template <typename T> void ValidateBitShift(const void *, const void *right) {
  if (ReadTyped<T>(right) >= static_cast<T>(sizeof(T) * 8)) {
    throw std::invalid_argument("onnx_light_cpu::BitShift: shift amount is out of range.");
  }
}

template <typename T> void ComputePRelu(const void *left, const void *right, void *out) {
  const T x = ReadTyped<T>(left);
  const T slope = ReadTyped<T>(right);
  if constexpr (std::is_integral_v<T>) {
    WriteTyped<T>(out, x < static_cast<T>(0) ? WrapMul(x, slope) : x);
  } else {
    WriteTyped<T>(out, x < static_cast<T>(0) ? static_cast<T>(x * slope) : x);
  }
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

struct HalfAdd {
  float operator()(float a, float b) const { return a + b; }
};
struct HalfSub {
  float operator()(float a, float b) const { return a - b; }
};
struct HalfMul {
  float operator()(float a, float b) const { return a * b; }
};
struct HalfDiv {
  float operator()(float a, float b) const { return a / b; }
};
struct HalfPRelu {
  float operator()(float a, float b) const { return a < 0.0f ? a * b : a; }
};

template <auto DecodeOne, auto Decode, auto Encode, typename Op>
void BulkHalf(const void *left, const void *right, void *out, std::size_t count, bool left_scalar,
              bool right_scalar, Op op) {
  constexpr std::size_t kBlockSize = 1024;
  const auto *a = static_cast<const std::uint16_t *>(left);
  const auto *b = static_cast<const std::uint16_t *>(right);
  auto *y = static_cast<std::uint16_t *>(out);
  alignas(32) float af[kBlockSize];
  alignas(32) float bf[kBlockSize];
  alignas(32) float yf[kBlockSize];
  const float scalar_a = left_scalar ? DecodeOne(*a) : 0.0f;
  const float scalar_b = right_scalar ? DecodeOne(*b) : 0.0f;
  for (std::size_t offset = 0; offset < count; offset += kBlockSize) {
    const std::size_t block = std::min(kBlockSize, count - offset);
    if (!left_scalar) {
      Decode(a + offset, af, block);
    }
    if (!right_scalar) {
      Decode(b + offset, bf, block);
    }
    for (std::size_t i = 0; i < block; ++i) {
      yf[i] = op(left_scalar ? scalar_a : af[i], right_scalar ? scalar_b : bf[i]);
    }
    Encode(yf, y + offset, block);
  }
}

#define ONNX_LIGHT_CPU_DEFINE_HALF_BULK(PREFIX, NAME, DECODE_ONE, DECODE, ENCODE, OP)              \
  void Bulk##PREFIX##NAME(const void *left, const void *right, void *out, std::size_t count) {     \
    BulkHalf<DECODE_ONE, DECODE, ENCODE>(left, right, out, count, false, false, OP{});             \
  }                                                                                                \
  void Bulk##PREFIX##NAME##Left(const void *left, const void *right, void *out,                    \
                                std::size_t count) {                                               \
    BulkHalf<DECODE_ONE, DECODE, ENCODE>(left, right, out, count, true, false, OP{});              \
  }                                                                                                \
  void Bulk##PREFIX##NAME##Right(const void *left, const void *right, void *out,                   \
                                 std::size_t count) {                                              \
    BulkHalf<DECODE_ONE, DECODE, ENCODE>(left, right, out, count, false, true, OP{});              \
  }

#define ONNX_LIGHT_CPU_DEFINE_HALF_OPERATION(NAME, OP)                                             \
  ONNX_LIGHT_CPU_DEFINE_HALF_BULK(Float16, NAME, detail::Float16BitsToFloat,                       \
                                  detail::ConvertFloat16ToFloat32,                                 \
                                  detail::ConvertFloat32ToFloat16, OP)                             \
  ONNX_LIGHT_CPU_DEFINE_HALF_BULK(Bfloat16, NAME, detail::Bfloat16BitsToFloat,                     \
                                  detail::ConvertBFloat16ToFloat32,                                \
                                  detail::ConvertFloat32ToBFloat16, OP)

ONNX_LIGHT_CPU_DEFINE_HALF_OPERATION(Add, HalfAdd)
ONNX_LIGHT_CPU_DEFINE_HALF_OPERATION(Sub, HalfSub)
ONNX_LIGHT_CPU_DEFINE_HALF_OPERATION(Mul, HalfMul)
ONNX_LIGHT_CPU_DEFINE_HALF_OPERATION(Div, HalfDiv)
ONNX_LIGHT_CPU_DEFINE_HALF_OPERATION(PRelu, HalfPRelu)

#undef ONNX_LIGHT_CPU_DEFINE_HALF_OPERATION
#undef ONNX_LIGHT_CPU_DEFINE_HALF_BULK

template <auto Decode, auto Encode, typename Op>
void BulkHalfContiguous(const void *left, const void *right, void *out, std::size_t count, Op op) {
  constexpr std::size_t kBlockSize = 1024;
  const auto *a = static_cast<const std::uint16_t *>(left);
  const auto *b = static_cast<const std::uint16_t *>(right);
  auto *y = static_cast<std::uint16_t *>(out);
  alignas(32) float af[kBlockSize];
  alignas(32) float bf[kBlockSize];
  alignas(32) float yf[kBlockSize];
  for (std::size_t offset = 0; offset < count; offset += kBlockSize) {
    const std::size_t block = std::min(kBlockSize, count - offset);
    Decode(a + offset, af, block);
    Decode(b + offset, bf, block);
    for (std::size_t i = 0; i < block; ++i) {
      yf[i] = op(af[i], bf[i]);
    }
    Encode(yf, y + offset, block);
  }
}

void BulkFloat16Mod(const void *left, const void *right, void *out, std::size_t count) {
  BulkHalfContiguous<detail::ConvertFloat16ToFloat32, detail::ConvertFloat32ToFloat16>(
      left, right, out, count, [](float a, float b) { return std::fmod(a, b); });
}

void BulkBfloat16Mod(const void *left, const void *right, void *out, std::size_t count) {
  BulkHalfContiguous<detail::ConvertBFloat16ToFloat32, detail::ConvertFloat32ToBFloat16>(
      left, right, out, count, [](float a, float b) { return std::fmod(a, b); });
}

void BulkFloat16Pow(const void *left, const void *right, void *out, std::size_t count) {
  constexpr std::size_t kBlockSize = 1024;
  const auto *bases = static_cast<const std::uint16_t *>(left);
  const auto *exponents = static_cast<const std::uint16_t *>(right);
  auto *output = static_cast<std::uint16_t *>(out);
  alignas(32) float base_f32[kBlockSize];
  alignas(32) float exponent_f32[kBlockSize];
  alignas(32) float output_f32[kBlockSize];
  for (std::size_t offset = 0; offset < count; offset += kBlockSize) {
    const std::size_t block = std::min(kBlockSize, count - offset);
    detail::ConvertFloat16ToFloat32(bases + offset, base_f32, block);
    detail::ConvertFloat16ToFloat32(exponents + offset, exponent_f32, block);
    EvaluateFloatPowBlock(base_f32, exponent_f32, output_f32, block);
    detail::ConvertFloat32ToFloat16(output_f32, output + offset, block);
  }
}

void BulkBfloat16Pow(const void *left, const void *right, void *out, std::size_t count) {
  BulkHalfContiguous<detail::ConvertBFloat16ToFloat32, detail::ConvertFloat32ToBFloat16>(
      left, right, out, count, [](float a, float b) { return FastFloatPower(a, b); });
}

template <auto DecodeOne, auto Decode, auto Encode>
void BulkHalfPowLeft(const void *left, const void *right, void *out, std::size_t count) {
  constexpr std::size_t kBlockSize = 1024;
  const float base = DecodeOne(*static_cast<const std::uint16_t *>(left));
  const auto *exponents = static_cast<const std::uint16_t *>(right);
  auto *output = static_cast<std::uint16_t *>(out);
  alignas(32) float exponent_f32[kBlockSize];
  alignas(32) float output_f32[kBlockSize];
  for (std::size_t offset = 0; offset < count; offset += kBlockSize) {
    const std::size_t block = std::min(kBlockSize, count - offset);
    Decode(exponents + offset, exponent_f32, block);
    EvaluateFloatPowLeftScalarBlock(base, exponent_f32, output_f32, block);
    Encode(output_f32, output + offset, block);
  }
}

template <auto DecodeOne, auto Decode, auto Encode>
void BulkHalfPowRight(const void *left, const void *right, void *out, std::size_t count) {
  constexpr std::size_t kBlockSize = 1024;
  const auto *bases = static_cast<const std::uint16_t *>(left);
  const float exponent = DecodeOne(*static_cast<const std::uint16_t *>(right));
  auto *output = static_cast<std::uint16_t *>(out);
  if (exponent == 1.0f) {
    std::copy_n(bases, count, output);
    return;
  }
  alignas(32) float base_f32[kBlockSize];
  alignas(32) float output_f32[kBlockSize];
  for (std::size_t offset = 0; offset < count; offset += kBlockSize) {
    const std::size_t block = std::min(kBlockSize, count - offset);
    Decode(bases + offset, base_f32, block);
    EvaluateFloatPowRightScalarBlock(base_f32, exponent, output_f32, block);
    Encode(output_f32, output + offset, block);
  }
}

void BulkFloat16PowLeft(const void *left, const void *right, void *out, std::size_t count) {
  BulkHalfPowLeft<detail::Float16BitsToFloat, detail::ConvertFloat16ToFloat32,
                  detail::ConvertFloat32ToFloat16>(left, right, out, count);
}

void BulkFloat16PowRight(const void *left, const void *right, void *out, std::size_t count) {
  BulkHalfPowRight<detail::Float16BitsToFloat, detail::ConvertFloat16ToFloat32,
                   detail::ConvertFloat32ToFloat16>(left, right, out, count);
}

void BulkBfloat16PowLeft(const void *left, const void *right, void *out, std::size_t count) {
  BulkHalfPowLeft<detail::Bfloat16BitsToFloat, detail::ConvertBFloat16ToFloat32,
                  detail::ConvertFloat32ToBFloat16>(left, right, out, count);
}

void BulkBfloat16PowRight(const void *left, const void *right, void *out, std::size_t count) {
  BulkHalfPowRight<detail::Bfloat16BitsToFloat, detail::ConvertBFloat16ToFloat32,
                   detail::ConvertFloat32ToBFloat16>(left, right, out, count);
}

template <typename TExp, auto DecodeExponent, bool IntegerExponent>
float EvaluateHalfPow(float base, TExp exponent) {
  if constexpr (IntegerExponent) {
    if (exponent == static_cast<TExp>(0)) {
      return 1.0f;
    }
    if (exponent == static_cast<TExp>(1)) {
      return base;
    }
    if (exponent == static_cast<TExp>(2)) {
      return FastFloatIntegerPower<2>(base);
    }
    if (exponent == static_cast<TExp>(3)) {
      return FastFloatIntegerPower<3>(base);
    }
    if (exponent == static_cast<TExp>(4)) {
      return FastFloatIntegerPower<4>(base);
    }
    if (exponent == static_cast<TExp>(5)) {
      return FastFloatIntegerPower<5>(base);
    }
    return PowIntegerExponent(base, exponent);
  } else {
    return FastFloatPower(base, DecodeExponent(exponent));
  }
}

template <int Exponent>
void EvaluateFixedIntegerPowBlock(const float *base, float *output, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = FastFloatIntegerPower<Exponent>(base[i]);
  }
}

template <typename TExp, auto DecodeBase, auto DecodeBaseBlock, auto Encode, auto DecodeExponent,
          bool IntegerExponent>
void BulkMixedHalfPow(const void *left, const void *right, void *out, std::size_t count,
                      bool left_scalar, bool right_scalar) {
  constexpr std::size_t kBlockSize = 1024;
  const auto *bases = static_cast<const std::uint16_t *>(left);
  const auto *exponents = static_cast<const TExp *>(right);
  auto *output = static_cast<std::uint16_t *>(out);
  const float scalar_base = left_scalar ? DecodeBase(*bases) : 0.0f;
  const TExp scalar_exponent = right_scalar ? *exponents : TExp{};
  if constexpr (IntegerExponent) {
    if (right_scalar && scalar_exponent == static_cast<TExp>(1)) {
      std::copy_n(bases, count, output);
      return;
    }
  }
  alignas(32) float base_f32[kBlockSize];
  alignas(32) float output_f32[kBlockSize];
  for (std::size_t offset = 0; offset < count; offset += kBlockSize) {
    const std::size_t block = std::min(kBlockSize, count - offset);
    if (!left_scalar) {
      DecodeBaseBlock(bases + offset, base_f32, block);
    }
    if constexpr (!IntegerExponent && std::is_same_v<TExp, float>) {
      if (!right_scalar) {
        if (left_scalar) {
          EvaluateFloatPowLeftScalarBlock(scalar_base, exponents + offset, output_f32, block);
        } else {
          EvaluateFloatPowBlock(base_f32, exponents + offset, output_f32, block);
        }
        Encode(output_f32, output + offset, block);
        continue;
      }
    }
    if constexpr (!IntegerExponent) {
      if (right_scalar && !left_scalar) {
        EvaluateFloatPowRightScalarBlock(base_f32, DecodeExponent(scalar_exponent), output_f32,
                                         block);
        Encode(output_f32, output + offset, block);
        continue;
      }
    }
    if constexpr (IntegerExponent) {
      if (right_scalar && !left_scalar) {
        if (scalar_exponent == static_cast<TExp>(2)) {
          EvaluateFixedIntegerPowBlock<2>(base_f32, output_f32, block);
          Encode(output_f32, output + offset, block);
          continue;
        }
        if (scalar_exponent == static_cast<TExp>(3)) {
          EvaluateFixedIntegerPowBlock<3>(base_f32, output_f32, block);
          Encode(output_f32, output + offset, block);
          continue;
        }
        if (scalar_exponent == static_cast<TExp>(4)) {
          EvaluateFixedIntegerPowBlock<4>(base_f32, output_f32, block);
          Encode(output_f32, output + offset, block);
          continue;
        }
        if (scalar_exponent == static_cast<TExp>(5)) {
          EvaluateFixedIntegerPowBlock<5>(base_f32, output_f32, block);
          Encode(output_f32, output + offset, block);
          continue;
        }
      }
    }
    for (std::size_t i = 0; i < block; ++i) {
      output_f32[i] = EvaluateHalfPow<TExp, DecodeExponent, IntegerExponent>(
          left_scalar ? scalar_base : base_f32[i],
          right_scalar ? scalar_exponent : exponents[offset + i]);
    }
    Encode(output_f32, output + offset, block);
  }
}

template <typename TExp, auto DecodeBase, auto DecodeBaseBlock, auto Encode, auto DecodeExponent,
          bool IntegerExponent>
void BulkMixedHalfPowContiguous(const void *left, const void *right, void *out, std::size_t count) {
  BulkMixedHalfPow<TExp, DecodeBase, DecodeBaseBlock, Encode, DecodeExponent, IntegerExponent>(
      left, right, out, count, false, false);
}

template <typename TExp, auto DecodeBase, auto DecodeBaseBlock, auto Encode, auto DecodeExponent,
          bool IntegerExponent>
void BulkMixedHalfPowLeft(const void *left, const void *right, void *out, std::size_t count) {
  BulkMixedHalfPow<TExp, DecodeBase, DecodeBaseBlock, Encode, DecodeExponent, IntegerExponent>(
      left, right, out, count, true, false);
}

template <typename TExp, auto DecodeBase, auto DecodeBaseBlock, auto Encode, auto DecodeExponent,
          bool IntegerExponent>
void BulkMixedHalfPowRight(const void *left, const void *right, void *out, std::size_t count) {
  BulkMixedHalfPow<TExp, DecodeBase, DecodeBaseBlock, Encode, DecodeExponent, IntegerExponent>(
      left, right, out, count, false, true);
}

template <typename TExp, auto DecodeBase, auto Encode, auto DecodeExponent>
void ComputeHalfPow(const void *left, const void *right, void *out) {
  const float base = DecodeBase(ReadTyped<std::uint16_t>(left));
  const float exponent = DecodeExponent(ReadTyped<TExp>(right));
  WriteTyped<std::uint16_t>(out, Encode(static_cast<float>(std::pow(base, exponent))));
}

template <typename TExp, auto DecodeBase, auto Encode>
void ComputeHalfIntegerPow(const void *left, const void *right, void *out) {
  const float base = DecodeBase(ReadTyped<std::uint16_t>(left));
  WriteTyped<std::uint16_t>(out, Encode(PowIntegerExponent(base, ReadTyped<TExp>(right))));
}

template <typename T> float CastExponent(T value) { return static_cast<float>(value); }

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
        return &ComputeHalfPow<float, detail::Float16BitsToFloat, detail::FloatToFloat16Bits,
                               CastExponent<float>>;
      case DT::FLOAT16:
        return &ComputeHalfPow<std::uint16_t, detail::Float16BitsToFloat,
                               detail::FloatToFloat16Bits, detail::Float16BitsToFloat>;
      case DT::BFLOAT16:
        return &ComputeHalfPow<std::uint16_t, detail::Float16BitsToFloat,
                               detail::FloatToFloat16Bits, detail::Bfloat16BitsToFloat>;
      case DT::INT32:
        return &ComputeHalfIntegerPow<std::int32_t, detail::Float16BitsToFloat,
                                      detail::FloatToFloat16Bits>;
      case DT::INT64:
        return &ComputeHalfIntegerPow<std::int64_t, detail::Float16BitsToFloat,
                                      detail::FloatToFloat16Bits>;
      case DT::UINT32:
        return &ComputeHalfIntegerPow<std::uint32_t, detail::Float16BitsToFloat,
                                      detail::FloatToFloat16Bits>;
      case DT::UINT64:
        return &ComputeHalfIntegerPow<std::uint64_t, detail::Float16BitsToFloat,
                                      detail::FloatToFloat16Bits>;
      default:
        break;
      }
    } else if (left == DT::BFLOAT16) {
      switch (right) {
      case DT::FLOAT:
        return &ComputeHalfPow<float, detail::Bfloat16BitsToFloat, detail::FloatToBFloat16Bits,
                               CastExponent<float>>;
      case DT::FLOAT16:
        return &ComputeHalfPow<std::uint16_t, detail::Bfloat16BitsToFloat,
                               detail::FloatToBFloat16Bits, detail::Float16BitsToFloat>;
      case DT::BFLOAT16:
        return &ComputeHalfPow<std::uint16_t, detail::Bfloat16BitsToFloat,
                               detail::FloatToBFloat16Bits, detail::Bfloat16BitsToFloat>;
      case DT::INT32:
        return &ComputeHalfIntegerPow<std::int32_t, detail::Bfloat16BitsToFloat,
                                      detail::FloatToBFloat16Bits>;
      case DT::INT64:
        return &ComputeHalfIntegerPow<std::int64_t, detail::Bfloat16BitsToFloat,
                                      detail::FloatToBFloat16Bits>;
      case DT::UINT32:
        return &ComputeHalfIntegerPow<std::uint32_t, detail::Bfloat16BitsToFloat,
                                      detail::FloatToBFloat16Bits>;
      case DT::UINT64:
        return &ComputeHalfIntegerPow<std::uint64_t, detail::Bfloat16BitsToFloat,
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

void SelectAdditionalBulk(BinaryOperator op, DT left, DT right, const Attrs &attributes,
                          BinaryKernelDescriptor::Adapter &adapter) {
#define ONNX_LIGHT_CPU_BIND_TYPED_BULK(TLEFT, TRIGHT, TOUT, ...)                                   \
  adapter.bulk_contiguous = &BulkComputeContiguous<TLEFT, TRIGHT, TOUT, &__VA_ARGS__>;             \
  adapter.bulk_left_scalar = &BulkComputeLeftScalar<TLEFT, TRIGHT, TOUT, &__VA_ARGS__>;            \
  adapter.bulk_right_scalar = &BulkComputeRightScalar<TLEFT, TRIGHT, TOUT, &__VA_ARGS__>;
#define ONNX_LIGHT_CPU_BIND_SAME_TYPE_CASE(TYPE, CPP_TYPE, COMPUTE)                                \
  case DT::TYPE:                                                                                   \
    ONNX_LIGHT_CPU_BIND_TYPED_BULK(CPP_TYPE, CPP_TYPE, CPP_TYPE, COMPUTE<CPP_TYPE>)                \
    break;
#define ONNX_LIGHT_CPU_BIND_INTEGER_TYPES(COMPUTE)                                                 \
  switch (left) {                                                                                  \
    ONNX_LIGHT_CPU_BIND_SAME_TYPE_CASE(INT8, std::int8_t, COMPUTE)                                 \
    ONNX_LIGHT_CPU_BIND_SAME_TYPE_CASE(INT16, std::int16_t, COMPUTE)                               \
    ONNX_LIGHT_CPU_BIND_SAME_TYPE_CASE(INT32, std::int32_t, COMPUTE)                               \
    ONNX_LIGHT_CPU_BIND_SAME_TYPE_CASE(INT64, std::int64_t, COMPUTE)                               \
    ONNX_LIGHT_CPU_BIND_SAME_TYPE_CASE(UINT8, std::uint8_t, COMPUTE)                               \
    ONNX_LIGHT_CPU_BIND_SAME_TYPE_CASE(UINT16, std::uint16_t, COMPUTE)                             \
    ONNX_LIGHT_CPU_BIND_SAME_TYPE_CASE(UINT32, std::uint32_t, COMPUTE)                             \
    ONNX_LIGHT_CPU_BIND_SAME_TYPE_CASE(UINT64, std::uint64_t, COMPUTE)                             \
  default:                                                                                         \
    break;                                                                                         \
  }
#define ONNX_LIGHT_CPU_BIND_COMPARE_CASE(TYPE, CPP_TYPE, COMPUTE)                                  \
  case DT::TYPE:                                                                                   \
    ONNX_LIGHT_CPU_BIND_TYPED_BULK(CPP_TYPE, CPP_TYPE, std::uint8_t, COMPUTE<CPP_TYPE>)            \
    break;
#define ONNX_LIGHT_CPU_BIND_COMPARE_TYPES(COMPUTE)                                                 \
  switch (left) {                                                                                  \
  case DT::FLOAT:                                                                                  \
    ONNX_LIGHT_CPU_BIND_TYPED_BULK(float, float, std::uint8_t, COMPUTE<float>)                     \
    break;                                                                                         \
  case DT::DOUBLE:                                                                                 \
    ONNX_LIGHT_CPU_BIND_TYPED_BULK(double, double, std::uint8_t, COMPUTE<double>)                  \
    break;                                                                                         \
    ONNX_LIGHT_CPU_BIND_COMPARE_CASE(INT8, std::int8_t, COMPUTE)                                   \
    ONNX_LIGHT_CPU_BIND_COMPARE_CASE(INT16, std::int16_t, COMPUTE)                                 \
    ONNX_LIGHT_CPU_BIND_COMPARE_CASE(INT32, std::int32_t, COMPUTE)                                 \
    ONNX_LIGHT_CPU_BIND_COMPARE_CASE(INT64, std::int64_t, COMPUTE)                                 \
    ONNX_LIGHT_CPU_BIND_COMPARE_CASE(UINT8, std::uint8_t, COMPUTE)                                 \
    ONNX_LIGHT_CPU_BIND_COMPARE_CASE(UINT16, std::uint16_t, COMPUTE)                               \
    ONNX_LIGHT_CPU_BIND_COMPARE_CASE(UINT32, std::uint32_t, COMPUTE)                               \
    ONNX_LIGHT_CPU_BIND_COMPARE_CASE(UINT64, std::uint64_t, COMPUTE)                               \
  default:                                                                                         \
    break;                                                                                         \
  }

  if (op == BinaryOperator::kPow && left != right) {
#define ONNX_LIGHT_CPU_BIND_MIXED_HALF_POW(TEXP, DECODE_BASE, DECODE_BASE_BLOCK, ENCODE,           \
                                           DECODE_EXPONENT, INTEGER_EXPONENT)                      \
  adapter.bulk_contiguous =                                                                        \
      &BulkMixedHalfPowContiguous<TEXP, DECODE_BASE, DECODE_BASE_BLOCK, ENCODE, DECODE_EXPONENT,   \
                                  INTEGER_EXPONENT>;                                               \
  adapter.bulk_left_scalar = &BulkMixedHalfPowLeft<TEXP, DECODE_BASE, DECODE_BASE_BLOCK, ENCODE,   \
                                                   DECODE_EXPONENT, INTEGER_EXPONENT>;             \
  adapter.bulk_right_scalar = &BulkMixedHalfPowRight<TEXP, DECODE_BASE, DECODE_BASE_BLOCK, ENCODE, \
                                                     DECODE_EXPONENT, INTEGER_EXPONENT>;
#define ONNX_LIGHT_CPU_BIND_POW_RIGHT_CASE(TYPE, RIGHT_CPP, BASE_CPP)                              \
  case DT::TYPE:                                                                                   \
    ONNX_LIGHT_CPU_BIND_TYPED_BULK(BASE_CPP, RIGHT_CPP, BASE_CPP, ComputePow<BASE_CPP, RIGHT_CPP>) \
    break;
    if (left == DT::FLOAT) {
      switch (right) {
        ONNX_LIGHT_CPU_BIND_POW_RIGHT_CASE(INT32, std::int32_t, float)
        ONNX_LIGHT_CPU_BIND_POW_RIGHT_CASE(INT64, std::int64_t, float)
        ONNX_LIGHT_CPU_BIND_POW_RIGHT_CASE(UINT32, std::uint32_t, float)
        ONNX_LIGHT_CPU_BIND_POW_RIGHT_CASE(UINT64, std::uint64_t, float)
      default:
        break;
      }
    } else if (left == DT::FLOAT16) {
      switch (right) {
      case DT::FLOAT:
        ONNX_LIGHT_CPU_BIND_MIXED_HALF_POW(
            float, detail::Float16BitsToFloat, detail::ConvertFloat16ToFloat32,
            detail::ConvertFloat32ToFloat16, CastExponent<float>, false)
        break;
      case DT::BFLOAT16:
        ONNX_LIGHT_CPU_BIND_MIXED_HALF_POW(
            std::uint16_t, detail::Float16BitsToFloat, detail::ConvertFloat16ToFloat32,
            detail::ConvertFloat32ToFloat16, detail::Bfloat16BitsToFloat, false)
        break;
      case DT::INT32:
        ONNX_LIGHT_CPU_BIND_MIXED_HALF_POW(
            std::int32_t, detail::Float16BitsToFloat, detail::ConvertFloat16ToFloat32,
            detail::ConvertFloat32ToFloat16, CastExponent<std::int32_t>, true)
        break;
      case DT::INT64:
        ONNX_LIGHT_CPU_BIND_MIXED_HALF_POW(
            std::int64_t, detail::Float16BitsToFloat, detail::ConvertFloat16ToFloat32,
            detail::ConvertFloat32ToFloat16, CastExponent<std::int64_t>, true)
        break;
      case DT::UINT32:
        ONNX_LIGHT_CPU_BIND_MIXED_HALF_POW(
            std::uint32_t, detail::Float16BitsToFloat, detail::ConvertFloat16ToFloat32,
            detail::ConvertFloat32ToFloat16, CastExponent<std::uint32_t>, true)
        break;
      case DT::UINT64:
        ONNX_LIGHT_CPU_BIND_MIXED_HALF_POW(
            std::uint64_t, detail::Float16BitsToFloat, detail::ConvertFloat16ToFloat32,
            detail::ConvertFloat32ToFloat16, CastExponent<std::uint64_t>, true)
        break;
      default:
        break;
      }
    } else if (left == DT::BFLOAT16) {
      switch (right) {
      case DT::FLOAT:
        ONNX_LIGHT_CPU_BIND_MIXED_HALF_POW(
            float, detail::Bfloat16BitsToFloat, detail::ConvertBFloat16ToFloat32,
            detail::ConvertFloat32ToBFloat16, CastExponent<float>, false)
        break;
      case DT::FLOAT16:
        ONNX_LIGHT_CPU_BIND_MIXED_HALF_POW(
            std::uint16_t, detail::Bfloat16BitsToFloat, detail::ConvertBFloat16ToFloat32,
            detail::ConvertFloat32ToBFloat16, detail::Float16BitsToFloat, false)
        break;
      case DT::INT32:
        ONNX_LIGHT_CPU_BIND_MIXED_HALF_POW(
            std::int32_t, detail::Bfloat16BitsToFloat, detail::ConvertBFloat16ToFloat32,
            detail::ConvertFloat32ToBFloat16, CastExponent<std::int32_t>, true)
        break;
      case DT::INT64:
        ONNX_LIGHT_CPU_BIND_MIXED_HALF_POW(
            std::int64_t, detail::Bfloat16BitsToFloat, detail::ConvertBFloat16ToFloat32,
            detail::ConvertFloat32ToBFloat16, CastExponent<std::int64_t>, true)
        break;
      case DT::UINT32:
        ONNX_LIGHT_CPU_BIND_MIXED_HALF_POW(
            std::uint32_t, detail::Bfloat16BitsToFloat, detail::ConvertBFloat16ToFloat32,
            detail::ConvertFloat32ToBFloat16, CastExponent<std::uint32_t>, true)
        break;
      case DT::UINT64:
        ONNX_LIGHT_CPU_BIND_MIXED_HALF_POW(
            std::uint64_t, detail::Bfloat16BitsToFloat, detail::ConvertBFloat16ToFloat32,
            detail::ConvertFloat32ToBFloat16, CastExponent<std::uint64_t>, true)
        break;
      default:
        break;
      }
    } else if (left == DT::INT32) {
      switch (right) {
        ONNX_LIGHT_CPU_BIND_POW_RIGHT_CASE(FLOAT, float, std::int32_t)
        ONNX_LIGHT_CPU_BIND_POW_RIGHT_CASE(INT64, std::int64_t, std::int32_t)
        ONNX_LIGHT_CPU_BIND_POW_RIGHT_CASE(UINT32, std::uint32_t, std::int32_t)
        ONNX_LIGHT_CPU_BIND_POW_RIGHT_CASE(UINT64, std::uint64_t, std::int32_t)
      default:
        break;
      }
    } else if (left == DT::INT64) {
      switch (right) {
        ONNX_LIGHT_CPU_BIND_POW_RIGHT_CASE(FLOAT, float, std::int64_t)
        ONNX_LIGHT_CPU_BIND_POW_RIGHT_CASE(INT32, std::int32_t, std::int64_t)
        ONNX_LIGHT_CPU_BIND_POW_RIGHT_CASE(UINT32, std::uint32_t, std::int64_t)
        ONNX_LIGHT_CPU_BIND_POW_RIGHT_CASE(UINT64, std::uint64_t, std::int64_t)
      default:
        break;
      }
    }
#undef ONNX_LIGHT_CPU_BIND_POW_RIGHT_CASE
#undef ONNX_LIGHT_CPU_BIND_MIXED_HALF_POW
    return;
  }
  if (left != right) {
    return;
  }
  switch (op) {
  case BinaryOperator::kDiv:
    ONNX_LIGHT_CPU_BIND_INTEGER_TYPES(ComputeDivUnchecked)
    break;
  case BinaryOperator::kMod:
#define ONNX_LIGHT_CPU_BIND_MOD_CASE(TYPE, CPP_TYPE)                                               \
  case DT::TYPE:                                                                                   \
    if (attributes.mod_fmod == 0) {                                                                \
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(CPP_TYPE, CPP_TYPE, CPP_TYPE,                                 \
                                     ComputeModIntUnchecked<CPP_TYPE, false>)                      \
    } else {                                                                                       \
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(CPP_TYPE, CPP_TYPE, CPP_TYPE,                                 \
                                     ComputeModIntUnchecked<CPP_TYPE, true>)                       \
    }                                                                                              \
    break;
    switch (left) {
      ONNX_LIGHT_CPU_BIND_MOD_CASE(INT8, std::int8_t)
      ONNX_LIGHT_CPU_BIND_MOD_CASE(INT16, std::int16_t)
      ONNX_LIGHT_CPU_BIND_MOD_CASE(INT32, std::int32_t)
      ONNX_LIGHT_CPU_BIND_MOD_CASE(INT64, std::int64_t)
      ONNX_LIGHT_CPU_BIND_MOD_CASE(UINT8, std::uint8_t)
      ONNX_LIGHT_CPU_BIND_MOD_CASE(UINT16, std::uint16_t)
      ONNX_LIGHT_CPU_BIND_MOD_CASE(UINT32, std::uint32_t)
      ONNX_LIGHT_CPU_BIND_MOD_CASE(UINT64, std::uint64_t)
    case DT::FLOAT:
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(float, float, float, ComputeModFloat<float>)
      break;
    case DT::DOUBLE:
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(double, double, double, ComputeModFloat<double>)
      break;
    case DT::FLOAT16:
      adapter.bulk_left_scalar =
          &BulkComputeLeftScalar<std::uint16_t, std::uint16_t, std::uint16_t, &ComputeFloat16Mod>;
      adapter.bulk_right_scalar =
          &BulkComputeRightScalar<std::uint16_t, std::uint16_t, std::uint16_t, &ComputeFloat16Mod>;
      break;
    case DT::BFLOAT16:
      adapter.bulk_left_scalar =
          &BulkComputeLeftScalar<std::uint16_t, std::uint16_t, std::uint16_t, &ComputeBfloat16Mod>;
      adapter.bulk_right_scalar =
          &BulkComputeRightScalar<std::uint16_t, std::uint16_t, std::uint16_t, &ComputeBfloat16Mod>;
      break;
    default:
      break;
    }
#undef ONNX_LIGHT_CPU_BIND_MOD_CASE
    break;
  case BinaryOperator::kPow:
    if (left == DT::FLOAT) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(float, float, float, ComputePow<float, float>)
      adapter.bulk_contiguous = &BulkFloatPow;
      adapter.bulk_left_scalar = &BulkFloatPowLeftScalar;
      adapter.bulk_right_scalar = &BulkFloatPowRightScalar;
    } else if (left == DT::DOUBLE) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(double, double, double, ComputePow<double, double>)
    } else if (left == DT::INT32) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(std::int32_t, std::int32_t, std::int32_t,
                                     ComputePow<std::int32_t, std::int32_t>)
    } else if (left == DT::INT64) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(std::int64_t, std::int64_t, std::int64_t,
                                     ComputePow<std::int64_t, std::int64_t>)
    } else if (left == DT::FLOAT16) {
      adapter.bulk_left_scalar = &BulkFloat16PowLeft;
      adapter.bulk_right_scalar = &BulkFloat16PowRight;
    } else if (left == DT::BFLOAT16) {
      adapter.bulk_left_scalar = &BulkBfloat16PowLeft;
      adapter.bulk_right_scalar = &BulkBfloat16PowRight;
    }
    break;
  case BinaryOperator::kEqual:
    ONNX_LIGHT_CPU_BIND_COMPARE_TYPES(ComputeEqual)
    if (left == DT::BOOL) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(std::uint8_t, std::uint8_t, std::uint8_t,
                                     ComputeEqual<std::uint8_t>)
    } else if (left == DT::FLOAT16) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(std::uint16_t, std::uint16_t, std::uint8_t,
                                     ComputeFloat16EqualBool)
    } else if (left == DT::BFLOAT16) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(std::uint16_t, std::uint16_t, std::uint8_t,
                                     ComputeBfloat16EqualBool)
    }
    break;
  case BinaryOperator::kGreater:
    ONNX_LIGHT_CPU_BIND_COMPARE_TYPES(ComputeGreater)
    if (left == DT::FLOAT16) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(std::uint16_t, std::uint16_t, std::uint8_t,
                                     ComputeFloat16Greater)
    } else if (left == DT::BFLOAT16) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(std::uint16_t, std::uint16_t, std::uint8_t,
                                     ComputeBfloat16Greater)
    }
    break;
  case BinaryOperator::kGreaterOrEqual:
    ONNX_LIGHT_CPU_BIND_COMPARE_TYPES(ComputeGreaterOrEqual)
    if (left == DT::FLOAT16) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(std::uint16_t, std::uint16_t, std::uint8_t,
                                     ComputeFloat16GreaterOrEqual)
    } else if (left == DT::BFLOAT16) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(std::uint16_t, std::uint16_t, std::uint8_t,
                                     ComputeBfloat16GreaterOrEqual)
    }
    break;
  case BinaryOperator::kLess:
    ONNX_LIGHT_CPU_BIND_COMPARE_TYPES(ComputeLess)
    if (left == DT::FLOAT16) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(std::uint16_t, std::uint16_t, std::uint8_t, ComputeFloat16Less)
    } else if (left == DT::BFLOAT16) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(std::uint16_t, std::uint16_t, std::uint8_t,
                                     ComputeBfloat16Less)
    }
    break;
  case BinaryOperator::kLessOrEqual:
    ONNX_LIGHT_CPU_BIND_COMPARE_TYPES(ComputeLessOrEqual)
    if (left == DT::FLOAT16) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(std::uint16_t, std::uint16_t, std::uint8_t,
                                     ComputeFloat16LessOrEqual)
    } else if (left == DT::BFLOAT16) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(std::uint16_t, std::uint16_t, std::uint8_t,
                                     ComputeBfloat16LessOrEqual)
    }
    break;
  case BinaryOperator::kAnd:
    adapter.bulk_contiguous = &BulkLogicalContiguous<'&'>;
    adapter.bulk_left_scalar = &BulkLogicalLeftScalar<'&'>;
    adapter.bulk_right_scalar = &BulkLogicalRightScalar<'&'>;
    break;
  case BinaryOperator::kOr:
    adapter.bulk_contiguous = &BulkLogicalContiguous<'|'>;
    adapter.bulk_left_scalar = &BulkLogicalLeftScalar<'|'>;
    adapter.bulk_right_scalar = &BulkLogicalRightScalar<'|'>;
    break;
  case BinaryOperator::kXor:
    adapter.bulk_contiguous = &BulkLogicalContiguous<'^'>;
    adapter.bulk_left_scalar = &BulkLogicalLeftScalar<'^'>;
    adapter.bulk_right_scalar = &BulkLogicalRightScalar<'^'>;
    break;
  case BinaryOperator::kBitwiseAnd:
    ONNX_LIGHT_CPU_BIND_INTEGER_TYPES(ComputeBitwiseAnd)
    break;
  case BinaryOperator::kBitwiseOr:
    ONNX_LIGHT_CPU_BIND_INTEGER_TYPES(ComputeBitwiseOr)
    break;
  case BinaryOperator::kBitwiseXor:
    ONNX_LIGHT_CPU_BIND_INTEGER_TYPES(ComputeBitwiseXor)
    break;
  case BinaryOperator::kBitShift:
#define ONNX_LIGHT_CPU_BIND_SHIFT_CASE(TYPE, CPP_TYPE)                                             \
  case DT::TYPE:                                                                                   \
    if (attributes.bitshift_direction == Attrs::BitShiftDirection::kLeft) {                        \
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(CPP_TYPE, CPP_TYPE, CPP_TYPE,                                 \
                                     ComputeBitShiftUnchecked<CPP_TYPE, true>)                     \
    } else {                                                                                       \
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(CPP_TYPE, CPP_TYPE, CPP_TYPE,                                 \
                                     ComputeBitShiftUnchecked<CPP_TYPE, false>)                    \
    }                                                                                              \
    break;
    switch (left) {
      ONNX_LIGHT_CPU_BIND_SHIFT_CASE(UINT8, std::uint8_t)
      ONNX_LIGHT_CPU_BIND_SHIFT_CASE(UINT16, std::uint16_t)
      ONNX_LIGHT_CPU_BIND_SHIFT_CASE(UINT32, std::uint32_t)
      ONNX_LIGHT_CPU_BIND_SHIFT_CASE(UINT64, std::uint64_t)
    default:
      break;
    }
#undef ONNX_LIGHT_CPU_BIND_SHIFT_CASE
    break;
  case BinaryOperator::kPRelu:
    if (left == DT::FLOAT) {
      adapter.bulk_contiguous = &BulkContiguousWrapper<float, &BinaryPReluFloat32Contiguous>;
      adapter.bulk_left_scalar = &BulkLeftScalarWrapper<float, &BinaryPReluFloat32LeftScalar>;
      adapter.bulk_right_scalar = &BulkRightScalarWrapper<float, &BinaryPReluFloat32RightScalar>;
    } else if (left == DT::DOUBLE) {
      ONNX_LIGHT_CPU_BIND_TYPED_BULK(double, double, double, ComputePRelu<double>)
    } else {
      ONNX_LIGHT_CPU_BIND_INTEGER_TYPES(ComputePRelu)
    }
    break;
  default:
    break;
  }
#undef ONNX_LIGHT_CPU_BIND_COMPARE_TYPES
#undef ONNX_LIGHT_CPU_BIND_COMPARE_CASE
#undef ONNX_LIGHT_CPU_BIND_INTEGER_TYPES
#undef ONNX_LIGHT_CPU_BIND_SAME_TYPE_CASE
#undef ONNX_LIGHT_CPU_BIND_TYPED_BULK
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
    SelectAdditionalBulk(manifest_.op, signature.left, signature.right, attributes_, adapter);
    adapters_.push_back(adapter);
  }
}

std::uint64_t BinaryKernelDescriptor::NextCacheIdentity() {
  static std::atomic<std::uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

DataType BinaryKernelDescriptor::ResolveOutputType(DataType left, DataType right) const {
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
BinaryKernelDescriptor::ResolveAdapter(DataType left, DataType right, DataType output) const {
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
