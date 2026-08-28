// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace onnx_light_cpu::normalization {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

inline bool IsSupportedFloatType(std::int32_t data_type) {
  const auto type = static_cast<rt_ns::DataType>(data_type);
  return type == rt_ns::DataType::FLOAT || type == rt_ns::DataType::DOUBLE ||
         type == rt_ns::DataType::FLOAT16 || type == rt_ns::DataType::BFLOAT16;
}

inline void RequireSupportedFloatType(const rt_ns::Tensor &tensor, std::string_view op,
                                      std::string_view role) {
  if (!IsSupportedFloatType(tensor.data_type)) {
    throw std::invalid_argument(std::string(op) + ": " + std::string(role) +
                                " must be FLOAT, DOUBLE, FLOAT16, or BFLOAT16.");
  }
}

class TensorReader {
public:
  TensorReader(const rt_ns::Tensor &tensor, std::string_view op, std::string_view role)
      : type_(static_cast<rt_ns::DataType>(tensor.data_type)), data_(tensor.bytes()) {
    RequireSupportedFloatType(tensor, op, role);
  }

  float LoadFloat(std::size_t index) const {
    switch (type_) {
    case rt_ns::DataType::FLOAT:
      return static_cast<const float *>(data_)[index];
    case rt_ns::DataType::DOUBLE:
      return static_cast<float>(static_cast<const double *>(data_)[index]);
    case rt_ns::DataType::FLOAT16:
      return detail::Float16BitsToFloat(static_cast<const std::uint16_t *>(data_)[index]);
    case rt_ns::DataType::BFLOAT16:
      return detail::Bfloat16BitsToFloat(static_cast<const std::uint16_t *>(data_)[index]);
    default:
      return 0.0F;
    }
  }

  double LoadDouble(std::size_t index) const {
    if (type_ == rt_ns::DataType::DOUBLE) {
      return static_cast<const double *>(data_)[index];
    }
    return static_cast<double>(LoadFloat(index));
  }

  rt_ns::DataType type() const noexcept { return type_; }

private:
  rt_ns::DataType type_;
  const void *data_;
};

class TensorWriter {
public:
  explicit TensorWriter(rt_ns::Tensor &tensor)
      : type_(static_cast<rt_ns::DataType>(tensor.data_type)), data_(tensor.mutable_bytes()) {}

  void StoreFloat(std::size_t index, float value) const {
    switch (type_) {
    case rt_ns::DataType::FLOAT:
      static_cast<float *>(data_)[index] = value;
      return;
    case rt_ns::DataType::DOUBLE:
      static_cast<double *>(data_)[index] = static_cast<double>(value);
      return;
    case rt_ns::DataType::FLOAT16:
      static_cast<std::uint16_t *>(data_)[index] = detail::FloatToFloat16Bits(value);
      return;
    case rt_ns::DataType::BFLOAT16:
      static_cast<std::uint16_t *>(data_)[index] = detail::FloatToBFloat16Bits(value);
      return;
    default:
      throw std::invalid_argument("normalization output has an unsupported data type.");
    }
  }

  void StoreDouble(std::size_t index, double value) const {
    if (type_ == rt_ns::DataType::DOUBLE) {
      static_cast<double *>(data_)[index] = value;
    } else {
      StoreFloat(index, static_cast<float>(value));
    }
  }

private:
  rt_ns::DataType type_;
  void *data_;
};

template <rt_ns::DataType Type> struct TypeTraits;

template <> struct TypeTraits<rt_ns::DataType::FLOAT> {
  using Storage = float;
  using Accumulator = float;

  static float Load(const float *data, std::size_t index) noexcept { return data[index]; }
  static void Store(float *data, std::size_t index, float value) noexcept { data[index] = value; }
};

template <> struct TypeTraits<rt_ns::DataType::DOUBLE> {
  using Storage = double;
  using Accumulator = double;

  static double Load(const double *data, std::size_t index) noexcept { return data[index]; }
  static void Store(double *data, std::size_t index, double value) noexcept { data[index] = value; }
};

template <> struct TypeTraits<rt_ns::DataType::FLOAT16> {
  using Storage = std::uint16_t;
  using Accumulator = float;

  static float Load(const std::uint16_t *data, std::size_t index) noexcept {
    return detail::Float16BitsToFloat(data[index]);
  }
  static void Store(std::uint16_t *data, std::size_t index, float value) noexcept {
    data[index] = detail::FloatToFloat16Bits(value);
  }
};

template <> struct TypeTraits<rt_ns::DataType::BFLOAT16> {
  using Storage = std::uint16_t;
  using Accumulator = float;

  static float Load(const std::uint16_t *data, std::size_t index) noexcept {
    return detail::Bfloat16BitsToFloat(data[index]);
  }
  static void Store(std::uint16_t *data, std::size_t index, float value) noexcept {
    data[index] = detail::FloatToBFloat16Bits(value);
  }
};

template <rt_ns::DataType Type> using StorageType = typename TypeTraits<Type>::Storage;
template <rt_ns::DataType Type> using AccumulatorType = typename TypeTraits<Type>::Accumulator;

template <rt_ns::DataType Type>
inline AccumulatorType<Type> RoundFloatToType(float value) noexcept {
  if constexpr (Type == rt_ns::DataType::DOUBLE) {
    return static_cast<double>(value);
  } else if constexpr (Type == rt_ns::DataType::FLOAT16) {
    return detail::Float16BitsToFloat(detail::FloatToFloat16Bits(value));
  } else if constexpr (Type == rt_ns::DataType::BFLOAT16) {
    return detail::Bfloat16BitsToFloat(detail::FloatToBFloat16Bits(value));
  } else {
    return value;
  }
}

template <rt_ns::DataType Type>
inline const StorageType<Type> *Data(const rt_ns::Tensor &tensor) noexcept {
  return reinterpret_cast<const StorageType<Type> *>(tensor.bytes());
}

template <rt_ns::DataType Type>
inline StorageType<Type> *MutableData(rt_ns::Tensor &tensor) noexcept {
  return reinterpret_cast<StorageType<Type> *>(tensor.mutable_bytes());
}

template <typename Fn> void DispatchFloatType(std::int32_t data_type, Fn &&fn) {
  switch (static_cast<rt_ns::DataType>(data_type)) {
  case rt_ns::DataType::FLOAT:
    std::forward<Fn>(fn).template operator()<rt_ns::DataType::FLOAT>();
    return;
  case rt_ns::DataType::DOUBLE:
    std::forward<Fn>(fn).template operator()<rt_ns::DataType::DOUBLE>();
    return;
  case rt_ns::DataType::FLOAT16:
    std::forward<Fn>(fn).template operator()<rt_ns::DataType::FLOAT16>();
    return;
  case rt_ns::DataType::BFLOAT16:
    std::forward<Fn>(fn).template operator()<rt_ns::DataType::BFLOAT16>();
    return;
  default:
    throw std::invalid_argument("normalization input has an unsupported data type.");
  }
}

template <typename T> struct Moments {
  T mean;
  T variance;
};

template <rt_ns::DataType Type>
Moments<AccumulatorType<Type>> ComputeContiguousMoments(const StorageType<Type> *data,
                                                        std::size_t count) {
  using Traits = TypeTraits<Type>;
  using Acc = AccumulatorType<Type>;
  if (count == 0) {
    throw std::invalid_argument("normalization reduction size must be positive.");
  }
  Acc sums[4] = {};
  std::size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    sums[0] += Traits::Load(data, i);
    sums[1] += Traits::Load(data, i + 1);
    sums[2] += Traits::Load(data, i + 2);
    sums[3] += Traits::Load(data, i + 3);
  }
  for (; i < count; ++i) {
    sums[i & 3] += Traits::Load(data, i);
  }
  const Acc mean = (sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<Acc>(count);
  Acc squared_sums[4] = {};
  i = 0;
  for (; i + 4 <= count; i += 4) {
    const Acc d0 = Traits::Load(data, i) - mean;
    const Acc d1 = Traits::Load(data, i + 1) - mean;
    const Acc d2 = Traits::Load(data, i + 2) - mean;
    const Acc d3 = Traits::Load(data, i + 3) - mean;
    squared_sums[0] += d0 * d0;
    squared_sums[1] += d1 * d1;
    squared_sums[2] += d2 * d2;
    squared_sums[3] += d3 * d3;
  }
  for (; i < count; ++i) {
    const Acc delta = Traits::Load(data, i) - mean;
    squared_sums[i & 3] += delta * delta;
  }
  const Acc variance = (squared_sums[0] + squared_sums[1] + squared_sums[2] + squared_sums[3]) /
                       static_cast<Acc>(count);
  return {mean, variance};
}

template <rt_ns::DataType Type>
Moments<float> ComputeContiguousFloatMoments(const StorageType<Type> *data, std::size_t count) {
  using Traits = TypeTraits<Type>;
  if (count == 0) {
    throw std::invalid_argument("normalization reduction size must be positive.");
  }
  float sums[4] = {};
  std::size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    sums[0] += static_cast<float>(Traits::Load(data, i));
    sums[1] += static_cast<float>(Traits::Load(data, i + 1));
    sums[2] += static_cast<float>(Traits::Load(data, i + 2));
    sums[3] += static_cast<float>(Traits::Load(data, i + 3));
  }
  for (; i < count; ++i) {
    sums[i & 3] += static_cast<float>(Traits::Load(data, i));
  }
  const float mean = (sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<float>(count);
  float squared_sums[4] = {};
  i = 0;
  for (; i + 4 <= count; i += 4) {
    const float d0 = static_cast<float>(Traits::Load(data, i)) - mean;
    const float d1 = static_cast<float>(Traits::Load(data, i + 1)) - mean;
    const float d2 = static_cast<float>(Traits::Load(data, i + 2)) - mean;
    const float d3 = static_cast<float>(Traits::Load(data, i + 3)) - mean;
    squared_sums[0] += d0 * d0;
    squared_sums[1] += d1 * d1;
    squared_sums[2] += d2 * d2;
    squared_sums[3] += d3 * d3;
  }
  for (; i < count; ++i) {
    const float delta = static_cast<float>(Traits::Load(data, i)) - mean;
    squared_sums[i & 3] += delta * delta;
  }
  const float variance = (squared_sums[0] + squared_sums[1] + squared_sums[2] + squared_sums[3]) /
                         static_cast<float>(count);
  return {mean, variance};
}

template <rt_ns::DataType Type>
AccumulatorType<Type> ComputeContiguousMeanSquare(const StorageType<Type> *data,
                                                  std::size_t count) {
  using Traits = TypeTraits<Type>;
  using Acc = AccumulatorType<Type>;
  if (count == 0) {
    throw std::invalid_argument("normalization reduction size must be positive.");
  }
  Acc sums[4] = {};
  std::size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    const Acc v0 = Traits::Load(data, i);
    const Acc v1 = Traits::Load(data, i + 1);
    const Acc v2 = Traits::Load(data, i + 2);
    const Acc v3 = Traits::Load(data, i + 3);
    sums[0] += v0 * v0;
    sums[1] += v1 * v1;
    sums[2] += v2 * v2;
    sums[3] += v3 * v3;
  }
  for (; i < count; ++i) {
    const Acc value = Traits::Load(data, i);
    sums[i & 3] += value * value;
  }
  return (sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<Acc>(count);
}

template <rt_ns::DataType Type>
float ComputeContiguousFloatMeanSquare(const StorageType<Type> *data, std::size_t count) {
  using Traits = TypeTraits<Type>;
  if (count == 0) {
    throw std::invalid_argument("normalization reduction size must be positive.");
  }
  float sums[4] = {};
  std::size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    const float v0 = static_cast<float>(Traits::Load(data, i));
    const float v1 = static_cast<float>(Traits::Load(data, i + 1));
    const float v2 = static_cast<float>(Traits::Load(data, i + 2));
    const float v3 = static_cast<float>(Traits::Load(data, i + 3));
    sums[0] += v0 * v0;
    sums[1] += v1 * v1;
    sums[2] += v2 * v2;
    sums[3] += v3 * v3;
  }
  for (; i < count; ++i) {
    const float value = static_cast<float>(Traits::Load(data, i));
    sums[i & 3] += value * value;
  }
  return (sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<float>(count);
}

inline void ApplyAffineFloatRange(const TensorReader &input, const TensorWriter &output,
                                  std::size_t base, std::size_t count, float multiplier,
                                  float offset) {
  for (std::size_t i = 0; i < count; ++i) {
    output.StoreFloat(base + i, input.LoadFloat(base + i) * multiplier + offset);
  }
}

inline void ApplyAffineDoubleRange(const TensorReader &input, const TensorWriter &output,
                                   std::size_t base, std::size_t count, double multiplier,
                                   double offset) {
  for (std::size_t i = 0; i < count; ++i) {
    output.StoreDouble(base + i, input.LoadDouble(base + i) * multiplier + offset);
  }
}

inline std::size_t ElementSize(std::int32_t data_type) {
  switch (static_cast<rt_ns::DataType>(data_type)) {
  case rt_ns::DataType::FLOAT:
    return sizeof(float);
  case rt_ns::DataType::DOUBLE:
    return sizeof(double);
  case rt_ns::DataType::FLOAT16:
  case rt_ns::DataType::BFLOAT16:
    return sizeof(std::uint16_t);
  default:
    throw std::invalid_argument("normalization output has an unsupported data type.");
  }
}

inline std::size_t Product(const rt_ns::Shape &shape, std::size_t begin, std::size_t end,
                           std::string_view op) {
  if (begin > end || end > shape.size()) {
    throw std::invalid_argument(std::string(op) + ": invalid shape range.");
  }
  std::size_t product = 1;
  for (std::size_t i = begin; i < end; ++i) {
    const std::int64_t dimension = shape[i];
    if (dimension < 0) {
      throw std::invalid_argument(std::string(op) + ": dimensions must be non-negative.");
    }
    const std::size_t value = static_cast<std::size_t>(dimension);
    if (value != 0 && product > std::numeric_limits<std::size_t>::max() / value) {
      throw std::invalid_argument(std::string(op) + ": shape product overflows size_t.");
    }
    product *= value;
  }
  return product;
}

inline std::int64_t NormalizeAxis(std::int64_t axis, std::size_t rank, std::string_view op) {
  const std::int64_t signed_rank = static_cast<std::int64_t>(rank);
  if (axis < 0) {
    axis += signed_rank;
  }
  if (axis < 0 || axis >= signed_rank) {
    throw std::invalid_argument(std::string(op) + ": axis is out of range.");
  }
  return axis;
}

inline rt_ns::Tensor AllocateOutput(std::int32_t data_type, const rt_ns::Shape &shape, int slot,
                                    rt_ns::RuntimeContext *rt) {
  const std::size_t count = Product(shape, 0, shape.size(), "normalization output");
  const std::size_t element_size = ElementSize(data_type);
  if (count > std::numeric_limits<std::size_t>::max() / element_size) {
    throw std::invalid_argument("normalization output byte size overflows size_t.");
  }
  const std::size_t bytes = count * element_size;
  return rt != nullptr ? rt->MakeOutputTensor(slot, data_type, shape, bytes)
                       : rt_ns::MakeOutputTensor(data_type, shape, bytes, nullptr);
}

inline void RequireSameType(const rt_ns::Tensor &left, const rt_ns::Tensor &right,
                            std::string_view op, std::string_view roles) {
  if (left.data_type != right.data_type) {
    throw std::invalid_argument(std::string(op) + ": " + std::string(roles) +
                                " data types must match.");
  }
}

inline void RequireVector(const rt_ns::Tensor &tensor, std::size_t length, std::string_view op,
                          std::string_view role) {
  if (tensor.shape.size() != 1 || tensor.shape[0] != static_cast<std::int64_t>(length)) {
    throw std::invalid_argument(std::string(op) + ": " + std::string(role) +
                                " must be a rank-1 tensor with the required length.");
  }
}

class BroadcastIndexer {
public:
  BroadcastIndexer(const rt_ns::Shape &target_shape, const rt_ns::Shape &parameter_shape,
                   std::string_view op, std::string_view role) {
    if (parameter_shape.size() > target_shape.size()) {
      throw std::invalid_argument(std::string(op) + ": " + std::string(role) +
                                  " rank exceeds the normalized rank.");
    }
    target_dims_.assign(target_shape.begin(), target_shape.end());
    strides_.assign(target_shape.size(), 0);
    const std::size_t offset = target_shape.size() - parameter_shape.size();
    std::size_t stride = 1;
    for (std::size_t reverse = 0; reverse < parameter_shape.size(); ++reverse) {
      const std::size_t parameter_axis = parameter_shape.size() - reverse - 1;
      const std::size_t target_axis = offset + parameter_axis;
      const std::int64_t parameter_dim = parameter_shape[parameter_axis];
      const std::int64_t target_dim = target_shape[target_axis];
      if (parameter_dim != 1 && parameter_dim != target_dim) {
        throw std::invalid_argument(std::string(op) + ": " + std::string(role) +
                                    " is not broadcastable to the normalized shape.");
      }
      strides_[target_axis] = parameter_dim == 1 ? 0 : stride;
      stride *= static_cast<std::size_t>(parameter_dim);
    }
    identity_ = parameter_shape.size() == target_shape.size();
    if (identity_) {
      for (std::size_t i = 0; i < target_shape.size(); ++i) {
        if (parameter_shape[i] != target_shape[i]) {
          identity_ = false;
          break;
        }
      }
    }
  }

  std::size_t Index(std::size_t flat) const {
    if (identity_) {
      return flat;
    }
    std::size_t index = 0;
    for (std::size_t reverse = 0; reverse < target_dims_.size(); ++reverse) {
      const std::size_t axis = target_dims_.size() - reverse - 1;
      const std::size_t dimension = static_cast<std::size_t>(target_dims_[axis]);
      const std::size_t coordinate = flat % dimension;
      flat /= dimension;
      index += coordinate * strides_[axis];
    }
    return index;
  }

  bool identity() const noexcept { return identity_; }

private:
  std::vector<std::int64_t> target_dims_;
  std::vector<std::size_t> strides_;
  bool identity_ = false;
};

struct FloatMoments {
  float mean;
  float variance;
};

struct DoubleMoments {
  double mean;
  double variance;
};

template <typename Loader> FloatMoments ComputeFloatMoments(std::size_t count, Loader &&load) {
  if (count == 0) {
    throw std::invalid_argument("normalization reduction size must be positive.");
  }
  float sum = 0.0F;
  float compensation = 0.0F;
  for (std::size_t i = 0; i < count; ++i) {
    const float value = load(i);
    const float corrected = value - compensation;
    const float next = sum + corrected;
    compensation = (next - sum) - corrected;
    sum = next;
  }
  const float mean = sum / static_cast<float>(count);
  float squared_sum = 0.0F;
  compensation = 0.0F;
  for (std::size_t i = 0; i < count; ++i) {
    const float value = load(i);
    const float term = (value - mean) * (value - mean);
    const float corrected = term - compensation;
    const float next = squared_sum + corrected;
    compensation = (next - squared_sum) - corrected;
    squared_sum = next;
  }
  return {mean, squared_sum / static_cast<float>(count)};
}

template <typename Loader> float ComputeFloatMeanSquare(std::size_t count, Loader &&load) {
  if (count == 0) {
    throw std::invalid_argument("normalization reduction size must be positive.");
  }
  float squared_sum = 0.0F;
  float compensation = 0.0F;
  for (std::size_t i = 0; i < count; ++i) {
    const float value = load(i);
    const float corrected = value * value - compensation;
    const float next = squared_sum + corrected;
    compensation = (next - squared_sum) - corrected;
    squared_sum = next;
  }
  return squared_sum / static_cast<float>(count);
}

template <typename Loader> DoubleMoments ComputeDoubleMoments(std::size_t count, Loader &&load) {
  if (count == 0) {
    throw std::invalid_argument("normalization reduction size must be positive.");
  }
  double sum = 0.0;
  double compensation = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double value = load(i);
    const double corrected = value - compensation;
    const double next = sum + corrected;
    compensation = (next - sum) - corrected;
    sum = next;
  }
  const double mean = sum / static_cast<double>(count);
  double squared_sum = 0.0;
  compensation = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double value = load(i);
    const double term = (value - mean) * (value - mean);
    const double corrected = term - compensation;
    const double next = squared_sum + corrected;
    compensation = (next - squared_sum) - corrected;
    squared_sum = next;
  }
  return {mean, squared_sum / static_cast<double>(count)};
}

} // namespace onnx_light_cpu::normalization
