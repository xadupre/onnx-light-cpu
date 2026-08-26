// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/variadic_elementwise_plan.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace onnx_light_cpu {
namespace {

std::size_t ElementSize(std::int32_t type) {
  switch (type) {
  case BinaryDataType::INT8:
  case BinaryDataType::UINT8:
    return 1;
  case BinaryDataType::INT16:
  case BinaryDataType::UINT16:
  case BinaryDataType::FLOAT16:
  case BinaryDataType::BFLOAT16:
    return 2;
  case BinaryDataType::INT32:
  case BinaryDataType::UINT32:
  case BinaryDataType::FLOAT:
    return 4;
  case BinaryDataType::INT64:
  case BinaryDataType::UINT64:
  case BinaryDataType::DOUBLE:
    return 8;
  default:
    throw std::invalid_argument("onnx_light_cpu::VariadicElementwisePlan: unsupported data type.");
  }
}

bool Supports(VariadicOperator op, std::int32_t type) {
  if (op == VariadicOperator::kMean) {
    return type == BinaryDataType::FLOAT || type == BinaryDataType::DOUBLE;
  }
  switch (type) {
  case BinaryDataType::FLOAT:
  case BinaryDataType::DOUBLE:
  case BinaryDataType::FLOAT16:
  case BinaryDataType::BFLOAT16:
  case BinaryDataType::INT8:
  case BinaryDataType::INT16:
  case BinaryDataType::INT32:
  case BinaryDataType::INT64:
  case BinaryDataType::UINT8:
  case BinaryDataType::UINT16:
  case BinaryDataType::UINT32:
  case BinaryDataType::UINT64:
    return true;
  default:
    return false;
  }
}

template <typename T> T Load(const std::byte *data) {
  T value;
  std::memcpy(&value, data, sizeof(T));
  return value;
}

template <typename T> void Store(std::byte *data, T value) { std::memcpy(data, &value, sizeof(T)); }

template <typename T> T Add(T left, T right) {
  if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    using U = std::make_unsigned_t<T>;
    return std::bit_cast<T>(static_cast<U>(static_cast<U>(left) + static_cast<U>(right)));
  } else {
    return static_cast<T>(left + right);
  }
}

template <typename T> T Apply(VariadicOperator op, T left, T right) {
  switch (op) {
  case VariadicOperator::kSum:
  case VariadicOperator::kMean:
    return Add(left, right);
  case VariadicOperator::kMin:
    return left < right ? left : right;
  case VariadicOperator::kMax:
    return left > right ? left : right;
  }
  return left;
}

template <typename T>
void ExecuteTyped(VariadicOperator op, std::span<const void *const> inputs,
                  std::span<const std::size_t> offsets, std::byte *output) {
  T value = Load<T>(static_cast<const std::byte *>(inputs[0]) + offsets[0] * sizeof(T));
  for (std::size_t input = 1; input < inputs.size(); ++input) {
    const T next =
        Load<T>(static_cast<const std::byte *>(inputs[input]) + offsets[input] * sizeof(T));
    value = Apply(op, value, next);
  }
  if (op == VariadicOperator::kMean && inputs.size() > 1) {
    value *= static_cast<T>(1) / static_cast<T>(inputs.size());
  }
  Store(output, value);
}

template <bool BFloat16>
void ExecuteHalf(VariadicOperator op, std::span<const void *const> inputs,
                 std::span<const std::size_t> offsets, std::byte *output) {
  const auto decode = [](std::uint16_t bits) {
    if constexpr (BFloat16) {
      return detail::Bfloat16BitsToFloat(bits);
    } else {
      return detail::Float16BitsToFloat(bits);
    }
  };
  const auto encode = [](float value) {
    if constexpr (BFloat16) {
      return detail::FloatToBFloat16Bits(value);
    } else {
      return detail::FloatToFloat16Bits(value);
    }
  };
  std::uint16_t bits =
      Load<std::uint16_t>(static_cast<const std::byte *>(inputs[0]) + offsets[0] * sizeof(bits));
  for (std::size_t input = 1; input < inputs.size(); ++input) {
    const std::uint16_t next = Load<std::uint16_t>(static_cast<const std::byte *>(inputs[input]) +
                                                   offsets[input] * sizeof(next));
    bits = encode(Apply(op, decode(bits), decode(next)));
  }
  Store(output, bits);
}

} // namespace

VariadicElementwisePlan::VariadicElementwisePlan(
    VariadicOperator op, std::span<const std::int32_t> input_types,
    std::span<const std::vector<std::int64_t>> input_shapes)
    : op_(op), data_type_(BinaryDataType::UNDEFINED), input_count_(input_types.size()),
      element_size_(0), element_count_(1) {
  if (input_types.empty() || input_types.size() != input_shapes.size()) {
    throw std::invalid_argument(
        "onnx_light_cpu::VariadicElementwisePlan: inputs must be non-empty and have shapes.");
  }
  data_type_ = input_types[0];
  if (!Supports(op, data_type_)) {
    throw std::invalid_argument("onnx_light_cpu::" + std::string(ToString(op)) +
                                ": unsupported data type.");
  }
  for (std::int32_t type : input_types) {
    if (type != data_type_) {
      throw std::invalid_argument("onnx_light_cpu::" + std::string(ToString(op)) +
                                  ": all inputs must have the same data type.");
    }
  }
  element_size_ = ElementSize(data_type_);

  std::size_t rank = 0;
  for (const auto &shape : input_shapes) {
    rank = std::max(rank, shape.size());
    for (std::int64_t dimension : shape) {
      if (dimension < 0) {
        throw std::invalid_argument("onnx_light_cpu::" + std::string(ToString(op)) +
                                    ": input dimensions must be non-negative.");
      }
    }
  }
  output_shape_.assign(rank, 1);
  for (const auto &shape : input_shapes) {
    const std::size_t offset = rank - shape.size();
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      const std::int64_t dimension = shape[axis];
      std::int64_t &output_dimension = output_shape_[offset + axis];
      if (output_dimension == dimension || dimension == 1) {
        continue;
      }
      if (output_dimension == 1) {
        output_dimension = dimension;
        continue;
      }
      throw std::invalid_argument("onnx_light_cpu::" + std::string(ToString(op)) +
                                  ": input shapes are not multidirectional-broadcastable.");
    }
  }

  input_strides_.assign(input_count_ * rank, 0);
  for (std::size_t input = 0; input < input_count_; ++input) {
    const auto &shape = input_shapes[input];
    const std::size_t offset = rank - shape.size();
    std::size_t stride = 1;
    for (std::size_t axis = rank; axis-- > offset;) {
      const std::int64_t dimension = shape[axis - offset];
      input_strides_[input * rank + axis] = dimension == 1 && output_shape_[axis] != 1 ? 0 : stride;
      if (dimension != 0 &&
          stride > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension)) {
        throw std::invalid_argument("onnx_light_cpu::" + std::string(ToString(op)) +
                                    ": input shape is too large.");
      }
      stride *= static_cast<std::size_t>(dimension);
    }
  }
  for (std::int64_t dimension : output_shape_) {
    if (dimension != 0 && element_count_ > std::numeric_limits<std::size_t>::max() /
                                               static_cast<std::size_t>(dimension)) {
      throw std::invalid_argument("onnx_light_cpu::" + std::string(ToString(op)) +
                                  ": output shape is too large.");
    }
    element_count_ *= static_cast<std::size_t>(dimension);
  }
  if (element_count_ > std::numeric_limits<std::size_t>::max() / element_size_) {
    throw std::invalid_argument("onnx_light_cpu::" + std::string(ToString(op)) +
                                ": output tensor is too large.");
  }
}

void VariadicElementwisePlan::Execute(std::span<const void *const> inputs, void *output) const {
  if (inputs.size() != input_count_) {
    throw std::invalid_argument("onnx_light_cpu::" + std::string(ToString(op_)) +
                                ": input count does not match the plan.");
  }
  if (element_count_ == 0) {
    return;
  }
  if (output == nullptr || std::any_of(inputs.begin(), inputs.end(),
                                       [](const void *input) { return input == nullptr; })) {
    throw std::invalid_argument("onnx_light_cpu::" + std::string(ToString(op_)) +
                                ": non-empty tensors require data.");
  }

  const std::size_t rank = output_shape_.size();
  std::vector<std::size_t> coordinates(rank, 0);
  std::vector<std::size_t> offsets(input_count_, 0);
  auto *out = static_cast<std::byte *>(output);
  const auto execute = [&](auto execute_element) {
    for (std::size_t index = 0; index < element_count_; ++index) {
      execute_element(out + index * element_size_);
      for (std::size_t axis = rank; axis-- > 0;) {
        ++coordinates[axis];
        for (std::size_t input = 0; input < input_count_; ++input) {
          offsets[input] += input_strides_[input * rank + axis];
        }
        if (coordinates[axis] < static_cast<std::size_t>(output_shape_[axis])) {
          break;
        }
        coordinates[axis] = 0;
        for (std::size_t input = 0; input < input_count_; ++input) {
          offsets[input] -=
              input_strides_[input * rank + axis] * static_cast<std::size_t>(output_shape_[axis]);
        }
      }
    }
  };
  switch (data_type_) {
  case BinaryDataType::FLOAT:
    execute([&](std::byte *element) { ExecuteTyped<float>(op_, inputs, offsets, element); });
    break;
  case BinaryDataType::DOUBLE:
    execute([&](std::byte *element) { ExecuteTyped<double>(op_, inputs, offsets, element); });
    break;
  case BinaryDataType::FLOAT16:
    execute([&](std::byte *element) { ExecuteHalf<false>(op_, inputs, offsets, element); });
    break;
  case BinaryDataType::BFLOAT16:
    execute([&](std::byte *element) { ExecuteHalf<true>(op_, inputs, offsets, element); });
    break;
#define ONNX_LIGHT_CPU_VARIADIC_INTEGER_CASE(TYPE, CPP_TYPE)                                       \
  case BinaryDataType::TYPE:                                                                       \
    execute([&](std::byte *element) { ExecuteTyped<CPP_TYPE>(op_, inputs, offsets, element); });   \
    break
    ONNX_LIGHT_CPU_VARIADIC_INTEGER_CASE(INT8, std::int8_t);
    ONNX_LIGHT_CPU_VARIADIC_INTEGER_CASE(INT16, std::int16_t);
    ONNX_LIGHT_CPU_VARIADIC_INTEGER_CASE(INT32, std::int32_t);
    ONNX_LIGHT_CPU_VARIADIC_INTEGER_CASE(INT64, std::int64_t);
    ONNX_LIGHT_CPU_VARIADIC_INTEGER_CASE(UINT8, std::uint8_t);
    ONNX_LIGHT_CPU_VARIADIC_INTEGER_CASE(UINT16, std::uint16_t);
    ONNX_LIGHT_CPU_VARIADIC_INTEGER_CASE(UINT32, std::uint32_t);
    ONNX_LIGHT_CPU_VARIADIC_INTEGER_CASE(UINT64, std::uint64_t);
#undef ONNX_LIGHT_CPU_VARIADIC_INTEGER_CASE
  default:
    break;
  }
}

const char *ToString(VariadicOperator op) noexcept {
  switch (op) {
  case VariadicOperator::kSum:
    return "Sum";
  case VariadicOperator::kMean:
    return "Mean";
  case VariadicOperator::kMin:
    return "Min";
  case VariadicOperator::kMax:
    return "Max";
  }
  return "Variadic";
}

} // namespace onnx_light_cpu
