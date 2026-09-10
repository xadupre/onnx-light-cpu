// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace onnx_light_cpu {

namespace tensor_validation {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

[[noreturn]] inline void InvalidCapacity(std::string_view kernel_name, std::string_view role,
                                         std::string_view message) {
  throw std::invalid_argument(std::string(kernel_name) + ": " + std::string(role) + " " +
                              std::string(message));
}

inline std::size_t RequiredBufferSize(const rt_ns::Tensor &tensor, std::string_view kernel_name,
                                      std::string_view role) {
  const std::int64_t count = tensor.shape.product(
      0, tensor.shape.size(), std::string(kernel_name) + " " + std::string(role));
  const auto type = static_cast<rt_ns::DataType>(tensor.data_type);
  if (type == rt_ns::DataType::STRING) {
    return 0;
  }

  const std::uint64_t elements = static_cast<std::uint64_t>(count);
  std::uint64_t bytes;
  switch (type) {
  case rt_ns::DataType::INT4:
  case rt_ns::DataType::UINT4:
  case rt_ns::DataType::FLOAT4E2M1:
    bytes = elements / 2 + elements % 2;
    break;
  case rt_ns::DataType::INT2:
  case rt_ns::DataType::UINT2:
    bytes = elements / 4 + (elements % 4 != 0);
    break;
  case rt_ns::DataType::FLOAT6E2M3:
  case rt_ns::DataType::FLOAT6E3M2:
    bytes = elements / 4 * 3 + (elements % 4 * 6 + 7) / 8;
    break;
  default: {
    const std::size_t width = rt_ns::ElementSize(tensor.data_type);
    if (elements > std::numeric_limits<std::size_t>::max() / width) {
      InvalidCapacity(kernel_name, role, "required buffer size overflows size_t.");
    }
    return static_cast<std::size_t>(elements) * width;
  }
  }
  if (bytes > std::numeric_limits<std::size_t>::max()) {
    InvalidCapacity(kernel_name, role, "required buffer size overflows size_t.");
  }
  return static_cast<std::size_t>(bytes);
}

inline std::size_t ValidateBufferCapacity(const rt_ns::Tensor &tensor, std::string_view kernel_name,
                                          std::string_view role) {
  const std::size_t required = RequiredBufferSize(tensor, kernel_name, role);
  if (tensor.data_type == static_cast<std::int32_t>(rt_ns::DataType::STRING)) {
    if (tensor.AsStrings().size() <
        static_cast<std::size_t>(tensor.shape.product(
            0, tensor.shape.size(), std::string(kernel_name) + " " + std::string(role)))) {
      InvalidCapacity(kernel_name, role,
                      "STRING storage has fewer entries than its shape requires.");
    }
  } else if (tensor.size_bytes() < required) {
    InvalidCapacity(kernel_name, role,
                    "buffer capacity is smaller than its shape and dtype require.");
  } else if (required != 0 && tensor.bytes() == nullptr) {
    InvalidCapacity(kernel_name, role, "buffer storage is null.");
  }
  return required;
}

} // namespace tensor_validation

} // namespace onnx_light_cpu
