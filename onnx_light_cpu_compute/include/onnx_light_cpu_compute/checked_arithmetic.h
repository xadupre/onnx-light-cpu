// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace onnx_light_cpu {

[[noreturn]] inline void ThrowArithmeticError(const char *kernel, const char *label,
                                              const char *reason) {
  throw std::invalid_argument(std::string(kernel) + ": " + label + " " + reason + ".");
}

inline std::size_t CheckedDimension(std::int64_t dimension, const char *kernel, const char *label) {
  if (dimension < 0) {
    ThrowArithmeticError(kernel, label, "has a negative dimension");
  }
  if (static_cast<std::uint64_t>(dimension) >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    ThrowArithmeticError(kernel, label, "has a dimension that overflows size_t");
  }
  return static_cast<std::size_t>(dimension);
}

inline std::size_t CheckedMultiply(std::size_t left, std::size_t right, const char *kernel,
                                   const char *label) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    ThrowArithmeticError(kernel, label, "overflows size_t");
  }
  return left * right;
}

inline std::size_t CheckedAdd(std::size_t left, std::size_t right, const char *kernel,
                              const char *label) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    ThrowArithmeticError(kernel, label, "overflows size_t");
  }
  return left + right;
}

inline std::size_t CheckedProduct(std::initializer_list<std::size_t> factors, const char *kernel,
                                  const char *label) {
  for (const std::size_t factor : factors) {
    if (factor == 0) {
      return 0;
    }
  }
  std::size_t product = 1;
  for (const std::size_t factor : factors) {
    product = CheckedMultiply(product, factor, kernel, label);
  }
  return product;
}

inline std::size_t CheckedByteSize(std::size_t count, std::size_t element_size, const char *kernel,
                                   const char *label) {
  return CheckedMultiply(count, element_size, kernel, label);
}

inline std::int64_t CheckedIndexMultiply(std::int64_t left, std::int64_t right, const char *kernel,
                                         const char *label) {
  if (left < 0 || right < 0) {
    ThrowArithmeticError(kernel, label, "has a negative operand");
  }
  if (right != 0 && left > std::numeric_limits<std::int64_t>::max() / right) {
    ThrowArithmeticError(kernel, label, "overflows int64_t");
  }
  return left * right;
}

inline std::int64_t CheckedIndexAdd(std::int64_t left, std::int64_t right, const char *kernel,
                                    const char *label) {
  if (left < 0 || right < 0) {
    ThrowArithmeticError(kernel, label, "has a negative operand");
  }
  if (left > std::numeric_limits<std::int64_t>::max() - right) {
    ThrowArithmeticError(kernel, label, "overflows int64_t");
  }
  return left + right;
}

template <typename Shape>
inline std::int64_t CheckedShapeIndexProduct(const Shape &shape, const char *kernel,
                                             const char *label) {
  bool empty = false;
  for (const std::int64_t dimension : shape) {
    if (dimension < 0) {
      ThrowArithmeticError(kernel, label, "has a negative dimension");
    }
    empty |= dimension == 0;
  }
  if (empty) {
    return 0;
  }
  std::int64_t product = 1;
  for (const std::int64_t dimension : shape) {
    product = CheckedIndexMultiply(product, dimension, kernel, label);
  }
  return product;
}

inline std::ptrdiff_t CheckedStride(std::size_t stride, const char *kernel, const char *label) {
  if (stride > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    ThrowArithmeticError(kernel, label, "overflows ptrdiff_t");
  }
  return static_cast<std::ptrdiff_t>(stride);
}

} // namespace onnx_light_cpu
