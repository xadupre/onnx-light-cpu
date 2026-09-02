// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/sigmoid_softmax_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/impl/math/unary_execution_tuning.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace onnx_light_cpu {

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

void ValidateOutput(const Tensor &x, const Tensor &output, const char *kernel_name) {
  if (output.data_type != x.data_type || output.shape != x.shape ||
      output.size_bytes() != x.size_bytes()) {
    throw std::invalid_argument(std::string(kernel_name) +
                                ": output dtype, shape, and size must match the input.");
  }
}

Tensor MakeLike(const Tensor &x, RuntimeContext *rt) {
  return rt != nullptr ? rt->MakeOutputTensor(0, x.data_type, x.shape, x.size_bytes())
                       : rt_ns::MakeOutputTensor(x.data_type, x.shape, x.size_bytes(), nullptr);
}

inline constexpr UnaryExecutionTuning kActivationExecutionTuning{256 * 1024, 256 * 1024, 32, false};
inline constexpr UnaryExecutionTuning kSerialExecutionTuning{};

template <typename T> void SigmoidRange(const T *input, T *output, std::size_t count) {
  if (input == output) {
    const std::vector<T> copy(input, input + count);
    SigmoidRange(copy.data(), output, count);
    return;
  }
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = -std::abs(input[i]);
  }
  if constexpr (std::is_same_v<T, float>) {
    ExpFloat32WithTuning(output, output, count, kSerialExecutionTuning);
  } else {
    for (std::size_t i = 0; i < count; ++i) {
      output[i] = std::exp(output[i]);
    }
  }
  for (std::size_t i = 0; i < count; ++i) {
    const T denominator = T{1} + output[i];
    const T negative_result = output[i] / denominator;
    const T positive_result = T{1} / denominator;
    using Bits =
        std::conditional_t<sizeof(T) == sizeof(std::uint32_t), std::uint32_t, std::uint64_t>;
    constexpr Bits sign_bit = Bits{1} << (sizeof(Bits) * 8 - 1);
    const Bits negative_mask =
        Bits{0} - static_cast<Bits>((std::bit_cast<Bits>(input[i]) & sign_bit) != 0);
    const Bits result_bits = (std::bit_cast<Bits>(negative_result) & negative_mask) |
                             (std::bit_cast<Bits>(positive_result) & ~negative_mask);
    output[i] = std::bit_cast<T>(result_bits);
  }
}

template <typename T> void Sigmoid(const T *input, T *output, std::size_t count) {
  ExecuteUnaryRanges<T>(
      count, kActivationExecutionTuning, [=](std::int64_t begin, std::int64_t end) {
        SigmoidRange(input + begin, output + begin, static_cast<std::size_t>(end - begin));
      });
}

template <typename DecodeBlock, typename EncodeBlock>
void SigmoidHalf(const std::uint16_t *input, std::uint16_t *output, std::size_t count,
                 DecodeBlock decode, EncodeBlock encode) {
  ExecuteUnaryRanges<std::uint16_t>(
      count, kActivationExecutionTuning, [=](std::int64_t begin, std::int64_t end) {
        constexpr std::size_t block_size = 256;
        std::array<float, block_size> input_buffer{};
        std::array<float, block_size> output_buffer{};
        for (std::size_t offset = static_cast<std::size_t>(begin);
             offset < static_cast<std::size_t>(end); offset += block_size) {
          const std::size_t block = std::min(block_size, static_cast<std::size_t>(end) - offset);
          decode(input + offset, input_buffer.data(), block);
          SigmoidRange(input_buffer.data(), output_buffer.data(), block);
          encode(output_buffer.data(), output + offset, block);
        }
      });
}

ExecutionSchedule MakeSoftmaxRowSchedule(std::size_t row_bytes) {
  constexpr std::size_t threshold_bytes = 128 * 1024;
  constexpr std::size_t target_block_bytes = 16 * 1024;
  const auto bytes_to_rows = [row_bytes](std::size_t bytes) {
    return static_cast<std::int64_t>(std::max<std::size_t>((bytes + row_bytes - 1) / row_bytes, 1));
  };
  return ExecutionSchedule{bytes_to_rows(threshold_bytes), bytes_to_rows(target_block_bytes), 48};
}

template <typename Fn> void DispatchSoftmaxRows(std::int64_t rows, std::size_t row_bytes, Fn fn) {
  ExecuteRanges(rows, MakeSoftmaxRowSchedule(row_bytes), std::int64_t{1}, std::move(fn));
}

template <typename T>
void SoftmaxLastAxis(const T *input, T *output, std::int64_t rows, std::int64_t columns) {
  if (rows == 0 || columns == 0) {
    return;
  }
  const std::size_t row_bytes = static_cast<std::size_t>(columns) * sizeof(T);
  DispatchSoftmaxRows(rows, row_bytes, [=](std::int64_t begin, std::int64_t end) {
    for (std::int64_t row = begin; row < end; ++row) {
      const T *row_input = input + row * columns;
      T *row_output = output + row * columns;
      T maximum = -std::numeric_limits<T>::infinity();
      for (std::int64_t column = 0; column < columns; ++column) {
        maximum = std::max(maximum, row_input[column]);
      }
      for (std::int64_t column = 0; column < columns; ++column) {
        row_output[column] = row_input[column] - maximum;
      }
    }
    T *range_output = output + begin * columns;
    const std::size_t range_count = static_cast<std::size_t>((end - begin) * columns);
    if constexpr (std::is_same_v<T, float>) {
      ExpFloat32WithTuning(range_output, range_output, range_count, kSerialExecutionTuning);
    } else {
      for (std::size_t i = 0; i < range_count; ++i) {
        range_output[i] = std::exp(range_output[i]);
      }
    }
    for (std::int64_t row = begin; row < end; ++row) {
      T *row_output = output + row * columns;
      T sum = T{0};
      for (std::int64_t column = 0; column < columns; ++column) {
        sum += row_output[column];
      }
      const T inverse_sum = T{1} / sum;
      for (std::int64_t column = 0; column < columns; ++column) {
        row_output[column] *= inverse_sum;
      }
    }
  });
}

template <typename DecodeBlock, typename EncodeBlock>
void SoftmaxHalfLastAxis(const std::uint16_t *input, std::uint16_t *output, std::int64_t rows,
                         std::int64_t columns, DecodeBlock decode, EncodeBlock encode) {
  if (rows == 0 || columns == 0) {
    return;
  }
  const std::size_t row_bytes = static_cast<std::size_t>(columns) * sizeof(std::uint16_t);
  DispatchSoftmaxRows(rows, row_bytes, [=](std::int64_t begin, std::int64_t end) {
    std::vector<float> buffer(static_cast<std::size_t>(columns));
    for (std::int64_t row = begin; row < end; ++row) {
      const std::size_t offset = static_cast<std::size_t>(row * columns);
      decode(input + offset, buffer.data(), buffer.size());
      SoftmaxLastAxis(buffer.data(), buffer.data(), 1, columns);
      encode(buffer.data(), output + offset, buffer.size());
    }
  });
}

template <typename T>
void Softmax(const T *input, T *output, std::int64_t outer, std::int64_t axis_dim,
             std::int64_t inner) {
  for (std::int64_t o = 0; o < outer; ++o) {
    for (std::int64_t i = 0; i < inner; ++i) {
      T maximum = -std::numeric_limits<T>::infinity();
      for (std::int64_t a = 0; a < axis_dim; ++a) {
        const std::size_t offset = static_cast<std::size_t>((o * axis_dim + a) * inner + i);
        maximum = std::max(maximum, input[offset]);
      }
      T sum = T{0};
      for (std::int64_t a = 0; a < axis_dim; ++a) {
        const std::size_t offset = static_cast<std::size_t>((o * axis_dim + a) * inner + i);
        output[offset] = std::exp(input[offset] - maximum);
        sum += output[offset];
      }
      for (std::int64_t a = 0; a < axis_dim; ++a) {
        const std::size_t offset = static_cast<std::size_t>((o * axis_dim + a) * inner + i);
        output[offset] /= sum;
      }
    }
  }
}

template <typename Decode, typename Encode>
void SoftmaxHalf(const std::uint16_t *input, std::uint16_t *output, std::int64_t outer,
                 std::int64_t axis_dim, std::int64_t inner, Decode decode, Encode encode) {
  for (std::int64_t o = 0; o < outer; ++o) {
    for (std::int64_t i = 0; i < inner; ++i) {
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::int64_t a = 0; a < axis_dim; ++a) {
        const std::size_t offset = static_cast<std::size_t>((o * axis_dim + a) * inner + i);
        maximum = std::max(maximum, decode(input[offset]));
      }
      float sum = 0.0F;
      for (std::int64_t a = 0; a < axis_dim; ++a) {
        const std::size_t offset = static_cast<std::size_t>((o * axis_dim + a) * inner + i);
        sum += std::exp(decode(input[offset]) - maximum);
      }
      for (std::int64_t a = 0; a < axis_dim; ++a) {
        const std::size_t offset = static_cast<std::size_t>((o * axis_dim + a) * inner + i);
        output[offset] = encode(std::exp(decode(input[offset]) - maximum) / sum);
      }
    }
  }
}

std::int64_t ResolveAxis(std::int64_t axis, std::int64_t rank) {
  const std::int64_t resolved = axis < 0 ? axis + rank : axis;
  if (resolved < 0 || resolved >= rank) {
    throw std::invalid_argument("onnx_light_cpu::Softmax: axis is out of range.");
  }
  return resolved;
}

} // namespace

Tensor SigmoidKernel::operator()(const Tensor &x, RuntimeContext *rt) const {
  Tensor output = MakeLike(x, rt);
  (*this)(x, output);
  return output;
}

void SigmoidKernel::operator()(const Tensor &x, Tensor &output) const {
  ValidateOutput(x, output, "onnx_light_cpu::Sigmoid");
  const std::size_t count = static_cast<std::size_t>(x.element_count());
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    Sigmoid(x.AsFloat(), output.AsFloat(), count);
    return;
  case DataType::DOUBLE:
    Sigmoid(x.AsDouble(), output.AsDouble(), count);
    return;
  case DataType::FLOAT16:
    SigmoidHalf(reinterpret_cast<const std::uint16_t *>(x.bytes()),
                reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), count,
                detail::ConvertFloat16ToFloat32, detail::ConvertFloat32ToFloat16);
    return;
  case DataType::BFLOAT16:
    SigmoidHalf(reinterpret_cast<const std::uint16_t *>(x.bytes()),
                reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), count,
                detail::ConvertBFloat16ToFloat32, detail::ConvertFloat32ToBFloat16);
    return;
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::Sigmoid: only FLOAT, DOUBLE, FLOAT16 and BFLOAT16 are supported.");
  }
}

void SigmoidKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &x = rt_ns::GetInput(node, 0, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(x, &rt), rt);
}

Tensor SoftmaxKernel::operator()(const Tensor &x, std::int64_t axis, RuntimeContext *rt) const {
  Tensor output = MakeLike(x, rt);
  (*this)(x, axis, output);
  return output;
}

void SoftmaxKernel::operator()(const Tensor &x, std::int64_t axis, Tensor &output) const {
  ValidateOutput(x, output, "onnx_light_cpu::Softmax");
  const std::int64_t rank = static_cast<std::int64_t>(x.shape.size());
  if (rank == 0) {
    throw std::invalid_argument("onnx_light_cpu::Softmax: input rank must be at least one.");
  }
  const std::int64_t resolved_axis = ResolveAxis(axis, rank);
  std::int64_t outer = 1;
  for (std::int64_t dimension = 0; dimension < resolved_axis; ++dimension) {
    outer *= x.shape[static_cast<std::size_t>(dimension)];
  }
  std::int64_t axis_dim = x.shape[static_cast<std::size_t>(resolved_axis)];
  std::int64_t inner = 1;
  for (std::int64_t dimension = resolved_axis + 1; dimension < rank; ++dimension) {
    inner *= x.shape[static_cast<std::size_t>(dimension)];
  }
  if (ctx_.opset.version < 13) {
    axis_dim *= inner;
    inner = 1;
  }

  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    if (inner == 1) {
      SoftmaxLastAxis(x.AsFloat(), output.AsFloat(), outer, axis_dim);
    } else {
      Softmax(x.AsFloat(), output.AsFloat(), outer, axis_dim, inner);
    }
    return;
  case DataType::DOUBLE:
    if (inner == 1) {
      SoftmaxLastAxis(x.AsDouble(), output.AsDouble(), outer, axis_dim);
    } else {
      Softmax(x.AsDouble(), output.AsDouble(), outer, axis_dim, inner);
    }
    return;
  case DataType::FLOAT16:
    if (inner == 1) {
      SoftmaxHalfLastAxis(reinterpret_cast<const std::uint16_t *>(x.bytes()),
                          reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), outer,
                          axis_dim, detail::ConvertFloat16ToFloat32,
                          detail::ConvertFloat32ToFloat16);
    } else {
      SoftmaxHalf(reinterpret_cast<const std::uint16_t *>(x.bytes()),
                  reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), outer, axis_dim, inner,
                  detail::Float16BitsToFloat, detail::FloatToFloat16Bits);
    }
    return;
  case DataType::BFLOAT16:
    if (inner == 1) {
      SoftmaxHalfLastAxis(reinterpret_cast<const std::uint16_t *>(x.bytes()),
                          reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), outer,
                          axis_dim, detail::ConvertBFloat16ToFloat32,
                          detail::ConvertFloat32ToBFloat16);
    } else {
      SoftmaxHalf(reinterpret_cast<const std::uint16_t *>(x.bytes()),
                  reinterpret_cast<std::uint16_t *>(output.mutable_bytes()), outer, axis_dim, inner,
                  detail::Bfloat16BitsToFloat, detail::FloatToBFloat16Bits);
    }
    return;
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::Softmax: only FLOAT, DOUBLE, FLOAT16 and BFLOAT16 are supported.");
  }
}

void SoftmaxKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const std::int64_t default_axis = ctx_.opset.version < 13 ? 1 : -1;
  const std::int64_t axis = rt_ns::GetAttributeIntOrDefault(node, "axis", default_axis);
  const Tensor &x = rt_ns::GetInput(node, 0, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(x, axis, &rt), rt);
}

void RegisterSigmoidKernel() {
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<SigmoidKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Sigmoid";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = SigmoidKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  RegisterKernel(std::move(info), std::move(factory));
}

void RegisterSoftmaxKernel() {
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<SoftmaxKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Softmax";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = SoftmaxKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
