// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/reduce_sum_kernel.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_extensions/kernels/kernels/reduction/include_reduction_kernels.h"
#include "onnx_lib/common/safe_math.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Tensor;
using BuiltinReduceSum = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::ReduceSum;

namespace {

[[noreturn]] void Invalid(const char *message) {
  throw std::invalid_argument(std::string("onnx_light_cpu::ReduceSum: ") + message);
}

std::size_t Bytes(int64_t count) {
  int64_t bytes;
  if (ONNX_LIGHT_NAMESPACE::checked_mul_overflow(count, int64_t{sizeof(int64_t)}, &bytes)) {
    Invalid("tensor byte size overflow.");
  }
  return ONNX_LIGHT_NAMESPACE::safe_cast_to_size(bytes, Invalid);
}

int64_t ElementCount(const rt_ns::Shape &shape) {
  return shape.product(0, shape.size(), "ReduceSum");
}

} // namespace

ReduceSumKernel::ReduceSumKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                 const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
}

Tensor ReduceSumKernel::Compute(const Tensor &data, const Tensor *axes, bool keepdims,
                                bool noop_with_empty_axes, rt_ns::RuntimeContext &rt) const {
  if (data.data_type != DataType::INT64) {
    throw std::invalid_argument("onnx_light_cpu::ReduceSum: data must be INT64.");
  }

  const int64_t input_count = ElementCount(data.shape);
  const std::size_t input_bytes = Bytes(input_count);
  if (data.size_bytes() != input_bytes || (input_bytes != 0 && data.bytes() == nullptr)) {
    Invalid("data buffer size does not match shape.");
  }

  rt_ns::Shape reduced(data.shape.size(), 0);
  int64_t axis_count = 0;
  if (axes != nullptr) {
    if (axes->data_type != DataType::INT64) {
      Invalid("axes must be INT64.");
    }
    axis_count = ElementCount(axes->shape);
    const std::size_t axes_bytes = Bytes(axis_count);
    if (axes->size_bytes() != axes_bytes || (axes_bytes != 0 && axes->bytes() == nullptr)) {
      Invalid("axes buffer size does not match shape.");
    }
  }
  const bool reduce_all = (axes == nullptr || axis_count == 0) && !noop_with_empty_axes;
  if (reduce_all) {
    std::fill(reduced.begin(), reduced.end(), 1);
  } else if (axes != nullptr && axis_count != 0) {
    const int64_t rank = static_cast<int64_t>(data.shape.size());
    const int64_t *values = axes->AsInt64();
    for (int64_t i = 0; i < axis_count; ++i) {
      const int64_t axis = values[i];
      const int64_t resolved = axis < 0 ? axis + rank : axis;
      if (resolved < 0 || resolved >= rank) {
        Invalid("axis is out of range.");
      }
      reduced[static_cast<std::size_t>(resolved)] = 1;
    }
  }

  rt_ns::Shape output_shape;
  output_shape.reserve(data.shape.size());
  for (std::size_t dim = 0; dim < data.shape.size(); ++dim) {
    if (reduced[dim] == 0 || keepdims) {
      output_shape.push_back(reduced[dim] == 0 ? data.shape[dim] : 1);
    }
  }
  const int64_t output_count = ElementCount(output_shape);
  const std::size_t output_bytes = Bytes(output_count);
  Tensor output = rt.MakeOutputTensor(0, DataType::INT64, output_shape, output_bytes);

  if ((axes == nullptr || axis_count == 0) && noop_with_empty_axes) {
    if (input_bytes != 0) {
      std::memcpy(output.mutable_bytes(), data.bytes(), input_bytes);
    }
    return output;
  }

  if (output_bytes != 0) {
    std::memset(output.mutable_bytes(), 0, output_bytes);
  }
  rt_ns::Shape reduction_shape;
  reduction_shape.reserve(data.shape.size());
  for (std::size_t dim = 0; dim < data.shape.size(); ++dim) {
    if (reduced[dim] != 0) {
      reduction_shape.push_back(data.shape[dim]);
    }
  }
  const int64_t reduction_count = ElementCount(reduction_shape);
  if (output_count == 0 || reduction_count == 0) {
    return output;
  }

  rt_ns::Shape input_strides(data.shape.size(), 1);
  if (input_count != 0) {
    for (std::size_t dim = data.shape.size(); dim-- > 1;) {
      input_strides[dim - 1] = input_strides[dim] * data.shape[dim];
    }
  }

  const int64_t *input = data.AsInt64();
  uint8_t *output_bytes_ptr = output.mutable_bytes();
  ExecuteRanges(
      output_count, static_cast<double>(reduction_count), [&](int64_t begin, int64_t end) {
        for (int64_t output_index = begin; output_index < end; ++output_index) {
          int64_t remaining_output = output_index;
          int64_t input_base = 0;
          for (std::size_t dim = data.shape.size(); dim-- > 0;) {
            if (reduced[dim] == 0) {
              const int64_t size = data.shape[dim];
              const int64_t coordinate = remaining_output % size;
              remaining_output /= size;
              input_base += coordinate * input_strides[dim];
            }
          }

          uint64_t sum = 0;
          for (int64_t reduction_index = 0; reduction_index < reduction_count; ++reduction_index) {
            int64_t remaining_reduction = reduction_index;
            int64_t input_index = input_base;
            for (std::size_t dim = data.shape.size(); dim-- > 0;) {
              if (reduced[dim] != 0) {
                const int64_t size = data.shape[dim];
                const int64_t coordinate = remaining_reduction % size;
                remaining_reduction /= size;
                input_index += coordinate * input_strides[dim];
              }
            }
            sum += static_cast<uint64_t>(input[input_index]);
          }

          // Unsigned accumulation gives modulo-2^64 sums without invoking
          // signed-overflow undefined behavior.
          std::memcpy(output_bytes_ptr + static_cast<std::size_t>(output_index) * sizeof(sum), &sum,
                      sizeof(sum));
        }
      });
  return output;
}

void ReduceSumKernel::Run(rt_ns::RuntimeContext &rt) {
  const auto &node = *node_;
  rt_ns::RequireMinInputCount(node, 1);
  if (node.input_size() > 2) {
    Invalid("expected one or two inputs.");
  }
  rt_ns::RequireOutputCount(node, 1);

  const Tensor &data = rt_ns::GetInput(node, 0, rt.tensors());
  if (data.data_type != DataType::INT64) {
    BuiltinReduceSum reference(ctx_);
    reference.set_node(node);
    reference.Run(rt);
    return;
  }

  rt.RecordKernelUsage(kName);
  const bool keepdims = rt_ns::GetAttributeIntOrDefault(node, "keepdims", 1) != 0;
  const bool noop_with_empty_axes =
      rt_ns::GetAttributeIntOrDefault(node, "noop_with_empty_axes", 0) != 0;
  const std::vector<int64_t> axes_attribute = rt_ns::GetAttributeIntsOrDefault(node, "axes", {});
  const Tensor *axes_input = rt_ns::GetOptionalInput(node, 1, rt.tensors());
  if (axes_input != nullptr) {
    rt_ns::SetOutput(node, 0, Compute(data, axes_input, keepdims, noop_with_empty_axes, rt), rt);
  } else if (!axes_attribute.empty()) {
    const Tensor axes =
        Tensor::FromInt64("", {static_cast<int64_t>(axes_attribute.size())}, axes_attribute);
    rt_ns::SetOutput(node, 0, Compute(data, &axes, keepdims, noop_with_empty_axes, rt), rt);
  } else {
    rt_ns::SetOutput(node, 0, Compute(data, nullptr, keepdims, noop_with_empty_axes, rt), rt);
  }
}

void RegisterReduceSumKernel() {
  rt_ns::NodeKernelFn factory =
      [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
         rt_ns::RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<ReduceSumKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "ReduceSum";
  info.kernel_name = ReduceSumKernel::kName;
  info.types = {DataType::INT64};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
