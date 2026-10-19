// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/causal_conv_with_state_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/causal_conv_with_state_common.h"
#include "onnx_light_cpu/kernels/execution.h"

#include "cpu_compute/checked_arithmetic.h"
#include "cpu_compute/execution.h"
#include "cpu_compute/math/half_conversion.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/tensor_buffer_validation.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;
using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;

template <typename T> float Read(const T *data, std::size_t index, DataType dtype) {
  if constexpr (sizeof(T) == sizeof(float)) {
    return data[index];
  } else {
    return dtype == DataType::FLOAT16 ? cpu_compute::detail::Float16BitsToFloat(data[index])
                                      : cpu_compute::detail::Bfloat16BitsToFloat(data[index]);
  }
}

template <typename T> void Write(T *data, std::size_t index, float value, DataType dtype) {
  if constexpr (sizeof(T) == sizeof(float)) {
    data[index] = value;
  } else {
    data[index] = dtype == DataType::FLOAT16 ? cpu_compute::detail::FloatToFloat16Bits(value)
                                             : cpu_compute::detail::FloatToBFloat16Bits(value);
  }
}

template <typename T>
void Compute(const Tensor &input, const Tensor &weight, const Tensor *bias, const Tensor *past,
             Tensor &output, Tensor &present, std::size_t batch, std::size_t channels,
             std::size_t length, std::size_t width, std::size_t pad, std::size_t dilation,
             bool channels_last, bool silu) {
  const auto *x = reinterpret_cast<const T *>(input.bytes());
  const auto *w = reinterpret_cast<const T *>(weight.bytes());
  const auto *b = bias ? reinterpret_cast<const T *>(bias->bytes()) : nullptr;
  const auto *p = past ? reinterpret_cast<const T *>(past->bytes()) : nullptr;
  auto *y = reinterpret_cast<T *>(output.mutable_bytes());
  auto *s = reinterpret_cast<T *>(present.mutable_bytes());
  const DataType dtype = static_cast<DataType>(input.data_type);
  cpu_compute::ExecuteRanges(
      cpu_compute::CheckedIndexMultiply(static_cast<std::int64_t>(batch),
                                        static_cast<std::int64_t>(channels), "CausalConvWithState",
                                        "batch channels"),
      static_cast<double>(length) * static_cast<double>(width) + 2.0 * static_cast<double>(pad),
      [&](std::int64_t first, std::int64_t last) {
        for (auto task = first; task < last; ++task) {
          const std::size_t n = static_cast<std::size_t>(task);
          const std::size_t row = n / channels;
          const std::size_t channel = n % channels;
          const auto input_at = [&](std::size_t t) {
            return channels_last ? (row * length + t) * channels + channel : n * length + t;
          };
          const auto state_at = [&](std::size_t t) {
            return channels_last ? (row * pad + t) * channels + channel : n * pad + t;
          };
          for (std::size_t t = 0; t < length; ++t) {
            float sum = b ? Read(b, channel, dtype) : 0.0f;
            for (std::size_t k = 0; k < width; ++k) {
              const std::size_t lag = (width - 1 - k) * dilation;
              float value = 0.0f;
              if (lag <= t) {
                value = Read(x, input_at(t - lag), dtype);
              } else if (lag - t <= pad && p != nullptr) {
                value = Read(p, state_at(pad - (lag - t)), dtype);
              }
              sum += Read(w, channel * width + k, dtype) * value;
            }
            Write(y, input_at(t), silu ? sum / (1.0f + std::exp(-sum)) : sum, dtype);
          }
          for (std::size_t t = 0; t < pad; ++t) {
            const std::size_t source = t + length;
            if (source >= pad) {
              s[state_at(t)] = x[input_at(source - pad)];
            } else if (p != nullptr) {
              s[state_at(t)] = p[state_at(source)];
            } else {
              Write(s, state_at(t), 0.0f, dtype);
            }
          }
        }
      });
}

} // namespace

detail::CausalConvWithStatePlan
detail::PlanCausalConvWithState(const Tensor &input, const Tensor &weight, const Tensor *bias,
                                const Tensor *past_state, const std::string &activation,
                                bool channels_last, std::int64_t dilation) {
  constexpr const char *name = "CausalConvWithState";
  const DataType dtype = static_cast<DataType>(input.data_type);
  if (dtype != DataType::FLOAT && dtype != DataType::FLOAT16 && dtype != DataType::BFLOAT16) {
    throw std::invalid_argument(
        "CausalConvWithState: only FLOAT, FLOAT16 and BFLOAT16 are supported.");
  }
  if (activation != "none" && activation != "silu" && activation != "swish") {
    throw std::invalid_argument("CausalConvWithState: unsupported activation.");
  }
  if (dilation < 1 || dilation > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("CausalConvWithState: dilation must be in [1, INT_MAX].");
  }
  if ((!channels_last && input.shape.size() != 3) || (channels_last && input.shape.size() < 3) ||
      weight.shape.size() != 3 || weight.shape[1] != 1 || weight.shape[2] < 1) {
    throw std::invalid_argument("CausalConvWithState: invalid input or depthwise weight shape.");
  }
  std::size_t channels = 1;
  if (channels_last) {
    for (std::size_t i = 2; i < input.shape.size(); ++i) {
      channels = cpu_compute::CheckedMultiply(
          channels, cpu_compute::CheckedDimension(input.shape[i], name, "channels"), name,
          "channels");
    }
  } else {
    channels = cpu_compute::CheckedDimension(input.shape[1], name, "channels");
  }
  const std::size_t batch = cpu_compute::CheckedDimension(input.shape[0], name, "batch");
  const std::size_t length =
      cpu_compute::CheckedDimension(input.shape[channels_last ? 1 : 2], name, "sequence");
  const std::size_t width = cpu_compute::CheckedDimension(weight.shape[2], name, "kernel width");
  const std::size_t pad = cpu_compute::CheckedMultiply(
      width - 1, static_cast<std::size_t>(dilation), name, "state length");
  if (channels > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      weight.shape[0] != static_cast<std::int64_t>(channels) || pad > INT32_MAX ||
      (bias != nullptr &&
       (bias->shape.size() != 1 || bias->shape[0] != static_cast<std::int64_t>(channels)))) {
    throw std::invalid_argument(
        "CausalConvWithState: weight channels, bias or state length invalid.");
  }
  const rt_ns::Shape state_shape = [&] {
    rt_ns::Shape shape = input.shape;
    shape[channels_last ? 1 : 2] = static_cast<std::int64_t>(pad);
    return shape;
  }();
  if ((bias && bias->data_type != input.data_type) || weight.data_type != input.data_type ||
      (past_state &&
       (past_state->data_type != input.data_type || past_state->shape != state_shape))) {
    throw std::invalid_argument("CausalConvWithState: tensor types or past_state shape mismatch.");
  }
  const std::size_t elements =
      cpu_compute::CheckedProduct({batch, channels, length}, name, "output");
  const std::size_t state_elements =
      cpu_compute::CheckedProduct({batch, channels, pad}, name, "state");
  const std::size_t bytes =
      cpu_compute::CheckedByteSize(elements, input.element_size(), name, "output");
  const std::size_t state_bytes =
      cpu_compute::CheckedByteSize(state_elements, input.element_size(), name, "state");
  cpu_compute::CheckedIndexMultiply(static_cast<std::int64_t>(batch),
                                    static_cast<std::int64_t>(channels), name, "batch channels");
  tensor_validation::ValidateBufferCapacity(input, name, "input");
  tensor_validation::ValidateBufferCapacity(weight, name, "weight");
  if (bias != nullptr) {
    tensor_validation::ValidateBufferCapacity(*bias, name, "bias");
  }
  if (past_state != nullptr) {
    tensor_validation::ValidateBufferCapacity(*past_state, name, "past_state");
  }
  return {batch, channels, length, width, pad, state_shape, bytes, state_bytes};
}

std::pair<Tensor, Tensor>
CausalConvWithStateKernel::operator()(const Tensor &input, const Tensor &weight, const Tensor *bias,
                                      const Tensor *past_state, const std::string &activation,
                                      bool channels_last, std::int64_t dilation,
                                      RuntimeContext *rt) const {
  const RuntimeExecutionScope execution_scope;
  const auto plan = detail::PlanCausalConvWithState(input, weight, bias, past_state, activation,
                                                    channels_last, dilation);
  Tensor output =
      rt ? rt->MakeOutputTensor(0, input.data_type, input.shape, plan.output_bytes)
         : rt_ns::MakeOutputTensor(input.data_type, input.shape, plan.output_bytes, nullptr);
  Tensor present =
      rt ? rt->MakeOutputTensor(1, input.data_type, plan.state_shape, plan.state_bytes)
         : rt_ns::MakeOutputTensor(input.data_type, plan.state_shape, plan.state_bytes, nullptr);
  const auto &[batch, channels, length, width, pad, state_shape, bytes, state_bytes] = plan;
  const DataType dtype = static_cast<DataType>(input.data_type);
  if (batch != 0 && channels != 0) {
    if (dtype == DataType::FLOAT) {
      Compute<float>(input, weight, bias, past_state, output, present, batch, channels, length,
                     width, pad, static_cast<std::size_t>(dilation), channels_last,
                     activation != "none");
    } else {
      Compute<std::uint16_t>(input, weight, bias, past_state, output, present, batch, channels,
                             length, width, pad, static_cast<std::size_t>(dilation), channels_last,
                             activation != "none");
    }
  }
  return {std::move(output), std::move(present)};
}

void CausalConvWithStateKernel::Run(RuntimeContext &rt) {
  const RuntimeExecutionScope execution_scope;
  detail::RunCausalConvWithState(*this, *node_, rt);
}

void RegisterCausalConvWithStateKernel() {
  rt_ns::NodeKernelFn factory =
      [](const NodeProto &node, RuntimeContext &,
         const rt_ns::KernelContext &ctx) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<CausalConvWithStateKernel>(ctx);
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "CausalConvWithState";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = CausalConvWithStateKernel::kName;
  info.types = {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
