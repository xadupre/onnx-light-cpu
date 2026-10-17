// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/naive_causal_conv_with_state_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/com_microsoft/causal_conv_with_state_common.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Tensor;

float Load(const Tensor &tensor, std::size_t index) {
  if (tensor.data_type == static_cast<int>(DataType::FLOAT)) {
    return tensor.AsFloat()[index];
  }
  const auto value = reinterpret_cast<const std::uint16_t *>(tensor.bytes())[index];
  return tensor.data_type == static_cast<int>(DataType::FLOAT16)
             ? detail::Float16BitsToFloat(value)
             : detail::Bfloat16BitsToFloat(value);
}

void Store(Tensor &tensor, std::size_t index, float value) {
  if (tensor.data_type == static_cast<int>(DataType::FLOAT)) {
    tensor.AsFloat()[index] = value;
  } else {
    reinterpret_cast<std::uint16_t *>(tensor.mutable_bytes())[index] =
        tensor.data_type == static_cast<int>(DataType::FLOAT16)
            ? detail::FloatToFloat16Bits(value)
            : detail::FloatToBFloat16Bits(value);
  }
}

} // namespace

std::pair<Tensor, Tensor>
NaiveCausalConvWithStateKernel::operator()(const Tensor &input, const Tensor &weight,
                                           const Tensor *bias, const Tensor *past_state,
                                           const std::string &activation, bool channels_last,
                                           std::int64_t dilation, rt_ns::RuntimeContext *rt) const {
  const auto plan = detail::PlanCausalConvWithState(input, weight, bias, past_state, activation,
                                                    channels_last, dilation);
  Tensor output =
      rt ? rt->MakeOutputTensor(0, input.data_type, input.shape, plan.output_bytes)
         : rt_ns::MakeOutputTensor(input.data_type, input.shape, plan.output_bytes, nullptr);
  Tensor present =
      rt ? rt->MakeOutputTensor(1, input.data_type, plan.state_shape, plan.state_bytes)
         : rt_ns::MakeOutputTensor(input.data_type, plan.state_shape, plan.state_bytes, nullptr);
  const auto offset = [&](std::size_t batch, std::size_t channel, std::size_t t,
                          std::size_t length) {
    return channels_last ? (batch * length + t) * plan.channels + channel
                         : (batch * plan.channels + channel) * length + t;
  };
  // Index a virtual concatenation [past_state, input], independently of the
  // production kernel's lag-based convolution and per-channel scheduling.
  const auto padded = [&](std::size_t batch, std::size_t channel, std::size_t t) {
    if (t < plan.pad) {
      return past_state ? Load(*past_state, offset(batch, channel, t, plan.pad)) : 0.0f;
    }
    return Load(input, offset(batch, channel, t - plan.pad, plan.length));
  };
  for (std::size_t batch = 0; batch < plan.batch; ++batch) {
    for (std::size_t t = 0; t < plan.length; ++t) {
      for (std::size_t channel = 0; channel < plan.channels; ++channel) {
        float sum = bias ? Load(*bias, channel) : 0.0f;
        for (std::size_t k = 0; k < plan.width; ++k) {
          sum += padded(batch, channel, t + k * static_cast<std::size_t>(dilation)) *
                 Load(weight, channel * plan.width + k);
        }
        if (activation != "none") {
          sum *= 1.0f / (1.0f + std::exp(-sum));
        }
        Store(output, offset(batch, channel, t, plan.length), sum);
      }
    }
    for (std::size_t t = 0; t < plan.pad; ++t) {
      for (std::size_t channel = 0; channel < plan.channels; ++channel) {
        Store(present, offset(batch, channel, t, plan.pad),
              padded(batch, channel, plan.length + t));
      }
    }
  }
  return {std::move(output), std::move(present)};
}

void NaiveCausalConvWithStateKernel::Run(rt_ns::RuntimeContext &rt) {
  detail::RunCausalConvWithState(*this, *node_, rt);
}

void RegisterNaiveCausalConvWithStateKernel() {
  rt_ns::NodeKernelFn factory =
      [](const ONNX_LIGHT_NAMESPACE::NodeProto &node, rt_ns::RuntimeContext &,
         const rt_ns::KernelContext &ctx) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<NaiveCausalConvWithStateKernel>(ctx);
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "CausalConvWithState";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = NaiveCausalConvWithStateKernel::kName;
  info.types = {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
