// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/linear_attention_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <limits>
#include <memory>
#include <utility>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

const LinearAttentionKernelConfig &Config() {
  static const LinearAttentionKernelConfig config{
      MicrosoftLinearAttentionKernel::kName,    {true, true, true}, {DataType::FLOAT}, true, true,
      std::numeric_limits<std::int32_t>::max(),
  };
  return config;
}

void RegisterKernelImpl() {
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<MicrosoftLinearAttentionKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "LinearAttention";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = MicrosoftLinearAttentionKernel::kName;
  info.types = Config().supported_types;
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace

LinearAttentionResult MicrosoftLinearAttentionKernel::operator()(
    const Tensor &query, const Tensor &key, const Tensor &value, const Attributes &attributes,
    const Tensor *past_state, const Tensor *decay, const Tensor *beta, RuntimeContext *rt) const {
  return InvokeLinearAttentionKernel(Config(), query, key, value, attributes, past_state, decay,
                                     beta, rt, true);
}

void MicrosoftLinearAttentionKernel::Run(RuntimeContext &rt) {
  RunLinearAttentionKernelNode(Config(), *node_, rt);
}

void RegisterMicrosoftLinearAttentionKernel() { RegisterKernelImpl(); }

void RegisterNaiveMicrosoftLinearAttentionKernel() { RegisterKernelImpl(); }

} // namespace onnx_light_cpu
