// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/attention/linear_attention_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/symbolic/sym_tensor.h"

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
      LinearAttentionKernel::kName,
      {},
      {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16},
  };
  return config;
}

} // namespace

LinearAttentionKernel::Result
LinearAttentionKernel::operator()(const Tensor &query, const Tensor &key, const Tensor &value,
                                  const Attributes &attributes, const Tensor *past_state,
                                  const Tensor *decay, const Tensor *beta, RuntimeContext *rt,
                                  bool has_state_output) const {
  return InvokeLinearAttentionKernel(Config(), query, key, value, attributes, past_state, decay,
                                     beta, rt, has_state_output);
}

void LinearAttentionKernel::Run(RuntimeContext &rt) {
  RunLinearAttentionKernelNode(Config(), *node_, rt);
}

void RegisterLinearAttentionKernel() {
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<LinearAttentionKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "LinearAttention";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = LinearAttentionKernel::kName;
  info.types = Config().supported_types;
  info.since_version = 27;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
