// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/attention/linear_attention_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

const std::vector<DataType> kSupportedTypes{DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16};

} // namespace

LinearAttentionKernel::Result
LinearAttentionKernel::operator()(const Tensor &query, const Tensor &key, const Tensor &value,
                                  const Attributes &attributes, const Tensor *past_state,
                                  const Tensor *decay, const Tensor *beta, RuntimeContext *rt,
                                  bool has_state_output) const {
  const LinearAttentionPlan plan = PlanLinearAttention(
      kName, query, key, value, past_state, decay, beta, attributes.update_rule,
      attributes.query_heads, attributes.key_value_heads, attributes.scale, {}, kSupportedTypes);
  return ExecuteLinearAttentionPlan(kName, plan, query, key, value, past_state, decay, beta, rt,
                                    has_state_output);
}

void LinearAttentionKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  if (node.input_size() < 3 || node.input_size() > 6 || node.output_size() < 1 ||
      node.output_size() > 2) {
    throw std::invalid_argument(std::string(kName) + ": expected 3-6 inputs and 1-2 outputs.");
  }
  const Tensor &query = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &key = rt_ns::GetInput(node, 1, rt.tensors());
  const Tensor &value = rt_ns::GetInput(node, 2, rt.tensors());
  const Tensor *past_state = rt_ns::GetOptionalInput(node, 3, rt.tensors());
  const Tensor *decay = rt_ns::GetOptionalInput(node, 4, rt.tensors());
  const Tensor *beta = rt_ns::GetOptionalInput(node, 5, rt.tensors());
  Attributes attributes;
  attributes.update_rule = rt_ns::GetAttributeStringOrDefault(node, "update_rule", "gated_delta");
  attributes.query_heads = rt_ns::GetAttributeIntOrDefault(node, "q_num_heads", 0);
  attributes.key_value_heads = rt_ns::GetAttributeIntOrDefault(node, "kv_num_heads", 0);
  attributes.scale = rt_ns::GetAttributeFloatOrDefault(node, "scale", 0.0f);
  const bool has_state_output = node.output_size() == 2 && !node.output(1).empty();
  Result result =
      (*this)(query, key, value, attributes, past_state, decay, beta, &rt, has_state_output);
  rt_ns::SetOutput(node, 0, std::move(result.output), rt);
  if (has_state_output) {
    rt_ns::SetOutput(node, 1, std::move(result.present_state), rt);
  }
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
  info.types = kSupportedTypes;
  info.since_version = 27;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
