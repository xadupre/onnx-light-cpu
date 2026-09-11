// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/naive_linear_attention_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

Tensor ExpandHeads(const Tensor &input, std::size_t heads, std::size_t expanded_heads,
                   std::size_t head_size) {
  if (heads == expanded_heads) {
    return input;
  }
  const std::size_t hidden = LinearAttentionCheckedMultiply(
      NaiveMicrosoftLinearAttentionKernel::kName, expanded_heads, head_size, "expanded heads");
  if (hidden > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("expanded LinearAttention hidden size exceeds int64_t.");
  }
  const rt_ns::Shape shape{input.shape[0], input.shape[1], static_cast<std::int64_t>(hidden)};
  std::vector<float> data(static_cast<std::size_t>(
      shape.product(0, shape.size(), NaiveMicrosoftLinearAttentionKernel::kName)));
  const std::size_t tokens =
      static_cast<std::size_t>(shape.product(0, 2, NaiveMicrosoftLinearAttentionKernel::kName));
  const std::size_t repeats = expanded_heads / heads;
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t head = 0; head < expanded_heads; ++head) {
      std::copy_n(input.AsFloat() + (token * heads + head / repeats) * head_size, head_size,
                  data.data() + (token * expanded_heads + head) * head_size);
    }
  }
  return Tensor::FromFloat("", shape, data);
}

} // namespace

LinearAttentionResult NaiveMicrosoftLinearAttentionKernel::operator()(
    const Tensor &query, const Tensor &key, const Tensor &value, const Attributes &attributes,
    const Tensor *past_state, const Tensor *decay, const Tensor *beta, RuntimeContext *rt) const {
  if (attributes.state_window != 0) {
    throw std::invalid_argument(std::string(kName) + ": nonzero state_window is not supported.");
  }
  if (attributes.query_heads > std::numeric_limits<std::int32_t>::max() ||
      attributes.key_value_heads > std::numeric_limits<std::int32_t>::max()) {
    throw std::invalid_argument(std::string(kName) + ": head counts exceed the supported maximum.");
  }
  const auto plan =
      PlanLinearAttention(kName, query, key, value, past_state, decay, beta, attributes.update_rule,
                          attributes.query_heads, attributes.key_value_heads, attributes.scale,
                          {true, true, true}, {DataType::FLOAT});
  const auto &p = plan.parameters;
  // The ONNX contract requires Hq >= Hkv and a physical key for each state head.
  // Replicate only the shared inputs; state, gates, values and outputs keep their layout.
  const Tensor reference_query =
      ExpandHeads(query, p.query_heads, plan.output_heads, p.key_head_size);
  const Tensor reference_key = ExpandHeads(key, p.key_heads, p.key_value_heads, p.key_head_size);
  using Reference = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::LinearAttention;
  Reference::Attributes reference_attributes;
  reference_attributes.update_rule = attributes.update_rule;
  reference_attributes.q_num_heads = static_cast<std::int64_t>(plan.output_heads);
  reference_attributes.kv_num_heads = attributes.key_value_heads;
  reference_attributes.has_scale = true;
  reference_attributes.scale = attributes.scale;
  reference_attributes.chunk_size = attributes.chunk_size;
  const Reference reference{ctx_};
  auto result = reference(reference_query, reference_key, value, reference_attributes, past_state,
                          LinearAttentionRuleUsesDecay(p.rule) ? decay : nullptr,
                          LinearAttentionRuleUsesBeta(p.rule) ? beta : nullptr, rt);
  return {std::move(result.output), std::move(result.present_state)};
}

void NaiveMicrosoftLinearAttentionKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const auto &node = *node_;
  if (node.input_size() < 3 || node.input_size() > 6 || node.output_size() != 2 ||
      node.output(0).empty() || node.output(1).empty()) {
    throw std::invalid_argument(std::string(kName) +
                                ": invalid LinearAttention input or output count.");
  }
  Attributes attributes;
  attributes.update_rule = rt_ns::GetAttributeStringOrDefault(node, "update_rule", "gated_delta");
  attributes.query_heads = rt_ns::GetAttributeIntOrDefault(node, "q_num_heads", 0);
  attributes.key_value_heads = rt_ns::GetAttributeIntOrDefault(node, "kv_num_heads", 0);
  attributes.scale = rt_ns::GetAttributeFloatOrDefault(node, "scale", 0.0f);
  attributes.chunk_size = rt_ns::GetAttributeIntOrDefault(node, "chunk_size", 64);
  attributes.state_window = rt_ns::GetAttributeIntOrDefault(node, "state_window", 0);
  auto result =
      (*this)(rt_ns::GetInput(node, 0, rt.tensors()), rt_ns::GetInput(node, 1, rt.tensors()),
              rt_ns::GetInput(node, 2, rt.tensors()), attributes,
              rt_ns::GetOptionalInput(node, 3, rt.tensors()),
              rt_ns::GetOptionalInput(node, 4, rt.tensors()),
              rt_ns::GetOptionalInput(node, 5, rt.tensors()), &rt);
  rt_ns::SetOutput(node, 0, std::move(result.output), rt);
  rt_ns::SetOutput(node, 1, std::move(result.present_state), rt);
}

void RegisterNaiveMicrosoftLinearAttentionKernel() {
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<NaiveMicrosoftLinearAttentionKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "LinearAttention";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = NaiveMicrosoftLinearAttentionKernel::kName;
  info.types = {DataType::FLOAT};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
