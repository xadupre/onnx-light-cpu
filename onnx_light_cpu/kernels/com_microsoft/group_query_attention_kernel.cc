// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/group_query_attention_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::NodeKernelFn;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

namespace {

bool HasInput(const NodeProto &node, int index) {
  return node.input_size() > index && !node.input(index).empty();
}

bool HasOutput(const NodeProto &node, int index) {
  return node.output_size() > index && !node.output(index).empty();
}

void AddIntAttribute(NodeProto &node, const char *name, std::int64_t value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::INT);
  attribute->set_i(value);
}

void AddFloatAttribute(NodeProto &node, const char *name, float value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::FLOAT);
  attribute->set_f(value);
}

const ONNX_LIGHT_NAMESPACE::AttributeProto *FindAttribute(const NodeProto &node, const char *name) {
  for (int i = 0; i < node.attribute_size(); ++i) {
    if (node.attribute(i).name() == name) {
      return &node.attribute(i);
    }
  }
  return nullptr;
}

NodeProto MakeAttentionNode(const NodeProto &node) {
  const std::int64_t num_heads = rt_ns::GetAttributeIntOrDefault(node, "num_heads", 0);
  const std::int64_t kv_num_heads = rt_ns::GetAttributeIntOrDefault(node, "kv_num_heads", 0);
  if (num_heads <= 0 || kv_num_heads <= 0) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: num_heads and kv_num_heads must be positive.");
  }

  NodeProto attention;
  attention.set_op_type("Attention");
  attention.add_input("query");
  attention.add_input("key");
  attention.add_input("value");
  if (HasInput(node, 10)) {
    attention.add_input("attention_bias");
  }
  attention.add_output("output");
  AddIntAttribute(attention, "q_num_heads", num_heads);
  AddIntAttribute(attention, "kv_num_heads", kv_num_heads);
  AddIntAttribute(attention, "is_causal", rt_ns::GetAttributeIntOrDefault(node, "causal", 1));
  if (const auto *scale = FindAttribute(node, "scale"); scale != nullptr) {
    AddFloatAttribute(attention, "scale", scale->f());
  }
  const float softcap = rt_ns::GetAttributeFloatOrDefault(node, "softcap", 0.0f);
  if (softcap != 0.0f) {
    AddFloatAttribute(attention, "softcap", softcap);
  }
  return attention;
}

void ValidateSupported(const NodeProto &node, const Tensor &query, const Tensor &key,
                       const Tensor &seqlens_k, const Tensor &total_sequence_length) {
  if (node.input_size() < 7 || HasInput(node, 3) || HasInput(node, 4) || HasInput(node, 7) ||
      HasInput(node, 8) || HasInput(node, 9) || HasInput(node, 11) || HasInput(node, 12) ||
      HasInput(node, 13) || HasInput(node, 14) || HasInput(node, 15)) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: only non-cached, non-rotary "
                                "Q/K/V attention is supported.");
  }
  if (HasOutput(node, 1) || HasOutput(node, 2) || HasOutput(node, 3)) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: only the output tensor is "
                                "supported; present and output_qk outputs are not supported.");
  }
  if (rt_ns::GetAttributeIntOrDefault(node, "do_rotary", 0) != 0 ||
      rt_ns::GetAttributeIntOrDefault(node, "sliding_window_cache", 0) != 0 ||
      rt_ns::GetAttributeIntOrDefault(node, "smooth_softmax", 0) != 0 ||
      rt_ns::GetAttributeIntOrDefault(node, "qk_output", 0) != 0 ||
      rt_ns::GetAttributeIntOrDefault(node, "kv_cache_bit_width", 0) != 0 ||
      rt_ns::GetAttributeIntOrDefault(node, "local_window_size", -1) != -1 ||
      rt_ns::GetAttributeStringOrDefault(node, "k_quant_type", "NONE") != "NONE" ||
      rt_ns::GetAttributeStringOrDefault(node, "v_quant_type", "NONE") != "NONE") {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: unsupported attribute.");
  }
  if (query.shape.size() != 3 || key.shape.size() != 3 || query.shape[0] != key.shape[0] ||
      query.shape[1] != key.shape[1] || key.shape[1] <= 0 ||
      key.shape[1] > std::numeric_limits<std::int32_t>::max()) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: Q and K must be non-empty "
                                "rank-3 self-attention tensors with matching batch and sequence.");
  }
  if (static_cast<DataType>(seqlens_k.data_type) != DataType::INT32 ||
      seqlens_k.element_count() != query.shape[0] ||
      static_cast<DataType>(total_sequence_length.data_type) != DataType::INT32 ||
      total_sequence_length.element_count() != 1) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: seqlens_k must be INT32 "
        "with one value per batch, and total_sequence_length an INT32 scalar.");
  }
  const std::int32_t expected_length = static_cast<std::int32_t>(key.shape[1]);
  if (total_sequence_length.AsInt32()[0] != expected_length) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: total_sequence_length must "
                                "equal the Q/K/V sequence length without a KV cache.");
  }
  for (std::int64_t batch = 0; batch < query.shape[0]; ++batch) {
    if (seqlens_k.AsInt32()[batch] != expected_length - 1) {
      throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: non-cached inputs require "
                                  "every seqlens_k value to equal sequence_length - 1.");
    }
  }
}

} // namespace

GroupQueryAttentionKernel::GroupQueryAttentionKernel(const NodeProto &node,
                                                     const rt_ns::KernelContext &ctx)
    : KernelBase(ctx), attention_(ctx) {
  set_node(node);
}

void GroupQueryAttentionKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireOutputCount(node, 1);
  if (node.input_size() < 7) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: query, key, value, seqlens_k and "
        "total_sequence_length inputs are required.");
  }
  const Tensor &query = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &key = rt_ns::GetInput(node, 1, rt.tensors());
  const Tensor &value = rt_ns::GetInput(node, 2, rt.tensors());
  const Tensor &seqlens_k = rt_ns::GetInput(node, 5, rt.tensors());
  const Tensor &total_sequence_length = rt_ns::GetInput(node, 6, rt.tensors());
  const Tensor *attention_bias =
      HasInput(node, 10) ? rt_ns::GetOptionalInput(node, 10, rt.tensors()) : nullptr;
  ValidateSupported(node, query, key, seqlens_k, total_sequence_length);
  const NodeProto attention_node = MakeAttentionNode(node);
  rt_ns::SetOutput(node, 0, attention_(attention_node, query, key, value, attention_bias, &rt), rt);
}

void RegisterGroupQueryAttentionKernel() {
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<GroupQueryAttentionKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "GroupQueryAttention";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = GroupQueryAttentionKernel::kName;
  info.types = {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
