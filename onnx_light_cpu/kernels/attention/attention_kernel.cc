// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/attention/attention_kernel.h"

#include "onnx_light_cpu/impl/attention/attention_plan.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::NodeKernelFn;
using rt_ns::RuntimeContext;
using rt_ns::Shape;
using rt_ns::Tensor;

namespace {

const ONNX_LIGHT_NAMESPACE::AttributeProto *FindAttribute(const NodeProto &node, const char *name) {
  for (int i = 0; i < node.attribute_size(); ++i) {
    if (node.attribute(i).name() == name) {
      return &node.attribute(i);
    }
  }
  return nullptr;
}

std::vector<std::int64_t> ShapeAsInt64(const Shape &shape) {
  return std::vector<std::int64_t>(shape.begin(), shape.end());
}

AttentionDescriptor BuildDescriptor(const NodeProto &node) {
  AttentionDescriptor descriptor;
  // The kernel does not have direct access to the node's opset from here;
  // `nonpad_kv_seqlen` (opset >= 24 only) is only wired when a 7th input is
  // present, so infer the minimum opset consistent with the wiring. This is
  // conservative: the materialized path rejects `nonpad_kv_seqlen` outright
  // regardless of the inferred opset.
  descriptor.opset = node.input_size() > 6 ? 24 : 23;

  if (const auto *attribute = FindAttribute(node, "scale"); attribute != nullptr) {
    descriptor.scale = attribute->f();
  }
  if (const auto *attribute = FindAttribute(node, "softmax_precision"); attribute != nullptr) {
    descriptor.softmax_precision = attribute->i();
  }
  if (const auto *attribute = FindAttribute(node, "q_num_heads"); attribute != nullptr) {
    descriptor.q_num_heads = attribute->i();
  }
  if (const auto *attribute = FindAttribute(node, "kv_num_heads"); attribute != nullptr) {
    descriptor.kv_num_heads = attribute->i();
  }
  descriptor.is_causal = rt_ns::GetAttributeIntOrDefault(node, "is_causal", 0) != 0;
  descriptor.softcap = rt_ns::GetAttributeFloatOrDefault(node, "softcap", 0.0f);
  descriptor.qk_matmul_output_mode =
      rt_ns::GetAttributeIntOrDefault(node, "qk_matmul_output_mode", 0);

  descriptor.has_attn_mask = node.input_size() > 3 && !node.input(3).empty();
  descriptor.has_past_key = node.input_size() > 4 && !node.input(4).empty();
  descriptor.has_past_value = node.input_size() > 5 && !node.input(5).empty();
  descriptor.has_nonpad_kv_seqlen = node.input_size() > 6 && !node.input(6).empty();

  descriptor.has_present_key = node.output_size() > 1 && !node.output(1).empty();
  descriptor.has_present_value = node.output_size() > 2 && !node.output(2).empty();
  descriptor.has_qk_matmul_output = node.output_size() > 3 && !node.output(3).empty();

  descriptor.Validate();
  descriptor.ValidateSupportedByMaterializedPath();
  return descriptor;
}

Tensor Compute(const Tensor &q, const Tensor &k, const Tensor &v, const Tensor *mask,
               const AttentionDescriptor &descriptor, RuntimeContext *rt) {
  if (q.data_type != static_cast<std::int32_t>(DataType::FLOAT) ||
      k.data_type != static_cast<std::int32_t>(DataType::FLOAT) ||
      v.data_type != static_cast<std::int32_t>(DataType::FLOAT)) {
    throw std::invalid_argument(
        "onnx_light_cpu::AttentionKernel: only FLOAT Q/K/V are supported so far.");
  }
  const AttentionLayout layout =
      q.shape.size() == 3 ? AttentionLayout::kRank3 : AttentionLayout::kRank4;

  AttentionMaskKind mask_kind = AttentionMaskKind::kNone;
  if (mask != nullptr) {
    if (static_cast<DataType>(mask->data_type) == DataType::BOOL) {
      mask_kind = AttentionMaskKind::kBoolean;
    } else if (static_cast<DataType>(mask->data_type) == DataType::FLOAT) {
      mask_kind = AttentionMaskKind::kAdditive;
    } else {
      throw std::invalid_argument("onnx_light_cpu::AttentionKernel: unsupported attn_mask data "
                                  "type; only BOOL and FLOAT are supported so far.");
    }
  }

  const AttentionPlan plan(
      descriptor, layout, ShapeAsInt64(q.shape), ShapeAsInt64(k.shape), ShapeAsInt64(v.shape),
      mask != nullptr ? ShapeAsInt64(mask->shape) : std::vector<std::int64_t>{}, mask_kind);

  const std::vector<std::int64_t> plan_output_shape = plan.output_shape();
  Shape output_shape;
  output_shape.reserve(plan_output_shape.size());
  std::size_t element_count = 1;
  for (const std::int64_t dimension : plan_output_shape) {
    output_shape.push_back(dimension);
    element_count *= static_cast<std::size_t>(dimension);
  }
  const std::size_t bytes = element_count * sizeof(float);
  Tensor y = rt != nullptr ? rt->MakeOutputTensor(0, q.data_type, output_shape, bytes)
                           : rt_ns::MakeOutputTensor(q.data_type, output_shape, bytes, nullptr);

  const void *mask_data = nullptr;
  if (mask != nullptr) {
    mask_data = mask_kind == AttentionMaskKind::kBoolean ? static_cast<const void *>(mask->AsBool())
                                                         : static_cast<const void *>(mask->bytes());
  }
  ComputeAttentionFloat32(plan, reinterpret_cast<const float *>(q.bytes()),
                          reinterpret_cast<const float *>(k.bytes()),
                          reinterpret_cast<const float *>(v.bytes()), mask_data,
                          reinterpret_cast<float *>(y.mutable_bytes()));
  return y;
}

} // namespace

AttentionKernel::AttentionKernel(const rt_ns::KernelContext &ctx) : KernelBase(ctx) {}

Tensor AttentionKernel::operator()(const NodeProto &node, const Tensor &q, const Tensor &k,
                                   const Tensor &v, const Tensor *mask, RuntimeContext *rt) const {
  const AttentionDescriptor descriptor = BuildDescriptor(node);
  return Compute(q, k, v, mask, descriptor, rt);
}

void AttentionKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  const AttentionDescriptor descriptor = BuildDescriptor(node);
  const Tensor &q = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &k = rt_ns::GetInput(node, 1, rt.tensors());
  const Tensor &v = rt_ns::GetInput(node, 2, rt.tensors());
  const Tensor *mask =
      descriptor.has_attn_mask ? rt_ns::GetOptionalInput(node, 3, rt.tensors()) : nullptr;
  Tensor y = Compute(q, k, v, mask, descriptor, &rt);
  rt_ns::SetOutput(node, 0, std::move(y), rt);
}

void RegisterAttentionKernel() {
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<AttentionKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Attention";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = AttentionKernel::kName;
  info.types = {DataType::FLOAT};
  info.since_version = 23;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
