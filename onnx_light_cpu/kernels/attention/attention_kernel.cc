// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/attention/attention_kernel.h"

#include "onnx_light_cpu/impl/attention/attention_plan.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
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
  // present, so infer the minimum opset consistent with the wiring. Window
  // attributes (opset >= 25) further raise the inferred opset.
  int inferred_opset = node.input_size() > 6 ? 24 : 23;
  if (rt_ns::GetAttributeIntOrDefault(node, "left_window_size", -1) >= 0 ||
      rt_ns::GetAttributeIntOrDefault(node, "right_window_size", -1) >= 0) {
    inferred_opset = std::max(inferred_opset, 25);
  }
  descriptor.opset = inferred_opset;

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

  // Window attributes (opset >= 25).
  descriptor.left_window_size = rt_ns::GetAttributeIntOrDefault(node, "left_window_size", -1);
  descriptor.right_window_size = rt_ns::GetAttributeIntOrDefault(node, "right_window_size", -1);

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

void ComputeHalfAttentionMaterialized(const AttentionPlan &plan, DataType data_type,
                                      const std::uint16_t *q, const std::uint16_t *k,
                                      const std::uint16_t *v, const void *mask, std::uint16_t *y,
                                      const std::uint16_t *past_k, const std::uint16_t *past_v,
                                      const std::int64_t *nonpad_kv_seqlen,
                                      std::uint16_t *qk_matmul_output) {
  const std::size_t q_count = plan.batch * plan.q_num_heads * plan.q_length * plan.head_dim;
  const std::size_t k_count = plan.batch * plan.kv_num_heads * plan.kv_length * plan.head_dim;
  const std::size_t v_count = plan.batch * plan.kv_num_heads * plan.kv_length * plan.v_head_dim;
  const std::size_t y_count = plan.batch * plan.q_num_heads * plan.q_length * plan.v_head_dim;
  const std::size_t past_k_count =
      plan.batch * plan.kv_num_heads * plan.past_length * plan.head_dim;
  const std::size_t past_v_count =
      plan.batch * plan.kv_num_heads * plan.past_length * plan.v_head_dim;
  const std::size_t qk_count =
      qk_matmul_output != nullptr
          ? plan.batch * plan.q_num_heads * plan.q_length * plan.total_kv_length
          : 0;
  std::vector<float> q_fp32(q_count);
  std::vector<float> k_fp32(k_count);
  std::vector<float> v_fp32(v_count);
  std::vector<float> y_fp32(y_count);
  std::vector<float> past_k_fp32(past_k_count);
  std::vector<float> past_v_fp32(past_v_count);
  std::vector<float> qk_fp32(qk_count);

  auto convert_to_float = [data_type](const std::uint16_t *source, float *destination,
                                      std::size_t count) {
    if (data_type == DataType::FLOAT16) {
      detail::ConvertFloat16ToFloat32(source, destination, count);
    } else {
      for (std::size_t index = 0; index < count; ++index) {
        destination[index] = detail::Bfloat16BitsToFloat(source[index]);
      }
    }
  };
  auto convert_from_float = [data_type](const float *source, std::uint16_t *destination,
                                        std::size_t count) {
    if (data_type == DataType::FLOAT16) {
      detail::ConvertFloat32ToFloat16(source, destination, count);
    } else {
      for (std::size_t index = 0; index < count; ++index) {
        destination[index] = detail::FloatToBFloat16Bits(source[index]);
      }
    }
  };

  convert_to_float(q, q_fp32.data(), q_count);
  convert_to_float(k, k_fp32.data(), k_count);
  convert_to_float(v, v_fp32.data(), v_count);
  if (past_k_count != 0) {
    convert_to_float(past_k, past_k_fp32.data(), past_k_count);
    convert_to_float(past_v, past_v_fp32.data(), past_v_count);
  }
  ComputeAttentionFloat32(plan, q_fp32.data(), k_fp32.data(), v_fp32.data(), mask, y_fp32.data(),
                          past_k_count != 0 ? past_k_fp32.data() : nullptr,
                          past_v_count != 0 ? past_v_fp32.data() : nullptr, nonpad_kv_seqlen,
                          qk_count != 0 ? qk_fp32.data() : nullptr);
  convert_from_float(y_fp32.data(), y, y_count);
  if (qk_count != 0) {
    convert_from_float(qk_fp32.data(), qk_matmul_output, qk_count);
  }
}

Tensor Compute(const Tensor &q, const Tensor &k, const Tensor &v, const Tensor *mask,
               const Tensor *past_k, const Tensor *past_v, const Tensor *nonpad_kv_seqlen,
               const AttentionDescriptor &descriptor, RuntimeContext *rt, Tensor *qk_output,
               Tensor *present_key_output, Tensor *present_value_output) {
  const DataType data_type = static_cast<DataType>(q.data_type);
  if (k.data_type != q.data_type || v.data_type != q.data_type ||
      (data_type != DataType::FLOAT && data_type != DataType::FLOAT16 &&
       data_type != DataType::BFLOAT16)) {
    throw std::invalid_argument("onnx_light_cpu::AttentionKernel: Q/K/V must have the same "
                                "FLOAT, FLOAT16, or BFLOAT16 data type.");
  }
  if ((past_k != nullptr && past_k->data_type != q.data_type) ||
      (past_v != nullptr && past_v->data_type != q.data_type)) {
    throw std::invalid_argument(
        "onnx_light_cpu::AttentionKernel: tensor cache data type must match Q/K/V.");
  }
  if (nonpad_kv_seqlen != nullptr &&
      static_cast<DataType>(nonpad_kv_seqlen->data_type) != DataType::INT64) {
    throw std::invalid_argument(
        "onnx_light_cpu::AttentionKernel: nonpad_kv_seqlen must have type INT64.");
  }
  const AttentionLayout layout =
      q.shape.size() == 3 ? AttentionLayout::kRank3 : AttentionLayout::kRank4;

  // --- Mask handling: convert 16-bit additive masks to FP32 on the fly ---
  AttentionMaskKind mask_kind = AttentionMaskKind::kNone;
  std::vector<float> mask_fp32_buffer;
  if (mask != nullptr) {
    if (static_cast<DataType>(mask->data_type) == DataType::BOOL) {
      mask_kind = AttentionMaskKind::kBoolean;
    } else if (static_cast<DataType>(mask->data_type) == DataType::FLOAT) {
      mask_kind = AttentionMaskKind::kAdditive;
    } else if (static_cast<DataType>(mask->data_type) == DataType::FLOAT16) {
      mask_kind = AttentionMaskKind::kAdditive;
      std::size_t mask_count = 1;
      for (const auto &d : mask->shape) {
        mask_count *= static_cast<std::size_t>(d);
      }
      mask_fp32_buffer.resize(mask_count);
      detail::ConvertFloat16ToFloat32(reinterpret_cast<const std::uint16_t *>(mask->bytes()),
                                      mask_fp32_buffer.data(), mask_count);
    } else if (static_cast<DataType>(mask->data_type) == DataType::BFLOAT16) {
      mask_kind = AttentionMaskKind::kAdditive;
      std::size_t mask_count = 1;
      for (const auto &d : mask->shape) {
        mask_count *= static_cast<std::size_t>(d);
      }
      mask_fp32_buffer.resize(mask_count);
      detail::ConvertBFloat16ToFloat32(reinterpret_cast<const std::uint16_t *>(mask->bytes()),
                                       mask_fp32_buffer.data(), mask_count);
    } else {
      throw std::invalid_argument("onnx_light_cpu::AttentionKernel: unsupported attn_mask data "
                                  "type; only BOOL, FLOAT, FLOAT16, and BFLOAT16 are supported.");
    }
  }

  const AttentionPlan plan(
      descriptor, layout, ShapeAsInt64(q.shape), ShapeAsInt64(k.shape), ShapeAsInt64(v.shape),
      mask != nullptr ? ShapeAsInt64(mask->shape) : std::vector<std::int64_t>{}, mask_kind,
      past_k != nullptr ? ShapeAsInt64(past_k->shape) : std::vector<std::int64_t>{},
      past_v != nullptr ? ShapeAsInt64(past_v->shape) : std::vector<std::int64_t>{});

  const std::vector<std::int64_t> plan_output_shape = plan.output_shape();
  Shape output_shape;
  output_shape.reserve(plan_output_shape.size());
  std::size_t element_count = 1;
  for (const std::int64_t dimension : plan_output_shape) {
    output_shape.push_back(dimension);
    element_count *= static_cast<std::size_t>(dimension);
  }
  const std::size_t element_bytes =
      data_type == DataType::FLOAT ? sizeof(float) : sizeof(std::uint16_t);
  const std::size_t bytes = element_count * element_bytes;
  Tensor y = rt != nullptr ? rt->MakeOutputTensor(0, q.data_type, output_shape, bytes)
                           : rt_ns::MakeOutputTensor(q.data_type, output_shape, bytes, nullptr);
  Tensor qk;
  float *qk_data = nullptr;
  std::uint16_t *qk_half_data = nullptr;
  if (plan.has_qk_matmul_output || plan.softmax_fp64) {
    const std::vector<std::int64_t> plan_qk_shape = plan.qk_matmul_output_shape();
    Shape qk_shape(plan_qk_shape);
    const std::size_t qk_bytes = static_cast<std::size_t>(plan.batch * plan.q_num_heads *
                                                          plan.q_length * plan.total_kv_length) *
                                 element_bytes;
    qk = rt != nullptr ? rt->MakeOutputTensor(3, q.data_type, qk_shape, qk_bytes)
                       : rt_ns::MakeOutputTensor(q.data_type, qk_shape, qk_bytes, nullptr);
    if (data_type == DataType::FLOAT) {
      qk_data = reinterpret_cast<float *>(qk.mutable_bytes());
    } else {
      qk_half_data = reinterpret_cast<std::uint16_t *>(qk.mutable_bytes());
    }
  }

  // Resolve the mask data pointer: use the converted FP32 buffer when the
  // original mask was FLOAT16, the boolean bytes for BOOL, or the raw FLOAT
  // bytes otherwise.
  const void *mask_data = nullptr;
  if (mask != nullptr) {
    if (!mask_fp32_buffer.empty()) {
      mask_data = static_cast<const void *>(mask_fp32_buffer.data());
    } else if (mask_kind == AttentionMaskKind::kBoolean) {
      mask_data = static_cast<const void *>(mask->AsBool());
    } else {
      mask_data = static_cast<const void *>(mask->bytes());
    }
  }
  const std::int64_t *nonpad =
      nonpad_kv_seqlen != nullptr
          ? reinterpret_cast<const std::int64_t *>(nonpad_kv_seqlen->bytes())
          : nullptr;
  if (data_type == DataType::FLOAT) {
    ComputeAttentionFloat32(
        plan, reinterpret_cast<const float *>(q.bytes()),
        reinterpret_cast<const float *>(k.bytes()), reinterpret_cast<const float *>(v.bytes()),
        mask_data, reinterpret_cast<float *>(y.mutable_bytes()),
        past_k != nullptr ? reinterpret_cast<const float *>(past_k->bytes()) : nullptr,
        past_v != nullptr ? reinterpret_cast<const float *>(past_v->bytes()) : nullptr, nonpad,
        qk_data);
  } else if (data_type == DataType::FLOAT16) {
    if (plan.has_qk_matmul_output || plan.softmax_fp64) {
      ComputeHalfAttentionMaterialized(
          plan, data_type, reinterpret_cast<const std::uint16_t *>(q.bytes()),
          reinterpret_cast<const std::uint16_t *>(k.bytes()),
          reinterpret_cast<const std::uint16_t *>(v.bytes()), mask_data,
          reinterpret_cast<std::uint16_t *>(y.mutable_bytes()),
          past_k != nullptr ? reinterpret_cast<const std::uint16_t *>(past_k->bytes()) : nullptr,
          past_v != nullptr ? reinterpret_cast<const std::uint16_t *>(past_v->bytes()) : nullptr,
          nonpad, qk_half_data);
    } else {
      ComputeAttentionFloat16Streaming(
          plan, reinterpret_cast<const std::uint16_t *>(q.bytes()),
          reinterpret_cast<const std::uint16_t *>(k.bytes()),
          reinterpret_cast<const std::uint16_t *>(v.bytes()), mask_data,
          reinterpret_cast<std::uint16_t *>(y.mutable_bytes()),
          past_k != nullptr ? reinterpret_cast<const std::uint16_t *>(past_k->bytes()) : nullptr,
          past_v != nullptr ? reinterpret_cast<const std::uint16_t *>(past_v->bytes()) : nullptr,
          nonpad);
    }
  } else {
    if (plan.has_qk_matmul_output || plan.softmax_fp64) {
      ComputeHalfAttentionMaterialized(
          plan, data_type, reinterpret_cast<const std::uint16_t *>(q.bytes()),
          reinterpret_cast<const std::uint16_t *>(k.bytes()),
          reinterpret_cast<const std::uint16_t *>(v.bytes()), mask_data,
          reinterpret_cast<std::uint16_t *>(y.mutable_bytes()),
          past_k != nullptr ? reinterpret_cast<const std::uint16_t *>(past_k->bytes()) : nullptr,
          past_v != nullptr ? reinterpret_cast<const std::uint16_t *>(past_v->bytes()) : nullptr,
          nonpad, qk_half_data);
    } else {
      ComputeAttentionBFloat16Streaming(
          plan, reinterpret_cast<const std::uint16_t *>(q.bytes()),
          reinterpret_cast<const std::uint16_t *>(k.bytes()),
          reinterpret_cast<const std::uint16_t *>(v.bytes()), mask_data,
          reinterpret_cast<std::uint16_t *>(y.mutable_bytes()),
          past_k != nullptr ? reinterpret_cast<const std::uint16_t *>(past_k->bytes()) : nullptr,
          past_v != nullptr ? reinterpret_cast<const std::uint16_t *>(past_v->bytes()) : nullptr,
          nonpad);
    }
  }
  if (qk_output != nullptr && plan.has_qk_matmul_output) {
    *qk_output = std::move(qk);
  }

  // --- Construct present_key / present_value outputs ---
  // These are data copies (concat(past, current) along the sequence axis),
  // always rank-4 (batch, kv_num_heads, total_kv_length, head_dim), in the
  // original Q/K/V data type.
  if (plan.has_present_output) {
    auto build_present = [&](int output_slot, const Tensor *past_tensor,
                             const Tensor &current_tensor,
                             const AttentionPlan::TensorStrides &past_strides,
                             const AttentionPlan::TensorStrides &current_strides, std::size_t dim,
                             Tensor *out) {
      const std::vector<std::int64_t> present_shape = {
          static_cast<std::int64_t>(plan.batch), static_cast<std::int64_t>(plan.kv_num_heads),
          static_cast<std::int64_t>(plan.total_kv_length), static_cast<std::int64_t>(dim)};
      Shape shape_s(present_shape);
      const std::size_t total_elements =
          plan.batch * plan.kv_num_heads * plan.total_kv_length * dim;
      const std::size_t total_bytes = total_elements * element_bytes;
      Tensor present = rt != nullptr
                           ? rt->MakeOutputTensor(output_slot, q.data_type, shape_s, total_bytes)
                           : rt_ns::MakeOutputTensor(q.data_type, shape_s, total_bytes, nullptr);
      // present is contiguous rank-4: (batch, kv_num_heads, total_kv_length, dim)
      for (std::size_t b = 0; b < plan.batch; ++b) {
        for (std::size_t h = 0; h < plan.kv_num_heads; ++h) {
          const std::size_t dst_offset = (b * plan.kv_num_heads + h) * plan.total_kv_length * dim;
          // Past segment
          if (past_tensor != nullptr && plan.past_length > 0) {
            const std::size_t src_offset =
                static_cast<std::size_t>(b * past_strides.batch + h * past_strides.head);
            for (std::size_t s = 0; s < plan.past_length; ++s) {
              const std::size_t from =
                  src_offset + s * static_cast<std::size_t>(past_strides.sequence);
              const std::size_t to = dst_offset + s * dim;
              std::memcpy(reinterpret_cast<char *>(present.mutable_bytes()) + to * element_bytes,
                          reinterpret_cast<const char *>(past_tensor->bytes()) +
                              from * element_bytes,
                          dim * element_bytes);
            }
          }
          // Current segment
          const std::size_t cur_offset =
              static_cast<std::size_t>(b * current_strides.batch + h * current_strides.head);
          for (std::size_t s = 0; s < plan.kv_length; ++s) {
            const std::size_t from =
                cur_offset + s * static_cast<std::size_t>(current_strides.sequence);
            const std::size_t to = dst_offset + (plan.past_length + s) * dim;
            std::memcpy(reinterpret_cast<char *>(present.mutable_bytes()) + to * element_bytes,
                        reinterpret_cast<const char *>(current_tensor.bytes()) +
                            from * element_bytes,
                        dim * element_bytes);
          }
        }
      }
      if (out != nullptr) {
        *out = std::move(present);
      }
    };

    build_present(1, past_k, k, plan.past_k_strides, plan.k_strides, plan.head_dim,
                  present_key_output);
    build_present(2, past_v, v, plan.past_v_strides, plan.v_strides, plan.v_head_dim,
                  present_value_output);
  }

  return y;
}

} // namespace

AttentionKernel::AttentionKernel(const rt_ns::KernelContext &ctx) : KernelBase(ctx) {}

Tensor AttentionKernel::operator()(const NodeProto &node, const Tensor &q, const Tensor &k,
                                   const Tensor &v, const Tensor *mask, RuntimeContext *rt,
                                   const Tensor *past_k, const Tensor *past_v,
                                   const Tensor *nonpad_kv_seqlen) const {
  const AttentionDescriptor descriptor = BuildDescriptor(node);
  Tensor qk_output, present_key, present_value;
  return Compute(q, k, v, mask, past_k, past_v, nonpad_kv_seqlen, descriptor, rt, &qk_output,
                 &present_key, &present_value);
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
  const Tensor *past_k =
      descriptor.has_past_key ? rt_ns::GetOptionalInput(node, 4, rt.tensors()) : nullptr;
  const Tensor *past_v =
      descriptor.has_past_value ? rt_ns::GetOptionalInput(node, 5, rt.tensors()) : nullptr;
  const Tensor *nonpad =
      descriptor.has_nonpad_kv_seqlen ? rt_ns::GetOptionalInput(node, 6, rt.tensors()) : nullptr;
  Tensor qk_output, present_key, present_value;
  Tensor y = Compute(q, k, v, mask, past_k, past_v, nonpad, descriptor, &rt, &qk_output,
                     &present_key, &present_value);
  rt_ns::SetOutput(node, 0, std::move(y), rt);
  if (descriptor.has_present_key) {
    rt_ns::SetOutput(node, 1, std::move(present_key), rt);
  }
  if (descriptor.has_present_value) {
    rt_ns::SetOutput(node, 2, std::move(present_value), rt);
  }
  if (descriptor.has_qk_matmul_output) {
    rt_ns::SetOutput(node, 3, std::move(qk_output), rt);
  }
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
  info.types = {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 23;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
