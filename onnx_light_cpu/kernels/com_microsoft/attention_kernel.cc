// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/attention_kernel.h"
#include "ml_cpu_compute/checked_arithmetic.h"
#include "ml_cpu_compute/com_microsoft/attention.h"
#include "ml_cpu_compute/execution.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_light_cpu/kernels/execution.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace onnx_light_cpu {
namespace {
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;
using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
bool Wired(const NodeProto &node, int slot, bool output = false) {
  return output ? node.output_size() > slot && !node.output(slot).empty()
                : node.input_size() > slot && !node.input(slot).empty();
}
const float *Floats(const Tensor *tensor) {
  return tensor ? reinterpret_cast<const float *>(tensor->bytes()) : nullptr;
}
} // namespace

MicrosoftAttentionKernel::MicrosoftAttentionKernel(const NodeProto &node,
                                                   const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
}

void MicrosoftAttentionKernel::Run(rt_ns::RuntimeContext &rt) {
  const RuntimeExecutionScope execution_scope(&rt);
  rt.RecordKernelUsage(kName);
  const auto &node = *node_;
  const auto fail = [](const char *message) {
    throw std::invalid_argument(std::string("com.microsoft Attention: ") + message);
  };
  const auto attr = [&](const char *name, int64_t fallback = 0) {
    return rt_ns::GetAttributeIntOrDefault(node, name, fallback);
  };
  if (!Wired(node, 0) || !Wired(node, 1) || !Wired(node, 0, true) || node.input_size() > 7 ||
      node.output_size() > 2 || Wired(node, 6) || attr("past_present_share_buffer") != 0 ||
      attr("do_rotary") != 0 || rt_ns::FindAttribute(node, "rotary_embedding_dim"))
    fail("requires input, weights and output; shared buffers and rotary are unsupported.");
  const auto optional = [&](int slot) -> const Tensor * {
    return Wired(node, slot) ? &rt_ns::GetInput(node, slot, rt) : nullptr;
  };
  const Tensor &x = *optional(0), &w = *optional(1);
  const Tensor *bias = optional(2), *mask = optional(3), *past = optional(4), *ab = optional(5);
  for (const auto *tensor : {&x, &w, bias, past, ab}) {
    if (tensor && tensor->data_type != static_cast<int>(DataType::FLOAT))
      fail("only matching FLOAT projection, bias and cache tensors are supported.");
    if (tensor)
      tensor->shape.product(0, tensor->shape.size(), "MicrosoftAttention input");
  }
  if (x.shape.size() != 3 || w.shape.size() != 2 || x.shape[2] <= 0 || w.shape[0] != x.shape[2] ||
      w.shape[1] <= 0)
    fail("input must be [B,S,I] and weights [I,Q+K+V] with positive hidden dimensions.");
  const int64_t b = x.shape[0], s = x.shape[1], width = w.shape[1], heads = attr("num_heads");
  int64_t q = width / 3, v = q;
  if (const auto *sizes = rt_ns::FindAttribute(node, "qkv_hidden_sizes")) {
    if (sizes->ints().size() != 3)
      fail("qkv_hidden_sizes must contain Q,K,V.");
    q = sizes->ints(0);
    v = sizes->ints(2);
    if (sizes->ints(1) != q ||
        ml_cpu_compute::CheckedIndexAdd(
            ml_cpu_compute::CheckedIndexMultiply(2, q, "MicrosoftAttention", "QK width"), v,
            "MicrosoftAttention", "QKV width") != width)
      fail("Q and K widths must match and Q+K+V must equal the weights width.");
  } else if (width % 3 != 0) {
    fail("weights width must be divisible by three without qkv_hidden_sizes.");
  }
  const auto causal = attr("unidirectional");
  if (heads <= 0 || q <= 0 || v <= 0 || q % heads || v % heads || b < 0 || s < 0 ||
      (causal != 0 && causal != 1))
    fail("invalid heads, hidden sizes, dimensions or unidirectional flag.");
  const auto h = q / heads;
  if (bias && bias->shape != Shape{width})
    fail("projection bias must have shape [Q+K+V].");
  const bool present_requested = Wired(node, 1, true);
  if ((past && !present_requested) || ((past || present_requested) && q != v))
    fail("past requires present, and packed cache requires equal Q/K/V widths.");
  const auto p = past ? (past->shape.size() == 5 ? past->shape[3] : -1) : 0;
  const auto t = ml_cpu_compute::CheckedIndexAdd(p, s, "MicrosoftAttention", "total length");
  if (p < 0 || (past && past->shape != Shape{2, b, heads, p, h}))
    fail("past must have shape [2,B,N,P,H].");
  int mask_rank = 0;
  if (mask) {
    mask_rank = mask->shape.size();
    if (mask_rank == 4 && causal)
      fail("4D masks require unidirectional=0.");
    mask->shape.product(0, mask_rank, "MicrosoftAttention mask");
    if (mask->data_type != static_cast<int>(DataType::INT32) ||
        !((mask_rank == 1 && (mask->shape[0] == b ||
                              mask->shape[0] == ml_cpu_compute::CheckedIndexMultiply(
                                                    2, b, "MicrosoftAttention", "mask lengths"))) ||
          mask->shape == Shape{b, t} || mask->shape == Shape{b, s, t} ||
          (mask_rank == 4 && mask->shape[0] == b && mask->shape[1] == 1 &&
           mask->shape[2] == mask->shape[3] && mask->shape[2] >= t)))
      fail("mask must be int32 [B], [2B], [B,T], [B,S,T], or [B,1,M,M], M>=T.");
    if (mask_rank == 1) {
      const auto *values = reinterpret_cast<const int32_t *>(mask->bytes());
      for (int64_t batch = 0; batch < b; ++batch) {
        const auto start = mask->shape[0] == b ? 0 : values[b + batch];
        if (start < 0 || values[batch] < start || values[batch] > t)
          fail("mask length/start positions must satisfy 0<=start<=end<=T.");
      }
    }
  }
  if (ab &&
      (ab->shape.size() != 4 || (ab->shape[0] != 1 && ab->shape[0] != b) ||
       (ab->shape[1] != 1 && ab->shape[1] != heads) || ab->shape[2] != s || ab->shape[3] != t))
    fail("attention_bias must be [1 or B,1 or N,S,T].");
  const auto scale = rt_ns::GetAttributeFloatOrDefault(node, "scale", 1.f / std::sqrt(float(h)));
  const auto filter = rt_ns::GetAttributeFloatOrDefault(node, "mask_filter_value", -10000.f);
  if (!std::isfinite(scale) || !std::isfinite(filter))
    fail("scale and mask_filter_value must be finite.");
  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.q_num_heads = descriptor.kv_num_heads = heads;
  descriptor.scale = scale;
  descriptor.has_past_key = descriptor.has_past_value = past != nullptr;
  const bool additive = mask || ab || causal;
  descriptor.has_attn_mask = additive;
  const std::vector<int64_t> qs{b, s, q}, vs{b, s, v}, ms{b, heads, s, t};
  const std::vector<int64_t> ps =
      past ? std::vector<int64_t>{b, heads, p, h} : std::vector<int64_t>{};
  ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank3, qs, qs,
                                     vs, additive ? ms : std::vector<int64_t>{},
                                     additive ? ml_cpu_compute::AttentionMaskKind::kAdditive
                                              : ml_cpu_compute::AttentionMaskKind::kNone,
                                     ps, ps);
  const auto mask_count =
      additive ? ml_cpu_compute::CheckedDimension(ml_cpu_compute::CheckedShapeIndexProduct(
                                                      ms, "MicrosoftAttention", "additive mask"),
                                                  "MicrosoftAttention", "additive mask")
               : 0;
  const auto work = ml_cpu_compute::CheckedStride(mask_count, "MicrosoftAttention", "mask work");
  ml_cpu_compute::ExecutionBuffer mask_buffer(ml_cpu_compute::CheckedByteSize(
      mask_count, sizeof(float), "MicrosoftAttention", "mask scratch"));
  auto *additive_mask = static_cast<float *>(mask_buffer.data());
  const auto *raw = mask ? reinterpret_cast<const int32_t *>(mask->bytes()) : nullptr;
  const auto *attention_bias = Floats(ab);
  ml_cpu_compute::ExecuteRanges(work, 3., [&](int64_t begin, int64_t end) {
    for (auto i = begin; i < end; ++i) {
      const auto key = i % t, query = (i / t) % s, head = (i / (t * s)) % heads;
      const auto batch = i / (t * s * heads);
      bool allowed = true;
      if (mask_rank == 1)
        allowed = key < raw[batch] && (mask->shape[0] == b || key >= raw[b + batch]);
      else if (mask_rank == 2)
        allowed = raw[batch * t + key] > 0;
      else if (mask_rank == 3)
        allowed = raw[(batch * s + query) * t + key] > 0;
      else if (mask_rank == 4) {
        const auto m = mask->shape[2];
        allowed = raw[(batch * m + p + query) * m + key] > 0;
      }
      float value = allowed ? 0.f : filter;
      if (causal && key > p + query)
        value = std::numeric_limits<float>::lowest();
      if (ab)
        value += attention_bias[((ab->shape[0] == 1 ? 0 : batch) * ab->shape[1] +
                                 (ab->shape[1] == 1 ? 0 : head)) *
                                    s * t +
                                query * t + key];
      additive_mask[i] = value;
    }
  });
  const auto make_output = [&](int slot, Shape shape) {
    const auto count = ml_cpu_compute::CheckedDimension(
        shape.product(0, shape.size(), "MicrosoftAttention output"), "MicrosoftAttention",
        "output elements");
    return rt.MakeOutputTensor(slot, DataType::FLOAT, shape,
                               ml_cpu_compute::CheckedByteSize(
                                   count, sizeof(float), "MicrosoftAttention", "output bytes"));
  };
  auto output = make_output(0, {b, s, v});
  Tensor present;
  if (present_requested)
    present = make_output(1, {2, b, heads, t, h});
  ml_cpu_compute::ComputeMicrosoftAttentionFloat32(
      plan, x.shape[2], Floats(&x), Floats(&w), Floats(bias), additive_mask, Floats(past),
      reinterpret_cast<float *>(output.mutable_bytes()),
      present_requested ? reinterpret_cast<float *>(present.mutable_bytes()) : nullptr);
  rt_ns::SetOutput(node, 0, std::move(output), rt);
  if (present_requested)
    rt_ns::SetOutput(node, 1, std::move(present), rt);
}

void RegisterMicrosoftAttentionKernel() {
  rt_ns::NodeKernelFn factory =
      [](const NodeProto &node, rt_ns::RuntimeContext &,
         const rt_ns::KernelContext &ctx) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<MicrosoftAttentionKernel>(node, ctx);
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "Attention";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = MicrosoftAttentionKernel::kName;
  info.types = {DataType::FLOAT};
  info.since_version = info.until_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}
} // namespace onnx_light_cpu
