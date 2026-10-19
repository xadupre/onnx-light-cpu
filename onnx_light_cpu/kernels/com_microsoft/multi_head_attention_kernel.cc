// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/multi_head_attention_kernel.h"
#include "onnx_light_cpu/kernels/execution.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu_compute/checked_arithmetic.h"
#include "onnx_light_cpu_compute/execution.h"
#include "onnx_light_cpu_compute/math/half_conversion.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace onnx_light_cpu {
namespace {
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
using rt_ns::DataType;
using rt_ns::Tensor;
using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;

bool Wired(const NodeProto &node, int slot, bool output = false) {
  return output ? node.output_size() > slot && !node.output(slot).empty()
                : node.input_size() > slot && !node.input(slot).empty();
}

float Read(const Tensor &tensor, std::size_t index) {
  switch (static_cast<DataType>(tensor.data_type)) {
  case DataType::FLOAT:
    return reinterpret_cast<const float *>(tensor.bytes())[index];
  case DataType::FLOAT16:
    return detail::Float16BitsToFloat(
        reinterpret_cast<const std::uint16_t *>(tensor.bytes())[index]);
  case DataType::BFLOAT16:
    return detail::Bfloat16BitsToFloat(
        reinterpret_cast<const std::uint16_t *>(tensor.bytes())[index]);
  default:
    throw std::invalid_argument("MultiHeadAttention: unsupported floating-point type.");
  }
}

void Write(Tensor &tensor, std::size_t index, float value) {
  switch (static_cast<DataType>(tensor.data_type)) {
  case DataType::FLOAT:
    reinterpret_cast<float *>(tensor.mutable_bytes())[index] = value;
    break;
  case DataType::FLOAT16:
    reinterpret_cast<std::uint16_t *>(tensor.mutable_bytes())[index] =
        detail::FloatToFloat16Bits(value);
    break;
  case DataType::BFLOAT16:
    reinterpret_cast<std::uint16_t *>(tensor.mutable_bytes())[index] =
        detail::FloatToBFloat16Bits(value);
    break;
  default:
    throw std::invalid_argument("MultiHeadAttention: unsupported floating-point type.");
  }
}

std::int64_t AttributeInt(const NodeProto &node, const char *name, std::int64_t fallback) {
  return rt_ns::GetAttributeIntOrDefault(node, name, fallback);
}

Tensor Output(rt_ns::RuntimeContext &rt, int slot, DataType dtype, rt_ns::Shape shape) {
  const std::size_t count =
      CheckedDimension(shape.product(0, shape.size(), "MultiHeadAttention output"),
                       "MultiHeadAttention", "output element count");
  return rt.MakeOutputTensor(slot, dtype, shape,
                             CheckedByteSize(count, dtype == DataType::FLOAT ? 4 : 2,
                                             "MultiHeadAttention", "output byte size"));
}

} // namespace

MultiHeadAttentionKernel::MultiHeadAttentionKernel(const NodeProto &node,
                                                   const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
}

void MultiHeadAttentionKernel::Run(rt_ns::RuntimeContext &rt) {
  const RuntimeExecutionScope execution_scope;
  rt.RecordKernelUsage(kName);
  const auto &node = *node_;
  if (!Wired(node, 0) || !Wired(node, 1) || !Wired(node, 2) || !Wired(node, 0, true) ||
      node.input_size() > 10 || node.output_size() > 3 || Wired(node, 8) || Wired(node, 9) ||
      Wired(node, 3, true)) {
    throw std::invalid_argument(
        "MultiHeadAttention: Q/K/V and output are required; packed QKV, shared buffers and qk "
        "output are unsupported.");
  }
  const Tensor &q = rt_ns::GetInput(node, 0, rt);
  const Tensor &k = rt_ns::GetInput(node, 1, rt);
  const Tensor &v = rt_ns::GetInput(node, 2, rt);
  const bool cross = k.shape.size() == 4;
  if (q.shape.size() != 3 || (k.shape.size() != 3 && !cross) || v.shape.size() != k.shape.size()) {
    throw std::invalid_argument("MultiHeadAttention: Q must be rank 3 and K/V rank 3 or 4.");
  }
  const auto dtype = static_cast<DataType>(q.data_type);
  if ((dtype != DataType::FLOAT && dtype != DataType::FLOAT16 && dtype != DataType::BFLOAT16) ||
      q.data_type != k.data_type || q.data_type != v.data_type) {
    throw std::invalid_argument("MultiHeadAttention: Q/K/V must have the same floating type.");
  }
  const std::int64_t heads = AttributeInt(node, "num_heads", 0);
  const std::int64_t causal = AttributeInt(node, "unidirectional", 0);
  if (heads <= 0 || (causal != 0 && causal != 1) || q.shape[2] <= 0 || q.shape[2] % heads != 0) {
    throw std::invalid_argument(
        "MultiHeadAttention: invalid head count, hidden size or causal flag.");
  }
  const std::int64_t b = q.shape[0], s = q.shape[1], h = q.shape[2] / heads;
  const std::int64_t l = cross ? k.shape[2] : k.shape[1];
  const std::int64_t hv = cross ? v.shape[3] : v.shape[2] / heads;
  const std::int64_t q_width = CheckedIndexMultiply(heads, h, "MultiHeadAttention", "query width");
  const std::int64_t v_width = CheckedIndexMultiply(heads, hv, "MultiHeadAttention", "value width");
  if (b < 0 || s < 0 || l < 0 || h <= 0 || hv <= 0 || k.shape[0] != b || v.shape[0] != b ||
      (cross ? (k.shape[1] != heads || v.shape[1] != heads || k.shape[3] != h || v.shape[2] != l)
             : (k.shape[1] != v.shape[1] || k.shape[2] != q.shape[2] || v.shape[2] % heads != 0))) {
    throw std::invalid_argument("MultiHeadAttention: incompatible Q/K/V dimensions.");
  }
  const auto optional = [&](int slot) -> const Tensor * {
    return Wired(node, slot) ? &rt_ns::GetInput(node, slot, rt) : nullptr;
  };
  const Tensor *bias = optional(3), *mask = optional(4), *attention_bias = optional(5);
  const Tensor *past_k = optional(6), *past_v = optional(7);
  if ((past_k == nullptr) != (past_v == nullptr)) {
    throw std::invalid_argument("MultiHeadAttention: past key/value must be paired.");
  }
  const std::int64_t p = past_k ? (past_k->shape.size() == 4 ? past_k->shape[2] : -1) : 0;
  if (p < 0) {
    throw std::invalid_argument("MultiHeadAttention: past key must have rank 4.");
  }
  const std::int64_t t = CheckedIndexAdd(p, l, "MultiHeadAttention", "total sequence length");
  if ((t == 0 && s != 0) || (cross && past_k) ||
      (past_k && (past_k->data_type != q.data_type || past_v->data_type != q.data_type ||
                  past_k->shape != rt_ns::Shape{b, heads, p, h} ||
                  past_v->shape != rt_ns::Shape{b, heads, p, hv}))) {
    throw std::invalid_argument("MultiHeadAttention: incompatible past key/value.");
  }
  const auto bias_width =
      CheckedIndexAdd(CheckedIndexMultiply(2, q_width, "MultiHeadAttention", "projection bias"),
                      v_width, "MultiHeadAttention", "projection bias");
  if (bias &&
      (bias->data_type != q.data_type || bias->shape.size() != 1 || bias->shape[0] != bias_width)) {
    throw std::invalid_argument("MultiHeadAttention: projection bias must be [2*N*H+N*Hv] "
                                "with the query type.");
  }
  if (mask) {
    if (mask->data_type != static_cast<int>(DataType::INT32) ||
        !(mask->shape == rt_ns::Shape{b} || mask->shape == rt_ns::Shape{b, t} ||
          mask->shape == rt_ns::Shape{b, s, t})) {
      throw std::invalid_argument("MultiHeadAttention: invalid int32 key padding mask shape.");
    }
  }
  if (attention_bias &&
      (attention_bias->data_type != q.data_type || attention_bias->shape.size() != 4 ||
       (attention_bias->shape[0] != 1 && attention_bias->shape[0] != b) ||
       (attention_bias->shape[1] != 1 && attention_bias->shape[1] != heads) ||
       attention_bias->shape[2] != s || attention_bias->shape[3] != t)) {
    throw std::invalid_argument("MultiHeadAttention: invalid attention_bias broadcast shape.");
  }
  const float scale = rt_ns::GetAttributeFloatOrDefault(node, "scale", 1.0f / std::sqrt(float(h)));
  const float filter = rt_ns::GetAttributeFloatOrDefault(node, "mask_filter_value", -10000.0f);
  if (!std::isfinite(scale) || std::isnan(filter) ||
      filter == std::numeric_limits<float>::infinity()) {
    throw std::invalid_argument("MultiHeadAttention: invalid scale or mask_filter_value.");
  }
  const auto count = [&](const Tensor &tensor) {
    tensor.shape.product(0, tensor.shape.size(), "MultiHeadAttention input");
  };
  for (const Tensor *tensor : {&q, &k, &v, bias, mask, attention_bias, past_k, past_v}) {
    if (tensor)
      count(*tensor);
  }
  Tensor output = Output(rt, 0, dtype, {b, s, v_width});
  Tensor present_k, present_v;
  if (Wired(node, 1, true)) {
    present_k = Output(rt, 1, dtype, {b, heads, t, h});
  }
  if (Wired(node, 2, true)) {
    present_v = Output(rt, 2, dtype, {b, heads, t, hv});
  }
  const auto key_at = [&](std::int64_t batch, std::int64_t head, std::int64_t pos,
                          std::int64_t feature, bool value) {
    const Tensor &current = value ? v : k;
    const Tensor *past = value ? past_v : past_k;
    const std::int64_t width = value ? hv : h;
    if (pos < p)
      return Read(*past, ((batch * heads + head) * p + pos) * width + feature);
    pos -= p;
    const std::int64_t offset = cross
                                    ? ((batch * heads + head) * l + pos) * width + feature
                                    : (batch * l + pos) * (heads * width) + head * width + feature;
    return Read(current, offset);
  };
  for (std::int64_t batch = 0; batch < b; ++batch) {
    for (std::int64_t head = 0; head < heads; ++head) {
      for (std::int64_t pos = 0; pos < t && (Wired(node, 1, true) || Wired(node, 2, true)); ++pos) {
        if (Wired(node, 1, true))
          for (std::int64_t feature = 0; feature < h; ++feature)
            Write(
                present_k, ((batch * heads + head) * t + pos) * h + feature,
                key_at(batch, head, pos, feature, false) +
                    (bias && !cross && pos >= p ? Read(*bias, heads * h + head * h + feature) : 0));
        if (Wired(node, 2, true))
          for (std::int64_t feature = 0; feature < hv; ++feature)
            Write(present_v, ((batch * heads + head) * t + pos) * hv + feature,
                  key_at(batch, head, pos, feature, true) +
                      (bias && !cross && pos >= p ? Read(*bias, 2 * heads * h + head * hv + feature)
                                                  : 0));
      }
    }
  }
  const std::int64_t rows =
      CheckedIndexMultiply(CheckedIndexMultiply(b, heads, "MultiHeadAttention", "rows"), s,
                           "MultiHeadAttention", "rows");
  if (rows == 0) {
    rt_ns::SetOutput(node, 0, std::move(output), rt);
    if (Wired(node, 1, true))
      rt_ns::SetOutput(node, 1, std::move(present_k), rt);
    if (Wired(node, 2, true))
      rt_ns::SetOutput(node, 2, std::move(present_v), rt);
    return;
  }
  const double cost = static_cast<double>(t) * (static_cast<double>(h) + hv);
  const std::int64_t blocks = ExecutionBlockCount(rows, cost);
  const std::int64_t block_size = rows / blocks + (rows % blocks != 0);
  const std::int64_t score_count =
      CheckedIndexMultiply(blocks, t, "MultiHeadAttention", "score scratch elements");
  rt_ns::detail::TemporaryTypedBuffer<float> score_scratch(
      CheckedDimension(score_count, "MultiHeadAttention", "score scratch elements"),
      rt.execution_allocator(), "MultiHeadAttention score scratch");
  ExecuteRanges(
      rows, ExecutionSchedule{1, 1, blocks, blocks}, [&](std::int64_t begin, std::int64_t end) {
        float *scores = score_scratch.data() + (begin / block_size) * t;
        for (std::int64_t row = begin; row < end; ++row) {
          const std::int64_t seq = row % s, head = row / s % heads, batch = row / s / heads;
          float maximum = -std::numeric_limits<float>::infinity();
          for (std::int64_t pos = 0; pos < t; ++pos) {
            float score = 0;
            for (std::int64_t feature = 0; feature < h; ++feature) {
              const float qf = Read(q, (batch * s + seq) * heads * h + head * h + feature) +
                               (bias ? Read(*bias, head * h + feature) : 0);
              const float kf =
                  key_at(batch, head, pos, feature, false) +
                  (bias && !cross && pos >= p ? Read(*bias, heads * h + head * h + feature) : 0);
              score += qf * kf;
            }
            score *= scale;
            if (attention_bias) {
              const std::int64_t ab = attention_bias->shape[0] == 1 ? 0 : batch;
              const std::int64_t ah = attention_bias->shape[1] == 1 ? 0 : head;
              score +=
                  Read(*attention_bias, ((ab * attention_bias->shape[1] + ah) * s + seq) * t + pos);
            }
            if (mask) {
              const bool invalid =
                  mask->shape.size() == 1
                      ? pos >= reinterpret_cast<const std::int32_t *>(mask->bytes())[batch]
                  : mask->shape.size() == 2
                      ? reinterpret_cast<const std::int32_t *>(mask->bytes())[batch * t + pos] == 0
                      : reinterpret_cast<const std::int32_t *>(
                            mask->bytes())[(batch * s + seq) * t + pos] == 0;
              if (invalid)
                score += filter;
            }
            if (causal && pos > p + seq)
              score += filter;
            scores[pos] = score;
            maximum = std::isnan(score) ? score : std::max(maximum, score);
          }
          float sum = 0;
          for (std::int64_t pos = 0; pos < t; ++pos) {
            scores[pos] = maximum == -std::numeric_limits<float>::infinity()
                              ? 0
                              : std::exp(scores[pos] - maximum);
            sum += scores[pos];
          }
          if (sum != 0)
            for (std::int64_t pos = 0; pos < t; ++pos)
              scores[pos] /= sum;
          for (std::int64_t feature = 0; feature < hv; ++feature) {
            float result = 0;
            for (std::int64_t pos = 0; pos < t; ++pos)
              result += scores[pos] * (key_at(batch, head, pos, feature, true) +
                                       (bias && !cross && pos >= p
                                            ? Read(*bias, 2 * heads * h + head * hv + feature)
                                            : 0));
            Write(output, (batch * s + seq) * heads * hv + head * hv + feature,
                  sum == 0 ? 0 : result);
          }
        }
      });
  rt_ns::SetOutput(node, 0, std::move(output), rt);
  if (Wired(node, 1, true)) {
    rt_ns::SetOutput(node, 1, std::move(present_k), rt);
  }
  if (Wired(node, 2, true)) {
    rt_ns::SetOutput(node, 2, std::move(present_v), rt);
  }
}

void RegisterMultiHeadAttentionKernel() {
  rt_ns::NodeKernelFn factory =
      [](const NodeProto &node, rt_ns::RuntimeContext &,
         const rt_ns::KernelContext &ctx) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<MultiHeadAttentionKernel>(node, ctx);
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "MultiHeadAttention";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = MultiHeadAttentionKernel::kName;
  info.types = {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}
} // namespace onnx_light_cpu
