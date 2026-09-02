// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/attention/linear_attention_kernel.h"

#include "onnx_light_cpu/impl/attention/linear_attention.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Shape;
using rt_ns::Tensor;

bool IsSupportedType(DataType type) {
  return type == DataType::FLOAT || type == DataType::FLOAT16 || type == DataType::BFLOAT16;
}

std::size_t ElementWidth(DataType type) {
  return type == DataType::FLOAT ? sizeof(float) : sizeof(std::uint16_t);
}

std::size_t CheckedMultiply(std::size_t left, std::size_t right, const char *label) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::invalid_argument(std::string(LinearAttentionKernel::kName) + ": " + label +
                                " element count overflows size_t.");
  }
  return left * right;
}

std::size_t CheckedCount(const Shape &shape, const char *label) {
  std::size_t count = 1;
  for (std::int64_t dimension : shape) {
    if (dimension < 0) {
      throw std::invalid_argument(std::string(LinearAttentionKernel::kName) + ": " + label +
                                  " has a negative dimension.");
    }
    count = CheckedMultiply(count, static_cast<std::size_t>(dimension), label);
  }
  return count;
}

void RequirePacked3D(const Tensor &tensor, const char *label) {
  if (!IsSupportedType(static_cast<DataType>(tensor.data_type)) || tensor.shape.size() != 3) {
    throw std::invalid_argument(std::string(LinearAttentionKernel::kName) + ": " + label +
                                " must be a rank-3 FLOAT, FLOAT16, or BFLOAT16 tensor.");
  }
  CheckedCount(tensor.shape, label);
}

LinearAttentionRule ParseRule(const std::string &rule) {
  if (rule == "linear") {
    return LinearAttentionRule::kLinear;
  }
  if (rule == "gated") {
    return LinearAttentionRule::kGated;
  }
  if (rule == "delta") {
    return LinearAttentionRule::kDelta;
  }
  if (rule == "gated_delta") {
    return LinearAttentionRule::kGatedDelta;
  }
  throw std::invalid_argument(std::string(LinearAttentionKernel::kName) +
                              ": update_rule must be linear, gated, delta, or gated_delta.");
}

bool UsesDecay(LinearAttentionRule rule) {
  return rule == LinearAttentionRule::kGated || rule == LinearAttentionRule::kGatedDelta;
}

bool UsesBeta(LinearAttentionRule rule) {
  return rule == LinearAttentionRule::kDelta || rule == LinearAttentionRule::kGatedDelta;
}

void ConvertToFloat(const Tensor &source, std::size_t count, float *destination) {
  const DataType type = static_cast<DataType>(source.data_type);
  if (type == DataType::FLOAT) {
    std::memcpy(destination, source.bytes(), count * sizeof(float));
  } else if (type == DataType::FLOAT16) {
    detail::ConvertFloat16ToFloat32(reinterpret_cast<const std::uint16_t *>(source.bytes()),
                                    destination, count);
  } else {
    detail::ConvertBFloat16ToFloat32(reinterpret_cast<const std::uint16_t *>(source.bytes()),
                                     destination, count);
  }
}

void ConvertFromFloat(const float *source, std::size_t count, DataType type, Tensor &destination) {
  if (type == DataType::FLOAT) {
    std::memcpy(destination.mutable_bytes(), source, count * sizeof(float));
  } else if (type == DataType::FLOAT16) {
    detail::ConvertFloat32ToFloat16(
        source, reinterpret_cast<std::uint16_t *>(destination.mutable_bytes()), count);
  } else {
    detail::ConvertFloat32ToBFloat16(
        source, reinterpret_cast<std::uint16_t *>(destination.mutable_bytes()), count);
  }
}

Tensor AllocateTensor(RuntimeContext *rt, int output_index, bool is_output, DataType type,
                      const Shape &shape, std::size_t count) {
  const std::size_t bytes = CheckedMultiply(count, ElementWidth(type), "output");
  if (rt == nullptr) {
    return rt_ns::MakeOutputTensor(static_cast<std::int32_t>(type), shape, bytes, nullptr);
  }
  if (is_output) {
    return rt->MakeOutputTensor(output_index, static_cast<std::int32_t>(type), shape, bytes);
  }
  return rt->MakeTemporaryTensor(static_cast<std::int32_t>(type), shape, bytes);
}

} // namespace

LinearAttentionKernel::Result
LinearAttentionKernel::operator()(const Tensor &query, const Tensor &key, const Tensor &value,
                                  const Attributes &attributes, const Tensor *past_state,
                                  const Tensor *decay, const Tensor *beta, RuntimeContext *rt,
                                  bool has_state_output) const {
  RequirePacked3D(query, "query");
  RequirePacked3D(key, "key");
  RequirePacked3D(value, "value");
  if (key.data_type != query.data_type || value.data_type != query.data_type) {
    throw std::invalid_argument(std::string(kName) + ": query, key, and value types must match.");
  }
  if (query.shape[0] != key.shape[0] || query.shape[0] != value.shape[0] ||
      query.shape[1] != key.shape[1] || query.shape[1] != value.shape[1]) {
    throw std::invalid_argument(
        std::string(kName) + ": query, key, and value batch and sequence dimensions must match.");
  }
  if (attributes.query_heads <= 0 || attributes.key_value_heads <= 0) {
    throw std::invalid_argument(std::string(kName) +
                                ": q_num_heads and kv_num_heads must be positive.");
  }

  const std::size_t batch = static_cast<std::size_t>(query.shape[0]);
  const std::size_t sequence = static_cast<std::size_t>(query.shape[1]);
  const std::size_t query_hidden = static_cast<std::size_t>(query.shape[2]);
  const std::size_t key_hidden = static_cast<std::size_t>(key.shape[2]);
  const std::size_t value_hidden = static_cast<std::size_t>(value.shape[2]);
  const std::size_t query_heads = static_cast<std::size_t>(attributes.query_heads);
  const std::size_t key_value_heads = static_cast<std::size_t>(attributes.key_value_heads);
  if (query_hidden % query_heads != 0 || query_hidden == 0) {
    throw std::invalid_argument(std::string(kName) +
                                ": query hidden size must be divisible by q_num_heads.");
  }
  const std::size_t key_head_size = query_hidden / query_heads;
  if (key_hidden == 0 || key_hidden % key_value_heads != 0 ||
      key_hidden / key_value_heads != key_head_size) {
    throw std::invalid_argument(
        std::string(kName) + ": key must contain kv_num_heads heads matching the query head size.");
  }
  if (value_hidden == 0 || value_hidden % key_value_heads != 0) {
    throw std::invalid_argument(std::string(kName) +
                                ": value hidden size must be divisible by kv_num_heads.");
  }
  if (query_heads % key_value_heads != 0) {
    throw std::invalid_argument(std::string(kName) +
                                ": q_num_heads must be divisible by kv_num_heads.");
  }

  const std::size_t value_head_size = value_hidden / key_value_heads;
  const LinearAttentionRule rule = ParseRule(attributes.update_rule);
  if ((decay != nullptr) != UsesDecay(rule)) {
    throw std::invalid_argument(std::string(kName) +
                                ": decay presence does not match update_rule.");
  }
  if ((beta != nullptr) != UsesBeta(rule)) {
    throw std::invalid_argument(std::string(kName) + ": beta presence does not match update_rule.");
  }
  LinearAttentionDecayLayout decay_layout = LinearAttentionDecayLayout::kNone;
  if (UsesDecay(rule)) {
    if (decay == nullptr || decay->data_type != query.data_type || decay->shape.size() != 3 ||
        decay->shape[0] != query.shape[0] || decay->shape[1] != query.shape[1]) {
      throw std::invalid_argument(
          std::string(kName) + ": decay must match the activation type and have shape (B,T,...).");
    }
    const std::size_t decay_hidden = static_cast<std::size_t>(decay->shape[2]);
    if (decay_hidden == key_value_heads) {
      decay_layout = LinearAttentionDecayLayout::kPerHead;
    } else if (decay_hidden == CheckedMultiply(key_value_heads, key_head_size, "decay")) {
      decay_layout = LinearAttentionDecayLayout::kPerKeyDimension;
    } else {
      throw std::invalid_argument(std::string(kName) +
                                  ": decay last dimension must be H_kv or H_kv*d_k.");
    }
  }

  LinearAttentionBetaLayout beta_layout = LinearAttentionBetaLayout::kNone;
  if (UsesBeta(rule)) {
    if (beta == nullptr || beta->data_type != query.data_type || beta->shape.size() != 3 ||
        beta->shape[0] != query.shape[0] || beta->shape[1] != query.shape[1]) {
      throw std::invalid_argument(
          std::string(kName) + ": beta must match the activation type and have shape (B,T,...).");
    }
    const std::size_t beta_hidden = static_cast<std::size_t>(beta->shape[2]);
    if (beta_hidden == 1) {
      beta_layout = LinearAttentionBetaLayout::kShared;
    } else if (beta_hidden == key_value_heads) {
      beta_layout = LinearAttentionBetaLayout::kPerHead;
    } else {
      throw std::invalid_argument(std::string(kName) + ": beta last dimension must be 1 or H_kv.");
    }
  }

  const DataType activation_type = static_cast<DataType>(query.data_type);
  const DataType state_type =
      past_state == nullptr ? activation_type : static_cast<DataType>(past_state->data_type);
  if (!IsSupportedType(state_type)) {
    throw std::invalid_argument(std::string(kName) +
                                ": past_state must be FLOAT, FLOAT16, or BFLOAT16.");
  }
  const Shape state_shape{
      static_cast<std::int64_t>(batch), static_cast<std::int64_t>(key_value_heads),
      static_cast<std::int64_t>(key_head_size), static_cast<std::int64_t>(value_head_size)};
  const std::size_t state_count =
      CheckedMultiply(CheckedMultiply(batch, key_value_heads, "state"),
                      CheckedMultiply(key_head_size, value_head_size, "state"), "state");
  if (past_state != nullptr && past_state->shape != state_shape) {
    throw std::invalid_argument(std::string(kName) +
                                ": past_state shape must be (B,H_kv,d_k,d_v).");
  }

  const std::size_t output_hidden = CheckedMultiply(query_heads, value_head_size, "output");
  if (output_hidden > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument(std::string(kName) + ": output hidden size exceeds int64_t.");
  }
  const Shape output_shape{static_cast<std::int64_t>(batch), static_cast<std::int64_t>(sequence),
                           static_cast<std::int64_t>(output_hidden)};
  const std::size_t output_count =
      CheckedMultiply(CheckedMultiply(batch, sequence, "output"), output_hidden, "output");
  Tensor output = AllocateTensor(rt, 0, true, activation_type, output_shape, output_count);
  Tensor present_state =
      AllocateTensor(rt, 1, has_state_output, state_type, state_shape, state_count);

  std::vector<float> query_buffer;
  std::vector<float> key_buffer;
  std::vector<float> value_buffer;
  std::vector<float> decay_buffer;
  std::vector<float> beta_buffer;
  std::vector<float> state_buffer;
  std::vector<float> output_buffer;
  auto as_float = [](const Tensor &tensor, std::size_t count,
                     std::vector<float> &buffer) -> const float * {
    if (static_cast<DataType>(tensor.data_type) == DataType::FLOAT) {
      return reinterpret_cast<const float *>(tensor.bytes());
    }
    buffer.resize(count);
    ConvertToFloat(tensor, count, buffer.data());
    return buffer.data();
  };

  const float *query_data = as_float(query, CheckedCount(query.shape, "query"), query_buffer);
  const float *key_data = as_float(key, CheckedCount(key.shape, "key"), key_buffer);
  const float *value_data = as_float(value, CheckedCount(value.shape, "value"), value_buffer);
  const float *decay_data =
      decay_layout == LinearAttentionDecayLayout::kNone
          ? nullptr
          : as_float(*decay, CheckedCount(decay->shape, "decay"), decay_buffer);
  const float *beta_data = beta_layout == LinearAttentionBetaLayout::kNone
                               ? nullptr
                               : as_float(*beta, CheckedCount(beta->shape, "beta"), beta_buffer);

  float *state_data;
  if (state_type == DataType::FLOAT) {
    state_data = reinterpret_cast<float *>(present_state.mutable_bytes());
    if (past_state == nullptr) {
      std::fill(state_data, state_data + state_count, 0.0f);
    } else if (past_state->bytes() != present_state.bytes()) {
      std::memcpy(state_data, past_state->bytes(), state_count * sizeof(float));
    }
  } else {
    state_buffer.resize(state_count);
    state_data = state_buffer.data();
    if (past_state == nullptr) {
      std::fill(state_data, state_data + state_count, 0.0f);
    } else {
      ConvertToFloat(*past_state, state_count, state_data);
    }
  }

  float *output_data;
  if (activation_type == DataType::FLOAT) {
    output_data = reinterpret_cast<float *>(output.mutable_bytes());
  } else {
    output_buffer.resize(output_count);
    output_data = output_buffer.data();
  }

  LinearAttentionParameters parameters;
  parameters.batch_size = batch;
  parameters.sequence_length = sequence;
  parameters.query_heads = query_heads;
  parameters.key_value_heads = key_value_heads;
  parameters.key_head_size = key_head_size;
  parameters.value_head_size = value_head_size;
  parameters.rule = rule;
  parameters.decay_layout = decay_layout;
  parameters.beta_layout = beta_layout;
  parameters.scale = attributes.scale == 0.0f ? 1.0f / std::sqrt(static_cast<float>(key_head_size))
                                              : attributes.scale;
  LinearAttentionFloat32(parameters, query_data, key_data, value_data, decay_data, beta_data,
                         state_data, output_data);

  if (activation_type != DataType::FLOAT) {
    ConvertFromFloat(output_data, output_count, activation_type, output);
  }
  if (state_type != DataType::FLOAT) {
    ConvertFromFloat(state_data, state_count, state_type, present_state);
  }
  return {std::move(output), std::move(present_state)};
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
  info.types = {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 27;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
