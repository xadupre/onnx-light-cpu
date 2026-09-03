// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/attention/linear_attention_shared.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Shape;
using rt_ns::Tensor;

bool Contains(const std::vector<DataType> &types, DataType type) {
  return std::find(types.begin(), types.end(), type) != types.end();
}

std::string TypeListForMessage(const std::vector<DataType> &types) {
  std::string joined;
  for (std::size_t i = 0; i < types.size(); ++i) {
    if (i != 0) {
      joined += ", ";
    }
    switch (types[i]) {
    case DataType::FLOAT:
      joined += "FLOAT";
      break;
    case DataType::FLOAT16:
      joined += "FLOAT16";
      break;
    case DataType::BFLOAT16:
      joined += "BFLOAT16";
      break;
    default:
      joined += "?";
      break;
    }
  }
  return joined;
}

std::size_t CheckedCount(const std::string &kernel_name, const Shape &shape, const char *label) {
  std::size_t count = 1;
  for (std::int64_t dimension : shape) {
    if (dimension < 0) {
      throw std::invalid_argument(kernel_name + ": " + label + " has a negative dimension.");
    }
    count = LinearAttentionCheckedMultiply(kernel_name, count, static_cast<std::size_t>(dimension),
                                           label);
  }
  return count;
}

void RequirePacked3D(const std::string &kernel_name, const Tensor &tensor, const char *label,
                     const std::vector<DataType> &allowed_types) {
  if (!Contains(allowed_types, static_cast<DataType>(tensor.data_type)) ||
      tensor.shape.size() != 3) {
    throw std::invalid_argument(kernel_name + ": " + label + " must be a rank-3 " +
                                TypeListForMessage(allowed_types) + " tensor.");
  }
  CheckedCount(kernel_name, tensor.shape, label);
}

} // namespace

LinearAttentionRule ParseLinearAttentionRule(const std::string &kernel_name,
                                             const std::string &rule) {
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
  throw std::invalid_argument(kernel_name +
                              ": update_rule must be linear, gated, delta, or gated_delta.");
}

bool LinearAttentionRuleUsesDecay(LinearAttentionRule rule) {
  return rule == LinearAttentionRule::kGated || rule == LinearAttentionRule::kGatedDelta;
}

bool LinearAttentionRuleUsesBeta(LinearAttentionRule rule) {
  return rule == LinearAttentionRule::kDelta || rule == LinearAttentionRule::kGatedDelta;
}

std::size_t LinearAttentionCheckedMultiply(const std::string &kernel_name, std::size_t left,
                                           std::size_t right, const char *label) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::invalid_argument(kernel_name + ": " + label + " element count overflows size_t.");
  }
  return left * right;
}

Tensor AllocateLinearAttentionTensor(const std::string &kernel_name, RuntimeContext *rt,
                                     int output_index, bool is_output, DataType type,
                                     const Shape &shape, std::size_t count) {
  const std::size_t width = type == DataType::FLOAT ? sizeof(float) : sizeof(std::uint16_t);
  const std::size_t bytes = LinearAttentionCheckedMultiply(kernel_name, count, width, "output");
  if (rt == nullptr) {
    return rt_ns::MakeOutputTensor(static_cast<std::int32_t>(type), shape, bytes, nullptr);
  }
  if (is_output) {
    return rt->MakeOutputTensor(output_index, static_cast<std::int32_t>(type), shape, bytes);
  }
  return rt->MakeTemporaryTensor(static_cast<std::int32_t>(type), shape, bytes);
}

void ConvertLinearAttentionToFloat(const Tensor &source, std::size_t count, float *destination) {
  if (count == 0) {
    return;
  }
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

void ConvertLinearAttentionFromFloat(const float *source, std::size_t count, DataType type,
                                     Tensor &destination) {
  if (count == 0) {
    return;
  }
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

LinearAttentionPlan PlanLinearAttention(const std::string &kernel_name, const Tensor &query,
                                        const Tensor &key, const Tensor &value,
                                        const Tensor *past_state, const Tensor *decay,
                                        const Tensor *beta, const std::string &update_rule,
                                        std::int64_t q_num_heads, std::int64_t kv_num_heads,
                                        float scale, const LinearAttentionGroupingPolicy &policy,
                                        const std::vector<DataType> &allowed_types) {
  RequirePacked3D(kernel_name, query, "query", allowed_types);
  RequirePacked3D(kernel_name, key, "key", allowed_types);
  RequirePacked3D(kernel_name, value, "value", allowed_types);
  if (key.data_type != query.data_type || value.data_type != query.data_type) {
    throw std::invalid_argument(kernel_name + ": query, key, and value types must match.");
  }
  if (query.shape[0] != key.shape[0] || query.shape[0] != value.shape[0] ||
      query.shape[1] != key.shape[1] || query.shape[1] != value.shape[1]) {
    throw std::invalid_argument(
        kernel_name + ": query, key, and value batch and sequence dimensions must match.");
  }
  if (q_num_heads <= 0 || kv_num_heads <= 0) {
    throw std::invalid_argument(kernel_name + ": q_num_heads and kv_num_heads must be positive.");
  }

  const std::size_t batch = static_cast<std::size_t>(query.shape[0]);
  const std::size_t sequence = static_cast<std::size_t>(query.shape[1]);
  const std::size_t query_hidden = static_cast<std::size_t>(query.shape[2]);
  const std::size_t key_hidden = static_cast<std::size_t>(key.shape[2]);
  const std::size_t value_hidden = static_cast<std::size_t>(value.shape[2]);
  const std::size_t query_heads = static_cast<std::size_t>(q_num_heads);
  const std::size_t key_value_heads = static_cast<std::size_t>(kv_num_heads);
  if (query_hidden % query_heads != 0 || query_hidden == 0) {
    throw std::invalid_argument(kernel_name +
                                ": query hidden size must be divisible by q_num_heads.");
  }
  const std::size_t key_head_size = query_hidden / query_heads;
  if (key_hidden == 0 || key_head_size == 0 || key_hidden % key_head_size != 0) {
    throw std::invalid_argument(
        kernel_name + ": key must contain a positive whole number of heads matching the query "
                      "head size.");
  }
  const std::size_t key_heads_actual = key_hidden / key_head_size;
  if (key_heads_actual != key_value_heads) {
    if (!policy.allow_key_head_sharing || key_heads_actual == 0 ||
        key_value_heads % key_heads_actual != 0) {
      throw std::invalid_argument(
          kernel_name + ": key must contain kv_num_heads heads matching the query head size.");
    }
  }
  if (value_hidden == 0 || value_hidden % key_value_heads != 0) {
    throw std::invalid_argument(kernel_name +
                                ": value hidden size must be divisible by kv_num_heads.");
  }
  if (query_heads >= key_value_heads) {
    if (query_heads % key_value_heads != 0) {
      throw std::invalid_argument(kernel_name + ": q_num_heads must be divisible by kv_num_heads.");
    }
  } else {
    if (!policy.allow_inverse_grouping) {
      throw std::invalid_argument(kernel_name + ": q_num_heads must be divisible by kv_num_heads.");
    }
    if (key_value_heads % query_heads != 0) {
      throw std::invalid_argument(
          kernel_name + ": kv_num_heads must be divisible by q_num_heads when kv_num_heads is "
                        "larger (inverse grouping).");
    }
  }

  const std::size_t value_head_size = value_hidden / key_value_heads;
  const LinearAttentionRule rule = ParseLinearAttentionRule(kernel_name, update_rule);
  const bool uses_decay = LinearAttentionRuleUsesDecay(rule);
  const bool uses_beta = LinearAttentionRuleUsesBeta(rule);
  if (decay == nullptr && uses_decay) {
    throw std::invalid_argument(kernel_name + ": decay is required by update_rule.");
  }
  if (decay != nullptr && !uses_decay && !policy.allow_irrelevant_optional_inputs) {
    throw std::invalid_argument(kernel_name + ": decay presence does not match update_rule.");
  }
  if (beta == nullptr && uses_beta) {
    throw std::invalid_argument(kernel_name + ": beta is required by update_rule.");
  }
  if (beta != nullptr && !uses_beta && !policy.allow_irrelevant_optional_inputs) {
    throw std::invalid_argument(kernel_name + ": beta presence does not match update_rule.");
  }
  LinearAttentionDecayLayout decay_layout = LinearAttentionDecayLayout::kNone;
  if (decay != nullptr) {
    if (decay->data_type != query.data_type || decay->shape.size() != 3 ||
        decay->shape[0] != query.shape[0] || decay->shape[1] != query.shape[1]) {
      throw std::invalid_argument(
          kernel_name + ": decay must match the activation type and have shape (B,T,...).");
    }
    const std::size_t decay_hidden = static_cast<std::size_t>(decay->shape[2]);
    if (decay_hidden == key_value_heads) {
      decay_layout = LinearAttentionDecayLayout::kPerHead;
    } else if (decay_hidden == LinearAttentionCheckedMultiply(kernel_name, key_value_heads,
                                                              key_head_size, "decay")) {
      decay_layout = LinearAttentionDecayLayout::kPerKeyDimension;
    } else {
      throw std::invalid_argument(kernel_name + ": decay last dimension must be H_kv or H_kv*d_k.");
    }
    if (!uses_decay) {
      decay_layout = LinearAttentionDecayLayout::kNone;
    }
  }

  LinearAttentionBetaLayout beta_layout = LinearAttentionBetaLayout::kNone;
  if (beta != nullptr) {
    if (beta->data_type != query.data_type || beta->shape.size() != 3 ||
        beta->shape[0] != query.shape[0] || beta->shape[1] != query.shape[1]) {
      throw std::invalid_argument(
          kernel_name + ": beta must match the activation type and have shape (B,T,...).");
    }
    const std::size_t beta_hidden = static_cast<std::size_t>(beta->shape[2]);
    if (beta_hidden == 1) {
      beta_layout = LinearAttentionBetaLayout::kShared;
    } else if (beta_hidden == key_value_heads) {
      beta_layout = LinearAttentionBetaLayout::kPerHead;
    } else {
      throw std::invalid_argument(kernel_name + ": beta last dimension must be 1 or H_kv.");
    }
    if (!uses_beta) {
      beta_layout = LinearAttentionBetaLayout::kNone;
    }
  }

  const DataType activation_type = static_cast<DataType>(query.data_type);
  const DataType state_type =
      past_state == nullptr ? activation_type : static_cast<DataType>(past_state->data_type);
  if (!Contains(allowed_types, state_type)) {
    throw std::invalid_argument(kernel_name + ": past_state must be " +
                                TypeListForMessage(allowed_types) + ".");
  }
  const Shape state_shape{
      static_cast<std::int64_t>(batch), static_cast<std::int64_t>(key_value_heads),
      static_cast<std::int64_t>(key_head_size), static_cast<std::int64_t>(value_head_size)};
  const std::size_t state_count = LinearAttentionCheckedMultiply(
      kernel_name, LinearAttentionCheckedMultiply(kernel_name, batch, key_value_heads, "state"),
      LinearAttentionCheckedMultiply(kernel_name, key_head_size, value_head_size, "state"),
      "state");
  if (past_state != nullptr && past_state->shape != state_shape) {
    throw std::invalid_argument(kernel_name + ": past_state shape must be (B,H_kv,d_k,d_v).");
  }

  const std::size_t output_heads = std::max(query_heads, key_value_heads);
  const std::size_t output_hidden =
      LinearAttentionCheckedMultiply(kernel_name, output_heads, value_head_size, "output");
  if (output_hidden > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument(kernel_name + ": output hidden size exceeds int64_t.");
  }
  const Shape output_shape{static_cast<std::int64_t>(batch), static_cast<std::int64_t>(sequence),
                           static_cast<std::int64_t>(output_hidden)};
  const std::size_t output_count = LinearAttentionCheckedMultiply(
      kernel_name, LinearAttentionCheckedMultiply(kernel_name, batch, sequence, "output"),
      output_hidden, "output");

  LinearAttentionPlan plan;
  plan.parameters.batch_size = batch;
  plan.parameters.sequence_length = sequence;
  plan.parameters.query_heads = query_heads;
  plan.parameters.key_value_heads = key_value_heads;
  plan.parameters.key_heads = key_heads_actual;
  plan.parameters.key_head_size = key_head_size;
  plan.parameters.value_head_size = value_head_size;
  plan.parameters.rule = rule;
  plan.parameters.decay_layout = decay_layout;
  plan.parameters.beta_layout = beta_layout;
  plan.parameters.scale =
      scale == 0.0f ? 1.0f / std::sqrt(static_cast<float>(key_head_size)) : scale;
  plan.output_heads = output_heads;
  plan.activation_type = activation_type;
  plan.state_type = state_type;
  plan.state_shape = state_shape;
  plan.output_shape = output_shape;
  plan.state_count = state_count;
  plan.output_count = output_count;
  return plan;
}

LinearAttentionResult ExecuteLinearAttentionPlan(const std::string &kernel_name,
                                                 const LinearAttentionPlan &plan,
                                                 const Tensor &query, const Tensor &key,
                                                 const Tensor &value, const Tensor *past_state,
                                                 const Tensor *decay, const Tensor *beta,
                                                 RuntimeContext *rt, bool has_state_output) {
  Tensor output = AllocateLinearAttentionTensor(kernel_name, rt, 0, true, plan.activation_type,
                                                plan.output_shape, plan.output_count);
  Tensor present_state = AllocateLinearAttentionTensor(
      kernel_name, rt, 1, has_state_output, plan.state_type, plan.state_shape, plan.state_count);

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
    ConvertLinearAttentionToFloat(tensor, count, buffer.data());
    return buffer.data();
  };

  const float *query_data =
      as_float(query, CheckedCount(kernel_name, query.shape, "query"), query_buffer);
  const float *key_data = as_float(key, CheckedCount(kernel_name, key.shape, "key"), key_buffer);
  const float *value_data =
      as_float(value, CheckedCount(kernel_name, value.shape, "value"), value_buffer);
  const float *decay_data =
      plan.parameters.decay_layout == LinearAttentionDecayLayout::kNone
          ? nullptr
          : as_float(*decay, CheckedCount(kernel_name, decay->shape, "decay"), decay_buffer);
  const float *beta_data =
      plan.parameters.beta_layout == LinearAttentionBetaLayout::kNone
          ? nullptr
          : as_float(*beta, CheckedCount(kernel_name, beta->shape, "beta"), beta_buffer);

  float *state_data;
  if (plan.state_type == DataType::FLOAT) {
    state_data = reinterpret_cast<float *>(present_state.mutable_bytes());
    if (past_state == nullptr && plan.state_count != 0) {
      std::fill(state_data, state_data + plan.state_count, 0.0f);
    } else if (past_state != nullptr && plan.state_count != 0 &&
               past_state->bytes() != present_state.bytes()) {
      std::memcpy(state_data, past_state->bytes(), plan.state_count * sizeof(float));
    }
  } else {
    state_buffer.resize(plan.state_count);
    state_data = state_buffer.data();
    if (past_state == nullptr && plan.state_count != 0) {
      std::fill(state_data, state_data + plan.state_count, 0.0f);
    } else if (past_state != nullptr) {
      ConvertLinearAttentionToFloat(*past_state, plan.state_count, state_data);
    }
  }

  float *output_data;
  if (plan.activation_type == DataType::FLOAT) {
    output_data = reinterpret_cast<float *>(output.mutable_bytes());
  } else {
    output_buffer.resize(plan.output_count);
    output_data = output_buffer.data();
  }

  LinearAttentionFloat32(plan.parameters, query_data, key_data, value_data, decay_data, beta_data,
                         state_data, output_data);
  if (plan.activation_type != DataType::FLOAT) {
    ConvertLinearAttentionFromFloat(output_data, plan.output_count, plan.activation_type, output);
  }
  if (plan.state_type != DataType::FLOAT) {
    ConvertLinearAttentionFromFloat(state_data, plan.state_count, plan.state_type, present_state);
  }
  return {std::move(output), std::move(present_state)};
}

LinearAttentionResult InvokeLinearAttentionKernel(const LinearAttentionKernelConfig &config,
                                                  const Tensor &query, const Tensor &key,
                                                  const Tensor &value,
                                                  const LinearAttentionKernelAttributes &attributes,
                                                  const Tensor *past_state, const Tensor *decay,
                                                  const Tensor *beta, RuntimeContext *rt,
                                                  bool has_state_output) {
  if (config.maximum_head_count > 0 && (attributes.query_heads > config.maximum_head_count ||
                                        attributes.key_value_heads > config.maximum_head_count)) {
    throw std::invalid_argument(std::string(config.kernel_name) +
                                ": head counts exceed the supported maximum.");
  }
  if (config.reject_nonzero_state_window && attributes.state_window != 0) {
    throw std::invalid_argument(std::string(config.kernel_name) +
                                ": nonzero state_window is not supported.");
  }
  const LinearAttentionPlan plan = PlanLinearAttention(
      config.kernel_name, query, key, value, past_state, decay, beta, attributes.update_rule,
      attributes.query_heads, attributes.key_value_heads, attributes.scale, config.grouping,
      config.supported_types);
  return ExecuteLinearAttentionPlan(config.kernel_name, plan, query, key, value, past_state, decay,
                                    beta, rt, has_state_output);
}

void RunLinearAttentionKernelNode(const LinearAttentionKernelConfig &config,
                                  const ONNX_LIGHT_NAMESPACE::NodeProto &node, RuntimeContext &rt) {
  RecordKernelUsage(config.kernel_name);
  const bool valid_outputs =
      config.require_present_state
          ? node.output_size() == 2 && !node.output(0).empty() && !node.output(1).empty()
          : node.output_size() >= 1 && node.output_size() <= 2 && !node.output(0).empty();
  if (node.input_size() < 3 || node.input_size() > 6 || !valid_outputs) {
    throw std::invalid_argument(std::string(config.kernel_name) +
                                ": invalid LinearAttention input or output count.");
  }
  const Tensor &query = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &key = rt_ns::GetInput(node, 1, rt.tensors());
  const Tensor &value = rt_ns::GetInput(node, 2, rt.tensors());
  const Tensor *past_state = rt_ns::GetOptionalInput(node, 3, rt.tensors());
  const Tensor *decay = rt_ns::GetOptionalInput(node, 4, rt.tensors());
  const Tensor *beta = rt_ns::GetOptionalInput(node, 5, rt.tensors());
  LinearAttentionKernelAttributes attributes;
  attributes.update_rule = rt_ns::GetAttributeStringOrDefault(node, "update_rule", "gated_delta");
  attributes.query_heads = rt_ns::GetAttributeIntOrDefault(node, "q_num_heads", 0);
  attributes.key_value_heads = rt_ns::GetAttributeIntOrDefault(node, "kv_num_heads", 0);
  attributes.scale = rt_ns::GetAttributeFloatOrDefault(node, "scale", 0.0f);
  attributes.chunk_size = rt_ns::GetAttributeIntOrDefault(node, "chunk_size", 64);
  attributes.state_window = rt_ns::GetAttributeIntOrDefault(node, "state_window", 0);
  const bool has_state_output = node.output_size() == 2 && !node.output(1).empty();
  LinearAttentionResult result = InvokeLinearAttentionKernel(
      config, query, key, value, attributes, past_state, decay, beta, &rt, has_state_output);
  rt_ns::SetOutput(node, 0, std::move(result.output), rt);
  if (has_state_output) {
    rt_ns::SetOutput(node, 1, std::move(result.present_state), rt);
  }
}

} // namespace onnx_light_cpu
