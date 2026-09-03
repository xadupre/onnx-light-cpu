// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/attention/linear_attention.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

/// Selects which LinearAttention head-count relationships
/// :cpp:func:`PlanLinearAttention` accepts beyond the always-required "the
/// larger head count is an exact multiple of the smaller one" rule for
/// query/state grouping.
///
/// The ai.onnx opset-27 contract uses the default (all false): only
/// ``query_heads >= key_value_heads`` grouping and a key tensor carrying
/// exactly ``key_value_heads`` heads. The com.microsoft contract sets both to
/// true, matching ONNX Runtime's CPU kernel.
struct LinearAttentionGroupingPolicy {
  /// Accepts ``key_value_heads > query_heads`` (several state heads share one
  /// query head), in addition to the always-accepted
  /// ``query_heads >= key_value_heads`` direction.
  bool allow_inverse_grouping = false;
  /// Accepts a key tensor carrying fewer heads than ``key_value_heads``,
  /// shared across groups of state heads
  /// (``key_value_heads % key_heads == 0``).
  bool allow_key_head_sharing = false;
  /// Accepts well-shaped decay/beta inputs that the selected update rule does
  /// not consume. ONNX Runtime's contrib operator validates and ignores them.
  bool allow_irrelevant_optional_inputs = false;
};

/// Fully validated shape/attribute plan for one LinearAttention invocation.
///
/// Both the ai.onnx and com.microsoft kernels call
/// :cpp:func:`PlanLinearAttention` to obtain this instead of duplicating its
/// ~150 lines of tensor-shape and attribute validation.
struct LinearAttentionPlan {
  LinearAttentionParameters parameters;
  /// ``max(query_heads, key_value_heads)``: the number of heads in the output
  /// tensor's last dimension.
  std::size_t output_heads = 0;
  ONNX_LIGHT_NAMESPACE::core::runtime::DataType activation_type =
      static_cast<ONNX_LIGHT_NAMESPACE::core::runtime::DataType>(0);
  /// Element type of ``past_state``/``present_state`` (defaults to
  /// ``activation_type`` when there is no ``past_state`` input).
  ONNX_LIGHT_NAMESPACE::core::runtime::DataType state_type =
      static_cast<ONNX_LIGHT_NAMESPACE::core::runtime::DataType>(0);
  ONNX_LIGHT_NAMESPACE::core::runtime::Shape state_shape;
  ONNX_LIGHT_NAMESPACE::core::runtime::Shape output_shape;
  std::size_t state_count = 0;
  std::size_t output_count = 0;
};

struct LinearAttentionResult {
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor output;
  ONNX_LIGHT_NAMESPACE::core::runtime::Tensor present_state;
};

/// Parses `rule` into a :cpp:enum:`LinearAttentionRule`, throwing
/// ``std::invalid_argument`` (prefixed with `kernel_name`) for anything else.
LinearAttentionRule ParseLinearAttentionRule(const std::string &kernel_name,
                                             const std::string &rule);
bool LinearAttentionRuleUsesDecay(LinearAttentionRule rule);
bool LinearAttentionRuleUsesBeta(LinearAttentionRule rule);

/// Validates every LinearAttention input/attribute relationship (packed-3D
/// query/key/value ranks and types, batch/sequence agreement, head-count
/// divisibility per `policy`, decay/beta shape and layout, and `past_state`
/// shape) and returns the resulting compute plan. Throws
/// ``std::invalid_argument`` (prefixed with `kernel_name`) on any violation.
/// `allowed_types` bounds the accepted query/key/value/state element types.
LinearAttentionPlan PlanLinearAttention(
    const std::string &kernel_name, const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &query,
    const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &key,
    const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &value,
    const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *past_state,
    const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *decay,
    const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *beta, const std::string &update_rule,
    std::int64_t q_num_heads, std::int64_t kv_num_heads, float scale,
    const LinearAttentionGroupingPolicy &policy,
    const std::vector<ONNX_LIGHT_NAMESPACE::core::runtime::DataType> &allowed_types);

/// Returns `left * right`, throwing ``std::invalid_argument`` (prefixed with
/// `kernel_name`) if the product overflows ``size_t``.
std::size_t LinearAttentionCheckedMultiply(const std::string &kernel_name, std::size_t left,
                                           std::size_t right, const char *label);

/// Allocates a tensor of `shape`/`count` elements of `type`, using `rt`'s
/// allocator (an output slot when `is_output`, otherwise scratch) when
/// non-null, or a standalone buffer otherwise.
ONNX_LIGHT_NAMESPACE::core::runtime::Tensor AllocateLinearAttentionTensor(
    const std::string &kernel_name, ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt,
    int output_index, bool is_output, ONNX_LIGHT_NAMESPACE::core::runtime::DataType type,
    const ONNX_LIGHT_NAMESPACE::core::runtime::Shape &shape, std::size_t count);

/// Converts `count` elements of `source` (FLOAT, FLOAT16, or BFLOAT16) into
/// `destination` as float32.
void ConvertLinearAttentionToFloat(const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &source,
                                   std::size_t count, float *destination);

/// Converts `count` float32 `source` elements into `destination`, encoding
/// them as `type` (FLOAT, FLOAT16, or BFLOAT16).
void ConvertLinearAttentionFromFloat(const float *source, std::size_t count,
                                     ONNX_LIGHT_NAMESPACE::core::runtime::DataType type,
                                     ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &destination);

LinearAttentionResult
ExecuteLinearAttentionPlan(const std::string &kernel_name, const LinearAttentionPlan &plan,
                           const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &query,
                           const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &key,
                           const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor &value,
                           const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *past_state,
                           const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *decay,
                           const ONNX_LIGHT_NAMESPACE::core::runtime::Tensor *beta,
                           ONNX_LIGHT_NAMESPACE::core::runtime::RuntimeContext *rt,
                           bool has_state_output);

} // namespace onnx_light_cpu
