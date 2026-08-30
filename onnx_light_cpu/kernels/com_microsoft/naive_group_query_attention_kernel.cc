// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/naive_group_query_attention_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// Independent, scalar reference implementation of
// com.microsoft::GroupQueryAttention. Every numeric step below (RoPE, the
// causal/attention_bias score, softmax, and the value reduction) is written
// as a plain loop over batch/head/sequence/kv indices; nothing here calls
// AttentionKernel or any other optimized attention compute helper. Basic
// tensor construction/attribute-reading helpers from onnx_core (Tensor,
// MakeOutputTensor, GetAttribute*, GetInput/SetOutput) are reused since they
// carry no attention-specific math. Parsing/validation intentionally
// duplicates GroupQueryAttentionKernel's so this kernel has no compile-time
// or runtime dependency on it.
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

bool HasInput(const NodeProto &node, int index) {
  return node.input_size() > index && !node.input(index).empty();
}

bool HasOutput(const NodeProto &node, int index) {
  return node.output_size() > index && !node.output(index).empty();
}

std::size_t ElementByteWidth(DataType dtype) {
  return dtype == DataType::FLOAT ? sizeof(float) : sizeof(std::uint16_t);
}

float ReadElementAsFloat(DataType dtype, const std::uint8_t *base, std::size_t index) {
  switch (dtype) {
  case DataType::FLOAT:
    return reinterpret_cast<const float *>(base)[index];
  case DataType::FLOAT16:
    return detail::Float16BitsToFloat(reinterpret_cast<const std::uint16_t *>(base)[index]);
  case DataType::BFLOAT16:
    return detail::Bfloat16BitsToFloat(reinterpret_cast<const std::uint16_t *>(base)[index]);
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: unsupported floating-point element type.");
  }
}

void WriteElementFromFloat(DataType dtype, std::uint8_t *base, std::size_t index, float value) {
  switch (dtype) {
  case DataType::FLOAT:
    reinterpret_cast<float *>(base)[index] = value;
    return;
  case DataType::FLOAT16:
    reinterpret_cast<std::uint16_t *>(base)[index] = detail::FloatToFloat16Bits(value);
    return;
  case DataType::BFLOAT16:
    reinterpret_cast<std::uint16_t *>(base)[index] = detail::FloatToBFloat16Bits(value);
    return;
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: unsupported floating-point element type.");
  }
}

// Resolved, validated view of one GroupQueryAttention invocation: every
// tensor pointer honored by the computation below plus the attributes and
// derived shape scalars the naive math needs.
struct GqaArgs {
  const Tensor *query = nullptr;
  const Tensor *key = nullptr;
  const Tensor *value = nullptr;
  const Tensor *past_key = nullptr;
  const Tensor *past_value = nullptr;
  const Tensor *seqlens_k = nullptr;
  const Tensor *total_sequence_length = nullptr;
  const Tensor *cos_cache = nullptr;
  const Tensor *sin_cache = nullptr;
  const Tensor *position_ids = nullptr;
  const Tensor *attention_bias = nullptr;

  std::int64_t num_heads = 0;
  std::int64_t kv_num_heads = 0;
  bool causal = true;
  std::optional<float> scale;
  float softcap = 0.0f;
  bool do_rotary = false;

  bool has_present_key = false;
  bool has_present_value = false;

  DataType dtype = DataType::FLOAT;
  std::int64_t batch = 0;
  std::int64_t sequence_length = 0;
  std::int64_t head_dim = 0;
  std::int64_t v_head_dim = 0;
  std::int64_t past_length = 0;
  std::int64_t total_length = 0;
};

void ValidateUnsupportedAttributesAndInputs(const NodeProto &node) {
  if (rt_ns::GetAttributeIntOrDefault(node, "sliding_window_cache", 0) != 0 ||
      rt_ns::GetAttributeIntOrDefault(node, "smooth_softmax", 0) != 0 ||
      rt_ns::GetAttributeIntOrDefault(node, "qk_output", 0) != 0 ||
      rt_ns::GetAttributeIntOrDefault(node, "kv_cache_bit_width", 0) != 0 ||
      rt_ns::GetAttributeIntOrDefault(node, "local_window_size", -1) != -1 ||
      rt_ns::GetAttributeStringOrDefault(node, "k_quant_type", "NONE") != "NONE" ||
      rt_ns::GetAttributeStringOrDefault(node, "v_quant_type", "NONE") != "NONE") {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: unsupported attribute (sliding-window cache, "
        "smooth softmax, qk_output, quantized KV cache, and local/sliding window attention are "
        "not implemented).");
  }
  if (HasInput(node, 11)) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: head_sink is not supported.");
  }
  if (HasInput(node, 12) || HasInput(node, 13)) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: quantized KV cache scales (k_scale/v_scale) "
        "are not supported.");
  }
  if (HasInput(node, 14) || HasInput(node, 15)) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: q_norm_weight/k_norm_weight (per-head RMS "
        "norm fusion) are not supported.");
  }
  if (HasOutput(node, 3)) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: output_qk is not supported.");
  }
}

// Reads attributes, resolves which optional tensors participate from
// `node`'s declared inputs/outputs, and fully validates shapes/types/bounds.
// `lookup` resolves a wired (non-empty name) input at `index` to its concrete
// tensor; the caller (`Run` vs. the direct `operator()`) supplies it so this
// function does not depend on `RuntimeContext`.
template <typename Lookup> GqaArgs ResolveAndValidate(const NodeProto &node, Lookup &&lookup) {
  if (node.input_size() < 7) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: query, key, value, seqlens_k and "
        "total_sequence_length inputs are required.");
  }
  if (node.output_size() < 1) {
    throw std::invalid_argument("onnx_light_cpu::NaiveGroupQueryAttention: an output is required.");
  }
  ValidateUnsupportedAttributesAndInputs(node);

  GqaArgs args;
  args.num_heads = rt_ns::GetAttributeIntOrDefault(node, "num_heads", 0);
  args.kv_num_heads = rt_ns::GetAttributeIntOrDefault(node, "kv_num_heads", 0);
  if (args.num_heads <= 0 || args.kv_num_heads <= 0 || args.num_heads % args.kv_num_heads != 0) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: num_heads and kv_num_heads must be positive, "
        "and num_heads must be a multiple of kv_num_heads.");
  }
  args.causal = rt_ns::GetAttributeIntOrDefault(node, "causal", 1) != 0;
  if (const auto *scale = rt_ns::FindAttribute(node, "scale"); scale != nullptr) {
    args.scale = scale->f();
  }
  args.softcap = rt_ns::GetAttributeFloatOrDefault(node, "softcap", 0.0f);
  args.do_rotary = rt_ns::GetAttributeIntOrDefault(node, "do_rotary", 0) != 0;
  const bool rotary_interleaved =
      rt_ns::GetAttributeIntOrDefault(node, "rotary_interleaved", 0) != 0;
  if (args.do_rotary && rotary_interleaved) {
    throw std::invalid_argument("onnx_light_cpu::NaiveGroupQueryAttention: only split-half rotary "
                                "(rotary_interleaved=0) is supported.");
  }

  if (!HasInput(node, 1) || !HasInput(node, 2)) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: packed QKV (empty key/value inputs) is not "
        "supported; key and value must be wired.");
  }
  const bool has_past_key = HasInput(node, 3);
  const bool has_past_value = HasInput(node, 4);
  if (has_past_key != has_past_value) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: past_key and past_value must be used "
        "together.");
  }
  const bool has_cos = HasInput(node, 7);
  const bool has_sin = HasInput(node, 8);
  if (has_cos != has_sin) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: cos_cache and sin_cache must be used "
        "together.");
  }
  if (args.do_rotary && !has_cos) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: do_rotary requires cos_cache and sin_cache.");
  }
  if (!args.do_rotary && has_cos) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: cos_cache/sin_cache require do_rotary=1.");
  }
  const bool has_position_ids = HasInput(node, 9);
  if (has_position_ids && !args.do_rotary) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: position_ids is only supported with "
        "do_rotary=1.");
  }
  const bool has_attention_bias = HasInput(node, 10);

  args.has_present_key = HasOutput(node, 1);
  args.has_present_value = HasOutput(node, 2);
  if (args.has_present_key != args.has_present_value) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: present_key and present_value must be "
        "requested together.");
  }

  args.query = &lookup(0);
  args.key = &lookup(1);
  args.value = &lookup(2);
  args.past_key = has_past_key ? &lookup(3) : nullptr;
  args.past_value = has_past_value ? &lookup(4) : nullptr;
  args.seqlens_k = &lookup(5);
  args.total_sequence_length = &lookup(6);
  args.cos_cache = has_cos ? &lookup(7) : nullptr;
  args.sin_cache = has_sin ? &lookup(8) : nullptr;
  args.position_ids = has_position_ids ? &lookup(9) : nullptr;
  args.attention_bias = has_attention_bias ? &lookup(10) : nullptr;

  args.dtype = static_cast<DataType>(args.query->data_type);
  if (args.dtype != DataType::FLOAT && args.dtype != DataType::FLOAT16 &&
      args.dtype != DataType::BFLOAT16) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: query/key/value must be FLOAT, FLOAT16, or "
        "BFLOAT16.");
  }
  if (static_cast<DataType>(args.key->data_type) != args.dtype ||
      static_cast<DataType>(args.value->data_type) != args.dtype) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: query, key, and value types must match.");
  }
  if (args.past_key != nullptr &&
      (static_cast<DataType>(args.past_key->data_type) != args.dtype ||
       static_cast<DataType>(args.past_value->data_type) != args.dtype)) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: past_key/past_value type must match query.");
  }
  if (args.cos_cache != nullptr &&
      (static_cast<DataType>(args.cos_cache->data_type) != args.dtype ||
       static_cast<DataType>(args.sin_cache->data_type) != args.dtype)) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: cos_cache/sin_cache type must match query.");
  }
  if (args.attention_bias != nullptr &&
      static_cast<DataType>(args.attention_bias->data_type) != args.dtype) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: attention_bias type must match query.");
  }

  if (args.query->shape.size() != 3 || args.key->shape.size() != 3 ||
      args.value->shape.size() != 3) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: query, key, and value must have rank 3.");
  }
  args.batch = args.query->shape[0];
  args.sequence_length = args.query->shape[1];
  if (args.key->shape[0] != args.batch || args.value->shape[0] != args.batch ||
      args.key->shape[1] != args.sequence_length || args.value->shape[1] != args.sequence_length) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: query, key, and value must share the same "
        "batch and sequence length.");
  }
  if (args.batch <= 0 || args.sequence_length <= 0) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: batch and sequence length must be positive.");
  }
  if (args.query->shape[2] % args.num_heads != 0) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: query hidden size must be a multiple of "
        "num_heads.");
  }
  args.head_dim = args.query->shape[2] / args.num_heads;
  if (args.key->shape[2] % args.kv_num_heads != 0 ||
      args.key->shape[2] / args.kv_num_heads != args.head_dim) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: key hidden size must equal kv_num_heads * "
        "head_size (matching query's head_size).");
  }
  if (args.value->shape[2] % args.kv_num_heads != 0) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: value hidden size must be a multiple of "
        "kv_num_heads.");
  }
  args.v_head_dim = args.value->shape[2] / args.kv_num_heads;

  if (args.past_key != nullptr) {
    if (args.past_key->shape.size() != 4 || args.past_value->shape.size() != 4) {
      throw std::invalid_argument(
          "onnx_light_cpu::NaiveGroupQueryAttention: past_key/past_value must have rank 4.");
    }
    if (args.past_key->shape[0] != args.batch || args.past_value->shape[0] != args.batch ||
        args.past_key->shape[1] != args.kv_num_heads ||
        args.past_value->shape[1] != args.kv_num_heads) {
      throw std::invalid_argument(
          "onnx_light_cpu::NaiveGroupQueryAttention: past_key/past_value must share the batch "
          "size and kv_num_heads.");
    }
    if (args.past_key->shape[2] != args.past_value->shape[2]) {
      throw std::invalid_argument(
          "onnx_light_cpu::NaiveGroupQueryAttention: past_key and past_value must share the "
          "same past sequence length.");
    }
    if (args.past_key->shape[3] != args.head_dim) {
      throw std::invalid_argument(
          "onnx_light_cpu::NaiveGroupQueryAttention: past_key must share Q/K's head_size.");
    }
    if (args.past_value->shape[3] != args.v_head_dim) {
      throw std::invalid_argument(
          "onnx_light_cpu::NaiveGroupQueryAttention: past_value must share V's head_size.");
    }
    args.past_length = args.past_key->shape[2];
  }
  args.total_length = args.past_length + args.sequence_length;

  if (static_cast<DataType>(args.seqlens_k->data_type) != DataType::INT32 ||
      args.seqlens_k->shape.size() != 1 || args.seqlens_k->shape[0] != args.batch) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: seqlens_k must be an INT32 tensor with one "
        "value per batch.");
  }
  if (static_cast<DataType>(args.total_sequence_length->data_type) != DataType::INT32 ||
      args.total_sequence_length->element_count() != 1) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: total_sequence_length must be an INT32 "
        "scalar.");
  }
  if (args.total_sequence_length->AsInt32()[0] != args.total_length) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: total_sequence_length must equal "
        "past_sequence_length + sequence_length.");
  }
  for (std::int64_t b = 0; b < args.batch; ++b) {
    if (args.seqlens_k->AsInt32()[b] != args.total_length - 1) {
      throw std::invalid_argument(
          "onnx_light_cpu::NaiveGroupQueryAttention: every seqlens_k value must equal "
          "total_sequence_length - 1 (a uniform batch is required).");
    }
  }

  if (args.attention_bias != nullptr && args.attention_bias->shape.size() > 4) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: attention_bias must have at most 4 "
        "dimensions.");
  }

  if (args.do_rotary) {
    if (args.cos_cache->shape.size() != 2 || args.sin_cache->shape.size() != 2) {
      throw std::invalid_argument(
          "onnx_light_cpu::NaiveGroupQueryAttention: cos_cache/sin_cache must have rank 2.");
    }
    if (args.head_dim % 2 != 0) {
      throw std::invalid_argument(
          "onnx_light_cpu::NaiveGroupQueryAttention: rotary embeddings require an even "
          "head_size.");
    }
    const std::int64_t half = args.head_dim / 2;
    if (args.cos_cache->shape[1] != half || args.sin_cache->shape[1] != half ||
        args.cos_cache->shape[0] != args.sin_cache->shape[0]) {
      throw std::invalid_argument(
          "onnx_light_cpu::NaiveGroupQueryAttention: cos_cache/sin_cache must both have shape "
          "(max_sequence_length, head_size / 2).");
    }
    if (args.position_ids != nullptr) {
      if (static_cast<DataType>(args.position_ids->data_type) != DataType::INT64 ||
          args.position_ids->shape.size() != 2 || args.position_ids->shape[0] != args.batch ||
          args.position_ids->shape[1] != args.sequence_length) {
        throw std::invalid_argument(
            "onnx_light_cpu::NaiveGroupQueryAttention: position_ids must be an INT64 tensor "
            "with shape (batch_size, sequence_length).");
      }
    }
    const std::int64_t max_position = args.cos_cache->shape[0];
    const std::int64_t *position_ids_data =
        args.position_ids != nullptr ? args.position_ids->AsInt64() : nullptr;
    const std::int32_t *seqlens_k_data = args.seqlens_k->AsInt32();
    for (std::int64_t b = 0; b < args.batch; ++b) {
      if (position_ids_data != nullptr) {
        for (std::int64_t s = 0; s < args.sequence_length; ++s) {
          const std::int64_t position = position_ids_data[b * args.sequence_length + s];
          if (position < 0 || position >= max_position) {
            throw std::invalid_argument(
                "onnx_light_cpu::NaiveGroupQueryAttention: a position_ids value falls outside "
                "cos_cache/sin_cache.");
          }
        }
        continue;
      }
      const std::int64_t start =
          static_cast<std::int64_t>(seqlens_k_data[b]) + 1 - args.sequence_length;
      if (start < 0 || start + args.sequence_length > max_position) {
        throw std::invalid_argument(
            "onnx_light_cpu::NaiveGroupQueryAttention: the RoPE position derived for a batch "
            "falls outside cos_cache/sin_cache.");
      }
    }
  }

  return args;
}

std::vector<std::int64_t> ResolveRotaryPositions(const GqaArgs &args) {
  std::vector<std::int64_t> positions(static_cast<std::size_t>(args.batch * args.sequence_length));
  const std::int64_t *position_ids_data =
      args.position_ids != nullptr ? args.position_ids->AsInt64() : nullptr;
  const std::int32_t *seqlens_k_data = args.seqlens_k->AsInt32();
  for (std::int64_t b = 0; b < args.batch; ++b) {
    const std::int64_t start =
        static_cast<std::int64_t>(seqlens_k_data[b]) + 1 - args.sequence_length;
    for (std::int64_t s = 0; s < args.sequence_length; ++s) {
      positions[static_cast<std::size_t>(b * args.sequence_length + s)] =
          position_ids_data != nullptr ? position_ids_data[b * args.sequence_length + s]
                                       : start + s;
    }
  }
  return positions;
}

// Applies split-half RoPE (`rotary_interleaved=0`) to every head of `input`
// (rank-3, `(batch, sequence, heads * head_dim)`) at the absolute position
// supplied in `positions`, reading `cos_cache`/`sin_cache` rows
// `(position, 0:head_dim/2)`. Returns a freshly materialized tensor of the
// same shape/type; `input` is left unmodified.
Tensor ApplyRotaryHalf(RuntimeContext *rt, const Tensor &input, std::int64_t heads,
                       std::int64_t head_dim, const Tensor &cos_cache, const Tensor &sin_cache,
                       const std::vector<std::int64_t> &positions) {
  const DataType dtype = static_cast<DataType>(input.data_type);
  const DataType cache_dtype = static_cast<DataType>(cos_cache.data_type);
  const std::int64_t batch = input.shape[0];
  const std::int64_t sequence_length = input.shape[1];
  const std::int64_t hidden = input.shape[2];
  const std::int64_t half = head_dim / 2;
  const std::size_t element_bytes = ElementByteWidth(dtype);
  const std::size_t total_bytes =
      static_cast<std::size_t>(batch * sequence_length * hidden) * element_bytes;
  Tensor output = rt != nullptr
                      ? rt->MakeTemporaryTensor(input.data_type, input.shape, total_bytes)
                      : rt_ns::MakeOutputTensor(input.data_type, input.shape, total_bytes, nullptr);
  const std::uint8_t *src = input.bytes();
  std::uint8_t *dst = output.mutable_bytes();
  const std::uint8_t *cos_bytes = cos_cache.bytes();
  const std::uint8_t *sin_bytes = sin_cache.bytes();
  for (std::int64_t b = 0; b < batch; ++b) {
    for (std::int64_t s = 0; s < sequence_length; ++s) {
      const std::int64_t position = positions[static_cast<std::size_t>(b * sequence_length + s)];
      const std::size_t row_base = static_cast<std::size_t>((b * sequence_length + s) * hidden);
      for (std::int64_t h = 0; h < heads; ++h) {
        const std::size_t head_base = row_base + static_cast<std::size_t>(h * head_dim);
        for (std::int64_t d = 0; d < half; ++d) {
          const std::size_t cache_index = static_cast<std::size_t>(position * half + d);
          const float cos_value = ReadElementAsFloat(cache_dtype, cos_bytes, cache_index);
          const float sin_value = ReadElementAsFloat(cache_dtype, sin_bytes, cache_index);
          const float x0 = ReadElementAsFloat(dtype, src, head_base + static_cast<std::size_t>(d));
          const float x1 =
              ReadElementAsFloat(dtype, src, head_base + static_cast<std::size_t>(half + d));
          WriteElementFromFloat(dtype, dst, head_base + static_cast<std::size_t>(d),
                                x0 * cos_value - x1 * sin_value);
          WriteElementFromFloat(dtype, dst, head_base + static_cast<std::size_t>(half + d),
                                x1 * cos_value + x0 * sin_value);
        }
      }
    }
  }
  return output;
}

// Materializes a rank-4 `(batch, heads, past_length + sequence_length, dim)`
// tensor holding `past` (when non-null) followed by `current` (rank-3,
// `(batch, sequence, heads * dim)`, already rotated for a key cache). Used
// both to expose `present_key`/`present_value` (when requested, via
// `output_slot >= 0`) and, unconditionally, to give the naive attention loop
// below a single contiguous view of every KV position (`output_slot < 0`
// materializes a temporary instead of a graph output).
Tensor ConcatenatePastAndCurrent(RuntimeContext *rt, int output_slot, DataType dtype,
                                 std::int64_t batch, std::int64_t heads, std::int64_t past_length,
                                 std::int64_t sequence_length, std::int64_t dim, const Tensor *past,
                                 const Tensor &current) {
  const std::int64_t total_length = past_length + sequence_length;
  const Shape shape{batch, heads, total_length, dim};
  const std::size_t element_bytes = ElementByteWidth(dtype);
  const std::size_t total_bytes =
      static_cast<std::size_t>(batch * heads * total_length * dim) * element_bytes;
  Tensor result;
  if (output_slot >= 0 && rt != nullptr) {
    result =
        rt->MakeOutputTensor(output_slot, static_cast<std::int32_t>(dtype), shape, total_bytes);
  } else if (rt != nullptr) {
    result = rt->MakeTemporaryTensor(static_cast<std::int32_t>(dtype), shape, total_bytes);
  } else {
    result = rt_ns::MakeOutputTensor(static_cast<std::int32_t>(dtype), shape, total_bytes, nullptr);
  }
  std::uint8_t *dst = result.mutable_bytes();
  if (past_length > 0) {
    const std::uint8_t *past_bytes = past->bytes();
    const std::size_t row_bytes = static_cast<std::size_t>(past_length * dim) * element_bytes;
    for (std::int64_t b = 0; b < batch; ++b) {
      for (std::int64_t h = 0; h < heads; ++h) {
        const std::size_t src_offset =
            static_cast<std::size_t>((b * heads + h) * past_length * dim) * element_bytes;
        const std::size_t dst_offset =
            static_cast<std::size_t>((b * heads + h) * total_length * dim) * element_bytes;
        std::memcpy(dst + dst_offset, past_bytes + src_offset, row_bytes);
      }
    }
  }
  const std::uint8_t *current_bytes = current.bytes();
  const std::int64_t hidden = heads * dim;
  const std::size_t element_row_bytes = static_cast<std::size_t>(dim) * element_bytes;
  for (std::int64_t b = 0; b < batch; ++b) {
    for (std::int64_t s = 0; s < sequence_length; ++s) {
      for (std::int64_t h = 0; h < heads; ++h) {
        const std::size_t src_offset =
            static_cast<std::size_t>((b * sequence_length + s) * hidden + h * dim) * element_bytes;
        const std::size_t dst_offset =
            static_cast<std::size_t>(((b * heads + h) * total_length + past_length + s) * dim) *
            element_bytes;
        std::memcpy(dst + dst_offset, current_bytes + src_offset, element_row_bytes);
      }
    }
  }
  return result;
}

// Per-axis element strides (0 for a broadcast axis) that right-justify
// `mask_shape` (up to rank 4) against the target
// `(batch, num_heads, sequence_length, total_kv_length)` shape, following
// ONNX numpy-style broadcasting. Throws when a dimension is neither `1` nor
// the matching target dimension.
struct MaskStrides {
  std::ptrdiff_t batch = 0;
  std::ptrdiff_t head = 0;
  std::ptrdiff_t q = 0;
  std::ptrdiff_t kv = 0;
};

MaskStrides ResolveMaskStrides(const Shape &mask_shape, std::int64_t batch, std::int64_t heads,
                               std::int64_t q_length, std::int64_t kv_length) {
  if (mask_shape.size() == 0 || mask_shape.size() > 4) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveGroupQueryAttention: attention_bias must have between 1 and 4 "
        "dimensions.");
  }
  const std::array<std::int64_t, 4> target = {batch, heads, q_length, kv_length};
  std::array<std::int64_t, 4> aligned = {1, 1, 1, 1};
  const std::size_t offset = 4 - mask_shape.size();
  for (std::size_t i = 0; i < mask_shape.size(); ++i) {
    aligned[offset + i] = mask_shape[i];
  }
  for (std::size_t i = 0; i < 4; ++i) {
    if (aligned[i] != 1 && aligned[i] != target[i]) {
      throw std::invalid_argument(
          "onnx_light_cpu::NaiveGroupQueryAttention: attention_bias is not broadcastable to "
          "(batch_size, num_heads, sequence_length, total_sequence_length).");
    }
  }
  std::array<std::ptrdiff_t, 4> contiguous_stride = {0, 0, 0, 0};
  std::ptrdiff_t running = 1;
  for (std::size_t i = 4; i-- > 0;) {
    contiguous_stride[i] = running;
    running *= static_cast<std::ptrdiff_t>(aligned[i]);
  }
  MaskStrides strides;
  strides.batch = aligned[0] == 1 ? 0 : contiguous_stride[0];
  strides.head = aligned[1] == 1 ? 0 : contiguous_stride[1];
  strides.q = aligned[2] == 1 ? 0 : contiguous_stride[2];
  strides.kv = aligned[3] == 1 ? 0 : contiguous_stride[3];
  return strides;
}

constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();

// Naive scalar attention core: for every (batch, query head, query position)
// row, scores every KV position with `scale * dot(q, k)`, optionally
// softcaps it, adds the bottom-right causal bias (`j <= i + past_length`)
// and the broadcast `attention_bias` additive bias, then applies a
// numerically-stable softmax and accumulates the weighted `value` reduction.
// `full_key`/`full_value` are rank-4 `(batch, kv_num_heads, total_length,
// dim)` tensors already holding `past` followed by the (rotated) current
// step, exactly as produced by `ConcatenatePastAndCurrent`.
Tensor ComputeNaiveAttention(const GqaArgs &args, RuntimeContext *rt, const Tensor &q,
                             const Tensor &full_key, const Tensor &full_value) {
  const DataType dtype = args.dtype;
  const std::int64_t batch = args.batch;
  const std::int64_t num_heads = args.num_heads;
  const std::int64_t kv_num_heads = args.kv_num_heads;
  const std::int64_t group_size = num_heads / kv_num_heads;
  const std::int64_t sequence_length = args.sequence_length;
  const std::int64_t head_dim = args.head_dim;
  const std::int64_t v_head_dim = args.v_head_dim;
  const std::int64_t total_length = args.total_length;
  const std::int64_t past_length = args.past_length;
  const float scale =
      args.scale.has_value()
          ? *args.scale
          : (head_dim > 0 ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)))
                          : 1.0f);
  const bool has_softcap = args.softcap != 0.0f;

  MaskStrides mask_strides;
  const std::uint8_t *mask_bytes = nullptr;
  DataType mask_dtype = dtype;
  if (args.attention_bias != nullptr) {
    mask_strides = ResolveMaskStrides(args.attention_bias->shape, batch, num_heads, sequence_length,
                                      total_length);
    mask_bytes = args.attention_bias->bytes();
    mask_dtype = static_cast<DataType>(args.attention_bias->data_type);
  }

  const std::int64_t q_hidden = num_heads * head_dim;
  const std::int64_t out_hidden = num_heads * v_head_dim;
  const Shape output_shape{batch, sequence_length, out_hidden};
  const std::size_t element_bytes = ElementByteWidth(dtype);
  const std::size_t output_bytes =
      static_cast<std::size_t>(batch * sequence_length * out_hidden) * element_bytes;
  Tensor output = rt != nullptr ? rt->MakeOutputTensor(0, static_cast<std::int32_t>(dtype),
                                                       output_shape, output_bytes)
                                : rt_ns::MakeOutputTensor(static_cast<std::int32_t>(dtype),
                                                          output_shape, output_bytes, nullptr);

  const std::uint8_t *q_bytes = q.bytes();
  const std::uint8_t *k_bytes = full_key.bytes();
  const std::uint8_t *v_bytes = full_value.bytes();
  std::uint8_t *out_bytes = output.mutable_bytes();

  // Reused per row: the (masked, softcapped) score of every KV position,
  // overwritten with its softmax probability once the row's max is known.
  std::vector<float> scores(static_cast<std::size_t>(total_length));

  for (std::int64_t b = 0; b < batch; ++b) {
    for (std::int64_t h = 0; h < num_heads; ++h) {
      const std::int64_t kv_h = h / group_size;
      for (std::int64_t i = 0; i < sequence_length; ++i) {
        const std::size_t q_row_base =
            static_cast<std::size_t>((b * sequence_length + i) * q_hidden + h * head_dim);
        // Bottom-right causal frontier: query row `i` (absolute position
        // `past_length + i`) attends KV position `j` iff `j <= i +
        // past_length`.
        const std::int64_t window_center = i + past_length;

        float row_max = kNegativeInfinity;
        float row_bias_max = kNegativeInfinity;
        for (std::int64_t j = 0; j < total_length; ++j) {
          const bool allowed = !args.causal || j <= window_center;
          float additive_bias = 0.0f;
          if (mask_bytes != nullptr) {
            const std::ptrdiff_t index = b * mask_strides.batch + h * mask_strides.head +
                                         i * mask_strides.q + j * mask_strides.kv;
            additive_bias =
                ReadElementAsFloat(mask_dtype, mask_bytes, static_cast<std::size_t>(index));
          }
          const float bias = (allowed ? 0.0f : kNegativeInfinity) + additive_bias;
          row_bias_max = std::max(row_bias_max, bias);

          const std::size_t k_row_base =
              static_cast<std::size_t>(((b * kv_num_heads + kv_h) * total_length + j) * head_dim);
          float dot = 0.0f;
          for (std::int64_t d = 0; d < head_dim; ++d) {
            dot += ReadElementAsFloat(dtype, q_bytes, q_row_base + static_cast<std::size_t>(d)) *
                   ReadElementAsFloat(dtype, k_bytes, k_row_base + static_cast<std::size_t>(d));
          }
          const float raw = scale * dot;
          const float capped = has_softcap ? args.softcap * std::tanh(raw / args.softcap) : raw;
          const float score = capped + bias;
          scores[static_cast<std::size_t>(j)] = score;
          row_max = std::max(row_max, score);
        }

        const std::size_t out_row_base =
            static_cast<std::size_t>((b * sequence_length + i) * out_hidden + h * v_head_dim);
        const bool row_all_masked = row_bias_max == kNegativeInfinity;
        if (row_all_masked) {
          // Every KV position of this row is disallowed by the combined
          // causal/attention_bias bias: zero output rather than NaN.
          for (std::int64_t d = 0; d < v_head_dim; ++d) {
            WriteElementFromFloat(dtype, out_bytes, out_row_base + static_cast<std::size_t>(d),
                                  0.0f);
          }
          continue;
        }

        double sum = 0.0;
        for (std::int64_t j = 0; j < total_length; ++j) {
          const float score = scores[static_cast<std::size_t>(j)];
          const float probability = score == kNegativeInfinity ? 0.0f : std::exp(score - row_max);
          scores[static_cast<std::size_t>(j)] = probability;
          sum += probability;
        }
        const float inv_sum = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;

        for (std::int64_t d = 0; d < v_head_dim; ++d) {
          double accumulator = 0.0;
          for (std::int64_t j = 0; j < total_length; ++j) {
            const std::size_t v_row_base = static_cast<std::size_t>(
                ((b * kv_num_heads + kv_h) * total_length + j) * v_head_dim);
            accumulator +=
                static_cast<double>(scores[static_cast<std::size_t>(j)]) *
                ReadElementAsFloat(dtype, v_bytes, v_row_base + static_cast<std::size_t>(d));
          }
          WriteElementFromFloat(dtype, out_bytes, out_row_base + static_cast<std::size_t>(d),
                                static_cast<float>(accumulator * inv_sum));
        }
      }
    }
  }

  return output;
}

// Shared computation core used by both `Run` and `operator()`: applies RoPE
// (when configured), materializes the concatenated past+current KV (always,
// exposing it as `present_key`/`present_value` when requested), and runs the
// naive scalar attention loop above.
Tensor Compute(const GqaArgs &args, RuntimeContext *rt, Tensor *present_key,
               Tensor *present_value) {
  const Tensor *q_for_attention = args.query;
  const Tensor *k_for_attention = args.key;
  Tensor rotated_query;
  Tensor rotated_key;
  if (args.do_rotary) {
    const std::vector<std::int64_t> positions = ResolveRotaryPositions(args);
    rotated_query = ApplyRotaryHalf(rt, *args.query, args.num_heads, args.head_dim, *args.cos_cache,
                                    *args.sin_cache, positions);
    rotated_key = ApplyRotaryHalf(rt, *args.key, args.kv_num_heads, args.head_dim, *args.cos_cache,
                                  *args.sin_cache, positions);
    q_for_attention = &rotated_query;
    k_for_attention = &rotated_key;
  }

  Tensor full_key = ConcatenatePastAndCurrent(
      rt, args.has_present_key ? 1 : -1, args.dtype, args.batch, args.kv_num_heads,
      args.past_length, args.sequence_length, args.head_dim, args.past_key, *k_for_attention);
  Tensor full_value = ConcatenatePastAndCurrent(
      rt, args.has_present_value ? 2 : -1, args.dtype, args.batch, args.kv_num_heads,
      args.past_length, args.sequence_length, args.v_head_dim, args.past_value, *args.value);

  Tensor output = ComputeNaiveAttention(args, rt, *q_for_attention, full_key, full_value);

  if (args.has_present_key && present_key != nullptr) {
    *present_key = std::move(full_key);
  }
  if (args.has_present_value && present_value != nullptr) {
    *present_value = std::move(full_value);
  }
  return output;
}

} // namespace

NaiveGroupQueryAttentionKernel::NaiveGroupQueryAttentionKernel(const NodeProto &node,
                                                               const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
}

void NaiveGroupQueryAttentionKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  const auto lookup = [&node, &rt](int index) -> const Tensor & {
    return rt_ns::GetInput(node, index, rt.tensors());
  };
  const GqaArgs args = ResolveAndValidate(node, lookup);

  Tensor present_key;
  Tensor present_value;
  Tensor output = Compute(args, &rt, &present_key, &present_value);
  if (args.has_present_key) {
    rt_ns::SetOutput(node, 1, std::move(present_key), rt);
    rt_ns::SetOutput(node, 2, std::move(present_value), rt);
  }
  rt_ns::SetOutput(node, 0, std::move(output), rt);
}

Tensor NaiveGroupQueryAttentionKernel::operator()(
    const NodeProto &node, const Tensor &query, const Tensor &key, const Tensor &value,
    const Tensor &seqlens_k, const Tensor &total_sequence_length, const Tensor *past_key,
    const Tensor *past_value, const Tensor *cos_cache, const Tensor *sin_cache,
    const Tensor *position_ids, const Tensor *attention_bias, RuntimeContext *rt,
    Tensor *present_key, Tensor *present_value) const {
  std::array<const Tensor *, 11> inputs{};
  inputs[0] = &query;
  inputs[1] = &key;
  inputs[2] = &value;
  inputs[3] = past_key;
  inputs[4] = past_value;
  inputs[5] = &seqlens_k;
  inputs[6] = &total_sequence_length;
  inputs[7] = cos_cache;
  inputs[8] = sin_cache;
  inputs[9] = position_ids;
  inputs[10] = attention_bias;
  const auto lookup = [&inputs](int index) -> const Tensor & {
    if (index >= static_cast<int>(inputs.size()) ||
        inputs[static_cast<std::size_t>(index)] == nullptr) {
      throw std::invalid_argument(
          "onnx_light_cpu::NaiveGroupQueryAttention: node declares an input that has no "
          "matching tensor argument.");
    }
    return *inputs[static_cast<std::size_t>(index)];
  };
  const GqaArgs args = ResolveAndValidate(node, lookup);
  return Compute(args, rt, present_key, present_value);
}

void RegisterNaiveGroupQueryAttentionKernel() {
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<NaiveGroupQueryAttentionKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "GroupQueryAttention";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = NaiveGroupQueryAttentionKernel::kName;
  info.types = {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
