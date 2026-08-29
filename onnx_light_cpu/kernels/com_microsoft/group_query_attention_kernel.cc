// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/group_query_attention_kernel.h"

#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
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
        "onnx_light_cpu::GroupQueryAttention: unsupported floating-point element type.");
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
        "onnx_light_cpu::GroupQueryAttention: unsupported floating-point element type.");
  }
}

// Resolved, validated view of one GroupQueryAttention invocation: every
// tensor pointer honored by the computation below plus the attributes and
// derived shape scalars needed by RoPE, cache concatenation, and the
// delegated Attention call.
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
        "onnx_light_cpu::GroupQueryAttention: unsupported attribute (sliding-window cache, "
        "smooth softmax, qk_output, quantized KV cache, and local/sliding window attention are "
        "not implemented).");
  }
  if (HasInput(node, 11)) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: head_sink is not supported.");
  }
  if (HasInput(node, 12) || HasInput(node, 13)) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: quantized KV cache scales (k_scale/v_scale) are "
        "not supported.");
  }
  if (HasInput(node, 14) || HasInput(node, 15)) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: q_norm_weight/k_norm_weight "
                                "(per-head RMS norm fusion) are not supported.");
  }
  if (HasOutput(node, 3)) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: output_qk is not supported.");
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
        "onnx_light_cpu::GroupQueryAttention: query, key, value, seqlens_k and "
        "total_sequence_length inputs are required.");
  }
  if (node.output_size() < 1) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: an output is required.");
  }
  ValidateUnsupportedAttributesAndInputs(node);

  GqaArgs args;
  args.num_heads = rt_ns::GetAttributeIntOrDefault(node, "num_heads", 0);
  args.kv_num_heads = rt_ns::GetAttributeIntOrDefault(node, "kv_num_heads", 0);
  if (args.num_heads <= 0 || args.kv_num_heads <= 0 || args.num_heads % args.kv_num_heads != 0) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: num_heads and kv_num_heads must be positive, and "
        "num_heads must be a multiple of kv_num_heads.");
  }
  args.causal = rt_ns::GetAttributeIntOrDefault(node, "causal", 1) != 0;
  if (const auto *scale = FindAttribute(node, "scale"); scale != nullptr) {
    args.scale = scale->f();
  }
  args.softcap = rt_ns::GetAttributeFloatOrDefault(node, "softcap", 0.0f);
  args.do_rotary = rt_ns::GetAttributeIntOrDefault(node, "do_rotary", 0) != 0;
  const bool rotary_interleaved =
      rt_ns::GetAttributeIntOrDefault(node, "rotary_interleaved", 0) != 0;
  if (args.do_rotary && rotary_interleaved) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: only split-half rotary "
                                "(rotary_interleaved=0) is supported.");
  }

  if (!HasInput(node, 1) || !HasInput(node, 2)) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: packed QKV (empty key/value "
                                "inputs) is not supported; key and value must be wired.");
  }
  const bool has_past_key = HasInput(node, 3);
  const bool has_past_value = HasInput(node, 4);
  if (has_past_key != has_past_value) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: past_key and past_value must be used together.");
  }
  const bool has_cos = HasInput(node, 7);
  const bool has_sin = HasInput(node, 8);
  if (has_cos != has_sin) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: cos_cache and sin_cache must be used together.");
  }
  if (args.do_rotary && !has_cos) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: do_rotary requires cos_cache and sin_cache.");
  }
  if (!args.do_rotary && has_cos) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: cos_cache/sin_cache require "
                                "do_rotary=1.");
  }
  const bool has_position_ids = HasInput(node, 9);
  if (has_position_ids && !args.do_rotary) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: position_ids is only supported with do_rotary=1.");
  }
  const bool has_attention_bias = HasInput(node, 10);

  args.has_present_key = HasOutput(node, 1);
  args.has_present_value = HasOutput(node, 2);
  if (args.has_present_key != args.has_present_value) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: present_key and "
                                "present_value must be requested together.");
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
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: query/key/value must be "
                                "FLOAT, FLOAT16, or BFLOAT16.");
  }
  if (static_cast<DataType>(args.key->data_type) != args.dtype ||
      static_cast<DataType>(args.value->data_type) != args.dtype) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: query, key, and value types must match.");
  }
  if (args.past_key != nullptr &&
      (static_cast<DataType>(args.past_key->data_type) != args.dtype ||
       static_cast<DataType>(args.past_value->data_type) != args.dtype)) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: past_key/past_value type must match query.");
  }
  if (args.cos_cache != nullptr &&
      (static_cast<DataType>(args.cos_cache->data_type) != args.dtype ||
       static_cast<DataType>(args.sin_cache->data_type) != args.dtype)) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: cos_cache/sin_cache type must match query.");
  }

  if (args.query->shape.size() != 3 || args.key->shape.size() != 3 ||
      args.value->shape.size() != 3) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: query, key, and value must have rank 3.");
  }
  args.batch = args.query->shape[0];
  args.sequence_length = args.query->shape[1];
  if (args.key->shape[0] != args.batch || args.value->shape[0] != args.batch ||
      args.key->shape[1] != args.sequence_length || args.value->shape[1] != args.sequence_length) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: query, key, and value must "
                                "share the same batch and sequence length.");
  }
  if (args.batch <= 0 || args.sequence_length <= 0) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: batch and sequence length must be positive.");
  }
  if (args.query->shape[2] % args.num_heads != 0) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: query hidden size must be a multiple of num_heads.");
  }
  args.head_dim = args.query->shape[2] / args.num_heads;
  if (args.key->shape[2] % args.kv_num_heads != 0 ||
      args.key->shape[2] / args.kv_num_heads != args.head_dim) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: key hidden size must equal "
                                "kv_num_heads * head_size (matching query's head_size).");
  }
  if (args.value->shape[2] % args.kv_num_heads != 0) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: value hidden size must be a multiple of "
        "kv_num_heads.");
  }
  args.v_head_dim = args.value->shape[2] / args.kv_num_heads;

  if (args.past_key != nullptr) {
    if (args.past_key->shape.size() != 4 || args.past_value->shape.size() != 4) {
      throw std::invalid_argument(
          "onnx_light_cpu::GroupQueryAttention: past_key/past_value must have rank 4.");
    }
    if (args.past_key->shape[0] != args.batch || args.past_value->shape[0] != args.batch ||
        args.past_key->shape[1] != args.kv_num_heads ||
        args.past_value->shape[1] != args.kv_num_heads) {
      throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: past_key/past_value must "
                                  "share the batch size and kv_num_heads.");
    }
    if (args.past_key->shape[2] != args.past_value->shape[2]) {
      throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: past_key and past_value "
                                  "must share the same past sequence length.");
    }
    if (args.past_key->shape[3] != args.head_dim) {
      throw std::invalid_argument(
          "onnx_light_cpu::GroupQueryAttention: past_key must share Q/K's head_size.");
    }
    if (args.past_value->shape[3] != args.v_head_dim) {
      throw std::invalid_argument(
          "onnx_light_cpu::GroupQueryAttention: past_value must share V's head_size.");
    }
    args.past_length = args.past_key->shape[2];
  }
  args.total_length = args.past_length + args.sequence_length;

  if (static_cast<DataType>(args.seqlens_k->data_type) != DataType::INT32 ||
      args.seqlens_k->shape.size() != 1 || args.seqlens_k->shape[0] != args.batch) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: seqlens_k must be an INT32 "
                                "tensor with one value per batch.");
  }
  if (static_cast<DataType>(args.total_sequence_length->data_type) != DataType::INT32 ||
      args.total_sequence_length->element_count() != 1) {
    throw std::invalid_argument(
        "onnx_light_cpu::GroupQueryAttention: total_sequence_length must be an INT32 scalar.");
  }
  if (args.total_sequence_length->AsInt32()[0] != args.total_length) {
    throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: total_sequence_length must "
                                "equal past_sequence_length + sequence_length.");
  }
  for (std::int64_t b = 0; b < args.batch; ++b) {
    if (args.seqlens_k->AsInt32()[b] != args.total_length - 1) {
      throw std::invalid_argument(
          "onnx_light_cpu::GroupQueryAttention: every seqlens_k value must equal "
          "total_sequence_length - 1 (a uniform batch is required).");
    }
  }

  if (args.do_rotary) {
    if (args.cos_cache->shape.size() != 2 || args.sin_cache->shape.size() != 2) {
      throw std::invalid_argument(
          "onnx_light_cpu::GroupQueryAttention: cos_cache/sin_cache must have rank 2.");
    }
    if (args.head_dim % 2 != 0) {
      throw std::invalid_argument(
          "onnx_light_cpu::GroupQueryAttention: rotary embeddings require an even head_size.");
    }
    const std::int64_t half = args.head_dim / 2;
    if (args.cos_cache->shape[1] != half || args.sin_cache->shape[1] != half ||
        args.cos_cache->shape[0] != args.sin_cache->shape[0]) {
      throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: cos_cache/sin_cache must "
                                  "both have shape (max_sequence_length, head_size / 2).");
    }
    if (args.position_ids != nullptr) {
      if (static_cast<DataType>(args.position_ids->data_type) != DataType::INT64 ||
          args.position_ids->shape.size() != 2 || args.position_ids->shape[0] != args.batch ||
          args.position_ids->shape[1] != args.sequence_length) {
        throw std::invalid_argument(
            "onnx_light_cpu::GroupQueryAttention: position_ids must be an INT64 tensor with "
            "shape (batch_size, sequence_length).");
      }
    }
    const std::int64_t max_position = args.cos_cache->shape[0];
    const std::int64_t *position_ids_data =
        args.position_ids != nullptr ? args.position_ids->AsInt64() : nullptr;
    const std::int32_t *seqlens_k_data = args.seqlens_k->AsInt32();
    for (std::int64_t b = 0; b < args.batch; ++b) {
      const std::int64_t start =
          position_ids_data != nullptr
              ? position_ids_data[b * args.sequence_length]
              : static_cast<std::int64_t>(seqlens_k_data[b]) + 1 - args.sequence_length;
      if (start < 0 || start + args.sequence_length > max_position) {
        throw std::invalid_argument("onnx_light_cpu::GroupQueryAttention: the RoPE position "
                                    "derived for a batch falls outside cos_cache/sin_cache.");
      }
    }
  }

  return args;
}

// Returns, for each batch, the absolute position of the first token of this
// step: `position_ids[b, 0]` when supplied, otherwise
// `seqlens_k[b] + 1 - sequence_length` (equals `past_length` for a uniform
// batch, per `ResolveAndValidate`'s bounds check above).
std::vector<std::int64_t> ResolveRotaryStarts(const GqaArgs &args) {
  std::vector<std::int64_t> starts(static_cast<std::size_t>(args.batch));
  const std::int64_t *position_ids_data =
      args.position_ids != nullptr ? args.position_ids->AsInt64() : nullptr;
  const std::int32_t *seqlens_k_data = args.seqlens_k->AsInt32();
  for (std::int64_t b = 0; b < args.batch; ++b) {
    starts[static_cast<std::size_t>(b)] =
        position_ids_data != nullptr
            ? position_ids_data[b * args.sequence_length]
            : static_cast<std::int64_t>(seqlens_k_data[b]) + 1 - args.sequence_length;
  }
  return starts;
}

// Applies split-half RoPE (`rotary_interleaved=0`) to every head of `input`
// (rank-3, `(batch, sequence, heads * head_dim)`) at the absolute position
// `starts[batch] + sequence_index`, reading `cos_cache`/`sin_cache` rows
// `(position, 0:head_dim/2)`. Returns a freshly materialized tensor of the
// same shape/type; `input` is left unmodified.
Tensor ApplyRotaryHalf(RuntimeContext *rt, const Tensor &input, std::int64_t heads,
                       std::int64_t head_dim, const Tensor &cos_cache, const Tensor &sin_cache,
                       const std::vector<std::int64_t> &starts) {
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
    const std::int64_t start = starts[static_cast<std::size_t>(b)];
    for (std::int64_t s = 0; s < sequence_length; ++s) {
      const std::int64_t position = start + s;
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

// Materializes `present_key`/`present_value`: a rank-4
// `(batch, heads, past_length + sequence_length, dim)` tensor holding
// `past` (when non-null) followed by `current` (rank-3,
// `(batch, sequence, heads * dim)`, already rotated for a key cache).
Tensor MakePresentCache(RuntimeContext *rt, int output_slot, DataType dtype, std::int64_t batch,
                        std::int64_t heads, std::int64_t past_length, std::int64_t sequence_length,
                        std::int64_t dim, const Tensor *past, const Tensor &current) {
  const std::int64_t total_length = past_length + sequence_length;
  const Shape shape{batch, heads, total_length, dim};
  const std::size_t element_bytes = ElementByteWidth(dtype);
  const std::size_t total_bytes =
      static_cast<std::size_t>(batch * heads * total_length * dim) * element_bytes;
  Tensor present =
      rt != nullptr
          ? rt->MakeOutputTensor(output_slot, static_cast<std::int32_t>(dtype), shape, total_bytes)
          : rt_ns::MakeOutputTensor(static_cast<std::int32_t>(dtype), shape, total_bytes, nullptr);
  std::uint8_t *dst = present.mutable_bytes();
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
  return present;
}

NodeProto MakeAttentionNode(const GqaArgs &args) {
  NodeProto attention;
  attention.set_op_type("Attention");
  attention.add_input("query");
  attention.add_input("key");
  attention.add_input("value");
  if (args.attention_bias != nullptr) {
    attention.add_input("attention_bias");
  }
  attention.add_output("output");
  AddIntAttribute(attention, "q_num_heads", args.num_heads);
  AddIntAttribute(attention, "kv_num_heads", args.kv_num_heads);
  AddIntAttribute(attention, "is_causal", args.causal ? 1 : 0);
  if (args.scale.has_value()) {
    AddFloatAttribute(attention, "scale", *args.scale);
  }
  if (args.softcap != 0.0f) {
    AddFloatAttribute(attention, "softcap", args.softcap);
  }
  return attention;
}

// Shared computation core used by both `Run` and `operator()`: applies RoPE
// (when configured), materializes `present_key`/`present_value` (when
// requested), and delegates the attention score/softmax/value reduction to
// `attention`.
Tensor Compute(const AttentionKernel &attention, const GqaArgs &args, RuntimeContext *rt,
               Tensor *present_key, Tensor *present_value) {
  const Tensor *q_for_attention = args.query;
  const Tensor *k_for_attention = args.key;
  Tensor rotated_query;
  Tensor rotated_key;
  if (args.do_rotary) {
    const std::vector<std::int64_t> starts = ResolveRotaryStarts(args);
    rotated_query = ApplyRotaryHalf(rt, *args.query, args.num_heads, args.head_dim, *args.cos_cache,
                                    *args.sin_cache, starts);
    rotated_key = ApplyRotaryHalf(rt, *args.key, args.kv_num_heads, args.head_dim, *args.cos_cache,
                                  *args.sin_cache, starts);
    q_for_attention = &rotated_query;
    k_for_attention = &rotated_key;
  }

  if (args.has_present_key) {
    Tensor key_cache =
        MakePresentCache(rt, 1, args.dtype, args.batch, args.kv_num_heads, args.past_length,
                         args.sequence_length, args.head_dim, args.past_key, *k_for_attention);
    Tensor value_cache =
        MakePresentCache(rt, 2, args.dtype, args.batch, args.kv_num_heads, args.past_length,
                         args.sequence_length, args.v_head_dim, args.past_value, *args.value);
    if (present_key != nullptr) {
      *present_key = std::move(key_cache);
    }
    if (present_value != nullptr) {
      *present_value = std::move(value_cache);
    }
  }

  const NodeProto attention_node = MakeAttentionNode(args);
  return attention(attention_node, *q_for_attention, *k_for_attention, *args.value,
                   args.attention_bias, rt, args.past_key, args.past_value, nullptr);
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
  const auto lookup = [&node, &rt](int index) -> const Tensor & {
    return rt_ns::GetInput(node, index, rt.tensors());
  };
  const GqaArgs args = ResolveAndValidate(node, lookup);

  Tensor present_key;
  Tensor present_value;
  Tensor output = Compute(attention_, args, &rt, &present_key, &present_value);
  if (args.has_present_key) {
    rt_ns::SetOutput(node, 1, std::move(present_key), rt);
    rt_ns::SetOutput(node, 2, std::move(present_value), rt);
  }
  rt_ns::SetOutput(node, 0, std::move(output), rt);
}

Tensor GroupQueryAttentionKernel::operator()(
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
          "onnx_light_cpu::GroupQueryAttention: node declares an input that has no matching "
          "tensor argument.");
    }
    return *inputs[static_cast<std::size_t>(index)];
  };
  const GqaArgs args = ResolveAndValidate(node, lookup);
  return Compute(attention_, args, rt, present_key, present_value);
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
