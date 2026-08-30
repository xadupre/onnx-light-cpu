// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/attention/include_attention_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/impl/attention/attention_plan.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/attention/attention_kernel.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/random.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {

namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::Expect;
using bt_ns::IoData;
using bt_ns::TestCase;
using bt_ns::TestMode;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::DefaultOpset;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Tensor;

enum class Geometry { kMha, kGqa, kMqa };
enum class Mask { kNone, kCausal, kBoolean, kAdditive };
enum class Cache { kStateless, kInternal, kNonpad };

const char *GeometryName(Geometry geometry) {
  switch (geometry) {
  case Geometry::kMha:
    return "mha";
  case Geometry::kGqa:
    return "gqa";
  case Geometry::kMqa:
    return "mqa";
  }
  return "mha";
}

const char *MaskName(Mask mask) {
  switch (mask) {
  case Mask::kNone:
    return "none";
  case Mask::kCausal:
    return "causal";
  case Mask::kBoolean:
    return "bool";
  case Mask::kAdditive:
    return "additive";
  }
  return "none";
}

const char *CacheName(Cache cache) {
  switch (cache) {
  case Cache::kStateless:
    return "stateless";
  case Cache::kInternal:
    return "internal_cache";
  case Cache::kNonpad:
    return "nonpad";
  }
  return "stateless";
}

// (q_num_heads, kv_num_heads) for each geometry, kept small for correctness
// cases and moderate for benchmark cases.
std::pair<std::int64_t, std::int64_t> HeadCounts(Geometry geometry, bool benchmark) {
  switch (geometry) {
  case Geometry::kMha:
    return benchmark ? std::pair<std::int64_t, std::int64_t>{12, 12}
                     : std::pair<std::int64_t, std::int64_t>{2, 2};
  case Geometry::kGqa:
    return benchmark ? std::pair<std::int64_t, std::int64_t>{16, 4}
                     : std::pair<std::int64_t, std::int64_t>{4, 2};
  case Geometry::kMqa:
    return benchmark ? std::pair<std::int64_t, std::int64_t>{16, 1}
                     : std::pair<std::int64_t, std::int64_t>{4, 1};
  }
  return {1, 1};
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

NodeProto MakeAttentionNode(bool rank3, std::int64_t q_num_heads, std::int64_t kv_num_heads,
                            bool is_causal, bool has_mask, Cache cache,
                            std::optional<float> scale = std::nullopt, bool present_outputs = false,
                            bool qk_output = false,
                            std::optional<std::int64_t> softmax_precision = std::nullopt) {
  NodeProto node;
  node.set_op_type("Attention");
  node.add_input("Q");
  node.add_input("K");
  node.add_input("V");
  if (has_mask) {
    node.add_input("attn_mask");
  }
  if (cache == Cache::kInternal) {
    if (!has_mask) {
      node.add_input("");
    }
    node.add_input("past_key");
    node.add_input("past_value");
  } else if (cache == Cache::kNonpad) {
    while (node.input_size() < 6) {
      node.add_input("");
    }
    node.add_input("nonpad_kv_seqlen");
  }
  node.add_output("Y");
  if (present_outputs || qk_output) {
    node.add_output(present_outputs ? "present_key" : "");
    node.add_output(present_outputs ? "present_value" : "");
  }
  if (qk_output) {
    node.add_output("qk_matmul_output");
  }
  if (rank3) {
    AddIntAttribute(node, "q_num_heads", q_num_heads);
    AddIntAttribute(node, "kv_num_heads", kv_num_heads);
  }
  if (is_causal) {
    AddIntAttribute(node, "is_causal", 1);
  }
  if (scale.has_value()) {
    AddFloatAttribute(node, "scale", *scale);
  }
  if (softmax_precision.has_value()) {
    AddIntAttribute(node, "softmax_precision", *softmax_precision);
  }
  return node;
}

std::vector<std::int64_t> QkvShape(bool rank3, std::int64_t batch, std::int64_t num_heads,
                                   std::int64_t length, std::int64_t head_dim) {
  if (rank3) {
    return {batch, length, num_heads * head_dim};
  }
  return {batch, num_heads, length, head_dim};
}

// Registers one Attention correctness/benchmark case. ``head_dim`` is shared
// by Q/K/V (v_head_size == head_size) for this stateless FP32 baseline; this
// is why a single ``head_dim`` sizes both the Q/K/V input shapes below and
// the ``y_count`` output element count (``y_count == q_count``, derived from
// the same ``head_dim``).
void RegisterAttentionCase(std::vector<TestCase> &registry, const OpsetId &opset,
                           std::int64_t opset_version, bool rank3, Geometry geometry, Mask mask,
                           std::int64_t batch, std::int64_t q_len, std::int64_t kv_len,
                           std::int64_t head_dim, DataType data_type, Cache cache, bool benchmark,
                           std::int64_t explicit_q_heads = 0, std::int64_t explicit_kv_heads = 0,
                           std::string_view profile = "",
                           std::optional<float> scale = std::nullopt) {
  const auto [default_q_heads, default_kv_heads] = HeadCounts(geometry, benchmark);
  const std::int64_t q_heads = explicit_q_heads > 0 ? explicit_q_heads : default_q_heads;
  const std::int64_t kv_heads = explicit_kv_heads > 0 ? explicit_kv_heads : default_kv_heads;
  const bool is_causal = mask == Mask::kCausal;
  const bool has_mask = mask == Mask::kBoolean || mask == Mask::kAdditive;

  std::string name = "test_cpu_attention_";
  if (!profile.empty()) {
    name += std::string(profile) + "_";
  }
  name += "opset" + std::to_string(opset_version) + "_" +
          (rank3 ? std::string("rank3") : std::string("rank4")) + "_" + GeometryName(geometry) +
          "_q" + std::to_string(q_len) + "_kv" + std::to_string(kv_len) + "_hd" +
          std::to_string(head_dim);
  if (!profile.empty()) {
    name += "_qh" + std::to_string(q_heads) + "_kvh" + std::to_string(kv_heads);
  }
  name += "_" + std::string(MaskName(mask)) + "_" + CacheName(cache) + "_" +
          DataTypeSuffix(data_type) + (benchmark ? "_benchmark" : "");

  const NodeProto node =
      MakeAttentionNode(rank3, q_heads, kv_heads, is_causal, has_mask, cache, scale);
  const std::int64_t current_kv_len = cache == Cache::kInternal ? 1 : kv_len;
  const std::int64_t past_len = cache == Cache::kInternal ? kv_len - current_kv_len : 0;
  const std::vector<std::int64_t> q_shape = QkvShape(rank3, batch, q_heads, q_len, head_dim);
  const std::vector<std::int64_t> k_shape =
      QkvShape(rank3, batch, kv_heads, current_kv_len, head_dim);
  const std::vector<std::int64_t> v_shape =
      QkvShape(rank3, batch, kv_heads, current_kv_len, head_dim);
  const std::vector<std::int64_t> past_shape = {batch, kv_heads, past_len, head_dim};
  const std::int64_t q_count = batch * q_heads * q_len * head_dim;
  const std::int64_t k_count = batch * kv_heads * current_kv_len * head_dim;
  const std::int64_t v_count = k_count;
  const std::int64_t y_count = q_count;
  const std::int64_t mask_count = q_len * kv_len;

  std::vector<std::int64_t> input_counts = {q_count, k_count, v_count};
  if (has_mask) {
    input_counts.push_back(mask_count);
  }
  if (cache == Cache::kInternal) {
    const std::int64_t past_count = batch * kv_heads * past_len * head_dim;
    input_counts.push_back(past_count);
    input_counts.push_back(past_count);
  } else if (cache == Cache::kNonpad) {
    input_counts.push_back(batch);
  }

  auto build_data = [q_shape, k_shape, v_shape, past_shape, has_mask, mask, cache, data_type, node,
                     q_len, kv_len, batch](bool generate_expected_outputs) -> IoData {
    Tensor q = MakeBenchmarkTensor(data_type, q_shape, 501);
    Tensor k = MakeBenchmarkTensor(data_type, k_shape, 502);
    Tensor v = MakeBenchmarkTensor(data_type, v_shape, 503);
    std::optional<Tensor> mask_tensor;
    const Tensor *mask_ptr = nullptr;
    if (has_mask) {
      if (mask == Mask::kBoolean) {
        const std::vector<float> random = rt_ns::Randn<float>({q_len, kv_len}, 504);
        std::vector<std::uint8_t> values(random.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
          values[i] = random[i] > 0.0f ? 1 : 0;
        }
        mask_tensor.emplace(Tensor("", DataType::BOOL, {q_len, kv_len}, std::move(values)));
      } else {
        mask_tensor.emplace(MakeBenchmarkTensor(DataType::FLOAT, {q_len, kv_len}, 505));
      }
      mask_ptr = &(*mask_tensor);
    }
    std::optional<Tensor> past_k;
    std::optional<Tensor> past_v;
    std::optional<Tensor> nonpad;
    if (cache == Cache::kInternal) {
      past_k.emplace(MakeBenchmarkTensor(data_type, past_shape, 506));
      past_v.emplace(MakeBenchmarkTensor(data_type, past_shape, 507));
    } else if (cache == Cache::kNonpad) {
      const std::int64_t nonpad_length = std::max<std::int64_t>(1, kv_len - 17);
      nonpad.emplace(
          Tensor::FromInt64("", {batch}, std::vector<std::int64_t>(batch, nonpad_length)));
    }
    std::vector<Tensor> inputs{std::move(q), std::move(k), std::move(v)};
    if (mask_tensor) {
      inputs.push_back(std::move(*mask_tensor));
    }
    if (past_k) {
      inputs.push_back(std::move(*past_k));
      inputs.push_back(std::move(*past_v));
    } else if (nonpad) {
      inputs.push_back(std::move(*nonpad));
    }
    if (!generate_expected_outputs) {
      return IoData{std::move(inputs), {}, {}, false};
    }
    const onnx_light_cpu::AttentionKernel kernel{KernelContext{DefaultOpset(23)}};
    Tensor y = kernel(node, inputs[0], inputs[1], inputs[2], has_mask ? &inputs[3] : nullptr,
                      nullptr, cache == Cache::kInternal ? &inputs[has_mask ? 4 : 3] : nullptr,
                      cache == Cache::kInternal ? &inputs[has_mask ? 5 : 4] : nullptr,
                      cache == Cache::kNonpad ? &inputs.back() : nullptr);
    return IoData{std::move(inputs), {std::move(y)}};
  };
  if (benchmark) {
    Expect(registry, node, name, {opset}, input_counts, {y_count}, std::move(build_data),
           "backend-test", bt_ns::TestCaseTag::NONE,
           {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(data_type), q_shape)});
  } else {
    Expect(registry, node, name, {opset}, input_counts, {y_count},
           [build_data = std::move(build_data)]() mutable { return build_data(true); });
  }
}

// Registers a present-key/present-value test case. The node has present_key
// and present_value outputs wired; the expected outputs are built by
// manually concatenating past_key + key (and past_value + value).
void RegisterAttentionPresentCase(std::vector<TestCase> &registry, const OpsetId &opset,
                                  std::int64_t opset_version, bool rank3, Geometry geometry,
                                  Mask mask, std::int64_t batch, std::int64_t q_len,
                                  std::int64_t kv_len, std::int64_t head_dim, DataType data_type,
                                  bool benchmark) {
  const auto [q_heads, kv_heads] = HeadCounts(geometry, benchmark);
  const bool is_causal = mask == Mask::kCausal;
  const bool has_mask = mask == Mask::kBoolean || mask == Mask::kAdditive;
  const std::int64_t past_len = std::max<std::int64_t>(1, kv_len / 2);
  const std::int64_t current_kv_len = kv_len - past_len;

  std::string name = "test_cpu_attention_present_opset" + std::to_string(opset_version) + "_" +
                     (rank3 ? std::string("rank3") : std::string("rank4")) + "_" +
                     GeometryName(geometry) + "_q" + std::to_string(q_len) + "_kv" +
                     std::to_string(kv_len) + "_hd" + std::to_string(head_dim) + "_" +
                     MaskName(mask) + "_" + DataTypeSuffix(data_type);

  const NodeProto node = MakeAttentionNode(rank3, q_heads, kv_heads, is_causal, has_mask,
                                           Cache::kInternal, std::nullopt, true);
  const std::vector<std::int64_t> q_shape = QkvShape(rank3, batch, q_heads, q_len, head_dim);
  const std::vector<std::int64_t> k_shape =
      QkvShape(rank3, batch, kv_heads, current_kv_len, head_dim);
  const std::vector<std::int64_t> v_shape =
      QkvShape(rank3, batch, kv_heads, current_kv_len, head_dim);
  const std::vector<std::int64_t> past_shape = {batch, kv_heads, past_len, head_dim};
  const std::int64_t y_count = batch * q_heads * q_len * head_dim;
  const std::int64_t present_count = batch * kv_heads * kv_len * head_dim;
  const std::int64_t mask_count = has_mask ? q_len * kv_len : 0;

  std::vector<std::int64_t> input_counts = {batch * q_heads * q_len * head_dim,
                                            batch * kv_heads * current_kv_len * head_dim,
                                            batch * kv_heads * current_kv_len * head_dim};
  if (has_mask) {
    input_counts.push_back(mask_count);
  }
  const std::int64_t past_count = batch * kv_heads * past_len * head_dim;
  input_counts.push_back(past_count);
  input_counts.push_back(past_count);

  Expect(
      registry, node, name, {opset}, input_counts, {y_count, present_count, present_count},
      [q_shape, k_shape, v_shape, past_shape, has_mask, mask, data_type, node, q_len, kv_len, batch,
       kv_heads, head_dim, past_len, current_kv_len, rank3]() -> IoData {
        const onnx_light_cpu::AttentionKernel kernel{KernelContext{DefaultOpset(23)}};
        Tensor q = MakeBenchmarkTensor(data_type, q_shape, 601);
        Tensor k = MakeBenchmarkTensor(data_type, k_shape, 602);
        Tensor v = MakeBenchmarkTensor(data_type, v_shape, 603);
        Tensor past_k_t = MakeBenchmarkTensor(data_type, past_shape, 604);
        Tensor past_v_t = MakeBenchmarkTensor(data_type, past_shape, 605);
        std::optional<Tensor> mask_tensor;
        const Tensor *mask_ptr = nullptr;
        if (has_mask) {
          if (mask == Mask::kBoolean) {
            const std::vector<float> random = rt_ns::Randn<float>({q_len, kv_len}, 606);
            std::vector<std::uint8_t> values(random.size());
            for (std::size_t i = 0; i < values.size(); ++i) {
              values[i] = random[i] > 0.0f ? 1 : 0;
            }
            mask_tensor.emplace(Tensor("", DataType::BOOL, {q_len, kv_len}, std::move(values)));
          } else {
            mask_tensor.emplace(MakeBenchmarkTensor(DataType::FLOAT, {q_len, kv_len}, 607));
          }
          mask_ptr = &(*mask_tensor);
        }
        Tensor y = kernel(node, q, k, v, mask_ptr, nullptr, &past_k_t, &past_v_t);

        // Build expected present_key = concat(past_key, key) along sequence.
        // Present output is always rank-4 (batch, kv_heads, total_kv_len, head_dim).
        const std::size_t element_bytes =
            data_type == DataType::FLOAT ? sizeof(float) : sizeof(std::uint16_t);
        auto build_present = [&](const Tensor &past, const Tensor &current, bool is_rank3) {
          const std::int64_t total = past_len + current_kv_len;
          const std::vector<std::int64_t> shape = {batch, kv_heads, total, head_dim};
          const std::size_t count = batch * kv_heads * total * head_dim;
          std::vector<std::uint8_t> data(count * element_bytes, 0);
          for (std::int64_t b = 0; b < batch; ++b) {
            for (std::int64_t h = 0; h < kv_heads; ++h) {
              const std::size_t dst_base =
                  static_cast<std::size_t>((b * kv_heads + h) * total * head_dim);
              // Past: always rank-4 (batch, kv_heads, past_len, head_dim)
              const std::size_t past_base =
                  static_cast<std::size_t>((b * kv_heads + h) * past_len * head_dim);
              std::memcpy(data.data() + dst_base * element_bytes,
                          reinterpret_cast<const char *>(past.bytes()) + past_base * element_bytes,
                          static_cast<std::size_t>(past_len * head_dim) * element_bytes);
              // Current segment
              for (std::int64_t s = 0; s < current_kv_len; ++s) {
                std::size_t src;
                if (is_rank3) {
                  // rank-3: (batch, kv_length, kv_heads * head_dim)
                  src = static_cast<std::size_t>((b * current_kv_len + s) * kv_heads * head_dim +
                                                 h * head_dim);
                } else {
                  // rank-4: (batch, kv_heads, kv_length, head_dim)
                  src = static_cast<std::size_t>((b * kv_heads + h) * current_kv_len * head_dim +
                                                 s * head_dim);
                }
                const std::size_t dst =
                    dst_base + static_cast<std::size_t>((past_len + s) * head_dim);
                std::memcpy(data.data() + dst * element_bytes,
                            reinterpret_cast<const char *>(current.bytes()) + src * element_bytes,
                            static_cast<std::size_t>(head_dim) * element_bytes);
              }
            }
          }
          return Tensor("", data_type, shape, std::move(data));
        };
        Tensor present_key = build_present(past_k_t, k, rank3);
        Tensor present_value = build_present(past_v_t, v, rank3);

        std::vector<Tensor> inputs{std::move(q), std::move(k), std::move(v)};
        if (mask_tensor) {
          inputs.push_back(std::move(*mask_tensor));
        }
        inputs.push_back(std::move(past_k_t));
        inputs.push_back(std::move(past_v_t));
        return IoData{std::move(inputs),
                      {std::move(y), std::move(present_key), std::move(present_value)}};
      });
}

void RegisterAttentionQkOutputCase(std::vector<TestCase> &registry, const OpsetId &opset,
                                   std::int64_t opset_version, bool rank3, Geometry geometry,
                                   Mask mask, std::int64_t batch, std::int64_t q_len,
                                   std::int64_t kv_len, std::int64_t head_dim, DataType data_type,
                                   bool benchmark) {
  const auto [q_heads, kv_heads] = HeadCounts(geometry, benchmark);
  const bool is_causal = mask == Mask::kCausal;
  const bool has_mask = mask == Mask::kBoolean || mask == Mask::kAdditive;

  std::string name = "test_cpu_attention_qk_output_opset" + std::to_string(opset_version) + "_" +
                     (rank3 ? std::string("rank3") : std::string("rank4")) + "_" +
                     GeometryName(geometry) + "_q" + std::to_string(q_len) + "_kv" +
                     std::to_string(kv_len) + "_hd" + std::to_string(head_dim) + "_" +
                     MaskName(mask) + (data_type == DataType::FLOAT ? "_float32" : "_float16");

  const NodeProto node = MakeAttentionNode(rank3, q_heads, kv_heads, is_causal, has_mask,
                                           Cache::kStateless, std::nullopt, false, true);
  const std::vector<std::int64_t> q_shape = QkvShape(rank3, batch, q_heads, q_len, head_dim);
  const std::vector<std::int64_t> k_shape = QkvShape(rank3, batch, kv_heads, kv_len, head_dim);
  const std::vector<std::int64_t> v_shape = QkvShape(rank3, batch, kv_heads, kv_len, head_dim);
  const std::int64_t q_count = batch * q_heads * q_len * head_dim;
  const std::int64_t k_count = batch * kv_heads * kv_len * head_dim;
  const std::int64_t y_count = q_count;
  // qk_matmul_output shape: (batch, q_heads, q_len, kv_len)
  const std::int64_t qk_count = batch * q_heads * q_len * kv_len;
  const std::int64_t mask_count = has_mask ? q_len * kv_len : 0;

  std::vector<std::int64_t> input_counts = {q_count, k_count, k_count};
  if (has_mask) {
    input_counts.push_back(mask_count);
  }

  // Node has empty present outputs but wired qk_matmul_output at index 3.
  // The framework only expects tensors for non-empty outputs (Y and qk).
  Expect(
      registry, node, name, {opset}, input_counts, {y_count, qk_count},
      [q_shape, k_shape, v_shape, has_mask, mask, node, q_len, kv_len, batch, q_heads, kv_heads,
       head_dim, data_type]() -> IoData {
        const onnx_light_cpu::AttentionKernel kernel{KernelContext{DefaultOpset(23)}};
        Tensor q = MakeBenchmarkTensor(data_type, q_shape, 701);
        Tensor k = MakeBenchmarkTensor(data_type, k_shape, 702);
        Tensor v = MakeBenchmarkTensor(data_type, v_shape, 703);
        std::optional<Tensor> mask_tensor;
        const Tensor *mask_ptr = nullptr;
        if (has_mask) {
          if (mask == Mask::kBoolean) {
            const std::vector<float> random = rt_ns::Randn<float>({q_len, kv_len}, 704);
            std::vector<std::uint8_t> values(random.size());
            for (std::size_t i = 0; i < values.size(); ++i) {
              values[i] = random[i] > 0.0f ? 1 : 0;
            }
            mask_tensor.emplace(Tensor("", DataType::BOOL, {q_len, kv_len}, std::move(values)));
          } else {
            mask_tensor.emplace(MakeBenchmarkTensor(DataType::FLOAT, {q_len, kv_len}, 705));
          }
          mask_ptr = &(*mask_tensor);
        }
        Tensor y = kernel(node, q, k, v, mask_ptr);
        // Compute qk_matmul_output via the materialized path directly.
        onnx_light_cpu::AttentionDescriptor desc;
        desc.has_qk_matmul_output = true;
        desc.is_causal = (mask == Mask::kCausal);
        const onnx_light_cpu::AttentionLayout layout =
            q_shape.size() == 3 ? onnx_light_cpu::AttentionLayout::kRank3
                                : onnx_light_cpu::AttentionLayout::kRank4;
        if (q_shape.size() == 3) {
          desc.q_num_heads = q_heads;
          desc.kv_num_heads = kv_heads;
        }
        onnx_light_cpu::AttentionMaskKind mk = onnx_light_cpu::AttentionMaskKind::kNone;
        if (mask == Mask::kBoolean)
          mk = onnx_light_cpu::AttentionMaskKind::kBoolean;
        else if (mask == Mask::kAdditive)
          mk = onnx_light_cpu::AttentionMaskKind::kAdditive;
        std::vector<std::int64_t> mask_shp;
        if (has_mask)
          mask_shp = {q_len, kv_len};
        onnx_light_cpu::AttentionPlan plan(desc, layout, q_shape, k_shape, v_shape, mask_shp, mk);
        const std::size_t qk_count_s = static_cast<std::size_t>(batch * q_heads * q_len * kv_len);
        std::vector<float> qk_data(qk_count_s);
        std::vector<float> y_mat(static_cast<std::size_t>(y.element_count()));
        std::vector<float> q_float(static_cast<std::size_t>(q.element_count()));
        std::vector<float> k_float(static_cast<std::size_t>(k.element_count()));
        std::vector<float> v_float(static_cast<std::size_t>(v.element_count()));
        const float *q_data = reinterpret_cast<const float *>(q.bytes());
        const float *k_data = reinterpret_cast<const float *>(k.bytes());
        const float *v_data = reinterpret_cast<const float *>(v.bytes());
        if (data_type == DataType::FLOAT16) {
          detail::ConvertFloat16ToFloat32(reinterpret_cast<const std::uint16_t *>(q.bytes()),
                                          q_float.data(), q_float.size());
          detail::ConvertFloat16ToFloat32(reinterpret_cast<const std::uint16_t *>(k.bytes()),
                                          k_float.data(), k_float.size());
          detail::ConvertFloat16ToFloat32(reinterpret_cast<const std::uint16_t *>(v.bytes()),
                                          v_float.data(), v_float.size());
          q_data = q_float.data();
          k_data = k_float.data();
          v_data = v_float.data();
        }
        const void *mask_raw = mask_ptr ? (mk == onnx_light_cpu::AttentionMaskKind::kBoolean
                                               ? static_cast<const void *>(mask_ptr->AsBool())
                                               : static_cast<const void *>(mask_ptr->bytes()))
                                        : nullptr;
        onnx_light_cpu::ComputeAttentionFloat32Materialized(plan, q_data, k_data, v_data, mask_raw,
                                                            y_mat.data(), nullptr, nullptr, nullptr,
                                                            qk_data.data());

        std::vector<std::uint8_t> qk_raw;
        if (data_type == DataType::FLOAT) {
          const auto *qk_bytes = reinterpret_cast<const std::uint8_t *>(qk_data.data());
          qk_raw.assign(qk_bytes, qk_bytes + qk_count_s * sizeof(float));
        } else {
          std::vector<std::uint16_t> qk_half(qk_count_s);
          detail::ConvertFloat32ToFloat16(qk_data.data(), qk_half.data(), qk_count_s);
          const auto *qk_bytes = reinterpret_cast<const std::uint8_t *>(qk_half.data());
          qk_raw.assign(qk_bytes, qk_bytes + qk_count_s * sizeof(std::uint16_t));
        }
        Tensor qk_tensor("", static_cast<int32_t>(data_type), {batch, q_heads, q_len, kv_len},
                         std::move(qk_raw));

        std::vector<Tensor> inputs{std::move(q), std::move(k), std::move(v)};
        if (mask_tensor) {
          inputs.push_back(std::move(*mask_tensor));
        }
        // Only return non-empty output tensors (Y and qk_matmul_output).
        return IoData{std::move(inputs), {std::move(y), std::move(qk_tensor)}};
      });
}

// Registers a test case with explicit softmax_precision attribute.
void RegisterAttentionSoftmaxPrecisionCase(std::vector<TestCase> &registry, const OpsetId &opset,
                                           std::int64_t opset_version,
                                           std::int64_t softmax_precision_value, bool benchmark) {
  std::string name = "test_cpu_attention_softmax_precision_" +
                     std::to_string(softmax_precision_value) + "_opset" +
                     std::to_string(opset_version) + "_float32";

  const NodeProto node = MakeAttentionNode(false, 2, 2, false, false, Cache::kStateless,
                                           std::nullopt, false, false, softmax_precision_value);
  constexpr std::int64_t batch = 1, q_heads = 2, q_len = 3, kv_len = 4, head_dim = 4;
  const std::vector<std::int64_t> q_shape = {batch, q_heads, q_len, head_dim};
  const std::vector<std::int64_t> k_shape = {batch, q_heads, kv_len, head_dim};
  const std::vector<std::int64_t> v_shape = {batch, q_heads, kv_len, head_dim};
  const std::int64_t count = batch * q_heads * q_len * head_dim;
  const std::int64_t k_count = batch * q_heads * kv_len * head_dim;

  Expect(registry, node, name, {opset}, {count, k_count, k_count}, {count},
         [q_shape, k_shape, v_shape, node]() -> IoData {
           const onnx_light_cpu::AttentionKernel kernel{KernelContext{DefaultOpset(23)}};
           Tensor q = MakeBenchmarkTensor(DataType::FLOAT, q_shape, 801);
           Tensor k = MakeBenchmarkTensor(DataType::FLOAT, k_shape, 802);
           Tensor v = MakeBenchmarkTensor(DataType::FLOAT, v_shape, 803);
           Tensor y = kernel(node, q, k, v, nullptr);
           std::vector<Tensor> inputs{std::move(q), std::move(k), std::move(v)};
           return IoData{std::move(inputs), {std::move(y)}};
         });
}

// Registers one Attention window test case. Builds the expected output using
// the Q=0, K=0, position-value V pattern (same as upstream
// MakeUniformWindowReference4). This tests the full kernel path including
// attribute parsing.
void RegisterAttentionWindowCase(std::vector<TestCase> &registry, const OpsetId &opset,
                                 const std::string &name, std::int64_t batch, std::int64_t q_heads,
                                 std::int64_t kv_heads, std::int64_t q_len, std::int64_t kv_len,
                                 std::int64_t head_dim, bool is_causal, std::int64_t left_window,
                                 std::int64_t right_window) {
  NodeProto node;
  node.set_op_type("Attention");
  node.add_input("Q");
  node.add_input("K");
  node.add_input("V");
  node.add_output("Y");
  if (is_causal) {
    AddIntAttribute(node, "is_causal", 1);
  }
  if (left_window >= 0) {
    AddIntAttribute(node, "left_window_size", left_window);
  }
  if (right_window >= 0) {
    AddIntAttribute(node, "right_window_size", right_window);
  }

  const std::int64_t q_count = batch * q_heads * q_len * head_dim;
  const std::int64_t k_count = batch * kv_heads * kv_len * head_dim;
  Expect(
      registry, node, name, {opset}, {q_count, k_count, k_count}, {q_count},
      [batch, q_heads, kv_heads, q_len, kv_len, head_dim, node]() -> IoData {
        const onnx_light_cpu::AttentionKernel kernel{KernelContext{DefaultOpset(25)}};
        const std::vector<std::int64_t> q_shape = {batch, q_heads, q_len, head_dim};
        const std::vector<std::int64_t> k_shape = {batch, kv_heads, kv_len, head_dim};
        const std::vector<std::int64_t> v_shape = k_shape;
        Tensor q = Tensor::FromFloat(
            "Q", q_shape,
            std::vector<float>(static_cast<std::size_t>(batch * q_heads * q_len * head_dim), 0.0f));
        Tensor k = Tensor::FromFloat(
            "K", k_shape,
            std::vector<float>(static_cast<std::size_t>(batch * kv_heads * kv_len * head_dim),
                               0.0f));
        // V[b,h,s,d] = 100*h + 10*d + s (position value tensor).
        std::vector<float> v_vals(static_cast<std::size_t>(batch * kv_heads * kv_len * head_dim));
        for (std::int64_t b = 0; b < batch; ++b)
          for (std::int64_t h = 0; h < kv_heads; ++h)
            for (std::int64_t s = 0; s < kv_len; ++s)
              for (std::int64_t d = 0; d < head_dim; ++d)
                v_vals[static_cast<std::size_t>(((b * kv_heads + h) * kv_len + s) * head_dim + d)] =
                    static_cast<float>(100 * h + 10 * d + s);
        Tensor v = Tensor::FromFloat("V", v_shape, v_vals);
        Tensor y = kernel(node, q, k, v, nullptr);
        return IoData{{std::move(q), std::move(k), std::move(v)}, {std::move(y)}};
      });
}

} // namespace

void RegisterCpuAttentionCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset23 = DefaultOpset(23);
  const OpsetId opset24 = DefaultOpset(24);

  if (mode == TestMode::BENCHMARK) {
    constexpr std::int64_t kHeadDim = 64;
    for (DataType data_type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
      for (Mask mask : {Mask::kNone, Mask::kCausal, Mask::kBoolean, Mask::kAdditive}) {
        RegisterAttentionCase(registry, opset23, 23, false, Geometry::kMha, mask, 1, 128, 128,
                              kHeadDim, data_type, Cache::kStateless, true);
      }
      for (Geometry geometry : {Geometry::kGqa, Geometry::kMqa}) {
        RegisterAttentionCase(registry, opset23, 23, false, geometry, Mask::kCausal, 1, 128, 128,
                              kHeadDim, data_type, Cache::kStateless, true);
      }
      RegisterAttentionCase(registry, opset23, 23, true, Geometry::kMha, Mask::kNone, 1, 128, 128,
                            kHeadDim, data_type, Cache::kStateless, true);
      for (std::int64_t q_len : {std::int64_t{1}, std::int64_t{2}, std::int64_t{8},
                                 std::int64_t{16}, std::int64_t{512}}) {
        RegisterAttentionCase(registry, opset23, 23, false, Geometry::kMha, Mask::kNone, 1, q_len,
                              128, kHeadDim, data_type, Cache::kStateless, true);
      }
      for (std::int64_t kv_len :
           {std::int64_t{1}, std::int64_t{1024}, std::int64_t{4096}, std::int64_t{8192}}) {
        RegisterAttentionCase(registry, opset23, 23, false, Geometry::kMha, Mask::kNone, 1, 128,
                              kv_len, kHeadDim, data_type, Cache::kStateless, true);
      }
      RegisterAttentionCase(registry, opset23, 23, false, Geometry::kMha, Mask::kCausal, 1, 128,
                            128, 128, data_type, Cache::kStateless, true);
      RegisterAttentionCase(registry, opset23, 23, false, Geometry::kMha, Mask::kNone, 1, 1, 1024,
                            kHeadDim, data_type, Cache::kInternal, true);
      RegisterAttentionCase(registry, opset24, 24, false, Geometry::kMha, Mask::kCausal, 1, 8, 1024,
                            kHeadDim, data_type, Cache::kNonpad, true);
    }
    constexpr float kQwen3Scale = 0.0883883461356163f;
    RegisterAttentionCase(registry, opset23, 23, true, Geometry::kGqa, Mask::kCausal, 1, 128, 128,
                          128, DataType::FLOAT16, Cache::kStateless, true, 32, 8, "llm_qwen3_8b",
                          kQwen3Scale);
    for (std::int64_t total_kv_len : {std::int64_t{128}, std::int64_t{1024}, std::int64_t{4096}}) {
      RegisterAttentionCase(registry, opset23, 23, true, Geometry::kGqa, Mask::kCausal, 1, 1,
                            total_kv_len, 128, DataType::FLOAT16, Cache::kInternal, true, 32, 8,
                            "llm_qwen3_8b", kQwen3Scale);
    }
    return;
  }

  // Correctness cases: pair the full mask x geometry x layout matrix with
  // small, cheap shapes for both opsets (stateless behavior is identical).
  for (const auto &[opset, opset_version] : {std::pair<OpsetId, std::int64_t>{opset23, 23},
                                             std::pair<OpsetId, std::int64_t>{opset24, 24}}) {
    for (bool rank3 : {false, true}) {
      for (Geometry geometry : {Geometry::kMha, Geometry::kGqa, Geometry::kMqa}) {
        for (Mask mask : {Mask::kNone, Mask::kCausal, Mask::kBoolean, Mask::kAdditive}) {
          RegisterAttentionCase(registry, opset, opset_version, rank3, geometry, mask, 2, 4, 5, 8,
                                DataType::FLOAT, Cache::kStateless, false);
        }
      }
    }
  }

  // Present key/value output cases: internal cache, FP32, and FP16.
  for (DataType data_type : {DataType::FLOAT, DataType::FLOAT16}) {
    RegisterAttentionPresentCase(registry, opset23, 23, false, Geometry::kMha, Mask::kNone, 1, 2, 4,
                                 8, data_type, false);
    RegisterAttentionPresentCase(registry, opset23, 23, false, Geometry::kGqa, Mask::kCausal, 1, 2,
                                 5, 8, data_type, false);
    RegisterAttentionPresentCase(registry, opset23, 23, true, Geometry::kMha, Mask::kNone, 1, 2, 4,
                                 8, data_type, false);
  }

  RegisterAttentionQkOutputCase(registry, opset23, 23, false, Geometry::kMha, Mask::kNone, 1, 3, 4,
                                8, DataType::FLOAT, false);
  RegisterAttentionQkOutputCase(registry, opset23, 23, false, Geometry::kMha, Mask::kCausal, 1, 3,
                                4, 8, DataType::FLOAT, false);
  RegisterAttentionQkOutputCase(registry, opset23, 23, false, Geometry::kGqa, Mask::kNone, 1, 3, 4,
                                8, DataType::FLOAT, false);
  RegisterAttentionQkOutputCase(registry, opset23, 23, false, Geometry::kMha, Mask::kNone, 1, 3, 4,
                                8, DataType::FLOAT16, false);

  // Explicit softmax_precision = 0 (UNDEFINED, default) and 1 (FLOAT):
  // both must not throw.
  RegisterAttentionSoftmaxPrecisionCase(registry, opset23, 23, 0, false);
  RegisterAttentionSoftmaxPrecisionCase(registry, opset23, 23, 1, false);

  // Window attention cases (opset 25): local window and bidirectional window.
  const OpsetId opset25 = DefaultOpset(25);
  RegisterAttentionWindowCase(registry, opset25, "test_cpu_attention_local_window_causal_float32",
                              2, 3, 3, 4, 6, 8, true, 2, -1);
  RegisterAttentionWindowCase(registry, opset25, "test_cpu_attention_bidirectional_window_float32",
                              1, 1, 1, 5, 5, 1, false, 1, 2);
}

} // namespace onnx_light_cpu::backend_test
