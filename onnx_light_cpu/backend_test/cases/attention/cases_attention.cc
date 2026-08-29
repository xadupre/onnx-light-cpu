// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/attention/include_attention_cases.h"

#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"
#include "onnx_light_cpu/kernels/attention/attention_kernel.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/random.h"

#include <algorithm>
#include <cstdint>
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
                            std::optional<float> scale = std::nullopt) {
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
      return IoData{std::move(inputs), {}, false};
    }
    const onnx_light_cpu::AttentionKernel kernel{KernelContext{DefaultOpset(23)}};
    Tensor y = kernel(node, inputs[0], inputs[1], inputs[2], has_mask ? &inputs[3] : nullptr,
                      nullptr, cache == Cache::kInternal ? &inputs[has_mask ? 4 : 3] : nullptr,
                      cache == Cache::kInternal ? &inputs[has_mask ? 5 : 4] : nullptr,
                      cache == Cache::kNonpad ? &inputs.back() : nullptr);
    return IoData{std::move(inputs), {std::move(y)}};
  };
  if (benchmark) {
    Expect(registry, node, name, {opset}, input_counts, {y_count}, std::move(build_data));
  } else {
    Expect(registry, node, name, {opset}, input_counts, {y_count},
           [build_data = std::move(build_data)]() mutable { return build_data(true); });
  }
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
}

} // namespace onnx_light_cpu::backend_test
