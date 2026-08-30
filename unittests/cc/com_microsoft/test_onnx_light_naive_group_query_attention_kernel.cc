// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/com_microsoft/group_query_attention_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_group_query_attention_kernel.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId("com.microsoft", 1)); }

void AddIntAttribute(NodeProto &node, const char *name, std::int64_t value) {
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, name, value);
}

void AddFloatAttribute(NodeProto &node, const char *name, float value) {
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, name, value);
}

// Builds an 11-input/{1,3}-output GroupQueryAttention node; both
// NaiveGroupQueryAttentionKernel and (the already-shipping)
// GroupQueryAttentionKernel accept the exact same wiring, so the same node is
// reused unmodified to drive both kernels in the differential tests below.
NodeProto MakeGqaNode(std::int64_t num_heads, std::int64_t kv_num_heads, bool causal,
                      bool with_cache, bool with_rotary, bool with_bias, bool with_present,
                      bool with_position_ids = false) {
  NodeProto node;
  node.set_op_type("GroupQueryAttention");
  node.set_domain("com.microsoft");
  node.add_input("query");
  node.add_input("key");
  node.add_input("value");
  node.add_input(with_cache ? "past_key" : "");
  node.add_input(with_cache ? "past_value" : "");
  node.add_input("seqlens_k");
  node.add_input("total_sequence_length");
  node.add_input(with_rotary ? "cos_cache" : "");
  node.add_input(with_rotary ? "sin_cache" : "");
  node.add_input(with_position_ids ? "position_ids" : "");
  node.add_input(with_bias ? "attention_bias" : "");
  node.add_output("output");
  if (with_present) {
    node.add_output("present_key");
    node.add_output("present_value");
  }
  AddIntAttribute(node, "num_heads", num_heads);
  AddIntAttribute(node, "kv_num_heads", kv_num_heads);
  AddIntAttribute(node, "causal", causal ? 1 : 0);
  AddIntAttribute(node, "do_rotary", with_rotary ? 1 : 0);
  AddIntAttribute(node, "rotary_interleaved", std::int64_t{0});
  return node;
}

// Deterministic, seeded pseudo-random values in roughly [-0.2, 0.2] so tests
// are reproducible without depending on <random>.
std::vector<float> PseudoRandom(std::size_t count, int seed) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    const int bucket = static_cast<int>((i * 7 + static_cast<std::size_t>(seed) * 13) % 41) - 20;
    values[i] = 0.01f * static_cast<float>(bucket);
  }
  return values;
}

Tensor MakeFloatTensor(const Shape &shape, const std::vector<float> &values) {
  return Tensor::FromFloat("", shape, values);
}

// Encodes `values` as FLOAT16 or BFLOAT16 raw bytes and wraps them into a
// Tensor of the requested `dtype`.
Tensor MakeHalfTensor(DataType dtype, const Shape &shape, const std::vector<float> &values) {
  std::vector<std::uint8_t> bytes(values.size() * sizeof(std::uint16_t));
  auto *words = reinterpret_cast<std::uint16_t *>(bytes.data());
  for (std::size_t i = 0; i < values.size(); ++i) {
    words[i] = dtype == DataType::FLOAT16 ? onnx_light_cpu::detail::FloatToFloat16Bits(values[i])
                                          : onnx_light_cpu::detail::FloatToBFloat16Bits(values[i]);
  }
  return Tensor("", static_cast<std::int32_t>(dtype), shape, std::move(bytes));
}

// Rounds `values` to the nearest value exactly representable in `dtype`, so a
// FLOAT16/BFLOAT16 differential comparison is not dominated by the tensor
// construction's own quantization noise.
std::vector<float> RoundTripThroughHalf(DataType dtype, const std::vector<float> &values) {
  std::vector<float> rounded(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (dtype == DataType::FLOAT16) {
      rounded[i] = onnx_light_cpu::detail::Float16BitsToFloat(
          onnx_light_cpu::detail::FloatToFloat16Bits(values[i]));
    } else {
      rounded[i] = onnx_light_cpu::detail::Bfloat16BitsToFloat(
          onnx_light_cpu::detail::FloatToBFloat16Bits(values[i]));
    }
  }
  return rounded;
}

// A deterministic, non-unit-norm rotary table -- cos_cache/sin_cache do not
// need to come from a genuine RoPE base for this test, only to be shared
// identically between the naive and optimized kernels.
std::vector<float> MakeCosOrSinTable(std::int64_t max_position, std::int64_t half_head_dim,
                                     bool cosine) {
  std::vector<float> values(static_cast<std::size_t>(max_position * half_head_dim));
  for (std::int64_t p = 0; p < max_position; ++p) {
    for (std::int64_t d = 0; d < half_head_dim; ++d) {
      const float angle = 0.05f * static_cast<float>(p + 1) * static_cast<float>(d + 1);
      values[static_cast<std::size_t>(p * half_head_dim + d)] =
          cosine ? std::cos(angle) : std::sin(angle);
    }
  }
  return values;
}

void ExpectTensorsNear(const Tensor &actual, const Tensor &expected, float tolerance) {
  ASSERT_EQ(actual.shape.size(), expected.shape.size());
  for (std::size_t i = 0; i < actual.shape.size(); ++i) {
    EXPECT_EQ(actual.shape[i], expected.shape[i]) << "dim " << i;
  }
  ASSERT_EQ(actual.element_count(), expected.element_count());
  ASSERT_EQ(static_cast<DataType>(actual.data_type), static_cast<DataType>(expected.data_type));
  const DataType dtype = static_cast<DataType>(actual.data_type);
  for (std::int64_t i = 0; i < actual.element_count(); ++i) {
    float actual_value = 0.0f;
    float expected_value = 0.0f;
    if (dtype == DataType::FLOAT) {
      actual_value = actual.AsFloat()[i];
      expected_value = expected.AsFloat()[i];
    } else if (dtype == DataType::FLOAT16) {
      actual_value = onnx_light_cpu::detail::Float16BitsToFloat(
          reinterpret_cast<const std::uint16_t *>(actual.bytes())[i]);
      expected_value = onnx_light_cpu::detail::Float16BitsToFloat(
          reinterpret_cast<const std::uint16_t *>(expected.bytes())[i]);
    } else {
      actual_value = onnx_light_cpu::detail::Bfloat16BitsToFloat(
          reinterpret_cast<const std::uint16_t *>(actual.bytes())[i]);
      expected_value = onnx_light_cpu::detail::Bfloat16BitsToFloat(
          reinterpret_cast<const std::uint16_t *>(expected.bytes())[i]);
    }
    EXPECT_NEAR(actual_value, expected_value, tolerance) << "element " << i;
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Hand-verifiable smoke test.
// ---------------------------------------------------------------------------

// A single query token attending to a single key (no past cache) always
// produces a softmax weight of exactly 1 for that one position, regardless
// of the query/key values or the scale -- so the output must equal `value`
// exactly (broadcast across every query head sharing that one KV head, here
// an MQA configuration with kv_num_heads=1). This is verifiable by hand
// without deriving the softmax/scale arithmetic.
TEST(OnnxLightNaiveGroupQueryAttentionKernel, OneTokenOneKeyOutputEqualsValueForEveryHead) {
  const Tensor query = Tensor::FromFloat("", {1, 1, 4}, {0.1f, -0.2f, 0.3f, -0.4f});
  const Tensor key = Tensor::FromFloat("", {1, 1, 2}, {0.5f, -0.1f});
  const Tensor value = Tensor::FromFloat("", {1, 1, 2}, {0.2f, -0.3f});
  const Tensor seqlens_k = Tensor::FromInt32("", {1}, {0});
  const Tensor total_sequence_length = Tensor::FromInt32("", {}, {1});

  const NodeProto node = MakeGqaNode(/*num_heads=*/2, /*kv_num_heads=*/1, /*causal=*/true,
                                     /*with_cache=*/false, /*with_rotary=*/false,
                                     /*with_bias=*/false, /*with_present=*/false);
  onnx_light_cpu::NaiveGroupQueryAttentionKernel kernel(node, MakeCtx());
  const Tensor output = kernel(node, query, key, value, seqlens_k, total_sequence_length);

  ASSERT_EQ(output.shape.size(), 3u);
  EXPECT_EQ(output.shape[0], 1);
  EXPECT_EQ(output.shape[1], 1);
  EXPECT_EQ(output.shape[2], 4);
  const std::vector<float> expected = {0.2f, -0.3f, 0.2f, -0.3f};
  const float *actual = output.AsFloat();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-6f) << i;
  }
}

// A hand-verifiable two-token causal case: with `causal=1` the first query
// row may only see the first key, so its output must equal `value[0]`
// exactly; the second query row sees both keys with equal QK scores (equal
// query/key rows below), so its output must equal the plain average of both
// value rows.
TEST(OnnxLightNaiveGroupQueryAttentionKernel, TwoTokenCausalFirstRowSeesOnlyFirstKey) {
  const Tensor query = Tensor::FromFloat("", {1, 2, 2}, {0.3f, 0.3f, 0.3f, 0.3f});
  const Tensor key = Tensor::FromFloat("", {1, 2, 2}, {0.3f, 0.3f, 0.3f, 0.3f});
  const Tensor value = Tensor::FromFloat("", {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  const Tensor seqlens_k = Tensor::FromInt32("", {1}, {1});
  const Tensor total_sequence_length = Tensor::FromInt32("", {}, {2});

  const NodeProto node = MakeGqaNode(/*num_heads=*/1, /*kv_num_heads=*/1, /*causal=*/true,
                                     /*with_cache=*/false, /*with_rotary=*/false,
                                     /*with_bias=*/false, /*with_present=*/false);
  onnx_light_cpu::NaiveGroupQueryAttentionKernel kernel(node, MakeCtx());
  const Tensor output = kernel(node, query, key, value, seqlens_k, total_sequence_length);

  ASSERT_EQ(output.element_count(), 4);
  const float *actual = output.AsFloat();
  EXPECT_NEAR(actual[0], 1.0f, 1e-6f);
  EXPECT_NEAR(actual[1], 2.0f, 1e-6f);
  EXPECT_NEAR(actual[2], 2.0f, 1e-5f); // average of value rows [1,2] and [3,4].
  EXPECT_NEAR(actual[3], 3.0f, 1e-5f);
}

TEST(OnnxLightNaiveGroupQueryAttentionKernel, UsesEveryExplicitPositionIdForRotaryCache) {
  const Tensor query = Tensor::FromFloat("", {1, 2, 2}, {1.0f, 0.0f, 1.0f, 0.0f});
  const Tensor key = Tensor::FromFloat("", {1, 2, 2}, {1.0f, 0.0f, 1.0f, 0.0f});
  const Tensor value = Tensor::FromFloat("", {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  const Tensor seqlens_k = Tensor::FromInt32("", {1}, {1});
  const Tensor total_sequence_length = Tensor::FromInt32("", {}, {2});
  const Tensor cos_cache = Tensor::FromFloat("", {3, 1}, {1.0f, 0.0f, -1.0f});
  const Tensor sin_cache = Tensor::FromFloat("", {3, 1}, {0.0f, 1.0f, 0.0f});
  const Tensor position_ids = Tensor::FromInt64("", {1, 2}, {2, 0});
  const NodeProto node = MakeGqaNode(1, 1, true, false, true, false, true, true);

  onnx_light_cpu::NaiveGroupQueryAttentionKernel kernel(node, MakeCtx());
  Tensor present_key;
  Tensor present_value;
  (void)kernel(node, query, key, value, seqlens_k, total_sequence_length, nullptr, nullptr,
               &cos_cache, &sin_cache, &position_ids, nullptr, nullptr, &present_key,
               &present_value);

  ASSERT_EQ(present_key.element_count(), 4);
  EXPECT_FLOAT_EQ(present_key.AsFloat()[0], -1.0f);
  EXPECT_FLOAT_EQ(present_key.AsFloat()[1], 0.0f);
  EXPECT_FLOAT_EQ(present_key.AsFloat()[2], 1.0f);
  EXPECT_FLOAT_EQ(present_key.AsFloat()[3], 0.0f);
}

// ---------------------------------------------------------------------------
// Optimized-vs-naive differential coverage.
// ---------------------------------------------------------------------------

namespace {

struct DifferentialCase {
  std::string name;
  std::int64_t num_heads;
  std::int64_t kv_num_heads;
  std::int64_t batch;
  std::int64_t sequence_length;
  std::int64_t past_length;
  std::int64_t head_dim;
  std::int64_t v_head_dim;
  bool causal;
  bool with_rotary;
  bool with_bias;
  bool with_present;
  int seed;
  bool with_position_ids = false;
  std::optional<float> scale = std::nullopt;
  std::optional<float> softcap = std::nullopt;
  bool expect_parallel = false;
};

struct InlineExecutor {
  std::int64_t dispatches = 0;

  static void Run(void *context, std::int64_t num_blocks, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    for (std::int64_t block = 0; block < num_blocks; ++block) {
      task(task_context, block);
    }
  }
};

// Runs one case through both the optimized GroupQueryAttentionKernel and the
// independent NaiveGroupQueryAttentionKernel and asserts the main output and
// (when requested) present_key/present_value match within `tolerance`.
// `dtype` selects FLOAT, FLOAT16, or BFLOAT16 tensors for every
// floating-point input/output.
void RunDifferentialCase(const DifferentialCase &c, DataType dtype, float tolerance) {
  const std::int64_t total_length = c.past_length + c.sequence_length;
  const std::int64_t q_hidden = c.num_heads * c.head_dim;
  const std::int64_t k_hidden = c.kv_num_heads * c.head_dim;
  const std::int64_t v_hidden = c.kv_num_heads * c.v_head_dim;

  auto to_dtype = [&](const std::vector<float> &values) {
    return dtype == DataType::FLOAT ? values : RoundTripThroughHalf(dtype, values);
  };
  auto make_tensor = [&](const Shape &shape, const std::vector<float> &values) {
    return dtype == DataType::FLOAT ? MakeFloatTensor(shape, values)
                                    : MakeHalfTensor(dtype, shape, values);
  };

  const Tensor query = make_tensor(
      {c.batch, c.sequence_length, q_hidden},
      to_dtype(PseudoRandom(static_cast<std::size_t>(c.batch * c.sequence_length * q_hidden),
                            c.seed + 1)));
  const Tensor key = make_tensor(
      {c.batch, c.sequence_length, k_hidden},
      to_dtype(PseudoRandom(static_cast<std::size_t>(c.batch * c.sequence_length * k_hidden),
                            c.seed + 2)));
  const Tensor value = make_tensor(
      {c.batch, c.sequence_length, v_hidden},
      to_dtype(PseudoRandom(static_cast<std::size_t>(c.batch * c.sequence_length * v_hidden),
                            c.seed + 3)));

  Tensor past_key;
  Tensor past_value;
  const bool with_cache = c.past_length > 0;
  if (with_cache) {
    past_key =
        make_tensor({c.batch, c.kv_num_heads, c.past_length, c.head_dim},
                    to_dtype(PseudoRandom(static_cast<std::size_t>(c.batch * c.kv_num_heads *
                                                                   c.past_length * c.head_dim),
                                          c.seed + 4)));
    past_value =
        make_tensor({c.batch, c.kv_num_heads, c.past_length, c.v_head_dim},
                    to_dtype(PseudoRandom(static_cast<std::size_t>(c.batch * c.kv_num_heads *
                                                                   c.past_length * c.v_head_dim),
                                          c.seed + 5)));
  }

  Tensor cos_cache;
  Tensor sin_cache;
  if (c.with_rotary) {
    const std::int64_t half = c.head_dim / 2;
    const std::int64_t max_position = total_length + (c.with_position_ids ? 2 : 0);
    cos_cache = make_tensor({max_position, half}, MakeCosOrSinTable(max_position, half, true));
    sin_cache = make_tensor({max_position, half}, MakeCosOrSinTable(max_position, half, false));
  }

  Tensor position_ids;
  if (c.with_position_ids) {
    std::vector<std::int64_t> positions(static_cast<std::size_t>(c.batch * c.sequence_length));
    for (std::int64_t batch = 0; batch < c.batch; ++batch) {
      for (std::int64_t token = 0; token < c.sequence_length; ++token) {
        positions[static_cast<std::size_t>(batch * c.sequence_length + token)] =
            c.past_length + c.sequence_length - token;
      }
    }
    position_ids = Tensor::FromInt64("", {c.batch, c.sequence_length}, positions);
  }

  Tensor attention_bias;
  if (c.with_bias) {
    // Shape (1, 1, sequence_length, total_length): broadcasts across batch
    // and heads, exercising the additive-mask broadcast path.
    attention_bias =
        make_tensor({1, 1, c.sequence_length, total_length},
                    to_dtype(PseudoRandom(
                        static_cast<std::size_t>(c.sequence_length * total_length), c.seed + 6)));
  }

  const Tensor seqlens_k =
      Tensor::FromInt32("", {c.batch},
                        std::vector<std::int32_t>(static_cast<std::size_t>(c.batch),
                                                  static_cast<std::int32_t>(total_length - 1)));
  const Tensor total_sequence_length =
      Tensor::FromInt32("", {}, {static_cast<std::int32_t>(total_length)});

  NodeProto node = MakeGqaNode(c.num_heads, c.kv_num_heads, c.causal, with_cache, c.with_rotary,
                               c.with_bias, c.with_present, c.with_position_ids);
  if (c.scale.has_value()) {
    AddFloatAttribute(node, "scale", *c.scale);
  }
  if (c.softcap.has_value()) {
    AddFloatAttribute(node, "softcap", *c.softcap);
  }

  onnx_light_cpu::GroupQueryAttentionKernel optimized_kernel(node, MakeCtx());
  onnx_light_cpu::NaiveGroupQueryAttentionKernel naive_kernel(node, MakeCtx());

  const Tensor *past_key_ptr = with_cache ? &past_key : nullptr;
  const Tensor *past_value_ptr = with_cache ? &past_value : nullptr;
  const Tensor *cos_cache_ptr = c.with_rotary ? &cos_cache : nullptr;
  const Tensor *sin_cache_ptr = c.with_rotary ? &sin_cache : nullptr;
  const Tensor *position_ids_ptr = c.with_position_ids ? &position_ids : nullptr;
  const Tensor *attention_bias_ptr = c.with_bias ? &attention_bias : nullptr;

  Tensor optimized_present_key;
  Tensor optimized_present_value;
  Tensor optimized_output;
  InlineExecutor executor;
  if (c.expect_parallel) {
    onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
    onnx_light_cpu::ExecutionExecutorScope scope(&view);
    optimized_output = optimized_kernel(
        node, query, key, value, seqlens_k, total_sequence_length, past_key_ptr, past_value_ptr,
        cos_cache_ptr, sin_cache_ptr, position_ids_ptr, attention_bias_ptr,
        /*rt=*/nullptr, c.with_present ? &optimized_present_key : nullptr,
        c.with_present ? &optimized_present_value : nullptr);
  } else {
    optimized_output = optimized_kernel(
        node, query, key, value, seqlens_k, total_sequence_length, past_key_ptr, past_value_ptr,
        cos_cache_ptr, sin_cache_ptr, position_ids_ptr, attention_bias_ptr,
        /*rt=*/nullptr, c.with_present ? &optimized_present_key : nullptr,
        c.with_present ? &optimized_present_value : nullptr);
  }

  Tensor naive_present_key;
  Tensor naive_present_value;
  const Tensor naive_output = naive_kernel(
      node, query, key, value, seqlens_k, total_sequence_length, past_key_ptr, past_value_ptr,
      cos_cache_ptr, sin_cache_ptr, position_ids_ptr, attention_bias_ptr,
      /*rt=*/nullptr, c.with_present ? &naive_present_key : nullptr,
      c.with_present ? &naive_present_value : nullptr);

  SCOPED_TRACE(c.name);
  if (c.expect_parallel) {
    EXPECT_GT(executor.dispatches, 0);
  }
  ExpectTensorsNear(naive_output, optimized_output, tolerance);
  if (c.with_present) {
    ExpectTensorsNear(naive_present_key, optimized_present_key, tolerance);
    ExpectTensorsNear(naive_present_value, optimized_present_value, tolerance);
  }
}

} // namespace

TEST(OnnxLightNaiveGroupQueryAttentionKernel, MatchesOptimizedKernelAcrossMhaGqaMqaFloat) {
  const std::vector<DifferentialCase> cases = {
      {"MHA_NoCache_NoRotary_Causal", /*num_heads=*/2, /*kv_num_heads=*/2, /*batch=*/2,
       /*sequence_length=*/3, /*past_length=*/0, /*head_dim=*/8, /*v_head_dim=*/8,
       /*causal=*/true, /*with_rotary=*/false, /*with_bias=*/false, /*with_present=*/false,
       /*seed=*/101},
      {"GQA_Cache_Rotary_Causal_Present", /*num_heads=*/4, /*kv_num_heads=*/2, /*batch=*/1,
       /*sequence_length=*/2, /*past_length=*/3, /*head_dim=*/8, /*v_head_dim=*/8,
       /*causal=*/true, /*with_rotary=*/true, /*with_bias=*/false, /*with_present=*/true,
       /*seed=*/202},
      {"MQA_Cache_Rotary_NonCausal_Bias_Present", /*num_heads=*/4, /*kv_num_heads=*/1, /*batch=*/2,
       /*sequence_length=*/2, /*past_length=*/2, /*head_dim=*/8, /*v_head_dim=*/8,
       /*causal=*/false, /*with_rotary=*/true, /*with_bias=*/true, /*with_present=*/true,
       /*seed=*/303},
      {"GQA_NoCache_Bias_Causal_DifferentVHeadDim", /*num_heads=*/6, /*kv_num_heads=*/3,
       /*batch=*/1, /*sequence_length=*/4, /*past_length=*/0, /*head_dim=*/8, /*v_head_dim=*/4,
       /*causal=*/true, /*with_rotary=*/false, /*with_bias=*/true, /*with_present=*/false,
       /*seed=*/404},
      {"GQA_Cache_Rotary_PositionIds_Present", /*num_heads=*/4, /*kv_num_heads=*/2, /*batch=*/2,
       /*sequence_length=*/2, /*past_length=*/2, /*head_dim=*/8, /*v_head_dim=*/8,
       /*causal=*/true, /*with_rotary=*/true, /*with_bias=*/false, /*with_present=*/true,
       /*seed=*/405, /*with_position_ids=*/true},
      {"GQA_VectorTails_Scale_Softcap_Parallel", /*num_heads=*/4, /*kv_num_heads=*/2, /*batch=*/2,
       /*sequence_length=*/96, /*past_length=*/0, /*head_dim=*/17, /*v_head_dim=*/19,
       /*causal=*/false, /*with_rotary=*/false, /*with_bias=*/false, /*with_present=*/false,
       /*seed=*/406, /*with_position_ids=*/false, /*scale=*/0.17f, /*softcap=*/1.5f,
       /*expect_parallel=*/true},
  };
  for (const auto &c : cases) {
    RunDifferentialCase(c, DataType::FLOAT, 1e-4f);
  }
}

TEST(OnnxLightNaiveGroupQueryAttentionKernel, MatchesOptimizedKernelWithFloat16CacheAndRotary) {
  const DifferentialCase c{"GQA_Cache_Rotary_Causal_Present_FP16",
                           /*num_heads=*/4,
                           /*kv_num_heads=*/2,
                           /*batch=*/1,
                           /*sequence_length=*/2,
                           /*past_length=*/3,
                           /*head_dim=*/8,
                           /*v_head_dim=*/8,
                           /*causal=*/true,
                           /*with_rotary=*/true,
                           /*with_bias=*/false,
                           /*with_present=*/true,
                           /*seed=*/505};
  RunDifferentialCase(c, DataType::FLOAT16, 5e-3f);
}

TEST(OnnxLightNaiveGroupQueryAttentionKernel, MatchesOptimizedKernelWithBFloat16CacheNonCausal) {
  const DifferentialCase c{"MQA_Cache_NonCausal_BF16",
                           /*num_heads=*/4,
                           /*kv_num_heads=*/1,
                           /*batch=*/1,
                           /*sequence_length=*/3,
                           /*past_length=*/2,
                           /*head_dim=*/8,
                           /*v_head_dim=*/8,
                           /*causal=*/false,
                           /*with_rotary=*/false,
                           /*with_bias=*/true,
                           /*with_present=*/true,
                           /*seed=*/606};
  RunDifferentialCase(c, DataType::BFLOAT16, 3e-2f);
}

TEST(OnnxLightNaiveGroupQueryAttentionKernel, BothVariantsRejectMismatchedAttentionBiasType) {
  const Tensor query = Tensor::FromFloat("", {1, 1, 2}, {0.1f, -0.2f});
  const Tensor key = Tensor::FromFloat("", {1, 1, 2}, {0.3f, -0.4f});
  const Tensor value = Tensor::FromFloat("", {1, 1, 2}, {0.5f, -0.6f});
  const Tensor seqlens_k = Tensor::FromInt32("", {1}, {0});
  const Tensor total_sequence_length = Tensor::FromInt32("", {}, {1});
  const Tensor attention_bias = MakeHalfTensor(DataType::FLOAT16, {1, 1, 1, 1}, {0.0f});
  const NodeProto node = MakeGqaNode(1, 1, true, false, false, true, false);

  onnx_light_cpu::GroupQueryAttentionKernel optimized(node, MakeCtx());
  onnx_light_cpu::NaiveGroupQueryAttentionKernel naive(node, MakeCtx());
  EXPECT_THROW((void)optimized(node, query, key, value, seqlens_k, total_sequence_length, nullptr,
                               nullptr, nullptr, nullptr, nullptr, &attention_bias),
               std::invalid_argument);
  EXPECT_THROW((void)naive(node, query, key, value, seqlens_k, total_sequence_length, nullptr,
                           nullptr, nullptr, nullptr, nullptr, &attention_bias),
               std::invalid_argument);
}
