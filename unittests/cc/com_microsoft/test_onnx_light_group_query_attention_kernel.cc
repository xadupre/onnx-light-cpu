// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/attention/attention_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/group_query_attention_kernel.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::Tensor;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId("com.microsoft", 1)); }

void AddIntAttribute(NodeProto &node, const char *name, std::int64_t value) {
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, name, value);
}

void AddFloatAttribute(NodeProto &node, const char *name, float value) {
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, name, value);
}

// Builds a GroupQueryAttention node using the exact 12-input/3-output wiring
// of the Qwen3-8B-int4 export (query/key/value, past_key/past_value,
// seqlens_k/total_sequence_length, cos_cache/sin_cache, empty
// position_ids/attention_bias/head_sink, output/present_key/present_value),
// optionally omitting the cache/rotary/present slots for the stateless tests.
NodeProto MakeGqaNode(std::int64_t num_heads, std::int64_t kv_num_heads, bool with_cache,
                      bool with_rotary, bool with_present, std::int64_t rotary_interleaved = 0) {
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
  node.add_input(""); // position_ids
  node.add_input(""); // attention_bias
  node.add_input(""); // head_sink
  node.add_output("output");
  if (with_present) {
    node.add_output("present_key");
    node.add_output("present_value");
  }
  AddIntAttribute(node, "num_heads", num_heads);
  AddIntAttribute(node, "kv_num_heads", kv_num_heads);
  AddIntAttribute(node, "do_rotary", with_rotary ? 1 : 0);
  AddIntAttribute(node, "rotary_interleaved", rotary_interleaved);
  return node;
}

TEST(OnnxLightGroupQueryAttentionKernel, ExactModelWiringMatchesPrecomputedReference) {
  // batch=1, current sequence=2, past_sequence=3, num_heads=4, kv_num_heads=2,
  // head_size=8; expected values were computed independently in Python from
  // the split-half RoPE and bottom-right causal attention formulas and cross
  // checked against onnxruntime's com.microsoft::GroupQueryAttention CPU
  // kernel for this exact input (see the skill/session notes for the script).
  const Tensor query = Tensor::FromFloat(
      "", {1, 2, 32},
      {-0.42714751f, 0.37911853f,  -0.26119852f, -0.07775197f, -0.02260299f, -0.22226539f,
       -0.41033781f, 0.19466785f,  0.10831743f,  -0.58585894f, 0.70422292f,  0.29054907f,
       -0.22781615f, 0.27065948f,  -0.14008595f, -0.01820686f, 0.23665330f,  -0.37700045f,
       0.17275725f,  0.41969371f,  0.39668941f,  -0.08990955f, 0.27087581f,  -0.48647481f,
       -0.04745678f, 0.13484518f,  -0.40308031f, -0.02450628f, 0.51742196f,  0.78544784f,
       0.23320840f,  0.24858996f,  -0.28769648f, -0.36281648f, -0.42368761f, 0.16246405f,
       0.22558182f,  -0.19762810f, -0.36860248f, 0.07726733f,  0.09387088f,  -0.03924351f,
       0.38099495f,  -0.02788874f, -0.01984527f, -0.33246434f, 0.04078706f,  0.40412334f,
       0.01834321f,  0.02127438f,  0.13009636f,  0.08324510f,  0.15907572f,  0.16101629f,
       0.18550500f,  -0.23850524f, 0.09000929f,  -0.48081046f, 0.08003965f,  -0.37848714f,
       -0.02138124f, 0.14221492f,  -0.12445613f, 0.02931495f});
  const Tensor key = Tensor::FromFloat(
      "", {1, 2, 16},
      {-0.49212536f, -0.25717765f, 0.20648454f,  -0.34635887f, 0.19513571f,  -0.41650799f,
       -0.27221474f, -0.32862759f, 0.00214371f,  0.16030797f,  -0.31974235f, -0.05444182f,
       0.48658553f,  -0.09521759f, -0.24474449f, 0.11597370f,  -0.06709168f, -0.21050724f,
       -0.53871393f, 0.24549769f,  -0.17130987f, 0.00023566f,  -0.31909281f, 0.39051434f,
       0.22436188f,  0.29426277f,  -0.03312561f, 0.14037555f,  0.26718214f,  0.30690280f,
       0.09371502f,  -0.01857141f});
  const Tensor value = Tensor::FromFloat(
      "", {1, 2, 16},
      {-0.10784389f, -0.22459319f, -0.28964368f, 0.10801040f,  -0.07336576f, -0.59875697f,
       -0.04657428f, 0.31914926f,  -0.08255147f, -0.55600077f, -0.03730258f, 0.23549236f,
       0.06059958f,  -0.12842233f, 0.55448669f,  0.56998587f,  -0.02952751f, 0.24403363f,
       0.11774832f,  0.23443288f,  0.43598145f,  0.24605581f,  0.02631160f,  -0.19605170f,
       -0.24356607f, -0.00766145f, 0.34745535f,  0.09015626f,  0.01591699f,  0.07718146f,
       0.01072286f,  0.16417101f});
  const Tensor past_key = Tensor::FromFloat(
      "", {1, 2, 3, 8},
      {-0.33688846f, -0.59257430f, -0.12754501f, -0.34472215f, 0.48454142f,  -0.04754306f,
       -0.07586201f, -0.46144620f, 0.08462581f,  -0.18708365f, 0.33654669f,  0.25236630f,
       -0.23276883f, 0.12321493f,  -0.81672484f, -0.20199144f, 0.37386647f,  0.23706241f,
       0.05260227f,  -0.00878838f, -0.42585427f, -0.40798989f, 0.06702347f,  0.52853382f,
       -0.65126693f, 0.18854645f,  0.18035896f,  0.28522736f,  -0.26077399f, -0.15870212f,
       0.01370523f,  -0.30826554f, -0.36878678f, -0.26500756f, -0.02126804f, 0.11221600f,
       -0.00737812f, 0.02317820f,  -0.20517397f, -0.21625130f, 0.33618686f,  -0.01644425f,
       -0.02472412f, 0.28079596f,  0.37156114f,  0.38183865f,  0.12176766f,  -0.01509757f});
  const Tensor past_value = Tensor::FromFloat(
      "", {1, 2, 3, 8},
      {0.08679526f,  0.05379170f,  0.41924417f,  0.08761404f,  0.19152170f,  -0.00836631f,
       0.41131556f,  -0.61584228f, 0.11415272f,  0.22661720f,  -0.34773776f, 0.64509302f,
       -0.04508106f, -0.04834928f, -0.32383275f, 0.26338986f,  0.06734022f,  -0.17747803f,
       0.06787884f,  0.20585476f,  0.36450139f,  0.06481783f,  -0.28944707f, -0.16698234f,
       -0.68951631f, -0.21962464f, 0.22094072f,  0.13971502f,  -0.03236281f, -0.10243089f,
       0.47536013f,  0.08467236f,  0.27286392f,  0.11852147f,  -0.20081295f, 0.46661070f,
       -0.37144172f, -0.35885203f, -0.12874486f, -0.21889797f, -0.16724066f, -0.17998593f,
       0.29604816f,  0.01625840f,  0.10557223f,  -0.47639084f, -0.25408539f, 0.32537109f});
  const Tensor cos_cache = Tensor::FromFloat(
      "", {8, 4},
      {-0.36114800f, 0.35355926f,  -0.30919975f, 0.08976550f,  -0.25387198f, 0.05898609f,
       -0.26989135f, -0.07698163f, 0.50176436f,  -0.11258090f, 0.61103845f,  -0.13763802f,
       -0.35273090f, 0.02251575f,  -0.12269706f, 0.52695960f,  0.25827691f,  0.35438219f,
       0.18950105f,  0.74129915f,  0.23829076f,  0.15940586f,  -0.24881957f, -0.27279115f,
       0.05527078f,  0.29932144f,  0.33508772f,  -0.28320017f, 0.15942201f,  0.05800380f,
       -0.33547908f, 0.15355368f});
  const Tensor sin_cache = Tensor::FromFloat(
      "", {8, 4},
      {-0.68116987f, 0.07894906f,  0.74139404f,  -0.30596557f, 0.00562578f,  -0.56827927f,
       -0.22650050f, 0.22685932f,  -0.31273860f, -0.01027744f, -0.10655049f, -0.11352851f,
       0.05719461f,  0.14531890f,  0.36908033f,  0.24989119f,  -0.16948253f, 0.42440879f,
       0.37448436f,  -0.46768442f, 0.19956978f,  0.24767855f,  0.28989565f,  0.16415259f,
       -0.38915497f, -0.08029030f, -0.62106198f, -0.04736777f, 0.60868281f,  0.20447126f,
       0.25086108f,  -0.15404642f});
  const Tensor seqlens_k = Tensor::FromInt32("", {1}, {4});
  const Tensor total_sequence_length = Tensor::FromInt32("", {}, {5});

  const std::vector<float> expected_output = {
      0.04011499f,  -0.03126945f, -0.04358935f, 0.26507425f,  0.10958266f,  -0.14726005f,
      -0.07057313f, -0.04284225f, 0.03882301f,  -0.03573211f, -0.03132916f, 0.25424054f,
      0.11318220f,  -0.14877345f, -0.05795662f, -0.05642154f, -0.17373003f, -0.20582926f,
      0.07062818f,  0.21520983f,  -0.06191964f, -0.26419181f, 0.16492207f,  0.18360150f,
      -0.16291133f, -0.20999458f, 0.06578688f,  0.21747601f,  -0.06145441f, -0.26491684f,
      0.16516756f,  0.18983750f,  0.02755510f,  0.03034053f,  -0.00752039f, 0.26124233f,
      0.17477928f,  -0.06389167f, -0.04567275f, -0.07897687f, 0.02505292f,  0.01972749f,
      -0.00815785f, 0.25487310f,  0.18327484f,  -0.06359573f, -0.05554926f, -0.07603817f,
      -0.18106356f, -0.16718511f, 0.12602295f,  0.18929057f,  -0.04455392f, -0.19862808f,
      0.12827682f,  0.18395641f,  -0.17697665f, -0.17044286f, 0.12171482f,  0.19194539f,
      -0.04540814f, -0.19714347f, 0.13347986f,  0.18642747f};
  const std::vector<float> expected_present_key = {
      -0.33688846f, -0.59257430f, -0.12754501f, -0.34472215f, 0.48454142f,  -0.04754306f,
      -0.07586201f, -0.46144620f, 0.08462581f,  -0.18708365f, 0.33654669f,  0.25236630f,
      -0.23276883f, 0.12321493f,  -0.81672484f, -0.20199144f, 0.37386647f,  0.23706241f,
      0.05260227f,  -0.00878838f, -0.42585427f, -0.40798989f, 0.06702347f,  0.52853382f,
      0.16242711f,  0.05473594f,  0.07513405f,  -0.10039598f, -0.09697731f, -0.04675076f,
      0.10960934f,  -0.25972548f, -0.04636227f, -0.07470004f, 0.01740841f,  0.36462468f,
      -0.03287452f, -0.08925761f, -0.26220834f, 0.17467251f,  -0.65126693f, 0.18854645f,
      0.18035896f,  0.28522736f,  -0.26077399f, -0.15870212f, 0.01370523f,  -0.30826554f,
      -0.36878678f, -0.26500756f, -0.02126804f, 0.11221600f,  -0.00737812f, 0.02317820f,
      -0.20517397f, -0.21625130f, 0.33618686f,  -0.01644425f, -0.02472412f, 0.28079596f,
      0.37156114f,  0.38183865f,  0.12176766f,  -0.01509757f, -0.02858622f, 0.01744637f,
      0.12956183f,  -0.05766945f, -0.17151114f, 0.02115188f,  -0.08798119f, 0.04750893f,
      0.10323020f,  -0.02597076f, -0.04137215f, 0.09537472f,  0.03098156f,  0.23364860f,
      0.00535407f,  -0.07941843f};
  const std::vector<float> expected_present_value = {
      0.08679526f,  0.05379170f,  0.41924417f,  0.08761404f,  0.19152170f,  -0.00836631f,
      0.41131556f,  -0.61584228f, 0.11415272f,  0.22661720f,  -0.34773776f, 0.64509302f,
      -0.04508106f, -0.04834928f, -0.32383275f, 0.26338986f,  0.06734022f,  -0.17747803f,
      0.06787884f,  0.20585476f,  0.36450139f,  0.06481783f,  -0.28944707f, -0.16698234f,
      -0.10784389f, -0.22459319f, -0.28964368f, 0.10801040f,  -0.07336576f, -0.59875697f,
      -0.04657428f, 0.31914926f,  -0.02952751f, 0.24403363f,  0.11774832f,  0.23443288f,
      0.43598145f,  0.24605581f,  0.02631160f,  -0.19605170f, -0.68951631f, -0.21962464f,
      0.22094072f,  0.13971502f,  -0.03236281f, -0.10243089f, 0.47536013f,  0.08467236f,
      0.27286392f,  0.11852147f,  -0.20081295f, 0.46661070f,  -0.37144172f, -0.35885203f,
      -0.12874486f, -0.21889797f, -0.16724066f, -0.17998593f, 0.29604816f,  0.01625840f,
      0.10557223f,  -0.47639084f, -0.25408539f, 0.32537109f,  -0.08255147f, -0.55600077f,
      -0.03730258f, 0.23549236f,  0.06059958f,  -0.12842233f, 0.55448669f,  0.56998587f,
      -0.24356607f, -0.00766145f, 0.34745535f,  0.09015626f,  0.01591699f,  0.07718146f,
      0.01072286f,  0.16417101f};

  const NodeProto node = MakeGqaNode(4, 2, /*with_cache=*/true, /*with_rotary=*/true,
                                     /*with_present=*/true);
  onnx_light_cpu::GroupQueryAttentionKernel kernel(node, MakeCtx());
  Tensor present_key;
  Tensor present_value;
  const Tensor output = kernel(node, query, key, value, seqlens_k, total_sequence_length, &past_key,
                               &past_value, &cos_cache, &sin_cache,
                               /*position_ids=*/nullptr, /*attention_bias=*/nullptr,
                               /*rt=*/nullptr, &present_key, &present_value);

  ASSERT_EQ(output.shape.size(), 3u);
  EXPECT_EQ(output.shape[0], 1);
  EXPECT_EQ(output.shape[1], 2);
  EXPECT_EQ(output.shape[2], 32);
  ASSERT_EQ(output.element_count(), static_cast<std::int64_t>(expected_output.size()));
  const float *actual_output = output.AsFloat();
  for (std::size_t i = 0; i < expected_output.size(); ++i) {
    EXPECT_NEAR(actual_output[i], expected_output[i], 1e-4f) << i;
  }

  ASSERT_EQ(present_key.shape.size(), 4u);
  EXPECT_EQ(present_key.shape[0], 1);
  EXPECT_EQ(present_key.shape[1], 2);
  EXPECT_EQ(present_key.shape[2], 5);
  EXPECT_EQ(present_key.shape[3], 8);
  ASSERT_EQ(present_key.element_count(), static_cast<std::int64_t>(expected_present_key.size()));
  const float *actual_present_key = present_key.AsFloat();
  for (std::size_t i = 0; i < expected_present_key.size(); ++i) {
    EXPECT_NEAR(actual_present_key[i], expected_present_key[i], 1e-4f) << i;
  }

  ASSERT_EQ(present_value.shape, present_key.shape);
  ASSERT_EQ(present_value.element_count(),
            static_cast<std::int64_t>(expected_present_value.size()));
  const float *actual_present_value = present_value.AsFloat();
  for (std::size_t i = 0; i < expected_present_value.size(); ++i) {
    EXPECT_NEAR(actual_present_value[i], expected_present_value[i], 1e-4f) << i;
  }
}

TEST(OnnxLightGroupQueryAttentionKernel, StatelessPathMatchesDelegatedAttentionKernel) {
  const Tensor query = Tensor::FromFloat("", {2, 3, 32}, std::vector<float>(2 * 3 * 32, 0.0f));
  std::vector<float> query_values(2 * 3 * 32);
  std::vector<float> key_values(2 * 3 * 16);
  std::vector<float> value_values(2 * 3 * 16);
  for (std::size_t i = 0; i < query_values.size(); ++i) {
    query_values[i] = 0.01f * static_cast<float>(i % 23) - 0.1f;
  }
  for (std::size_t i = 0; i < key_values.size(); ++i) {
    key_values[i] = 0.02f * static_cast<float>(i % 17) - 0.15f;
    value_values[i] = 0.03f * static_cast<float>(i % 11) - 0.05f;
  }
  const Tensor q = Tensor::FromFloat("", {2, 3, 32}, query_values);
  const Tensor k = Tensor::FromFloat("", {2, 3, 16}, key_values);
  const Tensor v = Tensor::FromFloat("", {2, 3, 16}, value_values);
  const Tensor seqlens_k = Tensor::FromInt32("", {2}, {2, 2});
  const Tensor total_sequence_length = Tensor::FromInt32("", {}, {3});

  const NodeProto node = MakeGqaNode(4, 2, /*with_cache=*/false, /*with_rotary=*/false,
                                     /*with_present=*/false);
  onnx_light_cpu::GroupQueryAttentionKernel kernel(node, MakeCtx());
  const Tensor output = kernel(node, q, k, v, seqlens_k, total_sequence_length);

  NodeProto attention_node;
  attention_node.set_op_type("Attention");
  attention_node.add_input("query");
  attention_node.add_input("key");
  attention_node.add_input("value");
  attention_node.add_output("output");
  AddIntAttribute(attention_node, "q_num_heads", 4);
  AddIntAttribute(attention_node, "kv_num_heads", 2);
  AddIntAttribute(attention_node, "is_causal", 1);
  onnx_light_cpu::AttentionKernel attention(rt_ns::KernelContext(rt_ns::DefaultOpset(23)));
  const Tensor expected = attention(attention_node, q, k, v, nullptr);

  ASSERT_EQ(output.element_count(), expected.element_count());
  const float *actual_data = output.AsFloat();
  const float *expected_data = expected.AsFloat();
  for (std::int64_t i = 0; i < output.element_count(); ++i) {
    EXPECT_NEAR(actual_data[i], expected_data[i], 1e-5f) << i;
  }
}

TEST(OnnxLightGroupQueryAttentionKernel, RejectsPastKeyWithoutPastValue) {
  NodeProto node = MakeGqaNode(4, 2, /*with_cache=*/true, /*with_rotary=*/false,
                               /*with_present=*/false);
  node.ref_input()[4] = ""; // Drop past_value while keeping past_key wired.
  onnx_light_cpu::GroupQueryAttentionKernel kernel(node, MakeCtx());
  const Tensor query = Tensor::FromFloat("", {1, 1, 32}, std::vector<float>(32, 0.1f));
  const Tensor key = Tensor::FromFloat("", {1, 1, 16}, std::vector<float>(16, 0.1f));
  const Tensor value = Tensor::FromFloat("", {1, 1, 16}, std::vector<float>(16, 0.1f));
  const Tensor past_key = Tensor::FromFloat("", {1, 2, 3, 8}, std::vector<float>(48, 0.1f));
  const Tensor seqlens_k = Tensor::FromInt32("", {1}, {3});
  const Tensor total_sequence_length = Tensor::FromInt32("", {}, {4});
  EXPECT_THROW((void)kernel(node, query, key, value, seqlens_k, total_sequence_length, &past_key),
               std::invalid_argument);
}

TEST(OnnxLightGroupQueryAttentionKernel, RejectsPastKeyRankMismatch) {
  const NodeProto node = MakeGqaNode(4, 2, /*with_cache=*/true, /*with_rotary=*/false,
                                     /*with_present=*/false);
  onnx_light_cpu::GroupQueryAttentionKernel kernel(node, MakeCtx());
  const Tensor query = Tensor::FromFloat("", {1, 1, 32}, std::vector<float>(32, 0.1f));
  const Tensor key = Tensor::FromFloat("", {1, 1, 16}, std::vector<float>(16, 0.1f));
  const Tensor value = Tensor::FromFloat("", {1, 1, 16}, std::vector<float>(16, 0.1f));
  // past_key must be rank 4; this is rank 3.
  const Tensor past_key = Tensor::FromFloat("", {2, 3, 8}, std::vector<float>(48, 0.1f));
  const Tensor past_value = Tensor::FromFloat("", {1, 2, 3, 8}, std::vector<float>(48, 0.1f));
  const Tensor seqlens_k = Tensor::FromInt32("", {1}, {3});
  const Tensor total_sequence_length = Tensor::FromInt32("", {}, {4});
  EXPECT_THROW((void)kernel(node, query, key, value, seqlens_k, total_sequence_length, &past_key,
                            &past_value),
               std::invalid_argument);
}

TEST(OnnxLightGroupQueryAttentionKernel, RejectsPresentKeyWithoutPresentValue) {
  NodeProto node = MakeGqaNode(4, 2, /*with_cache=*/false, /*with_rotary=*/false,
                               /*with_present=*/false);
  node.add_output("present_key");
  node.add_output(""); // present_value intentionally left unwired.
  onnx_light_cpu::GroupQueryAttentionKernel kernel(node, MakeCtx());
  const Tensor query = Tensor::FromFloat("", {1, 1, 32}, std::vector<float>(32, 0.1f));
  const Tensor key = Tensor::FromFloat("", {1, 1, 16}, std::vector<float>(16, 0.1f));
  const Tensor value = Tensor::FromFloat("", {1, 1, 16}, std::vector<float>(16, 0.1f));
  const Tensor seqlens_k = Tensor::FromInt32("", {1}, {0});
  const Tensor total_sequence_length = Tensor::FromInt32("", {}, {1});
  EXPECT_THROW((void)kernel(node, query, key, value, seqlens_k, total_sequence_length),
               std::invalid_argument);
}

TEST(OnnxLightGroupQueryAttentionKernel, RejectsRotaryWithoutCosSinCache) {
  NodeProto node = MakeGqaNode(4, 2, /*with_cache=*/false, /*with_rotary=*/false,
                               /*with_present=*/false);
  // do_rotary=1 is set without wiring cos_cache/sin_cache: overwrite the
  // attribute MakeGqaNode already added rather than duplicating it (the first
  // matching attribute wins and duplicates would silently mask this test).
  ASSERT_EQ(node.attribute(2).name(), "do_rotary");
  node.mutable_attribute(2)->ref_i() = 1;
  onnx_light_cpu::GroupQueryAttentionKernel kernel(node, MakeCtx());
  const Tensor query = Tensor::FromFloat("", {1, 1, 32}, std::vector<float>(32, 0.1f));
  const Tensor key = Tensor::FromFloat("", {1, 1, 16}, std::vector<float>(16, 0.1f));
  const Tensor value = Tensor::FromFloat("", {1, 1, 16}, std::vector<float>(16, 0.1f));
  const Tensor seqlens_k = Tensor::FromInt32("", {1}, {0});
  const Tensor total_sequence_length = Tensor::FromInt32("", {}, {1});
  EXPECT_THROW((void)kernel(node, query, key, value, seqlens_k, total_sequence_length),
               std::invalid_argument);
}

TEST(OnnxLightGroupQueryAttentionKernel, RejectsInterleavedRotary) {
  const NodeProto node = MakeGqaNode(4, 2, /*with_cache=*/false, /*with_rotary=*/true,
                                     /*with_present=*/false, /*rotary_interleaved=*/1);
  onnx_light_cpu::GroupQueryAttentionKernel kernel(node, MakeCtx());
  const Tensor query = Tensor::FromFloat("", {1, 1, 32}, std::vector<float>(32, 0.1f));
  const Tensor key = Tensor::FromFloat("", {1, 1, 16}, std::vector<float>(16, 0.1f));
  const Tensor value = Tensor::FromFloat("", {1, 1, 16}, std::vector<float>(16, 0.1f));
  const Tensor cos_cache = Tensor::FromFloat("", {8, 4}, std::vector<float>(32, 0.1f));
  const Tensor sin_cache = Tensor::FromFloat("", {8, 4}, std::vector<float>(32, 0.1f));
  const Tensor seqlens_k = Tensor::FromInt32("", {1}, {0});
  const Tensor total_sequence_length = Tensor::FromInt32("", {}, {1});
  EXPECT_THROW((void)kernel(node, query, key, value, seqlens_k, total_sequence_length, nullptr,
                            nullptr, &cos_cache, &sin_cache),
               std::invalid_argument);
}

TEST(OnnxLightGroupQueryAttentionKernel, RejectsCosCacheHalfHeadDimMismatch) {
  const NodeProto node = MakeGqaNode(4, 2, /*with_cache=*/false, /*with_rotary=*/true,
                                     /*with_present=*/false);
  onnx_light_cpu::GroupQueryAttentionKernel kernel(node, MakeCtx());
  const Tensor query = Tensor::FromFloat("", {1, 1, 32}, std::vector<float>(32, 0.1f));
  const Tensor key = Tensor::FromFloat("", {1, 1, 16}, std::vector<float>(16, 0.1f));
  const Tensor value = Tensor::FromFloat("", {1, 1, 16}, std::vector<float>(16, 0.1f));
  // head_size is 8, so cos_cache/sin_cache must have 4 columns, not 3.
  const Tensor cos_cache = Tensor::FromFloat("", {8, 3}, std::vector<float>(24, 0.1f));
  const Tensor sin_cache = Tensor::FromFloat("", {8, 3}, std::vector<float>(24, 0.1f));
  const Tensor seqlens_k = Tensor::FromInt32("", {1}, {0});
  const Tensor total_sequence_length = Tensor::FromInt32("", {}, {1});
  EXPECT_THROW((void)kernel(node, query, key, value, seqlens_k, total_sequence_length, nullptr,
                            nullptr, &cos_cache, &sin_cache),
               std::invalid_argument);
}

TEST(OnnxLightGroupQueryAttentionKernel, RejectsRotaryPositionOutOfCacheBounds) {
  const NodeProto node = MakeGqaNode(4, 2, /*with_cache=*/true, /*with_rotary=*/true,
                                     /*with_present=*/false);
  onnx_light_cpu::GroupQueryAttentionKernel kernel(node, MakeCtx());
  const Tensor query = Tensor::FromFloat("", {1, 1, 32}, std::vector<float>(32, 0.1f));
  const Tensor key = Tensor::FromFloat("", {1, 1, 16}, std::vector<float>(16, 0.1f));
  const Tensor value = Tensor::FromFloat("", {1, 1, 16}, std::vector<float>(16, 0.1f));
  // Past length 10 puts the current token's rotary position (10) at or past
  // a cos_cache/sin_cache with only 8 rows (max_sequence_length=8).
  const Tensor past_key = Tensor::FromFloat("", {1, 2, 10, 8}, std::vector<float>(160, 0.1f));
  const Tensor past_value = Tensor::FromFloat("", {1, 2, 10, 8}, std::vector<float>(160, 0.1f));
  const Tensor cos_cache = Tensor::FromFloat("", {8, 4}, std::vector<float>(32, 0.1f));
  const Tensor sin_cache = Tensor::FromFloat("", {8, 4}, std::vector<float>(32, 0.1f));
  const Tensor seqlens_k = Tensor::FromInt32("", {1}, {10});
  const Tensor total_sequence_length = Tensor::FromInt32("", {}, {11});
  EXPECT_THROW((void)kernel(node, query, key, value, seqlens_k, total_sequence_length, &past_key,
                            &past_value, &cos_cache, &sin_cache),
               std::invalid_argument);
}

TEST(OnnxLightGroupQueryAttentionKernel, RejectsNonUniformSeqlensK) {
  const NodeProto node = MakeGqaNode(4, 2, /*with_cache=*/false, /*with_rotary=*/false,
                                     /*with_present=*/false);
  onnx_light_cpu::GroupQueryAttentionKernel kernel(node, MakeCtx());
  const Tensor query = Tensor::FromFloat("", {2, 3, 32}, std::vector<float>(2 * 3 * 32, 0.1f));
  const Tensor key = Tensor::FromFloat("", {2, 3, 16}, std::vector<float>(2 * 3 * 16, 0.1f));
  const Tensor value = Tensor::FromFloat("", {2, 3, 16}, std::vector<float>(2 * 3 * 16, 0.1f));
  // Every batch entry must equal total_sequence_length - 1 == 2; the second
  // batch is inconsistent.
  const Tensor seqlens_k = Tensor::FromInt32("", {2}, {2, 1});
  const Tensor total_sequence_length = Tensor::FromInt32("", {}, {3});
  EXPECT_THROW((void)kernel(node, query, key, value, seqlens_k, total_sequence_length),
               std::invalid_argument);
}

TEST(OnnxLightGroupQueryAttentionKernel, RegisteredKernelRunsThroughRuntimeContext) {
  onnx_light_cpu::RegisterAllKernels();
  const NodeProto node = MakeGqaNode(2, 1, /*with_cache=*/false, /*with_rotary=*/false,
                                     /*with_present=*/false);
  rt_ns::RuntimeContext rt(MakeCtx());
  rt.tensors()["query"] = Tensor::FromFloat("query", {1, 2, 8}, std::vector<float>(16, 0.1f));
  rt.tensors()["key"] = Tensor::FromFloat("key", {1, 2, 4}, std::vector<float>(8, 0.1f));
  rt.tensors()["value"] = Tensor::FromFloat("value", {1, 2, 4}, std::vector<float>(8, 0.1f));
  rt.tensors()["seqlens_k"] = Tensor::FromInt32("seqlens_k", {1}, {1});
  rt.tensors()["total_sequence_length"] = Tensor::FromInt32("total_sequence_length", {}, {2});
  onnx_light_cpu::GroupQueryAttentionKernel kernel(node, MakeCtx());
  kernel.Run(rt);
  ASSERT_TRUE(rt.tensors().count("output"));
  const Tensor &output = rt.tensors().at("output");
  ASSERT_EQ(output.shape.size(), 3u);
  EXPECT_EQ(output.shape[0], 1);
  EXPECT_EQ(output.shape[1], 2);
  EXPECT_EQ(output.shape[2], 8);
}

} // namespace
