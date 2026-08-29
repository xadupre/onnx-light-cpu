// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/com_microsoft/include_com_microsoft_cases.h"
#include "onnx_light_cpu/backend_test/cases/math/benchmark_helpers.h"

#include "onnx_light_cpu/kernels/attention/attention_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/group_query_attention_kernel.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/kernel_context.h"

#include <cstdint>
#include <optional>
#include <string>
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

NodeProto MakeGroupQueryAttentionNode(std::int64_t num_heads, std::int64_t kv_num_heads,
                                      std::optional<std::int64_t> causal,
                                      std::optional<float> scale, std::optional<float> softcap) {
  NodeProto node;
  node.set_op_type("GroupQueryAttention");
  node.set_domain(kMicrosoftDomain);
  node.add_input("query");
  node.add_input("key");
  node.add_input("value");
  node.add_input("");
  node.add_input("");
  node.add_input("seqlens_k");
  node.add_input("total_sequence_length");
  node.add_output("output");
  AddIntAttribute(node, "num_heads", num_heads);
  AddIntAttribute(node, "kv_num_heads", kv_num_heads);
  if (causal.has_value()) {
    AddIntAttribute(node, "causal", *causal);
  }
  if (scale.has_value()) {
    AddFloatAttribute(node, "scale", *scale);
  }
  if (softcap.has_value()) {
    AddFloatAttribute(node, "softcap", *softcap);
  }
  return node;
}

NodeProto MakeAttentionNode(std::int64_t num_heads, std::int64_t kv_num_heads, bool causal,
                            std::optional<float> scale, std::optional<float> softcap) {
  NodeProto node;
  node.set_op_type("Attention");
  node.add_input("query");
  node.add_input("key");
  node.add_input("value");
  node.add_output("output");
  AddIntAttribute(node, "q_num_heads", num_heads);
  AddIntAttribute(node, "kv_num_heads", kv_num_heads);
  AddIntAttribute(node, "is_causal", causal ? 1 : 0);
  if (scale.has_value()) {
    AddFloatAttribute(node, "scale", *scale);
  }
  if (softcap.has_value()) {
    AddFloatAttribute(node, "softcap", *softcap);
  }
  return node;
}

// Registers a generic (non-cached, non-rotary) MHA/MQA/GQA case. The expected
// output is computed by directly invoking `AttentionKernel` (independently
// tested elsewhere), which `GroupQueryAttentionKernel` itself delegates to for
// this exact stateless path, so this is a positive same-adapter check of that
// delegation rather than an approximation of a cached/rotary contract.
void RegisterGroupQueryAttentionCase(
    std::vector<TestCase> &registry, const OpsetId &microsoft_opset, const std::string &name,
    std::int64_t batch, std::int64_t sequence, std::int64_t num_heads, std::int64_t kv_num_heads,
    std::int64_t head_dim, DataType data_type, std::optional<std::int64_t> causal = std::nullopt,
    std::optional<float> scale = std::nullopt, std::optional<float> softcap = std::nullopt,
    std::int64_t standard_opset_version = 23) {
  const std::int64_t query_width = num_heads * head_dim;
  const std::int64_t kv_width = kv_num_heads * head_dim;
  const std::int64_t query_count = batch * sequence * query_width;
  const std::int64_t kv_count = batch * sequence * kv_width;
  const NodeProto node =
      MakeGroupQueryAttentionNode(num_heads, kv_num_heads, causal, scale, softcap);

  Expect(registry, node, name, {DefaultOpset(standard_opset_version), microsoft_opset},
         {query_count, kv_count, kv_count, batch, 1}, {query_count}, [=]() -> IoData {
           Tensor query =
               MakeBenchmarkTensor(data_type, {batch, sequence, query_width}, 987654321ULL);
           Tensor key = MakeBenchmarkTensor(data_type, {batch, sequence, kv_width}, 246813579ULL);
           Tensor value = MakeBenchmarkTensor(data_type, {batch, sequence, kv_width}, 135792468ULL);
           Tensor seqlens_k = Tensor::FromInt32(
               "", {batch},
               std::vector<std::int32_t>(static_cast<std::size_t>(batch),
                                         static_cast<std::int32_t>(sequence - 1)));
           Tensor total_sequence_length =
               Tensor::FromInt32("", {}, {static_cast<std::int32_t>(sequence)});
           const AttentionKernel attention{KernelContext{DefaultOpset(standard_opset_version)}};
           Tensor output = attention(
               MakeAttentionNode(num_heads, kv_num_heads, causal.value_or(1) != 0, scale, softcap),
               query, key, value, nullptr);
           return IoData{{std::move(query), std::move(key), std::move(value), std::move(seqlens_k),
                          std::move(total_sequence_length)},
                         {std::move(output)}};
         });
}

// The Qwen3-8B-int4 model's 36 GroupQueryAttention nodes all share this exact
// contract: FLOAT activations, 32 query heads, 8 KV heads, head_size 128,
// this explicit scale, softcap disabled, do_rotary=1 with rotary_interleaved=0
// (split-half RoPE), a tensor past/present KV cache, and empty
// position_ids/attention_bias/head_sink. `sequence` is the number of new
// (query) tokens for this step and `past_length` the KV cache length carried
// in from earlier steps (0 for the prefill step, non-zero for token
// generation); `total_sequence_length = past_length + sequence` always.
NodeProto MakeQwen3GroupQueryAttentionNode() {
  NodeProto node;
  node.set_op_type("GroupQueryAttention");
  node.set_domain(kMicrosoftDomain);
  node.add_input("query");
  node.add_input("key");
  node.add_input("value");
  node.add_input("past_key");
  node.add_input("past_value");
  node.add_input("seqlens_k");
  node.add_input("total_sequence_length");
  node.add_input("cos_cache");
  node.add_input("sin_cache");
  node.add_input(""); // position_ids
  node.add_input(""); // attention_bias
  node.add_input(""); // head_sink
  node.add_output("output");
  node.add_output("present_key");
  node.add_output("present_value");
  AddIntAttribute(node, "num_heads", 32);
  AddIntAttribute(node, "kv_num_heads", 8);
  AddFloatAttribute(node, "scale", 0.0883883461356163f);
  AddFloatAttribute(node, "softcap", 0.0f);
  AddIntAttribute(node, "do_rotary", 1);
  AddIntAttribute(node, "rotary_interleaved", 0);
  return node;
}

// Registers one Qwen3-8B-int4 GroupQueryAttention BENCHMARK case using the
// full 9-wired-input/3-output model contract (rather than a stateless
// approximation): a tensor past/present KV cache and split-half RoPE applied
// to Q/K at the position derived from `seqlens_k`, exactly as the exported
// model uses it. The expected output is produced by the
// `GroupQueryAttentionKernel` adapter under test itself, which is acceptable
// here (per repository policy) only for BENCHMARK-mode timing baselines: the
// GQA/ORT parity test in `unittests/python/test_kernels_e2e.py`
// (`test_group_query_attention_benchmarks_match_onnx_runtime`) independently
// compares this exact case against onnxruntime before any timing is trusted.
void RegisterQwen3GroupQueryAttentionBenchmarkCase(std::vector<TestCase> &registry,
                                                   const OpsetId &microsoft_opset,
                                                   const std::string &name, std::int64_t sequence,
                                                   std::int64_t past_length) {
  constexpr std::int64_t kBatch = 1;
  constexpr std::int64_t kNumHeads = 32;
  constexpr std::int64_t kKvNumHeads = 8;
  constexpr std::int64_t kHeadDim = 128;
  constexpr std::int64_t kHalfHeadDim = kHeadDim / 2;
  constexpr std::int64_t kMaxPosition = 40960;
  const std::int64_t query_width = kNumHeads * kHeadDim;
  const std::int64_t kv_width = kKvNumHeads * kHeadDim;
  const std::int64_t total_length = past_length + sequence;
  const NodeProto node = MakeQwen3GroupQueryAttentionNode();

  Expect(
      registry, node, name, {DefaultOpset(27), microsoft_opset},
      {kBatch * sequence * query_width, kBatch * sequence * kv_width, kBatch * sequence * kv_width,
       kBatch * kKvNumHeads * past_length * kHeadDim, kBatch * kKvNumHeads * past_length * kHeadDim,
       kBatch, 1, kMaxPosition * kHalfHeadDim, kMaxPosition * kHalfHeadDim},
      {kBatch * sequence * query_width, kBatch * kKvNumHeads * total_length * kHeadDim,
       kBatch * kKvNumHeads * total_length * kHeadDim},
      [=](bool generate_expected_outputs) -> IoData {
        Tensor query =
            MakeBenchmarkTensor(DataType::FLOAT, {kBatch, sequence, query_width}, 987654321ULL);
        Tensor key =
            MakeBenchmarkTensor(DataType::FLOAT, {kBatch, sequence, kv_width}, 246813579ULL);
        Tensor value =
            MakeBenchmarkTensor(DataType::FLOAT, {kBatch, sequence, kv_width}, 135792468ULL);
        Tensor past_key = MakeBenchmarkTensor(
            DataType::FLOAT, {kBatch, kKvNumHeads, past_length, kHeadDim}, 111111111ULL);
        Tensor past_value = MakeBenchmarkTensor(
            DataType::FLOAT, {kBatch, kKvNumHeads, past_length, kHeadDim}, 222222222ULL);
        Tensor seqlens_k =
            Tensor::FromInt32("", {kBatch}, {static_cast<std::int32_t>(total_length - 1)});
        Tensor total_sequence_length =
            Tensor::FromInt32("", {}, {static_cast<std::int32_t>(total_length)});
        Tensor cos_cache =
            MakeBenchmarkTensor(DataType::FLOAT, {kMaxPosition, kHalfHeadDim}, 333333333ULL);
        Tensor sin_cache =
            MakeBenchmarkTensor(DataType::FLOAT, {kMaxPosition, kHalfHeadDim}, 444444444ULL);

        if (!generate_expected_outputs) {
          return IoData{{std::move(query), std::move(key), std::move(value), std::move(past_key),
                         std::move(past_value), std::move(seqlens_k),
                         std::move(total_sequence_length), std::move(cos_cache),
                         std::move(sin_cache)},
                        {},
                        {},
                        false};
        }
        const GroupQueryAttentionKernel kernel{node, KernelContext{OpsetId(kMicrosoftDomain, 1)}};
        Tensor present_key;
        Tensor present_value;
        Tensor output =
            kernel(node, query, key, value, seqlens_k, total_sequence_length, &past_key,
                   &past_value, &cos_cache, &sin_cache, /*position_ids=*/nullptr,
                   /*attention_bias=*/nullptr, /*rt=*/nullptr, &present_key, &present_value);
        return IoData{{std::move(query), std::move(key), std::move(value), std::move(past_key),
                       std::move(past_value), std::move(seqlens_k),
                       std::move(total_sequence_length), std::move(cos_cache),
                       std::move(sin_cache)},
                      {std::move(output), std::move(present_key), std::move(present_value)}};
      },
      "backend-test", "",
      {bt_ns::TensorTypeSpec(static_cast<std::int32_t>(DataType::FLOAT),
                             {kBatch, sequence, query_width}),
       bt_ns::TensorTypeSpec(static_cast<std::int32_t>(DataType::FLOAT),
                             {kBatch, kKvNumHeads, total_length, kHeadDim}),
       bt_ns::TensorTypeSpec(static_cast<std::int32_t>(DataType::FLOAT),
                             {kBatch, kKvNumHeads, total_length, kHeadDim})});
}

// Registers a positive correctness case for the exact cached/rotary contract.
// Q/K/V/cache/output values were computed independently in Python from the
// split-half RoPE and bottom-right causal attention formulas (not via this
// adapter) and cross checked against onnxruntime's
// com.microsoft::GroupQueryAttention CPU kernel for this exact input; see
// `test_onnx_light_group_query_attention_kernel.cc`'s
// `ExactModelWiringMatchesPrecomputedReference` for the identical inputs and a
// from-first-principles derivation of these numbers.
void RegisterCachedRotaryCorrectnessCase(std::vector<TestCase> &registry,
                                         const OpsetId &microsoft_opset) {
  NodeProto node;
  node.set_op_type("GroupQueryAttention");
  node.set_domain(kMicrosoftDomain);
  node.add_input("query");
  node.add_input("key");
  node.add_input("value");
  node.add_input("past_key");
  node.add_input("past_value");
  node.add_input("seqlens_k");
  node.add_input("total_sequence_length");
  node.add_input("cos_cache");
  node.add_input("sin_cache");
  node.add_output("output");
  node.add_output("present_key");
  node.add_output("present_value");
  AddIntAttribute(node, "num_heads", 4);
  AddIntAttribute(node, "kv_num_heads", 2);
  AddIntAttribute(node, "do_rotary", 1);
  AddIntAttribute(node, "rotary_interleaved", 0);

  Expect(
      registry, node, "test_cpu_group_query_attention_cached_rotary_float32",
      {DefaultOpset(27), microsoft_opset}, [=]() -> IoData {
        Tensor query = Tensor::FromFloat(
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
        Tensor key = Tensor::FromFloat(
            "", {1, 2, 16},
            {-0.49212536f, -0.25717765f, 0.20648454f,  -0.34635887f, 0.19513571f,  -0.41650799f,
             -0.27221474f, -0.32862759f, 0.00214371f,  0.16030797f,  -0.31974235f, -0.05444182f,
             0.48658553f,  -0.09521759f, -0.24474449f, 0.11597370f,  -0.06709168f, -0.21050724f,
             -0.53871393f, 0.24549769f,  -0.17130987f, 0.00023566f,  -0.31909281f, 0.39051434f,
             0.22436188f,  0.29426277f,  -0.03312561f, 0.14037555f,  0.26718214f,  0.30690280f,
             0.09371502f,  -0.01857141f});
        Tensor value = Tensor::FromFloat(
            "", {1, 2, 16},
            {-0.10784389f, -0.22459319f, -0.28964368f, 0.10801040f,  -0.07336576f, -0.59875697f,
             -0.04657428f, 0.31914926f,  -0.08255147f, -0.55600077f, -0.03730258f, 0.23549236f,
             0.06059958f,  -0.12842233f, 0.55448669f,  0.56998587f,  -0.02952751f, 0.24403363f,
             0.11774832f,  0.23443288f,  0.43598145f,  0.24605581f,  0.02631160f,  -0.19605170f,
             -0.24356607f, -0.00766145f, 0.34745535f,  0.09015626f,  0.01591699f,  0.07718146f,
             0.01072286f,  0.16417101f});
        Tensor past_key = Tensor::FromFloat(
            "", {1, 2, 3, 8},
            {-0.33688846f, -0.59257430f, -0.12754501f, -0.34472215f, 0.48454142f,  -0.04754306f,
             -0.07586201f, -0.46144620f, 0.08462581f,  -0.18708365f, 0.33654669f,  0.25236630f,
             -0.23276883f, 0.12321493f,  -0.81672484f, -0.20199144f, 0.37386647f,  0.23706241f,
             0.05260227f,  -0.00878838f, -0.42585427f, -0.40798989f, 0.06702347f,  0.52853382f,
             -0.65126693f, 0.18854645f,  0.18035896f,  0.28522736f,  -0.26077399f, -0.15870212f,
             0.01370523f,  -0.30826554f, -0.36878678f, -0.26500756f, -0.02126804f, 0.11221600f,
             -0.00737812f, 0.02317820f,  -0.20517397f, -0.21625130f, 0.33618686f,  -0.01644425f,
             -0.02472412f, 0.28079596f,  0.37156114f,  0.38183865f,  0.12176766f,  -0.01509757f});
        Tensor past_value = Tensor::FromFloat(
            "", {1, 2, 3, 8},
            {0.08679526f,  0.05379170f,  0.41924417f,  0.08761404f,  0.19152170f,  -0.00836631f,
             0.41131556f,  -0.61584228f, 0.11415272f,  0.22661720f,  -0.34773776f, 0.64509302f,
             -0.04508106f, -0.04834928f, -0.32383275f, 0.26338986f,  0.06734022f,  -0.17747803f,
             0.06787884f,  0.20585476f,  0.36450139f,  0.06481783f,  -0.28944707f, -0.16698234f,
             -0.68951631f, -0.21962464f, 0.22094072f,  0.13971502f,  -0.03236281f, -0.10243089f,
             0.47536013f,  0.08467236f,  0.27286392f,  0.11852147f,  -0.20081295f, 0.46661070f,
             -0.37144172f, -0.35885203f, -0.12874486f, -0.21889797f, -0.16724066f, -0.17998593f,
             0.29604816f,  0.01625840f,  0.10557223f,  -0.47639084f, -0.25408539f, 0.32537109f});
        Tensor cos_cache = Tensor::FromFloat(
            "", {8, 4},
            {-0.36114800f, 0.35355926f,  -0.30919975f, 0.08976550f,  -0.25387198f, 0.05898609f,
             -0.26989135f, -0.07698163f, 0.50176436f,  -0.11258090f, 0.61103845f,  -0.13763802f,
             -0.35273090f, 0.02251575f,  -0.12269706f, 0.52695960f,  0.25827691f,  0.35438219f,
             0.18950105f,  0.74129915f,  0.23829076f,  0.15940586f,  -0.24881957f, -0.27279115f,
             0.05527078f,  0.29932144f,  0.33508772f,  -0.28320017f, 0.15942201f,  0.05800380f,
             -0.33547908f, 0.15355368f});
        Tensor sin_cache = Tensor::FromFloat(
            "", {8, 4},
            {-0.68116987f, 0.07894906f,  0.74139404f,  -0.30596557f, 0.00562578f,  -0.56827927f,
             -0.22650050f, 0.22685932f,  -0.31273860f, -0.01027744f, -0.10655049f, -0.11352851f,
             0.05719461f,  0.14531890f,  0.36908033f,  0.24989119f,  -0.16948253f, 0.42440879f,
             0.37448436f,  -0.46768442f, 0.19956978f,  0.24767855f,  0.28989565f,  0.16415259f,
             -0.38915497f, -0.08029030f, -0.62106198f, -0.04736777f, 0.60868281f,  0.20447126f,
             0.25086108f,  -0.15404642f});
        Tensor seqlens_k = Tensor::FromInt32("", {1}, {4});
        Tensor total_sequence_length = Tensor::FromInt32("", {}, {5});

        Tensor output = Tensor::FromFloat(
            "", {1, 2, 32},
            {0.04011499f,  -0.03126945f, -0.04358935f, 0.26507425f,  0.10958266f,  -0.14726005f,
             -0.07057313f, -0.04284225f, 0.03882301f,  -0.03573211f, -0.03132916f, 0.25424054f,
             0.11318220f,  -0.14877345f, -0.05795662f, -0.05642154f, -0.17373003f, -0.20582926f,
             0.07062818f,  0.21520983f,  -0.06191964f, -0.26419181f, 0.16492207f,  0.18360150f,
             -0.16291133f, -0.20999458f, 0.06578688f,  0.21747601f,  -0.06145441f, -0.26491684f,
             0.16516756f,  0.18983750f,  0.02755510f,  0.03034053f,  -0.00752039f, 0.26124233f,
             0.17477928f,  -0.06389167f, -0.04567275f, -0.07897687f, 0.02505292f,  0.01972749f,
             -0.00815785f, 0.25487310f,  0.18327484f,  -0.06359573f, -0.05554926f, -0.07603817f,
             -0.18106356f, -0.16718511f, 0.12602295f,  0.18929057f,  -0.04455392f, -0.19862808f,
             0.12827682f,  0.18395641f,  -0.17697665f, -0.17044286f, 0.12171482f,  0.19194539f,
             -0.04540814f, -0.19714347f, 0.13347986f,  0.18642747f});
        Tensor present_key = Tensor::FromFloat(
            "", {1, 2, 5, 8},
            {-0.33688846f, -0.59257430f, -0.12754501f, -0.34472215f, 0.48454142f,  -0.04754306f,
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
             0.00535407f,  -0.07941843f});
        Tensor present_value = Tensor::FromFloat(
            "", {1, 2, 5, 8},
            {0.08679526f,  0.05379170f,  0.41924417f,  0.08761404f,  0.19152170f,  -0.00836631f,
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
             0.01072286f,  0.16417101f});

        return IoData{{std::move(query), std::move(key), std::move(value), std::move(past_key),
                       std::move(past_value), std::move(seqlens_k),
                       std::move(total_sequence_length), std::move(cos_cache),
                       std::move(sin_cache)},
                      {std::move(output), std::move(present_key), std::move(present_value)}};
      });
}

} // namespace

void RegisterCpuGroupQueryAttentionCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId microsoft_opset(kMicrosoftDomain, 1);
  if (mode == TestMode::BENCHMARK) {
    struct Profile {
      const char *name;
      std::int64_t batch;
      std::int64_t sequence;
      std::int64_t num_heads;
      std::int64_t kv_num_heads;
      std::int64_t head_dim;
    };
    for (const Profile &profile : {Profile{"mha", 1, 16, 4, 4, 32}, Profile{"mqa", 4, 32, 8, 1, 64},
                                   Profile{"gqa", 2, 64, 8, 2, 64}}) {
      for (const DataType data_type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
        const std::string name =
            "test_cpu_group_query_attention_" + std::string(profile.name) + "_b" +
            std::to_string(profile.batch) + "_s" + std::to_string(profile.sequence) + "_qh" +
            std::to_string(profile.num_heads) + "_kvh" + std::to_string(profile.kv_num_heads) +
            "_hd" + std::to_string(profile.head_dim) + "_causal_" + DataTypeSuffix(data_type) +
            "_benchmark";
        RegisterGroupQueryAttentionCase(registry, microsoft_opset, name, profile.batch,
                                        profile.sequence, profile.num_heads, profile.kv_num_heads,
                                        profile.head_dim, data_type);
      }
    }
    // Prefill: no KV cache yet (past_length=0), representative dynamic
    // sequence lengths.
    for (const std::int64_t sequence : {std::int64_t{1}, std::int64_t{16}, std::int64_t{128}}) {
      const std::string name =
          "test_cpu_group_query_attention_model_qwen3_8b_int4_mb_prefill_b1_s" +
          std::to_string(sequence) + "_qh32_kvh8_hd128_pastlen0_causal_float32_benchmark";
      RegisterQwen3GroupQueryAttentionBenchmarkCase(registry, microsoft_opset, name, sequence,
                                                    /*past_length=*/0);
    }
    // Token generation (decode): a single new token attending to a
    // previously accumulated KV cache; past lengths are bounded well under
    // the model's 40960-token rotary cache so the case stays cheap to run.
    for (const std::int64_t past_length :
         {std::int64_t{16}, std::int64_t{128}, std::int64_t{1024}}) {
      const std::string name =
          "test_cpu_group_query_attention_model_qwen3_8b_int4_mb_decode_b1_s1_qh32_kvh8_hd128_"
          "pastlen" +
          std::to_string(past_length) + "_causal_float32_benchmark";
      RegisterQwen3GroupQueryAttentionBenchmarkCase(registry, microsoft_opset, name,
                                                    /*sequence=*/1, past_length);
    }
    return;
  }

  RegisterGroupQueryAttentionCase(registry, microsoft_opset,
                                  "test_cpu_group_query_attention_gqa_causal_float32", 2, 3, 4, 2,
                                  8, DataType::FLOAT);
  RegisterGroupQueryAttentionCase(registry, microsoft_opset,
                                  "test_cpu_group_query_attention_mqa_bidirectional_scale_float16",
                                  1, 4, 4, 1, 8, DataType::FLOAT16, 0, 0.25f);
  RegisterGroupQueryAttentionCase(registry, microsoft_opset,
                                  "test_cpu_group_query_attention_mha_causal_softcap_bfloat16", 1,
                                  4, 4, 4, 8, DataType::BFLOAT16, 1, std::nullopt, 2.0f);
  RegisterCachedRotaryCorrectnessCase(registry, microsoft_opset);
}

} // namespace onnx_light_cpu::backend_test
