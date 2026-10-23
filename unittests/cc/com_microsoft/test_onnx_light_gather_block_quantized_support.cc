// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace shapes_ns = ONNX_LIGHT_NAMESPACE::core::shapes;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using sym_ns::SymDim;
using sym_ns::SymShape;
using sym_ns::SymTensor;
using sym_ns::TensorType;

NodeProto MakeGather(bool zero_points = true) {
  NodeProto node;
  node.set_domain(onnx_light_cpu::kMicrosoftDomain);
  node.set_op_type("GatherBlockQuantized");
  for (const char *name : {"data", "indices", "scales"})
    node.add_input(name);
  if (zero_points)
    node.add_input("zero_points");
  node.add_output("output");
  return node;
}

void SetInputs(shapes_ns::ShapesContext &ctx, int64_t bits = 4,
               TensorType output_type = TensorType::kFloat,
               TensorType indices_type = TensorType::kInt64,
               SymShape indices_shape = {SymDim(2), SymDim(3)}) {
  const int64_t logical_last = 70 * (8 / bits);
  const int64_t blocks = logical_last / 128 + (logical_last % 128 != 0);
  const int64_t packing = 8 / bits;
  ctx.Set("data", SymTensor(nullptr, TensorType::kUint8, {SymDim(5), SymDim(70)}));
  ctx.Set("indices", SymTensor(nullptr, indices_type, std::move(indices_shape)));
  ctx.Set("scales", SymTensor(nullptr, output_type, {SymDim(5), SymDim(blocks)}));
  ctx.Set("zero_points",
          SymTensor(nullptr, TensorType::kUint8,
                    {SymDim(5), SymDim(blocks / packing + (blocks % packing != 0))}));
}

TEST(GatherBlockQuantizedSupport, SchemaDescribesOnlySupportedTypesAndDefaults) {
  const auto schemas = onnx_light_cpu::GetMicrosoftOpSchemasWithHistory("GatherBlockQuantized");
  ASSERT_EQ(schemas.size(), 1U);
  const auto &schema = schemas[0];
  EXPECT_EQ(schema.domain(), "com.microsoft");
  EXPECT_EQ(schema.since_version(), 1);
  ASSERT_EQ(schema.inputs().size(), 4U);
  EXPECT_EQ(schema.inputs()[3].name, "zero_points");
  ASSERT_EQ(schema.outputs().size(), 1U);
  ASSERT_EQ(schema.type_constraints().size(), 3U);
  EXPECT_EQ(schema.type_constraints()[0].allowed_type_strs,
            std::vector<TensorType>({TensorType::kUint8}));
  EXPECT_EQ(schema.type_constraints()[1].allowed_type_strs,
            std::vector<TensorType>({TensorType::kInt32, TensorType::kInt64}));
  EXPECT_EQ(schema.type_constraints()[2].allowed_type_strs,
            std::vector<TensorType>({TensorType::kFloat, TensorType::kFloat16}));
  ASSERT_EQ(schema.attributes().size(), 4U);
  for (const auto &[name, value] : std::vector<std::pair<std::string, int64_t>>{
           {"gather_axis", 0}, {"quantize_axis", 1}, {"block_size", 128}, {"bits", 4}}) {
    const auto found = std::find_if(schema.attributes().begin(), schema.attributes().end(),
                                    [&](const auto &attr) { return attr.name == name; });
    ASSERT_NE(found, schema.attributes().end());
    EXPECT_FALSE(found->required);
    EXPECT_EQ(std::get<int64_t>(found->default_value), value);
  }
}

TEST(GatherBlockQuantizedSupport, InfersEveryPackedWidthAndTypeWithScalarAndTensorIndices) {
  for (const int64_t bits : {2, 4, 8}) {
    for (const auto output_type : {TensorType::kFloat, TensorType::kFloat16}) {
      for (const auto indices_type : {TensorType::kInt32, TensorType::kInt64}) {
        for (const bool scalar : {false, true}) {
          shapes_ns::ShapesContext ctx;
          SetInputs(ctx, bits, output_type, indices_type,
                    scalar ? SymShape{} : SymShape{SymDim(2), SymDim(3)});
          auto node = MakeGather();
          ONNX_LIGHT_NAMESPACE::AddAttribute(node, "bits", bits);
          onnx_light_cpu::ComputeShapeGatherBlockQuantized(ctx, node);
          EXPECT_EQ(ctx.Get("output").Dtype(), output_type);
          EXPECT_EQ(ctx.Get("output").Shape(),
                    scalar ? SymShape{SymDim(70 * (8 / bits))}
                           : SymShape({SymDim(2), SymDim(3), SymDim(70 * (8 / bits))}));
        }
      }
    }
  }
}

TEST(GatherBlockQuantizedSupport, AcceptsDefaultsOmittedZeroPointsAndEmptyIndices) {
  for (const bool empty_slot : {false, true}) {
    shapes_ns::ShapesContext ctx;
    SetInputs(ctx, 4, TensorType::kFloat, TensorType::kInt64, {SymDim(int64_t{0})});
    auto node = MakeGather(false);
    if (empty_slot)
      node.add_input("");
    onnx_light_cpu::ComputeShapeGatherBlockQuantized(ctx, node);
    EXPECT_EQ(ctx.Get("output").Shape(), SymShape({SymDim(int64_t{0}), SymDim(140)}));
  }
}

TEST(GatherBlockQuantizedSupport, PreservesHigherRankSymbolicDimensionsAndConstrainsParameters) {
  shapes_ns::ShapesContext ctx;
  ctx.Set("data", SymTensor(nullptr, TensorType::kUint8,
                            {SymDim("rows"), SymDim("heads"), SymDim("groups"), SymDim("packed")}));
  ctx.Set("indices", SymTensor(nullptr, TensorType::kInt32, {SymDim("I"), SymDim("J")}));
  ctx.Set("scales", SymTensor(nullptr, TensorType::kFloat16,
                              {SymDim("R"), SymDim("H"), SymDim("G"), SymDim("blocks")}));
  ctx.Set("zero_points", SymTensor(nullptr, TensorType::kUint8,
                                   {SymDim("ZR"), SymDim("ZH"), SymDim("ZG"), SymDim("ZB")}));
  auto node = MakeGather();
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "gather_axis", int64_t{-4});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "quantize_axis", int64_t{-1});
  onnx_light_cpu::ComputeShapeGatherBlockQuantized(ctx, node);
  EXPECT_EQ(ctx.Get("output").Shape(), SymShape({SymDim("I"), SymDim("J"), SymDim("heads"),
                                                 SymDim("groups"), SymDim("(packed)*2")}));
  EXPECT_EQ(ctx.Constraints().size(), 8U);
  EXPECT_EQ(ctx.Constraints().count({"rows", "R"}) + ctx.Constraints().count({"R", "rows"}), 1U);
  EXPECT_EQ(ctx.Constraints().count({"blocks", "ceil(((packed)*2)/128)"}), 1U);
  EXPECT_EQ(ctx.Constraints().count({"ZB", "ceil((blocks)/2)"}), 1U);
}

TEST(GatherBlockQuantizedSupport, AcceptsHigherRankConcreteShapesAndNonDefaultBlockSize) {
  shapes_ns::ShapesContext ctx;
  ctx.Set("data", SymTensor(nullptr, TensorType::kUint8, {SymDim(5), SymDim(2), SymDim(35)}));
  ctx.Set("indices", SymTensor(nullptr, TensorType::kInt64, {}));
  ctx.Set("scales", SymTensor(nullptr, TensorType::kFloat, {SymDim(5), SymDim(2), SymDim(5)}));
  ctx.Set("zero_points", SymTensor(nullptr, TensorType::kUint8, {SymDim(5), SymDim(2), SymDim(3)}));
  auto node = MakeGather();
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "block_size", int64_t{16});
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "quantize_axis", int64_t{2});
  onnx_light_cpu::ComputeShapeGatherBlockQuantized(ctx, node);
  EXPECT_EQ(ctx.Get("output").Shape(), SymShape({SymDim(2), SymDim(70)}));
  auto defaults = MakeGather();
  EXPECT_THROW(onnx_light_cpu::ComputeShapeGatherBlockQuantized(ctx, defaults),
               std::invalid_argument);
}

TEST(GatherBlockQuantizedSupport, GraphBuilderRetainsSymbolicPackedWidthAndGroupCounts) {
  onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions();
  ONNX_LIGHT_NAMESPACE::core::builder::GraphBuilder builder(
      "symbolic_quantized_gather", [](const std::string &op_type) {
        return onnx_light_cpu::GetMicrosoftOpSchemasWithHistory(op_type, false);
      });
  builder.SetOpsetVersion("", 23);
  builder.SetOpsetVersion("com.microsoft", 1);
  builder.MakeInput("data", TensorType::kUint8,
                    {SymDim("rows"), SymDim("heads"), SymDim("packed")});
  builder.MakeInput("indices", TensorType::kInt64, {SymDim("batch"), SymDim("sequence")});
  builder.MakeInput("scales", TensorType::kFloat16,
                    {SymDim("rows"), SymDim("heads"), SymDim("groups")});
  builder.MakeInput("zero_points", TensorType::kUint8,
                    {SymDim("rows"), SymDim("heads"), SymDim("zero_groups")});
  auto node = MakeGather();
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "quantize_axis", int64_t{-1});
  builder.MakeNode("GatherBlockQuantized", {"data", "indices", "scales", "zero_points"}, {"output"},
                   "com.microsoft", "", node.attribute());
  builder.MakeOutput("output");
  EXPECT_EQ(builder.GetShape("output").Dtype(), TensorType::kFloat16);
  EXPECT_EQ(builder.GetShape("output").Shape(),
            SymShape({SymDim("batch"), SymDim("sequence"), SymDim("heads"), SymDim("(packed)*2")}));
}

TEST(GatherBlockQuantizedSupport, InfersWithUnknownTypesButRejectsKnownIncompatibleTypes) {
  for (const char *name : {"data", "indices", "scales", "zero_points"}) {
    shapes_ns::ShapesContext ctx;
    SetInputs(ctx);
    const auto shape = ctx.Get(name).Shape();
    ctx.Set(name, SymTensor(nullptr, TensorType::kUndefined, shape));
    onnx_light_cpu::ComputeShapeGatherBlockQuantized(ctx, MakeGather());
    EXPECT_EQ(ctx.Get("output").Shape(), SymShape({SymDim(2), SymDim(3), SymDim(140)}));
    EXPECT_EQ(ctx.Get("output").Dtype(),
              std::string(name) == "scales" ? TensorType::kUndefined : TensorType::kFloat);
  }
}

TEST(GatherBlockQuantizedSupport, RejectsUnsupportedAttributes) {
  for (const auto &[name, value] :
       std::vector<std::pair<std::string, int64_t>>{{"bits", 1},
                                                    {"bits", 3},
                                                    {"bits", 16},
                                                    {"block_size", 0},
                                                    {"block_size", -16},
                                                    {"block_size", 8},
                                                    {"block_size", 24},
                                                    {"gather_axis", 1},
                                                    {"gather_axis", -3},
                                                    {"quantize_axis", 0},
                                                    {"quantize_axis", 2},
                                                    {"quantize_axis", -3}}) {
    shapes_ns::ShapesContext ctx;
    SetInputs(ctx);
    auto node = MakeGather();
    ONNX_LIGHT_NAMESPACE::AddAttribute(node, name.c_str(), value);
    EXPECT_THROW(onnx_light_cpu::ComputeShapeGatherBlockQuantized(ctx, node), std::invalid_argument)
        << name << "=" << value;
  }
  shapes_ns::ShapesContext ctx;
  SetInputs(ctx);
  auto node = MakeGather();
  ONNX_LIGHT_NAMESPACE::AddAttribute(node, "bits", 4.0f);
  EXPECT_THROW(onnx_light_cpu::ComputeShapeGatherBlockQuantized(ctx, node), std::invalid_argument);
}

TEST(GatherBlockQuantizedSupport, RejectsInvalidInputTypesRanksAndDimensions) {
  const std::vector<std::pair<std::string, SymTensor>> invalid = {
      {"data", SymTensor(nullptr, TensorType::kInt8, {SymDim(5), SymDim(70)})},
      {"data", SymTensor(nullptr, TensorType::kUint8, {SymDim(70)})},
      {"data", SymTensor(nullptr, TensorType::kUint8, {SymDim(5), SymDim(-1)})},
      {"indices", SymTensor(nullptr, TensorType::kFloat, {SymDim(2)})},
      {"scales", SymTensor(nullptr, TensorType::kBfloat16, {SymDim(5), SymDim(2)})},
      {"scales", SymTensor(nullptr, TensorType::kDouble, {SymDim(5), SymDim(2)})},
      {"scales", SymTensor(nullptr, TensorType::kFloat, {SymDim(2)})},
      {"scales", SymTensor(nullptr, TensorType::kFloat, {SymDim(4), SymDim(2)})},
      {"scales", SymTensor(nullptr, TensorType::kFloat, {SymDim(5), SymDim(1)})},
      {"zero_points", SymTensor(nullptr, TensorType::kFloat, {SymDim(5), SymDim(1)})},
      {"zero_points", SymTensor(nullptr, TensorType::kUint8, {SymDim(1)})},
      {"zero_points", SymTensor(nullptr, TensorType::kUint8, {SymDim(4), SymDim(1)})},
      {"zero_points", SymTensor(nullptr, TensorType::kUint8, {SymDim(5), SymDim(2)})}};
  for (const auto &[name, tensor] : invalid) {
    shapes_ns::ShapesContext ctx;
    SetInputs(ctx);
    ctx.Set(name, SymTensor(tensor));
    EXPECT_THROW(onnx_light_cpu::ComputeShapeGatherBlockQuantized(ctx, MakeGather()),
                 std::invalid_argument)
        << name;
  }
}

TEST(GatherBlockQuantizedSupport, RejectsMissingInputsAndLogicalDimensionOverflow) {
  shapes_ns::ShapesContext ctx;
  SetInputs(ctx);
  auto node = MakeGather();
  node.ref_input()[2] = "missing";
  EXPECT_THROW(onnx_light_cpu::ComputeShapeGatherBlockQuantized(ctx, node), std::invalid_argument);
  node = MakeGather();
  node.add_input("extra");
  EXPECT_THROW(onnx_light_cpu::ComputeShapeGatherBlockQuantized(ctx, node), std::invalid_argument);
  node = MakeGather();
  node.add_output("extra");
  EXPECT_THROW(onnx_light_cpu::ComputeShapeGatherBlockQuantized(ctx, node), std::invalid_argument);
  ctx.Set("data", SymTensor(nullptr, TensorType::kUint8,
                            {SymDim(5), SymDim(std::numeric_limits<int64_t>::max())}));
  EXPECT_THROW(onnx_light_cpu::ComputeShapeGatherBlockQuantized(ctx, MakeGather()),
               std::invalid_argument);
}

TEST(GatherBlockQuantizedSupport, ExposesInferenceOnlyInventoryAndZeroScratchMemory) {
  const auto support = onnx_light_cpu::CollectOperatorSupport();
  const auto found = std::find_if(support.begin(), support.end(), [](const auto &entry) {
    return entry.domain == "com.microsoft" && entry.op_type == "GatherBlockQuantized";
  });
  ASSERT_NE(found, support.end());
  EXPECT_FALSE(found->has_gradient);
  EXPECT_TRUE(found->fusion_patterns.empty());
  EXPECT_EQ(found->shape_inference_function, "onnx_light_cpu::ComputeShapeGatherBlockQuantized");
  EXPECT_EQ(found->peak_memory_function, "onnx_light_cpu::ComputePeakMemoryGatherBlockQuantized");
  EXPECT_EQ(onnx_light_cpu::ComputePeakMemoryGatherBlockQuantized(sym_ns::Device::kCPU, {}), 0);
  onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions();
  EXPECT_EQ(shapes_ns::ComputePeakMemory("com.microsoft", "GatherBlockQuantized",
                                         sym_ns::Device::kCPU, {}),
            0);
}

} // namespace
