// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/tensor/split_kernel.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/tensor/split_kernel.h"

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;
using BuiltinSplit = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Split;
using Values = std::vector<int64_t>;

rt_ns::KernelContext MakeCtx(int64_t version = 18) {
  return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), version));
}

constexpr std::array<DataType, 18> kTypes = {
    DataType::FLOAT,      DataType::DOUBLE,         DataType::FLOAT16,
    DataType::BFLOAT16,   DataType::INT8,           DataType::UINT8,
    DataType::INT16,      DataType::UINT16,         DataType::INT32,
    DataType::UINT32,     DataType::INT64,          DataType::UINT64,
    DataType::BOOL,       DataType::FLOAT8E4M3FN,   DataType::FLOAT8E4M3FNUZ,
    DataType::FLOAT8E5M2, DataType::FLOAT8E5M2FNUZ, DataType::FLOAT8E8M0};

Tensor Payload(DataType type, const Shape &shape) {
  std::vector<uint8_t> bytes(static_cast<std::size_t>(shape.product()) * rt_ns::ElementSize(type));
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<uint8_t>((i * 37 + i / 251 + 131) % 256);
  }
  return Tensor("data", type, shape, std::move(bytes));
}

void EqualBytes(const Tensor &actual, const Tensor &expected) {
  ASSERT_EQ(actual.data_type, expected.data_type);
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  if (actual.size_bytes() != 0) {
    EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
  }
}

void Compare(const Tensor &data, int64_t axis, const Values &sizes, int64_t count = 0) {
  const auto expected = BuiltinSplit{MakeCtx()}(data, axis, sizes, count);
  const auto actual = onnx_light_cpu::SplitKernel{MakeCtx()}(data, axis, sizes, count);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    SCOPED_TRACE(i);
    EqualBytes(actual[i], expected[i]);
    if (actual[i].size_bytes() != 0) {
      EXPECT_NE(actual[i].bytes(), data.bytes());
    }
  }
}

TEST(OnnxLightSplitKernel, AllFixedByteTypesAxesAndTails) {
  for (DataType type : kTypes) {
    SCOPED_TRACE(static_cast<int>(type));
    const Tensor data = Payload(type, {3, 4, 17});
    Compare(data, 0, {1, 2});
    Compare(data, -3, {1, 2});
    Compare(data, 1, {1, 2, 1});
    Compare(data, -2, {1, 2, 1});
    Compare(data, 2, {1, 7, 9});
    Compare(data, -1, {1, 7, 9});
    Compare(data, -1, {0, 8, 0, 9, 0});
    Compare(data, -1, {}, 3);
    for (int64_t tail : {7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65}) {
      Compare(Payload(type, {2, tail + 1}), -1, {1, tail});
    }
  }
  Shape shape;
  shape.assign(Shape::kMaxRank, 1);
  shape.back() = 3;
  Compare(Payload(DataType::FLOAT, shape), -1, {1, 2});
  Compare(Payload(DataType::UINT8, {20}), 0, Values(20, 1));
}

TEST(OnnxLightSplitKernel, LowLevelUnalignedRowsAndBoundaryGuards) {
  constexpr std::size_t rows = 5;
  constexpr std::size_t stride = 37;
  std::array<uint8_t, 1 + rows * stride> source{};
  for (std::size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>(i * 19);
  }
  for (std::size_t width :
       {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}, std::size_t{13}, stride}) {
    for (std::size_t offset : {std::size_t{0}, stride - width}) {
      std::vector<uint8_t> destination(2 + rows * width, 0xa5);
      onnx_light_cpu::SplitCopy(source.data() + 1, destination.data() + 1, rows, stride, width,
                                offset);
      EXPECT_EQ(destination.front(), 0xa5);
      EXPECT_EQ(destination.back(), 0xa5);
      for (std::size_t row = 0; row < rows; ++row) {
        EXPECT_EQ(std::memcmp(destination.data() + 1 + row * width,
                              source.data() + 1 + row * stride + offset, width),
                  0);
      }
    }
  }
}

TEST(OnnxLightSplitKernel, QkvSingleAndMultipleRowsAndUneven18) {
  for (DataType type : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (int64_t rows : {1, 2, 17, 257}) {
      Compare(Payload(type, {rows, 4096}), -1, {2048, 1024, 1024});
    }
  }
  Compare(Payload(DataType::FLOAT, {4096}), 0, {2048, 1024, 1024});
  Compare(Payload(DataType::FLOAT, {2, 10, 3}), 1, {}, 3);
  Compare(Payload(DataType::FLOAT, {2, 7}), -1, {}, 4);
  Compare(Payload(DataType::FLOAT, {2, 3}), -1, {}, 4);
  Compare(Payload(DataType::FLOAT, {3, 12}), -1, {}, 3);
}

TEST(OnnxLightSplitKernel, SpecialValuesAndOwnedMaterializedOutputs) {
  Tensor data = Tensor::From<uint32_t>("data", {2, 4},
                                       {0x7fc12345, 0xff812345, 0x80000000, 0x00000000, 0x7f800000,
                                        0xff800000, 0x00000001, 0x807fffff});
  data.data_type = DataType::FLOAT;
  Compare(data, -1, {1, 3});
  const auto expected = BuiltinSplit{MakeCtx()}(data, -1, {1, 3}, 0);
  auto actual = onnx_light_cpu::SplitKernel{MakeCtx()}(data, -1, {1, 3}, 0);
  std::memset(data.mutable_bytes(), 0, data.size_bytes());
  EqualBytes(actual[0], expected[0]);
  EqualBytes(actual[1], expected[1]);
  std::memset(actual[0].mutable_bytes(), 0, actual[0].size_bytes());
  EqualBytes(actual[1], expected[1]);
  data = Tensor{};
  EqualBytes(actual[1], expected[1]);
}

TEST(OnnxLightSplitKernel, ZeroDimensionsAndEmptyOutputs) {
  onnx_light_cpu::SplitKernel kernel(MakeCtx());
  for (DataType type : kTypes) {
    for (const Shape &shape : {Shape{0, 4}, Shape{3, 0}, Shape{0, 0}, Shape{2, 0, 4}}) {
      const Tensor data = Payload(type, shape);
      for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        const auto outputs = kernel(data, static_cast<int64_t>(axis), {0, shape[axis], 0}, 0);
        ASSERT_EQ(outputs.size(), 3);
        for (std::size_t i = 0; i < outputs.size(); ++i) {
          Shape expected = shape;
          expected[axis] = i == 1 ? shape[axis] : 0;
          EXPECT_EQ(outputs[i].shape, expected);
          EXPECT_EQ(outputs[i].data_type, type);
          EXPECT_EQ(outputs[i].size_bytes(), 0);
        }
      }
    }
    const auto outputs = kernel(Payload(type, {2, 0}), 1, {}, 3);
    ASSERT_EQ(outputs.size(), 3);
    for (const auto &output : outputs) {
      EXPECT_EQ(output.shape, (Shape{2, 0}));
      EXPECT_EQ(output.size_bytes(), 0);
    }
  }
  Tensor empty;
  empty.data_type = DataType::FLOAT;
  empty.shape = {std::numeric_limits<int64_t>::max(), 2, 0};
  EXPECT_EQ(kernel(empty, 1, {1, 1}, 0).size(), 2);
  onnx_light_cpu::SplitCopy(nullptr, nullptr, 0, 0, 1, 0);
  onnx_light_cpu::SplitCopy(nullptr, nullptr, 3, 0, 0, 0);
}

TEST(OnnxLightSplitKernel, InvalidShapesTypesAxesSizesAndOverflow) {
  onnx_light_cpu::SplitKernel kernel(MakeCtx());
  const Tensor data = Payload(DataType::FLOAT, {2, 4});
  for (int64_t axis : {std::numeric_limits<int64_t>::min(), int64_t{-3}, int64_t{2},
                       std::numeric_limits<int64_t>::max()}) {
    EXPECT_THROW(kernel(data, axis, {2, 2}, 0), std::invalid_argument);
  }
  EXPECT_THROW(kernel(Payload(DataType::FLOAT, {}), 0, {1}, 0), std::invalid_argument);
  for (const Values &sizes :
       {Values{-1, 5}, Values{1, 2}, Values{5}, Values{std::numeric_limits<int64_t>::max(), 1}}) {
    EXPECT_THROW(kernel(data, 1, sizes, 0), std::invalid_argument);
  }
  EXPECT_THROW(kernel(data, 1, {2, 2}, 2), std::invalid_argument);
  EXPECT_THROW(kernel(data, 1, {2, 2}, -1), std::invalid_argument);
  for (int64_t count : {int64_t{-1}, int64_t{0}, int64_t{6}, std::numeric_limits<int64_t>::max()}) {
    EXPECT_THROW(kernel(data, 1, {}, count), std::invalid_argument);
  }
  // Ceil(5 / 4) leaves a negative final chunk; the builtin clamps it and overreads.
  EXPECT_THROW(kernel(Payload(DataType::FLOAT, {5}), 0, {}, 4), std::invalid_argument);
  Tensor malformed;
  malformed.data_type = DataType::FLOAT;
  for (const Shape &shape :
       {Shape{-1}, Shape{0, -1}, Shape{1}, Shape{std::numeric_limits<int64_t>::max(), 2},
        Shape{std::numeric_limits<int64_t>::max() / 2}}) {
    malformed.shape = shape;
    EXPECT_THROW(kernel(malformed, 0, {1}, 0), std::invalid_argument);
  }
  for (DataType type : {DataType::STRING, DataType::INT4, DataType::UINT4, DataType::FLOAT4E2M1,
                        DataType::INT2, DataType::UINT2, DataType::FLOAT6E2M3, DataType::FLOAT6E3M2,
                        DataType::COMPLEX64, DataType::COMPLEX128, DataType::UNDEFINED}) {
    malformed.data_type = type;
    malformed.shape = {0};
    EXPECT_THROW(kernel(malformed, 0, {0}, 0), std::invalid_argument);
  }
}

void IntAttribute(ONNX_LIGHT_NAMESPACE::NodeProto &node, const char *name, int64_t value) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
  attribute->set_i(value);
}

void SplitAttribute(ONNX_LIGHT_NAMESPACE::NodeProto &node, const Values &values) {
  auto *attribute = node.add_attribute();
  attribute->set_name("split");
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INTS);
  for (int64_t value : values) {
    attribute->add_ints(value);
  }
}

ONNX_LIGHT_NAMESPACE::NodeProto Node(int outputs = 2) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Split");
  node.add_input("data");
  for (int i = 0; i < outputs; ++i) {
    node.add_output("out" + std::to_string(i));
  }
  return node;
}

TEST(OnnxLightSplitKernel, LegacyAndModernRunExplicitAndImplicitSizes) {
  for (int64_t version : {1, 2, 11, 12, 13, 17, 18}) {
    for (bool explicit_split : {false, true}) {
      auto node = Node(3);
      IntAttribute(node, "axis", -1);
      const Tensor data = Payload(DataType::FLOAT, {2, 12});
      const Values sizes = explicit_split ? Values{3, 4, 5} : Values{4, 4, 4};
      rt_ns::RuntimeContext rt(MakeCtx(version));
      rt.Set("data", data);
      if (explicit_split) {
        if (version < 13) {
          SplitAttribute(node, sizes);
        } else {
          node.add_input("split");
          rt.Set("split", Tensor::From<int64_t>("split", {3}, sizes));
        }
      } else if (version >= 18) {
        node.add_input("");
        IntAttribute(node, "num_outputs", 3);
      }
      onnx_light_cpu::SplitKernel kernel(node, MakeCtx(version));
      kernel.Run(rt);
      const auto expected = BuiltinSplit{MakeCtx()}(data, -1, sizes, 0);
      for (int i = 0; i < 3; ++i) {
        EqualBytes(rt.Get("out" + std::to_string(i)), expected[static_cast<std::size_t>(i)]);
      }
    }
  }
  auto node = Node(3);
  IntAttribute(node, "num_outputs", 3);
  const Tensor data = Payload(DataType::FLOAT, {10, 2});
  rt_ns::RuntimeContext rt(MakeCtx());
  rt.Set("data", data);
  onnx_light_cpu::SplitKernel kernel(node, MakeCtx());
  kernel.Run(rt);
  const auto expected = BuiltinSplit{MakeCtx()}(data, 0, {}, 3);
  for (int i = 0; i < 3; ++i) {
    EqualBytes(rt.Get("out" + std::to_string(i)), expected[static_cast<std::size_t>(i)]);
  }
}

void ExpectInvalidNode(const ONNX_LIGHT_NAMESPACE::NodeProto &node, int64_t version,
                       const Tensor *split = nullptr) {
  rt_ns::RuntimeContext rt(MakeCtx(version));
  rt.Set("data", Payload(DataType::FLOAT, {5, 4}));
  if (split) {
    rt.Set("split", *split);
  }
  onnx_light_cpu::SplitKernel kernel(node, MakeCtx(version));
  EXPECT_THROW(kernel.Run(rt), std::invalid_argument);
}

TEST(OnnxLightSplitKernel, InvalidNodeContractsAndSplitTensorValidation) {
  for (int64_t version : {2, 11, 13, 17, 18}) {
    ExpectInvalidNode(Node(), version);
    ExpectInvalidNode(Node(0), version);
    auto node = Node();
    node.clear_input();
    ExpectInvalidNode(node, version);
  }
  auto node = Node();
  node.add_input("split");
  for (Tensor split :
       {Payload(DataType::INT32, {2}), Payload(DataType::FLOAT, {2}),
        Tensor::From<int64_t>("split", {}, {5}), Tensor::From<int64_t>("split", {1, 2}, {2, 3}),
        Tensor::From<int64_t>("split", {0}, {}), Tensor::From<int64_t>("split", {1}, {5}),
        Tensor::From<int64_t>("split", {2}, {-1, 6}),
        Tensor::From<int64_t>("split", {2}, {2, 2})}) {
    ExpectInvalidNode(node, 13, &split);
    ExpectInvalidNode(node, 18, &split);
  }
  Tensor split = Tensor::From<int64_t>("split", {2}, {2, 3});
  Tensor malformed = split;
  for (const Shape &shape : {Shape{-1}, Shape{3}, Shape{std::numeric_limits<int64_t>::max()}}) {
    malformed.shape = shape;
    ExpectInvalidNode(node, 18, &malformed);
  }
  ExpectInvalidNode(node, 12, &split);
  IntAttribute(node, "num_outputs", 2);
  ExpectInvalidNode(node, 18, &split);
  node = Node();
  SplitAttribute(node, {2, 3});
  ExpectInvalidNode(node, 13);
  node = Node();
  SplitAttribute(node, {});
  ExpectInvalidNode(node, 12);
  for (int64_t value : {-1, 0, 1, 3}) {
    node = Node();
    IntAttribute(node, "num_outputs", value);
    ExpectInvalidNode(node, 18);
  }
  node = Node();
  IntAttribute(node, "num_outputs", 2);
  ExpectInvalidNode(node, 17);
  node.add_input("");
  node.add_input("extra");
  ExpectInvalidNode(node, 18);
}

TEST(OnnxLightSplitKernel, UnalignedSplitTensorAndExplicitZeroSizesRun) {
  auto node = Node(3);
  node.add_input("split");
  const Values values{0, 5, 0};
  std::array<uint8_t, 1 + 3 * sizeof(int64_t)> storage{};
  std::memcpy(storage.data() + 1, values.data(), 3 * sizeof(int64_t));
  const Tensor split =
      Tensor::Borrow("split", DataType::INT64, {3}, storage.data() + 1, 3 * sizeof(int64_t));
  const Tensor data = Payload(DataType::FLOAT, {5, 4});
  rt_ns::RuntimeContext rt(MakeCtx());
  rt.Set("data", data);
  rt.Set("split", split);
  onnx_light_cpu::SplitKernel kernel(node, MakeCtx());
  kernel.Run(rt);
  EqualBytes(rt.Get("out1"), data);
  EXPECT_EQ(rt.Get("out0").shape, (Shape{0, 4}));
  EXPECT_EQ(rt.Get("out2").size_bytes(), 0);
}

struct InlineExecutor {
  int64_t dispatches = 0;
  int64_t blocks = 0;
  int64_t depth = 0;
  int64_t maximum_depth = 0;
  bool nested = false;

  static void Run(void *context, int64_t count, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = count;
    self.maximum_depth = std::max(self.maximum_depth, ++self.depth);
    for (int64_t i = 0; i < count; ++i) {
      task(task_context, i);
      if (self.nested) {
        onnx_light_cpu::detail::ExecutionRegionScope region;
        Compare(Payload(DataType::FLOAT, {257, 4096}), -1, {2048, 1024, 1024});
      }
    }
    --self.depth;
  }
};

TEST(OnnxLightSplitKernel, RuntimeSchedulingSmallLargeSingleRowAndNested) {
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  Compare(Payload(DataType::FLOAT, {1, 4096}), -1, {2048, 1024, 1024});
  EXPECT_EQ(executor.dispatches, 0);
  Compare(Payload(DataType::FLOAT, {257, 4096}), -1, {2048, 1024, 1024});
  EXPECT_GT(executor.dispatches, 0);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_LE(executor.blocks, view.effective_threads);
  int64_t previous = executor.dispatches;
  Compare(Payload(DataType::FLOAT, {1, 1048583}), -1, {524291, 524292});
  EXPECT_GT(executor.dispatches, previous);
  previous = executor.dispatches;
  Compare(Payload(DataType::FLOAT, {262145, 4}), -1, {1, 1, 1, 1});
  EXPECT_GT(executor.dispatches, previous);
  previous = executor.dispatches;
  {
    onnx_light_cpu::detail::ExecutionRegionScope region;
    Compare(Payload(DataType::FLOAT, {257, 4096}), -1, {2048, 1024, 1024});
  }
  EXPECT_EQ(executor.dispatches, previous);
  executor.nested = true;
  Compare(Payload(DataType::FLOAT, {257, 4096}), -1, {2048, 1024, 1024});
  EXPECT_EQ(executor.maximum_depth, 1);
  executor.nested = false;
  previous = executor.dispatches;
  view.effective_threads = 1;
  Compare(Payload(DataType::FLOAT, {257, 4096}), -1, {2048, 1024, 1024});
  EXPECT_EQ(executor.dispatches, previous);
  view.effective_threads = 8;
  view.run_blocks = nullptr;
  Compare(Payload(DataType::FLOAT, {257, 4096}), -1, {2048, 1024, 1024});
  EXPECT_EQ(executor.dispatches, previous);
}

} // namespace
