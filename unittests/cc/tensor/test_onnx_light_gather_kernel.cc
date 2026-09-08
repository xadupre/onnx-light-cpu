// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/tensor/gather_kernel.h"

#include "onnx_light_cpu/impl/execution.h"

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;
using BuiltinGather = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Gather;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

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
    bytes[i] = static_cast<uint8_t>(i * 37 + 131);
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

void Compare(const Tensor &data, const Tensor &indices, int64_t axis) {
  onnx_light_cpu::GatherKernel kernel(MakeCtx(), axis);
  BuiltinGather reference(MakeCtx());
  const Tensor expected = reference(data, indices, axis);
  EqualBytes(kernel(data, indices), expected);
  Tensor output =
      rt_ns::MakeOutputTensor(expected.data_type, expected.shape, expected.size_bytes(), nullptr);
  kernel(data, indices, output);
  EqualBytes(output, expected);
}

TEST(OnnxLightGatherKernel, AllFixedByteTypesAndIndexLayouts) {
  for (DataType type : kTypes) {
    SCOPED_TRACE(static_cast<int>(type));
    const Tensor data = Payload(type, {3, 4, 5});
    for (int64_t axis = -3; axis < 3; ++axis) {
      SCOPED_TRACE(axis);
      const int64_t dim = data.shape[axis < 0 ? axis + 3 : axis];
      const std::vector<int64_t> values = {dim - 1, 0, -dim, -1, 0, dim - 1};
      std::vector<int32_t> values32;
      for (int64_t value : values) {
        values32.push_back(static_cast<int32_t>(value));
      }
      for (const Shape &shape : {Shape{6}, Shape{2, 3}, Shape{1, 2, 1, 3}}) {
        Compare(data, Tensor::From<int64_t>("indices", shape, values), axis);
        Compare(data, Tensor::From<int32_t>("indices", shape, values32), axis);
      }
      Compare(data, Tensor::From<int64_t>("indices", {}, {-1}), axis);
      Compare(data, Tensor::From<int32_t>("indices", {}, {0}), axis);
    }
    Compare(Payload(type, {3}), Tensor::From<int64_t>("indices", {}, {-1}), 0);
  }
}

TEST(OnnxLightGatherKernel, FloatingPointPayloadsAreBitwisePreserved) {
  const std::vector<uint32_t> patterns = {0x7fc12345, 0xff812345, 0x80000000, 0x00000000,
                                          0x7f800000, 0xff800000, 0x00000001, 0x807fffff};
  Tensor data = Tensor::From<uint32_t>("data", {2, 4}, patterns);
  data.data_type = DataType::FLOAT;
  Compare(data, Tensor::From<int64_t>("indices", {4}, {-1, 0, 0, 1}), 0);
  Compare(data, Tensor::From<int32_t>("indices", {4}, {3, 0, 1, 2}), 1);
}

TEST(OnnxLightGatherKernel, EmptyInputsAndIndices) {
  for (DataType type : kTypes) {
    for (const Shape &shape : {Shape{3, 4}, Shape{0, 4}, Shape{3, 0}, Shape{0, 0}}) {
      for (int64_t axis : {0, 1}) {
        Compare(Payload(type, shape), Tensor::From<int32_t>("indices", {0}, {}), axis);
        Compare(Payload(type, shape), Tensor::From<int64_t>("indices", {2, 0, 3}, {}), axis);
      }
    }
    Compare(Payload(type, {0, 3}), Tensor::From<int64_t>("indices", {2}, {0, -1}), 1);
    Compare(Payload(type, {3, 0}), Tensor::From<int32_t>("indices", {}, {-1}), 0);
  }
}

TEST(OnnxLightGatherKernel, InvalidRankAxesAndIndices) {
  const Tensor data = Payload(DataType::FLOAT, {3, 4});
  const Tensor indices = Tensor::From<int64_t>("indices", {1}, {0});
  BuiltinGather reference(MakeCtx());
  for (int64_t axis : {std::numeric_limits<int64_t>::min(), int64_t{-3}, int64_t{2},
                       std::numeric_limits<int64_t>::max()}) {
    EXPECT_THROW(onnx_light_cpu::GatherKernel(MakeCtx(), axis)(data, indices),
                 std::invalid_argument);
    EXPECT_THROW(reference(data, indices, axis), std::invalid_argument);
  }
  const Tensor scalar = Payload(DataType::FLOAT, {});
  EXPECT_THROW(onnx_light_cpu::GatherKernel{MakeCtx()}(scalar, indices), std::invalid_argument);
  EXPECT_THROW(reference(scalar, indices), std::invalid_argument);
  for (int64_t value : {std::numeric_limits<int64_t>::min(), int64_t{-4}, int64_t{3},
                        std::numeric_limits<int64_t>::max()}) {
    const Tensor bad = Tensor::From<int64_t>("indices", {2}, {0, value});
    EXPECT_THROW(onnx_light_cpu::GatherKernel{MakeCtx()}(data, bad), std::invalid_argument);
    EXPECT_THROW(reference(data, bad), std::invalid_argument);
  }
  for (int32_t value : {std::numeric_limits<int32_t>::min(), int32_t{-4}, int32_t{3},
                        std::numeric_limits<int32_t>::max()}) {
    const Tensor bad = Tensor::From<int32_t>("indices", {1}, {value});
    EXPECT_THROW(onnx_light_cpu::GatherKernel{MakeCtx()}(data, bad), std::invalid_argument);
    EXPECT_THROW(reference(data, bad), std::invalid_argument);
  }
  const Tensor bad_type = Payload(DataType::UINT32, {1});
  EXPECT_THROW(onnx_light_cpu::GatherKernel{MakeCtx()}(data, bad_type), std::invalid_argument);
  EXPECT_THROW(reference(data, bad_type), std::invalid_argument);
}

TEST(OnnxLightGatherKernel, ValidatesIndicesEvenWhenOutputIsEmpty) {
  const Tensor indices = Tensor::From<int64_t>("indices", {1}, {0});
  EXPECT_THROW(onnx_light_cpu::GatherKernel{MakeCtx()}(Payload(DataType::FLOAT, {0, 3}), indices),
               std::invalid_argument);
  EXPECT_THROW(
      onnx_light_cpu::GatherKernel(MakeCtx(), 1)(Payload(DataType::FLOAT, {0, 0}), indices),
      std::invalid_argument);
  const Tensor bad = Tensor::From<int64_t>("indices", {1}, {3});
  EXPECT_THROW(onnx_light_cpu::GatherKernel{MakeCtx()}(Payload(DataType::FLOAT, {3, 0}), bad),
               std::invalid_argument);
}

TEST(OnnxLightGatherKernel, UnsupportedTypesPreserveBuiltinErrors) {
  const Tensor indices = Tensor::From<int64_t>("indices", {1}, {0});
  BuiltinGather reference(MakeCtx());
  for (DataType type : {DataType::STRING, DataType::INT4, DataType::UINT4, DataType::FLOAT4E2M1,
                        DataType::INT2, DataType::UINT2, DataType::FLOAT6E2M3, DataType::FLOAT6E3M2,
                        DataType::COMPLEX64, DataType::COMPLEX128, DataType::UNDEFINED}) {
    Tensor data;
    data.data_type = type;
    data.shape = {1};
    EXPECT_THROW(onnx_light_cpu::GatherKernel{MakeCtx()}(data, indices), std::invalid_argument);
    EXPECT_THROW(reference(data, indices), std::invalid_argument);
  }
}

TEST(OnnxLightGatherKernel, CheckedShapesBytesAndBuffers) {
  onnx_light_cpu::GatherKernel kernel(MakeCtx());
  const Tensor indices = Tensor::From<int64_t>("indices", {1}, {0});
  Tensor malformed;
  malformed.data_type = DataType::FLOAT;
  for (const Shape &shape : {Shape{-1}, Shape{std::numeric_limits<int64_t>::max(), 2},
                             Shape{std::numeric_limits<int64_t>::max() / 2}, Shape{1}}) {
    malformed.shape = shape;
    EXPECT_THROW(kernel(malformed, indices), std::invalid_argument);
  }
  const Tensor data = Payload(DataType::FLOAT, {2, 3});
  Tensor bad_indices = indices;
  bad_indices.shape = {2};
  EXPECT_THROW(kernel(data, bad_indices), std::invalid_argument);
  bad_indices.shape = {-1};
  EXPECT_THROW(kernel(data, bad_indices), std::invalid_argument);
  bad_indices.shape = {std::numeric_limits<int64_t>::max(), 2};
  EXPECT_THROW(kernel(data, bad_indices), std::invalid_argument);
  const Tensor two_indices = Tensor::From<int64_t>("indices", {2}, {0, 0});
  malformed.shape = {0, std::numeric_limits<int64_t>::max()};
  EXPECT_THROW(kernel(malformed, two_indices), std::invalid_argument);
  malformed.shape = {0, std::numeric_limits<int64_t>::max() / 2};
  EXPECT_THROW(kernel(malformed, two_indices), std::invalid_argument);

  Shape max_rank;
  max_rank.assign(Shape::kMaxRank, 1);
  EXPECT_THROW(
      kernel(Payload(DataType::FLOAT, max_rank), Tensor::From<int64_t>("indices", {1, 1}, {0})),
      std::invalid_argument);
}

TEST(OnnxLightGatherKernel, PreallocatedOutputValidationAndNoPartialWrites) {
  onnx_light_cpu::GatherKernel kernel(MakeCtx());
  const Tensor data = Payload(DataType::FLOAT, {3, 2});
  const Tensor indices = Tensor::From<int64_t>("indices", {2}, {2, 0});
  Tensor output = Payload(DataType::FLOAT, {2, 2});
  const Tensor original = Payload(DataType::FLOAT, {2, 2});
  const Tensor invalid = Tensor::From<int64_t>("indices", {2}, {0, 3});
  EXPECT_THROW(kernel(data, invalid, output), std::invalid_argument);
  EqualBytes(output, original);
  output.data_type = DataType::INT32;
  EXPECT_THROW(kernel(data, indices, output), std::invalid_argument);
  output = original;
  output.shape = {4};
  EXPECT_THROW(kernel(data, indices, output), std::invalid_argument);
  output = Payload(DataType::FLOAT, {3});
  output.shape = {2, 2};
  EXPECT_THROW(kernel(data, indices, output), std::invalid_argument);

  Tensor alias = Tensor::Borrow("output", data.data_type, {2, 2}, data.bytes(), 16);
  EXPECT_THROW(kernel(data, indices, alias), std::invalid_argument);
  const Tensor int_data = Payload(DataType::INT64, {2});
  const Tensor valid_indices = Tensor::From<int64_t>("indices", {2}, {1, 0});
  Tensor index_alias = Tensor::Borrow("output", DataType::INT64, {2}, valid_indices.bytes(), 16);
  EXPECT_THROW(kernel(int_data, valid_indices, index_alias), std::invalid_argument);
}

TEST(OnnxLightGatherKernel, NodeConstructorReadsAxis) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Gather");
  auto *axis = node.add_attribute();
  axis->set_name("axis");
  axis->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
  axis->set_i(-1);
  onnx_light_cpu::GatherKernel kernel(node, MakeCtx());
  const Tensor data = Payload(DataType::FLOAT, {2, 3});
  const Tensor indices = Tensor::From<int64_t>("indices", {2}, {2, 0});
  EqualBytes(kernel(data, indices), BuiltinGather(MakeCtx())(data, indices, -1));
}

struct InlineExecutor {
  int64_t dispatches = 0;
  int64_t blocks = 0;
  int64_t depth = 0;
  int64_t maximum_depth = 0;
  bool run_nested = false;

  static void Run(void *context, int64_t count, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = count;
    ++self.depth;
    self.maximum_depth = std::max(self.maximum_depth, self.depth);
    for (int64_t i = 0; i < count; ++i) {
      if (self.run_nested) {
        onnx_light_cpu::detail::ExecutionRegionScope region;
        Compare(Payload(DataType::FLOAT, {16, 32768}),
                Tensor::From<int32_t>("indices", {16}, std::vector<int32_t>(16, -1)), 0);
      }
      task(task_context, i);
    }
    --self.depth;
  }
};

TEST(OnnxLightGatherKernel, RuntimeSchedulingSmallLargeAndNested) {
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  Compare(Payload(DataType::FLOAT, {3, 4}), Tensor::From<int32_t>("indices", {2}, {2, 0}), 0);
  EXPECT_EQ(executor.dispatches, 0);
  executor.run_nested = true;
  Compare(
      Payload(DataType::FLOAT, {16, 32768}),
      Tensor::From<int64_t>("indices", {16}, {15, 0, 3, 2, 1, -1, 7, 8, 3, 3, 2, 2, 1, 1, 0, 0}),
      0);
  EXPECT_GT(executor.dispatches, 0);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_LE(executor.blocks, view.effective_threads);
  EXPECT_EQ(executor.maximum_depth, 1);
  executor.run_nested = false;
  Compare(Payload(DataType::FLOAT, {65537, 8}),
          Tensor::From<int32_t>("indices", {8}, {7, 0, -1, 3, 2, 1, 0, 0}), 1);
  EXPECT_GT(executor.blocks, 1);
  const int64_t calls = executor.dispatches;
  {
    onnx_light_cpu::detail::ExecutionRegionScope region;
    Compare(Payload(DataType::FLOAT, {16, 32768}),
            Tensor::From<int32_t>("indices", {16}, std::vector<int32_t>(16, -1)), 0);
  }
  EXPECT_EQ(executor.dispatches, calls);
}

} // namespace
