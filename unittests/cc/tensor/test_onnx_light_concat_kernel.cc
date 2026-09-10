// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/tensor/concat_kernel.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;
using rt_ns::Tensors;
using BuiltinConcat = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Concat;

rt_ns::KernelContext MakeCtx(int64_t opset = 13) {
  return rt_ns::KernelContext(rt_ns::DefaultOpset(opset));
}

constexpr std::array<DataType, 18> kTypes = {
    DataType::FLOAT,      DataType::DOUBLE,         DataType::FLOAT16,
    DataType::BFLOAT16,   DataType::INT8,           DataType::UINT8,
    DataType::INT16,      DataType::UINT16,         DataType::INT32,
    DataType::UINT32,     DataType::INT64,          DataType::UINT64,
    DataType::BOOL,       DataType::FLOAT8E4M3FN,   DataType::FLOAT8E4M3FNUZ,
    DataType::FLOAT8E5M2, DataType::FLOAT8E5M2FNUZ, DataType::FLOAT8E8M0};

Tensor Payload(DataType type, const Shape &shape, std::size_t seed = 131) {
  std::vector<uint8_t> bytes(static_cast<std::size_t>(shape.product()) * rt_ns::ElementSize(type));
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<uint8_t>(i * 37 + seed);
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

void Compare(const Tensors &inputs, int64_t axis, int64_t opset = 13) {
  BuiltinConcat reference(MakeCtx(opset));
  const Tensor expected = reference(inputs, axis);
  onnx_light_cpu::ConcatKernel kernel(MakeCtx(opset));
  EqualBytes(kernel(inputs, axis), expected);
  Tensor output =
      rt_ns::MakeOutputTensor(expected.data_type, expected.shape, expected.size_bytes(), nullptr);
  kernel(inputs, axis, output);
  EqualBytes(output, expected);
}

TEST(OnnxLightConcatKernel, AllFixedByteTypesAxesRanksAndUnequalExtents) {
  for (DataType type : kTypes) {
    SCOPED_TRACE(static_cast<int>(type));
    for (const Shape &shape : {Shape{3}, Shape{3, 5}, Shape{2, 3, 5}, Shape{2, 1, 3, 5}}) {
      const int64_t rank = static_cast<int64_t>(shape.size());
      for (int64_t axis = 0; axis < rank; ++axis) {
        SCOPED_TRACE(axis);
        Shape singleton = shape;
        singleton[static_cast<std::size_t>(axis)] = 1;
        Shape tail = shape;
        tail[static_cast<std::size_t>(axis)] = 7;
        const Tensors inputs = {Payload(type, shape), Payload(type, singleton, 93),
                                Payload(type, tail, 47)};
        Compare(inputs, axis);
        Compare(inputs, axis - rank);
        Compare({inputs.front()}, axis);
        Compare({inputs[0], inputs[1]}, axis);
      }
    }
  }
}

TEST(OnnxLightConcatKernel, ManyNarrowInputsAndTails) {
  for (DataType type : kTypes) {
    for (int count : {2, 3, 4, 5, 17}) {
      Tensors uniform;
      for (int i = 0; i < count; ++i) {
        uniform.push_back(Payload(type, {17, 1}, static_cast<std::size_t>(i)));
      }
      Compare(uniform, -1);
    }
    Tensors inputs;
    for (int64_t i = 0; i < 65; ++i) {
      inputs.push_back(Payload(type, {17, i % 4}, static_cast<std::size_t>(i)));
    }
    Compare(inputs, -1);
  }
  for (int64_t size : {7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65}) {
    Compare({Payload(DataType::FLOAT, {size}), Payload(DataType::FLOAT, {size + 1}, 19)}, 0);
  }
}

TEST(OnnxLightConcatKernel, FloatingPointPayloadsRemainBitwiseIdentical) {
  Tensor first = Tensor::From<uint32_t>("first", {2, 4},
                                        {0x7fc12345u, 0xff812345u, 0x80000000u, 0u, 0x7f800000u,
                                         0xff800000u, 0x00000001u, 0x807fffffu});
  first.data_type = DataType::FLOAT;
  Tensor second = Tensor::From<uint32_t>("second", {2, 1}, {0x7fc54321u, 0x80000000u});
  second.data_type = DataType::FLOAT;
  Compare({first, second}, 1);
  Compare({first, first}, 0);
}

TEST(OnnxLightConcatKernel, EmptyTensorsAndZeroAxisPieces) {
  for (DataType type : kTypes) {
    Compare({Payload(type, {0}), Payload(type, {0})}, 0);
    Compare({Payload(type, {0}), Payload(type, {7}), Payload(type, {0})}, 0);
    Compare({Payload(type, {3, 0}), Payload(type, {3, 5}), Payload(type, {3, 0})}, 1);
    Compare({Payload(type, {0, 3}), Payload(type, {0, 5})}, 1);
    Compare({Payload(type, {3, 0}), Payload(type, {5, 0})}, 0);
    Compare({Payload(type, {0, 0, 3}), Payload(type, {0, 0, 7})}, -1);
  }
  Tensor enormous_empty;
  enormous_empty.data_type = DataType::FLOAT;
  enormous_empty.shape = {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(),
                          0};
  const Tensors inputs{enormous_empty};
  const Tensor output = onnx_light_cpu::ConcatKernel{MakeCtx()}(inputs, 2);
  EXPECT_EQ(output.size_bytes(), 0u);
  EXPECT_EQ(output.shape, enormous_empty.shape);
}

TEST(OnnxLightConcatKernel, SingleAndRepeatedInputsDoNotAliasOutput) {
  const Tensor data = Payload(DataType::INT32, {3, 5});
  const Tensor alias =
      Tensor::Borrow("alias", data.data_type, data.shape, data.bytes(), data.size_bytes());
  Compare({alias, alias, alias}, 0);
  Compare({alias, alias}, 1);
  const std::vector<Tensor> inputs{alias};
  const Tensor output = onnx_light_cpu::ConcatKernel{MakeCtx()}(inputs, 0);
  EqualBytes(output, data);
  EXPECT_NE(output.bytes(), data.bytes());
}

TEST(OnnxLightConcatKernel, InvalidInputsAxesAndShapes) {
  onnx_light_cpu::ConcatKernel kernel(MakeCtx());
  EXPECT_THROW(kernel(std::span<const Tensor>{}, 0), std::invalid_argument);
  EXPECT_THROW(BuiltinConcat{MakeCtx()}(Tensors{}, 0), std::invalid_argument);
  const Tensors inputs{Payload(DataType::FLOAT, {3, 4})};
  for (int64_t axis : {std::numeric_limits<int64_t>::min(), int64_t{-3}, int64_t{2},
                       std::numeric_limits<int64_t>::max()}) {
    EXPECT_THROW(kernel(inputs, axis), std::invalid_argument);
    EXPECT_THROW(BuiltinConcat{MakeCtx()}(inputs, axis), std::invalid_argument);
  }
  for (const Tensors &invalid :
       {Tensors{Payload(DataType::FLOAT, {})},
        Tensors{Payload(DataType::FLOAT, {3}), Payload(DataType::INT32, {3})},
        Tensors{Payload(DataType::FLOAT, {3}), Payload(DataType::FLOAT, {3, 1})},
        Tensors{Payload(DataType::FLOAT, {3, 4}), Payload(DataType::FLOAT, {3, 5})}}) {
    EXPECT_THROW(kernel(invalid, 0), std::invalid_argument);
    EXPECT_THROW(BuiltinConcat{MakeCtx()}(invalid, 0), std::invalid_argument);
  }
}

TEST(OnnxLightConcatKernel, UnsupportedTypesMatchBuiltinRejection) {
  for (DataType type : {DataType::STRING, DataType::INT4, DataType::UINT4, DataType::FLOAT4E2M1,
                        DataType::INT2, DataType::UINT2, DataType::FLOAT6E2M3, DataType::FLOAT6E3M2,
                        DataType::COMPLEX64, DataType::COMPLEX128, DataType::UNDEFINED}) {
    Tensor data;
    data.data_type = type;
    data.shape = {1};
    const Tensors inputs{data};
    EXPECT_THROW(onnx_light_cpu::ConcatKernel{MakeCtx()}(inputs, 0), std::invalid_argument);
    EXPECT_THROW(BuiltinConcat{MakeCtx()}(inputs, 0), std::invalid_argument);
  }
  const Tensors strings{Tensor::FromStrings("strings", {2}, {"hello", "world"})};
  EXPECT_THROW(onnx_light_cpu::ConcatKernel{MakeCtx()}(strings, 0), std::invalid_argument);
  EXPECT_THROW(BuiltinConcat{MakeCtx()}(strings, 0), std::invalid_argument);
}

TEST(OnnxLightConcatKernel, CheckedDimensionsAxisSumAndByteProducts) {
  onnx_light_cpu::ConcatKernel kernel(MakeCtx());
  Tensor malformed;
  malformed.data_type = DataType::FLOAT;
  for (const Shape &shape : {Shape{-1}, Shape{1}, Shape{std::numeric_limits<int64_t>::max(), 2},
                             Shape{std::numeric_limits<int64_t>::max() / 2}, Shape{0, -1}}) {
    malformed.shape = shape;
    const Tensors inputs{malformed};
    EXPECT_THROW(kernel(inputs, 0), std::invalid_argument);
  }
  Tensor huge_empty;
  huge_empty.data_type = DataType::UINT8;
  huge_empty.shape = {std::numeric_limits<int64_t>::max(), 0};
  const Tensors overflowing_axis{huge_empty, Payload(DataType::UINT8, {1, 0})};
  EXPECT_THROW(kernel(overflowing_axis, 0), std::invalid_argument);
  if constexpr (sizeof(std::size_t) >= sizeof(int64_t)) {
    // Claimed large buffers are never read: output overflow must fail during preparation.
    uint8_t byte = 0;
    constexpr int64_t half = std::numeric_limits<int64_t>::max() / 2;
    const Tensor huge = Tensor::Borrow("huge", DataType::UINT8, {half, 2}, &byte,
                                       static_cast<std::size_t>(half) * 2);
    const Tensors overflowing_shape{huge, huge};
    EXPECT_THROW(kernel(overflowing_shape, 1), std::invalid_argument);
    constexpr int64_t elements = std::numeric_limits<int64_t>::max() / 8;
    const Tensor wide = Tensor::Borrow("wide", DataType::DOUBLE, {elements}, &byte,
                                       static_cast<std::size_t>(elements) * 8);
    const Tensors overflowing_bytes{wide, wide};
    EXPECT_THROW(kernel(overflowing_bytes, 0), std::invalid_argument);
  }
}

TEST(OnnxLightConcatKernel, ValidatesOutputBeforeAnyWrites) {
  onnx_light_cpu::ConcatKernel kernel(MakeCtx());
  const Tensors inputs{Payload(DataType::FLOAT, {2, 3}), Payload(DataType::FLOAT, {2, 1})};
  Tensor output = Payload(DataType::FLOAT, {2, 4}, 29);
  const Tensor snapshot = Payload(DataType::FLOAT, {2, 4}, 29);
  const Tensors invalid{inputs[0], Payload(DataType::FLOAT, {3, 1})};
  EXPECT_THROW(kernel(invalid, 1, output), std::invalid_argument);
  EqualBytes(output, snapshot);
  output.data_type = DataType::INT32;
  EXPECT_THROW(kernel(inputs, 1, output), std::invalid_argument);
  output = Payload(DataType::FLOAT, {8});
  EXPECT_THROW(kernel(inputs, 1, output), std::invalid_argument);
  output = Payload(DataType::FLOAT, {1});
  output.shape = {2, 4};
  EXPECT_THROW(kernel(inputs, 1, output), std::invalid_argument);
}

TEST(OnnxLightConcatKernel, RejectsOutputOverlapWithAnyInput) {
  onnx_light_cpu::ConcatKernel kernel(MakeCtx());
  const Tensor storage = Payload(DataType::INT32, {32});
  for (std::size_t offset : {std::size_t{0}, std::size_t{4}, std::size_t{12}}) {
    const Tensor first = Tensor::Borrow("first", DataType::INT32, {2}, storage.bytes() + 8, 8);
    const Tensor second = Payload(DataType::INT32, {2});
    Tensor output = Tensor::Borrow("output", DataType::INT32, {4}, storage.bytes() + offset, 16);
    const Tensors inputs{first, second};
    EXPECT_THROW(kernel(inputs, 0, output), std::invalid_argument);
    const Tensors reversed{second, first};
    EXPECT_THROW(kernel(reversed, 0, output), std::invalid_argument);
  }
  const Tensor input = Tensor::Borrow("input", storage.data_type, {2}, storage.bytes(), 8);
  Tensor output = Tensor::Borrow("output", storage.data_type, {2}, storage.bytes(), 8);
  const Tensors single{input};
  EXPECT_THROW(kernel(single, 0, output), std::invalid_argument);
}

ONNX_LIGHT_NAMESPACE::NodeProto MakeNode() {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Concat");
  node.add_input("first");
  node.add_input("second");
  node.add_output("output");
  return node;
}

void AddAxis(ONNX_LIGHT_NAMESPACE::NodeProto &node, int64_t axis) {
  auto *attribute = node.add_attribute();
  attribute->set_name("axis");
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
  attribute->set_i(axis);
}

TEST(OnnxLightConcatKernel, NodeRunAcrossOpsetsAndRepeatedNames) {
  for (int64_t opset : {1, 4, 11, 13}) {
    auto node = MakeNode();
    node.add_input("first");
    AddAxis(node, -1);
    const Tensors inputs{Payload(DataType::FLOAT, {3, 5}), Payload(DataType::FLOAT, {3, 2}, 19)};
    rt_ns::RuntimeContext runtime(MakeCtx(opset));
    runtime.Set("first", inputs[0]);
    runtime.Set("second", inputs[1]);
    onnx_light_cpu::ConcatKernel kernel(node, MakeCtx(opset));
    onnx_light_cpu::ClearUsedKernelNames();
    onnx_light_cpu::SetKernelUsageRecording(true);
    EXPECT_NO_THROW(kernel.Run(runtime));
    onnx_light_cpu::SetKernelUsageRecording(false);
    EXPECT_EQ(onnx_light_cpu::UsedKernelNames(),
              std::vector<std::string>{onnx_light_cpu::ConcatKernel::kName});
    EqualBytes(runtime.Get("output"),
               BuiltinConcat{MakeCtx(opset)}({inputs[0], inputs[1], inputs[0]}, -1));
  }
}

TEST(OnnxLightConcatKernel, NodeRequiresAxisInputsAndOneOutput) {
  auto node = MakeNode();
  rt_ns::RuntimeContext runtime(MakeCtx());
  runtime.Set("first", Payload(DataType::FLOAT, {3}));
  runtime.Set("second", Payload(DataType::FLOAT, {2}));
  onnx_light_cpu::ConcatKernel kernel(node, MakeCtx());
  EXPECT_THROW(kernel.Run(runtime), std::invalid_argument);
  AddAxis(node, 0);
  auto *attribute = node.mutable_attribute(0);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::FLOAT);
  EXPECT_THROW(kernel.Run(runtime), std::invalid_argument);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
  node.add_input("");
  EXPECT_THROW(kernel.Run(runtime), std::invalid_argument);
  node = MakeNode();
  AddAxis(node, 0);
  node.add_output("extra");
  EXPECT_THROW(kernel.Run(runtime), std::invalid_argument);
  ONNX_LIGHT_NAMESPACE::NodeProto no_inputs;
  no_inputs.set_op_type("Concat");
  no_inputs.add_output("output");
  AddAxis(no_inputs, 0);
  onnx_light_cpu::ConcatKernel empty_kernel(no_inputs, MakeCtx());
  EXPECT_THROW(empty_kernel.Run(runtime), std::invalid_argument);
}

struct InlineExecutor {
  int64_t dispatches = 0;
  int64_t blocks = 0;

  static void Run(void *context, int64_t count, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = count;
    for (int64_t i = 0; i < count; ++i) {
      task(task_context, i);
    }
  }
};

TEST(OnnxLightConcatKernel, SmallAxisZeroLargeRowsManyInputsAndNestedScheduling) {
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  Compare({Payload(DataType::FLOAT, {17}), Payload(DataType::FLOAT, {33})}, 0);
  EXPECT_EQ(executor.dispatches, 0);
  Compare({Payload(DataType::FLOAT, {400000}), Payload(DataType::FLOAT, {700001})}, 0);
  EXPECT_EQ(executor.dispatches, 2);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_LE(executor.blocks, view.effective_threads);
  Compare({Payload(DataType::FLOAT, {4097, 65}), Payload(DataType::FLOAT, {4097, 33})}, -1);
  EXPECT_EQ(executor.dispatches, 4);
  Compare({Payload(DataType::FLOAT, {1048577})}, 0);
  EXPECT_EQ(executor.dispatches, 6);
  Tensors many;
  for (int i = 0; i < 64; ++i) {
    many.push_back(Payload(DataType::FLOAT, {8193, 1}, static_cast<std::size_t>(i)));
  }
  Compare(many, 1);
  EXPECT_EQ(executor.dispatches, 8);
  Compare({Payload(DataType::FLOAT, {100001, 1}), Payload(DataType::FLOAT, {100001, 1}, 19),
           Payload(DataType::FLOAT, {100001, 1}, 47)},
          1);
  EXPECT_EQ(executor.dispatches, 10);
  {
    onnx_light_cpu::detail::ExecutionRegionScope region;
    Compare(many, 1);
    EXPECT_EQ(onnx_light_cpu::detail::ExecutionRegionDepth(), 1);
  }
  EXPECT_EQ(executor.dispatches, 10);
}

} // namespace
