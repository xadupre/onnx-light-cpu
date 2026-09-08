// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/tensor/slice_kernel.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/tensor/slice_kernel.h"

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;
using BuiltinSlice = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Slice;
using Values = std::vector<int64_t>;

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

Tensor Payload(DataType type, const Shape &shape) {
  std::vector<uint8_t> bytes(static_cast<std::size_t>(shape.product()) * rt_ns::ElementSize(type));
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<uint8_t>(i * 37 + 131);
  }
  return Tensor("data", type, shape, std::move(bytes));
}

Tensor Indices(const Values &values, DataType type = DataType::INT64) {
  const Shape shape{static_cast<int64_t>(values.size())};
  if (type == DataType::INT32) {
    std::vector<int32_t> narrowed;
    for (int64_t value : values) {
      narrowed.push_back(static_cast<int32_t>(value));
    }
    return Tensor::From<int32_t>("indices", shape, narrowed);
  }
  return Tensor::From<int64_t>("indices", shape, values);
}

void EqualBytes(const Tensor &actual, const Tensor &expected) {
  ASSERT_EQ(actual.data_type, expected.data_type);
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  if (actual.size_bytes() != 0) {
    EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
  }
}

void Compare(const Tensor &data, const Values &starts, const Values &ends,
             const std::optional<Values> &axes = std::nullopt,
             const std::optional<Values> &steps = std::nullopt,
             DataType index_type = DataType::INT64) {
  const Tensor start_tensor = Indices(starts, index_type);
  const Tensor end_tensor = Indices(ends, index_type);
  const Tensor axis_tensor = Indices(axes.value_or(Values{}), index_type);
  const Tensor step_tensor = Indices(steps.value_or(Values{}), index_type);
  const Tensor *axis = axes ? &axis_tensor : nullptr;
  const Tensor *step = steps ? &step_tensor : nullptr;
  BuiltinSlice reference(MakeCtx());
  const Tensor expected = reference(data, start_tensor, end_tensor, axis, step);
  onnx_light_cpu::SliceKernel kernel(MakeCtx());
  EqualBytes(kernel(data, start_tensor, end_tensor, axis, step), expected);
  Tensor output =
      rt_ns::MakeOutputTensor(expected.data_type, expected.shape, expected.size_bytes(), nullptr);
  kernel(data, start_tensor, end_tensor, axis, step, output);
  EqualBytes(output, expected);
}

TEST(OnnxLightSliceKernel, FixedByteTypesDefaultsPermutationsAndSteps) {
  for (DataType type : kTypes) {
    SCOPED_TRACE(static_cast<int>(type));
    for (DataType index_type : {DataType::INT32, DataType::INT64}) {
      SCOPED_TRACE(static_cast<int>(index_type));
      const Tensor data = Payload(type, {3, 4, 5});
      Compare(data, {1}, {3}, std::nullopt, std::nullopt, index_type);
      Compare(data, {-9, 1}, {99, 3}, std::nullopt, std::nullopt, index_type);
      Compare(data, {0, 0, 0}, {3, 4, 5}, std::nullopt, Values{2, 2, 2}, index_type);
      Compare(data, {-1, -1, -1}, {-99, -99, -99}, Values{2, 0, 1}, Values{-2, -1, -2}, index_type);
      Compare(data, {1, -3}, {5, -1}, Values{-1, 0}, Values{1, 1}, index_type);
      Compare(data, {}, {}, Values{}, Values{}, index_type);
      Compare(Payload(type, {1, 3, 1, 5}), {-1}, {-99}, Values{1}, Values{-1}, index_type);
      for (int64_t length : {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65}) {
        Compare(Payload(type, {length}), {0}, {length}, std::nullopt, Values{2}, index_type);
        Compare(Payload(type, {length}), {-1}, {-999}, std::nullopt, Values{-1}, index_type);
      }
    }
  }
}

TEST(OnnxLightSliceKernel, DeterministicDifferentialCorpus) {
  std::mt19937 random(2026);
  for (int trial = 0; trial < 120; ++trial) {
    SCOPED_TRACE(trial);
    const int64_t rank = 1 + static_cast<int64_t>(random() % 5);
    Shape shape;
    Values axes;
    for (int64_t i = 0; i < rank; ++i) {
      shape.push_back(1 + static_cast<int64_t>(random() % 5));
      axes.push_back(i);
    }
    std::shuffle(axes.begin(), axes.end(), random);
    axes.resize(static_cast<std::size_t>(random() % (rank + 1)));
    Values starts, ends, steps;
    for (int64_t &axis : axes) {
      starts.push_back(static_cast<int64_t>(random() % 17) - 8);
      ends.push_back(static_cast<int64_t>(random() % 17) - 8);
      steps.push_back((random() % 2 ? 1 : -1) * (1 + static_cast<int64_t>(random() % 4)));
      if (random() % 2) {
        axis -= rank;
      }
    }
    Compare(Payload(kTypes[static_cast<std::size_t>(trial) % kTypes.size()], shape), starts, ends,
            axes, steps, trial % 2 ? DataType::INT32 : DataType::INT64);
  }
}

TEST(OnnxLightSliceKernel, FloatingPointSpecialPayloads) {
  Tensor data = Tensor::From<uint32_t>("data", {2, 4},
                                       {0x7fc12345u, 0xff812345u, 0x80000000u, 0u, 0x7f800000u,
                                        0xff800000u, 0x00000001u, 0x807fffffu});
  data.data_type = DataType::FLOAT;
  Compare(data, {-1, -1}, {-99, -99}, Values{0, 1}, Values{-1, -1});
}

TEST(OnnxLightSliceKernel, EmptyDimensionsAndScalarIdentity) {
  for (DataType type : kTypes) {
    Compare(Payload(type, {0, 3}), {0}, {9});
    Compare(Payload(type, {3, 0}), {-1}, {-99}, Values{1}, Values{-1});
    Compare(Payload(type, {3, 4}), {3}, {0});
    Compare(Payload(type, {3, 4}), {0}, {3}, Values{0}, Values{-1});
    Compare(Payload(type, {}), {}, {});
  }
  Shape shape;
  shape.assign(Shape::kMaxRank, 1);
  Compare(Payload(DataType::FLOAT, shape), {}, {});
}

TEST(OnnxLightSliceKernel, ClippingAndSentinels) {
  const Tensor data = Payload(DataType::FLOAT, {3, 4, 5});
  constexpr int64_t lo = std::numeric_limits<int64_t>::min();
  constexpr int64_t hi = std::numeric_limits<int64_t>::max();
  Compare(data, {lo, lo, lo}, {hi, hi, hi});
  Compare(data, {hi, hi, hi}, {lo, lo, lo}, std::nullopt, Values{-1, -1, -1});
  Compare(data, {lo}, {hi}, Values{0}, Values{-1});
  Compare(data, {hi}, {lo}, Values{0}, Values{1});
  Compare(data, {std::numeric_limits<int32_t>::max()}, {std::numeric_limits<int32_t>::min()},
          Values{1}, Values{-1}, DataType::INT32);
}

TEST(OnnxLightSliceKernel, ExtremeStepsUseDefinedOnnxSemantics) {
  constexpr int64_t lo = std::numeric_limits<int64_t>::min();
  constexpr int64_t hi = std::numeric_limits<int64_t>::max();
  onnx_light_cpu::SliceKernel kernel(MakeCtx());
  const Tensor data = Tensor::From<int32_t>("data", {2, 3}, {0, 1, 2, 3, 4, 5});
  const Tensor axes = Indices({0});
  const Tensor reverse_step = Indices({lo});
  const Tensor forward_step = Indices({hi});
  EqualBytes(kernel(data, Indices({hi}), Indices({lo}), &axes, &reverse_step),
             Tensor::From<int32_t>("expected", {1, 3}, {3, 4, 5}));
  EqualBytes(kernel(data, Indices({0}), Indices({hi}), &axes, &forward_step),
             Tensor::From<int32_t>("expected", {1, 3}, {0, 1, 2}));
  const Tensor both_axes = Indices({1, 0});
  const Tensor both_steps = Indices({lo, hi});
  EqualBytes(kernel(data, Indices({hi, 0}), Indices({lo, hi}), &both_axes, &both_steps),
             Tensor::From<int32_t>("expected", {1, 1}, {2}));
  EXPECT_EQ(kernel(data, Indices({0}), Indices({1}), &axes, &reverse_step).element_count(), 0);
  Tensor empty;
  empty.data_type = DataType::FLOAT;
  empty.shape = {0, hi, hi};
  EXPECT_EQ(kernel(empty, Indices({0}), Indices({0})).size_bytes(), 0u);
}

TEST(OnnxLightSliceKernel, InvalidAxesLengthsStepsAndMetadata) {
  onnx_light_cpu::SliceKernel kernel(MakeCtx());
  const Tensor data = Payload(DataType::FLOAT, {3, 4});
  const Tensor starts = Indices({0});
  const Tensor ends = Indices({3});
  for (int64_t axis : {int64_t{-3}, int64_t{2}, std::numeric_limits<int64_t>::min(),
                       std::numeric_limits<int64_t>::max()}) {
    const Tensor axes = Indices({axis});
    EXPECT_THROW(kernel(data, starts, ends, &axes), std::invalid_argument);
    EXPECT_THROW(BuiltinSlice{MakeCtx()}(data, starts, ends, &axes), std::invalid_argument);
  }
  const Tensor duplicates = Indices({0, -2});
  EXPECT_THROW(kernel(data, Indices({0, 0}), Indices({1, 1}), &duplicates), std::invalid_argument);
  const Tensor zero_step = Indices({0});
  EXPECT_THROW(kernel(data, starts, ends, nullptr, &zero_step), std::invalid_argument);
  EXPECT_THROW(kernel(Payload(DataType::FLOAT, {0}), starts, ends, nullptr, &zero_step),
               std::invalid_argument);
  const Tensor empty = Indices({});
  EXPECT_THROW(kernel(data, starts, empty), std::invalid_argument);
  EXPECT_THROW(kernel(data, starts, ends, &empty), std::invalid_argument);
  EXPECT_THROW(kernel(data, starts, ends, nullptr, &empty), std::invalid_argument);
  EXPECT_THROW(kernel(data, Indices({0, 0, 0}), Indices({1, 1, 1})), std::invalid_argument);
  EXPECT_THROW(kernel(Payload(DataType::FLOAT, {}), starts, ends), std::invalid_argument);

  for (Tensor bad : {Payload(DataType::FLOAT, {1}), Payload(DataType::INT32, {}),
                     Payload(DataType::INT64, {1, 1}), Payload(DataType::FLOAT, {0})}) {
    EXPECT_THROW(kernel(data, bad, ends), std::invalid_argument);
    EXPECT_THROW(kernel(data, starts, bad), std::invalid_argument);
    EXPECT_THROW(kernel(data, starts, ends, &bad), std::invalid_argument);
    EXPECT_THROW(kernel(data, starts, ends, nullptr, &bad), std::invalid_argument);
  }
  Tensor malformed = starts;
  malformed.shape = {-1};
  EXPECT_THROW(kernel(data, malformed, ends), std::invalid_argument);
  malformed.shape = {2};
  EXPECT_THROW(kernel(data, malformed, ends), std::invalid_argument);
  malformed.shape = {std::numeric_limits<int64_t>::max()};
  EXPECT_THROW(kernel(data, malformed, ends), std::invalid_argument);
}

TEST(OnnxLightSliceKernel, UnsupportedTypesAndCheckedDataBuffers) {
  onnx_light_cpu::SliceKernel kernel(MakeCtx());
  const Tensor starts = Indices({0});
  const Tensor ends = Indices({1});
  for (DataType type : {DataType::STRING, DataType::INT4, DataType::UINT4, DataType::FLOAT4E2M1,
                        DataType::INT2, DataType::UINT2, DataType::FLOAT6E2M3, DataType::FLOAT6E3M2,
                        DataType::COMPLEX64, DataType::COMPLEX128, DataType::UNDEFINED}) {
    Tensor data;
    data.data_type = type;
    data.shape = {1};
    EXPECT_THROW(kernel(data, starts, ends), std::invalid_argument);
    EXPECT_THROW(BuiltinSlice{MakeCtx()}(data, starts, ends), std::invalid_argument);
  }
  Tensor data;
  data.data_type = DataType::FLOAT;
  for (const Shape &shape : {Shape{-1}, Shape{1}, Shape{std::numeric_limits<int64_t>::max(), 2},
                             Shape{std::numeric_limits<int64_t>::max() / 2}}) {
    data.shape = shape;
    EXPECT_THROW(kernel(data, starts, ends), std::invalid_argument);
  }
}

TEST(OnnxLightSliceKernel, PreallocatedValidationAndNoPartialWrites) {
  onnx_light_cpu::SliceKernel kernel(MakeCtx());
  const Tensor data = Payload(DataType::FLOAT, {3, 4});
  const Tensor starts = Indices({0});
  const Tensor ends = Indices({2});
  Tensor output = Payload(DataType::FLOAT, {2, 4});
  const Tensor original = Payload(DataType::FLOAT, {2, 4});
  const Tensor invalid_step = Indices({0});
  EXPECT_THROW(kernel(data, starts, ends, nullptr, &invalid_step, output), std::invalid_argument);
  EqualBytes(output, original);
  output.data_type = DataType::INT32;
  EXPECT_THROW(kernel(data, starts, ends, nullptr, nullptr, output), std::invalid_argument);
  output = Payload(DataType::FLOAT, {8});
  EXPECT_THROW(kernel(data, starts, ends, nullptr, nullptr, output), std::invalid_argument);
  output = Payload(DataType::FLOAT, {1});
  output.shape = {2, 4};
  EXPECT_THROW(kernel(data, starts, ends, nullptr, nullptr, output), std::invalid_argument);
  output = Tensor::Borrow("output", data.data_type, {2, 4}, data.bytes() + 4, 32);
  EXPECT_THROW(kernel(data, starts, ends, nullptr, nullptr, output), std::invalid_argument);

  const Tensor integers = Payload(DataType::INT64, {1});
  const Tensor scalar_starts = Indices({0});
  const Tensor scalar_ends = Indices({1});
  const Tensor axes = Indices({0});
  const Tensor steps = Indices({1});
  for (const Tensor *input : {&scalar_starts, &scalar_ends, &axes, &steps}) {
    output = Tensor::Borrow("output", DataType::INT64, {1}, input->bytes(), sizeof(int64_t));
    EXPECT_THROW(kernel(integers, scalar_starts, scalar_ends, &axes, &steps, output),
                 std::invalid_argument);
  }
}

void AddAttribute(ONNX_LIGHT_NAMESPACE::NodeProto &node, const char *name, const Values &values) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INTS);
  for (int64_t value : values) {
    attribute->add_ints(value);
  }
}

TEST(OnnxLightSliceKernel, ModernRunMissingAxesWithPresentSteps) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Slice");
  for (const char *input : {"data", "starts", "ends", "", "steps"}) {
    node.add_input(input);
  }
  node.add_output("output");
  const Tensor data = Payload(DataType::FLOAT, {5, 3});
  const Tensor starts = Indices({-1}, DataType::INT32);
  const Tensor ends = Indices({-99}, DataType::INT32);
  const Tensor steps = Indices({-2}, DataType::INT32);
  rt_ns::RuntimeContext runtime(MakeCtx());
  runtime.Set("data", data);
  runtime.Set("starts", starts);
  runtime.Set("ends", ends);
  runtime.Set("steps", steps);
  onnx_light_cpu::SliceKernel kernel(node, MakeCtx());
  kernel.Run(runtime);
  EqualBytes(runtime.Get("output"), BuiltinSlice{MakeCtx()}(data, starts, ends, nullptr, &steps));
}

TEST(OnnxLightSliceKernel, LegacyRunAttributesAndDefaults) {
  for (int64_t opset : {1, 9}) {
    for (bool explicit_axes : {false, true}) {
      ONNX_LIGHT_NAMESPACE::NodeProto node;
      node.set_op_type("Slice");
      node.add_input("data");
      node.add_output("output");
      AddAttribute(node, "starts", {-2});
      AddAttribute(node, "ends", {99});
      if (explicit_axes) {
        AddAttribute(node, "axes", {1});
      }
      const Tensor data = Payload(DataType::FLOAT, {3, 4});
      const Tensor axes = Indices({1});
      const Tensor expected = BuiltinSlice{MakeCtx()}(data, Indices({-2}), Indices({99}),
                                                      explicit_axes ? &axes : nullptr);
      rt_ns::RuntimeContext runtime(MakeCtx(opset));
      runtime.Set("data", data);
      onnx_light_cpu::SliceKernel kernel(node, MakeCtx(opset));
      kernel.Run(runtime);
      EqualBytes(runtime.Get("output"), expected);
    }
  }
}

TEST(OnnxLightSliceKernel, InvalidNodeInputsAndLegacyAttributes) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Slice");
  node.add_input("data");
  node.add_output("output");
  rt_ns::RuntimeContext runtime(MakeCtx());
  runtime.Set("data", Payload(DataType::FLOAT, {3}));
  onnx_light_cpu::SliceKernel modern(node, MakeCtx());
  EXPECT_THROW(modern.Run(runtime), std::invalid_argument);
  onnx_light_cpu::SliceKernel legacy(node, MakeCtx(9));
  EXPECT_THROW(legacy.Run(runtime), std::invalid_argument);
  AddAttribute(node, "starts", {0});
  EXPECT_THROW(legacy.Run(runtime), std::invalid_argument);
  AddAttribute(node, "ends", {});
  EXPECT_THROW(legacy.Run(runtime), std::invalid_argument);
  for (const char *input : {"starts", "ends", "", "", "extra"}) {
    node.add_input(input);
  }
  EXPECT_THROW(modern.Run(runtime), std::invalid_argument);
}

struct InlineExecutor {
  int64_t dispatches = 0;
  int64_t blocks = 0;
  int64_t maximum_depth = 0;
  bool nested = false;

  static void Run(void *context, int64_t count, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.dispatches;
    self.blocks = count;
    for (int64_t block = 0; block < count; ++block) {
      task(task_context, block);
      if (self.nested) {
        onnx_light_cpu::detail::ExecutionRegionScope region;
        self.maximum_depth =
            std::max(self.maximum_depth,
                     static_cast<int64_t>(onnx_light_cpu::detail::ExecutionRegionDepth()));
        Compare(Payload(DataType::FLOAT, {1024, 1024}), {0}, {1024});
      }
    }
  }
};

TEST(OnnxLightSliceKernel, SmallLargeContiguousStridedAndNestedScheduling) {
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  Compare(Payload(DataType::FLOAT, {17, 9}), {1}, {16});
  EXPECT_EQ(executor.dispatches, 0);
  Compare(Payload(DataType::FLOAT, {1024, 1024}), {0}, {1024});
  EXPECT_GT(executor.dispatches, 0);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_LE(executor.blocks, view.effective_threads);
  const int64_t previous = executor.dispatches;
  Compare(Payload(DataType::FLOAT, {1025, 1025}), {-1}, {-9999}, Values{-1}, Values{-2});
  EXPECT_GT(executor.dispatches, previous);
  executor.nested = true;
  Compare(Payload(DataType::FLOAT, {1024, 1024}), {0}, {1024}, Values{0}, Values{2});
  EXPECT_EQ(executor.maximum_depth, 1);
  const int64_t calls = executor.dispatches;
  {
    onnx_light_cpu::detail::ExecutionRegionScope region;
    Compare(Payload(DataType::FLOAT, {1024, 1024}), {-1}, {-9999}, Values{0}, Values{-1});
  }
  EXPECT_EQ(executor.dispatches, calls);
}

} // namespace
