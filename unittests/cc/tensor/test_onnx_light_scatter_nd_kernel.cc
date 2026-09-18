// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/tensor/scatter_nd_kernel.h"

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_light_cpu/impl/execution.h"

#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;
using BuiltinScatterND = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::ScatterND;
using onnx_light_cpu::ScatterNDKernel;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::DefaultOpset(18)); }

Tensor Payload(int32_t type, const Shape &shape, int seed = 0) {
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

void Compare(const Tensor &data, const Shape &index_shape, const std::vector<int64_t> &values,
             const Tensor &updates) {
  const auto ctx = MakeCtx();
  const ScatterNDKernel kernel(ctx);
  const BuiltinScatterND reference(ctx);
  const Tensor indices = Tensor::FromInt64("indices", index_shape, values);
  const Tensor expected = reference(data, indices, updates, {});
  EqualBytes(kernel(data, indices, updates), expected);
  std::vector<int32_t> values32(values.begin(), values.end());
  const Tensor indices32 = Tensor::FromInt32("indices", index_shape, values32);
  EqualBytes(kernel(data, indices32, updates), expected);
  Tensor output = Payload(data.data_type, data.shape, 255);
  kernel(data, indices32, updates, output);
  EqualBytes(output, expected);
}

TEST(OnnxLightScatterNDKernel, BitwisePayloadsScalarSlicesAndTupleGrids) {
  for (const DataType type :
       {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16, DataType::INT8,
        DataType::UINT8, DataType::INT16, DataType::UINT16, DataType::INT32, DataType::UINT32,
        DataType::INT64, DataType::UINT64, DataType::BOOL, DataType::FLOAT8E4M3FN,
        DataType::FLOAT8E4M3FNUZ, DataType::FLOAT8E5M2, DataType::FLOAT8E5M2FNUZ,
        DataType::FLOAT8E8M0}) {
    SCOPED_TRACE(static_cast<int>(type));
    Compare(Payload(type, {3, 4}), {2}, {-1, -4}, Payload(type, {}, 11));
    Compare(Payload(type, {3, 4}), {2, 2}, {0, 1, 2, -1}, Payload(type, {2}, 11));
    Compare(Payload(type, {3, 4, 5}), {2, 1}, {-3, -1}, Payload(type, {2, 4, 5}, 11));
    Compare(Payload(type, {3, 4, 5}), {2, 1, 2}, {0, 1, 2, -1}, Payload(type, {2, 1, 5}, 11));
    Compare(Payload(type, {3, 4}), {0, 1}, {}, Payload(type, {0, 4}));
    Compare(Payload(type, {0, 4}), {0, 1}, {}, Payload(type, {0, 4}));
    Compare(Payload(type, {3, 0}), {2, 1}, {0, -1}, Payload(type, {2, 0}));
    Compare(Payload(type, {16, 6656}), {3, 1}, {0, -1, 7}, Payload(type, {3, 6656}, 11));
  }
}

TEST(OnnxLightScatterNDKernel, DuplicatesUseLastTupleIncludingNegativeAliases) {
  const auto ctx = MakeCtx();
  const ScatterNDKernel kernel(ctx);
  const Tensor data = Tensor::FromFloat("data", {3, 2}, {0, 1, 2, 3, 4, 5});
  const Tensor updates = Tensor::FromFloat("updates", {3, 2}, {10, 11, 20, 21, 30, 31});
  const Tensor indices = Tensor::FromInt64("indices", {3, 1}, {1, -2, 1});
  EqualBytes(kernel(data, indices, updates),
             Tensor::FromFloat("expected", {3, 2}, {0, 1, 30, 31, 4, 5}));
  Compare(data, {3, 1}, {1, -2, 1}, updates);
}

TEST(OnnxLightScatterNDKernel, RejectsRanksShapesTypesReductionsAndMalformedBuffers) {
  const auto ctx = MakeCtx();
  const ScatterNDKernel kernel(ctx);
  const Tensor data = Payload(DataType::FLOAT, {3, 2});
  const Tensor indices = Tensor::FromInt64("indices", {1, 1}, {0});
  const Tensor updates = Payload(DataType::FLOAT, {1, 2});
  for (const char *reduction : {"add", "mul", "min", "max", "", "invalid"}) {
    EXPECT_THROW(ScatterNDKernel(ctx, reduction), std::invalid_argument);
    ONNX_LIGHT_NAMESPACE::NodeProto node;
    auto *attr = node.add_attribute();
    attr->set_name("reduction");
    attr->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::STRING);
    attr->set_s(reduction);
    EXPECT_THROW(ScatterNDKernel(node, ctx), std::invalid_argument);
  }
  EXPECT_THROW(kernel(Payload(DataType::FLOAT, {}), indices, updates), std::invalid_argument);
  EXPECT_THROW(kernel(data, Tensor::FromInt64("indices", {}, {0}), updates), std::invalid_argument);
  EXPECT_THROW(kernel(data, Tensor::FromInt64("indices", {1, 0}, {}), updates),
               std::invalid_argument);
  EXPECT_THROW(kernel(data, Tensor::FromInt64("indices", {1, 3}, {0, 0, 0}), updates),
               std::invalid_argument);
  EXPECT_THROW(kernel(data, Payload(DataType::FLOAT, {1, 1}), updates), std::invalid_argument);
  EXPECT_THROW(kernel(data, indices, Payload(DataType::FLOAT16, {1, 2})), std::invalid_argument);
  EXPECT_THROW(kernel(data, indices, Payload(DataType::FLOAT, {2})), std::invalid_argument);
  for (const Shape &shape : {Shape{-1, 2}, Shape{std::numeric_limits<int64_t>::max(), 2},
                             Shape{std::numeric_limits<int64_t>::max() / 2, 2}, Shape{4, 2}}) {
    Tensor malformed = data;
    malformed.shape = shape;
    EXPECT_THROW(kernel(malformed, indices, updates), std::invalid_argument);
  }
  Tensor malformed = indices;
  malformed.shape = {1, 2};
  EXPECT_THROW(kernel(data, malformed, Payload(DataType::FLOAT, {1})), std::invalid_argument);
  malformed = updates;
  malformed.shape = {2, 2};
  EXPECT_THROW(kernel(data, Tensor::FromInt64("indices", {2, 1}, {0, 1}), malformed),
               std::invalid_argument);
  for (const DataType type : {DataType::STRING, DataType::COMPLEX64, DataType::INT4}) {
    Tensor unsupported = data;
    unsupported.data_type = type;
    Tensor unsupported_updates = updates;
    unsupported_updates.data_type = type;
    EXPECT_THROW(kernel(unsupported, indices, unsupported_updates), std::invalid_argument);
  }
}

TEST(OnnxLightScatterNDKernel, BoundsFailureNeverWritesEvenWithEmptySlices) {
  const auto ctx = MakeCtx();
  const ScatterNDKernel kernel(ctx);
  for (int64_t width : {0, 2}) {
    const Tensor data = Payload(DataType::FLOAT, {3, width});
    const Tensor updates = Payload(DataType::FLOAT, {2, width}, 11);
    for (int64_t bad : {std::numeric_limits<int64_t>::min(), int64_t{-4}, int64_t{3},
                        std::numeric_limits<int64_t>::max()}) {
      Tensor output = Payload(DataType::FLOAT, {3, width}, 55);
      const Tensor original = Payload(DataType::FLOAT, {3, width}, 55);
      const Tensor indices = Tensor::FromInt64("indices", {2, 1}, {0, bad});
      EXPECT_THROW(kernel(data, indices, updates, output), std::invalid_argument);
      EqualBytes(output, original);
    }
    for (int32_t bad : {std::numeric_limits<int32_t>::min(), int32_t{-4}, int32_t{3},
                        std::numeric_limits<int32_t>::max()}) {
      EXPECT_THROW(kernel(data, Tensor::FromInt32("indices", {2, 1}, {0, bad}), updates),
                   std::invalid_argument);
    }
  }
  EXPECT_THROW(kernel(Payload(DataType::FLOAT, {0, 2}), Tensor::FromInt64("indices", {1, 1}, {0}),
                      Payload(DataType::FLOAT, {1, 2})),
               std::invalid_argument);
}

TEST(OnnxLightScatterNDKernel, OutputValidationAndAllInputOverlaps) {
  const auto ctx = MakeCtx();
  const ScatterNDKernel kernel(ctx);
  const Tensor data = Payload(DataType::FLOAT, {2, 2});
  const Tensor indices = Tensor::FromInt64("indices", {2, 1}, {1, 0});
  const Tensor updates = Payload(DataType::FLOAT, {2, 2}, 11);
  for (const Tensor *input : {&data, &indices, &updates}) {
    Tensor alias = Tensor::Borrow("output", DataType::FLOAT, data.shape, input->bytes(), 16);
    EXPECT_THROW(kernel(data, indices, updates, alias), std::invalid_argument);
  }
  Tensor storage = Payload(DataType::FLOAT, {8});
  for (int offset : {0, 8}) {
    const Tensor input =
        Tensor::Borrow("data", DataType::FLOAT, {2, 2}, storage.bytes() + offset, 16);
    Tensor alias =
        Tensor::Borrow("output", DataType::FLOAT, {2, 2}, storage.bytes() + (8 - offset), 16);
    EXPECT_THROW(kernel(input, indices, updates, alias), std::invalid_argument);
  }
  for (Tensor output : {Payload(DataType::FLOAT16, {2, 2}), Payload(DataType::FLOAT, {4})}) {
    EXPECT_THROW(kernel(data, indices, updates, output), std::invalid_argument);
  }
  Tensor short_output = Payload(DataType::FLOAT, {3});
  short_output.shape = data.shape;
  EXPECT_THROW(kernel(data, indices, updates, short_output), std::invalid_argument);
  // Inputs may alias each other; only the independently owned output is written.
  Compare(data, {2, 1}, {1, 0}, data);
}

TEST(OnnxLightScatterNDKernel, OwnedOutputOutlivesInputsAndEmptyUpdatesStillCopy) {
  const auto ctx = MakeCtx();
  const ScatterNDKernel kernel(ctx);
  EXPECT_FALSE(ScatterNDKernel::CanRunInPlace());
  Tensor result;
  {
    Tensor data = Tensor::FromFloat("data", {3}, {1, 2, 3});
    Tensor updates = Tensor::FromFloat("updates", {1}, {99});
    result = kernel(data, Tensor::FromInt64("indices", {1, 1}, {1}), updates);
    EXPECT_NE(result.bytes(), data.bytes());
    EXPECT_NE(result.bytes(), updates.bytes());
    EqualBytes(data, Tensor::FromFloat("original", {3}, {1, 2, 3}));
    data.As<float>()[0] = -1;
    updates.As<float>()[0] = -1;
    Tensor copy = kernel(data, Tensor::FromInt64("indices", {0, 1}, {}),
                         Tensor::FromFloat("updates", {0}, {}));
    EXPECT_NE(copy.bytes(), data.bytes());
    EqualBytes(copy, data);
    copy.As<float>()[0] = 10;
    EXPECT_EQ(data.As<float>()[0], -1);
  }
  EqualBytes(result, Tensor::FromFloat("expected", {3}, {1, 99, 3}));
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

TEST(OnnxLightScatterNDKernel, CopySchedulingIsRuntimeOwnedAndNestedCallsStaySerial) {
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 4, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  Compare(Payload(DataType::FLOAT, {3, 2}), {1, 1}, {1}, Payload(DataType::FLOAT, {1, 2}, 11));
  EXPECT_EQ(executor.dispatches, 0);
  Compare(Payload(DataType::FLOAT, {32, 6656}), {2, 1}, {1, -31},
          Payload(DataType::FLOAT, {2, 6656}, 11));
  EXPECT_GT(executor.dispatches, 0);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_LE(executor.blocks, view.effective_threads);
  const int64_t calls = executor.dispatches;
  {
    onnx_light_cpu::detail::ExecutionRegionScope region;
    Compare(Payload(DataType::FLOAT, {32, 6656}), {1, 1}, {1},
            Payload(DataType::FLOAT, {1, 6656}, 11));
  }
  EXPECT_EQ(executor.dispatches, calls);
}

} // namespace
