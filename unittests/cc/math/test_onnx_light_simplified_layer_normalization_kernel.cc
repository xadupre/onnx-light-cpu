// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/simplified_layer_normalization_reference.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/kernels/math/rms_normalization_kernel.h"
#include "onnx_light_cpu/kernels/math/simplified_layer_normalization_kernel.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {

namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace builtin = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel;
using rt::DataType;
using rt::Shape;
using rt::Tensor;

rt::KernelContext Context() { return rt::KernelContext(rt::DefaultOpset(23)); }

Tensor Payload(DataType type, const Shape &shape) {
  std::vector<float> values(static_cast<std::size_t>(shape.product()));
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i % 17) * 0.25F - 1.5F;
  }
  return builtin::Cast(Context())(Tensor::FromFloat("input", shape, values),
                                  static_cast<std::int32_t>(type));
}

void Equal(const Tensor &actual, const Tensor &expected) {
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.data_type, expected.data_type);
  const builtin::Cast cast(Context());
  const auto a = cast(actual, static_cast<std::int32_t>(DataType::DOUBLE));
  const auto e = cast(expected, static_cast<std::int32_t>(DataType::DOUBLE));
  const double tolerance = actual.data_type == DataType::BFLOAT16  ? 0.01
                           : actual.data_type == DataType::FLOAT16 ? 0.002
                                                                   : 1.0e-6;
  for (std::size_t i = 0; i < a.size_bytes() / sizeof(double); ++i) {
    if (std::isnan(e.AsDouble()[i])) {
      EXPECT_TRUE(std::isnan(a.AsDouble()[i])) << i;
    } else if (std::isinf(e.AsDouble()[i])) {
      EXPECT_EQ(a.AsDouble()[i], e.AsDouble()[i]) << i;
    } else {
      EXPECT_NEAR(a.AsDouble()[i], e.AsDouble()[i], tolerance * (1.0 + std::abs(e.AsDouble()[i])))
          << i;
    }
  }
}

void Compare(const Tensor &x, const Tensor &scale, std::int64_t axis = -1, float epsilon = 1.0e-6F,
             std::int64_t stash = 1) {
  const auto expected = onnx_light_cpu::backend_test::ReferenceSimplifiedLayerNormalization(
      x, scale, axis, epsilon, stash, true, Context());
  const onnx_light_cpu::SimplifiedLayerNormalizationKernel kernel(Context());
  const auto result = kernel(x, scale, axis, epsilon, stash, true);
  Equal(result.y, expected[0]);
  Equal(result.inv_std_var, expected[1]);
  const auto y_only = kernel(x, scale, axis, epsilon, stash);
  Equal(y_only.y, result.y);
}

TEST(OnnxLightSimplifiedLayerNormalizationKernel, TypesTailsAndStatistics) {
  for (auto x_type : {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (auto scale_type :
         {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
      for (std::int64_t width : {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129}) {
        for (std::int64_t stash : {1, 11}) {
          Compare(Payload(x_type, {3, width}), Payload(scale_type, {width}), -1, 1.0e-6F, stash);
        }
      }
    }
  }
}

TEST(OnnxLightSimplifiedLayerNormalizationKernel, BroadcastAndMiddleAxes) {
  for (const Shape &shape : {Shape{}, Shape{4}, Shape{1, 4}, Shape{2, 1, 4}, Shape{2, 3, 4}}) {
    for (std::int64_t axis : {0, 1, -1, -2, -3}) {
      Compare(Payload(DataType::FLOAT, {2, 3, 4}), Payload(DataType::FLOAT, shape), axis);
    }
  }
}

TEST(OnnxLightSimplifiedLayerNormalizationKernel, DoubleDoesNotRoundThroughFloatStash) {
  const auto x = Tensor::FromDouble("X", {1, 3}, {1.0e100, 2.0e100, 3.0e100});
  const auto scale = Tensor::FromDouble("scale", {3}, {1.0, 1.0, 1.0});
  Compare(x, scale);
  const auto result = onnx_light_cpu::SimplifiedLayerNormalizationKernel(Context())(x, scale);
  EXPECT_GT(result.y.AsDouble()[0], 0.4);
  EXPECT_LT(result.y.AsDouble()[0], 0.5);
}

TEST(OnnxLightSimplifiedLayerNormalizationKernel, LowPrecisionRoundsOnlyAfterScale) {
  const builtin::Cast cast(Context());
  for (auto type : {DataType::FLOAT16, DataType::BFLOAT16}) {
    const auto x = cast(Tensor::FromFloat("X", {1, 4}, {-3.0F, -2.0F, -1.0F, -0.5F}),
                        static_cast<std::int32_t>(type));
    const auto scale = cast(Tensor::FromFloat("scale", {4}, {0.3F, 0.3F, 0.3F, 0.3F}),
                            static_cast<std::int32_t>(type));
    const auto expected = onnx_light_cpu::backend_test::ReferenceSimplifiedLayerNormalization(
        x, scale, -1, 1.0e-5F, 1, false, Context());
    const auto result = onnx_light_cpu::SimplifiedLayerNormalizationKernel(Context())(x, scale);
    EXPECT_EQ(std::memcmp(result.y.bytes(), expected[0].bytes(), result.y.size_bytes()), 0);
    const auto rms = onnx_light_cpu::RmsNormalizationKernel(Context())(x, scale);
    EXPECT_NE(std::memcmp(result.y.bytes(), rms.bytes(), result.y.size_bytes()), 0);
  }
}

TEST(OnnxLightSimplifiedLayerNormalizationKernel, EmptyOuterAndInvalidInputs) {
  const onnx_light_cpu::SimplifiedLayerNormalizationKernel kernel(Context());
  const auto scale = Payload(DataType::FLOAT, {4});
  const auto empty = kernel(Payload(DataType::FLOAT, {0, 4}), scale, -1, 1.0e-6F, 1, true);
  EXPECT_EQ(empty.y.shape, (Shape{0, 4}));
  EXPECT_EQ(empty.inv_std_var.shape, (Shape{0, 1}));
  EXPECT_EQ(empty.y.size_bytes(), 0);
  const auto x = Payload(DataType::FLOAT, {2, 4});
  EXPECT_THROW(kernel(Payload(DataType::FLOAT, {2, 0}), Payload(DataType::FLOAT, {0})),
               std::invalid_argument);
  EXPECT_THROW(kernel(Payload(DataType::FLOAT, {}), Payload(DataType::FLOAT, {})),
               std::invalid_argument);
  EXPECT_THROW(kernel(x, scale, 2), std::invalid_argument);
  EXPECT_THROW(kernel(x, scale, -3), std::invalid_argument);
  EXPECT_THROW(kernel(x, scale, -1, 1.0e-6F, 10), std::invalid_argument);
  EXPECT_THROW(kernel(x, Payload(DataType::FLOAT, {8})), std::invalid_argument);
  EXPECT_THROW(kernel(x, Payload(DataType::INT32, {4})), std::invalid_argument);
  auto malformed = x;
  malformed.shape = {3, 4};
  EXPECT_THROW(kernel(malformed, scale), std::invalid_argument);
  malformed.shape = {-1, 4};
  EXPECT_THROW(kernel(malformed, scale), std::invalid_argument);
  malformed.shape = {std::numeric_limits<std::int64_t>::max(), 4};
  EXPECT_THROW(kernel(malformed, scale), std::invalid_argument);
}

TEST(OnnxLightSimplifiedLayerNormalizationKernel, UnalignedBorrowedInputs) {
  const auto x = Payload(DataType::FLOAT, {2, 9});
  const auto scale = Payload(DataType::FLOAT, {9});
  std::vector<std::uint8_t> x_storage(x.size_bytes() + 1), s_storage(scale.size_bytes() + 1);
  std::memcpy(x_storage.data() + 1, x.bytes(), x.size_bytes());
  std::memcpy(s_storage.data() + 1, scale.bytes(), scale.size_bytes());
  const auto borrowed_x =
      Tensor::Borrow("X", DataType::FLOAT, x.shape, x_storage.data() + 1, x.size_bytes());
  const auto borrowed_scale = Tensor::Borrow("scale", DataType::FLOAT, scale.shape,
                                             s_storage.data() + 1, scale.size_bytes());
  const onnx_light_cpu::SimplifiedLayerNormalizationKernel kernel(Context());
  const auto result = kernel(borrowed_x, borrowed_scale, -1, 1.0e-6F, 1, true);
  const auto expected = kernel(x, scale, -1, 1.0e-6F, 1, true);
  Equal(result.y, expected.y);
  Equal(result.inv_std_var, expected.inv_std_var);
}

TEST(OnnxLightSimplifiedLayerNormalizationKernel, IeeeValuesAndEpsilon) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const auto scale = Tensor::FromFloat("scale", {4}, {1.0F, 1.0F, 1.0F, 1.0F});
  for (float epsilon : {0.0F, -1.0F, nan, inf}) {
    Compare(Tensor::FromFloat("X", {1, 4}, {1.0F, 1.0F, 1.0F, 1.0F}), scale, -1, epsilon);
  }
  Compare(Tensor::FromFloat(
              "X", {3, 4}, {nan, 1.0F, 2.0F, 3.0F, inf, 1.0F, 2.0F, 3.0F, -0.0F, 0.0F, 0.0F, 0.0F}),
          scale);
}

TEST(OnnxLightSimplifiedLayerNormalizationKernel, RuntimeDefaultsAndEmptyOptionalOutput) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("SimplifiedLayerNormalization");
  node.add_input("X");
  node.add_input("scale");
  node.add_output("Y");
  node.add_output("");
  rt::RuntimeContext context(Context());
  const auto x = Payload(DataType::FLOAT, {2, 4});
  const auto scale = Payload(DataType::FLOAT, {4});
  context.Set("X", x);
  context.Set("scale", scale);
  onnx_light_cpu::SimplifiedLayerNormalizationKernel kernel(Context());
  kernel.set_node(node);
  kernel.Run(context);
  Equal(context.Get("Y"), kernel(x, scale).y);
}

struct InlineExecutor {
  std::int64_t calls = 0;
  std::int64_t blocks = 0;
  static void Run(void *context, std::int64_t count, void *task_context,
                  onnx_light_cpu::ExecutionBlockFn task) {
    auto &self = *static_cast<InlineExecutor *>(context);
    ++self.calls;
    self.blocks = count;
    for (std::int64_t i = 0; i < count; ++i) {
      task(task_context, i);
    }
  }
};

TEST(OnnxLightSimplifiedLayerNormalizationKernel, RuntimeOwnedSchedulingAndNestedSuppression) {
  InlineExecutor executor;
  onnx_light_cpu::ExecutionExecutorView view{&executor, 8, &InlineExecutor::Run};
  onnx_light_cpu::ExecutionExecutorScope scope(&view);
  Compare(Payload(DataType::FLOAT, {1, 128}), Payload(DataType::FLOAT, {128}));
  EXPECT_EQ(executor.calls, 0);
  Compare(Payload(DataType::FLOAT, {256, 1024}), Payload(DataType::FLOAT, {1024}));
  EXPECT_GT(executor.calls, 0);
  EXPECT_GT(executor.blocks, 1);
  EXPECT_LE(executor.blocks, 256);
  const auto previous = executor.calls;
  {
    onnx_light_cpu::detail::ExecutionRegionScope nested;
    Compare(Payload(DataType::FLOAT, {256, 1024}), Payload(DataType::FLOAT, {1024}));
  }
  EXPECT_EQ(executor.calls, previous);
  Compare(Payload(DataType::FLOAT16, {256, 1024}), Payload(DataType::FLOAT16, {1024}), -1, 1.0e-6F,
          11);
  EXPECT_GT(executor.calls, previous);
}

} // namespace
