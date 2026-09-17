// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/math/simplified_layer_normalization_reference.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/normalization_kernel.h"
#include "onnx_light_cpu/kernels/math/rms_normalization_kernel.h"
#include "onnx_light_cpu/kernels/math/simplified_layer_normalization_kernel.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
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
  for (auto type : {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (const Shape &shape : {Shape{}, Shape{17}, Shape{1, 17}, Shape{2, 1, 17}, Shape{2, 3, 17},
                               Shape{2, 3, 1}, Shape{3, 1}, Shape{2, 1, 1}}) {
      for (std::int64_t axis : {0, 1, -1, -2, -3}) {
        for (std::int64_t stash : {1, 11}) {
          Compare(Payload(type, {2, 3, 17}), Payload(type, shape), axis, 1.0e-6F, stash);
        }
      }
    }
  }
}

TEST(OnnxLightSimplifiedLayerNormalizationKernel,
     StashControlsAccumulationWithAndWithoutStatistics) {
  const onnx_light_cpu::SimplifiedLayerNormalizationKernel kernel(Context());
  for (auto type : {DataType::FLOAT, DataType::DOUBLE}) {
    for (std::int64_t width : {1, 7, 8, 9, 16, 33, 65}) {
      const auto x = builtin::Cast(Context())(
          Tensor::FromFloat("X", {2, width}, std::vector<float>(2 * width, 1.0e20F)),
          static_cast<std::int32_t>(type));
      for (const Shape &shape : {Shape{width}, Shape{2, 1}, Shape{2, width}}) {
        const auto scale = builtin::Cast(Context())(
            Tensor::FromFloat("scale", shape, std::vector<float>(shape.product(), 1.0F)),
            static_cast<std::int32_t>(type));
        for (bool statistics : {false, true}) {
          const auto fp32 = kernel(x, scale, -1, 0.0F, 1, statistics);
          const auto fp64 = kernel(x, scale, -1, 0.0F, 11, statistics);
          const auto y32 = builtin::Cast(Context())(fp32.y, 11);
          const auto y64 = builtin::Cast(Context())(fp64.y, 11);
          EXPECT_EQ(y32.AsDouble()[0], 0.0);
          EXPECT_NEAR(y64.AsDouble()[0], 1.0, 1.0e-12);
          if (statistics) {
            EXPECT_EQ(fp32.inv_std_var.AsFloat()[0], 0.0F);
            EXPECT_GT(fp64.inv_std_var.AsDouble()[0], 0.0);
          }
        }
        Compare(x, scale, -1, 0.0F, 1);
        Compare(x, scale, -1, 0.0F, 11);
      }
    }
  }
}

TEST(OnnxLightSimplifiedLayerNormalizationKernel, DoubleStashRetainsInputAndScalePrecision) {
  const onnx_light_cpu::SimplifiedLayerNormalizationKernel kernel(Context());
  const double value = 1.0 + 0x1p-30;
  for (std::int64_t width : {1, 7, 8, 9, 32, 33}) {
    const auto x = Tensor::FromDouble("X", {1, width}, std::vector<double>(width, value));
    const auto scale = Tensor::FromDouble("scale", {width}, std::vector<double>(width, value));
    const auto fp32 = kernel(x, scale, -1, 0.0F, 1, true);
    const auto fp64 = kernel(x, scale, -1, 0.0F, 11, true);
    EXPECT_EQ(fp32.y.AsDouble()[0], 1.0);
    EXPECT_EQ(fp32.inv_std_var.AsFloat()[0], 1.0F);
    EXPECT_NEAR(fp64.y.AsDouble()[0], value, 1.0e-15);
    EXPECT_NEAR(fp64.inv_std_var.AsDouble()[0], 1.0 / value, 1.0e-15);
  }
}

TEST(OnnxLightSimplifiedLayerNormalizationKernel, LowPrecisionRoundsOnlyAfterScale) {
  const builtin::Cast cast(Context());
  for (auto type : {DataType::FLOAT16, DataType::BFLOAT16}) {
    for (std::int64_t width : {4, 8, 32, 64}) {
      const float pattern[] = {-3.0F, -2.0F, -1.0F, -0.5F};
      std::vector<float> values(width);
      for (std::int64_t i = 0; i < width; ++i) {
        values[i] = pattern[i % 4];
      }
      const auto x =
          cast(Tensor::FromFloat("X", {1, width}, values), static_cast<std::int32_t>(type));
      const auto scale = cast(Tensor::FromFloat("scale", {width}, std::vector<float>(width, 0.3F)),
                              static_cast<std::int32_t>(type));
      const auto expected = onnx_light_cpu::backend_test::ReferenceSimplifiedLayerNormalization(
          x, scale, -1, 1.0e-5F, 1, false, Context());
      const auto result = onnx_light_cpu::SimplifiedLayerNormalizationKernel(Context())(x, scale);
      EXPECT_EQ(std::memcmp(result.y.bytes(), expected[0].bytes(), result.y.size_bytes()), 0);
      const auto rms = onnx_light_cpu::RmsNormalizationKernel(Context())(x, scale);
      EXPECT_NE(std::memcmp(result.y.bytes(), rms.bytes(), result.y.size_bytes()), 0);
    }
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
  for (auto type : {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    const auto x = Payload(type, {2, 3, 9});
    for (const Shape &shape : {Shape{9}, Shape{2, 1, 9}}) {
      const auto scale = Payload(type, shape);
      std::vector<std::uint8_t> x_storage(x.size_bytes() + 1), s_storage(scale.size_bytes() + 1);
      std::memcpy(x_storage.data() + 1, x.bytes(), x.size_bytes());
      std::memcpy(s_storage.data() + 1, scale.bytes(), scale.size_bytes());
      const auto borrowed_x =
          Tensor::Borrow("X", type, x.shape, x_storage.data() + 1, x.size_bytes());
      const auto borrowed_scale =
          Tensor::Borrow("scale", type, scale.shape, s_storage.data() + 1, scale.size_bytes());
      for (std::int64_t stash : {1, 11}) {
        const onnx_light_cpu::SimplifiedLayerNormalizationKernel kernel(Context());
        const auto result = kernel(borrowed_x, borrowed_scale, -1, 1.0e-6F, stash, true);
        const auto expected = kernel(x, scale, -1, 1.0e-6F, stash, true);
        Equal(result.y, expected.y);
        Equal(result.inv_std_var, expected.inv_std_var);
      }
    }
  }
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
  for (auto type : {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (float value : {nan, inf, -inf, 0.0F, -0.0F}) {
      const auto x =
          builtin::Cast(Context())(Tensor::FromFloat("X", {2, 17}, std::vector<float>(34, value)),
                                   static_cast<std::int32_t>(type));
      for (std::int64_t stash : {1, 11}) {
        Compare(x, Payload(type, {17}), -1, 1.0e-6F, stash);
      }
    }
  }
}

TEST(OnnxLightSimplifiedLayerNormalizationKernel, RecordsSelectedPathAndBroadcastLayout) {
  using namespace onnx_light_cpu;
  for (auto type : {DataType::FLOAT, DataType::FLOAT16, DataType::DOUBLE, DataType::BFLOAT16}) {
    for (std::int64_t axis : {-1, 1}) {
      rt::RuntimeContext context(Context());
      context.set_kernel_usage_enabled(true);
      const auto x = Payload(type, {2, 3, 17});
      const auto scale = Payload(type, {2, 1, 17});
      const SimplifiedLayerNormalizationKernel kernel(Context());
      kernel(x, scale, axis, 1.0e-6F, 1, false, &context);
      const auto paths = context.GetKernelUsage();
      ASSERT_EQ(paths.size(), 1);
      const std::string path = type == DataType::FLOAT     ? "float32/normalization"
                               : type == DataType::FLOAT16 ? NormalizationFloat16Path()
                               : type == DataType::DOUBLE  ? NormalizationFloat64Path()
                                                           : "generic";
      EXPECT_EQ(paths[0], std::string(SimplifiedLayerNormalizationKernel::kName) + "/" + path +
                              (axis == -1 ? "/row-scale" : "/broadcast-blocks"));
      context.ClearKernelUsage();
      context.set_kernel_usage_enabled(false);
      kernel(x, scale, axis, 1.0e-6F, 1, false, &context);
      EXPECT_TRUE(context.GetKernelUsage().empty());
    }
  }
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
  for (auto type : {DataType::FLOAT16, DataType::DOUBLE, DataType::FLOAT}) {
    const auto calls = executor.calls;
    Compare(Payload(type, {4, 64, 1025}), Payload(type, {4, 1, 1025}));
    EXPECT_GT(executor.calls, calls);
    const auto parallel_calls = executor.calls;
    onnx_light_cpu::detail::ExecutionRegionScope nested;
    Compare(Payload(type, {4, 64, 1025}), Payload(type, {4, 1, 1025}));
    EXPECT_EQ(executor.calls, parallel_calls);
  }
}

} // namespace
