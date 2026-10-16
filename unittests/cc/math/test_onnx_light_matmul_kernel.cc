#include "onnx_light_cpu/kernels/math/matmul_kernel.h"

#include "onnx_light_cpu/impl/math/gemm/gemm_plan.h"
#include "unittests/cc/heap_allocation_tracker.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

void CheckBfloat16BoundedConversion(std::size_t m, std::size_t n, std::size_t k,
                                    onnx_light_cpu::GemmAlgorithm algorithm) {
  const rt_ns::Shape a_shape{static_cast<std::int64_t>(m), static_cast<std::int64_t>(k)};
  const rt_ns::Shape b_shape{static_cast<std::int64_t>(k), static_cast<std::int64_t>(n)};
  std::vector<float> a(m * k), b(k * n);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 9) - 4) / 8.0f;
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 8.0f;
  }

  const onnx_light_cpu::GemmHalfPlan plan({true, false, false, m, n, k});
  ASSERT_EQ(plan.algorithm(), algorithm);
  onnx_light_cpu::MatMulKernel kernel(MakeCtx());
  const auto expected = kernel(rt_ns::Tensor::From<float>("A", a_shape, a),
                               rt_ns::Tensor::From<float>("B", b_shape, b));

  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("MatMul");
  node.add_input("A");
  node.add_input("B");
  node.add_output("Y");
  kernel.set_node(node);
  rt_ns::RuntimeContext runtime(MakeCtx());
  runtime.Set("A", rt_ns::MakeBfloat16Tensor("A", a_shape, a));
  runtime.Set("B", rt_ns::MakeBfloat16Tensor("B", b_shape, b));
  // Measure the actual workspace allocation, including STL alignment bookkeeping
  // (MSVC adds overhead to large vector allocations).
  const std::size_t workspace_peak = PeakAllocationBytes([&] {
    rt_ns::detail::TemporaryTypedBuffer<float> workspace(m * n, nullptr, "MatMul FP32 workspace");
    EXPECT_NE(workspace.data(), nullptr);
  });
  const std::size_t peak = PeakAllocationBytes([&] { kernel.Run(runtime); });

  // The output-sized FP32 workspace is necessary; neither full input widening
  // (both operands for general, the large B for direct) fits this budget.
  const std::size_t workspace_bytes = m * n * sizeof(float);
  ASSERT_GE(workspace_peak, workspace_bytes);
  EXPECT_LE(peak, workspace_peak);
  ASSERT_GT(k * n * sizeof(float), workspace_peak);
  if (algorithm == onnx_light_cpu::GemmAlgorithm::kGeneral) {
    ASSERT_GT(m * k * sizeof(float), workspace_peak);
  }
  const auto &actual = runtime.Get("Y");
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.data_type, static_cast<std::int32_t>(rt_ns::DataType::BFLOAT16));
  const auto *actual_bits = reinterpret_cast<const std::uint16_t *>(actual.bytes());
  for (std::size_t i = 0; i < m * n; ++i) {
    ASSERT_FLOAT_EQ(rt_ns::Bfloat16BitsToFloat(actual_bits[i]),
                    rt_ns::Bfloat16BitsToFloat(rt_ns::FloatToBfloat16Bits(expected.AsFloat()[i])))
        << i;
  }
}

TEST(MatMulKernel, Bfloat16GeneralUsesBoundedConversion) {
  CheckBfloat16BoundedConversion(512, 512, 1024, onnx_light_cpu::GemmAlgorithm::kGeneral);
}

TEST(MatMulKernel, Bfloat16DirectUsesBoundedConversion) {
  CheckBfloat16BoundedConversion(8, 4097, 32, onnx_light_cpu::GemmAlgorithm::kDirect);
}

TEST(MatMulKernel, BroadcastsBatches) {
  onnx_light_cpu::MatMulKernel kernel(MakeCtx());
  const auto a = rt_ns::Tensor::From<float>("a", {2, 1, 2}, {1, 2, 3, 4});
  const auto b = rt_ns::Tensor::From<float>("b", {1, 2, 2}, {5, 6, 7, 8});

  const auto y = kernel(a, b);

  EXPECT_EQ(y.shape, (rt_ns::Shape{2, 1, 2}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 19.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 22.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 43.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[3], 50.0f);
}

TEST(MatMulKernel, SupportsVectorDot) {
  onnx_light_cpu::MatMulKernel kernel(MakeCtx());
  const auto vector_a = rt_ns::Tensor::From<float>("va", {2}, {2, 3});
  const auto vector_b = rt_ns::Tensor::From<float>("vb", {2}, {4, 5});
  const auto dot = kernel(vector_a, vector_b);
  EXPECT_TRUE(dot.shape.empty());
  EXPECT_FLOAT_EQ(dot.AsFloat()[0], 23.0f);
}

TEST(MatMulKernel, RuntimeCacheReusesShapesAndInvalidatesOnShapeChange) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("MatMul");
  node.add_input("A");
  node.add_input("B");
  node.add_output("Y");
  rt_ns::RuntimeContext runtime(MakeCtx());
  runtime.Set("A", rt_ns::Tensor::From<float>("A", {2, 2}, {1, 2, 3, 4}));
  runtime.Set("B", rt_ns::Tensor::From<float>("B", {2, 2}, {1, 0, 0, 1}));

  onnx_light_cpu::MatMulKernel kernel(MakeCtx());
  kernel.set_node(node);
  kernel.Run(runtime);
  EXPECT_FLOAT_EQ(runtime.Get("Y").AsFloat()[3], 4.0f);

  rt_ns::RuntimeContext same_shape_runtime(MakeCtx());
  same_shape_runtime.Set("A", rt_ns::Tensor::From<float>("A", {2, 2}, {5, 6, 7, 8}));
  same_shape_runtime.Set("B", rt_ns::Tensor::From<float>("B", {2, 2}, {1, 0, 0, 1}));
  kernel.Run(same_shape_runtime);
  EXPECT_FLOAT_EQ(same_shape_runtime.Get("Y").AsFloat()[0], 5.0f);
  EXPECT_FLOAT_EQ(same_shape_runtime.Get("Y").AsFloat()[3], 8.0f);

  rt_ns::RuntimeContext changed_shape_runtime(MakeCtx());
  changed_shape_runtime.Set("A", rt_ns::Tensor::From<float>("A", {1, 2}, {2, 3}));
  changed_shape_runtime.Set("B", rt_ns::Tensor::From<float>("B", {2, 1}, {4, 5}));
  kernel.Run(changed_shape_runtime);
  EXPECT_EQ(changed_shape_runtime.Get("Y").shape, (rt_ns::Shape{1, 1}));
  EXPECT_FLOAT_EQ(changed_shape_runtime.Get("Y").AsFloat()[0], 23.0f);
}

} // namespace
