#include "onnx_light_cpu/kernels/math/matmul_kernel.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#include "ml_cpu_compute/math/gemm/gemm_plan.h"
#include "unittests/cc/heap_allocation_tracker.h"
#include "unittests/cc/scoped_x86_gradual_underflow.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/value_store.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_session.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

void CheckBfloat16BoundedConversion(std::size_t m, std::size_t n, std::size_t k,
                                    ml_cpu_compute::GemmAlgorithm algorithm,
                                    bool subnormal_inputs = false) {
  const rt_ns::Shape a_shape{static_cast<std::int64_t>(m), static_cast<std::int64_t>(k)};
  const rt_ns::Shape b_shape{static_cast<std::int64_t>(k), static_cast<std::int64_t>(n)};
  std::vector<float> a(m * k), b(k * n);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = subnormal_inputs ? rt_ns::Bfloat16BitsToFloat(0x0010)
                            : static_cast<float>(static_cast<int>(i % 9) - 4) / 8.0f;
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = subnormal_inputs ? rt_ns::Bfloat16BitsToFloat(0x7e80)
                            : static_cast<float>(static_cast<int>(i % 7) - 3) / 8.0f;
  }

  const ml_cpu_compute::GemmHalfPlan plan({true, false, false, m, n, k});
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
  if (algorithm == ml_cpu_compute::GemmAlgorithm::kGeneral) {
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
  CheckBfloat16BoundedConversion(512, 512, 1024, ml_cpu_compute::GemmAlgorithm::kGeneral);
}

TEST(MatMulKernel, Bfloat16DirectUsesBoundedConversion) {
  CheckBfloat16BoundedConversion(8, 4097, 32, ml_cpu_compute::GemmAlgorithm::kDirect);
}

TEST(MatMulKernel, Bfloat16SubnormalConversionStaysBounded) {
  onnx_light_cpu::test::ScopedX86GradualUnderflow gradual_underflow;
  CheckBfloat16BoundedConversion(512, 512, 1024, ml_cpu_compute::GemmAlgorithm::kGeneral, true);
}

TEST(MatMulKernel, Bfloat16SubnormalInputsMatchFloat32) {
  onnx_light_cpu::test::ScopedX86GradualUnderflow gradual_underflow;
  constexpr std::size_t size = 128;
  constexpr std::uint16_t input_pairs[][2] = {
      {0x0080, 0x7e80}, {0x0010, 0x7e80}, {0x0080, 0x3e80}, {0x0080, 0x7e80}};
  const rt_ns::Shape shape{size, size};
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("MatMul");
  node.add_input("A");
  node.add_input("B");
  node.add_output("Y");
  onnx_light_cpu::MatMulKernel kernel(MakeCtx());
  kernel.set_node(node);
  rt_ns::RuntimeContext runtime(MakeCtx());
  for (const auto &[bits, large_bits] : input_pairs) {
    for (bool small_b : {false, true}) {
      SCOPED_TRACE(bits);
      SCOPED_TRACE(small_b);
      const float small = rt_ns::Bfloat16BitsToFloat(bits);
      const float large = rt_ns::Bfloat16BitsToFloat(large_bits);
      const std::vector<float> a(size * size, small_b ? large : small);
      const std::vector<float> b(size * size, small_b ? small : large);
      const auto expected = kernel(rt_ns::Tensor::From<float>("A", shape, a),
                                   rt_ns::Tensor::From<float>("B", shape, b));
      runtime.Put("A", rt_ns::MakeBfloat16Tensor("A", shape, a));
      runtime.Put("B", rt_ns::MakeBfloat16Tensor("B", shape, b));
      kernel.Run(runtime);
      const auto &actual = runtime.Get("Y");
      ASSERT_EQ(actual.shape, shape);
      const auto *actual_bits = reinterpret_cast<const std::uint16_t *>(actual.bytes());
      for (std::size_t i = 0; i < size * size; ++i) {
        ASSERT_EQ(actual_bits[i], rt_ns::FloatToBfloat16Bits(expected.AsFloat()[i])) << i;
      }
    }
  }
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

struct AllocationCheckingExecutor {
  std::int64_t dispatches = 0;
  std::int64_t maximum_blocks = 0;
  std::size_t worker_allocation = 0;

  static void Run(void *context, std::int64_t blocks, void *task_context,
                  ml_cpu_compute::ExecutionBlockFn task) {
    auto &self = *static_cast<AllocationCheckingExecutor *>(context);
    ++self.dispatches;
    self.maximum_blocks = std::max(self.maximum_blocks, blocks);
    self.worker_allocation = std::max(self.worker_allocation, PeakAllocationBytes([&] {
                                        for (std::int64_t block = 0; block < blocks; ++block) {
                                          task(task_context, block);
                                        }
                                      }));
  }
};

TEST(MatMulKernel, SharesImmutableWideWeightsAndPreservesShapeFallbacks) {
  constexpr std::size_t k = 512, n = 32016;
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("MatMul");
  node.add_input("A");
  node.add_input("B");
  node.add_output("Y");
  onnx_light_cpu::MatMulKernel first(MakeCtx()), second(MakeCtx());
  first.set_node(node);
  second.set_node(node);
  std::vector<rt_ns::KernelPreparation> preparations;
  first.DeclarePreparations({}, preparations);
  EXPECT_TRUE(preparations.empty());
  const bool available =
      ml_cpu_compute::EffectiveGemmSimdLevel() >= ml_cpu_compute::SimdLevel::kAVX2;
  first.DeclarePreparations({"B"}, preparations);
  second.DeclarePreparations({"B"}, preparations);
  ASSERT_EQ(preparations.size(), available ? 2U : 0U);
  if (!available) {
    return;
  }

  rt_ns::RuntimeContext runtime(MakeCtx());
  std::vector<float> b(k * n);
  for (std::size_t col = 0; col < n; ++col) {
    b[col] = static_cast<float>(col % 7) / 8;
    b[(k - 1) * n + col] = static_cast<float>(col % 11) / 16;
  }
  runtime.Set("B", rt_ns::Tensor::FromFloat("B", {k, n}, b));
  const auto memory = preparations[0].estimate_memory(runtime);
  EXPECT_EQ(memory.source_bytes, b.size() * sizeof(float));
  EXPECT_EQ(memory.temporary_bytes, 0U);
  EXPECT_EQ(memory.prepared_bytes, b.size() * sizeof(float));
  EXPECT_EQ(runtime.value_store().prepared_bytes(), 0U);
  EXPECT_EQ(preparations[0].info.output_slots, preparations[1].info.output_slots);
  preparations[0].execute(runtime);
  EXPECT_EQ(preparations[1].estimate_memory(runtime).prepared_bytes, 0U);
  preparations[1].execute(runtime);
  EXPECT_EQ(runtime.value_store().prepared_bytes(), memory.prepared_bytes);
  EXPECT_EQ(runtime.value_store().slot_count(), 1U);

  for (std::size_t m : {0U, 1U, 17U, 128U, 129U}) {
    SCOPED_TRACE(m);
    std::vector<float> a(m * k, 0.25f);
    for (std::size_t row = 0; row < m; ++row) {
      a[row * k] = static_cast<float>(row % 13) / 4;
      a[(row + 1) * k - 1] = static_cast<float>(row % 5) / 2;
    }
    runtime.Put("A", rt_ns::Tensor::FromFloat("A", {1, static_cast<std::int64_t>(m), k}, a));
    if (m >= 16) {
      AllocationCheckingExecutor executor;
      ml_cpu_compute::ExecutionExecutorView view{&executor, 4, &AllocationCheckingExecutor::Run};
      ml_cpu_compute::ExecutionExecutorScope scope(&view);
      second.Run(runtime);
      EXPECT_EQ(executor.dispatches, 1);
      EXPECT_GT(executor.maximum_blocks, 1);
      EXPECT_LE(executor.maximum_blocks, 4);
      EXPECT_EQ(executor.worker_allocation, 0U);
    } else {
      second.Run(runtime);
    }
    const auto &output = runtime.Get("Y");
    ASSERT_EQ(output.shape, (rt_ns::Shape{1, static_cast<std::int64_t>(m), n}));
    for (std::size_t row = 0; row < m; ++row) {
      for (std::size_t col = 0; col < n; ++col) {
        const float expected = a[row * k] * b[col] + a[(row + 1) * k - 1] * b[(k - 1) * n + col];
        ASSERT_FLOAT_EQ(output.AsFloat()[row * n + col], expected) << row << ',' << col;
      }
    }
    EXPECT_EQ(runtime.value_store().prepared_bytes(), memory.prepared_bytes);
  }

  b[0] = 2.0f;
  runtime.Put("B", rt_ns::Tensor::FromFloat("B", {k, n}, b));
  EXPECT_EQ(preparations[0].estimate_memory(runtime).prepared_bytes, memory.prepared_bytes);
  preparations[0].execute(runtime);
  EXPECT_EQ(runtime.value_store().slot_count(), 2U);
  EXPECT_EQ(runtime.value_store().prepared_bytes(), 2 * memory.prepared_bytes);
  EXPECT_EQ(preparations[0].estimate_memory(runtime).prepared_bytes, 0U);

  rt_ns::RuntimeContext independent(MakeCtx());
  independent.Set("A", rt_ns::Tensor::FromFloat("A", {16, k}, std::vector<float>(16 * k, 1)));
  independent.Set("B", rt_ns::Tensor::FromFloat("B", {k, n}, std::vector<float>(k * n, 0)));
  first.Run(independent);
  for (std::size_t index = 0; index < 16 * n; ++index) {
    ASSERT_FLOAT_EQ(independent.Get("Y").AsFloat()[index], 0);
  }
  EXPECT_EQ(independent.value_store().prepared_bytes(), 0U);
}

TEST(MatMulKernel, WidePreparationRejectsTruncatedStorageAndSkipsUnsupportedWeights) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("MatMul");
  node.add_input("A");
  node.add_input("B");
  node.add_output("Y");
  onnx_light_cpu::MatMulKernel kernel(MakeCtx());
  kernel.set_node(node);
  std::vector<rt_ns::KernelPreparation> preparations;
  kernel.DeclarePreparations({"B"}, preparations);
  if (preparations.empty()) {
    return;
  }
  rt_ns::RuntimeContext runtime(MakeCtx());
  const std::uint8_t value = 0;
  for (const rt_ns::Shape shape : {rt_ns::Shape{511, 32000}, rt_ns::Shape{512, 31984},
                                   rt_ns::Shape{512, 32001}, rt_ns::Shape{1, 512, 32000}}) {
    runtime.Put("B", rt_ns::Tensor::Borrow("B", static_cast<int>(rt_ns::DataType::FLOAT), shape,
                                           &value, sizeof(value)));
    EXPECT_EQ(preparations[0].estimate_memory(runtime).prepared_bytes, 0U);
    preparations[0].execute(runtime);
    EXPECT_EQ(runtime.value_store().prepared_bytes(), 0U);
  }
  runtime.Put("B", rt_ns::Tensor::Borrow("B", static_cast<int>(rt_ns::DataType::FLOAT16),
                                         {512, 32000}, &value, sizeof(value)));
  EXPECT_EQ(preparations[0].estimate_memory(runtime).prepared_bytes, 0U);
  preparations[0].execute(runtime);
  runtime.Put("B", rt_ns::Tensor::Borrow("B", static_cast<int>(rt_ns::DataType::FLOAT),
                                         {512, 32000}, &value, sizeof(value)));
  EXPECT_THROW(preparations[0].estimate_memory(runtime), std::invalid_argument);
  EXPECT_THROW(preparations[0].execute(runtime), std::invalid_argument);
  runtime.Put("B", rt_ns::Tensor::Borrow("B", static_cast<int>(rt_ns::DataType::FLOAT),
                                         {512, std::numeric_limits<std::int64_t>::max() - 15},
                                         &value, sizeof(value)));
  EXPECT_THROW(preparations[0].estimate_memory(runtime), std::invalid_argument);
  EXPECT_EQ(runtime.value_store().prepared_bytes(), 0U);
}

TEST(MatMulKernel, RuntimeSessionPreparesWideInitializersBeforeInference) {
  constexpr std::int64_t k = 512, n = 32000;
  ONNX_LIGHT_NAMESPACE::ModelProto model;
  model.add_opset_import()->set_version(18);
  auto *graph = model.mutable_graph();
  auto *input = graph->add_input();
  input->set_name("A");
  input->mutable_type()->mutable_tensor_type()->set_elem_type(rt_ns::DataType::FLOAT);
  graph->add_output()->set_name("Y");
  auto *node = graph->add_node();
  node->set_op_type("MatMul");
  node->add_input("A");
  node->add_input("B");
  node->add_output("Y");
  auto *weights = graph->add_initializer();
  weights->set_name("B");
  weights->set_data_type(rt_ns::DataType::FLOAT);
  weights->add_dims(k);
  weights->add_dims(n);
  const std::vector<float> b(k * n, 0.25f);
  weights->set_raw_data(
      std::string(reinterpret_cast<const char *>(b.data()), b.size() * sizeof(float)));
  rt_ns::RuntimeContext runtime(MakeCtx());
  ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(runtime, "", "MatMul"));
  rt_ns::RuntimeSession session(model);
  session.Prepare(runtime);
  const bool available =
      ml_cpu_compute::EffectiveGemmSimdLevel() >= ml_cpu_compute::SimdLevel::kAVX2;
  const std::size_t prepared_bytes = available ? b.size() * sizeof(float) : 0;
  EXPECT_EQ(session.prepared_bytes(), prepared_bytes);
  EXPECT_FALSE(runtime.HasValue("Y"));
  runtime.Set("A", rt_ns::Tensor::FromFloat("A", {17, k}, std::vector<float>(17 * k, 0.5f)));
  session.Run(runtime);
  EXPECT_EQ(session.prepared_bytes(), prepared_bytes);
  for (std::size_t index = 0; index < 17 * n; ++index) {
    ASSERT_FLOAT_EQ(runtime.Get("Y").AsFloat()[index], 64);
  }
}

} // namespace
