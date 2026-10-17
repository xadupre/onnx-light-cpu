#include "onnx_light_cpu/kernels/math/matmul_kernel.h"

#include "onnx_light_cpu/impl/math/gemm/gemm_plan.h"
#include "onnx_light_cpu/kernels/register_kernels.h"
#include "unittests/cc/heap_allocation_tracker.h"
#include "unittests/cc/scoped_x86_gradual_underflow.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/value_store.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_session.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

rt_ns::Tensor MakeInput(rt_ns::DataType dtype, const std::string &name, const rt_ns::Shape &shape,
                        const std::vector<float> &values) {
  if (dtype == rt_ns::DataType::FLOAT16) {
    return rt_ns::MakeFloat16Tensor(name, shape, values);
  }
  if (dtype == rt_ns::DataType::BFLOAT16) {
    return rt_ns::MakeBfloat16Tensor(name, shape, values);
  }
  return rt_ns::Tensor::FromFloat(name, shape, values);
}

ONNX_LIGHT_NAMESPACE::ModelProto MakeMatMulModel(const rt_ns::Tensor &b, bool mutable_b = false,
                                                 bool duplicate = false) {
  ONNX_LIGHT_NAMESPACE::ModelProto model;
  model.add_opset_import()->set_version(18);
  auto &graph = *model.mutable_graph();
  graph.add_input()->set_name("A");
  if (mutable_b) {
    graph.add_input()->set_name("B");
  }
  auto &initializer = *graph.add_initializer();
  initializer.set_name("B");
  initializer.set_data_type(b.data_type);
  for (const auto dimension : b.shape) {
    initializer.add_dims(dimension);
  }
  initializer.set_raw_data(std::string(reinterpret_cast<const char *>(b.bytes()), b.size_bytes()));
  for (const std::string output :
       duplicate ? std::vector<std::string>{"Y", "Z"} : std::vector<std::string>{"Y"}) {
    auto &node = *graph.add_node();
    node.set_op_type("MatMul");
    node.add_input("A");
    node.add_input("B");
    node.add_output(output);
    graph.add_output()->set_name(output);
  }
  return model;
}

TEST(MatMulKernel, SessionsSharePackedWeightsAcrossNodesShapesAndDtypes) {
  constexpr std::int64_t k = 257;
  constexpr std::int64_t n = 35;
  for (const auto dtype :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    std::vector<float> weights(k * n);
    for (std::size_t index = 0; index < weights.size(); ++index) {
      weights[index] = static_cast<float>(static_cast<int>(index % 7) - 3) / 32.0f;
    }
    const auto b = MakeInput(dtype, "B", {k, n}, weights);
    const auto model = MakeMatMulModel(b, false, true);
    auto store = std::make_shared<rt_ns::ValueStore>();
    rt_ns::RuntimeSession session(model, rt_ns::RuntimeSessionOptions{.value_store = store});
    rt_ns::RuntimeContext runtime(MakeCtx());
    ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(runtime, "", "MatMul"));
    EXPECT_EQ(store->slot_count(), 0U);
    const std::uint8_t *packed_address = nullptr;
    for (const rt_ns::Shape shape : {rt_ns::Shape{k}, rt_ns::Shape{7, k}, rt_ns::Shape{2, 3, k},
                                     rt_ns::Shape{0, k}, rt_ns::Shape{1, k}}) {
      SCOPED_TRACE(::testing::PrintToString(shape));
      std::size_t elements = 1;
      for (const auto dimension : shape) {
        elements *= static_cast<std::size_t>(dimension);
      }
      std::vector<float> values(elements);
      for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<float>(static_cast<int>(index % 11) - 5) / 16.0f;
      }
      runtime.Put("A", MakeInput(dtype, "A", shape, values));
      session.Run(runtime);
      ASSERT_EQ(store->slot_count(), 1U);
      EXPECT_EQ(session.prepared_bytes(), static_cast<std::size_t>(k * n) * sizeof(float));
      const auto &packed = store->Get(0).tensor;
      if (packed_address == nullptr) {
        packed_address = packed.bytes();
      }
      EXPECT_EQ(packed.bytes(), packed_address);
      onnx_light_cpu::MatMulKernel reference(MakeCtx());
      const auto expected = reference(runtime.Get("A"), b);
      for (const auto *output : {"Y", "Z"}) {
        const auto &actual = runtime.Get(output);
        ASSERT_EQ(actual.shape, expected.shape);
        ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
        if (actual.size_bytes() != 0) {
          EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
        }
      }
    }
    rt_ns::RuntimeSession other(model, rt_ns::RuntimeSessionOptions{.value_store = store});
    rt_ns::RuntimeContext other_runtime(MakeCtx());
    ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(other_runtime, "", "MatMul"));
    other_runtime.Put("A", MakeInput(dtype, "A", {1, k}, std::vector<float>(k, 1.0f)));
    other.Run(other_runtime);
    EXPECT_EQ(store->slot_count(), 1U);
    EXPECT_EQ(store->Get(0).tensor.bytes(), packed_address);
  }
}

TEST(MatMulKernel, MutableWeightsAndUnsupportedLayoutsKeepDynamicExecution) {
  for (const auto dtype :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    const auto b = MakeInput(dtype, "B", {2, 2}, {1, 0, 0, 1});
    const auto model = MakeMatMulModel(b, true);
    rt_ns::RuntimeSession session(model);
    rt_ns::RuntimeContext runtime(MakeCtx());
    ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(runtime, "", "MatMul"));
    runtime.Put("A", MakeInput(dtype, "A", {1, 2}, {2, 3}));
    session.Run(runtime);
    const auto original = runtime.Get("Y");
    runtime.Put("B", MakeInput(dtype, "B", {2, 2}, {2, 0, 0, 2}));
    session.Run(runtime);
    onnx_light_cpu::MatMulKernel reference(MakeCtx());
    const auto expected = reference(runtime.Get("A"), runtime.Get("B"));
    EXPECT_EQ(std::memcmp(runtime.Get("Y").bytes(), expected.bytes(), expected.size_bytes()), 0);
    EXPECT_NE(std::memcmp(runtime.Get("Y").bytes(), original.bytes(), original.size_bytes()), 0);
    EXPECT_EQ(session.prepared_bytes(), 0U);

    const auto immutable_model = MakeMatMulModel(b);
    rt_ns::RuntimeSession overridden(immutable_model);
    rt_ns::RuntimeContext override_runtime(MakeCtx());
    ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(override_runtime, "", "MatMul"));
    override_runtime.Put("A", MakeInput(dtype, "A", {1, 2}, {2, 3}));
    override_runtime.Put("B", MakeInput(dtype, "B", {2, 2}, {2, 0, 0, 2}));
    overridden.Run(override_runtime);
    EXPECT_EQ(overridden.prepared_bytes(), 0U);
    EXPECT_EQ(
        std::memcmp(override_runtime.Get("Y").bytes(), expected.bytes(), expected.size_bytes()), 0);
  }
  for (const rt_ns::Shape shape : {rt_ns::Shape{2}, rt_ns::Shape{1, 2, 2}, rt_ns::Shape{0, 2}}) {
    rt_ns::RuntimeContext runtime(MakeCtx());
    std::size_t count = 1;
    for (const auto dim : shape) {
      count *= static_cast<std::size_t>(dim);
    }
    runtime.Put("B", rt_ns::Tensor::FromFloat("B", shape, std::vector<float>(count, 1)));
    ONNX_LIGHT_NAMESPACE::NodeProto node;
    node.add_input("A");
    node.add_input("B");
    onnx_light_cpu::MatMulKernel kernel(MakeCtx());
    kernel.set_node(node);
    kernel.Prepare(runtime, {"B"});
    EXPECT_EQ(runtime.value_store().prepared_bytes(), 0U);
  }
}

TEST(MatMulKernel, PreparationDistinguishesWeightContentTypeAndShape) {
  auto store = std::make_shared<rt_ns::ValueStore>();
  for (const auto dtype :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    for (const float value : {1.0f, 2.0f}) {
      for (const rt_ns::Shape shape : {rt_ns::Shape{2, 3}, rt_ns::Shape{3, 2}}) {
        const auto model =
            MakeMatMulModel(MakeInput(dtype, "B", shape, std::vector<float>(6, value)));
        rt_ns::RuntimeSession session(model, rt_ns::RuntimeSessionOptions{.value_store = store});
        rt_ns::RuntimeContext runtime(MakeCtx());
        ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(runtime, "", "MatMul"));
        runtime.Put("A", MakeInput(dtype, "A", {1, shape[0]},
                                   std::vector<float>(static_cast<std::size_t>(shape[0]), 1)));
        session.Run(runtime);
        const auto expected =
            MakeInput(dtype, "Y", {1, shape[1]},
                      std::vector<float>(static_cast<std::size_t>(shape[1]), value * shape[0]));
        EXPECT_EQ(std::memcmp(runtime.Get("Y").bytes(), expected.bytes(), expected.size_bytes()),
                  0);
      }
    }
  }
  EXPECT_EQ(store->slot_count(), 12U);
  EXPECT_EQ(store->prepared_bytes(), 12U * 6U * sizeof(float));
}

TEST(MatMulKernel, PackedWeightsValidateShapesAndInputTypes) {
  const auto model = MakeMatMulModel(rt_ns::Tensor::FromFloat("B", {2, 2}, {1, 0, 0, 1}));
  rt_ns::RuntimeSession session(model);
  rt_ns::RuntimeContext runtime(MakeCtx());
  ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(runtime, "", "MatMul"));
  runtime.Put("A", rt_ns::Tensor::FromFloat("A", {1, 2}, {1, 2}));
  session.Run(runtime);
  runtime.Put("A", rt_ns::Tensor::FromFloat("A", {1, 3}, {1, 2, 3}));
  EXPECT_THROW(session.Run(runtime), std::invalid_argument);
  runtime.Put("A", rt_ns::Tensor::FromDouble("A", {1, 2}, {1, 2}));
  EXPECT_THROW(session.Run(runtime), std::invalid_argument);
}

TEST(MatMulKernel, PreparedHalfWeightsKeepConversionBoundedAcrossKChunks) {
  constexpr std::int64_t k = 1377;
  constexpr std::int64_t n = 65;
  for (const auto dtype :
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    const auto b = MakeInput(dtype, "B", {k, n}, std::vector<float>(k * n, 0.125f));
    const auto model = MakeMatMulModel(b);
    rt_ns::RuntimeSession session(model);
    rt_ns::RuntimeContext runtime(MakeCtx());
    ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(runtime, "", "MatMul"));
    for (const std::int64_t m : {128, 1}) {
      runtime.Put("A", MakeInput(dtype, "A", {m, k}, std::vector<float>(m * k, 0.25f)));
      session.Run(runtime);
      const auto expected = MakeInput(dtype, "Y", {m, n}, std::vector<float>(m * n, k * 0.03125f));
      EXPECT_EQ(std::memcmp(runtime.Get("Y").bytes(), expected.bytes(), expected.size_bytes()), 0);
      if (m == 1) {
        const auto largest_allocation = PeakAllocationBytes([&] { session.Run(runtime); });
        EXPECT_LT(largest_allocation, static_cast<std::size_t>(k * n) * sizeof(float));
      }
    }
  }
}

void CheckBfloat16BoundedConversion(std::size_t m, std::size_t n, std::size_t k,
                                    onnx_light_cpu::GemmAlgorithm algorithm,
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

TEST(MatMulKernel, Bfloat16SubnormalConversionStaysBounded) {
  onnx_light_cpu::test::ScopedX86GradualUnderflow gradual_underflow;
  CheckBfloat16BoundedConversion(512, 512, 1024, onnx_light_cpu::GemmAlgorithm::kGeneral, true);
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

} // namespace
