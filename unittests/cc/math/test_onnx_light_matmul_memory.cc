// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/math/gemm/gemm_plan.h"
#include "onnx_light_cpu/kernels/math/gemm_kernel.h"
#include "onnx_light_cpu/kernels/math/matmul_kernel.h"
#include "onnx_light_cpu/kernels/register_kernels.h"
#include "onnx_light_cpu/shapes/ai_onnx/shape_inference.h"

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#if defined(ONNX_LIGHT_CPU_HAS_ONNX_SHAPE)
#include "onnx_extensions/shapes/dispatch_table.h"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace shapes_ns = ONNX_LIGHT_NAMESPACE::core::shapes;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
using sym_ns::SymDim;
using sym_ns::SymShape;

ONNX_LIGHT_NAMESPACE::NodeProto MatMulNode() {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("MatMul");
  node.add_input("A");
  node.add_input("B");
  node.add_output("Y");
  return node;
}

TEST(MatMulMemory, DecoderShapesUseCpuPlansWithoutTensorAllocation) {
  onnx_light_cpu::RegisterAllKernels();
  for (const std::size_t sequence : {1U, 128U, 32000U}) {
    SCOPED_TRACE(sequence);
    const std::array<std::size_t, 3> a{1, sequence, 1024};
    const std::array<std::size_t, 2> b{1024, 151936};
    const ml_cpu_compute::MatMulPlan<float> f32(a, b);
    const ml_cpu_compute::MatMulPlan<double> f64(a, b);
    const std::size_t expected =
        std::max({f32.ScratchBytes(), f64.ScratchBytes(), f32.HalfScratchBytes(false),
                  f32.HalfScratchBytes(true)});
    const auto estimate = shapes_ns::ComputePeakMemory(
        sym_ns::Device::kCPU, MatMulNode(),
        {SymShape({1, static_cast<std::int64_t>(sequence), 1024}), SymShape({1024, 151936})});
    RecordProperty("sequence_" + std::to_string(sequence), std::to_string(expected));
    const ml_cpu_compute::GemmHalfPlan half({false, false, false, sequence, 151936, 1024});
    const auto property = [&](const std::string &name, std::size_t value) {
      RecordProperty("sequence_" + std::to_string(sequence) + "_" + name, std::to_string(value));
    };
    property("float_participants", f32.gemm_plan().maximum_participants());
    property("double_participants", f64.gemm_plan().maximum_participants());
    property("half_participants", half.maximum_participants());
    property("half_bytes_48", f32.HalfScratchBytes(false, 48));
    EXPECT_EQ(estimate.worst_case.AsInt(), expected);
    EXPECT_FALSE(estimate.average_case.has_value());
    EXPECT_GE(expected, sequence * 151936 * sizeof(float));
    // Explicit executor bounds are available at the plan layer. Packing is
    // panel-sized, not a gratuitous full FP32 copy of the large weights.
    const std::size_t workspace = sequence * 151936 * sizeof(float);
    EXPECT_LT(f32.HalfScratchBytes(false, 48) - workspace, 1024 * 151936 * sizeof(float));
    if (sequence == 1) {
      EXPECT_LE(expected, 2 * 151936 * sizeof(float));
    }
  }
}

TEST(MatMulMemory, CpuOverrideSurvivesEveryLazyRegistrationOrder) {
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  const std::vector<SymShape> shapes{SymShape({1, 1024}), SymShape({1024, 151936})};
  const auto expected =
      onnx_light_cpu::ComputePeakMemoryMatMul(sym_ns::Device::kCPU, MatMulNode(), shapes);
#if defined(ONNX_LIGHT_CPU_HAS_ONNX_SHAPE)
  ONNX_LIGHT_NAMESPACE::onnx_shapes::RegisterShapeFunctions();
  ONNX_LIGHT_NAMESPACE::onnx_shapes::RegisterPeakMemoryFunctions();
#endif
  onnx_light_cpu::RegisterAllKernels();
  onnx_light_cpu::RegisterAllKernels(onnx_light_cpu::MicrosoftKernelImplementation::NAIVE);
  onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
  for (const std::string domain : {"", "ai.onnx"}) {
    auto node = MatMulNode();
    node.set_domain(domain);
    EXPECT_EQ(shapes_ns::ComputePeakMemory(sym_ns::Device::kCPU, node, shapes), expected);
    EXPECT_EQ(shapes_ns::ComputePeakMemory(domain, "MatMul", sym_ns::Device::kCPU, shapes),
              expected);
  }
}

TEST(MatMulMemory, VectorsBroadcastsAndEmptyProducts) {
  const auto peak = [](const std::vector<SymShape> &shapes) {
    return onnx_light_cpu::ComputePeakMemoryMatMul(sym_ns::Device::kCPU, MatMulNode(), shapes)
        .worst_case.AsInt();
  };
  EXPECT_GE(peak({SymShape({17}), SymShape({17})}), sizeof(float));
  EXPECT_GE(peak({SymShape({17}), SymShape({17, 9})}), 9 * sizeof(float));
  EXPECT_GE(peak({SymShape({8, 17}), SymShape({17})}), 8 * sizeof(float));
  EXPECT_GE(peak({SymShape({2, 1, 8, 17}), SymShape({1, 3, 17, 9})}),
            2 * 3 * 8 * 9 * sizeof(float));
  EXPECT_EQ(peak({SymShape({2, int64_t{0}, 17}), SymShape({17, 9})}), 0);
  EXPECT_EQ(peak({SymShape({int64_t{0}, 8, 17}), SymShape({17, 9})}), 0);
  EXPECT_EQ(peak({SymShape({8, int64_t{0}}), SymShape({int64_t{0}, 9})}), 8 * 9 * sizeof(float));
}

TEST(MatMulMemory, UnresolvedAndInvalidShapes) {
  const auto estimate = [](const std::vector<SymShape> &shapes) {
    return onnx_light_cpu::ComputePeakMemoryMatMul(sym_ns::Device::kCPU, MatMulNode(), shapes);
  };
  EXPECT_TRUE(estimate({}).worst_case.IsExpr());
  EXPECT_TRUE(estimate({SymShape({}), SymShape({3, 4})}).worst_case.IsExpr());
  EXPECT_TRUE(estimate({SymShape({SymDim("S"), SymDim(3)}), SymShape({3, 4})}).worst_case.IsExpr());
  EXPECT_THROW(estimate({SymShape({-1, 3}), SymShape({3, 4})}), std::invalid_argument);
  EXPECT_THROW(estimate({SymShape({2, 3}), SymShape({4, 5})}), std::invalid_argument);
  EXPECT_THROW(estimate({SymShape({2, 2, 3}), SymShape({3, 3, 4})}), std::invalid_argument);
  const auto huge = std::numeric_limits<std::int64_t>::max();
  EXPECT_THROW(estimate({SymShape({huge, 2}), SymShape({2, 2})}), std::invalid_argument);
}

class RecordingAllocator : public rt_ns::SimpleRawBufferAllocator {
public:
  RecordingAllocator() : SimpleRawBufferAllocator(4) {}
  rt_ns::RawBuffer *Allocate(std::size_t bytes) override {
    requests.push_back(bytes);
    return SimpleRawBufferAllocator::Allocate(bytes);
  }
  std::vector<std::size_t> requests;
};

TEST(MatMulMemory, RuntimeAllocatorOwnsGenericPackingAndWorkspaceNotDeclaredOutput) {
  constexpr std::size_t m = 17, n = 1057, k = 129;
  const std::vector<float> a(m * k, 0.25f), b(k * n, 0.5f);
  ml_cpu_compute::ExecutionExecutorScope serial(nullptr);
  for (const auto dtype : {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE,
                           rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    RecordingAllocator scratch, outputs;
    const rt_ns::KernelContext context(rt_ns::OpsetId("", 18));
    rt_ns::RuntimeContext runtime(
        context, rt_ns::RuntimeContextOptions{.allocator = &scratch, .io_allocator = &outputs});
    runtime.SetActiveAllocator(&outputs);
    if (dtype == rt_ns::DataType::FLOAT16 || dtype == rt_ns::DataType::BFLOAT16) {
      const bool bf16 = dtype == rt_ns::DataType::BFLOAT16;
      runtime.Set("A", bf16 ? rt_ns::MakeBfloat16Tensor("A", {m, k}, a)
                            : rt_ns::MakeFloat16Tensor("A", {m, k}, a));
      runtime.Set("B", bf16 ? rt_ns::MakeBfloat16Tensor("B", {k, n}, b)
                            : rt_ns::MakeFloat16Tensor("B", {k, n}, b));
    } else if (dtype == rt_ns::DataType::FLOAT) {
      runtime.Set("A", rt_ns::Tensor::From<float>("A", {m, k}, a));
      runtime.Set("B", rt_ns::Tensor::From<float>("B", {k, n}, b));
    } else {
      runtime.Set("A", rt_ns::Tensor::From<double>("A", {m, k}, std::vector<double>(m * k, 0.25)));
      runtime.Set("B", rt_ns::Tensor::From<double>("B", {k, n}, std::vector<double>(k * n, 0.5)));
    }
    // RuntimeContext owns imported inputs too; only measure subsequent kernel allocations.
    const std::size_t baseline = scratch.TotalAllocatedSize();
    scratch.requests.clear();
    scratch.ResetPeak();
    onnx_light_cpu::MatMulKernel kernel(context);
    const auto node = MatMulNode();
    kernel.set_node(node);
    kernel.Run(runtime);
    const bool half = dtype == rt_ns::DataType::FLOAT16 || dtype == rt_ns::DataType::BFLOAT16;
    EXPECT_FALSE(scratch.requests.empty());
    if (half) {
      EXPECT_EQ(scratch.requests.front(), m * n * sizeof(float));
    }
    EXPECT_EQ(scratch.TotalAllocatedSize(), baseline);
    const ml_cpu_compute::MatMulPlan<float> f32(m, n, k);
    const ml_cpu_compute::MatMulPlan<double> f64(m, n, k);
    const std::size_t bound = half ? f32.HalfScratchBytes(dtype == rt_ns::DataType::BFLOAT16, 1)
                              : dtype == rt_ns::DataType::FLOAT ? f32.ScratchBytes(1)
                                                                : f64.ScratchBytes(1);
    // The raw runtime allocator need not return cache-line-aligned storage.
    EXPECT_LE(scratch.PeakAllocatedSize() - baseline, bound + 63 * scratch.requests.size());
    const std::size_t output_width = half ? 2 : dtype == rt_ns::DataType::FLOAT ? 4 : 8;
    EXPECT_EQ(outputs.requests, (std::vector<std::size_t>{m * n * output_width}));
    EXPECT_EQ(outputs.TotalAllocatedSize(), m * n * output_width);
  }
}

TEST(GemmMemory, RuntimeAllocatorOwnsTransposedGenericPacking) {
  constexpr std::size_t m = 49, n = 1057, k = 259;
  ml_cpu_compute::ExecutionExecutorScope serial(nullptr);
  for (const auto dtype : {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE,
                           rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    RecordingAllocator scratch, outputs;
    const rt_ns::KernelContext context(rt_ns::OpsetId("", 18));
    rt_ns::RuntimeContext runtime(
        context, rt_ns::RuntimeContextOptions{.allocator = &scratch, .io_allocator = &outputs});
    runtime.SetActiveAllocator(&outputs);
    const std::vector<float> a(m * k, 0.25f), b(n * k, 0.5f);
    if (dtype == rt_ns::DataType::FLOAT16 || dtype == rt_ns::DataType::BFLOAT16) {
      const bool bf16 = dtype == rt_ns::DataType::BFLOAT16;
      runtime.Set("A", bf16 ? rt_ns::MakeBfloat16Tensor("A", {k, m}, a)
                            : rt_ns::MakeFloat16Tensor("A", {k, m}, a));
      runtime.Set("B", bf16 ? rt_ns::MakeBfloat16Tensor("B", {n, k}, b)
                            : rt_ns::MakeFloat16Tensor("B", {n, k}, b));
    } else if (dtype == rt_ns::DataType::FLOAT) {
      runtime.Set("A", rt_ns::Tensor::From<float>("A", {k, m}, a));
      runtime.Set("B", rt_ns::Tensor::From<float>("B", {n, k}, b));
    } else {
      runtime.Set("A", rt_ns::Tensor::From<double>("A", {k, m}, std::vector<double>(m * k, 0.25)));
      runtime.Set("B", rt_ns::Tensor::From<double>("B", {n, k}, std::vector<double>(n * k, 0.5)));
    }
    const std::size_t baseline = scratch.TotalAllocatedSize();
    scratch.requests.clear();
    scratch.ResetPeak();
    onnx_light_cpu::GemmKernel kernel(context);
    ONNX_LIGHT_NAMESPACE::NodeProto node;
    node.set_op_type("Gemm");
    node.add_input("A");
    node.add_input("B");
    node.add_output("Y");
    for (const char *name : {"transA", "transB"}) {
      auto *attribute = node.add_attribute();
      attribute->set_name(name);
      attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
      attribute->set_i(1);
    }
    kernel.set_node(node);
    kernel.Run(runtime);
    EXPECT_FALSE(scratch.requests.empty());
    EXPECT_EQ(scratch.TotalAllocatedSize(), baseline);
    EXPECT_EQ(outputs.requests.size(), 1U);
  }
}

} // namespace
