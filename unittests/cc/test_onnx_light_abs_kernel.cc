// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/abs_kernel.h"

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext MakeCtx() { return rt_ns::KernelContext(rt_ns::OpsetId(std::string(), 18)); }

TEST(OnnxLightAbsKernel, Float32) {
  onnx_light_cpu::AbsKernel kernel(MakeCtx());
  const std::vector<float> values = {-1.0f, 0.0f, 3.0f, -7.5f};
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {4}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], std::fabs(values[i]));
  }
}

TEST(OnnxLightAbsKernel, Float64) {
  onnx_light_cpu::AbsKernel kernel(MakeCtx());
  const std::vector<double> values = {-1.0, 0.0, 3.0, -7.5, 0.001};
  const rt_ns::Tensor x = rt_ns::Tensor::FromDouble("x", {5}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 5);
  const double *py = y.AsDouble();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_DOUBLE_EQ(py[i], std::fabs(values[i]));
  }
}

TEST(OnnxLightAbsKernel, Int64) {
  onnx_light_cpu::AbsKernel kernel(MakeCtx());
  const std::vector<int64_t> values = {-1, 0, 3, -7, 100};
  const rt_ns::Tensor x = rt_ns::Tensor::FromInt64("x", {5}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), 5);
  const int64_t *py = y.AsInt64();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(py[i], values[i] < 0 ? -values[i] : values[i]);
  }
}

TEST(OnnxLightAbsKernel, OutputUsesSlotAllocator) {
  onnx_light_cpu::AbsKernel kernel(MakeCtx());
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {2}, {-1.0f, 3.0f});
  rt_ns::ExecutionArena execution_arena(1);
  auto io_arena = rt_ns::IOArena::Create(1);
  rt_ns::RuntimeContextOptions options;
  options.allocator = &execution_arena;
  options.io_allocator = io_arena.get();
  rt_ns::RuntimeContext rt(options);
  rt.set_output_slot_io_roles({true});

  {
    const rt_ns::Tensor y = kernel(x, &rt);
    EXPECT_EQ(y.allocation_owner(), io_arena.get());
    EXPECT_EQ(execution_arena.allocated_count(), 0);
    EXPECT_EQ(io_arena->allocated_count(), 1);
  }

  rt.set_output_slot_io_roles({false});
  {
    const rt_ns::Tensor y = kernel(x, &rt);
    EXPECT_EQ(y.allocation_owner(), &execution_arena);
    EXPECT_EQ(execution_arena.allocated_count(), 1);
    EXPECT_EQ(io_arena->allocated_count(), 0);
  }
}

// Exercises the ParallelFor path: the array is large enough (above
// ``kParallelForGrainSize``) that ``AbsKernel`` splits it across the shared
// thread pool. The result must stay bit-exact regardless of how many threads
// process it.
TEST(OnnxLightAbsKernel, Float32LargeParallel) {
  onnx_light_cpu::AbsKernel kernel(MakeCtx());
  const int64_t n = 2000003; // Above Abs's discounted parallel grain.
  std::vector<float> values(static_cast<std::size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    values[static_cast<std::size_t>(i)] = static_cast<float>((i % 2 == 0 ? -1 : 1) * (i % 97));
  }
  const rt_ns::Tensor x = rt_ns::Tensor::FromFloat("x", {n}, values);
  const rt_ns::Tensor y = kernel(x);
  ASSERT_EQ(y.element_count(), n);
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], std::fabs(values[i]));
  }
}

} // namespace
