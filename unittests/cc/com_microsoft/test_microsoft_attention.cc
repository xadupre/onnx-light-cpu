// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/com_microsoft/attention.h"
#include "ml_cpu_compute/execution.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
using ml_cpu_compute::AttentionDescriptor;
using ml_cpu_compute::AttentionLayout;
using ml_cpu_compute::AttentionMaskKind;
using ml_cpu_compute::AttentionPlan;

AttentionPlan Plan(bool past = false) {
  AttentionDescriptor descriptor;
  descriptor.q_num_heads = descriptor.kv_num_heads = 1;
  descriptor.has_past_key = descriptor.has_past_value = past;
  const std::vector<int64_t> shape{1, 2, 1};
  const std::vector<int64_t> cache =
      past ? std::vector<int64_t>{1, 1, 1, 1} : std::vector<int64_t>{};
  return AttentionPlan(descriptor, AttentionLayout::kRank3, shape, shape, shape, {},
                       AttentionMaskKind::kNone, cache, cache);
}

TEST(MicrosoftAttentionCompute, ProjectionBiasAndPackedCache) {
  for (bool cached : {false, true}) {
    const auto plan = Plan(cached);
    const std::vector<float> x{1, 2}, w{1, 1, 2}, bias{0, 0, 1}, past{0, 8};
    std::vector<float> y(2), present(cached ? 6 : 4);
    ml_cpu_compute::ComputeMicrosoftAttentionFloat32(plan, 1, x.data(), w.data(), bias.data(),
                                                     nullptr, cached ? past.data() : nullptr,
                                                     y.data(), present.data());
    for (int i = 0; i < 2; ++i) {
      const float a = std::exp(float(i + 1)), b = a * a;
      const float expected = (3 * a + 5 * b + (cached ? 8 : 0)) / (a + b + (cached ? 1 : 0));
      EXPECT_NEAR(y[i], expected, 1.e-5);
    }
    const auto expected =
        cached ? std::vector<float>{0, 1, 2, 8, 3, 5} : std::vector<float>{1, 2, 3, 5};
    EXPECT_EQ(present, expected);
  }
}

TEST(MicrosoftAttentionCompute, UnequalValuesAndAdditiveMask) {
  AttentionDescriptor descriptor;
  descriptor.q_num_heads = descriptor.kv_num_heads = 1;
  descriptor.has_attn_mask = true;
  const std::vector<int64_t> qshape{1, 2, 1}, vshape{1, 2, 2}, mask_shape{1, 1, 2, 2};
  const AttentionPlan plan(descriptor, AttentionLayout::kRank3, qshape, qshape, vshape, mask_shape,
                           AttentionMaskKind::kAdditive);
  const std::vector<float> x{1, 2}, w{0, 0, 2, 3}, mask{0, std::log(3.f), 0, 0};
  std::vector<float> y(4);
  ml_cpu_compute::ComputeMicrosoftAttentionFloat32(plan, 1, x.data(), w.data(), nullptr,
                                                   mask.data(), nullptr, y.data(), nullptr);
  const std::vector<float> expected{3.5f, 5.25f, 3, 4.5f};
  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(y[i], expected[i], 1.e-5);
}

TEST(MicrosoftAttentionCompute, RejectsMissingPointersBeforeDispatch) {
  const auto plan = Plan();
  const std::vector<float> x{1, 2}, w{1, 1, 2};
  std::vector<float> y(2);
  EXPECT_THROW(ml_cpu_compute::ComputeMicrosoftAttentionFloat32(
                   plan, 1, nullptr, w.data(), nullptr, nullptr, nullptr, y.data(), nullptr),
               std::invalid_argument);
  EXPECT_THROW(ml_cpu_compute::ComputeMicrosoftAttentionFloat32(
                   plan, 0, x.data(), w.data(), nullptr, nullptr, nullptr, y.data(), nullptr),
               std::invalid_argument);
}

TEST(MicrosoftAttentionCompute, ImplicitFiniteCausalMaskAcrossStreamingAndTiledPaths) {
  for (int64_t length : {2, 32}) {
    AttentionDescriptor descriptor;
    descriptor.q_num_heads = descriptor.kv_num_heads = 1;
    descriptor.is_causal = true;
    const std::vector<int64_t> shape{1, length, 1};
    const AttentionPlan plan(descriptor, AttentionLayout::kRank3, shape, shape, shape, {},
                             AttentionMaskKind::kNone);
    std::vector<float> x(length * 2, 0.f), y(length);
    x[0] = 1;
    x[3] = 1;
    const std::vector<float> w{1, 0, 1, 0, std::numeric_limits<float>::max(), 2};
    ml_cpu_compute::ComputeMicrosoftAttentionFloat32(plan, 2, x.data(), w.data(), nullptr, nullptr,
                                                     nullptr, y.data(), nullptr);
    EXPECT_FLOAT_EQ(y[0], 1.5f);
    EXPECT_FLOAT_EQ(y[1], 1.5f);
    EXPECT_NEAR(y.back(), 3.f / length, 1.e-6);
  }
}

TEST(MicrosoftAttentionCompute, EmptyQueriesAndBatchPreservePackedCache) {
  for (int64_t batch : {0, 1}) {
    AttentionDescriptor descriptor;
    descriptor.q_num_heads = descriptor.kv_num_heads = 1;
    descriptor.has_past_key = descriptor.has_past_value = true;
    const std::vector<int64_t> shape{batch, 0, 1}, cache{batch, 1, 1, 1};
    const AttentionPlan plan(descriptor, AttentionLayout::kRank3, shape, shape, shape, {},
                             AttentionMaskKind::kNone, cache, cache);
    const std::vector<float> w{1, 1, 2}, past{3, 4};
    std::vector<float> present(batch * 2);
    ml_cpu_compute::ComputeMicrosoftAttentionFloat32(plan, 1, nullptr, w.data(), nullptr, nullptr,
                                                     batch ? past.data() : nullptr, nullptr,
                                                     batch ? present.data() : nullptr);
    if (batch)
      EXPECT_EQ(present, past);
  }
}

TEST(MicrosoftAttentionCompute, RejectsOverflowBeforeAllocation) {
  auto plan = Plan();
  plan.batch = 2;
  plan.q_length = plan.kv_length = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(ml_cpu_compute::ComputeMicrosoftAttentionFloat32(plan, 1, nullptr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr),
               std::invalid_argument);
  EXPECT_THROW(ml_cpu_compute::MicrosoftAttentionPeakMemoryBytes(plan, 1), std::invalid_argument);
}

TEST(MicrosoftAttentionCompute, BorrowsAllocatorAndBoundsScratchBeforeDispatch) {
  struct State {
    std::size_t live = 0, peak = 0, allocations = 0;
  } state;
  struct Handle {
    std::size_t bytes, alignment;
  };
  ml_cpu_compute::ExecutionAllocatorView allocator;
  allocator.context = &state;
  allocator.allocate = [](void *context, std::size_t bytes, std::size_t alignment) {
    auto &state = *static_cast<State *>(context);
    auto *handle = new Handle{bytes, alignment};
    void *data = ::operator new(bytes, std::align_val_t(alignment));
    state.live += bytes;
    state.peak = std::max(state.peak, state.live);
    ++state.allocations;
    return ml_cpu_compute::ExecutionAllocation{data, handle};
  };
  allocator.release = [](void *context, ml_cpu_compute::ExecutionAllocation allocation) noexcept {
    auto &state = *static_cast<State *>(context);
    auto *handle = static_cast<Handle *>(allocation.handle);
    state.live -= handle->bytes;
    ::operator delete(allocation.data, std::align_val_t(handle->alignment));
    delete handle;
  };
  AttentionDescriptor descriptor;
  descriptor.q_num_heads = descriptor.kv_num_heads = 2;
  const std::vector<int64_t> shape{2, 3, 2};
  const AttentionPlan plan(descriptor, AttentionLayout::kRank3, shape, shape, shape, {},
                           AttentionMaskKind::kNone);
  const std::vector<float> x{1, 2, 3, 4, 5, 6, 2, 4, 6, 8, 10, 12};
  const std::vector<float> w{0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1};
  std::vector<float> y(12), present(24);
  const auto bound = ml_cpu_compute::MicrosoftAttentionPeakMemoryBytes(plan, 2);
  {
    const ml_cpu_compute::ExecutionExecutorScope serial(nullptr);
    const ml_cpu_compute::ExecutionAllocatorScope scope(&allocator);
    ml_cpu_compute::ComputeMicrosoftAttentionFloat32(plan, 2, x.data(), w.data(), nullptr, nullptr,
                                                     nullptr, y.data(), present.data());
  }
  EXPECT_GT(state.allocations, 0u);
  EXPECT_EQ(state.live, 0u);
  EXPECT_LE(state.peak, bound);
  for (int i = 0; i < 12; ++i)
    EXPECT_NEAR(y[i], (i < 6 ? 1 : 2) * (i % 2 ? 4 : 3), 1.e-5);
}

TEST(MicrosoftAttentionCompute, AttentionWorkerScratchUsesAllocatorBeforeDispatch) {
  struct State {
    std::size_t live = 0, allocations = 0, largest_pre_dispatch_allocation = 0;
    std::size_t pre_dispatch_allocations = 0, allocations_during_dispatch = 0, dispatches = 0;
    bool dispatching = false;
  } state;
  struct Handle {
    std::size_t bytes, alignment;
  };
  ml_cpu_compute::ExecutionAllocatorView allocator;
  allocator.context = &state;
  allocator.allocate = [](void *context, std::size_t bytes, std::size_t alignment) {
    auto &state = *static_cast<State *>(context);
    auto *handle = new Handle{bytes, alignment};
    void *data = ::operator new(bytes, std::align_val_t(alignment));
    state.live += bytes;
    if (state.dispatching) {
      ++state.allocations_during_dispatch;
    } else {
      ++state.pre_dispatch_allocations;
      state.largest_pre_dispatch_allocation =
          std::max(state.largest_pre_dispatch_allocation, bytes);
    }
    ++state.allocations;
    return ml_cpu_compute::ExecutionAllocation{data, handle};
  };
  allocator.release = [](void *context, ml_cpu_compute::ExecutionAllocation allocation) noexcept {
    auto &state = *static_cast<State *>(context);
    auto *handle = static_cast<Handle *>(allocation.handle);
    state.live -= handle->bytes;
    ::operator delete(allocation.data, std::align_val_t(handle->alignment));
    delete handle;
  };
  ml_cpu_compute::ExecutionExecutorView executor;
  executor.context = &state;
  executor.effective_threads = 4;
  executor.run_blocks = [](void *context, std::int64_t num_blocks, void *task_context,
                           ml_cpu_compute::ExecutionBlockFn task) {
    auto &state = *static_cast<State *>(context);
    ++state.dispatches;
    state.dispatching = true;
    for (std::int64_t block = 0; block < num_blocks; ++block) {
      task(task_context, block);
    }
    state.dispatching = false;
  };

  ml_cpu_compute::AttentionDescriptor descriptor;
  descriptor.q_num_heads = descriptor.kv_num_heads = 1;
  const std::vector<std::int64_t> shape{1, 300, 1};
  const std::vector<std::int64_t> value_shape{1, 300, 128};
  const ml_cpu_compute::AttentionPlan plan(descriptor, ml_cpu_compute::AttentionLayout::kRank3,
                                           shape, shape, value_shape, {},
                                           ml_cpu_compute::AttentionMaskKind::kNone);
  const std::vector<float> q(300), k(300), v(300 * 128);
  std::vector<float> y(300 * 128);
  ml_cpu_compute::AttentionExecutionInfo tiled_info;
  ml_cpu_compute::AttentionDescriptor cached_descriptor;
  cached_descriptor.q_num_heads = cached_descriptor.kv_num_heads = 2;
  cached_descriptor.has_past_key = cached_descriptor.has_past_value = true;
  const std::vector<std::int64_t> cached_shape{4, 8, 64}, past_shape{4, 2, 1024, 32};
  const ml_cpu_compute::AttentionPlan cached_plan(
      cached_descriptor, ml_cpu_compute::AttentionLayout::kRank3, cached_shape, cached_shape,
      cached_shape, {}, ml_cpu_compute::AttentionMaskKind::kNone, past_shape, past_shape);
  const std::vector<float> cached_q(4 * 8 * 64), cached_k(4 * 8 * 64), cached_v(4 * 8 * 64);
  const std::vector<float> past_k(4 * 2 * 1024 * 32), past_v(4 * 2 * 1024 * 32);
  std::vector<float> cached_y(4 * 8 * 64);
  ml_cpu_compute::AttentionExecutionInfo streaming_info;
  std::size_t tiled_pre_dispatch_allocations = 0, tiled_dispatch_allocations = 0;
  std::size_t tiled_workspace_bytes = 0, streaming_dispatch_allocations = 0;
  {
    const ml_cpu_compute::ExecutionExecutorScope executor_scope(&executor);
    const ml_cpu_compute::ExecutionAllocatorScope allocator_scope(&allocator);
    ml_cpu_compute::ComputeAttentionFloat32Streaming(plan, q.data(), k.data(), v.data(), nullptr,
                                                     y.data(), nullptr, nullptr, nullptr,
                                                     &tiled_info);
    tiled_pre_dispatch_allocations = state.pre_dispatch_allocations;
    tiled_dispatch_allocations = state.allocations_during_dispatch;
    tiled_workspace_bytes = state.largest_pre_dispatch_allocation;
    ml_cpu_compute::ComputeAttentionFloat32Streaming(
        cached_plan, cached_q.data(), cached_k.data(), cached_v.data(), nullptr, cached_y.data(),
        past_k.data(), past_v.data(), nullptr, &streaming_info);
    streaming_dispatch_allocations = state.allocations_during_dispatch - tiled_dispatch_allocations;
  }
  EXPECT_EQ(tiled_info.path, ml_cpu_compute::AttentionExecutionPath::kTiled);
  EXPECT_EQ(tiled_info.query_tile, 128u);
  EXPECT_EQ(streaming_info.path, ml_cpu_compute::AttentionExecutionPath::kStreaming);
  EXPECT_EQ(tiled_pre_dispatch_allocations, 1u);
  EXPECT_EQ(state.pre_dispatch_allocations - tiled_pre_dispatch_allocations, 1u);
  EXPECT_EQ(tiled_dispatch_allocations, 0u);
  EXPECT_EQ(streaming_dispatch_allocations, 0u);
  EXPECT_EQ(state.dispatches, 2u);
  EXPECT_EQ(state.live, 0u);
  EXPECT_GE(tiled_workspace_bytes, 2 * (plan.head_dim + 1) * sizeof(float));
  EXPECT_TRUE(std::all_of(y.begin(), y.end(), [](float value) { return value == 0.0f; }));
  EXPECT_TRUE(
      std::all_of(cached_y.begin(), cached_y.end(), [](float value) { return value == 0.0f; }));
}
} // namespace
