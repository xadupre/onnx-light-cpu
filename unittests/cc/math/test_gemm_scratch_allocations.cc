// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/execution.h"
#include "ml_cpu_compute/math/gemm/gemm_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <thread>
#include <vector>

namespace {

// GEMM packing and split-K buffers use aligned new. Count their live payloads,
// not the output, thread-launch bookkeeping, or merely the largest allocation.
struct AllocationHeader {
  void *allocation;
  std::size_t bytes;
  std::uint64_t generation;
};

std::atomic<std::uint64_t> generation{0};
std::atomic<bool> recording{false};
std::atomic<std::size_t> live_bytes{0}, peak_bytes{0};

void RecordPeak(std::size_t bytes) {
  std::size_t previous = peak_bytes.load();
  while (previous < bytes && !peak_bytes.compare_exchange_weak(previous, bytes)) {
  }
}

template <typename Fn> std::size_t PackingPeak(Fn fn) {
  fn();
  ++generation;
  live_bytes = 0;
  peak_bytes = 0;
  struct RecordingScope {
    ~RecordingScope() { recording = false; }
  } scope;
  recording = true;
  fn();
  recording = false;
  EXPECT_EQ(live_bytes.load(), 0U);
  return peak_bytes.load();
}

struct ConcurrentExecutor {
  static void Run(void *, std::int64_t blocks, void *context,
                  ml_cpu_compute::ExecutionBlockFn task) {
    std::vector<std::thread> threads;
    for (std::int64_t block = 1; block < blocks; ++block) {
      threads.emplace_back([=] { task(context, block); });
    }
    task(context, 0);
    for (auto &thread : threads) {
      thread.join();
    }
  }
};

} // namespace

void *operator new(std::size_t bytes, std::align_val_t alignment) {
  const std::size_t align =
      std::max(static_cast<std::size_t>(alignment), alignof(AllocationHeader));
  const std::size_t overhead = align - 1 + sizeof(AllocationHeader);
  if (bytes > std::numeric_limits<std::size_t>::max() - overhead) {
    throw std::bad_alloc();
  }
  void *allocation = std::malloc(bytes + overhead);
  if (allocation == nullptr) {
    throw std::bad_alloc();
  }
  const auto address = reinterpret_cast<std::uintptr_t>(allocation) + sizeof(AllocationHeader);
  const auto aligned = (address + align - 1) & ~(align - 1);
  auto *header = reinterpret_cast<AllocationHeader *>(aligned) - 1;
  const bool tracked = recording.load();
  *header = {allocation, bytes, tracked ? generation.load() : 0};
  if (tracked) {
    RecordPeak(live_bytes.fetch_add(bytes) + bytes);
  }
  return reinterpret_cast<void *>(aligned);
}

void operator delete(void *pointer, std::align_val_t) noexcept {
  if (pointer == nullptr) {
    return;
  }
  const auto *header = static_cast<AllocationHeader *>(pointer) - 1;
  if (header->generation != 0 && header->generation == generation.load()) {
    live_bytes.fetch_sub(header->bytes);
  }
  std::free(header->allocation);
}

void *operator new[](std::size_t bytes, std::align_val_t alignment) {
  return ::operator new(bytes, alignment);
}
void operator delete[](void *pointer, std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}
void operator delete(void *pointer, std::size_t, std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}
void operator delete[](void *pointer, std::size_t, std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}

namespace {

TEST(GemmScratchAllocations, ConcurrentPackingAndSplitKFitPlanBounds) {
  for (const auto shape : {std::array<std::size_t, 3>{128, 513, 257}, {32, 32, 16384}}) {
    const auto [m, n, k] = shape;
    const ml_cpu_compute::GemmPlan<float> f32({false, false, m, n, k});
    const ml_cpu_compute::GemmPlan<double> f64({false, false, m, n, k});
    const ml_cpu_compute::GemmHalfPlan half({false, false, false, m, n, k});
    const ml_cpu_compute::GemmHalfPlan bf16({true, false, false, m, n, k});
    const std::vector<float> a(m * k, 0.25f), b(k * n, 0.5f);
    const std::vector<double> ad(m * k, 0.25), bd(k * n, 0.5);
    const std::vector<std::uint16_t> ah(m * k, 0x3400), bh(k * n, 0x3800);
    const std::vector<std::uint16_t> ab(m * k, 0x3e80), bb(k * n, 0x3f00);
    std::vector<float> y(m * n);
    std::vector<double> yd(m * n);
    ml_cpu_compute::GemmEpilogue<float> epilogue;
    ml_cpu_compute::ExecutionExecutorView executor{nullptr, 4, &ConcurrentExecutor::Run};
    ml_cpu_compute::ExecutionExecutorScope scope(&executor);
    const std::size_t float_peak =
        PackingPeak([&] { f32.Execute(a.data(), b.data(), nullptr, y.data()); });
    EXPECT_GT(float_peak, 0U);
    EXPECT_LE(float_peak, f32.ScratchBytes(4));
    EXPECT_LE(PackingPeak([&] { f64.Execute(ad.data(), bd.data(), nullptr, yd.data()); }),
              f64.ScratchBytes(4));
    EXPECT_LE(PackingPeak([&] { half.Execute(ah.data(), bh.data(), epilogue, y.data()); }),
              half.ScratchBytes(4));
    EXPECT_LE(PackingPeak([&] { bf16.Execute(ab.data(), bb.data(), epilogue, y.data()); }),
              bf16.ScratchBytes(4));
  }
}

TEST(GemmScratchAllocations, ConcurrentBroadcastProductsFitPlanBounds) {
  constexpr std::size_t batches = 4, m = 128, n = 513, k = 257;
  const ml_cpu_compute::MatMulPlan<float> plan(std::array<std::size_t, 3>{batches, m, k},
                                               std::array<std::size_t, 2>{k, n});
  const ml_cpu_compute::GemmHalfPlan half({false, false, false, m, n, k});
  const std::vector<float> a(batches * m * k, 0.25f), b(k * n, 0.5f);
  const std::vector<std::uint16_t> ah(batches * m * k, 0x3400), bh(k * n, 0x3800);
  std::vector<float> workspace(batches * m * n);
  std::vector<std::uint16_t> y(batches * m * n);
  ml_cpu_compute::GemmEpilogue<float> epilogue;
  epilogue.output_conversion = ml_cpu_compute::GemmOutputConversion::kFloat16;
  epilogue.converted_output = y.data();
  ml_cpu_compute::ExecutionExecutorView executor{nullptr, 4, &ConcurrentExecutor::Run};
  ml_cpu_compute::ExecutionExecutorScope scope(&executor);
  EXPECT_LE(PackingPeak([&] { plan.Execute(a.data(), b.data(), workspace.data()); }),
            plan.ScratchBytes(4));
  const std::size_t packing = PackingPeak(
      [&] { plan.ExecuteHalf(half, ah.data(), bh.data(), epilogue, workspace.data()); });
  EXPECT_LE(packing + workspace.size() * sizeof(float), plan.HalfScratchBytes(false, 4));
}

TEST(GemmScratchAllocations, ChangingTransposedAShapeDoesNotRetainPackedCapacity) {
  constexpr std::size_t n = 65, k = 128;
  const ml_cpu_compute::GemmPlan<float> large(
      ml_cpu_compute::GemmPlanOptions<float>{true, false, 4096, n, k});
  const ml_cpu_compute::GemmPlan<float> small(
      ml_cpu_compute::GemmPlanOptions<float>{true, false, 64, n, k});
  const std::vector<float> large_a(4096 * k, 0.25f), small_a(64 * k, 0.25f);
  const std::vector<float> b(k * n, 0.5f);
  std::vector<float> large_y(4096 * n), small_y(64 * n);
  ml_cpu_compute::ExecutionExecutorScope serial(nullptr);

  large.Execute(large_a.data(), b.data(), nullptr, large_y.data());
  EXPECT_EQ(small.ScratchBytes(1), 64U * k * sizeof(float));
  EXPECT_EQ(PackingPeak([&] { small.Execute(small_a.data(), b.data(), nullptr, small_y.data()); }),
            small.ScratchBytes(1));
}

TEST(GemmScratchAllocations, CustomAvx512ColumnOverrideFitsPlanBound) {
  if (ml_cpu_compute::EffectiveGemmSimdLevel() < ml_cpu_compute::SimdLevel::kAVX512) {
    GTEST_SKIP() << "Requires AVX-512 to select the 16-column FP32 packing override.";
  }
  ml_cpu_compute::GemmPlanOptions<float> options;
  options.trans_a = true;
  options.m = 128;
  options.n = 4096;
  options.k = 256;
  options.blocking = {128, 1, 256};
  options.maximum_participants = 1;
  const ml_cpu_compute::GemmPlan<float> plan(options);
  const std::vector<float> a(options.m * options.k, 0.25f), b(options.k * options.n, 0.5f);
  std::vector<float> y(options.m * options.n);
  ml_cpu_compute::ExecutionExecutorScope serial(nullptr);
  EXPECT_EQ(plan.ScratchBytes(1), 147456U);
  EXPECT_EQ(PackingPeak([&] { plan.Execute(a.data(), b.data(), nullptr, y.data()); }), 147456U);
}

template <typename T> void CheckSubMrBlockingFitsPlanBound() {
  for (const bool trans_b : {false, true}) {
    ml_cpu_compute::GemmPlanOptions<T> options;
    options.trans_a = true;
    options.trans_b = trans_b;
    options.m = 24;
    options.n = 4096;
    options.k = 256;
    options.blocking = {1, 4096, 256};
    options.maximum_participants = 1;
    options.m = 2 * ml_cpu_compute::GemmPlan<T>(options).blocking().mr;
    const ml_cpu_compute::GemmPlan<T> plan(options);
    SCOPED_TRACE(::testing::Message()
                 << "element_size=" << sizeof(T) << ", M=" << options.m << ", trans_b=" << trans_b);
    ASSERT_LT(plan.blocking().mc, plan.blocking().mr);
    const std::vector<T> a(options.m * options.k, T(0.25)), b(options.k * options.n, T(0.5));
    std::vector<T> y(options.m * options.n);
    ml_cpu_compute::ExecutionExecutorScope serial(nullptr);
    const std::size_t peak =
        PackingPeak([&] { plan.Execute(a.data(), b.data(), nullptr, y.data()); });
    EXPECT_GE(peak, options.k * options.n * sizeof(T));
    EXPECT_LE(peak, plan.ScratchBytes(1));
  }
}

TEST(GemmScratchAllocations, CustomSubMrBlockingFitsPlanBound) {
  CheckSubMrBlockingFitsPlanBound<float>();
  CheckSubMrBlockingFitsPlanBound<double>();
}

TEST(GemmScratchAllocations, CustomHalfBlockingFitsMatMulPlanEstimate) {
  constexpr std::size_t m = 128, n = 4096, k = 512;
  const ml_cpu_compute::MatMulPlan<float> matmul(m, n, k, false, true);
  ml_cpu_compute::GemmHalfPlanOptions options{true, false, true, m, n, k};
  options.blocking = {128, 4096, 512};
  options.maximum_participants = 1;
  const ml_cpu_compute::GemmHalfPlan half(options);
  const std::vector<std::uint16_t> a(m * k, 0x3e80), b(n * k, 0x3f00);
  std::vector<float> workspace(m * n);
  std::vector<std::uint16_t> output(m * n);
  ml_cpu_compute::GemmEpilogue<float> epilogue;
  epilogue.output_conversion = ml_cpu_compute::GemmOutputConversion::kBFloat16;
  epilogue.converted_output = output.data();
  ml_cpu_compute::ExecutionExecutorScope serial(nullptr);
  const std::size_t workspace_bytes = workspace.size() * sizeof(float);
  ASSERT_GE(half.ScratchBytes(1), 8U * 1024 * 1024);
  const std::size_t packing = PackingPeak(
      [&] { matmul.ExecuteHalf(half, a.data(), b.data(), epilogue, workspace.data()); });
  EXPECT_LE(packing + workspace_bytes, matmul.HalfScratchBytes(half, 1));
  EXPECT_THROW(
      matmul.HalfScratchBytes(ml_cpu_compute::GemmHalfPlan({true, false, false, m, n, k}), 1),
      std::invalid_argument);
}

} // namespace
