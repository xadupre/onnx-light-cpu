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

} // namespace
