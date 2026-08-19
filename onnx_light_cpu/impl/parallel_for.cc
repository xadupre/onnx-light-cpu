// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/parallel_for.h"

#include <atomic>

namespace onnx_light_cpu {
namespace {

const ParallelForExecutorView *&CurrentExecutorSlot() noexcept {
  thread_local const ParallelForExecutorView *executor = nullptr;
  return executor;
}

std::atomic<uint64_t> &StandalonePoolCreations() noexcept {
  static std::atomic<uint64_t> count{0};
  return count;
}

} // namespace

namespace detail {

void RecordStandaloneThreadPoolCreation() noexcept {
  StandalonePoolCreations().fetch_add(1, std::memory_order_relaxed);
}

} // namespace detail

const ParallelForExecutorView *CurrentParallelForExecutor() noexcept {
  return CurrentExecutorSlot();
}

ParallelForExecutorScope::ParallelForExecutorScope(const ParallelForExecutorView *executor) noexcept
    : previous_(CurrentExecutorSlot()) {
  CurrentExecutorSlot() = executor;
}

ParallelForExecutorScope::~ParallelForExecutorScope() { CurrentExecutorSlot() = previous_; }

uint64_t StandaloneThreadPoolCreationCount() noexcept {
  return StandalonePoolCreations().load(std::memory_order_relaxed);
}

} // namespace onnx_light_cpu
