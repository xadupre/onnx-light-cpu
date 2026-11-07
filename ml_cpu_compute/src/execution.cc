// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/execution.h"

#include <new>
#include <stdexcept>

namespace ml_cpu_compute {
namespace {

const ExecutionExecutorView *&CurrentExecutorSlot() noexcept {
  thread_local const ExecutionExecutorView *executor = nullptr;
  return executor;
}

int &ExecutionDepthSlot() noexcept {
  thread_local int depth = 0;
  return depth;
}

int &ExecutorScopeDepthSlot() noexcept {
  thread_local int depth = 0;
  return depth;
}

const ExecutionAllocatorView *&CurrentAllocatorSlot() noexcept {
  thread_local const ExecutionAllocatorView *allocator = nullptr;
  return allocator;
}

int &AllocatorScopeDepthSlot() noexcept {
  thread_local int depth = 0;
  return depth;
}

} // namespace

const ExecutionExecutorView *CurrentExecutionExecutor() noexcept { return CurrentExecutorSlot(); }

bool HasExecutionExecutorScope() noexcept { return ExecutorScopeDepthSlot() != 0; }

ExecutionExecutorScope::ExecutionExecutorScope(const ExecutionExecutorView *executor) noexcept
    : previous_(CurrentExecutorSlot()) {
  CurrentExecutorSlot() = executor;
  ++ExecutorScopeDepthSlot();
}

ExecutionExecutorScope::~ExecutionExecutorScope() {
  CurrentExecutorSlot() = previous_;
  --ExecutorScopeDepthSlot();
}

const ExecutionAllocatorView *CurrentExecutionAllocator() noexcept {
  return CurrentAllocatorSlot();
}

bool HasExecutionAllocatorScope() noexcept { return AllocatorScopeDepthSlot() != 0; }

ExecutionAllocatorScope::ExecutionAllocatorScope(const ExecutionAllocatorView *allocator) noexcept
    : previous_(CurrentAllocatorSlot()) {
  CurrentAllocatorSlot() = allocator;
  ++AllocatorScopeDepthSlot();
}

ExecutionAllocatorScope::~ExecutionAllocatorScope() {
  CurrentAllocatorSlot() = previous_;
  --AllocatorScopeDepthSlot();
}

ExecutionBuffer::ExecutionBuffer(std::size_t bytes, std::size_t alignment) : alignment_(alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("ExecutionBuffer alignment must be a power of two.");
  }
  if (bytes == 0) {
    return;
  }
  if (const ExecutionAllocatorView *allocator = CurrentExecutionAllocator()) {
    if (allocator->allocate == nullptr || allocator->release == nullptr) {
      throw std::invalid_argument("ExecutionBuffer allocator requires allocate and release.");
    }
    allocator_ = *allocator;
    allocation_ = allocator_.allocate(allocator_.context, bytes, alignment);
    if (allocation_.data == nullptr ||
        reinterpret_cast<std::uintptr_t>(allocation_.data) % alignment != 0) {
      if (allocation_.data != nullptr || allocation_.handle != nullptr) {
        allocator_.release(allocator_.context, allocation_);
      }
      throw std::bad_alloc();
    }
  } else {
    allocation_.data = ::operator new(bytes, std::align_val_t{alignment});
  }
}

ExecutionBuffer::~ExecutionBuffer() {
  if (allocation_.data == nullptr) {
    return;
  }
  if (allocator_.release != nullptr) {
    allocator_.release(allocator_.context, allocation_);
  } else {
    ::operator delete(allocation_.data, std::align_val_t{alignment_});
  }
}

namespace detail {

ExecutionRegionScope::ExecutionRegionScope() noexcept { ++ExecutionDepthSlot(); }

ExecutionRegionScope::~ExecutionRegionScope() { --ExecutionDepthSlot(); }

int ExecutionRegionDepth() noexcept { return ExecutionDepthSlot(); }

} // namespace detail
} // namespace ml_cpu_compute
