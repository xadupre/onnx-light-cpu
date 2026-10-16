// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

// Include in only one translation unit per executable: this replaces global
// allocation operators to observe kernel allocations, including worker threads.
namespace {
std::atomic<bool> g_alloc_recording{false};
std::atomic<std::size_t> g_alloc_peak{0};

void RecordAllocation(std::size_t bytes) {
  if (!g_alloc_recording.load(std::memory_order_relaxed)) {
    return;
  }
  std::size_t previous = g_alloc_peak.load(std::memory_order_relaxed);
  while (bytes > previous &&
         !g_alloc_peak.compare_exchange_weak(previous, bytes, std::memory_order_relaxed)) {
  }
}

void *AlignedAlloc(std::size_t alignment, std::size_t size) {
#if defined(_MSC_VER)
  return _aligned_malloc(size, alignment);
#else
  return std::aligned_alloc(alignment, size);
#endif
}

void AlignedFree(void *pointer) noexcept {
#if defined(_MSC_VER)
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
}

// Warm up one-time allocations, then measure the largest single heap request.
template <typename Fn> std::size_t PeakAllocationBytes(Fn fn) {
  fn();
  g_alloc_peak.store(0, std::memory_order_relaxed);
  struct RecordingScope {
    RecordingScope() { g_alloc_recording.store(true, std::memory_order_relaxed); }
    ~RecordingScope() { g_alloc_recording.store(false, std::memory_order_relaxed); }
  } recording;
  fn();
  return g_alloc_peak.load(std::memory_order_relaxed);
}
} // namespace

void *operator new(std::size_t bytes) {
  RecordAllocation(bytes);
  void *pointer = std::malloc(bytes != 0 ? bytes : 1);
  if (pointer == nullptr) {
    throw std::bad_alloc();
  }
  return pointer;
}

void *operator new[](std::size_t bytes) { return ::operator new(bytes); }

void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete[](void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void *pointer, std::size_t) noexcept { std::free(pointer); }

void *operator new(std::size_t bytes, std::align_val_t alignment) {
  RecordAllocation(bytes);
  const std::size_t align = static_cast<std::size_t>(alignment);
  const std::size_t size = (bytes + align - 1) / align * align;
  void *pointer = AlignedAlloc(align, size != 0 ? size : align);
  if (pointer == nullptr) {
    throw std::bad_alloc();
  }
  return pointer;
}

void *operator new[](std::size_t bytes, std::align_val_t alignment) {
  return ::operator new(bytes, alignment);
}

void operator delete(void *pointer, std::align_val_t) noexcept { AlignedFree(pointer); }
void operator delete[](void *pointer, std::align_val_t) noexcept { AlignedFree(pointer); }
void operator delete(void *pointer, std::size_t, std::align_val_t) noexcept {
  AlignedFree(pointer);
}
void operator delete[](void *pointer, std::size_t, std::align_val_t) noexcept {
  AlignedFree(pointer);
}
