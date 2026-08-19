// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_light_cpu/kernels/kernel_registry.h"

#include <atomic>
#include <mutex>

namespace onnx_light_cpu {

namespace {

// Process-wide kernel-usage log. Guarded by ``UsageMutex`` because a model can
// dispatch nodes from more than one thread; the kernels themselves record their
// own name here from :cpp:func:`KernelBase::Run`.
std::vector<std::string> &UsageLog() {
  static std::vector<std::string> log;
  return log;
}

std::mutex &UsageMutex() {
  static std::mutex mutex;
  return mutex;
}

std::atomic<bool> &UsageRecordingEnabled() {
  static std::atomic<bool> enabled{true};
  return enabled;
}

} // namespace

void RecordKernelUsage(std::string_view name) {
  if (!UsageRecordingEnabled().load(std::memory_order_relaxed)) {
    return;
  }
  std::lock_guard<std::mutex> guard(UsageMutex());
  UsageLog().emplace_back(name);
}

std::vector<std::string> UsedKernelNames() {
  std::lock_guard<std::mutex> guard(UsageMutex());
  return UsageLog();
}

void ClearUsedKernelNames() {
  std::lock_guard<std::mutex> guard(UsageMutex());
  UsageLog().clear();
}

void SetKernelUsageRecording(bool enabled) noexcept {
  UsageRecordingEnabled().store(enabled, std::memory_order_relaxed);
}

const std::vector<std::pair<std::string, std::string>> &RegisteredKernelNames() {
  // Built once from the self-registration registry (see kernel_registry.h) so
  // adding a kernel needs no edit here: the registry already lists every
  // {op_type, kName} pair, sorted by op_type for a stable order.
  static const std::vector<std::pair<std::string, std::string>> names = [] {
    std::vector<std::pair<std::string, std::string>> pairs;
    for (const KernelRegistration &registration : KernelRegistrations()) {
      pairs.emplace_back(registration.op_type, registration.kernel_name);
    }
    return pairs;
  }();
  return names;
}

} // namespace onnx_light_cpu
