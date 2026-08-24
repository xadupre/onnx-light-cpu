// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"

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
  // Derived from ``CollectRegisteredKernels()`` (the same structured
  // inventory :cpp:func:`onnx_light_cpu::CollectRegisteredKernels` builds
  // from every ``Register*Kernel[s]`` call) rather than a second,
  // hand-maintained ``op_type -> kernel name`` list, so the two never drift
  // apart. The result uses the inventory's deterministic
  // ``(domain, op_type, device, kernel_name)`` order.
  static const std::vector<std::pair<std::string, std::string>> names = [] {
    std::vector<std::pair<std::string, std::string>> result;
    for (const KernelRegistration &record : CollectRegisteredKernels()) {
      result.emplace_back(record.op_type, record.kernel_name);
    }
    return result;
  }();
  return names;
}

} // namespace onnx_light_cpu
