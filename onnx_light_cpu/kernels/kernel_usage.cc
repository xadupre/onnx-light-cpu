// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_light_cpu/kernels/logical/not_kernel.h"
#include "onnx_light_cpu/kernels/math/abs_kernel.h"
#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"
#include "onnx_light_cpu/kernels/math/gemm_kernel.h"
#include "onnx_light_cpu/kernels/math/integer_matmul_kernel.h"

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
  static const std::vector<std::pair<std::string, std::string>> names = {
      {"Abs", AbsKernel::kName},
      {"Exp", ExpKernel::kName},
      {"Log", LogKernel::kName},
      {"Gemm", GemmKernel::kName},
      {"MatMulInteger", MatMulIntegerKernel::kName},
      {"QLinearMatMul", QLinearMatMulKernel::kName},
      {"Not", NotKernel::kName},
  };
  return names;
}

} // namespace onnx_light_cpu
