// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/kernel_registry.h"

#include <algorithm>
#include <mutex>

namespace onnx_light_cpu {

namespace {

// Process-wide kernel registry. Kernels append to it from namespace-scope
// static initialisers (via ONNX_LIGHT_CPU_REGISTER_KERNEL), which can run on
// different threads if translation units are initialised concurrently, so the
// vector is guarded by a mutex.
std::vector<KernelRegistration> &Registry() {
  static std::vector<KernelRegistration> registry;
  return registry;
}

std::mutex &RegistryMutex() {
  static std::mutex mutex;
  return mutex;
}

} // namespace

int AddKernelRegistration(const char *op_type, const char *kernel_name, KernelInstallFn install) {
  std::lock_guard<std::mutex> guard(RegistryMutex());
  Registry().push_back(KernelRegistration{op_type, kernel_name, install});
  return 0;
}

const std::vector<KernelRegistration> &KernelRegistrations() {
  // Registrations only happen during static initialisation (through
  // ONNX_LIGHT_CPU_REGISTER_KERNEL), so snapshot the registry into an immutable
  // sorted copy exactly once, on the first call. Returning a reference to that
  // stable static avoids both re-sorting on every call and handing back a
  // reference to the still-mutable registry.
  static const std::vector<KernelRegistration> sorted = [] {
    std::lock_guard<std::mutex> guard(RegistryMutex());
    std::vector<KernelRegistration> copy = Registry();
    std::sort(copy.begin(), copy.end(),
              [](const KernelRegistration &a, const KernelRegistration &b) {
                return a.op_type < b.op_type;
              });
    return copy;
  }();
  return sorted;
}

} // namespace onnx_light_cpu
