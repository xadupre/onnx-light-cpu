// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"

namespace onnx_light_cpu {

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
