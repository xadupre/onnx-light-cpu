// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_light_cpu/kernels/kernel_registry.h"

namespace onnx_light_cpu {

void RegisterAllKernels() {
  // Every kernel self-registers its installer through
  // ONNX_LIGHT_CPU_REGISTER_KERNEL, so this simply drives the registry instead
  // of maintaining a hand-written list of RegisterXKernel() calls.
  for (const KernelRegistration &registration : KernelRegistrations()) {
    registration.install();
  }
}

} // namespace onnx_light_cpu
