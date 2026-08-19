// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu {

/// Installs a single onnx-light-cpu kernel into onnx-light's shared
/// ``KernelDispatchTable`` for the CPU device (the per-operator
/// ``RegisterAbsKernel`` / ``RegisterGemmKernel`` / ... functions).
using KernelInstallFn = void (*)();

/// Describes one onnx-light-cpu kernel: the ONNX ``op_type`` it overrides, the
/// library-qualified name it records through :cpp:func:`RecordKernelUsage`, and
/// the function that installs it into onnx-light's dispatch table.
struct KernelRegistration {
  std::string op_type;
  std::string kernel_name;
  KernelInstallFn install;
};

/// Appends a kernel to the process-wide registry and returns a dummy value.
///
/// It is meant to initialise a namespace-scope static (see
/// :c:macro:`ONNX_LIGHT_CPU_REGISTER_KERNEL`) so every kernel translation unit
/// registers itself without any central bookkeeping. It only stores the
/// supplied pointers/names; it never touches onnx-light, so it is safe to call
/// during static initialisation before onnx-light's runtime is loaded.
int AddKernelRegistration(const char *op_type, const char *kernel_name, KernelInstallFn install);

/// Returns every self-registered kernel, sorted by ``op_type`` so the order is
/// stable regardless of static-initialisation order across translation units.
const std::vector<KernelRegistration> &KernelRegistrations();

} // namespace onnx_light_cpu

/// Self-registers a kernel so :cpp:func:`onnx_light_cpu::RegisterAllKernels` and
/// :cpp:func:`onnx_light_cpu::RegisteredKernelNames` pick it up automatically.
///
/// Place at namespace scope in a kernel's ``.cc`` file, after its installer is
/// defined, passing the ONNX ``op_type``, the kernel's ``kName`` and the
/// installer function (for example
/// ``ONNX_LIGHT_CPU_REGISTER_KERNEL("Abs", AbsKernel::kName, RegisterAbsKernel)``).
#define ONNX_LIGHT_CPU_REGISTER_KERNEL(op_type, kernel_name, install_fn)                           \
  namespace {                                                                                      \
  const int kOnnxLightCpuKernelReg_##install_fn =                                                  \
      ::onnx_light_cpu::AddKernelRegistration((op_type), (kernel_name), (install_fn));             \
  } // namespace
