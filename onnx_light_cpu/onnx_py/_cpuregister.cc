// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>

#include "onnx_light_cpu/kernels/register_kernels.h"

namespace nb = nanobind;

NB_MODULE(_cpuregister, m) {
  m.doc() = "Python bindings for onnx-light-cpu kernel registration: registers "
            "the SIMD-accelerated Abs/Exp/Log/Not kernel classes into "
            "onnx-light's C++ KernelDispatchTable for the CPU device.";

  m.def(
      "register_all_kernels", []() { onnx_light_cpu::RegisterAllKernels(); },
      "Registers every onnx-light-cpu kernel class (Abs, Exp, Log, Not) into "
      "onnx-light's shared KernelDispatchTable for the CPU device, replacing "
      "the corresponding built-in entries for the default ONNX domain.");
}
