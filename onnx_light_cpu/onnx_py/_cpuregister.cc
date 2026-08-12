// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#ifdef ONNX_LIGHT_CPU_HAS_BACKEND_TEST
#include "onnx_light_cpu/backend_test/kernel_backend_test.h"
#endif

namespace nb = nanobind;

NB_MODULE(_cpuregister, m) {
  m.doc() = "Python bindings for onnx-light-cpu kernel registration: registers "
            "the SIMD-accelerated Abs/Exp/Log/Gemm/Not kernel classes into "
            "onnx-light's C++ KernelDispatchTable for the CPU device.";

  m.def(
      "register_all_kernels", []() { onnx_light_cpu::RegisterAllKernels(); },
      "Registers every onnx-light-cpu kernel class (Abs, Exp, Log, Gemm, Not) into "
      "onnx-light's shared KernelDispatchTable for the CPU device, replacing "
      "the corresponding built-in entries for the default ONNX domain.");

  m.def(
      "registered_kernel_names", []() { return onnx_light_cpu::RegisteredKernelNames(); },
      "Returns the (op_type, kernel name) pairs of every onnx-light-cpu kernel, "
      "e.g. ('Abs', 'onnx_light_cpu::Abs'). The kernel name is the "
      "library-qualified name each kernel records when it runs, so callers can "
      "check the accelerated kernels are the ones actually used.");

  m.def(
      "used_kernel_names", []() { return onnx_light_cpu::UsedKernelNames(); },
      "Returns the library-qualified names of the onnx-light-cpu kernels that "
      "ran since the last clear_used_kernel_names() call, in invocation order.");

  m.def(
      "clear_used_kernel_names", []() { onnx_light_cpu::ClearUsedKernelNames(); },
      "Clears the record of onnx-light-cpu kernels that have run.");

  m.def("set_kernel_usage_recording", &onnx_light_cpu::SetKernelUsageRecording, nb::arg("enabled"),
        "Enables or disables per-invocation kernel usage recording. Disabling it "
        "removes diagnostic logging overhead from performance measurements.");

#ifdef ONNX_LIGHT_CPU_HAS_BACKEND_TEST
  m.def(
      "register_backend_test_cases",
      []() { onnx_light_cpu::backend_test::RegisterCpuKernelBackendTestCases(); },
      "Registers the onnx-light-cpu backend test cases (named ``test_cpu_*``, "
      "covering every dtype each kernel implements) into onnx-light's shared "
      "backend test case registry. After this call they are returned by "
      "onnx_light.onnx.backend.collect_test_cases alongside onnx-light's own "
      "cases. Registration is process-wide and idempotent.");

  m.attr("has_backend_test_cases") = true;
#else
  m.attr("has_backend_test_cases") = false;
#endif
}
