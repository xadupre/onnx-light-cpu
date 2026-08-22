// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_proto/onnx_helper.h"

#ifdef ONNX_LIGHT_CPU_HAS_BACKEND_TEST
#include "onnx_light_cpu/backend_test/collect_test_cases.h"
#endif

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace nb = nanobind;

namespace {

// One onnx-light-cpu kernel registration, with the enum fields
// (``KernelRegistration::device`` and ``KernelRegistration::types``)
// rendered as the human-readable strings ``registered_kernels()`` exposes to
// Python: ``domain, op_type, device, kernel_name, types, since_version,
// until_version``.
using RegisteredKernelTuple =
    std::tuple<std::string, std::string, std::string, std::string, std::vector<std::string>,
               std::optional<std::int64_t>, std::optional<std::int64_t>>;

std::vector<RegisteredKernelTuple> RegisteredKernelsForPython() {
  std::vector<RegisteredKernelTuple> result;
  const std::vector<onnx_light_cpu::KernelRegistration> records =
      onnx_light_cpu::CollectRegisteredKernels();
  result.reserve(records.size());
  for (const onnx_light_cpu::KernelRegistration &record : records) {
    std::vector<std::string> type_names;
    type_names.reserve(record.types.size());
    for (const auto &type : record.types) {
      type_names.emplace_back(ONNX_LIGHT_NAMESPACE::TensorProto::DataType_Name(type));
    }
    result.emplace_back(record.domain, record.op_type,
                        ONNX_LIGHT_NAMESPACE::core::symbolic::DeviceName(record.device),
                        record.kernel_name, std::move(type_names), record.since_version,
                        record.until_version);
  }
  return result;
}

} // namespace

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

  m.def("registered_kernels", &RegisteredKernelsForPython,
        "Returns one (domain, op_type, device, kernel_name, types, since_version, "
        "until_version) tuple per onnx-light-cpu kernel registration, sorted by "
        "(domain, op_type, device, kernel_name). Collected from "
        "CollectRegisteredKernels() without mutating onnx-light's shared "
        "KernelDispatchTable.");

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
