// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "onnx_light_cpu/gradient/com_microsoft/gradients.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/kernels/register_kernels.h"
#include "onnx_light_cpu/patterns/com_microsoft/patterns.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"
#include "onnx_light_cpu/shapes/com_microsoft/shape_inference.h"

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
using OperatorSupportTuple =
    std::tuple<std::string, std::string, std::string, std::string, std::vector<std::string>, bool>;

std::vector<RegisteredKernelTuple>
RegisteredKernelsForPython(onnx_light_cpu::MicrosoftKernelImplementation implementation) {
  std::vector<RegisteredKernelTuple> result;
  const std::vector<onnx_light_cpu::KernelRegistration> records =
      onnx_light_cpu::CollectRegisteredKernels(implementation);
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

std::vector<OperatorSupportTuple> OperatorSupportForPython() {
  std::vector<OperatorSupportTuple> result;
  for (const auto &record : onnx_light_cpu::CollectOperatorSupport()) {
    result.emplace_back(record.domain, record.op_type, record.shape_inference_function,
                        record.peak_memory_function, record.fusion_patterns, record.has_gradient);
  }
  return result;
}

} // namespace

NB_MODULE(_cpuregister, m) {
  m.doc() = "Python bindings for registering and inspecting onnx-light-cpu "
            "kernels and custom operator support.";

  nb::enum_<onnx_light_cpu::MicrosoftKernelImplementation>(m, "MicrosoftKernelImplementation")
      .value("NAIVE", onnx_light_cpu::MicrosoftKernelImplementation::NAIVE)
      .value("OPTIMIZED", onnx_light_cpu::MicrosoftKernelImplementation::OPTIMIZED);

  m.def(
      "register_all_kernels",
      [](onnx_light_cpu::MicrosoftKernelImplementation implementation) {
        onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions();
        onnx_light_cpu::RegisterCustomOperatorPatterns();
        onnx_light_cpu::RegisterAllKernels(implementation);
      },
      nb::arg("microsoft_implementation") =
          onnx_light_cpu::MicrosoftKernelImplementation::OPTIMIZED,
      "Registers every onnx-light-cpu kernel class into onnx-light's shared "
      "KernelDispatchTable for the CPU device.");

  m.def("microsoft_op_schemas", &onnx_light_cpu::GetMicrosoftOpSchemasWithHistory,
        nb::arg("op_type") = std::string(), nb::arg("init_doc") = true,
        "Returns the LightOpSchema history provided for com.microsoft operators.");

  m.def(
      "register_custom_operator_support",
      []() {
        onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions();
        onnx_light_cpu::RegisterCustomOperatorPatterns();
      },
      "Registers com.microsoft shape inference, peak-memory functions, and fusion patterns.");

  m.def("register_custom_gradients", &onnx_light_cpu::RegisterCustomOperatorGradients,
        nb::arg("registry"),
        "Adds the com.microsoft BiasGelu, CDist, and GroupQueryAttention backward rules to a "
        "GradRegistry.");

  m.def(
      "registered_kernel_names", []() { return onnx_light_cpu::RegisteredKernelNames(); },
      "Returns the (op_type, kernel name) pairs of every onnx-light-cpu kernel, "
      "e.g. ('Abs', 'onnx_light_cpu::Abs'). The kernel name is the "
      "library-qualified name each kernel records when it runs, so callers can "
      "check the accelerated kernels are the ones actually used.");

  m.def("registered_kernels", &RegisteredKernelsForPython,
        nb::arg("microsoft_implementation") =
            onnx_light_cpu::MicrosoftKernelImplementation::OPTIMIZED,
        "Returns one (domain, op_type, device, kernel_name, types, since_version, "
        "until_version) tuple per onnx-light-cpu kernel registration, sorted by "
        "(domain, op_type, device, kernel_name). Collected from "
        "CollectRegisteredKernels() without mutating onnx-light's shared "
        "KernelDispatchTable.");

  m.def("operator_support", &OperatorSupportForPython,
        "Returns one (domain, op_type, shape_inference_function, "
        "peak_memory_function, fusion_patterns, has_gradient) tuple per custom "
        "operator supported by onnx-light-cpu, without mutating any registry.");

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
