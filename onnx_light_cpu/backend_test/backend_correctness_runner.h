// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/kernels/register_kernels.h"

#include <cstddef>
#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {

/// The result of one skipped or failed backend correctness case.
struct BackendCaseResult {
  std::string op_type;
  std::string case_name;
  std::string reason;
};

/// Summary of the regular onnx-light backend correctness corpus.
struct BackendCorrectnessReport {
  size_t executed = 0;
  size_t passed = 0;
  std::vector<BackendCaseResult> skipped;
  std::vector<BackendCaseResult> failed;

  /// Formats all skipped and failed cases for test diagnostics.
  std::string Describe() const;
};

/// Registers the CPU kernels and runs every applicable ``TestMode::TEST`` case.
///
/// Backend cases are selected from the registrations returned by
/// :cpp:func:`CollectRegisteredKernels`, including their domain, opset range,
/// and input element types. Unsupported candidates are retained in ``skipped``;
/// a registration without an applicable case is retained in ``failed``.
BackendCorrectnessReport RunBackendCorrectnessTests(
    MicrosoftKernelImplementation implementation = MicrosoftKernelImplementation::OPTIMIZED);

} // namespace onnx_light_cpu::backend_test
