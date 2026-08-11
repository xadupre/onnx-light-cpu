// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Thin gtest wrapper that drives the onnx-light-cpu backend test library
// (``lib_onnx_light_cpu_backend_test``). The backend test itself lives in that
// library, which takes a dependency on the onnx-light-cpu kernels and mirrors
// onnx-light's own ``lib_onnx_backend_test`` pattern. Each function collects
// onnx-light's C++-registered backend test cases (``CollectTestCases``) for an
// operator, runs them through the corresponding SIMD-accelerated kernel and
// returns any mismatches, which this test surfaces as gtest failures.

#include "onnx_light_cpu/backend_test/kernel_backend_test.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

// Turns the failure descriptions returned by a backend test runner into a
// single gtest failure message.
std::string Describe(const std::vector<std::string> &failures) {
  std::string message;
  for (const std::string &failure : failures) {
    message += failure;
    message += '\n';
  }
  return message;
}

TEST(OnnxLightBackendKernels, AbsRunsOnBackendTestCases) {
  const std::vector<std::string> failures = onnx_light_cpu::backend_test::RunAbsBackendCases();
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, ExpRunsOnBackendTestCases) {
  const std::vector<std::string> failures = onnx_light_cpu::backend_test::RunExpBackendCases();
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, LogRunsOnBackendTestCases) {
  const std::vector<std::string> failures = onnx_light_cpu::backend_test::RunLogBackendCases();
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, NotRunsOnBackendTestCases) {
  const std::vector<std::string> failures = onnx_light_cpu::backend_test::RunNotBackendCases();
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

TEST(OnnxLightBackendKernels, GemmRunsOnBackendTestCases) {
  const std::vector<std::string> failures = onnx_light_cpu::backend_test::RunGemmBackendCases();
  EXPECT_TRUE(failures.empty()) << Describe(failures);
}

} // namespace
