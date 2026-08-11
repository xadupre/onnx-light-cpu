// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Backend-test driver for the onnx-light-cpu kernels.
//
// onnx-light ships its ONNX backend test cases as a C++-registered registry
// (``onnx_light/onnx_extensions/backend_test/cases/``) exposed through
// :cpp:func:`onnx_light::core::backend_test::CollectTestCases`. Each entry is a
// ``TestCase`` bundling a single-node ``ModelProto`` with one or more
// input/output ``DataSet`` computed by onnx-light itself.
//
// Following onnx-light's own ``lib_onnx_backend_test`` pattern, this backend
// test lives in its own library (``lib_onnx_light_cpu_backend_test``) that
// takes a dependency on the onnx-light-cpu kernels. The functions below collect
// the backend test cases registered for an operator (covering every element
// type onnx-light registers for it), run each case's input through the
// corresponding SIMD-accelerated onnx-light-cpu kernel, and compare the output
// against the reference output shipped with the case (using the case's own
// ``rtol``/``atol``).
//
// Each function returns the list of human-readable failure descriptions found
// while running the cases; an empty vector means every case passed. Keeping the
// assertion mechanism out of this library (mirroring onnx-light's gtest-free
// ``lib_onnx_backend_test``) lets the thin C++ unit test drive it through any
// test framework.

#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {

/// Runs every onnx-light-registered ``Abs`` backend test case through the
/// onnx-light-cpu ``Abs`` kernel. Returns the failure descriptions (empty when
/// all cases pass).
std::vector<std::string> RunAbsBackendCases();

/// Runs every onnx-light-registered ``Exp`` backend test case through the
/// onnx-light-cpu ``Exp`` kernel.
std::vector<std::string> RunExpBackendCases();

/// Runs every onnx-light-registered ``Log`` backend test case through the
/// onnx-light-cpu ``Log`` kernel.
std::vector<std::string> RunLogBackendCases();

/// Runs every onnx-light-registered ``Not`` backend test case through the
/// onnx-light-cpu ``Not`` kernel.
std::vector<std::string> RunNotBackendCases();

/// Runs every onnx-light-registered ``Gemm`` backend test case through the
/// onnx-light-cpu ``Gemm`` kernel.
std::vector<std::string> RunGemmBackendCases();

} // namespace onnx_light_cpu::backend_test
