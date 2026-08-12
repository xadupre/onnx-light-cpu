// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Backend-test *registration* library for the onnx-light-cpu kernels.
//
// onnx-light exposes a global backend test case registry
// (``onnx_core/backend_test/test_case_registry.h``): every operator category
// registers a collector through
// :cpp:func:`onnx_light::core::backend_test::RegisterTestCasesCollector`, and
// :cpp:func:`onnx_light::core::backend_test::CollectTestCases` walks the
// registered collectors to materialize the cases. Both the C++ runtime and the
// Python ``onnx_light.onnx.backend.collect_test_cases`` API read that same
// registry.
//
// Mirroring onnx-light's own ``lib_onnx_backend_test`` pattern, this library
// takes a dependency on the onnx-light-cpu kernels and *registers* backend test
// cases for the accelerated ``Abs``/``Exp``/``Log``/``Gemm``/``Not`` kernels
// (covering every element type each kernel implements) into that shared
// onnx-light registry. Following onnx-light's per-operator layout, the actual
// cases live in one header + one ``.cc`` per kernel
// (``abs_backend_test.{h,cc}``, ``exp_backend_test.{h,cc}``,
// ``log_backend_test.{h,cc}``, ``gemm_backend_test.{h,cc}``,
// ``not_backend_test.{h,cc}``); each registers both correctness
// (``TestMode::TEST``) and benchmark (``TestMode::BENCHMARK``) cases. This
// translation unit is only the *collector* that dispatches by op_type to those
// helpers (mirroring onnx-light's ``collect_<category>_cases.cc``).
//
// It does **not** run or check anything itself: the cases are executed and
// validated by the unit tests, which register them, register the kernels and
// then drive them through onnx-light's regular API (``CollectTestCases`` + the
// runtime in C++, ``collect_test_cases`` + ``ReferenceEvaluator`` in Python).

#include "onnx_core/backend_test/test_case.h"

#include <string>
#include <vector>

namespace onnx_light_cpu::backend_test {

/// Appends the onnx-light-cpu backend test cases matching ``op_type`` (or every
/// case when ``op_type`` is empty) to ``registry``. This is the collector that
/// :cpp:func:`RegisterCpuKernelBackendTestCases` installs into onnx-light's
/// global registry; it is exposed so tests can also materialize the cpu cases
/// directly.
void CollectCpuKernelTestCases(
    std::vector<ONNX_LIGHT_NAMESPACE::core::backend_test::TestCase> &registry,
    const std::string &op_type,
    ONNX_LIGHT_NAMESPACE::core::backend_test::TestMode mode =
        ONNX_LIGHT_NAMESPACE::core::backend_test::TestMode::TEST);

/// Registers the onnx-light-cpu backend test cases into onnx-light's shared
/// backend test case registry (via
/// :cpp:func:`onnx_light::core::backend_test::RegisterTestCasesCollector`).
///
/// After this call the cpu cases are reachable through onnx-light's regular
/// :cpp:func:`onnx_light::core::backend_test::CollectTestCases` (C++) and
/// ``onnx_light.onnx.backend.collect_test_cases`` (Python) APIs, exactly like a
/// built-in onnx-light operator category. The registration is performed at most
/// once regardless of how many times this function is called, so it is safe to
/// invoke from every test's setup.
void RegisterCpuKernelBackendTestCases();

} // namespace onnx_light_cpu::backend_test
