// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Backend-test *registration* helper for the onnx-light-cpu ``Abs`` kernel.
//
// Mirrors onnx-light's per-operator ``cases_<op>.cc`` layout (one header + one
// ``.cc`` per kernel): this translation unit registers the ``Abs`` backend test
// cases into a caller-provided registry. Both correctness (``TestMode::TEST``)
// and benchmark (``TestMode::BENCHMARK``) cases are registered, exactly like
// onnx-light's own ``RegisterAbsCases``.

#include "onnx_core/backend_test/test_case.h"

#include <vector>

namespace onnx_light_cpu::backend_test {

/// Appends the onnx-light-cpu ``Abs`` backend test cases to ``registry``.
///
/// In :cpp:enumerator:`TestMode::TEST` this registers a ``test_cpu_abs_<dtype>``
/// correctness case for every element type ``AbsKernel`` implements (float32,
/// float64, int8, int16, int32, int64, float16, bfloat16). In
/// :cpp:enumerator:`TestMode::BENCHMARK` it registers the large
/// ``test_cpu_abs_benchmark`` (+ ``_float16``) timing cases.
void RegisterCpuAbsCases(std::vector<ONNX_LIGHT_NAMESPACE::core::backend_test::TestCase> &registry,
                         ONNX_LIGHT_NAMESPACE::core::backend_test::TestMode mode);

} // namespace onnx_light_cpu::backend_test
