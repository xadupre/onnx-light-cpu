// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Backend-test *registration* helper for the onnx-light-cpu ``Gemm`` kernel.
// See ``abs_backend_test.h`` for the shared rationale (one header + one ``.cc``
// per kernel, both correctness and benchmark cases).

#include "onnx_core/backend_test/test_case.h"

#include <vector>

namespace onnx_light_cpu::backend_test {

/// Appends the onnx-light-cpu ``Gemm`` backend test cases to ``registry``.
///
/// In :cpp:enumerator:`TestMode::TEST` this registers a
/// ``test_cpu_gemm_<dtype>`` correctness case for every element type
/// ``GemmKernel`` implements (float32, float64, float16, bfloat16). In
/// :cpp:enumerator:`TestMode::BENCHMARK` it registers the large
/// ``test_cpu_gemm_benchmark`` timing case.
void RegisterCpuGemmCases(std::vector<ONNX_LIGHT_NAMESPACE::core::backend_test::TestCase> &registry,
                          ONNX_LIGHT_NAMESPACE::core::backend_test::TestMode mode);

} // namespace onnx_light_cpu::backend_test
