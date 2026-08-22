// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/simd_level.h"
#include "onnx_light_cpu/onnx_py/_cpupy_kernels.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace onnx_light_cpu {

void RegisterMathKernels(nb::module_ &m) {
  m.def(
      "detect_simd_level",
      []() -> int { return static_cast<int>(onnx_light_cpu::DetectSimdLevel()); },
      "Returns the detected SIMD level: 0=None, 1=SSE2, 2=AVX, 3=AVX2, 4=AVX512.");

  m.def(
      "has_cpu_kernels", []() -> bool { return true; },
      "Returns True when the CPU kernel extension is available.");
}

} // namespace onnx_light_cpu

NB_MODULE(_cpukernels, m) {
  m.doc() = "Python bindings for onnx-light-cpu: exposes SIMD introspection helpers for the "
            "highly optimized CPU kernels (Abs, Exp, Log, Gemm, Not) with "
            "AVX/AVX2/AVX-512 dispatch. The kernels themselves are only "
            "reachable through onnx-light's runtime after registration. Also exposes the "
            "versioned processor performance profile (benchmark_processor_performance_raw), "
            "which links neither onnx-light nor any kernel dispatch table.";

  onnx_light_cpu::RegisterMathKernels(m);
  onnx_light_cpu::RegisterProcessorPerformanceProfile(m);
}
