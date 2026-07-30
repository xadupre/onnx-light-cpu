// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/cpu_kernels.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <stdexcept>

namespace nb = nanobind;

NB_MODULE(_cpukernels, m) {
  m.doc() = "Python bindings for onnx-light-cpu: "
            "highly optimized CPU kernels (Abs) with AVX/AVX2/AVX-512 dispatch.";

  m.def(
      "detect_simd_level",
      []() -> int { return static_cast<int>(onnx_light_cpu::DetectSimdLevel()); },
      "Returns the detected SIMD level: 0=None, 1=SSE2, 2=AVX, 3=AVX2, 4=AVX512.");

  m.def(
      "abs_float32",
      [](nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> input,
         nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> output) {
        if (input.shape(0) != output.shape(0)) {
          throw std::invalid_argument("input and output must have the same length");
        }
        onnx_light_cpu::AbsFloat32(input.data(), output.data(),
                                   static_cast<std::size_t>(input.shape(0)));
      },
      nb::arg("input"), nb::arg("output"),
      "Computes elementwise abs of a float32 array using optimized SIMD.");

  m.def(
      "abs_float64",
      [](nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> input,
         nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> output) {
        if (input.shape(0) != output.shape(0)) {
          throw std::invalid_argument("input and output must have the same length");
        }
        onnx_light_cpu::AbsFloat64(input.data(), output.data(),
                                   static_cast<std::size_t>(input.shape(0)));
      },
      nb::arg("input"), nb::arg("output"),
      "Computes elementwise abs of a float64 array using optimized SIMD.");

  m.def(
      "abs_int32",
      [](nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> input,
         nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> output) {
        if (input.shape(0) != output.shape(0)) {
          throw std::invalid_argument("input and output must have the same length");
        }
        onnx_light_cpu::AbsInt32(input.data(), output.data(),
                                 static_cast<std::size_t>(input.shape(0)));
      },
      nb::arg("input"), nb::arg("output"),
      "Computes elementwise abs of an int32 array using optimized SIMD.");

  m.def(
      "abs_int64",
      [](nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> input,
         nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> output) {
        if (input.shape(0) != output.shape(0)) {
          throw std::invalid_argument("input and output must have the same length");
        }
        onnx_light_cpu::AbsInt64(input.data(), output.data(),
                                 static_cast<std::size_t>(input.shape(0)));
      },
      nb::arg("input"), nb::arg("output"),
      "Computes elementwise abs of an int64 array using optimized SIMD.");

  m.def(
      "has_cpu_kernels", []() -> bool { return true; },
      "Returns True when the CPU kernel extension is available.");
}
