// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/cpu_kernels.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace nb = nanobind;

namespace {

// Untyped 1-D, C-contiguous, CPU input array. The element type is resolved at
// runtime from the array dtype so a single ``abs`` entry point can dispatch to
// the matching kernel, mirroring the behaviour of ``numpy.abs``.
using AbsInput = nb::ndarray<nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Allocates a new output array of the same length as ``input``, runs the
// matching kernel, and returns the result as a NumPy array that owns its data.
template <typename T, void (*Kernel)(const T *, T *, std::size_t)>
nb::object AbsTyped(const AbsInput &input) {
  const std::size_t count = static_cast<std::size_t>(input.shape(0));
  T *output = new T[count == 0 ? 1 : count];
  Kernel(reinterpret_cast<const T *>(input.data()), output, count);
  nb::capsule owner(output, [](void *p) noexcept { delete[] reinterpret_cast<T *>(p); });
  return nb::cast(nb::ndarray<nb::numpy, T, nb::ndim<1>>(output, {count}, owner));
}

} // namespace

NB_MODULE(_cpukernels, m) {
  m.doc() = "Python bindings for onnx-light-cpu: "
            "highly optimized CPU kernels (Abs) with AVX/AVX2/AVX-512 dispatch.";

  m.def(
      "detect_simd_level",
      []() -> int { return static_cast<int>(onnx_light_cpu::DetectSimdLevel()); },
      "Returns the detected SIMD level: 0=None, 1=SSE2, 2=AVX, 3=AVX2, 4=AVX512.");

  m.def(
      "abs",
      [](const AbsInput &input) -> nb::object {
        const nb::dlpack::dtype dt = input.dtype();
        if (dt == nb::dtype<float>()) {
          return AbsTyped<float, onnx_light_cpu::AbsFloat32>(input);
        }
        if (dt == nb::dtype<double>()) {
          return AbsTyped<double, onnx_light_cpu::AbsFloat64>(input);
        }
        if (dt == nb::dtype<int32_t>()) {
          return AbsTyped<int32_t, onnx_light_cpu::AbsInt32>(input);
        }
        if (dt == nb::dtype<int64_t>()) {
          return AbsTyped<int64_t, onnx_light_cpu::AbsInt64>(input);
        }
        throw std::invalid_argument(
            "abs: unsupported dtype; expected float32, float64, int32 or int64");
      },
      nb::arg("input"),
      "Computes the elementwise absolute value of a 1-D array using optimized "
      "SIMD. Dispatches on the array dtype (float32, float64, int32, int64) and "
      "returns a new array, like numpy.abs.");

  m.def(
      "has_cpu_kernels", []() -> bool { return true; },
      "Returns True when the CPU kernel extension is available.");
}
