// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/math_kernels.h"
#include "onnx_light_cpu/onnx_py/_cpupy_kernels.h"

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
  const T *data = reinterpret_cast<const T *>(input.data());
  {
    // The kernel only touches the raw C buffers, so the GIL can be released for
    // the duration of the compute. This lets several Python threads run the
    // kernel on disjoint chunks of an array concurrently.
    nb::gil_scoped_release release;
    Kernel(data, output, count);
  }
  nb::capsule owner(output, [](void *p) noexcept { delete[] reinterpret_cast<T *>(p); });
  return nb::cast(nb::ndarray<nb::numpy, T, nb::ndim<1>>(output, {count}, owner));
}

// float16 has no portable C++ scalar type, so it is handled specially: the raw
// 16-bit bit patterns are processed by the matching ``*Float16`` kernel and the
// result is exposed as a NumPy ``float16`` array (dtype code Float, 16 bits)
// that owns its data.
template <void (*Kernel)(const std::uint16_t *, std::uint16_t *, std::size_t)>
nb::object Float16Typed(const AbsInput &input) {
  const std::size_t count = static_cast<std::size_t>(input.shape(0));
  auto *output = new std::uint16_t[count == 0 ? 1 : count];
  const std::uint16_t *data = reinterpret_cast<const std::uint16_t *>(input.data());
  {
    nb::gil_scoped_release release;
    Kernel(data, output, count);
  }
  nb::capsule owner(output,
                    [](void *p) noexcept { delete[] reinterpret_cast<std::uint16_t *>(p); });
  const nb::dlpack::dtype f16{static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};
  const std::size_t shape[1] = {count};
  return nb::cast(nb::ndarray<nb::numpy>(output, 1, shape, owner, nullptr, f16));
}

nb::object AbsFloat16Typed(const AbsInput &input) {
  return Float16Typed<onnx_light_cpu::AbsFloat16>(input);
}

} // namespace

namespace onnx_light_cpu {

void RegisterMathKernels(nb::module_ &m) {
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
        const nb::dlpack::dtype f16{static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};
        if (dt == f16) {
          return AbsFloat16Typed(input);
        }
        if (dt == nb::dtype<int8_t>()) {
          return AbsTyped<int8_t, onnx_light_cpu::AbsInt8>(input);
        }
        if (dt == nb::dtype<int32_t>()) {
          return AbsTyped<int32_t, onnx_light_cpu::AbsInt32>(input);
        }
        if (dt == nb::dtype<int64_t>()) {
          return AbsTyped<int64_t, onnx_light_cpu::AbsInt64>(input);
        }
        throw std::invalid_argument(
            "abs: unsupported dtype; expected float16, float32, float64, int8, int32 or int64");
      },
      nb::arg("input"),
      "Computes the elementwise absolute value of a 1-D array using optimized "
      "SIMD. Dispatches on the array dtype (float16, float32, float64, int8, "
      "int32, int64) and returns a new array, like numpy.abs.");

  m.def(
      "exp",
      [](const AbsInput &input) -> nb::object {
        const nb::dlpack::dtype dt = input.dtype();
        if (dt == nb::dtype<float>()) {
          return AbsTyped<float, onnx_light_cpu::ExpFloat32>(input);
        }
        if (dt == nb::dtype<double>()) {
          return AbsTyped<double, onnx_light_cpu::ExpFloat64>(input);
        }
        const nb::dlpack::dtype f16{static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};
        if (dt == f16) {
          return Float16Typed<onnx_light_cpu::ExpFloat16>(input);
        }
        throw std::invalid_argument("exp: unsupported dtype; expected float16, float32 or float64");
      },
      nb::arg("input"),
      "Computes the elementwise natural exponential of a 1-D array using "
      "optimized SIMD. Dispatches on the array dtype (float16, float32, "
      "float64) and returns a new array, like numpy.exp.");

  m.def(
      "log",
      [](const AbsInput &input) -> nb::object {
        const nb::dlpack::dtype dt = input.dtype();
        if (dt == nb::dtype<float>()) {
          return AbsTyped<float, onnx_light_cpu::LogFloat32>(input);
        }
        if (dt == nb::dtype<double>()) {
          return AbsTyped<double, onnx_light_cpu::LogFloat64>(input);
        }
        const nb::dlpack::dtype f16{static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};
        if (dt == f16) {
          return Float16Typed<onnx_light_cpu::LogFloat16>(input);
        }
        throw std::invalid_argument("log: unsupported dtype; expected float16, float32 or float64");
      },
      nb::arg("input"),
      "Computes the elementwise natural logarithm of a 1-D array using "
      "optimized SIMD. Dispatches on the array dtype (float16, float32, "
      "float64) and returns a new array, like numpy.log.");

  m.def(
      "has_cpu_kernels", []() -> bool { return true; },
      "Returns True when the CPU kernel extension is available.");
}

} // namespace onnx_light_cpu

NB_MODULE(_cpukernels, m) {
  m.doc() = "Python bindings for onnx-light-cpu: "
            "highly optimized CPU kernels (Abs, Exp, Log, Not) with AVX/AVX2/AVX-512 dispatch.";

  onnx_light_cpu::RegisterMathKernels(m);
  onnx_light_cpu::RegisterLogicalKernels(m);
}
