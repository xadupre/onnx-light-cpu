// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/logical/logical_kernels.h"
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
// runtime from the array dtype, mirroring the behaviour of
// ``numpy.logical_not``.
using LogicalInput = nb::ndarray<nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// ``bool`` arrays are stored as one byte per element, so the raw byte patterns
// are processed by the ``NotBool`` kernel and the result is exposed as a NumPy
// ``bool`` array that owns its data.
template <void (*Kernel)(const std::uint8_t *, std::uint8_t *, std::size_t)>
nb::object BoolTyped(const LogicalInput &input) {
  const std::size_t count = static_cast<std::size_t>(input.shape(0));
  auto *output = new std::uint8_t[count == 0 ? 1 : count];
  Kernel(reinterpret_cast<const std::uint8_t *>(input.data()), output, count);
  nb::capsule owner(output, [](void *p) noexcept { delete[] reinterpret_cast<std::uint8_t *>(p); });
  return nb::cast(nb::ndarray<nb::numpy, bool, nb::ndim<1>>(output, {count}, owner));
}

} // namespace

namespace onnx_light_cpu {

void RegisterLogicalKernels(nb::module_ &m) {
  m.def(
      "logical_not",
      [](const LogicalInput &input) -> nb::object {
        const nb::dlpack::dtype dt = input.dtype();
        if (dt == nb::dtype<bool>()) {
          return BoolTyped<onnx_light_cpu::NotBool>(input);
        }
        throw std::invalid_argument("logical_not: unsupported dtype; expected bool");
      },
      nb::arg("input"),
      "Computes the elementwise logical negation of a 1-D bool array using "
      "optimized SIMD and returns a new bool array, like numpy.logical_not.");
}

} // namespace onnx_light_cpu
