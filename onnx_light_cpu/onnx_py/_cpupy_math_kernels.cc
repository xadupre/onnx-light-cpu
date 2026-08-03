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

// Untyped 2-D, C-contiguous, CPU input matrix used by the ``gemm`` binding.
using GemmMatrix = nb::ndarray<nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// Runs the matching Gemm kernel and returns a freshly allocated ``M x N`` NumPy
// matrix that owns its data. ``c`` may be ``nullptr`` (no bias term).
template <typename T, void (*Kernel)(bool, bool, std::size_t, std::size_t, std::size_t, T,
                                     const T *, const T *, T, const T *, T *)>
nb::object GemmTyped(const GemmMatrix &a, const GemmMatrix &b, const GemmMatrix *c, double alpha,
                     double beta, bool trans_a, bool trans_b) {
  const std::size_t a0 = static_cast<std::size_t>(a.shape(0));
  const std::size_t a1 = static_cast<std::size_t>(a.shape(1));
  const std::size_t b0 = static_cast<std::size_t>(b.shape(0));
  const std::size_t b1 = static_cast<std::size_t>(b.shape(1));

  const std::size_t M = trans_a ? a1 : a0;
  const std::size_t K = trans_a ? a0 : a1;
  const std::size_t Kb = trans_b ? b1 : b0;
  const std::size_t N = trans_b ? b0 : b1;
  if (K != Kb) {
    throw std::invalid_argument("gemm: inner dimensions of A and B do not match");
  }

  const T *cptr = nullptr;
  if (c != nullptr) {
    if (static_cast<std::size_t>(c->shape(0)) != M || static_cast<std::size_t>(c->shape(1)) != N) {
      throw std::invalid_argument("gemm: C must have shape (M, N)");
    }
    cptr = reinterpret_cast<const T *>(c->data());
  }

  const std::size_t total = M * N;
  T *output = new T[total == 0 ? 1 : total];
  {
    nb::gil_scoped_release release;
    Kernel(trans_a, trans_b, M, N, K, static_cast<T>(alpha), reinterpret_cast<const T *>(a.data()),
           reinterpret_cast<const T *>(b.data()), static_cast<T>(beta), cptr, output);
  }
  nb::capsule owner(output, [](void *p) noexcept { delete[] reinterpret_cast<T *>(p); });
  return nb::cast(nb::ndarray<nb::numpy, T, nb::ndim<2>>(output, {M, N}, owner));
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
      "gemm",
      [](const GemmMatrix &a, const GemmMatrix &b, nb::object c, double alpha, double beta,
         bool trans_a, bool trans_b) -> nb::object {
        const nb::dlpack::dtype dt = a.dtype();
        if (b.dtype() != dt) {
          throw std::invalid_argument("gemm: A and B must have the same dtype");
        }
        GemmMatrix cmat;
        const GemmMatrix *cptr = nullptr;
        if (!c.is_none()) {
          cmat = nb::cast<GemmMatrix>(c);
          if (cmat.dtype() != dt) {
            throw std::invalid_argument("gemm: C must have the same dtype as A and B");
          }
          cptr = &cmat;
        }
        if (dt == nb::dtype<float>()) {
          return GemmTyped<float, onnx_light_cpu::GemmFloat32>(a, b, cptr, alpha, beta, trans_a,
                                                               trans_b);
        }
        if (dt == nb::dtype<double>()) {
          return GemmTyped<double, onnx_light_cpu::GemmFloat64>(a, b, cptr, alpha, beta, trans_a,
                                                                trans_b);
        }
        throw std::invalid_argument("gemm: unsupported dtype; expected float32 or float64");
      },
      nb::arg("a"), nb::arg("b"), nb::arg("c") = nb::none(), nb::arg("alpha") = 1.0,
      nb::arg("beta") = 1.0, nb::arg("trans_a") = false, nb::arg("trans_b") = false,
      "Computes the ONNX Gemm general matrix multiplication "
      "``Y = alpha * op(A) @ op(B) + beta * C`` for 2-D float32 or float64 "
      "matrices using an AVX-accelerated kernel. ``op(A)`` transposes ``A`` when "
      "``trans_a`` is True and ``op(B)`` transposes ``B`` when ``trans_b`` is "
      "True. ``c`` is an optional bias matrix of shape (M, N); pass None "
      "(or ``beta=0``) to skip it. Returns a new (M, N) array.");

  m.def(
      "has_cpu_kernels", []() -> bool { return true; },
      "Returns True when the CPU kernel extension is available.");
}

} // namespace onnx_light_cpu

NB_MODULE(_cpukernels, m) {
  m.doc() = "Python bindings for onnx-light-cpu: "
            "highly optimized CPU kernels (Abs, Exp, Log, Gemm, Not) with "
            "AVX/AVX2/AVX-512 dispatch.";

  onnx_light_cpu::RegisterMathKernels(m);
  onnx_light_cpu::RegisterLogicalKernels(m);
}
