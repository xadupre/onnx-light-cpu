// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// x86 VNNI INT8 matrix multiplication (Roadmap PR09.2). ``IntegerMatMul2D``
// computes a single ``MatMulInteger`` matrix product exactly, dispatching to a
// native AVX-512 VNNI (``vpdpbusd``) path when the running CPU reports the ISA
// (``CpuSupportsAvx512Vnni``) and otherwise to a portable scalar sibling that
// produces identical results. Both paths share the same packing and zero-point
// correction, so they are differentially testable over the portable PR09.1
// fallback.
//
// The native dot-product ``IntegerDotU8S8Avx512Vnni`` is only usable when
// ``ONNX_LIGHT_CPU_HAVE_AVX512VNNI`` is defined: integer_gemm_avx512vnni.cc
// (the file that implements it) is only compiled into lib_onnx_light_cpu when
// CMake confirms the compiler accepts -mavx512vnni -- see CMakeLists.txt.

#pragma once

#include <cstdint>

namespace onnx_light_cpu {

// Computes ``C[rows, cols]`` (row-major, INT32 accumulation defined as
// modulo-2^32 arithmetic) for ``MatMulInteger`` of the byte matrices
// ``A[rows, depth]`` (row-major) and ``B[depth, cols]`` (row-major). The
// ``a_signed`` / ``b_signed`` flags select whether the raw bytes are read as
// INT8 (``true``) or UINT8 (``false``). ``a_zero_point`` holds either a single
// value (``a_zero_point_count == 1``) applied to every row or one value per row
// (``a_zero_point_count == rows``); ``b_zero_point`` is scalar or per-column in
// the same way. Dispatches to the native AVX-512 VNNI kernel when available and
// otherwise to the portable scalar path, producing identical results.
void IntegerMatMul2D(const std::uint8_t *a, bool a_signed, const std::uint8_t *b, bool b_signed,
                     std::int32_t *c, std::int64_t rows, std::int64_t cols, std::int64_t depth,
                     const std::int32_t *a_zero_point, std::int64_t a_zero_point_count,
                     const std::int32_t *b_zero_point, std::int64_t b_zero_point_count);

// Returns whether ``IntegerMatMul2D`` dispatches to the native AVX-512 VNNI
// path on this CPU. ``false`` when the ISA is unavailable at runtime or the
// library was built without VNNI support. Exposed for differential tests.
bool IntegerMatMul2DUsesVnni();

namespace detail {

// Per-column dot-product used by the shared ``IntegerMatMul2D`` driver: returns
// ``sum_{i<depth} ua[i] * sb[i]`` (modulo-2^32) where ``ua`` is a UINT8 row and
// ``sb`` is an INT8 column, both contiguous of length ``depth``.
using IntegerVnniDotFn = std::int32_t (*)(const std::uint8_t *ua, const std::int8_t *sb,
                                          std::int64_t depth);

// Portable scalar reference for ``IntegerVnniDotFn``.
std::int32_t IntegerDotU8S8Scalar(const std::uint8_t *ua, const std::int8_t *sb,
                                  std::int64_t depth);

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512VNNI
// Native AVX-512 VNNI (``vpdpbusd``) implementation of ``IntegerVnniDotFn``,
// with a scalar tail for depths that are not a multiple of the 64-byte vector.
std::int32_t IntegerDotU8S8Avx512Vnni(const std::uint8_t *ua, const std::int8_t *sb,
                                      std::int64_t depth);
#endif

// Shared driver behind ``IntegerMatMul2D``, parameterised by the per-column
// dot-product implementation. Packs ``A`` into a UINT8 panel and ``B`` into a
// transposed INT8 panel (applying the signedness offsets), then reconstructs the
// zero-point-corrected accumulation exactly for every output element.
void IntegerMatMul2DWithDot(IntegerVnniDotFn dot, const std::uint8_t *a, bool a_signed,
                            const std::uint8_t *b, bool b_signed, std::int32_t *c,
                            std::int64_t rows, std::int64_t cols, std::int64_t depth,
                            const std::int32_t *a_zero_point, std::int64_t a_zero_point_count,
                            const std::int32_t *b_zero_point, std::int64_t b_zero_point_count);

} // namespace detail

} // namespace onnx_light_cpu
