// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared INT8 matrix multiplication driver for ``MatMulInteger`` (Roadmap PR09.2
// / PR09.3). ``IntegerMatMul2D`` computes a single ``MatMulInteger`` matrix
// product exactly, dispatching to the native ARM NEON dot-product kernel when
// the running CPU reports it (``CpuSupportsNeonDotProd``, Roadmap PR09.3), then
// to a native x86 AVX-512 VNNI (``vpdpbusd``) path when the CPU reports that ISA
// (``CpuSupportsAvx512Vnni``, Roadmap PR09.2), an exact AVX2 fallback, and
// finally a portable scalar sibling. All paths share the same packing and
// zero-point correction, so they are differentially testable over the portable
// PR09.1 fallback.
//
// The native dot-product ``IntegerDotU8S8Avx512Vnni`` is only usable when
// ``ONNX_LIGHT_CPU_HAVE_AVX512VNNI`` is defined: integer_gemm_avx512vnni.cc
// (the file that implements it) is only compiled into lib_onnx_light_cpu when
// CMake confirms the compiler accepts -mavx512vnni -- see CMakeLists.txt. The
// NEON path is likewise gated on ``ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD``.

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
// the same way. Dispatches to the native ARM NEON dot-product, x86 AVX-512 VNNI,
// or x86 AVX2 kernel when available and otherwise to the portable scalar path,
// producing identical results.
void IntegerMatMul2D(const std::uint8_t *a, bool a_signed, const std::uint8_t *b, bool b_signed,
                     std::int32_t *c, std::int64_t rows, std::int64_t cols, std::int64_t depth,
                     const std::int32_t *a_zero_point, std::int64_t a_zero_point_count,
                     const std::int32_t *b_zero_point, std::int64_t b_zero_point_count);

// Computes the same matrix product from packed INT4/UINT4 operands. Logical
// elements use the ONNX row-major packing order: the even-indexed element is in
// the low nibble and the odd-indexed element is in the high nibble, with no
// padding between matrix rows. When a matrix has an odd number of elements the
// unused high nibble of its final byte is ignored. Signed nibbles use two's
// complement in the range [-8, 7]; unsigned nibbles use [0, 15].
//
// Packing expands directly into the UINT8 x INT8 panels consumed by the scalar,
// AVX2, AVX-512 VNNI, or ARM NEON dot-product implementation. It never
// materializes separate unpacked source matrices.
void IntegerMatMul4Bit2D(const std::uint8_t *a, bool a_signed, const std::uint8_t *b, bool b_signed,
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

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER
// Exact AVX2 fallback. It splits each UINT8 value into low-seven-bit and
// high-bit terms before ``vpmaddubsw``, keeping both pair sums in range.
std::int32_t IntegerDotU8S8Avx2(const std::uint8_t *ua, const std::int8_t *sb, std::int64_t depth);

// Computes all raw dot products from packed row-major A and transposed B panels.
// The 2x2 output micro-kernel reuses loaded A and B vectors across outputs.
void IntegerMatMulU8S8Avx2(const std::uint8_t *a, const std::int8_t *b, std::int32_t *c,
                           std::int64_t rows, std::int64_t cols, std::int64_t depth);
#endif

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

// Scalar/native-testable implementation behind ``IntegerMatMul4Bit2D``.
void IntegerMatMul4Bit2DWithDot(IntegerVnniDotFn dot, const std::uint8_t *a, bool a_signed,
                                const std::uint8_t *b, bool b_signed, std::int32_t *c,
                                std::int64_t rows, std::int64_t cols, std::int64_t depth,
                                const std::int32_t *a_zero_point, std::int64_t a_zero_point_count,
                                const std::int32_t *b_zero_point, std::int64_t b_zero_point_count);

} // namespace detail

} // namespace onnx_light_cpu
