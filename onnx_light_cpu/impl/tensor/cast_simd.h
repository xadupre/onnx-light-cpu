// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu::detail {

// Convert whole vectors in possibly unaligned byte buffers; return the number
// converted so the dispatcher can use the generic codec for the tail.
using CastSimdFunction = std::size_t (*)(const std::uint8_t *, std::uint8_t *, std::size_t);

#ifdef ONNX_LIGHT_CPU_HAVE_AVX_F16C
std::size_t CastFloat32ToFloat16_F16C(const std::uint8_t *src, std::uint8_t *dst,
                                      std::size_t count);
std::size_t CastFloat16ToFloat32_F16C(const std::uint8_t *src, std::uint8_t *dst,
                                      std::size_t count);
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2
std::size_t CastFloat32ToInt32_AVX2(const std::uint8_t *src, std::uint8_t *dst, std::size_t count);
std::size_t CastFloat32ToInt64_AVX2(const std::uint8_t *src, std::uint8_t *dst, std::size_t count);
std::size_t CastFloat32ToInt8_AVX2(const std::uint8_t *src, std::uint8_t *dst, std::size_t count);
std::size_t CastFloat32ToUint8_AVX2(const std::uint8_t *src, std::uint8_t *dst, std::size_t count);
std::size_t CastFloat32ToBool_AVX2(const std::uint8_t *src, std::uint8_t *dst, std::size_t count);
std::size_t CastBoolToFloat32_AVX2(const std::uint8_t *src, std::uint8_t *dst, std::size_t count);
std::size_t CastFloat32ToBFloat16_AVX2(const std::uint8_t *src, std::uint8_t *dst,
                                       std::size_t count);
std::size_t CastBFloat16ToFloat32_AVX2(const std::uint8_t *src, std::uint8_t *dst,
                                       std::size_t count);
#endif

#ifdef ONNX_LIGHT_CPU_HAVE_AVX512DQ
std::size_t CastInt64ToFloat32_AVX512DQ(const std::uint8_t *src, std::uint8_t *dst,
                                        std::size_t count);
#endif

} // namespace onnx_light_cpu::detail
