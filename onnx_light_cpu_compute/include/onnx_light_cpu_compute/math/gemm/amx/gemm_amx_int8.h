// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnx_light_cpu {

// Native AMX-INT8 implementation of a contiguous 2D ``MatMulInteger``. Both
// byte operands are mapped to the unsigned domain before ``tdpbuud`` and the
// zero-point correction restores the exact modulo-2^32 accumulation. Only call
// this when ``CpuSupportsAmxInt8()`` and ``AmxTileStateAvailable()`` are true.
void GemmMatMulIntegerAmxInt8(const std::uint8_t *a, bool a_signed, const std::uint8_t *b,
                              bool b_signed, std::int32_t *c, std::size_t m, std::size_t n,
                              std::size_t k, const std::int32_t *a_zero_points,
                              std::size_t a_zero_point_count, const std::int32_t *b_zero_points,
                              std::size_t b_zero_point_count);

} // namespace onnx_light_cpu
