// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace ml_cpu_compute {

void AbsInt8_SSSE3(const std::int8_t *input, std::int8_t *output, std::size_t count);
void AbsInt16_SSSE3(const std::int16_t *input, std::int16_t *output, std::size_t count);

} // namespace ml_cpu_compute
