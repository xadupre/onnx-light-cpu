// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace ml_cpu_compute::detail {

inline constexpr std::size_t kNBitsRows = 128;
inline constexpr std::size_t kNBitsColumns = 32;
inline constexpr std::size_t kNBitsBlock = 32;

using NBitsPanelFn = void (*)(const float *, const float *, float *, std::size_t, std::size_t);

#ifdef ML_CPU_COMPUTE_HAVE_AVX2
void NBitsPanelAvx2(const float *a, const float *b, float *sums, std::size_t rows,
                    std::size_t depth);
#endif
#ifdef ML_CPU_COMPUTE_HAVE_AVX512
void NBitsPanelAvx512(const float *a, const float *b, float *sums, std::size_t rows,
                      std::size_t depth);
#endif

} // namespace ml_cpu_compute::detail
