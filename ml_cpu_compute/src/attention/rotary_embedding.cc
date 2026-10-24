// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "ml_cpu_compute/attention/rotary_embedding.h"
#include "ml_cpu_compute/simd_level.h"

namespace ml_cpu_compute {

#ifdef ML_CPU_COMPUTE_HAVE_AVX
std::size_t RotaryEmbeddingSplitHalf_AVX(const void *, const void *, const void *, void *,
                                         std::size_t);
#endif
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_F16C
std::size_t RotaryEmbeddingSplitHalf_AVX2_F16C(const void *, const void *, const void *, void *,
                                               std::size_t);
#endif

RotaryEmbeddingSplitHalfFn SelectRotaryEmbeddingSplitHalf(DataType type) {
#if defined(ML_CPU_COMPUTE_HAVE_AVX) || defined(ML_CPU_COMPUTE_HAVE_AVX2_F16C)
  static const auto level = DetectSimdLevel();
#endif
  if (type == DataType::FLOAT16) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX2_F16C
    static const bool use_f16c = level >= SimdLevel::kAVX2 && CpuSupportsF16C();
    if (use_f16c) {
      return RotaryEmbeddingSplitHalf_AVX2_F16C;
    }
#endif
  } else if (type == DataType::FLOAT) {
#ifdef ML_CPU_COMPUTE_HAVE_AVX
    if (level >= SimdLevel::kAVX) {
      return RotaryEmbeddingSplitHalf_AVX;
    }
#endif
  }
  return nullptr;
}

} // namespace ml_cpu_compute
