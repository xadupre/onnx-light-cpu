// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu_compute/math/gemm/gemm_common.h"

namespace onnx_light_cpu::detail {

enum class GemmBf16KernelKind { kNone, kAVX2, kAVX512BF16, kAMXBF16 };

struct GemmBf16Capabilities {
  bool avx2_fma = false;
  bool avx512_bf16 = false;
  bool amx_bf16 = false;
};

// Capabilities include build support and runtime CPU/OS availability.
constexpr GemmBf16KernelKind SelectGemmBf16KernelKind(GemmAlgorithm algorithm, bool trans_b,
                                                      GemmBf16Capabilities capabilities,
                                                      bool requires_gradual_underflow = false) {
  if (trans_b)
    return GemmBf16KernelKind::kNone;
  if (algorithm == GemmAlgorithm::kGeneral && !requires_gradual_underflow) {
    if (capabilities.amx_bf16)
      return GemmBf16KernelKind::kAMXBF16;
    if (capabilities.avx512_bf16)
      return GemmBf16KernelKind::kAVX512BF16;
  }
  if ((algorithm == GemmAlgorithm::kGeneral || algorithm == GemmAlgorithm::kDirect) &&
      capabilities.avx2_fma)
    return GemmBf16KernelKind::kAVX2;
  return GemmBf16KernelKind::kNone;
}

} // namespace onnx_light_cpu::detail
