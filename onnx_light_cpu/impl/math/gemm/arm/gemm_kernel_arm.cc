// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/arm/gemm_kernel_arm.h"

namespace onnx_light_cpu {

ArmGemmProfile SelectArmGemmProfile(ArmSimdLevel level, std::size_t sve_vector_bytes,
                                    bool sve_kernel_available) {
  // SVE has a runtime vector length. At 128 bits it offers no lane-width
  // advantage over the six-row NEON kernel, so retain that better-unrolled
  // fallback. Wider implementations use predicated SVE tails.
  if (sve_kernel_available && level >= ArmSimdLevel::kSve && sve_vector_bytes >= 32) {
    return {ArmGemmKernelKind::kSve, sve_vector_bytes, kGemmSveMR};
  }
  if (level >= ArmSimdLevel::kNeon) {
    return {ArmGemmKernelKind::kNeon, 16, kGemmNeonMR};
  }
  return {};
}

ArmGemmProfile DetectArmGemmProfile() {
  static const ArmGemmProfile profile = [] {
    const ArmSimdLevel level = DetectArmSimdLevel();
#ifdef ONNX_LIGHT_CPU_HAVE_SVE
    const std::size_t sve_vector_bytes = level >= ArmSimdLevel::kSve ? GemmSveVectorBytes() : 0;
    return SelectArmGemmProfile(level, sve_vector_bytes, true);
#else
    return SelectArmGemmProfile(level, 0, false);
#endif
  }();
  return profile;
}

} // namespace onnx_light_cpu
