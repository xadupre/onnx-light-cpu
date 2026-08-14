// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/arm_simd_level.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__linux__)
#include <sys/auxv.h>

#ifndef AT_HWCAP
#define AT_HWCAP 16
#endif
#ifndef AT_HWCAP2
#define AT_HWCAP2 26
#endif
#ifndef HWCAP_ASIMD
#define HWCAP_ASIMD (1UL << 1)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE (1UL << 22)
#endif
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1UL << 1)
#endif
#endif
#endif

namespace onnx_light_cpu {

ArmSimdLevel DetectArmSimdLevel() {
#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__linux__)
  const unsigned long hwcap = getauxval(AT_HWCAP);
  if ((hwcap & HWCAP_SVE) != 0) {
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
    return (hwcap2 & HWCAP2_SVE2) != 0 ? ArmSimdLevel::kSve2 : ArmSimdLevel::kSve;
  }
  return (hwcap & HWCAP_ASIMD) != 0 ? ArmSimdLevel::kNeon : ArmSimdLevel::kNone;
#else
  // Advanced SIMD is part of the AArch64 baseline. Current Apple and Windows
  // ARM64 systems do not expose SVE, so NEON is the strongest usable level.
  return ArmSimdLevel::kNeon;
#endif
#else
  return ArmSimdLevel::kNone;
#endif
}

} // namespace onnx_light_cpu
