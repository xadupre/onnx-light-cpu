// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace onnx_light_cpu {

enum class ArmSimdLevel {
  kNone,
  kNeon,
  kSve,
  kSve2,
};

ArmSimdLevel DetectArmSimdLevel();

/// Returns whether the Advanced SIMD dot-product instructions (``SDOT``/``UDOT``,
/// ARMv8.2-A ``+dotprod``) are available on the current CPU. Used to gate the
/// native NEON INT8 dot-product GEMM path; callers keep the portable scalar
/// fallback when this returns ``false`` (including on every non-AArch64 target).
bool CpuSupportsNeonDotProd();

} // namespace onnx_light_cpu
