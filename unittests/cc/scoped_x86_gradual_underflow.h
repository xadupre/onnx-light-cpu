// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace onnx_light_cpu::test {

class ScopedX86GradualUnderflow {
public:
  ScopedX86GradualUnderflow() {
#if defined(__x86_64__) || defined(_M_X64)
    original_ = _mm_getcsr();
    _mm_setcsr(original_ & ~0x8040u); // Disable FTZ and DAZ.
#endif
  }
  ~ScopedX86GradualUnderflow() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_setcsr(original_);
#endif
  }
  ScopedX86GradualUnderflow(const ScopedX86GradualUnderflow &) = delete;
  ScopedX86GradualUnderflow &operator=(const ScopedX86GradualUnderflow &) = delete;

private:
#if defined(__x86_64__) || defined(_M_X64)
  unsigned int original_;
#endif
};

} // namespace onnx_light_cpu::test
