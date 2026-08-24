// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_light_cpu/kernels/attention/attention_kernel.h"
#include "onnx_light_cpu/kernels/elementwise/binary_kernel.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/logical/not_kernel.h"
#include "onnx_light_cpu/kernels/math/abs_kernel.h"
#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"
#include "onnx_light_cpu/kernels/math/gemm_kernel.h"
#include "onnx_light_cpu/kernels/math/integer_matmul_kernel.h"
#include "onnx_light_cpu/kernels/math/matmul_kernel.h"
#include "onnx_light_cpu/kernels/math/swiglu_kernel.h"

namespace onnx_light_cpu {

void RegisterAllKernels() {
  // Opens a fresh duplicate-registration check for this call, unless one is
  // already active (``CollectRegisteredKernels`` opens one first so this
  // pass collects metadata instead of installing kernels; see
  // ``KernelRegistrationScope``).
  KernelRegistrationScope scope;
  RegisterAbsKernel();
  RegisterAttentionKernel();
  RegisterBinaryKernels();
  RegisterExpKernel();
  RegisterLogKernel();
  RegisterGemmKernel();
  RegisterMatMulKernel();
  RegisterIntegerMatMulKernels();
  RegisterNotKernel();
  RegisterSwiGLUKernel();
}

} // namespace onnx_light_cpu
