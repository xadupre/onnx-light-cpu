// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_light_cpu/kernels/logical/not_kernel.h"
#include "onnx_light_cpu/kernels/math/abs_kernel.h"
#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"
#include "onnx_light_cpu/kernels/math/gemm_kernel.h"

namespace onnx_light_cpu {

void RegisterAllKernels() {
  RegisterAbsKernel();
  RegisterExpKernel();
  RegisterLogKernel();
  RegisterGemmKernel();
  RegisterNotKernel();
}

} // namespace onnx_light_cpu
