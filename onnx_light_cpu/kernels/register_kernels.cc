// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_light_cpu/kernels/attention/attention_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/bias_gelu_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/cdist_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/group_query_attention_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_bias_gelu_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_cdist_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_group_query_attention_kernel.h"
#include "onnx_light_cpu/kernels/elementwise/binary_kernel.h"
#include "onnx_light_cpu/kernels/elementwise/variadic_kernel.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/logical/not_kernel.h"
#include "onnx_light_cpu/kernels/math/abs_kernel.h"
#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"
#include "onnx_light_cpu/kernels/math/gemm_kernel.h"
#include "onnx_light_cpu/kernels/math/integer_matmul_kernel.h"
#include "onnx_light_cpu/kernels/math/matmul_kernel.h"
#include "onnx_light_cpu/kernels/math/normalization_kernel.h"
#include "onnx_light_cpu/kernels/math/rms_normalization_kernel.h"
#include "onnx_light_cpu/kernels/math/swiglu_kernel.h"
#include "onnx_light_cpu/kernels/traditionalml/tree_ensemble_kernel.h"

#include <stdexcept>

namespace onnx_light_cpu {

void RegisterMicrosoftKernels(MicrosoftKernelImplementation implementation) {
  KernelRegistrationScope scope;
  switch (implementation) {
  case MicrosoftKernelImplementation::NAIVE:
    RegisterNaiveBiasGeluKernel();
    RegisterNaiveCDistKernel();
    RegisterNaiveGroupQueryAttentionKernel();
    return;
  case MicrosoftKernelImplementation::OPTIMIZED:
    RegisterBiasGeluKernel();
    RegisterCDistKernel();
    RegisterGroupQueryAttentionKernel();
    return;
  }
  throw std::invalid_argument("unknown MicrosoftKernelImplementation");
}

void RegisterAllKernels() { RegisterAllKernels(MicrosoftKernelImplementation::OPTIMIZED); }

void RegisterAllKernels(MicrosoftKernelImplementation implementation) {
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
  RegisterNormalizationKernels();
  RegisterRmsNormalizationKernel();
  RegisterSwiGLUKernel();
  RegisterTreeEnsembleKernel();
  RegisterVariadicElementwiseKernels();
  RegisterMicrosoftKernels(implementation);
}

} // namespace onnx_light_cpu
