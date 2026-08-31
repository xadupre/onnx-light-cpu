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

#include "onnx_proto/onnx_helper.h"

#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

using FactoryKey = std::pair<std::string, std::string>;

std::string NormalizedDomain(const std::string &domain) {
  return ONNX_LIGHT_NAMESPACE::NormaliseDomain(domain);
}

std::vector<KernelFactoryRegistration>
UniqueFactories(MicrosoftKernelImplementation implementation) {
  std::vector<KernelFactoryRegistration> unique;
  std::set<FactoryKey> seen;
  for (KernelFactoryRegistration &entry : CollectKernelFactories(implementation)) {
    const FactoryKey key{entry.info.domain, entry.info.op_type};
    if (seen.insert(key).second) {
      unique.push_back(std::move(entry));
    }
  }
  return unique;
}

KernelFactoryRegistration FindFactory(const std::string &domain, const std::string &op_type,
                                      MicrosoftKernelImplementation implementation) {
  const std::string normalized_domain = NormalizedDomain(domain);
  for (KernelFactoryRegistration &entry : UniqueFactories(implementation)) {
    if (entry.info.domain == normalized_domain && entry.info.op_type == op_type) {
      return std::move(entry);
    }
  }
  throw std::invalid_argument("onnx_light_cpu: unknown kernel '" + normalized_domain + ":" +
                              op_type + "'.");
}

rt_ns::CustomKernelFn AsSessionKernel(rt_ns::NodeKernelFn factory) {
  return [factory = std::move(factory)](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                        rt_ns::RuntimeContext &runtime) {
    std::unique_ptr<rt_ns::KernelBase> kernel = factory(node, runtime);
    if (kernel == nullptr) {
      throw std::runtime_error("onnx_light_cpu: kernel factory returned null.");
    }
    kernel->Run(runtime);
  };
}

} // namespace

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

bool RegisterKernelGlobal(const std::string &domain, const std::string &op_type, bool replace,
                          MicrosoftKernelImplementation implementation) {
  KernelFactoryRegistration entry = FindFactory(domain, op_type, implementation);
  return rt_ns::RegisterKernelFn(entry.info.domain, entry.info.op_type, entry.info.device,
                                 std::move(entry.factory), replace);
}

std::size_t RegisterAllKernelsGlobal(bool replace, MicrosoftKernelImplementation implementation) {
  std::size_t count = 0;
  for (KernelFactoryRegistration &entry : UniqueFactories(implementation)) {
    count += rt_ns::RegisterKernelFn(entry.info.domain, entry.info.op_type, entry.info.device,
                                     std::move(entry.factory), replace)
                 ? 1
                 : 0;
  }
  return count;
}

bool RegisterKernelForSession(rt_ns::RuntimeContext &session, const std::string &domain,
                              const std::string &op_type, bool replace,
                              MicrosoftKernelImplementation implementation) {
  KernelFactoryRegistration entry = FindFactory(domain, op_type, implementation);
  const std::string key = entry.info.domain + ":" + entry.info.op_type;
  if (!replace && session.custom_kernels().find(key) != session.custom_kernels().end()) {
    return false;
  }
  session.RegisterCustomKernel(entry.info.domain, entry.info.op_type,
                               AsSessionKernel(std::move(entry.factory)));
  return true;
}

std::size_t RegisterAllKernelsForSession(rt_ns::RuntimeContext &session, bool replace,
                                         MicrosoftKernelImplementation implementation) {
  std::size_t count = 0;
  for (KernelFactoryRegistration &entry : UniqueFactories(implementation)) {
    const std::string key = entry.info.domain + ":" + entry.info.op_type;
    if (!replace && session.custom_kernels().find(key) != session.custom_kernels().end()) {
      continue;
    }
    session.RegisterCustomKernel(entry.info.domain, entry.info.op_type,
                                 AsSessionKernel(std::move(entry.factory)));
    ++count;
  }
  return count;
}

} // namespace onnx_light_cpu
