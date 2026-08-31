// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

namespace {

// The scope currently bounding kernel registration on this thread, if any.
// ``RegisterAllKernels`` always opens one; ``CollectRegisteredKernels`` opens
// one first so the nested one ``RegisterAllKernels`` then opens reuses it
// (see ``KernelRegistrationScope``'s constructor).
KernelRegistrationScope *&CurrentScope() {
  static thread_local KernelRegistrationScope *current = nullptr;
  return current;
}

} // namespace

KernelRegistrationScope::KernelRegistrationScope(std::vector<KernelRegistration> *inventory) {
  if (CurrentScope() != nullptr) {
    // Reuse the enclosing scope (e.g. the one ``CollectRegisteredKernels``
    // opened around this ``RegisterAllKernels`` call) instead of shadowing
    // it: only the outermost scope should decide whether this pass installs
    // kernels or only collects metadata.
    owns_ = false;
    return;
  }
  owns_ = true;
  inventory_ = inventory;
  CurrentScope() = this;
}

KernelRegistrationScope::KernelRegistrationScope(
    std::vector<KernelFactoryRegistration> *factories) {
  if (CurrentScope() != nullptr) {
    owns_ = false;
    return;
  }
  owns_ = true;
  factories_ = factories;
  CurrentScope() = this;
}

KernelRegistrationScope::~KernelRegistrationScope() {
  if (owns_) {
    CurrentScope() = nullptr;
  }
}

void RegisterKernel(KernelRegistration info, rt_ns::NodeKernelFn fn) {
  info.domain = ONNX_LIGHT_NAMESPACE::NormaliseDomain(info.domain);

  // Defensive fallback for a ``RegisterKernel`` call made outside any
  // ``RegisterAllKernels``/``CollectRegisteredKernels`` pass (e.g. directly
  // from a test): still bounds the duplicate check to this one call instead
  // of dereferencing a null scope. Only constructed when no scope is already
  // active, so the common path (called from within ``RegisterAllKernels``)
  // does not pay for an unused scope.
  std::optional<KernelRegistrationScope> fallback_scope;
  if (CurrentScope() == nullptr) {
    fallback_scope.emplace();
  }
  KernelRegistrationScope *scope = CurrentScope();

  const auto dispatch_key = std::make_tuple(info.domain, info.op_type, info.device);
  const auto key = std::make_tuple(info.domain, info.op_type, info.device, info.since_version,
                                   info.until_version);
  if (!scope->seen_.insert(key).second) {
    throw std::invalid_argument("onnx_light_cpu::RegisterKernel: duplicate registration for "
                                "domain '" +
                                info.domain + "', operator '" + info.op_type + "', device " +
                                std::to_string(static_cast<int32_t>(info.device)) + ".");
  }

  if (scope->inventory_ != nullptr) {
    scope->inventory_->push_back(std::move(info));
    return;
  }
  if (scope->factories_ != nullptr) {
    scope->factories_->push_back({std::move(info), std::move(fn)});
    return;
  }

  if (scope->installed_.insert(dispatch_key).second) {
    rt_ns::RegisterKernelFn(info.domain, info.op_type, info.device, std::move(fn));
  }
}

std::vector<KernelRegistration> CollectRegisteredKernels() {
  return CollectRegisteredKernels(MicrosoftKernelImplementation::OPTIMIZED);
}

std::vector<KernelRegistration>
CollectRegisteredKernels(MicrosoftKernelImplementation implementation) {
  std::vector<KernelRegistration> inventory;
  {
    KernelRegistrationScope scope(&inventory);
    RegisterAllKernels(implementation);
  }

  std::sort(inventory.begin(), inventory.end(),
            [](const KernelRegistration &a, const KernelRegistration &b) {
              return std::tie(a.domain, a.op_type, a.device, a.kernel_name, a.since_version,
                              a.until_version) < std::tie(b.domain, b.op_type, b.device,
                                                          b.kernel_name, b.since_version,
                                                          b.until_version);
            });
  return inventory;
}

std::vector<KernelFactoryRegistration>
CollectKernelFactories(MicrosoftKernelImplementation implementation) {
  std::vector<KernelFactoryRegistration> factories;
  {
    KernelRegistrationScope scope(&factories);
    RegisterAllKernels(implementation);
  }
  return factories;
}

} // namespace onnx_light_cpu
