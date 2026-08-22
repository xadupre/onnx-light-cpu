// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_cpu {

/// Structured metadata describing one onnx-light-cpu kernel registration.
///
/// Every ``Register*Kernel[s]`` function builds one (or more) of these and
/// hands it to :cpp:func:`RegisterKernel` together with the node factory, so
/// this record always mirrors exactly what is (or would be) installed into
/// onnx-light's shared ``KernelDispatchTable``.
struct KernelRegistration {
  /// ONNX operator domain. Callers may pass the empty string (as
  /// :cpp:func:`onnx_light::core::runtime::RegisterKernelFn` also accepts) to
  /// mean the default ONNX domain; :cpp:func:`RegisterKernel` normalises this
  /// field to ``"ai.onnx"`` before storing or forwarding it, so every record
  /// this struct ends up in (installed or collected) always reads the
  /// normalised form.
  std::string domain;
  /// ONNX operator type name, e.g. ``"Abs"``.
  std::string op_type;
  /// Device the kernel runs on, e.g.
  /// :cpp:enumerator:`onnx_light::core::symbolic::Device::kCPU`.
  ONNX_LIGHT_NAMESPACE::core::symbolic::Device device =
      ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  /// Library-qualified C++ kernel class name (e.g. ``AbsKernel::kName``).
  std::string kernel_name;
  /// Element (``TensorProto::DataType``) values the kernel accepts for its
  /// primary tensor operands, in the order the kernel implementation checks
  /// them.
  std::vector<ONNX_LIGHT_NAMESPACE::core::runtime::DataType> types;
  /// Inclusive opset lower bound, when the kernel only supports opsets from
  /// a given version onward. ``std::nullopt`` means "no lower bound".
  std::optional<std::int64_t> since_version;
  /// Inclusive opset upper bound, when the kernel stops applying beyond a
  /// given version. ``std::nullopt`` means "no upper bound".
  std::optional<std::int64_t> until_version;
};

/// RAII scope bounding one kernel-registration pass.
///
/// :cpp:func:`RegisterAllKernels` opens one of these around its body so every
/// :cpp:func:`RegisterKernel` call it makes is checked for duplicates against
/// the others in that same call. :cpp:func:`CollectRegisteredKernels` opens
/// one first (with an inventory to collect into) and then calls
/// :cpp:func:`RegisterAllKernels`; because a scope is already active, the
/// nested one that function opens reuses it instead of starting a new,
/// non-collecting scope, so the whole pass stays in inventory mode.
///
/// This is an implementation detail shared by ``kernel_registration.cc`` and
/// ``register_kernels.cc``; callers outside those two translation units
/// should not need to construct it directly.
class KernelRegistrationScope {
public:
  /// When `inventory` is non-null and no scope is currently active on this
  /// thread, every :cpp:func:`RegisterKernel` call made for the lifetime of
  /// this object records its metadata into `*inventory` instead of
  /// installing a kernel.
  explicit KernelRegistrationScope(std::vector<KernelRegistration> *inventory = nullptr);
  ~KernelRegistrationScope();

  KernelRegistrationScope(const KernelRegistrationScope &) = delete;
  KernelRegistrationScope &operator=(const KernelRegistrationScope &) = delete;

private:
  friend void RegisterKernel(KernelRegistration info,
                             ONNX_LIGHT_NAMESPACE::core::runtime::NodeKernelFn fn);

  std::set<std::tuple<std::string, std::string, ONNX_LIGHT_NAMESPACE::core::symbolic::Device>>
      seen_;
  std::vector<KernelRegistration> *inventory_ = nullptr;
  bool owns_ = false;
};

/// Registers `info` for the shared onnx-light dispatch table.
///
/// In normal mode (the default, used by :cpp:func:`RegisterAllKernels`
/// itself), installs `fn` into
/// :cpp:func:`onnx_light::core::runtime::RegisterKernelFn` for
/// ``(info.domain, info.op_type, info.device)`` exactly as a direct call
/// would, so runtime dispatch is unchanged.
///
/// While an inventory collection started by
/// :cpp:func:`CollectRegisteredKernels` is in progress on the calling thread,
/// this instead records `info` into that inventory and returns without
/// constructing `fn` or touching the dispatch table, so collection can never
/// install, replace, or execute a kernel.
///
/// Every call (in either mode) is checked against every other call made
/// since the start of the current top-level :cpp:func:`RegisterAllKernels`
/// or :cpp:func:`CollectRegisteredKernels` invocation: a repeated
/// ``(domain, op_type, device)`` triple throws ``std::invalid_argument``
/// immediately, before any dispatch-table mutation or inventory insertion
/// happens for that call.
void RegisterKernel(KernelRegistration info, ONNX_LIGHT_NAMESPACE::core::runtime::NodeKernelFn fn);

/// Runs :cpp:func:`RegisterAllKernels` in inventory mode and returns the
/// structured metadata for every onnx-light-cpu kernel registration it
/// performs, sorted deterministically by
/// ``(domain, op_type, device, kernel_name)``.
///
/// Collection never mutates onnx-light's shared ``KernelDispatchTable``:
/// every :cpp:func:`RegisterKernel` call made while collecting only appends
/// to the returned vector.
std::vector<KernelRegistration> CollectRegisteredKernels();

} // namespace onnx_light_cpu
