// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu {

/// Returns the ``{op_type, library-qualified kernel name}`` pairs for every
/// kernel onnx-light-cpu registers, so callers can check the accelerated
/// kernels (rather than the built-in ones) are the ones being used.
///
/// Derived from :cpp:func:`CollectRegisteredKernels`'s structured inventory
/// rather than a second, hand-maintained list, so entries are ordered the
/// same way: by ``(domain, op_type, device, kernel_name)``.
const std::vector<std::pair<std::string, std::string>> &RegisteredKernelNames();

} // namespace onnx_light_cpu
