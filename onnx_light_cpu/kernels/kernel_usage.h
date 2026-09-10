// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace onnx_light_cpu {

/// Maximum number of invocations retained between clears; later records are dropped.
inline constexpr std::size_t kMaxKernelUsageRecords = 4096;

/// Records that the onnx-light-cpu kernel identified by ``name`` ran.
///
/// Every onnx-light-cpu kernel calls this from :cpp:func:`KernelBase::Run`
/// with its own library-qualified name (``AbsKernel::kName`` and friends), so
/// callers can observe which kernels a model actually dispatched to. The log
/// is a process-wide singleton guarded by a mutex, so recording is safe to
/// call from multiple threads; it is intended for tests and diagnostics.
/// Recording is disabled by default. When enabled, only the first
/// ``kMaxKernelUsageRecords`` invocations since the last clear are retained.
void RecordKernelUsage(std::string_view name);

/// Returns the library-qualified names of the kernels recorded by
/// :cpp:func:`RecordKernelUsage` since the last
/// :cpp:func:`ClearUsedKernelNames`, in mutex acquisition order.
/// Returns an independent snapshot without consuming records. Retrieval,
/// recording, and clearing are serialized by the same mutex.
std::vector<std::string> UsedKernelNames();

/// Clears the kernel-usage log populated by :cpp:func:`RecordKernelUsage`.
/// Does not change whether recording is enabled. Concurrent records may fall
/// before or after the clear according to mutex acquisition order.
void ClearUsedKernelNames();

/// Enables or disables per-invocation kernel usage recording.
///
/// Disabling recording removes the mutex and log-allocation overhead from
/// performance measurements while leaving kernel dispatch unchanged. The
/// disabled path only loads an atomic flag. Toggling does not clear the log.
/// Disabling waits for any active append; no further records are appended
/// until recording is enabled again.
void SetKernelUsageRecording(bool enabled) noexcept;

/// Returns the ``{op_type, library-qualified kernel name}`` pairs for every
/// kernel onnx-light-cpu registers, so callers can check the accelerated
/// kernels (rather than the built-in ones) are the ones being used.
///
/// Derived from :cpp:func:`CollectRegisteredKernels`'s structured inventory
/// rather than a second, hand-maintained list, so entries are ordered the
/// same way: by ``(domain, op_type, device, kernel_name)``.
const std::vector<std::pair<std::string, std::string>> &RegisteredKernelNames();

} // namespace onnx_light_cpu
