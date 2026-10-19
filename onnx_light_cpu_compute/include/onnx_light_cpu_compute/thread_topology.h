// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace onnx_light_cpu {

/// Relative performance class reported for one physical CPU core.
enum class CpuCoreKind {
  kUnknown,
  kPerformance,
  kEfficiency,
};

/// Cross-platform logical processor target.
struct CpuAffinity {
  std::uint16_t group = 0;
  std::uint16_t index = 0;
};

/// One logical processor and its physical-core relationship.
struct CpuThread {
  CpuAffinity affinity;
  std::uint32_t core_index = 0;
  CpuCoreKind core_kind = CpuCoreKind::kUnknown;
  bool smt_primary = true;
};

/// Processor topology visible to the current process.
struct CpuTopology {
  std::vector<CpuThread> threads;
  std::size_t logical_thread_count = 1;
  std::size_t physical_core_count = 1;
  std::size_t performance_core_count = 0;
  std::size_t efficiency_core_count = 0;
};

/// Returns the process-visible topology, detected once.
const CpuTopology &GetCpuTopology();

/// Chooses physical cores by performance class before appending SMT siblings.
std::vector<CpuAffinity> SelectCpuAffinities(const CpuTopology &topology, std::size_t count);

/// Pins the calling thread to one logical processor where supported.
bool SetCurrentThreadAffinity(const CpuAffinity &affinity) noexcept;

/// Returns the logical processor currently executing the calling thread.
bool GetCurrentThreadAffinity(CpuAffinity &affinity) noexcept;

} // namespace onnx_light_cpu
