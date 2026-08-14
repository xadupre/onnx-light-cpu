// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/thread_topology.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>

#include <fstream>
#include <string>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace onnx_light_cpu {

namespace {

#if defined(__linux__)

struct LinuxLogicalCpu {
  int cpu = 0;
  int package = 0;
  int core = 0;
  int core_type = 0;
  int capacity = 0;
};

bool ReadInteger(const std::string &path, int &value) {
  std::ifstream stream(path);
  return static_cast<bool>(stream >> value);
}

CpuTopology DetectLinuxTopology() {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  const bool affinity_available = sched_getaffinity(0, sizeof(allowed), &allowed) == 0;

  std::vector<LinuxLogicalCpu> logical_cpus;
  const unsigned int hardware_threads = std::max(1u, std::thread::hardware_concurrency());
  const int limit = affinity_available ? CPU_SETSIZE : static_cast<int>(hardware_threads);
  for (int cpu = 0; cpu < limit; ++cpu) {
    if (affinity_available && !CPU_ISSET(cpu, &allowed)) {
      continue;
    }
    LinuxLogicalCpu logical;
    logical.cpu = cpu;
    const std::string topology = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    if (!ReadInteger(topology + "physical_package_id", logical.package)) {
      logical.package = 0;
    }
    if (!ReadInteger(topology + "core_id", logical.core)) {
      logical.core = cpu;
    }
    ReadInteger(topology + "core_type", logical.core_type);
    ReadInteger("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpu_capacity",
                logical.capacity);
    logical_cpus.push_back(logical);
  }
  if (logical_cpus.empty()) {
    logical_cpus.push_back({});
  }

  const int maximum_capacity =
      std::max_element(logical_cpus.begin(), logical_cpus.end(),
                       [](const LinuxLogicalCpu &left, const LinuxLogicalCpu &right) {
                         return left.capacity < right.capacity;
                       })
          ->capacity;
  std::vector<std::pair<int, int>> cores;
  CpuTopology topology;
  topology.threads.clear();
  for (const LinuxLogicalCpu &logical : logical_cpus) {
    const std::pair<int, int> key{logical.package, logical.core};
    auto core = std::find(cores.begin(), cores.end(), key);
    const bool primary = core == cores.end();
    if (primary) {
      cores.push_back(key);
      core = cores.end() - 1;
    }

    CpuCoreKind kind = CpuCoreKind::kUnknown;
    if (logical.core_type == 2) {
      kind = CpuCoreKind::kPerformance;
    } else if (logical.core_type == 1) {
      kind = CpuCoreKind::kEfficiency;
    } else if (maximum_capacity > 0 && logical.capacity > 0) {
      kind = logical.capacity * 4 >= maximum_capacity * 3 ? CpuCoreKind::kPerformance
                                                          : CpuCoreKind::kEfficiency;
    }
    topology.threads.push_back({{0, static_cast<std::uint16_t>(logical.cpu)},
                                static_cast<std::uint32_t>(core - cores.begin()),
                                kind,
                                primary});
  }
  topology.logical_thread_count = topology.threads.size();
  topology.physical_core_count = cores.size();
  for (const CpuThread &thread : topology.threads) {
    if (!thread.smt_primary) {
      continue;
    }
    topology.performance_core_count += thread.core_kind == CpuCoreKind::kPerformance;
    topology.efficiency_core_count += thread.core_kind == CpuCoreKind::kEfficiency;
  }
  return topology;
}

#elif defined(_WIN32)

CpuTopology DetectWindowsTopology() {
  DWORD bytes = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
  std::vector<std::byte> buffer(bytes);
  if (bytes == 0 ||
      !GetLogicalProcessorInformationEx(
          RelationProcessorCore,
          reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &bytes)) {
    return {};
  }

  struct WindowsCore {
    BYTE efficiency_class = 0;
    std::vector<CpuAffinity> affinities;
  };
  std::vector<WindowsCore> cores;
  BYTE minimum_class = 255;
  BYTE maximum_class = 0;
  for (DWORD offset = 0; offset < bytes;) {
    const auto *entry =
        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buffer.data() + offset);
    if (entry->Relationship == RelationProcessorCore) {
      WindowsCore core;
      core.efficiency_class = entry->Processor.EfficiencyClass;
      minimum_class = std::min(minimum_class, core.efficiency_class);
      maximum_class = std::max(maximum_class, core.efficiency_class);
      for (WORD group_index = 0; group_index < entry->Processor.GroupCount; ++group_index) {
        const GROUP_AFFINITY &group = entry->Processor.GroupMask[group_index];
        for (std::uint16_t bit = 0; bit < sizeof(KAFFINITY) * 8; ++bit) {
          if ((group.Mask & (KAFFINITY{1} << bit)) != 0) {
            core.affinities.push_back({group.Group, bit});
          }
        }
      }
      cores.push_back(std::move(core));
    }
    offset += entry->Size;
  }

  CpuTopology topology;
  topology.threads.clear();
  for (std::size_t core_index = 0; core_index < cores.size(); ++core_index) {
    const WindowsCore &core = cores[core_index];
    CpuCoreKind kind = CpuCoreKind::kUnknown;
    if (maximum_class != minimum_class) {
      kind = core.efficiency_class == maximum_class ? CpuCoreKind::kPerformance
                                                    : CpuCoreKind::kEfficiency;
    }
    for (std::size_t index = 0; index < core.affinities.size(); ++index) {
      topology.threads.push_back(
          {core.affinities[index], static_cast<std::uint32_t>(core_index), kind, index == 0});
    }
    topology.performance_core_count += kind == CpuCoreKind::kPerformance;
    topology.efficiency_core_count += kind == CpuCoreKind::kEfficiency;
  }
  topology.logical_thread_count = std::max<std::size_t>(1, topology.threads.size());
  topology.physical_core_count = std::max<std::size_t>(1, cores.size());
  if (topology.threads.empty()) {
    topology.threads.push_back({});
  }
  return topology;
}

#elif defined(__APPLE__)

std::size_t ReadSysctlCount(const char *name) {
  std::uint32_t value = 0;
  std::size_t size = sizeof(value);
  return sysctlbyname(name, &value, &size, nullptr, 0) == 0 && value != 0 ? value : 1;
}

CpuTopology DetectAppleTopology() {
  CpuTopology topology;
  topology.logical_thread_count = ReadSysctlCount("hw.logicalcpu");
  topology.physical_core_count = ReadSysctlCount("hw.physicalcpu");
  topology.threads.reserve(topology.logical_thread_count);
  for (std::size_t index = 0; index < topology.logical_thread_count; ++index) {
    topology.threads.push_back({{0, static_cast<std::uint16_t>(index)},
                                static_cast<std::uint32_t>(index % topology.physical_core_count),
                                CpuCoreKind::kUnknown,
                                index < topology.physical_core_count});
  }
  return topology;
}

#else

CpuTopology DetectGenericTopology() {
  CpuTopology topology;
  topology.logical_thread_count = std::max(1u, std::thread::hardware_concurrency());
  topology.physical_core_count = topology.logical_thread_count;
  topology.threads.reserve(topology.logical_thread_count);
  for (std::size_t index = 0; index < topology.logical_thread_count; ++index) {
    topology.threads.push_back({{0, static_cast<std::uint16_t>(index)},
                                static_cast<std::uint32_t>(index),
                                CpuCoreKind::kUnknown,
                                true});
  }
  return topology;
}

#endif

CpuTopology DetectCpuTopology() {
#if defined(__linux__)
  return DetectLinuxTopology();
#elif defined(_WIN32)
  return DetectWindowsTopology();
#elif defined(__APPLE__)
  return DetectAppleTopology();
#else
  return DetectGenericTopology();
#endif
}

int CoreKindPriority(CpuCoreKind kind) {
  switch (kind) {
  case CpuCoreKind::kPerformance:
    return 0;
  case CpuCoreKind::kUnknown:
    return 1;
  case CpuCoreKind::kEfficiency:
    return 2;
  }
  return 1;
}

} // namespace

const CpuTopology &GetCpuTopology() {
  static const CpuTopology topology = DetectCpuTopology();
  return topology;
}

std::vector<CpuAffinity> SelectCpuAffinities(const CpuTopology &topology, std::size_t count) {
  std::vector<const CpuThread *> ordered;
  ordered.reserve(topology.threads.size());
  for (bool primary : {true, false}) {
    for (int priority = 0; priority <= 2; ++priority) {
      for (const CpuThread &thread : topology.threads) {
        if (thread.smt_primary == primary && CoreKindPriority(thread.core_kind) == priority) {
          ordered.push_back(&thread);
        }
      }
    }
  }

  count = std::min(count, ordered.size());
  std::vector<CpuAffinity> affinities;
  affinities.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    affinities.push_back(ordered[index]->affinity);
  }
  return affinities;
}

bool SetCurrentThreadAffinity(const CpuAffinity &affinity) noexcept {
#if defined(__linux__)
  if (affinity.index >= CPU_SETSIZE) {
    return false;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(affinity.index, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#elif defined(_WIN32)
  GROUP_AFFINITY group_affinity{};
  group_affinity.Group = affinity.group;
  group_affinity.Mask = KAFFINITY{1} << affinity.index;
  return SetThreadGroupAffinity(GetCurrentThread(), &group_affinity, nullptr) != 0;
#else
  (void)affinity;
  return false;
#endif
}

bool GetCurrentThreadAffinity(CpuAffinity &affinity) noexcept {
#if defined(__linux__)
  const int cpu = sched_getcpu();
  if (cpu < 0 || cpu > std::numeric_limits<std::uint16_t>::max()) {
    return false;
  }
  affinity = {0, static_cast<std::uint16_t>(cpu)};
  return true;
#elif defined(_WIN32)
  PROCESSOR_NUMBER processor{};
  GetCurrentProcessorNumberEx(&processor);
  affinity = {processor.Group, processor.Number};
  return true;
#else
  (void)affinity;
  return false;
#endif
}

} // namespace onnx_light_cpu
