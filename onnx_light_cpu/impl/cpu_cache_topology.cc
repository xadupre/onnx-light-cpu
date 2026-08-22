// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/cpu_cache_topology.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ONNX_LIGHT_CPU_CACHE_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#else
#define ONNX_LIGHT_CPU_CACHE_X86 0
#endif

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace onnx_light_cpu {

namespace detail {

namespace {

#if ONNX_LIGHT_CPU_CACHE_X86

struct CpuidResult {
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;
};

CpuidResult Cpuid(unsigned int leaf, unsigned int subleaf = 0) {
  CpuidResult result;
#if defined(_MSC_VER)
  int registers[4];
  __cpuidex(registers, static_cast<int>(leaf), static_cast<int>(subleaf));
  result.eax = static_cast<unsigned int>(registers[0]);
  result.ebx = static_cast<unsigned int>(registers[1]);
  result.ecx = static_cast<unsigned int>(registers[2]);
  result.edx = static_cast<unsigned int>(registers[3]);
#else
  __cpuid_count(leaf, subleaf, result.eax, result.ebx, result.ecx, result.edx);
#endif
  return result;
}

void ReadDeterministicCaches(unsigned int leaf, std::vector<CpuCacheRecord> &records) {
  for (unsigned int index = 0;; ++index) {
    const CpuidResult cache = Cpuid(leaf, index);
    const unsigned int type = cache.eax & 0x1fu;
    if (type == 0) {
      break;
    }
    if (type > 3) {
      continue;
    }
    CpuCacheRecord record;
    record.level = (cache.eax >> 5) & 0x7u;
    record.kind = type == 1   ? CpuCacheKind::kData
                  : type == 2 ? CpuCacheKind::kInstruction
                              : CpuCacheKind::kUnified;
    const std::size_t line_size = (cache.ebx & 0xfffu) + 1;
    const std::size_t partitions = ((cache.ebx >> 12) & 0x3ffu) + 1;
    const std::size_t ways = ((cache.ebx >> 22) & 0x3ffu) + 1;
    const std::size_t sets = static_cast<std::size_t>(cache.ecx) + 1;
    record.size_bytes = line_size * partitions * ways * sets;
    record.line_size_bytes = line_size;
    record.sharing_thread_count = ((cache.eax >> 14) & 0xfffu) + 1;
    record.confidence = CpuCacheConfidence::kDetected;
    record.valid = record.level >= 1 && record.level <= 4 && record.size_bytes > 0;
    records.push_back(record);
  }
}

std::vector<CpuCacheRecord> DetectX86CpuidCacheRecords() {
  std::vector<CpuCacheRecord> records;
  const unsigned int max_basic = Cpuid(0).eax;
  if (max_basic >= 4) {
    ReadDeterministicCaches(4, records);
  }
  if (records.empty()) {
    const unsigned int max_extended = Cpuid(0x80000000u).eax;
    if (max_extended >= 0x8000001du) {
      ReadDeterministicCaches(0x8000001du, records);
    }
  }
  return records;
}

#endif // ONNX_LIGHT_CPU_CACHE_X86

#if defined(__linux__)

std::size_t ParseCacheSize(const std::string &text) {
  if (text.empty()) {
    return 0;
  }
  std::size_t value = 0;
  std::size_t parsed = 0;
  try {
    value = std::stoull(text, &parsed);
  } catch (const std::exception &) {
    return 0;
  }
  if (parsed < text.size()) {
    const char suffix = static_cast<char>(std::tolower(static_cast<unsigned char>(text[parsed])));
    if (suffix == 'k') {
      value *= 1024;
    } else if (suffix == 'm') {
      value *= 1024 * 1024;
    } else if (suffix == 'g') {
      value *= 1024 * 1024 * 1024;
    }
  }
  return value;
}

std::size_t CountSharedCpuList(const std::string &text) {
  std::size_t count = 0;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    const std::size_t dash = token.find('-');
    if (dash == std::string::npos) {
      ++count;
      continue;
    }
    try {
      const long low = std::stol(token.substr(0, dash));
      const long high = std::stol(token.substr(dash + 1));
      if (high >= low) {
        count += static_cast<std::size_t>(high - low + 1);
      }
    } catch (const std::exception &) {
      // Ignore a malformed range instead of guessing its width.
    }
  }
  return count;
}

bool ReadFirstLine(const std::string &path, std::string &line) {
  std::ifstream stream(path);
  return static_cast<bool>(std::getline(stream, line));
}

std::vector<CpuCacheRecord> DetectLinuxCacheRecords() {
  std::vector<CpuCacheRecord> records;
  for (int index = 0; index < 16; ++index) {
    const std::string base =
        "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index) + "/";
    std::string level_text;
    if (!ReadFirstLine(base + "level", level_text)) {
      if (index == 0) {
        continue;
      }
      break;
    }
    CpuCacheRecord record;
    try {
      record.level = static_cast<unsigned int>(std::stoul(level_text));
    } catch (const std::exception &) {
      record.level = 0;
    }

    std::string type_text;
    ReadFirstLine(base + "type", type_text);
    if (type_text == "Data") {
      record.kind = CpuCacheKind::kData;
    } else if (type_text == "Instruction") {
      record.kind = CpuCacheKind::kInstruction;
    } else if (type_text == "Unified") {
      record.kind = CpuCacheKind::kUnified;
    }

    std::string size_text;
    ReadFirstLine(base + "size", size_text);
    record.size_bytes = ParseCacheSize(size_text);

    std::string line_size_text;
    if (ReadFirstLine(base + "coherency_line_size", line_size_text)) {
      record.line_size_bytes = ParseCacheSize(line_size_text);
    }

    std::string shared_cpu_list;
    if (ReadFirstLine(base + "shared_cpu_list", shared_cpu_list)) {
      record.sharing_thread_count = CountSharedCpuList(shared_cpu_list);
    }

    record.confidence = record.line_size_bytes > 0 && record.sharing_thread_count > 0
                            ? CpuCacheConfidence::kDetected
                            : CpuCacheConfidence::kInferred;
    record.valid = record.level >= 1 && record.level <= 4 && record.size_bytes > 0;
    records.push_back(record);
  }
  return records;
}

#elif defined(_WIN32)

CpuCacheKind WindowsCacheKind(PROCESSOR_CACHE_TYPE type) {
  switch (type) {
  case CacheData:
    return CpuCacheKind::kData;
  case CacheInstruction:
    return CpuCacheKind::kInstruction;
  case CacheUnified:
    return CpuCacheKind::kUnified;
  default:
    return CpuCacheKind::kUnknown;
  }
}

int CountSetBits(KAFFINITY mask) {
  int count = 0;
  for (std::size_t bit = 0; bit < sizeof(KAFFINITY) * 8; ++bit) {
    count += static_cast<int>((mask & (KAFFINITY{1} << bit)) != 0);
  }
  return count;
}

std::vector<CpuCacheRecord> DetectWindowsCacheRecords() {
  std::vector<CpuCacheRecord> records;
  DWORD bytes = 0;
  GetLogicalProcessorInformationEx(RelationCache, nullptr, &bytes);
  if (bytes == 0) {
    return records;
  }
  std::vector<std::byte> buffer(bytes);
  if (!GetLogicalProcessorInformationEx(
          RelationCache, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
          &bytes)) {
    return records;
  }
  for (DWORD offset = 0; offset < bytes;) {
    const auto *entry =
        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buffer.data() + offset);
    if (entry->Relationship == RelationCache) {
      const CACHE_RELATIONSHIP &cache = entry->Cache;
      CpuCacheRecord record;
      record.level = cache.Level;
      record.kind = WindowsCacheKind(cache.Type);
      record.size_bytes = cache.CacheSize;
      record.line_size_bytes = cache.LineSize;
      std::size_t sharing = 0;
      for (WORD group_index = 0; group_index < cache.GroupCount; ++group_index) {
        sharing += static_cast<std::size_t>(CountSetBits(cache.GroupMasks[group_index].Mask));
      }
      record.sharing_thread_count = sharing;
      record.confidence = CpuCacheConfidence::kDetected;
      record.valid = record.level >= 1 && record.level <= 4 && record.size_bytes > 0;
      records.push_back(record);
    }
    offset += entry->Size;
  }
  return records;
}

#elif defined(__APPLE__)

bool ReadSysctlUint64(const char *name, std::uint64_t &value) {
  std::size_t size = sizeof(value);
  return sysctlbyname(name, &value, &size, nullptr, 0) == 0 && value != 0;
}

void AddAppleCache(std::vector<CpuCacheRecord> &records, unsigned int level, CpuCacheKind kind,
                   const char *size_key, std::uint64_t line_size) {
  std::uint64_t size = 0;
  if (!ReadSysctlUint64(size_key, size)) {
    return;
  }
  CpuCacheRecord record;
  record.level = level;
  record.kind = kind;
  record.size_bytes = static_cast<std::size_t>(size);
  record.line_size_bytes = static_cast<std::size_t>(line_size);
  // sysctlbyname reports sizes without a matching per-level line size or
  // sharing count, so every macOS record is inferred rather than detected.
  record.confidence = CpuCacheConfidence::kInferred;
  record.valid = record.size_bytes > 0;
  records.push_back(record);
}

std::vector<CpuCacheRecord> DetectAppleCacheRecords() {
  std::vector<CpuCacheRecord> records;
  std::uint64_t line_size = 0;
  ReadSysctlUint64("hw.cachelinesize", line_size);
  AddAppleCache(records, 1, CpuCacheKind::kData, "hw.l1dcachesize", line_size);
  AddAppleCache(records, 1, CpuCacheKind::kInstruction, "hw.l1icachesize", line_size);
  AddAppleCache(records, 2, CpuCacheKind::kUnified, "hw.l2cachesize", line_size);
  AddAppleCache(records, 3, CpuCacheKind::kUnified, "hw.l3cachesize", line_size);
  return records;
}

#endif

std::vector<CpuCacheRecord> DetectPlatformCacheRecords() {
#if defined(__linux__)
  return DetectLinuxCacheRecords();
#elif defined(_WIN32)
  return DetectWindowsCacheRecords();
#elif defined(__APPLE__)
  return DetectAppleCacheRecords();
#else
  return {};
#endif
}

} // namespace

CpuCacheTopology BuildCpuCacheTopology(const std::vector<CpuCacheRecord> &records,
                                       bool platform_supported) {
  CpuCacheTopology topology;
  topology.caches.reserve(records.size());
  for (const CpuCacheRecord &record : records) {
    if (!record.valid) {
      continue;
    }
    CpuCacheDescriptor descriptor;
    descriptor.level = record.level;
    descriptor.kind = record.kind;
    descriptor.size_bytes = record.size_bytes;
    descriptor.line_size_bytes = record.line_size_bytes;
    descriptor.sharing_thread_count = record.sharing_thread_count;
    descriptor.confidence = record.confidence;
    topology.caches.push_back(descriptor);
  }
  topology.platform_detected = platform_supported && !topology.caches.empty();
  return topology;
}

std::vector<CpuCacheRecord> DetectRawCpuCacheRecords() {
  std::vector<CpuCacheRecord> records = DetectPlatformCacheRecords();
  if (!records.empty()) {
    return records;
  }
#if ONNX_LIGHT_CPU_CACHE_X86
  return DetectX86CpuidCacheRecords();
#else
  return records;
#endif
}

} // namespace detail

namespace {

const CpuCacheTopology &CachedCpuCacheTopology() {
  static const CpuCacheTopology topology =
      detail::BuildCpuCacheTopology(detail::DetectRawCpuCacheRecords(), true);
  return topology;
}

} // namespace

const CpuCacheTopology &GetCpuCacheTopology() { return CachedCpuCacheTopology(); }

const CpuCacheDescriptor *FindCpuCacheDescriptor(const CpuCacheTopology &topology,
                                                 unsigned int level, CpuCacheKind kind) {
  for (const CpuCacheDescriptor &descriptor : topology.caches) {
    if (descriptor.level != level) {
      continue;
    }
    if (kind == CpuCacheKind::kUnknown) {
      if (descriptor.kind == CpuCacheKind::kData || descriptor.kind == CpuCacheKind::kUnified) {
        return &descriptor;
      }
      continue;
    }
    if (descriptor.kind == kind) {
      return &descriptor;
    }
  }
  return nullptr;
}

std::size_t CpuCacheSizeBytesOrFallback(const CpuCacheTopology &topology, unsigned int level,
                                        std::size_t fallback_bytes) {
  std::size_t best = 0;
  bool found = false;
  for (const CpuCacheDescriptor &descriptor : topology.caches) {
    if (descriptor.level != level) {
      continue;
    }
    if (descriptor.kind != CpuCacheKind::kData && descriptor.kind != CpuCacheKind::kUnified) {
      continue;
    }
    found = true;
    best = std::max(best, descriptor.size_bytes);
  }
  return found ? best : fallback_bytes;
}

} // namespace onnx_light_cpu
