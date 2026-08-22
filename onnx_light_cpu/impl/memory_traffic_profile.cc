// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/memory_traffic_profile.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#include "onnx_light_cpu/impl/thread_topology.h"

#if defined(_MSC_VER)
#include <intrin.h>
#define ONNX_LIGHT_CPU_NOINLINE __declspec(noinline)
#else
#define ONNX_LIGHT_CPU_NOINLINE __attribute__((noinline))
#endif

namespace onnx_light_cpu {

namespace {

using Clock = std::chrono::steady_clock;

// Volatile module-level sink. Every measurement writes its post-timing
// checksum here so the compiler cannot prove the timed stores or loads are
// dead, without adding any consumption inside a timed sample.
volatile std::uint64_t g_memory_profile_sink = 0;

double ToSeconds(Clock::duration duration) {
  return std::chrono::duration<double>(duration).count();
}

double Median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t count = values.size();
  return count % 2 == 1 ? values[count / 2] : 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

double Quantile(const std::vector<double> &sorted_values, double q) {
  if (sorted_values.empty()) {
    return 0.0;
  }
  if (sorted_values.size() == 1) {
    return sorted_values.front();
  }
  const double position = q * static_cast<double>(sorted_values.size() - 1);
  const std::size_t low = static_cast<std::size_t>(std::floor(position));
  const std::size_t high = static_cast<std::size_t>(std::ceil(position));
  if (low == high) {
    return sorted_values[low];
  }
  const double fraction = position - static_cast<double>(low);
  return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction;
}

double InterquartileRange(std::vector<double> values) {
  if (values.size() < 2) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  return Quantile(values, 0.75) - Quantile(values, 0.25);
}

// Heap-allocated, 64-byte-aligned, prefaulted storage for one participant's
// stream. Allocation and prefaulting happen once, before any timing.
class AlignedBuffer {
public:
  AlignedBuffer() = default;

  explicit AlignedBuffer(std::size_t element_count) : count_(element_count) {
    if (count_ == 0) {
      return;
    }
    constexpr std::size_t kAlignment = 64;
    const std::size_t requested_bytes = count_ * sizeof(std::uint64_t);
    const std::size_t aligned_bytes =
        ((requested_bytes + kAlignment - 1) / kAlignment) * kAlignment;
#if defined(_WIN32)
    data_ = static_cast<std::uint64_t *>(_aligned_malloc(aligned_bytes, kAlignment));
#else
    data_ = static_cast<std::uint64_t *>(std::aligned_alloc(kAlignment, aligned_bytes));
#endif
    if (data_ == nullptr) {
      count_ = 0;
      return;
    }
    // Prefault every element so first-touch page faults never happen inside
    // a timed sample.
    for (std::size_t index = 0; index < count_; ++index) {
      data_[index] = static_cast<std::uint64_t>(index);
    }
  }

  AlignedBuffer(const AlignedBuffer &) = delete;
  AlignedBuffer &operator=(const AlignedBuffer &) = delete;

  AlignedBuffer(AlignedBuffer &&other) noexcept { *this = std::move(other); }

  AlignedBuffer &operator=(AlignedBuffer &&other) noexcept {
    if (this != &other) {
      Release();
      data_ = other.data_;
      count_ = other.count_;
      other.data_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  ~AlignedBuffer() { Release(); }

  std::uint64_t *data() { return data_; }
  const std::uint64_t *data() const { return data_; }
  std::size_t count() const { return count_; }
  bool valid() const { return count_ == 0 || data_ != nullptr; }

private:
  void Release() {
    if (data_ != nullptr) {
#if defined(_WIN32)
      _aligned_free(data_);
#else
      std::free(data_);
#endif
      data_ = nullptr;
    }
    count_ = 0;
  }

  std::uint64_t *data_ = nullptr;
  std::size_t count_ = 0;
};

ONNX_LIGHT_CPU_NOINLINE std::uint64_t RunReadKernel(const std::uint64_t *data, std::size_t count,
                                                    std::size_t passes) {
  std::uint64_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
  for (std::size_t pass = 0; pass < passes; ++pass) {
    std::size_t index = 0;
    for (; index + 4 <= count; index += 4) {
      acc0 += data[index];
      acc1 += data[index + 1];
      acc2 += data[index + 2];
      acc3 += data[index + 3];
    }
    for (; index < count; ++index) {
      acc0 += data[index];
    }
  }
  return acc0 + acc1 + acc2 + acc3;
}

ONNX_LIGHT_CPU_NOINLINE void RunWriteKernel(std::uint64_t *data, std::size_t count,
                                            std::size_t passes, std::uint64_t seed) {
  for (std::size_t pass = 0; pass < passes; ++pass) {
    const std::uint64_t base = seed + pass;
    for (std::size_t index = 0; index < count; ++index) {
      data[index] = base + static_cast<std::uint64_t>(index);
    }
  }
}

ONNX_LIGHT_CPU_NOINLINE void RunCopyKernel(const std::uint64_t *src, std::uint64_t *dst,
                                           std::size_t count, std::size_t passes) {
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t index = 0; index < count; ++index) {
      dst[index] = src[index];
    }
  }
}

ONNX_LIGHT_CPU_NOINLINE void RunReadModifyWriteKernel(std::uint64_t *data, std::size_t count,
                                                      std::size_t passes) {
  for (std::size_t pass = 0; pass < passes; ++pass) {
    for (std::size_t index = 0; index < count; ++index) {
      data[index] = data[index] * 2654435761ull + 1ull;
    }
  }
}

// One participant's disjoint storage for a bandwidth measurement.
struct BandwidthParticipant {
  AlignedBuffer primary;
  AlignedBuffer secondary; // Only used by copy (destination stream).
};

// Runs one full timed round (``passes`` internal repetitions of ``mode`` over
// ``participant``'s buffers) and returns a value that keeps the compiler from
// discarding the work; the caller stores it outside the timed interval.
std::uint64_t RunBandwidthRound(MemoryTrafficMode mode, BandwidthParticipant &participant,
                                std::size_t passes, std::uint64_t seed) {
  switch (mode) {
  case MemoryTrafficMode::kRead:
    return RunReadKernel(participant.primary.data(), participant.primary.count(), passes);
  case MemoryTrafficMode::kWrite:
    RunWriteKernel(participant.primary.data(), participant.primary.count(), passes, seed);
    return 0;
  case MemoryTrafficMode::kCopy:
    RunCopyKernel(participant.primary.data(), participant.secondary.data(),
                  participant.primary.count(), passes);
    return 0;
  case MemoryTrafficMode::kReadModifyWrite:
    RunReadModifyWriteKernel(participant.primary.data(), participant.primary.count(), passes);
    return 0;
  }
  return 0;
}

// Coordinates a fixed set of persistent worker threads (participants 1..N-1)
// through repeated timed rounds without creating or joining threads inside
// any timed sample. Participant 0 always runs on the calling thread.
class BandwidthWorkerPool {
public:
  BandwidthWorkerPool(MemoryTrafficMode mode, std::vector<BandwidthParticipant> &participants,
                      const std::vector<CpuAffinity> &affinities, std::uint64_t seed)
      : mode_(mode), participants_(participants),
        barrier_(static_cast<std::ptrdiff_t>(participants.size())) {
    for (std::size_t index = 1; index < participants_.size(); ++index) {
      const CpuAffinity *affinity = index < affinities.size() ? &affinities[index] : nullptr;
      workers_.emplace_back([this, index, affinity, seed]() {
        if (affinity != nullptr) {
          SetCurrentThreadAffinity(*affinity);
        }
        while (true) {
          barrier_.arrive_and_wait();
          const std::size_t passes = passes_.load(std::memory_order_acquire);
          if (passes == 0 && stop_.load(std::memory_order_relaxed)) {
            barrier_.arrive_and_wait();
            return;
          }
          RunBandwidthRound(mode_, participants_[index], passes, seed + index);
          barrier_.arrive_and_wait();
        }
      });
    }
  }

  ~BandwidthWorkerPool() {
    if (!workers_.empty()) {
      passes_.store(0, std::memory_order_release);
      stop_.store(true, std::memory_order_relaxed);
      barrier_.arrive_and_wait();
      barrier_.arrive_and_wait();
      for (std::thread &worker : workers_) {
        worker.join();
      }
    }
  }

  // Runs one synchronized round of ``passes`` iterations across every
  // participant and returns the aggregate wall-clock seconds elapsed.
  double RunRound(std::size_t passes, std::uint64_t seed) {
    passes_.store(passes, std::memory_order_release);
    barrier_.arrive_and_wait();
    const Clock::time_point start = Clock::now();
    const std::uint64_t checksum = RunBandwidthRound(mode_, participants_[0], passes, seed);
    barrier_.arrive_and_wait();
    const Clock::time_point end = Clock::now();
    g_memory_profile_sink += checksum;
    return ToSeconds(end - start);
  }

private:
  MemoryTrafficMode mode_;
  std::vector<BandwidthParticipant> &participants_;
  std::barrier<> barrier_;
  std::vector<std::thread> workers_;
  std::atomic<std::size_t> passes_{0};
  std::atomic<bool> stop_{false};
};

constexpr double kMemoryProfileGigabyte = 1.0e9;
constexpr std::size_t kMemoryProfileMaxCalibrationPasses = 1u << 20;

} // namespace

MemoryWorkingSet SelectMemoryWorkingSet(const CpuCacheTopology &topology, MemoryProfileLevel level,
                                        std::size_t participant_count,
                                        std::size_t memory_budget_bytes) {
  participant_count = std::max<std::size_t>(participant_count, 1);
  MemoryWorkingSet result;
  if (memory_budget_bytes == 0) {
    result.unavailable_reason = MemoryProfileUnavailableReason::kMemoryBudgetExceeded;
    return result;
  }
  const std::size_t per_participant_budget = memory_budget_bytes / participant_count;

  // Use the largest reported size at each level (the same deterministic,
  // confidence-agnostic byte count GEMM blocking already relies on) so a
  // heterogeneous topology with several private-cache clusters at the same
  // level yields one reproducible working set instead of depending on
  // descriptor insertion order.
  const std::size_t l1_bytes = CpuCacheSizeBytesOrFallback(topology, 1, 0);
  const std::size_t l2_bytes = CpuCacheSizeBytesOrFallback(topology, 2, 0);
  const std::size_t l3_bytes = CpuCacheSizeBytesOrFallback(topology, 3, 0);

  std::size_t candidate_bytes = 0;
  switch (level) {
  case MemoryProfileLevel::kL1: {
    if (l1_bytes == 0) {
      result.unavailable_reason = MemoryProfileUnavailableReason::kLevelUnavailable;
      return result;
    }
    candidate_bytes = l1_bytes / 2;
    break;
  }
  case MemoryProfileLevel::kL2: {
    if (l2_bytes == 0) {
      result.unavailable_reason = MemoryProfileUnavailableReason::kLevelUnavailable;
      return result;
    }
    candidate_bytes = l2_bytes / 2;
    if (candidate_bytes <= l1_bytes) {
      result.unavailable_reason = MemoryProfileUnavailableReason::kLevelUnavailable;
      return result;
    }
    break;
  }
  case MemoryProfileLevel::kL3: {
    if (l3_bytes == 0) {
      result.unavailable_reason = MemoryProfileUnavailableReason::kLevelUnavailable;
      return result;
    }
    const std::size_t aggregate_private_bytes = l1_bytes + l2_bytes;
    candidate_bytes = l3_bytes / 2;
    if (candidate_bytes <= aggregate_private_bytes) {
      result.unavailable_reason = MemoryProfileUnavailableReason::kLevelUnavailable;
      return result;
    }
    break;
  }
  case MemoryProfileLevel::kRam: {
    const std::size_t last_level_bytes =
        l3_bytes != 0 ? l3_bytes : (l2_bytes != 0 ? l2_bytes : l1_bytes);
    if (last_level_bytes == 0) {
      // No cache information at all: the whole budget is, vacuously, more
      // than twice an unknown (zero) last-level cache.
      candidate_bytes = per_participant_budget;
    } else {
      candidate_bytes = last_level_bytes * 2 + last_level_bytes / 4;
    }
    break;
  }
  }

  if (candidate_bytes == 0 || candidate_bytes > per_participant_budget) {
    result.unavailable_reason = MemoryProfileUnavailableReason::kMemoryBudgetExceeded;
    return result;
  }

  result.available = true;
  result.working_set_bytes = candidate_bytes;
  return result;
}

MemoryTrafficAccounting ComputeMemoryTrafficAccounting(MemoryTrafficMode mode,
                                                       std::size_t working_set_bytes,
                                                       std::size_t element_bytes) {
  MemoryTrafficAccounting accounting;
  if (element_bytes == 0) {
    return accounting;
  }
  switch (mode) {
  case MemoryTrafficMode::kRead:
  case MemoryTrafficMode::kWrite: {
    accounting.element_count = working_set_bytes / element_bytes;
    accounting.useful_bytes_per_pass =
        static_cast<std::uint64_t>(accounting.element_count) * element_bytes;
    break;
  }
  case MemoryTrafficMode::kCopy: {
    const std::size_t per_stream_bytes = working_set_bytes / 2;
    accounting.element_count = per_stream_bytes / element_bytes;
    accounting.useful_bytes_per_pass =
        static_cast<std::uint64_t>(accounting.element_count) * element_bytes * 2;
    break;
  }
  case MemoryTrafficMode::kReadModifyWrite: {
    accounting.element_count = working_set_bytes / element_bytes;
    accounting.useful_bytes_per_pass =
        static_cast<std::uint64_t>(accounting.element_count) * element_bytes * 2;
    break;
  }
  }
  return accounting;
}

const char *MemoryProfileTimerName() { return "std::chrono::steady_clock"; }

namespace detail {

std::vector<std::uint32_t> BuildPointerChasePermutation(std::size_t count, std::uint64_t seed) {
  std::vector<std::uint32_t> permutation(count);
  for (std::size_t index = 0; index < count; ++index) {
    permutation[index] = static_cast<std::uint32_t>(index);
  }
  if (count < 2) {
    return permutation;
  }
  // Sattolo's algorithm: produces a permutation that is a single cycle over
  // every index, so a pointer chase starting anywhere visits every element
  // exactly once before returning to its start.
  std::mt19937_64 generator(seed);
  for (std::size_t index = count - 1; index > 0; --index) {
    std::uniform_int_distribution<std::size_t> distribution(0, index - 1);
    const std::size_t swap_with = distribution(generator);
    std::swap(permutation[index], permutation[swap_with]);
  }
  return permutation;
}

} // namespace detail

namespace {

ONNX_LIGHT_CPU_NOINLINE std::uint32_t ChasePointers(const std::uint32_t *next, std::uint32_t start,
                                                    std::size_t iterations) {
  std::uint32_t index = start;
  for (std::size_t step = 0; step < iterations; ++step) {
    index = next[index];
  }
  return index;
}

struct LatencyParticipant {
  std::vector<std::uint32_t> permutation;
};

// Coordinates persistent worker threads (participants 1..N-1) through
// repeated timed pointer-chase rounds without creating or joining threads
// inside any timed sample. Participant 0 always runs on the calling thread.
// Each round records every participant's own elapsed seconds so aggregate
// latency reflects each independent dependent chain rather than only the
// outer wall-clock span.
class LatencyWorkerPool {
public:
  LatencyWorkerPool(std::vector<LatencyParticipant> &participants,
                    const std::vector<CpuAffinity> &affinities)
      : participants_(participants), barrier_(static_cast<std::ptrdiff_t>(participants.size())),
        elapsed_seconds_(participants.size(), 0.0) {
    for (std::size_t index = 1; index < participants_.size(); ++index) {
      const CpuAffinity *affinity = index < affinities.size() ? &affinities[index] : nullptr;
      workers_.emplace_back([this, index, affinity]() {
        if (affinity != nullptr) {
          SetCurrentThreadAffinity(*affinity);
        }
        while (true) {
          barrier_.arrive_and_wait();
          const std::size_t iterations = iterations_.load(std::memory_order_acquire);
          if (iterations == 0 && stop_.load(std::memory_order_relaxed)) {
            barrier_.arrive_and_wait();
            return;
          }
          const Clock::time_point local_start = Clock::now();
          const std::uint32_t last =
              ChasePointers(participants_[index].permutation.data(), 0, iterations);
          elapsed_seconds_[index] = ToSeconds(Clock::now() - local_start);
          g_memory_profile_sink += last;
          barrier_.arrive_and_wait();
        }
      });
    }
  }

  ~LatencyWorkerPool() {
    if (!workers_.empty()) {
      iterations_.store(0, std::memory_order_release);
      stop_.store(true, std::memory_order_relaxed);
      barrier_.arrive_and_wait();
      barrier_.arrive_and_wait();
      for (std::thread &worker : workers_) {
        worker.join();
      }
    }
  }

  // Runs one synchronized round of ``iterations`` dependent loads across
  // every participant and returns each participant's own elapsed seconds.
  const std::vector<double> &RunRound(std::size_t iterations) {
    iterations_.store(iterations, std::memory_order_release);
    barrier_.arrive_and_wait();
    const Clock::time_point start = Clock::now();
    const std::uint32_t last = ChasePointers(participants_[0].permutation.data(), 0, iterations);
    elapsed_seconds_[0] = ToSeconds(Clock::now() - start);
    barrier_.arrive_and_wait();
    g_memory_profile_sink += last;
    return elapsed_seconds_;
  }

private:
  std::vector<LatencyParticipant> &participants_;
  std::barrier<> barrier_;
  std::vector<std::thread> workers_;
  std::vector<double> elapsed_seconds_;
  std::atomic<std::size_t> iterations_{0};
  std::atomic<bool> stop_{false};
};

} // namespace

MemoryBandwidthResult MeasureMemoryBandwidth(MemoryProfileLevel level, MemoryTrafficMode mode,
                                             MemoryParticipantPolicy policy,
                                             const MemoryProfileOptions &options) {
  MemoryBandwidthResult result;
  result.level = level;
  result.mode = mode;
  result.policy = policy;
  result.timer_name = MemoryProfileTimerName();

  if (options.repeats == 0 || options.minimum_duration_ms <= 0.0 ||
      options.memory_budget_bytes == 0) {
    result.unavailable_reason = MemoryProfileUnavailableReason::kInvalidOptions;
    result.diagnostic = "invalid measurement options";
    return result;
  }

  std::vector<CpuAffinity> affinities;
  std::size_t participant_count = 1;
  bool affinity_pinned = false;
  if (policy == MemoryParticipantPolicy::kPhysical) {
    const CpuTopology &cpu_topology = GetCpuTopology();
    participant_count = std::max<std::size_t>(cpu_topology.physical_core_count, 1);
    affinities = SelectCpuAffinities(cpu_topology, participant_count);
    if (!affinities.empty()) {
      participant_count = affinities.size();
      affinity_pinned = true;
    } else {
      participant_count = 1;
    }
  } else if (options.explicit_single_affinity.has_value()) {
    affinity_pinned = SetCurrentThreadAffinity(*options.explicit_single_affinity);
  }

  const MemoryWorkingSet working_set = SelectMemoryWorkingSet(
      GetCpuCacheTopology(), level, participant_count, options.memory_budget_bytes);
  if (!working_set.available) {
    result.unavailable_reason = working_set.unavailable_reason;
    result.diagnostic = "working set unavailable for the requested memory level";
    return result;
  }
  result.working_set_bytes = working_set.working_set_bytes;
  result.participant_count = participant_count;
  result.affinity_pinned = affinity_pinned;

  constexpr std::size_t kElementBytes = sizeof(std::uint64_t);
  const MemoryTrafficAccounting accounting =
      ComputeMemoryTrafficAccounting(mode, working_set.working_set_bytes, kElementBytes);
  result.useful_bytes_per_pass_per_participant = accounting.useful_bytes_per_pass;
  if (accounting.element_count == 0) {
    result.unavailable_reason = MemoryProfileUnavailableReason::kLevelUnavailable;
    result.diagnostic = "working set too small to hold one element";
    return result;
  }

  std::vector<BandwidthParticipant> participants(participant_count);
  for (BandwidthParticipant &participant : participants) {
    participant.primary = AlignedBuffer(accounting.element_count);
    if (mode == MemoryTrafficMode::kCopy) {
      participant.secondary = AlignedBuffer(accounting.element_count);
    }
    if (!participant.primary.valid() || !participant.secondary.valid()) {
      result.unavailable_reason = MemoryProfileUnavailableReason::kMemoryBudgetExceeded;
      result.diagnostic = "allocation failed for the requested working set";
      return result;
    }
  }

  if (!affinity_pinned) {
    affinities.clear();
  }
  BandwidthWorkerPool pool(mode, participants, affinities, /*seed=*/0x9e3779b97f4a7c15ull);

  std::size_t passes = 1;
  const double calibration_elapsed = pool.RunRound(1, 1);
  {
    const double target_seconds = options.minimum_duration_ms / 1000.0;
    const double safe_elapsed = calibration_elapsed > 0.0 ? calibration_elapsed : 1e-9;
    const double estimated_passes = target_seconds / safe_elapsed;
    passes = static_cast<std::size_t>(std::ceil(std::max(1.0, estimated_passes)));
    passes = std::min(passes, kMemoryProfileMaxCalibrationPasses);
  }

  result.raw_gbps_samples.reserve(options.repeats);
  for (std::size_t repeat = 0; repeat < options.repeats; ++repeat) {
    const double elapsed_seconds = pool.RunRound(passes, static_cast<std::uint64_t>(repeat) + 2);
    const double total_bytes = static_cast<double>(accounting.useful_bytes_per_pass) *
                               static_cast<double>(passes) * static_cast<double>(participant_count);
    const double seconds = elapsed_seconds > 0.0 ? elapsed_seconds : 1e-12;
    result.raw_gbps_samples.push_back(total_bytes / seconds / kMemoryProfileGigabyte);
  }

  result.median_gbps = Median(result.raw_gbps_samples);
  result.dispersion_gbps = InterquartileRange(result.raw_gbps_samples);
  result.available = true;
  return result;
}

MemoryLatencyResult MeasureMemoryLatency(MemoryProfileLevel level, MemoryParticipantPolicy policy,
                                         const MemoryProfileOptions &options) {
  MemoryLatencyResult result;
  result.level = level;
  result.policy = policy;
  result.timer_name = MemoryProfileTimerName();

  if (options.repeats == 0 || options.minimum_duration_ms <= 0.0 ||
      options.memory_budget_bytes == 0) {
    result.unavailable_reason = MemoryProfileUnavailableReason::kInvalidOptions;
    result.diagnostic = "invalid measurement options";
    return result;
  }

  std::vector<CpuAffinity> affinities;
  std::size_t participant_count = 1;
  bool affinity_pinned = false;
  if (policy == MemoryParticipantPolicy::kPhysical) {
    const CpuTopology &cpu_topology = GetCpuTopology();
    participant_count = std::max<std::size_t>(cpu_topology.physical_core_count, 1);
    affinities = SelectCpuAffinities(cpu_topology, participant_count);
    if (!affinities.empty()) {
      participant_count = affinities.size();
      affinity_pinned = true;
    } else {
      participant_count = 1;
    }
  } else if (options.explicit_single_affinity.has_value()) {
    affinity_pinned = SetCurrentThreadAffinity(*options.explicit_single_affinity);
  }

  const MemoryWorkingSet working_set = SelectMemoryWorkingSet(
      GetCpuCacheTopology(), level, participant_count, options.memory_budget_bytes);
  if (!working_set.available) {
    result.unavailable_reason = working_set.unavailable_reason;
    result.diagnostic = "working set unavailable for the requested memory level";
    return result;
  }
  result.working_set_bytes = working_set.working_set_bytes;
  result.participant_count = participant_count;
  result.affinity_pinned = affinity_pinned;

  constexpr std::size_t kElementBytes = sizeof(std::uint32_t);
  const std::size_t element_count = working_set.working_set_bytes / kElementBytes;
  if (element_count < 2) {
    result.unavailable_reason = MemoryProfileUnavailableReason::kLevelUnavailable;
    result.diagnostic = "working set too small for a pointer chase";
    return result;
  }

  // Build and validate the permutation for every participant outside any
  // timed sample.
  std::vector<LatencyParticipant> participants(participant_count);
  for (std::size_t index = 0; index < participant_count; ++index) {
    participants[index].permutation =
        detail::BuildPointerChasePermutation(element_count, 0xda3e39cb94b95bdbull + index);
  }

  // Warm up (prefault the permutation array via one untimed chase) and
  // calibrate the number of dependent loads needed to exceed the requested
  // minimum duration.
  ChasePointers(participants[0].permutation.data(), 0, element_count);

  if (!affinity_pinned) {
    affinities.clear();
  }
  LatencyWorkerPool pool(participants, affinities);

  std::size_t iterations = element_count;
  {
    const std::vector<double> &calibration_elapsed = pool.RunRound(iterations);
    const double target_seconds = options.minimum_duration_ms / 1000.0;
    const double safe_elapsed = calibration_elapsed[0] > 0.0 ? calibration_elapsed[0] : 1e-9;
    const double scale = target_seconds / safe_elapsed;
    const double estimated_iterations = static_cast<double>(iterations) * std::max(1.0, scale);
    iterations = static_cast<std::size_t>(
        std::min(estimated_iterations, static_cast<double>(kMemoryProfileMaxCalibrationPasses) *
                                           static_cast<double>(element_count)));
    iterations = std::max<std::size_t>(iterations, element_count);
  }

  result.raw_ns_per_load_samples.reserve(options.repeats);
  for (std::size_t repeat = 0; repeat < options.repeats; ++repeat) {
    const std::vector<double> &per_participant_elapsed = pool.RunRound(iterations);
    // Each participant chases its own independent, disjoint permutation, so
    // the per-load latency for this round is the average of every
    // participant's own elapsed time divided by its own iteration count,
    // not the outer wall-clock span (which would conflate independent
    // chains with scheduling jitter across participants).
    double total_ns_per_load = 0.0;
    for (double elapsed_seconds : per_participant_elapsed) {
      const double safe_elapsed = elapsed_seconds > 0.0 ? elapsed_seconds : 1e-12;
      total_ns_per_load += safe_elapsed * 1.0e9 / static_cast<double>(iterations);
    }
    result.raw_ns_per_load_samples.push_back(total_ns_per_load /
                                             static_cast<double>(participant_count));
  }

  result.median_ns_per_load = Median(result.raw_ns_per_load_samples);
  result.dispersion_ns_per_load = InterquartileRange(result.raw_ns_per_load_samples);
  result.available = true;
  return result;
}

} // namespace onnx_light_cpu
