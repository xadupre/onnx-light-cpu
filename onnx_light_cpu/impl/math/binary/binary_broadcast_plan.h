// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_cpu/impl/math/binary/binary_execution_schedule.h"
#include "onnx_light_cpu/impl/math/binary/binary_kernel_descriptor.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <span>
#include <vector>

namespace onnx_light_cpu {

class BinaryBroadcastPlan {
public:
  enum class LoopFamily : std::uint8_t {
    kContiguous,
    kLeftScalar,
    kRightScalar,
    kRepeatedContiguousBlock,
    kInnerVectorBroadcast,
    kOuterBroadcast,
    kGeneralStrided,
  };

  struct Dimension {
    std::size_t extent = 1;
    std::ptrdiff_t left_stride = 0;
    std::ptrdiff_t right_stride = 0;
    std::ptrdiff_t output_stride = 0;
  };

  BinaryBroadcastPlan(const BinaryKernelDescriptor &descriptor, BinaryDataType left_type,
                      BinaryDataType right_type, BinaryDataType output_type,
                      std::span<const std::int64_t> left_shape,
                      std::span<const std::int64_t> right_shape);

  const BinaryKernelDescriptor &descriptor() const noexcept { return descriptor_; }
  const BinaryKernelDescriptor::Adapter &adapter() const noexcept { return adapter_; }
  std::span<const std::int64_t> output_shape() const noexcept { return output_shape_; }
  std::span<const Dimension> dimensions() const noexcept { return dimensions_; }
  LoopFamily loop_family() const noexcept { return loop_family_; }
  std::size_t inner_loop_elements() const noexcept { return inner_loop_elements_; }
  std::size_t outer_block_count() const noexcept { return outer_block_count_; }
  std::size_t prepared_outer_rank() const noexcept { return prepared_outer_rank_; }
  bool left_output_alias_safe() const noexcept { return left_output_alias_safe_; }
  bool right_output_alias_safe() const noexcept { return right_output_alias_safe_; }

  void Execute(const void *left, const void *right, void *output) const;
  void Execute(const void *left, const void *right, void *output,
               const BinaryExecutionTuning &tuning) const;

private:
  void ClassifyLoopFamily();
  void ValidateInputs(const std::byte *left, const std::byte *right) const;
  void ExecuteInner(const std::byte *left, const std::byte *right, std::byte *output,
                    std::ptrdiff_t left_offset, std::ptrdiff_t right_offset,
                    std::ptrdiff_t output_offset) const;
  // Binary PR03: single-dimension (kContiguous/kLeftScalar/kRightScalar) and
  // multi-dimensional (repeated block/inner-vector/outer/general-strided)
  // dispatch, each submitting independent, per-invocation work to the
  // session executor via ``ExecuteRanges`` instead of a private scheduler.
  void ExecuteFlat(const std::byte *left, const std::byte *right, std::byte *output,
                   const BinaryExecutionTuning &tuning) const;
  void ExecuteMultiDimensional(const std::byte *left, const std::byte *right, std::byte *output,
                               const BinaryExecutionTuning &tuning) const;
  void ExecuteOuterRange(const std::byte *left, const std::byte *right, std::byte *output,
                         std::size_t outer_begin, std::size_t outer_end) const;
  void ExecuteInnerBlock(const std::byte *left, const std::byte *right, std::byte *output,
                         std::ptrdiff_t left_offset, std::ptrdiff_t right_offset,
                         std::ptrdiff_t output_offset) const;
  template <std::size_t OuterRank>
  void ExecuteOuterRangeFixed(const std::byte *left, const std::byte *right, std::byte *output,
                              std::size_t outer_begin, std::size_t outer_end) const;
  void ComputeOuterOffsets(std::size_t outer_index, std::vector<std::size_t> &indices,
                           std::ptrdiff_t &left_offset, std::ptrdiff_t &right_offset,
                           std::ptrdiff_t &output_offset) const;
  void AdvanceOuterIndices(std::vector<std::size_t> &indices, std::ptrdiff_t &left_offset,
                           std::ptrdiff_t &right_offset, std::ptrdiff_t &output_offset) const;

  const BinaryKernelDescriptor &descriptor_;
  const BinaryKernelDescriptor::Adapter &adapter_;
  std::vector<std::int64_t> output_shape_;
  std::vector<Dimension> dimensions_;
  std::size_t element_count_ = 0;
  std::size_t inner_loop_elements_ = 0;
  std::size_t outer_block_count_ = 0;
  std::size_t prepared_outer_rank_ = 0;
  LoopFamily loop_family_ = LoopFamily::kContiguous;
  bool left_output_alias_safe_ = false;
  bool right_output_alias_safe_ = false;
};

class BinaryBroadcastPlanCache {
public:
  using PlanPtr = std::shared_ptr<const BinaryBroadcastPlan>;

  PlanPtr GetOrCreate(const BinaryKernelDescriptor &descriptor, BinaryDataType left_type,
                      BinaryDataType right_type, BinaryDataType output_type,
                      std::span<const std::int64_t> left_shape,
                      std::span<const std::int64_t> right_shape);

  std::size_t size() const noexcept { return entries_.size(); }

private:
  struct Key {
    std::uint64_t descriptor_identity = 0;
    BinaryDataType left_type = BinaryDataType::UNDEFINED;
    BinaryDataType right_type = BinaryDataType::UNDEFINED;
    BinaryDataType output_type = BinaryDataType::UNDEFINED;
    std::vector<std::int64_t> left_shape;
    std::vector<std::int64_t> right_shape;

    bool operator==(const Key &other) const noexcept;
  };

  struct Entry {
    Key key;
    PlanPtr plan;
  };

  static constexpr std::size_t kCapacity = 8;
  std::list<Entry> entries_;
};

const char *ToString(BinaryBroadcastPlan::LoopFamily family) noexcept;

} // namespace onnx_light_cpu
