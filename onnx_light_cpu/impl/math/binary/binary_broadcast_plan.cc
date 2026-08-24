// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_broadcast_plan.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/binary/binary_execution_schedule.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnx_light_cpu {
namespace {

std::vector<std::int64_t> AlignShape(std::span<const std::int64_t> shape, std::size_t rank) {
  std::vector<std::int64_t> aligned(rank, 1);
  for (std::size_t i = 0; i < shape.size(); ++i) {
    aligned[rank - shape.size() + i] = shape[i];
  }
  return aligned;
}

std::vector<std::int64_t> BroadcastShape(std::string_view op_name,
                                         std::span<const std::int64_t> left,
                                         std::span<const std::int64_t> right) {
  const std::size_t rank = std::max(left.size(), right.size());
  const std::vector<std::int64_t> aligned_left = AlignShape(left, rank);
  const std::vector<std::int64_t> aligned_right = AlignShape(right, rank);
  std::vector<std::int64_t> output(rank, 1);
  for (std::size_t i = 0; i < rank; ++i) {
    if (aligned_left[i] == aligned_right[i]) {
      output[i] = aligned_left[i];
    } else if (aligned_left[i] == 1) {
      output[i] = aligned_right[i];
    } else if (aligned_right[i] == 1) {
      output[i] = aligned_left[i];
    } else {
      throw std::invalid_argument("onnx_light_cpu::" + std::string(op_name) +
                                  ": input shapes are not multidirectional-broadcastable.");
    }
  }
  return output;
}

std::vector<std::ptrdiff_t> ComputeStrides(const std::vector<std::int64_t> &aligned_shape,
                                           const std::vector<std::int64_t> &output_shape) {
  std::vector<std::ptrdiff_t> strides(aligned_shape.size(), 0);
  std::ptrdiff_t stride = 1;
  for (std::size_t i = aligned_shape.size(); i-- > 0;) {
    strides[i] = aligned_shape[i] == 1 && output_shape[i] != 1 ? 0 : stride;
    stride *= static_cast<std::ptrdiff_t>(aligned_shape[i]);
  }
  return strides;
}

bool CanMerge(const BinaryBroadcastPlan::Dimension &outer,
              const BinaryBroadcastPlan::Dimension &inner) {
  const auto mergeable = [&](std::ptrdiff_t outer_stride, std::ptrdiff_t inner_stride) {
    if (outer_stride == 0 && inner_stride == 0) {
      return true;
    }
    return outer_stride == static_cast<std::ptrdiff_t>(inner.extent) * inner_stride;
  };
  return mergeable(outer.left_stride, inner.left_stride) &&
         mergeable(outer.right_stride, inner.right_stride) &&
         outer.output_stride == static_cast<std::ptrdiff_t>(inner.extent) * inner.output_stride;
}

} // namespace

BinaryBroadcastPlan::BinaryBroadcastPlan(const BinaryKernelDescriptor &descriptor,
                                         BinaryDataType left_type, BinaryDataType right_type,
                                         BinaryDataType output_type,
                                         std::span<const std::int64_t> left_shape,
                                         std::span<const std::int64_t> right_shape)
    : descriptor_(descriptor),
      adapter_(descriptor.ResolveAdapter(left_type, right_type, output_type)) {
  output_shape_ = BroadcastShape(descriptor.op_type(), left_shape, right_shape);
  const std::size_t rank = output_shape_.size();
  const std::vector<std::int64_t> aligned_left = AlignShape(left_shape, rank);
  const std::vector<std::int64_t> aligned_right = AlignShape(right_shape, rank);
  const std::vector<std::ptrdiff_t> left_strides = ComputeStrides(aligned_left, output_shape_);
  const std::vector<std::ptrdiff_t> right_strides = ComputeStrides(aligned_right, output_shape_);

  std::ptrdiff_t output_stride = 1;
  for (std::size_t i = rank; i-- > 0;) {
    if (output_shape_[i] != 1) {
      dimensions_.push_back(Dimension{static_cast<std::size_t>(output_shape_[i]), left_strides[i],
                                      right_strides[i], output_stride});
    }
    output_stride *= output_shape_[i];
  }
  std::reverse(dimensions_.begin(), dimensions_.end());
  std::vector<Dimension> coalesced;
  for (const Dimension &dimension : dimensions_) {
    if (!coalesced.empty() && CanMerge(coalesced.back(), dimension)) {
      coalesced.back().extent *= dimension.extent;
      coalesced.back().left_stride = dimension.left_stride;
      coalesced.back().right_stride = dimension.right_stride;
      coalesced.back().output_stride = dimension.output_stride;
    } else {
      coalesced.push_back(dimension);
    }
  }
  dimensions_ = std::move(coalesced);

  element_count_ = 1;
  for (std::int64_t dim : output_shape_) {
    element_count_ *= static_cast<std::size_t>(std::max<std::int64_t>(dim, 0));
  }
  if (element_count_ == 0) {
    inner_loop_elements_ = 0;
    outer_block_count_ = 0;
    return;
  }
  if (dimensions_.empty()) {
    dimensions_.push_back(Dimension{1, 0, 0, 1});
  }
  inner_loop_elements_ = dimensions_.back().extent;
  outer_block_count_ = 1;
  for (std::size_t i = 0; i + 1 < dimensions_.size(); ++i) {
    outer_block_count_ *= dimensions_[i].extent;
  }
  left_output_alias_safe_ =
      std::vector<std::int64_t>(left_shape.begin(), left_shape.end()) == output_shape_ &&
      left_type == output_type;
  right_output_alias_safe_ =
      std::vector<std::int64_t>(right_shape.begin(), right_shape.end()) == output_shape_ &&
      right_type == output_type;
  for (const Dimension &dimension : dimensions_) {
    if (dimension.left_stride == 0) {
      left_output_alias_safe_ = false;
    }
    if (dimension.right_stride == 0) {
      right_output_alias_safe_ = false;
    }
  }
  ClassifyLoopFamily();
}

void BinaryBroadcastPlan::ClassifyLoopFamily() {
  if (element_count_ == 0) {
    loop_family_ = LoopFamily::kContiguous;
    return;
  }
  if (dimensions_.size() == 1) {
    if (dimensions_[0].left_stride == 0 && dimensions_[0].right_stride == 1) {
      loop_family_ = LoopFamily::kLeftScalar;
      return;
    }
    if (dimensions_[0].right_stride == 0 && dimensions_[0].left_stride == 1) {
      loop_family_ = LoopFamily::kRightScalar;
      return;
    }
    loop_family_ = LoopFamily::kContiguous;
    return;
  }
  const Dimension &inner = dimensions_.back();
  bool has_outer_broadcast = false;
  int left_outer_nonzero = 0;
  int right_outer_nonzero = 0;
  for (std::size_t i = 0; i + 1 < dimensions_.size(); ++i) {
    has_outer_broadcast =
        has_outer_broadcast || dimensions_[i].left_stride == 0 || dimensions_[i].right_stride == 0;
    left_outer_nonzero += dimensions_[i].left_stride != 0 ? 1 : 0;
    right_outer_nonzero += dimensions_[i].right_stride != 0 ? 1 : 0;
  }
  if (inner.extent > 1 &&
      ((inner.left_stride == 1 && inner.right_stride == 0 && right_outer_nonzero == 1) ||
       (inner.left_stride == 0 && inner.right_stride == 1 && left_outer_nonzero == 1))) {
    loop_family_ = LoopFamily::kRepeatedContiguousBlock;
  } else if (inner.extent > 1 && inner.left_stride == 1 && inner.right_stride == 1 &&
             has_outer_broadcast) {
    loop_family_ = LoopFamily::kOuterBroadcast;
  } else if (inner.extent > 1 && ((inner.left_stride == 1 && has_outer_broadcast) ||
                                  (inner.right_stride == 1 && has_outer_broadcast))) {
    loop_family_ = LoopFamily::kInnerVectorBroadcast;
  } else {
    loop_family_ = LoopFamily::kGeneralStrided;
  }
}

void BinaryBroadcastPlan::ValidateInputs(const std::byte *left, const std::byte *right) const {
  if (adapter_.validate == nullptr) {
    return;
  }
  const std::size_t outer_dims = dimensions_.size() - 1;
  std::vector<std::size_t> indices(outer_dims, 0);
  std::ptrdiff_t left_offset = 0;
  std::ptrdiff_t right_offset = 0;
  std::ptrdiff_t output_offset = 0;
  ComputeOuterOffsets(0, indices, left_offset, right_offset, output_offset);
  const Dimension &inner = dimensions_.back();
  for (std::size_t block = 0; block < outer_block_count_; ++block) {
    for (std::size_t i = 0; i < inner_loop_elements_; ++i) {
      adapter_.validate(left + (left_offset + static_cast<std::ptrdiff_t>(i) * inner.left_stride) *
                                   adapter_.left_size,
                        right +
                            (right_offset + static_cast<std::ptrdiff_t>(i) * inner.right_stride) *
                                adapter_.right_size);
    }
    if (block + 1 != outer_block_count_) {
      AdvanceOuterIndices(indices, left_offset, right_offset, output_offset);
    }
  }
}

void BinaryBroadcastPlan::ExecuteInner(const std::byte *left, const std::byte *right,
                                       std::byte *output, std::ptrdiff_t left_offset,
                                       std::ptrdiff_t right_offset,
                                       std::ptrdiff_t output_offset) const {
  const std::ptrdiff_t left_stride = dimensions_.back().left_stride;
  const std::ptrdiff_t right_stride = dimensions_.back().right_stride;
  for (std::size_t i = 0; i < inner_loop_elements_; ++i) {
    adapter_.scalar(
        left + (left_offset + static_cast<std::ptrdiff_t>(i) * left_stride) * adapter_.left_size,
        right +
            (right_offset + static_cast<std::ptrdiff_t>(i) * right_stride) * adapter_.right_size,
        output + (output_offset + static_cast<std::ptrdiff_t>(i)) * adapter_.output_size);
  }
}

namespace {

// The general N-D fallback loop advances offsets by addition/subtraction only
// (see ``ClassifyLoopFamily``/generic traversal below); no offset division or
// modulo ever appears inside a per-element hot loop. Division/modulo is only
// used once per parallel block, to seed a block's starting offsets, which is
// outside the hot loop by construction.
std::size_t ByteThresholdToUnits(std::size_t threshold_bytes, std::size_t bytes_per_unit) {
  return bytes_per_unit == 0 ? 0 : (threshold_bytes + bytes_per_unit - 1) / bytes_per_unit;
}

} // namespace

void BinaryBroadcastPlan::ExecuteFlat(const std::byte *left, const std::byte *right,
                                      std::byte *output) const {
  const std::size_t count = dimensions_[0].extent;
  const std::ptrdiff_t left_stride = dimensions_[0].left_stride;
  const std::ptrdiff_t right_stride = dimensions_[0].right_stride;
  BinaryKernelDescriptor::Adapter::BulkContiguousFn bulk = nullptr;
  if (loop_family_ == LoopFamily::kContiguous) {
    bulk = adapter_.bulk_contiguous;
  } else if (loop_family_ == LoopFamily::kLeftScalar) {
    bulk = adapter_.bulk_left_scalar;
  } else if (loop_family_ == LoopFamily::kRightScalar) {
    bulk = adapter_.bulk_right_scalar;
  }

  const std::size_t bytes_per_unit = (left_stride != 0 ? adapter_.left_size : 0) +
                                     (right_stride != 0 ? adapter_.right_size : 0) +
                                     adapter_.output_size;
  const std::size_t threshold_bytes =
      bulk != nullptr ? kBinaryBulkParallelThresholdBytes : kBinaryScalarParallelThresholdBytes;
  const std::size_t min_units = ByteThresholdToUnits(threshold_bytes, bytes_per_unit);
  const std::int64_t max_participants =
      bulk != nullptr ? kBinaryBulkMaxParticipants : kBinaryUnboundedParticipants;
  const std::size_t element_size = std::max<std::size_t>(adapter_.output_size, 1);
  const std::int64_t block_multiple =
      std::max<std::int64_t>(static_cast<std::int64_t>(kExecutionSimdWidthBytes / element_size), 1);

  // ``min_block_size`` is deliberately a fraction of ``min_parallel_size``:
  // ``ExecuteRanges`` only submits work to the executor once it can form at
  // least two blocks, so keeping them equal would silently double the
  // effective crossover threshold.
  const std::size_t min_units_bound = std::max<std::size_t>(min_units, 1);
  const std::int64_t block_divisor =
      max_participants == kBinaryUnboundedParticipants ? 4 : max_participants;
  const std::int64_t min_block_size =
      std::max<std::int64_t>(static_cast<std::int64_t>(min_units_bound) / block_divisor, 1);
  const ExecutionSchedule schedule{static_cast<std::int64_t>(min_units_bound), min_block_size,
                                   max_participants};
  ExecuteRanges(static_cast<std::int64_t>(count), schedule, block_multiple,
                [&](std::int64_t begin, std::int64_t end) {
                  const std::size_t sub_count = static_cast<std::size_t>(end - begin);
                  const std::byte *sub_left =
                      left + (left_stride != 0 ? begin : 0) * adapter_.left_size;
                  const std::byte *sub_right =
                      right + (right_stride != 0 ? begin : 0) * adapter_.right_size;
                  std::byte *sub_output = output + begin * adapter_.output_size;
                  if (bulk != nullptr) {
                    bulk(sub_left, sub_right, sub_output, sub_count);
                    return;
                  }
                  for (std::size_t i = 0; i < sub_count; ++i) {
                    adapter_.scalar(sub_left + (left_stride != 0 ? i : 0) * adapter_.left_size,
                                    sub_right + (right_stride != 0 ? i : 0) * adapter_.right_size,
                                    sub_output + i * adapter_.output_size);
                  }
                });
}

void BinaryBroadcastPlan::ComputeOuterOffsets(std::size_t outer_index,
                                              std::vector<std::size_t> &indices,
                                              std::ptrdiff_t &left_offset,
                                              std::ptrdiff_t &right_offset,
                                              std::ptrdiff_t &output_offset) const {
  left_offset = 0;
  right_offset = 0;
  output_offset = 0;
  std::size_t remaining = outer_index;
  for (std::size_t d = indices.size(); d-- > 0;) {
    const Dimension &dimension = dimensions_[d];
    const std::size_t idx = remaining % dimension.extent;
    remaining /= dimension.extent;
    indices[d] = idx;
    left_offset += static_cast<std::ptrdiff_t>(idx) * dimension.left_stride;
    right_offset += static_cast<std::ptrdiff_t>(idx) * dimension.right_stride;
    output_offset += static_cast<std::ptrdiff_t>(idx) * dimension.output_stride;
  }
}

void BinaryBroadcastPlan::AdvanceOuterIndices(std::vector<std::size_t> &indices,
                                              std::ptrdiff_t &left_offset,
                                              std::ptrdiff_t &right_offset,
                                              std::ptrdiff_t &output_offset) const {
  for (std::size_t d = indices.size(); d-- > 0;) {
    const Dimension &dimension = dimensions_[d];
    ++indices[d];
    left_offset += dimension.left_stride;
    right_offset += dimension.right_stride;
    output_offset += dimension.output_stride;
    if (indices[d] < dimension.extent) {
      return;
    }
    left_offset -= static_cast<std::ptrdiff_t>(indices[d]) * dimension.left_stride;
    right_offset -= static_cast<std::ptrdiff_t>(indices[d]) * dimension.right_stride;
    output_offset -= static_cast<std::ptrdiff_t>(indices[d]) * dimension.output_stride;
    indices[d] = 0;
  }
}

void BinaryBroadcastPlan::ExecuteOuterRange(const std::byte *left, const std::byte *right,
                                            std::byte *output, std::size_t outer_begin,
                                            std::size_t outer_end) const {
  if (outer_begin >= outer_end) {
    return;
  }
  const std::size_t outer_dims = dimensions_.size() - 1;
  std::vector<std::size_t> indices(outer_dims, 0);
  std::ptrdiff_t left_offset = 0;
  std::ptrdiff_t right_offset = 0;
  std::ptrdiff_t output_offset = 0;
  ComputeOuterOffsets(outer_begin, indices, left_offset, right_offset, output_offset);
  const Dimension &inner = dimensions_.back();
  for (std::size_t block = outer_begin; block < outer_end; ++block) {
    // Vectorize the inner extent whenever the plan's inner dimension leaves
    // one side contiguous (repeated block / inner-vector broadcast), both
    // sides contiguous (outer broadcast), or by falling back to the
    // per-element scalar loop (general strided).
    if (inner.left_stride == 0 && inner.right_stride == 1 && adapter_.bulk_left_scalar != nullptr) {
      adapter_.bulk_left_scalar(
          left + left_offset * adapter_.left_size, right + right_offset * adapter_.right_size,
          output + output_offset * adapter_.output_size, inner_loop_elements_);
    } else if (inner.right_stride == 0 && inner.left_stride == 1 &&
               adapter_.bulk_right_scalar != nullptr) {
      adapter_.bulk_right_scalar(
          left + left_offset * adapter_.left_size, right + right_offset * adapter_.right_size,
          output + output_offset * adapter_.output_size, inner_loop_elements_);
    } else if (inner.left_stride == 1 && inner.right_stride == 1 &&
               adapter_.bulk_contiguous != nullptr) {
      adapter_.bulk_contiguous(left + left_offset * adapter_.left_size,
                               right + right_offset * adapter_.right_size,
                               output + output_offset * adapter_.output_size, inner_loop_elements_);
    } else {
      ExecuteInner(left, right, output, left_offset, right_offset, output_offset);
    }
    if (block + 1 == outer_end) {
      break;
    }
    AdvanceOuterIndices(indices, left_offset, right_offset, output_offset);
  }
}

void BinaryBroadcastPlan::ExecuteMultiDimensional(const std::byte *left, const std::byte *right,
                                                  std::byte *output) const {
  const Dimension &inner = dimensions_.back();
  const bool has_bulk_inner =
      (inner.left_stride == 0 && inner.right_stride == 1 && adapter_.bulk_left_scalar != nullptr) ||
      (inner.right_stride == 0 && inner.left_stride == 1 &&
       adapter_.bulk_right_scalar != nullptr) ||
      (inner.left_stride == 1 && inner.right_stride == 1 && adapter_.bulk_contiguous != nullptr);

  const std::size_t bytes_per_block =
      inner_loop_elements_ *
      ((inner.left_stride != 0 ? adapter_.left_size : 0) +
       (inner.right_stride != 0 ? adapter_.right_size : 0) + adapter_.output_size);
  const std::size_t threshold_bytes =
      has_bulk_inner ? kBinaryBlockParallelThresholdBytes : kBinaryScalarParallelThresholdBytes;
  const std::size_t min_units = ByteThresholdToUnits(threshold_bytes, bytes_per_block);
  const std::int64_t max_participants =
      has_bulk_inner ? kBinaryBlockMaxParticipants : kBinaryUnboundedParticipants;

  const ExecutionSchedule schedule{static_cast<std::int64_t>(std::max<std::size_t>(min_units, 1)),
                                   1, max_participants};
  ExecuteRanges(static_cast<std::int64_t>(outer_block_count_), schedule,
                [&](std::int64_t begin, std::int64_t end) {
                  ExecuteOuterRange(left, right, output, static_cast<std::size_t>(begin),
                                    static_cast<std::size_t>(end));
                });
}

void BinaryBroadcastPlan::Execute(const void *left, const void *right, void *output) const {
  if (element_count_ == 0) {
    return;
  }
  const auto *left_bytes = reinterpret_cast<const std::byte *>(left);
  const auto *right_bytes = reinterpret_cast<const std::byte *>(right);
  auto *output_bytes = reinterpret_cast<std::byte *>(output);
  ValidateInputs(left_bytes, right_bytes);

  // Binary PR02/PR03: contiguous and left/right-scalar loops collapse to a
  // single dimension (see ClassifyLoopFamily). Every remaining family
  // (repeated block, inner-vector broadcast, outer broadcast, general
  // strided) vectorizes its inner extent whenever the plan's strides allow a
  // bulk SIMD kernel, and submits independent per-invocation work to the
  // existing session executor instead of looping element-by-element serially.
  if (dimensions_.size() == 1) {
    ExecuteFlat(left_bytes, right_bytes, output_bytes);
    return;
  }
  ExecuteMultiDimensional(left_bytes, right_bytes, output_bytes);
}

bool BinaryBroadcastPlanCache::Key::operator==(const Key &other) const noexcept {
  return descriptor_identity == other.descriptor_identity && left_type == other.left_type &&
         right_type == other.right_type && output_type == other.output_type &&
         left_shape == other.left_shape && right_shape == other.right_shape;
}

BinaryBroadcastPlanCache::PlanPtr BinaryBroadcastPlanCache::GetOrCreate(
    const BinaryKernelDescriptor &descriptor, BinaryDataType left_type, BinaryDataType right_type,
    BinaryDataType output_type, std::span<const std::int64_t> left_shape,
    std::span<const std::int64_t> right_shape) {
  Key key{descriptor.cache_identity(),
          left_type,
          right_type,
          output_type,
          std::vector<std::int64_t>(left_shape.begin(), left_shape.end()),
          std::vector<std::int64_t>(right_shape.begin(), right_shape.end())};
  auto it = std::find_if(entries_.begin(), entries_.end(),
                         [&](const Entry &entry) { return entry.key == key; });
  if (it != entries_.end()) {
    Entry hit = *it;
    entries_.erase(it);
    entries_.push_front(hit);
    return hit.plan;
  }
  PlanPtr plan = std::make_shared<BinaryBroadcastPlan>(descriptor, left_type, right_type,
                                                       output_type, left_shape, right_shape);
  entries_.push_front(Entry{std::move(key), plan});
  if (entries_.size() > kCapacity) {
    entries_.pop_back();
  }
  return plan;
}

const char *ToString(BinaryBroadcastPlan::LoopFamily family) noexcept {
  switch (family) {
  case BinaryBroadcastPlan::LoopFamily::kContiguous:
    return "contiguous";
  case BinaryBroadcastPlan::LoopFamily::kLeftScalar:
    return "left_scalar";
  case BinaryBroadcastPlan::LoopFamily::kRightScalar:
    return "right_scalar";
  case BinaryBroadcastPlan::LoopFamily::kRepeatedContiguousBlock:
    return "repeated_block";
  case BinaryBroadcastPlan::LoopFamily::kInnerVectorBroadcast:
    return "inner_vector_broadcast";
  case BinaryBroadcastPlan::LoopFamily::kOuterBroadcast:
    return "outer_broadcast";
  case BinaryBroadcastPlan::LoopFamily::kGeneralStrided:
    return "general_strided";
  }
  return "unknown";
}

} // namespace onnx_light_cpu
