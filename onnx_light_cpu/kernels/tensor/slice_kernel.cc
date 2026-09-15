// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/tensor/slice_kernel.h"

#include "onnx_light_cpu/impl/tensor/slice_kernel.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_lib/common/safe_math.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Shape;
using rt_ns::Tensor;

namespace {

static_assert(Shape::kMaxRank <= SliceCopyPlan::kMaxRank);

[[noreturn]] void Invalid(const char *message) {
  throw std::invalid_argument(std::string("onnx_light_cpu::Slice: ") + message);
}

int64_t Multiply(int64_t left, int64_t right) {
  int64_t result;
  if (ONNX_LIGHT_NAMESPACE::checked_mul_overflow(left, right, &result)) {
    Invalid("shape or byte arithmetic overflow.");
  }
  return result;
}

int64_t Add(int64_t left, int64_t right) {
  int64_t result;
  if (ONNX_LIGHT_NAMESPACE::checked_add_overflow(left, right, &result)) {
    Invalid("byte offset overflow.");
  }
  return result;
}

std::size_t Size(int64_t value) { return ONNX_LIGHT_NAMESPACE::safe_cast_to_size(value, Invalid); }

int64_t Count(const Shape &shape) {
  bool empty = false;
  for (int64_t dim : shape) {
    if (dim < 0) {
      Invalid("negative tensor dimension.");
    }
    empty |= dim == 0;
  }
  return empty ? 0 : shape.product(0, shape.size(), "Slice");
}

void ValidateBuffer(const Tensor &tensor, std::size_t bytes) {
  if (tensor.size_bytes() != bytes || (bytes != 0 && tensor.bytes() == nullptr)) {
    Invalid("tensor buffer size does not match shape.");
  }
}

Shape ReadIndices(const Tensor &tensor) {
  if (tensor.shape.size() != 1 ||
      (tensor.data_type != DataType::INT32 && tensor.data_type != DataType::INT64)) {
    Invalid("starts, ends, axes and steps must be 1-D INT32 or INT64 tensors.");
  }
  const int64_t count = Count(tensor.shape);
  if (count > static_cast<int64_t>(Shape::kMaxRank)) {
    Invalid("too many slice axes.");
  }
  ValidateBuffer(tensor, Size(Multiply(count, static_cast<int64_t>(tensor.element_size()))));
  Shape values;
  for (int64_t i = 0; i < count; ++i) {
    int64_t value;
    if (tensor.data_type == DataType::INT32) {
      int32_t item;
      std::memcpy(&item, tensor.bytes() + static_cast<std::size_t>(i) * sizeof(item), sizeof(item));
      value = item;
    } else {
      std::memcpy(&value, tensor.bytes() + static_cast<std::size_t>(i) * sizeof(value),
                  sizeof(value));
    }
    values.push_back(value);
  }
  return values;
}

int64_t Normalize(int64_t dim, int64_t &start, int64_t end, int64_t step) {
  if (step == 0) {
    Invalid("step cannot be zero.");
  }
  if (dim == 0) {
    start = 0;
    return 0;
  }
  if (start < 0) {
    start += dim;
  }
  if (end < 0) {
    end += dim;
  }
  if (step > 0) {
    start = std::clamp(start, int64_t{0}, dim);
    end = std::clamp(end, int64_t{0}, dim);
    return end <= start ? 0 : 1 + (end - start - 1) / step;
  }
  start = std::clamp(start, int64_t{0}, dim - 1);
  end = std::clamp(end, int64_t{-1}, dim - 1);
  if (end >= start) {
    return 0;
  }
  // Unsigned magnitude also represents abs(INT64_MIN); the builtin's signed negation cannot.
  const uint64_t magnitude = uint64_t{0} - static_cast<uint64_t>(step);
  return 1 + static_cast<int64_t>(static_cast<uint64_t>(start - end - 1) / magnitude);
}

struct SlicePlan {
  Shape shape;
  SliceCopyPlan copy;
  std::size_t bytes = 0;
};

SlicePlan Prepare(const Tensor &data, const Shape &starts, const Shape &ends, const Shape *axes,
                  const Shape *steps) {
  if (starts.size() != ends.size() || (axes && axes->size() != starts.size()) ||
      (steps && steps->size() != starts.size())) {
    Invalid("starts, ends, axes and steps lengths must match.");
  }
  const std::size_t rank = data.shape.size();
  if (starts.size() > rank) {
    Invalid("too many slice axes for data rank.");
  }
  const auto width = static_cast<int64_t>(data.element_size());
  ValidateBuffer(data, Size(Multiply(Count(data.shape), width)));
  SlicePlan plan;
  plan.shape = data.shape;
  std::array<int64_t, SliceCopyPlan::kMaxRank> first{};
  std::array<int64_t, SliceCopyPlan::kMaxRank> step{};
  step.fill(1);
  std::array<bool, SliceCopyPlan::kMaxRank> seen{};
  for (std::size_t i = 0; i < starts.size(); ++i) {
    int64_t axis = axes ? (*axes)[i] : static_cast<int64_t>(i);
    const int64_t signed_rank = static_cast<int64_t>(rank);
    if (axis < -signed_rank || axis >= signed_rank) {
      Invalid("axis out of range.");
    }
    if (axis < 0) {
      axis += signed_rank;
    }
    const auto a = static_cast<std::size_t>(axis);
    if (seen[a]) {
      Invalid("duplicate slice axis.");
    }
    seen[a] = true;
    first[a] = starts[i];
    step[a] = steps ? (*steps)[i] : 1;
    plan.shape[a] = Normalize(data.shape[a], first[a], ends[i], step[a]);
  }
  const int64_t output_count = Count(plan.shape);
  plan.bytes = Size(Multiply(output_count, width));
  if (output_count == 0) {
    return plan;
  }
  auto &copy = plan.copy;
  copy.rank = rank;
  copy.element_bytes = Size(width);
  int64_t stride = width;
  for (std::size_t i = rank; i-- > 0;) {
    copy.dimensions[i] = plan.shape[i];
    copy.source_offset = Add(copy.source_offset, Multiply(first[i], stride));
    // Singleton dimensions never advance: their arbitrary step need not fit a byte stride.
    copy.strides[i] = plan.shape[i] > 1 ? Multiply(step[i], stride) : 0;
    copy.rewinds[i] = Multiply(plan.shape[i] - 1, copy.strides[i]);
    stride = Multiply(stride, data.shape[i]);
  }
  copy.chunk_bytes = copy.element_bytes;
  while (copy.rank > 0) {
    const std::size_t i = copy.rank - 1;
    if (plan.shape[i] > 1 && copy.strides[i] != static_cast<int64_t>(copy.chunk_bytes)) {
      break;
    }
    copy.chunk_bytes = Size(Multiply(static_cast<int64_t>(copy.chunk_bytes), plan.shape[i]));
    --copy.rank;
  }
  copy.chunks = static_cast<int64_t>(plan.bytes / copy.chunk_bytes);
  return plan;
}

SlicePlan Prepare(const Tensor &data, const Tensor &starts, const Tensor &ends, const Tensor *axes,
                  const Tensor *steps) {
  const Shape start_values = ReadIndices(starts);
  const Shape end_values = ReadIndices(ends);
  const Shape axis_values = axes ? ReadIndices(*axes) : Shape{};
  const Shape step_values = steps ? ReadIndices(*steps) : Shape{};
  return Prepare(data, start_values, end_values, axes ? &axis_values : nullptr,
                 steps ? &step_values : nullptr);
}

void RejectOverlap(const Tensor &input, const Tensor &output) {
  if (input.size_bytes() == 0 || output.size_bytes() == 0) {
    return;
  }
  const auto source = reinterpret_cast<std::uintptr_t>(input.bytes());
  const auto destination = reinterpret_cast<std::uintptr_t>(output.bytes());
  if (source <= destination ? destination - source < input.size_bytes()
                            : source - destination < output.size_bytes()) {
    Invalid("output must not overlap any input.");
  }
}

void Copy(const Tensor &data, Tensor &output, const SlicePlan &plan) {
  if (output.data_type != data.data_type || output.shape != plan.shape) {
    Invalid("output dtype or shape mismatch.");
  }
  ValidateBuffer(output, plan.bytes);
  RejectOverlap(data, output);
  SliceCopy(data.bytes(), output.mutable_bytes(), plan.copy);
}

Tensor Allocate(const Tensor &data, const SlicePlan &plan, rt_ns::RuntimeContext *rt,
                rt_ns::RawBufferAllocator *allocator) {
  return rt ? rt->MakeOutputTensor(0, data.data_type, plan.shape, plan.bytes)
            : rt_ns::MakeOutputTensor(data.data_type, plan.shape, plan.bytes, allocator);
}

} // namespace

SliceKernel::SliceKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                         const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
}

Tensor SliceKernel::operator()(const Tensor &data, const Tensor &starts, const Tensor &ends,
                               const Tensor *axes, const Tensor *steps,
                               rt_ns::RuntimeContext *rt) const {
  const SlicePlan plan = Prepare(data, starts, ends, axes, steps);
  Tensor output = Allocate(data, plan, rt, ctx_.allocator);
  Copy(data, output, plan);
  return output;
}

void SliceKernel::operator()(const Tensor &data, const Tensor &starts, const Tensor &ends,
                             const Tensor *axes, const Tensor *steps, Tensor &output) const {
  const SlicePlan plan = Prepare(data, starts, ends, axes, steps);
  for (const Tensor *input : {&starts, &ends, axes, steps}) {
    if (input) {
      RejectOverlap(*input, output);
    }
  }
  Copy(data, output, plan);
}

void SliceKernel::Run(rt_ns::RuntimeContext &rt) {
  rt.RecordKernelUsage(kName);
  const auto &node = *node_;
  rt_ns::RequireOutputCount(node, 1);
  if (ctx_.opset.version > 0 && ctx_.opset.version < 10) {
    rt_ns::RequireInputCount(node, 1);
    if (!rt_ns::FindAttribute(node, "starts") || !rt_ns::FindAttribute(node, "ends")) {
      Invalid("legacy Slice requires starts and ends attributes.");
    }
    const Shape starts = rt_ns::GetAttributeShapeOrDefault(node, "starts", {});
    const Shape ends = rt_ns::GetAttributeShapeOrDefault(node, "ends", {});
    const Shape axes = rt_ns::GetAttributeShapeOrDefault(node, "axes", {});
    const Tensor &data = rt_ns::GetInput(node, 0, rt.tensors());
    const SlicePlan plan =
        Prepare(data, starts, ends, rt_ns::FindAttribute(node, "axes") ? &axes : nullptr, nullptr);
    Tensor output = Allocate(data, plan, &rt, ctx_.allocator);
    Copy(data, output, plan);
    rt_ns::SetOutput(node, 0, std::move(output), rt);
    return;
  }
  rt_ns::RequireMinInputCount(node, 3);
  if (node.input_size() > 5) {
    Invalid("expected between three and five inputs.");
  }
  const Tensor &data = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &starts = rt_ns::GetInput(node, 1, rt.tensors());
  const Tensor &ends = rt_ns::GetInput(node, 2, rt.tensors());
  const Tensor *axes = rt_ns::GetOptionalInput(node, 3, rt.tensors());
  const Tensor *steps = rt_ns::GetOptionalInput(node, 4, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(data, starts, ends, axes, steps, &rt), rt);
}

void RegisterSliceKernel() {
  rt_ns::NodeKernelFn factory =
      [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
         rt_ns::RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<SliceKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Slice";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = SliceKernel::kName;
  info.types = {DataType::FLOAT,      DataType::DOUBLE,         DataType::FLOAT16,
                DataType::BFLOAT16,   DataType::INT8,           DataType::UINT8,
                DataType::INT16,      DataType::UINT16,         DataType::INT32,
                DataType::UINT32,     DataType::INT64,          DataType::UINT64,
                DataType::BOOL,       DataType::FLOAT8E4M3FN,   DataType::FLOAT8E4M3FNUZ,
                DataType::FLOAT8E5M2, DataType::FLOAT8E5M2FNUZ, DataType::FLOAT8E8M0};
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
