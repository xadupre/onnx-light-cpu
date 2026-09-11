// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/tensor/gather_kernel.h"

#include "onnx_light_cpu/impl/tensor/gather_kernel.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_lib/common/safe_math.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Tensor;

namespace {

[[noreturn]] void Invalid(const char *message) {
  throw std::invalid_argument(std::string("onnx_light_cpu::Gather: ") + message);
}

std::size_t Bytes(int64_t count, std::size_t width) {
  int64_t bytes;
  if (ONNX_LIGHT_NAMESPACE::checked_mul_overflow(count, static_cast<int64_t>(width), &bytes)) {
    Invalid("byte size overflow.");
  }
  return ONNX_LIGHT_NAMESPACE::safe_cast_to_size(bytes, Invalid);
}

void ValidateBuffer(const Tensor &tensor, std::size_t bytes) {
  if (tensor.size_bytes() != bytes || (bytes != 0 && tensor.bytes() == nullptr)) {
    Invalid("tensor buffer size does not match shape.");
  }
}

struct GatherPlan {
  rt_ns::Shape shape;
  int64_t outer;
  int64_t axis_size;
  int64_t index_count;
  std::size_t slice_bytes;
  std::size_t output_bytes;
};

template <typename Index> void ValidateIndices(const Tensor &indices, const GatherPlan &plan) {
  const Index *values = indices.As<Index>();
  for (int64_t i = 0; i < plan.index_count; ++i) {
    const int64_t index = values[i];
    if (index < -plan.axis_size || index >= plan.axis_size) {
      Invalid("index out of range for axis dimension.");
    }
  }
}

GatherPlan Prepare(const Tensor &data, const Tensor &indices, int64_t axis) {
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  if (rank == 0 || axis < -rank || axis >= rank) {
    Invalid("data must have rank >= 1 and axis must be in [-rank, rank).");
  }
  if (axis < 0) {
    axis += rank;
  }
  if (indices.data_type != DataType::INT32 && indices.data_type != DataType::INT64) {
    Invalid("indices must be INT32 or INT64.");
  }
  // ElementSize deliberately preserves the builtin's rejection of STRING,
  // complex and packed sub-byte types rather than treating them as raw elements.
  const std::size_t width = data.element_size();
  const int64_t data_count = data.shape.product(0, data.shape.size(), "Gather data");
  const int64_t index_count = indices.shape.product(0, indices.shape.size(), "Gather indices");
  ValidateBuffer(data, Bytes(data_count, width));
  ValidateBuffer(indices, Bytes(index_count, indices.element_size()));
  GatherPlan plan;
  for (int64_t i = 0; i < axis; ++i) {
    plan.shape.push_back(data.shape[i]);
  }
  for (int64_t dim : indices.shape) {
    plan.shape.push_back(dim);
  }
  for (int64_t i = axis + 1; i < rank; ++i) {
    plan.shape.push_back(data.shape[i]);
  }
  const int64_t output_count = plan.shape.product(0, plan.shape.size(), "Gather output");
  plan.output_bytes = Bytes(output_count, width);
  plan.outer = data.shape.product(0, static_cast<std::size_t>(axis), "Gather outer");
  plan.axis_size = data.shape[axis];
  plan.index_count = index_count;
  const int64_t inner =
      data.shape.product(static_cast<std::size_t>(axis + 1), data.shape.size(), "Gather inner");
  plan.slice_bytes = Bytes(inner, width);
  int64_t slices;
  if (ONNX_LIGHT_NAMESPACE::checked_mul_overflow(plan.outer, index_count, &slices)) {
    Invalid("slice count overflow.");
  }
  if (indices.data_type == DataType::INT32) {
    ValidateIndices<int32_t>(indices, plan);
  } else {
    ValidateIndices<int64_t>(indices, plan);
  }
  return plan;
}

bool Overlaps(const Tensor &input, const Tensor &output) {
  if (input.size_bytes() == 0 || output.size_bytes() == 0) {
    return false;
  }
  const auto source = reinterpret_cast<std::uintptr_t>(input.bytes());
  const auto destination = reinterpret_cast<std::uintptr_t>(output.bytes());
  return source <= destination ? destination - source < input.size_bytes()
                               : source - destination < output.size_bytes();
}

void Copy(const Tensor &data, const Tensor &indices, Tensor &output, const GatherPlan &plan) {
  if (plan.output_bytes == 0) {
    return;
  }
  if (Overlaps(data, output) || Overlaps(indices, output)) {
    Invalid("output must not overlap data or indices.");
  }
  if (indices.data_type == DataType::INT32) {
    GatherSlices(data.bytes(), indices.As<int32_t>(), output.mutable_bytes(), plan.outer,
                 plan.axis_size, plan.index_count, plan.slice_bytes);
  } else {
    GatherSlices(data.bytes(), indices.As<int64_t>(), output.mutable_bytes(), plan.outer,
                 plan.axis_size, plan.index_count, plan.slice_bytes);
  }
}

} // namespace

GatherKernel::GatherKernel(const rt_ns::KernelContext &ctx, int64_t axis)
    : KernelBase(ctx), axis_(axis) {}

GatherKernel::GatherKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                           const rt_ns::KernelContext &ctx)
    : KernelBase(ctx), axis_(rt_ns::GetAttributeIntOrDefault(node, "axis", 0)) {
  set_node(node);
}

Tensor GatherKernel::operator()(const Tensor &data, const Tensor &indices,
                                rt_ns::RuntimeContext *rt) const {
  const GatherPlan plan = Prepare(data, indices, axis_);
  Tensor output =
      rt ? rt->MakeOutputTensor(0, data.data_type, plan.shape, plan.output_bytes)
         : rt_ns::MakeOutputTensor(data.data_type, plan.shape, plan.output_bytes, ctx_.allocator);
  Copy(data, indices, output, plan);
  return output;
}

void GatherKernel::operator()(const Tensor &data, const Tensor &indices, Tensor &output) const {
  const GatherPlan plan = Prepare(data, indices, axis_);
  if (output.data_type != data.data_type || output.shape != plan.shape) {
    Invalid("output dtype or shape mismatch.");
  }
  ValidateBuffer(output, plan.output_bytes);
  Copy(data, indices, output, plan);
}

void GatherKernel::Run(rt_ns::RuntimeContext &rt) {
  rt.RecordKernelUsage(kName);
  const auto &node = *node_;
  rt_ns::RequireInputCount(node, 2);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &data = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &indices = rt_ns::GetInput(node, 1, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(data, indices, &rt), rt);
}

void RegisterGatherKernel() {
  rt_ns::NodeKernelFn factory =
      [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
         rt_ns::RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<GatherKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Gather";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = GatherKernel::kName;
  info.types = {DataType::FLOAT,      DataType::DOUBLE,         DataType::FLOAT16,
                DataType::BFLOAT16,   DataType::INT8,           DataType::UINT8,
                DataType::INT16,      DataType::UINT16,         DataType::INT32,
                DataType::UINT32,     DataType::INT64,          DataType::UINT64,
                DataType::BOOL,       DataType::FLOAT8E4M3FN,   DataType::FLOAT8E4M3FNUZ,
                DataType::FLOAT8E5M2, DataType::FLOAT8E5M2FNUZ, DataType::FLOAT8E8M0};
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
