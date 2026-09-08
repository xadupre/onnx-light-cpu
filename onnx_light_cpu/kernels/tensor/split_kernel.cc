// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/tensor/split_kernel.h"

#include "onnx_light_cpu/impl/tensor/split_kernel.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_lib/common/safe_math.h"

#include <cstring>
#include <limits>
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

[[noreturn]] void Invalid(const char *message) {
  throw std::invalid_argument(std::string("onnx_light_cpu::Split: ") + message);
}

int64_t Multiply(int64_t left, int64_t right) {
  int64_t result;
  if (ONNX_LIGHT_NAMESPACE::checked_mul_overflow(left, right, &result)) {
    Invalid("shape or byte arithmetic overflow.");
  }
  return result;
}

std::size_t Size(int64_t value) { return ONNX_LIGHT_NAMESPACE::safe_cast_to_size(value, Invalid); }

int64_t Count(const Shape &shape, std::size_t begin, std::size_t end) {
  bool empty = false;
  for (std::size_t i = begin; i < end; ++i) {
    if (shape[i] < 0) {
      Invalid("negative tensor dimension.");
    }
    empty |= shape[i] == 0;
  }
  return empty ? 0 : shape.product(begin, end, "Split");
}

void ValidateBuffer(const Tensor &tensor, std::size_t bytes) {
  if (tensor.size_bytes() != bytes || (bytes != 0 && tensor.bytes() == nullptr)) {
    Invalid("tensor buffer size does not match shape.");
  }
}

std::size_t Axis(const Tensor &data, int64_t axis) {
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  if (rank == 0 || axis < -rank || axis >= rank) {
    Invalid("axis out of range or scalar input.");
  }
  return static_cast<std::size_t>(axis < 0 ? axis + rank : axis);
}

std::vector<int64_t> Resolve(int64_t dim, std::span<const int64_t> split, int64_t num_outputs) {
  if (num_outputs < 0 || (!split.empty() && num_outputs != 0)) {
    Invalid("split and num_outputs are mutually exclusive; num_outputs must be positive.");
  }
  if (!split.empty()) {
    if (split.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      Invalid("too many outputs.");
    }
    int64_t sum = 0;
    for (int64_t size : split) {
      if (size < 0 || ONNX_LIGHT_NAMESPACE::checked_add_overflow(sum, size, &sum)) {
        Invalid("negative split size or split sum overflow.");
      }
    }
    if (sum != dim) {
      Invalid("split sizes must sum to the axis dimension.");
    }
    return {split.begin(), split.end()};
  }
  if (num_outputs <= 0 || num_outputs > std::numeric_limits<int>::max()) {
    Invalid("a positive, representable number of outputs is required.");
  }
  const int64_t chunk = dim / num_outputs + (dim % num_outputs != 0);
  const int64_t prefix = Multiply(chunk, num_outputs - 1);
  if (prefix > dim) {
    Invalid("num_outputs would produce a negative final split.");
  }
  std::vector<int64_t> sizes(Size(num_outputs), chunk);
  sizes.back() = dim - prefix;
  return sizes;
}

std::vector<int64_t> ReadSplit(const Tensor &tensor) {
  if (tensor.data_type != DataType::INT64 || tensor.shape.size() != 1) {
    Invalid("split must be a 1-D INT64 tensor.");
  }
  const int64_t count = Count(tensor.shape, 0, 1);
  if (count == 0 || count > std::numeric_limits<int>::max()) {
    Invalid("split must contain at least one size and a representable output count.");
  }
  const auto bytes = Size(Multiply(count, sizeof(int64_t)));
  ValidateBuffer(tensor, bytes);
  std::vector<int64_t> sizes(Size(count));
  std::memcpy(sizes.data(), tensor.bytes(), bytes);
  return sizes;
}

void RejectOverlap(const Tensor &left, const Tensor &right) {
  if (left.size_bytes() == 0 || right.size_bytes() == 0) {
    return;
  }
  const auto a = reinterpret_cast<std::uintptr_t>(left.bytes());
  const auto b = reinterpret_cast<std::uintptr_t>(right.bytes());
  if (a <= b ? b - a < left.size_bytes() : a - b < right.size_bytes()) {
    Invalid("outputs must not overlap inputs or each other.");
  }
}

} // namespace

SplitKernel::SplitKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                         const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
}

std::vector<Tensor> SplitKernel::operator()(const Tensor &data, int64_t axis,
                                            std::span<const int64_t> split, int64_t num_outputs,
                                            rt_ns::RuntimeContext *rt) const {
  return Compute(data, axis, split, num_outputs, rt, nullptr);
}

std::vector<Tensor> SplitKernel::Compute(const Tensor &data, int64_t axis,
                                         std::span<const int64_t> split, int64_t num_outputs,
                                         rt_ns::RuntimeContext *rt,
                                         const Tensor *split_input) const {
  const std::size_t a = Axis(data, axis);
  const auto width = static_cast<int64_t>(data.element_size());
  const int64_t count = Count(data.shape, 0, data.shape.size());
  ValidateBuffer(data, Size(Multiply(count, width)));
  const auto sizes = Resolve(data.shape[a], split, num_outputs);
  // Empty tensors require no strides or pointer arithmetic, even with huge unused dimensions.
  const int64_t rows = count == 0 ? 0 : Count(data.shape, 0, a);
  const int64_t inner_bytes =
      count == 0 ? 0 : Multiply(Count(data.shape, a + 1, data.shape.size()), width);
  const auto input_row_bytes = Size(Multiply(data.shape[a], inner_bytes));
  std::vector<Tensor> outputs;
  outputs.reserve(sizes.size());
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    Shape shape = data.shape;
    shape[a] = sizes[i];
    const auto bytes = Size(Multiply(Multiply(rows, sizes[i]), inner_bytes));
    outputs.push_back(rt ? rt->MakeOutputTensor(static_cast<int>(i), data.data_type, shape, bytes)
                         : rt_ns::MakeOutputTensor(data.data_type, shape, bytes, ctx_.allocator));
    ValidateBuffer(outputs.back(), bytes);
    RejectOverlap(data, outputs.back());
    if (split_input) {
      RejectOverlap(*split_input, outputs.back());
    }
    for (std::size_t j = 0; j < i; ++j) {
      RejectOverlap(outputs[j], outputs.back());
    }
  }
  std::size_t offset = 0;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    const auto row_bytes = Size(Multiply(sizes[i], inner_bytes));
    SplitCopy(data.bytes(), outputs[i].mutable_bytes(), rows, input_row_bytes, row_bytes, offset);
    offset += row_bytes;
  }
  return outputs;
}

void SplitKernel::Run(rt_ns::RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const auto &node = *node_;
  const int64_t version = ctx_.opset.version;
  const bool legacy = version > 0 && version < 13;
  const bool modern = version == 0 || version >= 18;
  rt_ns::RequireMinInputCount(node, 1);
  if (node.input_size() > (legacy ? 1 : 2) || node.output_size() < 1) {
    Invalid("invalid input or output count.");
  }
  const auto *split_attr = rt_ns::FindAttribute(node, "split");
  const auto *num_attr = rt_ns::FindAttribute(node, "num_outputs");
  if ((!legacy && split_attr) || (!modern && num_attr)) {
    Invalid("attribute is not supported in this opset.");
  }
  const Tensor &data = rt_ns::GetInput(node, 0, rt.tensors());
  const int64_t axis = rt_ns::GetAttributeIntOrDefault(node, "axis", 0);
  const Tensor *split_input = legacy ? nullptr : rt_ns::GetOptionalInput(node, 1, rt.tensors());
  const bool explicit_split = split_attr != nullptr || split_input != nullptr;
  std::vector<int64_t> sizes;
  if (split_input) {
    sizes = ReadSplit(*split_input);
  } else if (split_attr) {
    sizes = rt_ns::GetAttributeIntsOrDefault(node, "split", {});
  }
  if (explicit_split && (sizes.empty() || num_attr)) {
    Invalid("split must be nonempty and cannot accompany num_outputs.");
  }
  int64_t num_outputs = 0;
  if (explicit_split) {
    if (sizes.size() != static_cast<std::size_t>(node.output_size())) {
      Invalid("split size count must match node output count.");
    }
  } else if (modern) {
    num_outputs = rt_ns::GetAttributeIntOrDefault(node, "num_outputs", 0);
    if (num_outputs != node.output_size()) {
      Invalid("num_outputs is required and must match node output count.");
    }
  } else {
    const int64_t dim = data.shape[Axis(data, axis)];
    if (dim < 0 || dim % node.output_size() != 0) {
      Invalid("implicit splits before opset 18 require equal division.");
    }
    sizes.assign(static_cast<std::size_t>(node.output_size()), dim / node.output_size());
  }
  // Copying the split values above avoids alignment assumptions and keeps allocation independent
  // of the optional input's storage.
  auto outputs = Compute(data, axis, sizes, num_outputs, &rt, split_input);
  for (int i = 0; i < node.output_size(); ++i) {
    rt_ns::SetOutput(node, i, std::move(outputs[static_cast<std::size_t>(i)]), rt);
  }
}

void RegisterSplitKernel() {
  rt_ns::NodeKernelFn factory =
      [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
         rt_ns::RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<SplitKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Split";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = SplitKernel::kName;
  info.types = {DataType::FLOAT,      DataType::DOUBLE,         DataType::FLOAT16,
                DataType::BFLOAT16,   DataType::INT8,           DataType::UINT8,
                DataType::INT16,      DataType::UINT16,         DataType::INT32,
                DataType::UINT32,     DataType::INT64,          DataType::UINT64,
                DataType::BOOL,       DataType::FLOAT8E4M3FN,   DataType::FLOAT8E4M3FNUZ,
                DataType::FLOAT8E5M2, DataType::FLOAT8E5M2FNUZ, DataType::FLOAT8E8M0};
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
