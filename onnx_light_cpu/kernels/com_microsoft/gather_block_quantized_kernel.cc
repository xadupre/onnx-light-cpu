// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/gather_block_quantized_kernel.h"

#include "ml_cpu_compute/checked_arithmetic.h"
#include "ml_cpu_compute/com_microsoft/gather_block_quantized.h"
#include "onnx_light_cpu/kernels/execution.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace onnx_light_cpu {
namespace {

namespace runtime = ONNX_LIGHT_NAMESPACE::core::runtime;
using runtime::DataType;
using runtime::Tensor;
constexpr const char *kRoutine = "GatherBlockQuantized";

[[noreturn]] void Invalid(const char *message) {
  throw std::invalid_argument(std::string(kRoutine) + ": " + message);
}

void ValidateAttributes(const GatherBlockQuantizedAttributes &a) {
  if ((a.bits != 2 && a.bits != 4 && a.bits != 8) || a.block_size < 16 ||
      (a.block_size & (a.block_size - 1)) != 0) {
    Invalid("bits must be 2, 4, or 8 and block_size must be a power of two >= 16.");
  }
}

std::size_t Bytes(int64_t count, std::size_t width) {
  const auto bytes = ml_cpu_compute::CheckedByteSize(
      ml_cpu_compute::CheckedDimension(count, kRoutine, "element count"), width, kRoutine, "bytes");
  ml_cpu_compute::CheckedStride(bytes, kRoutine, "bytes");
  return bytes;
}

void ValidateBuffer(const Tensor &tensor) {
  const auto count = ml_cpu_compute::CheckedShapeIndexProduct(tensor.shape, kRoutine, "shape");
  const auto bytes = Bytes(count, tensor.element_size());
  if (tensor.size_bytes() != bytes || (bytes != 0 && tensor.bytes() == nullptr)) {
    Invalid("tensor buffer does not match shape.");
  }
}

int64_t NormalizeAxis(int64_t axis, int64_t rank) {
  if (axis < -rank || axis >= rank) {
    Invalid("axis out of range.");
  }
  return axis < 0 ? axis + rank : axis;
}

template <typename Scale>
void Compute(const Tensor &data, const Tensor &indices, const Tensor &scales,
             const Tensor *zero_points, Tensor &output,
             const ml_cpu_compute::GatherBlockQuantizedParameters &p) {
  const auto *zp = zero_points == nullptr ? nullptr : zero_points->As<uint8_t>();
  const auto *scale_data = reinterpret_cast<const Scale *>(scales.bytes());
  auto *output_data = reinterpret_cast<Scale *>(output.mutable_bytes());
  if (indices.data_type == DataType::INT32) {
    ml_cpu_compute::GatherBlockQuantized(data.As<uint8_t>(), indices.As<int32_t>(), scale_data, zp,
                                         output_data, p);
  } else {
    ml_cpu_compute::GatherBlockQuantized(data.As<uint8_t>(), indices.As<int64_t>(), scale_data, zp,
                                         output_data, p);
  }
}

} // namespace

GatherBlockQuantizedKernel::GatherBlockQuantizedKernel(const runtime::KernelContext &ctx,
                                                       GatherBlockQuantizedAttributes attributes)
    : KernelBase(ctx), attributes_(attributes) {
  ValidateAttributes(attributes_);
}

GatherBlockQuantizedKernel::GatherBlockQuantizedKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                                       const runtime::KernelContext &ctx)
    : GatherBlockQuantizedKernel(ctx, {runtime::GetAttributeIntOrDefault(node, "gather_axis", 0),
                                       runtime::GetAttributeIntOrDefault(node, "quantize_axis", 1),
                                       runtime::GetAttributeIntOrDefault(node, "block_size", 128),
                                       runtime::GetAttributeIntOrDefault(node, "bits", 4)}) {
  set_node(node);
}

Tensor GatherBlockQuantizedKernel::operator()(const Tensor &data, const Tensor &indices,
                                              const Tensor &scales, const Tensor *zero_points,
                                              runtime::RuntimeContext *rt) const {
  const RuntimeExecutionScope execution_scope;
  const auto &a = attributes_;
  const auto rank = static_cast<int64_t>(data.shape.size());
  if (rank < 2 || NormalizeAxis(a.gather_axis, rank) != 0 ||
      NormalizeAxis(a.quantize_axis, rank) != rank - 1) {
    Invalid("data must have rank >= 2, gather_axis first, and quantize_axis last.");
  }
  if (data.data_type != DataType::UINT8 ||
      (indices.data_type != DataType::INT32 && indices.data_type != DataType::INT64) ||
      (scales.data_type != DataType::FLOAT && scales.data_type != DataType::FLOAT16) ||
      (zero_points != nullptr && zero_points->data_type != DataType::UINT8)) {
    Invalid("expected UINT8 data/zero_points, INT32/INT64 indices, FLOAT/FLOAT16 scales.");
  }
  ValidateBuffer(data);
  ValidateBuffer(indices);
  ValidateBuffer(scales);
  if (zero_points != nullptr) {
    ValidateBuffer(*zero_points);
  }
  const int64_t components = 8 / a.bits;
  const int64_t columns =
      ml_cpu_compute::CheckedIndexMultiply(data.shape.back(), components, kRoutine, "columns");
  const int64_t blocks = columns / a.block_size + (columns % a.block_size != 0);
  auto scale_shape = data.shape;
  scale_shape.back() = blocks;
  if (scales.shape != scale_shape) {
    Invalid("scales shape must match data with ceil(logical_last / block_size) last.");
  }
  if (zero_points != nullptr) {
    scale_shape.back() = blocks / components + (blocks % components != 0);
    if (zero_points->shape != scale_shape) {
      Invalid("zero_points shape must match scales with a packed last dimension.");
    }
  }
  runtime::Shape output_shape = indices.shape;
  for (std::size_t axis = 1; axis < data.shape.size(); ++axis) {
    output_shape.push_back(data.shape[axis]);
  }
  output_shape.back() = columns;
  const auto output_count =
      ml_cpu_compute::CheckedShapeIndexProduct(output_shape, kRoutine, "output shape");
  const auto output_bytes = Bytes(output_count, scales.element_size());
  ml_cpu_compute::GatherBlockQuantizedParameters p;
  p.rows = data.shape[0];
  p.rows_per_gather = 1;
  for (std::size_t axis = 1; axis + 1 < data.shape.size(); ++axis) {
    p.rows_per_gather = ml_cpu_compute::CheckedIndexMultiply(p.rows_per_gather, data.shape[axis],
                                                             kRoutine, "rows_per_gather");
  }
  p.packed_columns = data.shape.back();
  p.index_count = ml_cpu_compute::CheckedShapeIndexProduct(indices.shape, kRoutine, "indices");
  p.bits = a.bits;
  p.block_size = a.block_size;
  Tensor output =
      rt ? rt->MakeOutputTensor(0, scales.data_type, output_shape, output_bytes)
         : runtime::MakeOutputTensor(scales.data_type, output_shape, output_bytes, ctx_.allocator);
  ValidateBuffer(output);
  if (scales.data_type == DataType::FLOAT) {
    Compute<float>(data, indices, scales, zero_points, output, p);
  } else {
    Compute<uint16_t>(data, indices, scales, zero_points, output, p);
  }
  return output;
}

void GatherBlockQuantizedKernel::Run(runtime::RuntimeContext &rt) {
  const RuntimeExecutionScope execution_scope;
  rt.RecordKernelUsage(kName);
  const auto &node = *node_;
  if (node.input_size() < 3 || node.input_size() > 4) {
    Invalid("expected three or four inputs.");
  }
  runtime::RequireOutputCount(node, 1);
  runtime::SetOutput(node, 0,
                     (*this)(runtime::GetInput(node, 0, rt), runtime::GetInput(node, 1, rt),
                             runtime::GetInput(node, 2, rt),
                             runtime::GetOptionalInput(node, 3, rt.tensors()), &rt),
                     rt);
}

void RegisterGatherBlockQuantizedKernel() {
  runtime::NodeKernelFn factory = [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                     runtime::RuntimeContext &, const runtime::KernelContext &ctx) {
    return std::make_unique<GatherBlockQuantizedKernel>(node, ctx);
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "GatherBlockQuantized";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = GatherBlockQuantizedKernel::kName;
  info.types = {DataType::UINT8};
  info.since_version = 1;
  info.until_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
