// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/tensor/scatter_nd_kernel.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_lib/common/safe_math.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Tensor;

namespace {

[[noreturn]] void Invalid(const char *message) {
  throw std::invalid_argument(std::string("onnx_light_cpu::ScatterND: ") + message);
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

struct ScatterPlan {
  std::size_t output_bytes;
  std::size_t slice_bytes;
  std::vector<std::size_t> offsets;
};

template <typename Index>
void PrepareOffsets(const Tensor &data, const Tensor &indices, int64_t tuples, int64_t k,
                    ScatterPlan &plan) {
  const Index *values = indices.As<Index>();
  if (plan.slice_bytes != 0) {
    plan.offsets.reserve(ONNX_LIGHT_NAMESPACE::safe_cast_to_size(tuples, Invalid));
  }
  for (int64_t t = 0; t < tuples; ++t) {
    std::size_t offset = 0;
    for (int64_t j = 0; j < k; ++j) {
      int64_t index = values[t * k + j];
      const int64_t dim = data.shape[j];
      if (index < -dim || index >= dim) {
        Invalid("index out of range.");
      }
      if (index < 0) {
        index += dim;
      }
      // With nonempty slices, the checked data size bounds every prefix.
      if (plan.slice_bytes != 0) {
        offset = offset * static_cast<std::size_t>(dim) + static_cast<std::size_t>(index);
      }
    }
    if (plan.slice_bytes != 0) {
      plan.offsets.push_back(offset * plan.slice_bytes);
    }
  }
}

ScatterPlan Prepare(const Tensor &data, const Tensor &indices, const Tensor &updates) {
  if (data.shape.empty() || indices.shape.empty()) {
    Invalid("data and indices must have rank >= 1.");
  }
  if (indices.data_type != DataType::INT32 && indices.data_type != DataType::INT64) {
    Invalid("indices must be INT32 or INT64.");
  }
  if (data.data_type != updates.data_type) {
    Invalid("updates dtype must match data.");
  }
  const int64_t k = indices.shape.back();
  if (k < 1 || k > static_cast<int64_t>(data.shape.size())) {
    Invalid("last dimension of indices must be in [1, rank(data)].");
  }
  rt_ns::Shape expected;
  expected.insert(expected.end(), indices.shape.begin(), indices.shape.end() - 1);
  expected.insert(expected.end(), data.shape.begin() + k, data.shape.end());
  if (updates.shape != expected) {
    Invalid("updates shape must be indices.shape[:-1] + data.shape[k:].");
  }
  const std::size_t width = data.element_size();
  const int64_t data_count = data.shape.product(0, data.shape.size(), "ScatterND data");
  const int64_t index_count = indices.shape.product(0, indices.shape.size(), "ScatterND indices");
  const int64_t update_count = updates.shape.product(0, updates.shape.size(), "ScatterND updates");
  ScatterPlan plan{Bytes(data_count, width),
                   Bytes(data.shape.product(k, data.shape.size(), "ScatterND slice"), width),
                   {}};
  ValidateBuffer(data, plan.output_bytes);
  ValidateBuffer(indices, Bytes(index_count, indices.element_size()));
  ValidateBuffer(updates, Bytes(update_count, width));
  const int64_t tuples = indices.shape.product(0, indices.shape.size() - 1, "ScatterND tuples");
  if (indices.data_type == DataType::INT32) {
    PrepareOffsets<int32_t>(data, indices, tuples, k, plan);
  } else {
    PrepareOffsets<int64_t>(data, indices, tuples, k, plan);
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

void Copy(const Tensor &data, const Tensor &indices, const Tensor &updates, Tensor &output,
          const ScatterPlan &plan) {
  if (output.data_type != data.data_type || output.shape != data.shape) {
    Invalid("output dtype or shape mismatch.");
  }
  ValidateBuffer(output, plan.output_bytes);
  if (Overlaps(data, output) || Overlaps(indices, output) || Overlaps(updates, output)) {
    Invalid("output must not overlap data, indices or updates.");
  }
  if (plan.output_bytes == 0) {
    return;
  }
  const auto *source = data.bytes();
  auto *destination = output.mutable_bytes();
  ExecuteRanges(static_cast<int64_t>(plan.output_bytes),
                ExecutionSchedule{262144, 65536, ExecutionThreadCount()}, int64_t{64},
                [source, destination](int64_t begin, int64_t end) {
                  std::memcpy(destination + begin, source + begin,
                              static_cast<std::size_t>(end - begin));
                });
  // Serial tuple order makes duplicate writes deterministic and race-free.
  for (std::size_t t = 0; t < plan.offsets.size(); ++t) {
    std::memcpy(destination + plan.offsets[t], updates.bytes() + t * plan.slice_bytes,
                plan.slice_bytes);
  }
}

} // namespace

ScatterNDKernel::ScatterNDKernel(const rt_ns::KernelContext &ctx, const std::string &reduction)
    : KernelBase(ctx) {
  if (reduction != "none") {
    Invalid("only reduction='none' is supported.");
  }
}

ScatterNDKernel::ScatterNDKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                 const rt_ns::KernelContext &ctx)
    : ScatterNDKernel(ctx, rt_ns::GetAttributeStringOrDefault(node, "reduction", "none")) {
  set_node(node);
}

Tensor ScatterNDKernel::operator()(const Tensor &data, const Tensor &indices, const Tensor &updates,
                                   rt_ns::RuntimeContext *rt) const {
  const ScatterPlan plan = Prepare(data, indices, updates);
  Tensor output =
      rt ? rt->MakeOutputTensor(0, data.data_type, data.shape, plan.output_bytes)
         : rt_ns::MakeOutputTensor(data.data_type, data.shape, plan.output_bytes, ctx_.allocator);
  Copy(data, indices, updates, output, plan);
  return output;
}

void ScatterNDKernel::operator()(const Tensor &data, const Tensor &indices, const Tensor &updates,
                                 Tensor &output) const {
  const ScatterPlan plan = Prepare(data, indices, updates);
  Copy(data, indices, updates, output, plan);
}

void ScatterNDKernel::Run(rt_ns::RuntimeContext &rt) {
  rt.RecordKernelUsage(kName);
  const auto &node = *node_;
  rt_ns::RequireInputCount(node, 3);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &data = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &indices = rt_ns::GetInput(node, 1, rt.tensors());
  const Tensor &updates = rt_ns::GetInput(node, 2, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(data, indices, updates, &rt), rt);
}

void RegisterScatterNDKernel() {
  rt_ns::NodeKernelFn factory =
      [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
         rt_ns::RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<ScatterNDKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "ScatterND";
  info.kernel_name = ScatterNDKernel::kName;
  info.since_version = 11;
  info.types = {DataType::FLOAT,      DataType::DOUBLE,         DataType::FLOAT16,
                DataType::BFLOAT16,   DataType::INT8,           DataType::UINT8,
                DataType::INT16,      DataType::UINT16,         DataType::INT32,
                DataType::UINT32,     DataType::INT64,          DataType::UINT64,
                DataType::BOOL,       DataType::FLOAT8E4M3FN,   DataType::FLOAT8E4M3FNUZ,
                DataType::FLOAT8E5M2, DataType::FLOAT8E5M2FNUZ, DataType::FLOAT8E8M0};
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
