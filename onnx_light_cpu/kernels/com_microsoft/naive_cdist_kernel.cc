// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/naive_cdist_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Tensor;

namespace {

void ValidateTensors(const Tensor &a, const Tensor &b, const Tensor &output) {
  if (a.data_type != b.data_type || a.data_type != output.data_type) {
    throw std::invalid_argument("onnx_light_cpu::NaiveCDist: A, B and output dtypes must match.");
  }
  if (a.shape.size() != 2 || b.shape.size() != 2) {
    throw std::invalid_argument("onnx_light_cpu::NaiveCDist: A and B must be rank-2 tensors.");
  }
  if (a.shape[1] != b.shape[1]) {
    throw std::invalid_argument(
        "onnx_light_cpu::NaiveCDist: A and B must share the same feature dimension.");
  }
  if (output.shape.size() != 2 || output.shape[0] != a.shape[0] || output.shape[1] != b.shape[0]) {
    throw std::invalid_argument("onnx_light_cpu::NaiveCDist: output shape must be (M, K).");
  }
}

bool IsEuclidean(const std::string &metric) {
  if (metric == "euclidean") {
    return true;
  }
  if (metric == "sqeuclidean") {
    return false;
  }
  throw std::invalid_argument(
      "onnx_light_cpu::NaiveCDist: metric must be \"sqeuclidean\" or \"euclidean\".");
}

template <typename T>
void Compute(const T *a, const T *b, T *output, std::size_t m, std::size_t k, std::size_t n,
             bool euclidean) {
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t column = 0; column < k; ++column) {
      T sum = T(0);
      for (std::size_t feature = 0; feature < n; ++feature) {
        const T difference = a[row * n + feature] - b[column * n + feature];
        sum += difference * difference;
      }
      output[row * k + column] = euclidean ? std::sqrt(sum) : sum;
    }
  }
}

} // namespace

NaiveCDistKernel::NaiveCDistKernel(const NodeProto &node, const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  set_node(node);
}

Tensor NaiveCDistKernel::operator()(const Tensor &a, const Tensor &b, const std::string &metric,
                                    RuntimeContext *rt) const {
  if (a.shape.size() != 2 || b.shape.size() != 2) {
    throw std::invalid_argument("onnx_light_cpu::NaiveCDist: A and B must be rank-2 tensors.");
  }
  const rt_ns::Shape output_shape{a.shape[0], b.shape[0]};
  const std::size_t bytes = static_cast<std::size_t>(a.shape[0]) *
                            static_cast<std::size_t>(b.shape[0]) * a.element_size();
  Tensor output = rt != nullptr
                      ? rt->MakeOutputTensor(0, a.data_type, output_shape, bytes)
                      : rt_ns::MakeOutputTensor(a.data_type, output_shape, bytes, nullptr);
  (*this)(a, b, metric, output);
  return output;
}

void NaiveCDistKernel::operator()(const Tensor &a, const Tensor &b, const std::string &metric,
                                  Tensor &output) const {
  ValidateTensors(a, b, output);
  const bool euclidean = IsEuclidean(metric);
  const std::size_t m = static_cast<std::size_t>(a.shape[0]);
  const std::size_t k = static_cast<std::size_t>(b.shape[0]);
  const std::size_t n = static_cast<std::size_t>(a.shape[1]);
  switch (static_cast<DataType>(a.data_type)) {
  case DataType::FLOAT:
    Compute(a.AsFloat(), b.AsFloat(), output.AsFloat(), m, k, n, euclidean);
    return;
  case DataType::DOUBLE:
    Compute(a.AsDouble(), b.AsDouble(), output.AsDouble(), m, k, n, euclidean);
    return;
  default:
    throw std::invalid_argument("onnx_light_cpu::NaiveCDist: only FLOAT and DOUBLE are supported.");
  }
}

void NaiveCDistKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 2);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &a = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &b = rt_ns::GetInput(node, 1, rt.tensors());
  const std::string metric = rt_ns::GetAttributeStringOrDefault(node, "metric", "sqeuclidean");
  rt_ns::SetOutput(node, 0, (*this)(a, b, metric, &rt), rt);
}

void RegisterNaiveCDistKernel() {
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<NaiveCDistKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "CDist";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = NaiveCDistKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE};
  info.since_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
