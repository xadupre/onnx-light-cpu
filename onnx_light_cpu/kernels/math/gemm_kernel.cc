// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/gemm_kernel.h"

#include "onnx_light_cpu/impl/math/math_kernels.h"

#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::NodeKernelFn;
using rt_ns::RuntimeContext;
using rt_ns::Shape;
using rt_ns::Tensor;

namespace {

// Returns the two matrix dimensions of a 2-D ``Gemm`` operand, throwing when
// the tensor is not rank 2.
void Require2D(const Tensor &t, const char *name, std::size_t &rows, std::size_t &cols) {
  if (t.shape.size() != 2) {
    throw std::invalid_argument(std::string("onnx_light_cpu::GemmKernel: input ") + name +
                                " must be a 2-D matrix.");
  }
  rows = static_cast<std::size_t>(t.shape[0]);
  cols = static_cast<std::size_t>(t.shape[1]);
}

// Materializes the optional bias ``C`` into a contiguous ``M x N`` row-major
// buffer, applying the ONNX unidirectional broadcasting rules (``C`` may be a
// scalar, a vector, or any 2-D shape broadcastable to ``M x N``). Returns an
// empty vector when there is no bias.
template <typename T>
std::vector<T> BroadcastBias(const Tensor &c, const T *c_data, std::size_t M, std::size_t N) {
  const std::size_t rank = c.shape.size();
  std::size_t c_rows = 1;
  std::size_t c_cols = 1;
  if (rank == 1) {
    c_cols = static_cast<std::size_t>(c.shape[0]);
  } else if (rank == 2) {
    c_rows = static_cast<std::size_t>(c.shape[0]);
    c_cols = static_cast<std::size_t>(c.shape[1]);
  } else if (rank != 0) {
    throw std::invalid_argument("onnx_light_cpu::GemmKernel: bias C must have rank 0, 1 or 2.");
  }
  if (!(c_rows == M || c_rows == 1) || !(c_cols == N || c_cols == 1)) {
    throw std::invalid_argument(
        "onnx_light_cpu::GemmKernel: bias C is not broadcastable to the output shape.");
  }
  std::vector<T> out(M * N);
  for (std::size_t m = 0; m < M; ++m) {
    const std::size_t cm = c_rows == 1 ? 0 : m;
    for (std::size_t n = 0; n < N; ++n) {
      const std::size_t cn = c_cols == 1 ? 0 : n;
      out[m * N + n] = c_data[cm * c_cols + cn];
    }
  }
  return out;
}

} // namespace

Tensor GemmKernel::operator()(const Tensor &a, const Tensor &b, const Tensor *c, float alpha,
                              float beta, bool trans_a, bool trans_b, RuntimeContext *rt) const {
  if (a.data_type != b.data_type) {
    throw std::invalid_argument("onnx_light_cpu::GemmKernel: A and B must share the same dtype.");
  }
  if (c != nullptr && c->data_type != a.data_type) {
    throw std::invalid_argument("onnx_light_cpu::GemmKernel: bias C dtype must match A and B.");
  }

  std::size_t a_rows = 0, a_cols = 0, b_rows = 0, b_cols = 0;
  Require2D(a, "A", a_rows, a_cols);
  Require2D(b, "B", b_rows, b_cols);

  const std::size_t M = trans_a ? a_cols : a_rows;
  const std::size_t K = trans_a ? a_rows : a_cols;
  const std::size_t K_b = trans_b ? b_cols : b_rows;
  const std::size_t N = trans_b ? b_rows : b_cols;
  if (K != K_b) {
    throw std::invalid_argument("onnx_light_cpu::GemmKernel: inner dimensions of A and B mismatch.");
  }

  const Shape out_shape{static_cast<std::int64_t>(M), static_cast<std::int64_t>(N)};
  const bool has_bias = c != nullptr && beta != 0.0f;

  switch (static_cast<DataType>(a.data_type)) {
  case DataType::FLOAT: {
    std::vector<float> bias;
    const float *c_ptr = nullptr;
    if (has_bias) {
      bias = BroadcastBias<float>(*c, c->AsFloat(), M, N);
      c_ptr = bias.data();
    }
    const std::size_t n_bytes = M * N * sizeof(float);
    Tensor y = rt_ns::MakeOutputTensor(a.data_type, out_shape, n_bytes,
                                       rt != nullptr ? rt->allocator() : nullptr);
    GemmFloat32(trans_a, trans_b, M, N, K, alpha, a.AsFloat(), b.AsFloat(), beta, c_ptr,
                y.AsFloat());
    return y;
  }
  case DataType::DOUBLE: {
    std::vector<double> bias;
    const double *c_ptr = nullptr;
    if (has_bias) {
      bias = BroadcastBias<double>(*c, c->AsDouble(), M, N);
      c_ptr = bias.data();
    }
    const std::size_t n_bytes = M * N * sizeof(double);
    Tensor y = rt_ns::MakeOutputTensor(a.data_type, out_shape, n_bytes,
                                       rt != nullptr ? rt->allocator() : nullptr);
    GemmFloat64(trans_a, trans_b, M, N, K, static_cast<double>(alpha), a.AsDouble(), b.AsDouble(),
                static_cast<double>(beta), c_ptr, y.AsDouble());
    return y;
  }
  default:
    throw std::invalid_argument("onnx_light_cpu::GemmKernel: unsupported data type " +
                                std::to_string(a.data_type) +
                                ", only FLOAT and DOUBLE are supported.");
  }
}

void GemmKernel::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &a = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &b = rt_ns::GetInput(node, 1, rt.tensors());
  const Tensor *c = rt_ns::GetOptionalInput(node, 2, rt.tensors());
  const float alpha = rt_ns::GetAttributeFloatOrDefault(node, "alpha", 1.0f);
  const float beta = rt_ns::GetAttributeFloatOrDefault(node, "beta", 1.0f);
  const bool trans_a = rt_ns::GetAttributeIntOrDefault(node, "transA", 0) != 0;
  const bool trans_b = rt_ns::GetAttributeIntOrDefault(node, "transB", 0) != 0;
  rt_ns::SetOutput(node, 0, (*this)(a, b, c, alpha, beta, trans_a, trans_b, &rt), rt);
}

void RegisterGemmKernel() {
  NodeKernelFn factory = [](const NodeProto &node,
                            RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<GemmKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  // Empty domain -> normalised to the default ONNX domain, overriding the
  // built-in Gemm entry with the SIMD-accelerated kernel for the CPU device.
  rt_ns::RegisterKernelFn("", "Gemm", sym_ns::Device::kCPU, std::move(factory));
}

} // namespace onnx_light_cpu
