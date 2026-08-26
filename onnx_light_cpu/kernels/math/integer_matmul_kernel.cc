// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/integer_matmul_kernel.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_vnni.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER
#include "onnx_light_cpu/impl/simd_level.h"
#endif

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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

bool IsInt8OrUint8(int32_t data_type) {
  return data_type == static_cast<int32_t>(DataType::INT8) ||
         data_type == static_cast<int32_t>(DataType::UINT8);
}

int32_t ReadInteger(const Tensor &tensor, int64_t index) {
  return tensor.data_type == static_cast<int32_t>(DataType::INT8)
             ? static_cast<int32_t>(tensor.AsInt8()[index])
             : static_cast<int32_t>(tensor.AsUint8()[index]);
}

Shape PromoteMatMulShape(const Shape &shape, bool left) {
  if (shape.empty()) {
    throw std::invalid_argument("integer MatMul does not accept rank-0 inputs.");
  }
  if (shape.size() != 1) {
    return shape;
  }
  return left ? Shape{1, shape[0]} : Shape{shape[0], 1};
}

Shape BroadcastPrefix(const Shape &a, const Shape &b) {
  const std::size_t rank = std::max(a.size(), b.size());
  Shape output(std::vector<int64_t>(rank, 1));
  for (std::size_t index = 0; index < rank; ++index) {
    const int64_t a_dim = index + a.size() >= rank ? a[index - (rank - a.size())] : int64_t{1};
    const int64_t b_dim = index + b.size() >= rank ? b[index - (rank - b.size())] : int64_t{1};
    if (a_dim == b_dim || a_dim == 1) {
      output[index] = b_dim;
    } else if (b_dim == 1) {
      output[index] = a_dim;
    } else {
      throw std::invalid_argument(
          "integer MatMul inputs are not broadcast-compatible on batch dimensions.");
    }
  }
  return output;
}

struct MatMulLayout {
  Shape a_shape;
  Shape b_shape;
  Shape output_shape;
  Shape output_prefix;
  Shape a_strides;
  Shape b_strides;
  Shape output_strides;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t batch_count = 1;
};

struct BatchWorkItem {
  int64_t a_base = 0;
  int64_t b_base = 0;
  int64_t output_base = 0;
};

Shape ComputeStrides(const Shape &shape) {
  Shape strides(std::vector<int64_t>(shape.size(), 1));
  for (std::size_t index = shape.size(); index > 1; --index) {
    strides[index - 2] = strides[index - 1] * shape[index - 1];
  }
  return strides;
}

int64_t ElementCount(const Shape &shape) {
  int64_t count = 1;
  for (int64_t dimension : shape) {
    if (dimension < 0) {
      throw std::invalid_argument("integer MatMul requires concrete non-negative dimensions.");
    }
    count *= dimension;
  }
  return count;
}

MatMulLayout ResolveLayout(const Tensor &a, const Tensor &b) {
  MatMulLayout layout;
  layout.a_shape = PromoteMatMulShape(a.shape, true);
  layout.b_shape = PromoteMatMulShape(b.shape, false);
  layout.m = layout.a_shape[layout.a_shape.size() - 2];
  layout.k = layout.a_shape.back();
  layout.n = layout.b_shape.back();
  if (layout.k != layout.b_shape[layout.b_shape.size() - 2]) {
    throw std::invalid_argument("integer MatMul inputs have incompatible inner dimensions.");
  }

  const Shape a_prefix(std::vector<int64_t>(layout.a_shape.begin(), layout.a_shape.end() - 2));
  const Shape b_prefix(std::vector<int64_t>(layout.b_shape.begin(), layout.b_shape.end() - 2));
  layout.output_prefix = BroadcastPrefix(a_prefix, b_prefix);
  layout.output_shape = layout.output_prefix;
  if (a.shape.size() != 1) {
    layout.output_shape.push_back(layout.m);
  }
  if (b.shape.size() != 1) {
    layout.output_shape.push_back(layout.n);
  }
  layout.a_strides = ComputeStrides(layout.a_shape);
  layout.b_strides = ComputeStrides(layout.b_shape);
  layout.output_strides = ComputeStrides(layout.output_shape);
  layout.batch_count = ElementCount(layout.output_prefix);
  return layout;
}

std::vector<int32_t> ReadZeroPoints(const Tensor *tensor, int32_t expected_type,
                                    int64_t expected_size, const char *name) {
  if (tensor == nullptr) {
    return {0};
  }
  if (tensor->data_type != expected_type) {
    throw std::invalid_argument(std::string(name) + " dtype must match its data input.");
  }
  if (tensor->shape.size() > 1) {
    throw std::invalid_argument(std::string(name) + " must be scalar or one-dimensional.");
  }
  const int64_t count = tensor->element_count();
  if (count != 1 && count != expected_size) {
    throw std::invalid_argument(std::string(name) +
                                " must contain one value or one value per matrix axis.");
  }
  std::vector<int32_t> values(static_cast<std::size_t>(count));
  for (int64_t index = 0; index < count; ++index) {
    values[static_cast<std::size_t>(index)] = ReadInteger(*tensor, index);
  }
  return values;
}

template <typename Fn> void ForEachBatch(const MatMulLayout &layout, Fn fn) {
  const Shape a_prefix(std::vector<int64_t>(layout.a_shape.begin(), layout.a_shape.end() - 2));
  const Shape b_prefix(std::vector<int64_t>(layout.b_shape.begin(), layout.b_shape.end() - 2));
  const std::size_t batch_rank = layout.output_prefix.size();
  Shape batch_index(std::vector<int64_t>(batch_rank, 0));

  for (int64_t batch = 0; batch < layout.batch_count; ++batch) {
    int64_t a_base = 0;
    int64_t b_base = 0;
    int64_t output_base = 0;
    for (std::size_t dimension = 0; dimension < batch_rank; ++dimension) {
      const int64_t coordinate = batch_index[dimension];
      if (dimension + a_prefix.size() >= batch_rank) {
        const std::size_t a_dimension = dimension - (batch_rank - a_prefix.size());
        a_base += (a_prefix[a_dimension] == 1 ? 0 : coordinate) * layout.a_strides[a_dimension];
      }

      if (dimension + b_prefix.size() >= batch_rank) {
        const std::size_t b_dimension = dimension - (batch_rank - b_prefix.size());
        b_base += (b_prefix[b_dimension] == 1 ? 0 : coordinate) * layout.b_strides[b_dimension];
      }
      output_base += coordinate * layout.output_strides[dimension];
    }

    fn(a_base, b_base, output_base);

    for (std::size_t dimension = batch_rank; dimension-- > 0;) {
      if (++batch_index[dimension] < layout.output_prefix[dimension]) {
        break;
      }
      batch_index[dimension] = 0;
    }
  }
}

std::vector<BatchWorkItem> BuildBatchWorkItems(const MatMulLayout &layout) {
  std::vector<BatchWorkItem> items;
  items.reserve(static_cast<std::size_t>(layout.batch_count));
  ForEachBatch(layout, [&](int64_t a_base, int64_t b_base, int64_t output_base) {
    items.push_back({a_base, b_base, output_base});
  });
  return items;
}

template <typename Fn>
void ForEachOutput(const Tensor &a, const Tensor &b, const MatMulLayout &layout, Fn fn) {
  const std::size_t batch_rank = layout.output_prefix.size();
  ForEachBatch(layout, [&](int64_t a_base, int64_t b_base, int64_t output_base) {
    for (int64_t row = 0; row < layout.m; ++row) {
      for (int64_t column = 0; column < layout.n; ++column) {
        int64_t output_index = output_base;
        if (a.shape.size() != 1 && b.shape.size() != 1) {
          output_index += row * layout.output_strides[batch_rank] +
                          column * layout.output_strides[batch_rank + 1];
        } else if (a.shape.size() == 1 && b.shape.size() != 1) {
          output_index += column * layout.output_strides[batch_rank];
        } else if (a.shape.size() != 1 && b.shape.size() == 1) {
          output_index += row * layout.output_strides[batch_rank];
        }
        fn(a_base, b_base, row, column, output_index);
      }
    }
  });
}

int64_t ReadScalarInteger(const Tensor &tensor, const char *name) {
  if (tensor.element_count() != 1 || !IsInt8OrUint8(tensor.data_type)) {
    throw std::invalid_argument(std::string(name) + " must be an INT8 or UINT8 scalar.");
  }
  return ReadInteger(tensor, 0);
}

float ReadScalarScale(const Tensor &tensor, const char *name) {
  if (tensor.element_count() != 1) {
    throw std::invalid_argument(std::string(name) + " must be scalar.");
  }
  if (tensor.data_type == static_cast<int32_t>(DataType::FLOAT)) {
    return tensor.AsFloat()[0];
  }
  if (tensor.data_type == static_cast<int32_t>(DataType::FLOAT16)) {
    return rt_ns::Float16BitsToFloat(*reinterpret_cast<const std::uint16_t *>(tensor.bytes()));
  }
  throw std::invalid_argument(std::string(name) + " must have FLOAT or FLOAT16 dtype.");
}

template <typename T> T Requantize(int64_t accumulator, float scale, int32_t zero_point) {
  const double rounded =
      std::nearbyint(static_cast<double>(accumulator) * static_cast<double>(scale));
  const double shifted = rounded + static_cast<double>(zero_point);
  const double clamped = std::clamp(shifted, static_cast<double>(std::numeric_limits<T>::min()),
                                    static_cast<double>(std::numeric_limits<T>::max()));
  return static_cast<T>(clamped);
}

void RequantizeInt32Buffer(const int32_t *src, int8_t *dst, int64_t count, float scale,
                           int32_t zero_point) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER
  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
    detail::RequantizeInt32ToInt8Avx2(src, dst, count, scale, zero_point);
    return;
  }
#endif
  for (int64_t index = 0; index < count; ++index) {
    dst[index] = Requantize<int8_t>(src[index], scale, zero_point);
  }
}

void RequantizeInt32Buffer(const int32_t *src, uint8_t *dst, int64_t count, float scale,
                           int32_t zero_point) {
#ifdef ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER
  if (DetectSimdLevel() >= SimdLevel::kAVX2) {
    detail::RequantizeInt32ToUint8Avx2(src, dst, count, scale, zero_point);
    return;
  }
#endif
  for (int64_t index = 0; index < count; ++index) {
    dst[index] = Requantize<uint8_t>(src[index], scale, zero_point);
  }
}

} // namespace

Tensor MatMulIntegerKernel::operator()(const Tensor &a, const Tensor &b, const Tensor *a_zero_point,
                                       const Tensor *b_zero_point, RuntimeContext *rt) const {
  if (!IsInt8OrUint8(a.data_type) || !IsInt8OrUint8(b.data_type)) {
    throw std::invalid_argument("MatMulInteger inputs must have INT8 or UINT8 dtype.");
  }
  const MatMulLayout layout = ResolveLayout(a, b);
  const std::vector<int32_t> a_zp =
      ReadZeroPoints(a_zero_point, a.data_type, layout.m, "a_zero_point");
  const std::vector<int32_t> b_zp =
      ReadZeroPoints(b_zero_point, b.data_type, layout.n, "b_zero_point");
  const std::size_t output_bytes =
      static_cast<std::size_t>(ElementCount(layout.output_shape)) * sizeof(int32_t);
  Tensor output = rt != nullptr
                      ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::INT32),
                                             layout.output_shape, output_bytes)
                      : rt_ns::MakeOutputTensor(static_cast<int32_t>(DataType::INT32),
                                                layout.output_shape, output_bytes, nullptr);
  int32_t *values = output.AsInt32();

  // Roadmap PR09.2 / PR09.3: the plain matrix product (both operands rank >= 2,
  // so their inner two dimensions are contiguous) is routed through the shared
  // integer GEMM, which dispatches to the native x86 VNNI ``vpdpbusd`` path or
  // the ARM NEON dot-product kernel when the CPU supports them and otherwise to
  // the portable scalar sibling. Vector and rank-1 promotions keep the PR09.1
  // scalar fallback below.
  if (a.shape.size() >= 2 && b.shape.size() >= 2) {
    const auto *a_bytes = reinterpret_cast<const std::uint8_t *>(a.bytes());
    const auto *b_bytes = reinterpret_cast<const std::uint8_t *>(b.bytes());
    const bool a_signed = a.data_type == static_cast<int32_t>(DataType::INT8);
    const bool b_signed = b.data_type == static_cast<int32_t>(DataType::INT8);
    if (layout.batch_count == 1) {
      IntegerMatMul2D(a_bytes, a_signed, b_bytes, b_signed, values, layout.m, layout.n, layout.k,
                      a_zp.data(), static_cast<int64_t>(a_zp.size()), b_zp.data(),
                      static_cast<int64_t>(b_zp.size()));
      return output;
    }
    const std::vector<BatchWorkItem> work_items = BuildBatchWorkItems(layout);
    const double cost = static_cast<double>(layout.m) * static_cast<double>(layout.n) *
                        static_cast<double>(std::max<int64_t>(layout.k, 1));
    ExecuteRanges(static_cast<int64_t>(work_items.size()), cost, [&](int64_t begin, int64_t end) {
      for (int64_t index = begin; index < end; ++index) {
        const BatchWorkItem &item = work_items[static_cast<std::size_t>(index)];
        IntegerMatMul2D(a_bytes + item.a_base, a_signed, b_bytes + item.b_base, b_signed,
                        values + item.output_base, layout.m, layout.n, layout.k, a_zp.data(),
                        static_cast<int64_t>(a_zp.size()), b_zp.data(),
                        static_cast<int64_t>(b_zp.size()));
      }
    });
    return output;
  }

  ForEachOutput(
      a, b, layout,
      [&](int64_t a_base, int64_t b_base, int64_t row, int64_t column, int64_t output_index) {
        const int32_t az = a_zp.size() == 1 ? a_zp[0] : a_zp[row];
        const int32_t bz = b_zp.size() == 1 ? b_zp[0] : b_zp[column];
        std::uint32_t accumulator = 0;
        for (int64_t depth = 0; depth < layout.k; ++depth) {
          const int32_t av = ReadInteger(
              a, a_base +
                     row * layout.a_strides[a.shape.size() == 1 ? 0 : layout.a_shape.size() - 2] +
                     depth * layout.a_strides.back());
          const int32_t bv =
              ReadInteger(b, b_base + depth * layout.b_strides[layout.b_shape.size() - 2] +
                                 column * layout.b_strides.back());
          const int32_t product = (av - az) * (bv - bz);
          accumulator += static_cast<std::uint32_t>(product);
        }
        values[output_index] = std::bit_cast<int32_t>(accumulator);
      });
  return output;
}

void MatMulIntegerKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireMinInputCount(node, 2);
  if (node.input_size() > 4) {
    throw std::invalid_argument("MatMulInteger expects at most four inputs.");
  }
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &a = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &b = rt_ns::GetInput(node, 1, rt.tensors());
  const Tensor *a_zero_point = rt_ns::GetOptionalInput(node, 2, rt.tensors());
  const Tensor *b_zero_point = rt_ns::GetOptionalInput(node, 3, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(a, b, a_zero_point, b_zero_point, &rt), rt);
}

Tensor QLinearMatMulKernel::operator()(const Tensor &a, const Tensor &a_scale,
                                       const Tensor &a_zero_point, const Tensor &b,
                                       const Tensor &b_scale, const Tensor &b_zero_point,
                                       const Tensor &y_scale, const Tensor &y_zero_point,
                                       RuntimeContext *rt) const {
  if (!IsInt8OrUint8(a.data_type) || !IsInt8OrUint8(b.data_type) ||
      !IsInt8OrUint8(y_zero_point.data_type)) {
    throw std::invalid_argument(
        "QLinearMatMul data and output zero point must have INT8 or UINT8 dtype.");
  }
  if (a_zero_point.data_type != a.data_type || b_zero_point.data_type != b.data_type) {
    throw std::invalid_argument("QLinearMatMul zero point dtype must match its data input.");
  }
  const int32_t az = static_cast<int32_t>(ReadScalarInteger(a_zero_point, "a_zero_point"));
  const int32_t bz = static_cast<int32_t>(ReadScalarInteger(b_zero_point, "b_zero_point"));
  const int32_t yz = static_cast<int32_t>(ReadScalarInteger(y_zero_point, "y_zero_point"));
  const float as = ReadScalarScale(a_scale, "a_scale");
  const float bs = ReadScalarScale(b_scale, "b_scale");
  const float ys = ReadScalarScale(y_scale, "y_scale");
  if (!std::isfinite(as) || !std::isfinite(bs) || !std::isfinite(ys) || as <= 0.0f || bs <= 0.0f ||
      ys <= 0.0f) {
    throw std::invalid_argument("QLinearMatMul scales must be finite and strictly positive.");
  }

  const MatMulLayout layout = ResolveLayout(a, b);
  const std::size_t output_bytes = static_cast<std::size_t>(ElementCount(layout.output_shape));
  Tensor output =
      rt != nullptr
          ? rt->MakeOutputTensor(0, y_zero_point.data_type, layout.output_shape, output_bytes)
          : rt_ns::MakeOutputTensor(y_zero_point.data_type, layout.output_shape, output_bytes,
                                    nullptr);
  const float combined_scale = as * bs / ys;

  if (a.shape.size() >= 2 && b.shape.size() >= 2) {
    const auto *a_bytes = reinterpret_cast<const std::uint8_t *>(a.bytes());
    const auto *b_bytes = reinterpret_cast<const std::uint8_t *>(b.bytes());
    const bool a_signed = a.data_type == static_cast<int32_t>(DataType::INT8);
    const bool b_signed = b.data_type == static_cast<int32_t>(DataType::INT8);
    const std::int32_t a_zp[1] = {az};
    const std::int32_t b_zp[1] = {bz};
    const std::vector<BatchWorkItem> work_items = BuildBatchWorkItems(layout);
    const int64_t matrix_size = layout.m * layout.n;
    const double cost =
        static_cast<double>(matrix_size) * static_cast<double>(std::max<int64_t>(layout.k, 1));
    ExecuteRanges(static_cast<int64_t>(work_items.size()), cost, [&](int64_t begin, int64_t end) {
      thread_local std::vector<std::int32_t> accum;
      accum.resize(static_cast<std::size_t>(matrix_size));
      for (int64_t index = begin; index < end; ++index) {
        const BatchWorkItem &item = work_items[static_cast<std::size_t>(index)];
        IntegerMatMul2D(a_bytes + item.a_base, a_signed, b_bytes + item.b_base, b_signed,
                        accum.data(), layout.m, layout.n, layout.k, a_zp, 1, b_zp, 1);
        if (output.data_type == static_cast<int32_t>(DataType::INT8)) {
          RequantizeInt32Buffer(accum.data(), output.AsInt8() + item.output_base, matrix_size,
                                combined_scale, yz);
        } else {
          RequantizeInt32Buffer(accum.data(), output.AsUint8() + item.output_base, matrix_size,
                                combined_scale, yz);
        }
      }
    });
    return output;
  }

  ForEachOutput(
      a, b, layout,
      [&](int64_t a_base, int64_t b_base, int64_t row, int64_t column, int64_t output_index) {
        int64_t accumulator = 0;
        for (int64_t depth = 0; depth < layout.k; ++depth) {
          const int32_t av = ReadInteger(
              a, a_base +
                     row * layout.a_strides[a.shape.size() == 1 ? 0 : layout.a_shape.size() - 2] +
                     depth * layout.a_strides.back());
          const int32_t bv =
              ReadInteger(b, b_base + depth * layout.b_strides[layout.b_shape.size() - 2] +
                                 column * layout.b_strides.back());
          accumulator += static_cast<int64_t>(av - az) * (bv - bz);
        }
        if (output.data_type == static_cast<int32_t>(DataType::INT8)) {
          output.AsInt8()[output_index] = Requantize<int8_t>(accumulator, combined_scale, yz);
        } else {
          output.AsUint8()[output_index] = Requantize<uint8_t>(accumulator, combined_scale, yz);
        }
      });
  return output;
}

void QLinearMatMulKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 8);
  rt_ns::RequireOutputCount(node, 1);
  rt_ns::SetOutput(
      node, 0,
      (*this)(rt_ns::GetInput(node, 0, rt.tensors()), rt_ns::GetInput(node, 1, rt.tensors()),
              rt_ns::GetInput(node, 2, rt.tensors()), rt_ns::GetInput(node, 3, rt.tensors()),
              rt_ns::GetInput(node, 4, rt.tensors()), rt_ns::GetInput(node, 5, rt.tensors()),
              rt_ns::GetInput(node, 6, rt.tensors()), rt_ns::GetInput(node, 7, rt.tensors()), &rt),
      rt);
}

void RegisterIntegerMatMulKernels() {
  NodeKernelFn matmul_integer = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<MatMulIntegerKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  {
    KernelRegistration info;
    info.domain = "";
    info.op_type = "MatMulInteger";
    info.device = sym_ns::Device::kCPU;
    info.kernel_name = MatMulIntegerKernel::kName;
    info.types = {DataType::INT8, DataType::UINT8};
    RegisterKernel(std::move(info), std::move(matmul_integer));
  }

  NodeKernelFn qlinear_matmul = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<QLinearMatMulKernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  {
    KernelRegistration info;
    info.domain = "";
    info.op_type = "QLinearMatMul";
    info.device = sym_ns::Device::kCPU;
    info.kernel_name = QLinearMatMulKernel::kName;
    info.types = {DataType::INT8, DataType::UINT8};
    RegisterKernel(std::move(info), std::move(qlinear_matmul));
  }
}

} // namespace onnx_light_cpu
