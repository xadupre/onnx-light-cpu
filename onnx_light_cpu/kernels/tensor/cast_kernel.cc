// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/tensor/cast_kernel.h"
#include "onnx_light_cpu/kernels/execution.h"

#include "cpu_compute/execution.h"
#include "cpu_compute/tensor/cast.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_lib/common/safe_math.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::Tensor;

namespace {

using rt_ns::DataType;
using BuiltinCast = ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Cast;

[[noreturn]] void Invalid(const char *message) {
  throw std::invalid_argument(std::string("onnx_light_cpu::Cast: ") + message);
}

bool Numeric(int32_t type) {
  return cpu_compute::IsCastNumericType(static_cast<cpu_compute::DataType>(type));
}

bool Supported(int32_t type) {
  if (Numeric(type)) {
    return true;
  }
  switch (static_cast<DataType>(type)) {
  case DataType::STRING:
  case DataType::COMPLEX64:
  case DataType::COMPLEX128:
  case DataType::FLOAT8E4M3FN:
  case DataType::FLOAT8E4M3FNUZ:
  case DataType::FLOAT8E5M2:
  case DataType::FLOAT8E5M2FNUZ:
  case DataType::FLOAT8E8M0:
  case DataType::INT4:
  case DataType::UINT4:
  case DataType::INT2:
  case DataType::UINT2:
  case DataType::FLOAT4E2M1:
  case DataType::FLOAT6E2M3:
  case DataType::FLOAT6E3M2:
    return true;
  default:
    return false;
  }
}

std::size_t Bytes(int32_t type, int64_t count) {
  if (!Supported(type)) {
    Invalid("unsupported source or destination type.");
  }
  if (type == DataType::STRING) {
    return 0;
  }
  // PackedByteSize's INT2/INT4 rounding adds up to three in signed arithmetic.
  // Fixed-byte arithmetic is checked separately before calling that helper.
  if (count > std::numeric_limits<int64_t>::max() - 3) {
    Invalid("element count exceeds safe packed-size arithmetic.");
  }
  if (Numeric(type) || (type >= DataType::FLOAT8E4M3FN && type <= DataType::FLOAT8E5M2FNUZ) ||
      type == DataType::FLOAT8E8M0 || type == DataType::COMPLEX64 || type == DataType::COMPLEX128) {
    int64_t bytes;
    const std::size_t width = type == DataType::COMPLEX64    ? 2 * sizeof(float)
                              : type == DataType::COMPLEX128 ? 2 * sizeof(double)
                                                             : rt_ns::ElementSize(type);
    if (ONNX_LIGHT_NAMESPACE::checked_mul_overflow(count, static_cast<int64_t>(width), &bytes)) {
      Invalid("byte size overflow.");
    }
    return ONNX_LIGHT_NAMESPACE::safe_cast_to_size(bytes, Invalid);
  }
  const std::size_t bytes = rt_ns::PackedByteSize(type, count);
  // The runtime's packed helpers address FLOAT6 by multiplying an index by six.
  if (static_cast<std::uint64_t>(count) > std::numeric_limits<std::size_t>::max() / 6) {
    Invalid("packed bit-offset overflow.");
  }
  return bytes;
}

void ValidateBuffer(const Tensor &tensor, int64_t count, std::size_t bytes) {
  if (tensor.size_bytes() != bytes || (bytes != 0 && tensor.bytes() == nullptr)) {
    Invalid("tensor buffer size does not match shape.");
  }
  if (tensor.data_type == DataType::STRING &&
      tensor.AsStrings().size() != static_cast<std::size_t>(count)) {
    Invalid("STRING storage must have one entry per element.");
  }
}

struct CastPlan {
  int64_t count;
  std::size_t output_bytes;
};

CastPlan PrepareCast(const Tensor &data, int32_t to) {
  const int64_t count = data.shape.product(0, data.shape.size(), "Cast");
  static_cast<void>(ONNX_LIGHT_NAMESPACE::safe_cast_to_size(count, Invalid));
  const std::size_t input_bytes = Bytes(data.data_type, count);
  const std::size_t output_bytes = Bytes(to, count);
  ValidateBuffer(data, count, input_bytes);
  return {count, output_bytes};
}

void ValidateOutput(const Tensor &data, int32_t to, const Tensor &output, const CastPlan &plan) {
  if (output.data_type != to || output.shape != data.shape) {
    Invalid("output dtype or shape mismatch.");
  }
  ValidateBuffer(output, plan.count, plan.output_bytes);
  if (&data == &output) {
    Invalid("output must not alias input.");
  }
  if (data.size_bytes() != 0 && output.size_bytes() != 0) {
    const auto source = reinterpret_cast<std::uintptr_t>(data.bytes());
    const auto destination = reinterpret_cast<std::uintptr_t>(output.bytes());
    if (source <= destination ? destination - source < data.size_bytes()
                              : source - destination < output.size_bytes()) {
      Invalid("output must not overlap input.");
    }
  }
  if (data.data_type == DataType::STRING && to == DataType::STRING && plan.count != 0 &&
      data.AsStrings().data() == output.AsStrings().data()) {
    Invalid("STRING output must not overlap input.");
  }
}

void ConvertNumeric(const Tensor &data, Tensor &output, int64_t count) {
  constexpr int64_t kDirectIdentityCopyLimit = 128 * 1024;
  if (data.data_type == output.data_type && count < kDirectIdentityCopyLimit) {
    if (data.size_bytes() != 0) {
      std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    }
    return;
  }
  cpu_compute::CastConvert(
      data.bytes(), static_cast<cpu_compute::DataType>(data.data_type), output.mutable_bytes(),
      static_cast<cpu_compute::DataType>(output.data_type), static_cast<std::size_t>(count));
}

bool Complex(int32_t type) { return type == DataType::COMPLEX64 || type == DataType::COMPLEX128; }

std::size_t ComplexElementSize(int32_t type) {
  return type == DataType::COMPLEX64 ? 2 * sizeof(float) : 2 * sizeof(double);
}

void ConvertComplexRange(const Tensor &data, Tensor &output, std::size_t begin, std::size_t end) {
  constexpr std::size_t kBlockSize = 256;
  std::array<double, kBlockSize> values;
  std::array<float, kBlockSize> float_values;
  const bool from_complex = Complex(data.data_type);
  const bool to_complex = Complex(output.data_type);
  const std::size_t input_width =
      from_complex
          ? ComplexElementSize(data.data_type)
          : cpu_compute::CastElementSize(static_cast<cpu_compute::DataType>(data.data_type));
  const std::size_t output_width =
      to_complex
          ? ComplexElementSize(output.data_type)
          : cpu_compute::CastElementSize(static_cast<cpu_compute::DataType>(output.data_type));
  for (std::size_t block_begin = begin; block_begin < end; block_begin += kBlockSize) {
    const std::size_t block_count = std::min(kBlockSize, end - block_begin);
    if (from_complex && to_complex) {
      const std::size_t input_part_width = input_width / 2;
      const std::size_t output_part_width = output_width / 2;
      for (std::size_t i = 0; i < block_count; ++i) {
        const auto *source = data.bytes() + (block_begin + i) * input_width;
        auto *destination = output.mutable_bytes() + (block_begin + i) * output_width;
        for (std::size_t part = 0; part < 2; ++part) {
          double value;
          if (data.data_type == DataType::COMPLEX64) {
            float source_part;
            std::memcpy(&source_part, source + part * input_part_width, input_part_width);
            value = source_part;
          } else {
            std::memcpy(&value, source + part * input_part_width, input_part_width);
          }
          if (output.data_type == DataType::COMPLEX64) {
            const float output_part = static_cast<float>(value);
            std::memcpy(destination + part * output_part_width, &output_part, output_part_width);
          } else {
            std::memcpy(destination + part * output_part_width, &value, output_part_width);
          }
        }
      }
      continue;
    }
    if (!from_complex && to_complex) {
      const std::size_t part_width = output_width / 2;
      if (output.data_type == DataType::COMPLEX64) {
        cpu_compute::CastConvert(data.bytes() + block_begin * input_width,
                                 static_cast<cpu_compute::DataType>(data.data_type),
                                 float_values.data(), cpu_compute::DataType::FLOAT, block_count);
        for (std::size_t i = 0; i < block_count; ++i) {
          auto *destination = output.mutable_bytes() + (block_begin + i) * output_width;
          std::memcpy(destination, &float_values[i], part_width);
          std::memset(destination + part_width, 0, part_width);
        }
      } else {
        cpu_compute::CastConvert(data.bytes() + block_begin * input_width,
                                 static_cast<cpu_compute::DataType>(data.data_type), values.data(),
                                 cpu_compute::DataType::DOUBLE, block_count);
        for (std::size_t i = 0; i < block_count; ++i) {
          auto *destination = output.mutable_bytes() + (block_begin + i) * output_width;
          std::memcpy(destination, &values[i], part_width);
          std::memset(destination + part_width, 0, part_width);
        }
      }
      continue;
    }
    if (from_complex) {
      const std::size_t part_width = input_width / 2;
      if (output.data_type == DataType::BOOL) {
        for (std::size_t i = 0; i < block_count; ++i) {
          const auto *source = data.bytes() + (block_begin + i) * input_width;
          bool nonzero;
          if (data.data_type == DataType::COMPLEX64) {
            float parts[2];
            std::memcpy(parts, source, input_width);
            nonzero = parts[0] != 0.0F || parts[1] != 0.0F;
          } else {
            double parts[2];
            std::memcpy(parts, source, input_width);
            nonzero = parts[0] != 0.0 || parts[1] != 0.0;
          }
          output.mutable_bytes()[block_begin + i] = static_cast<std::uint8_t>(nonzero);
        }
        continue;
      }
      for (std::size_t i = 0; i < block_count; ++i) {
        const auto *source = data.bytes() + (block_begin + i) * input_width;
        if (data.data_type == DataType::COMPLEX64) {
          float real;
          std::memcpy(&real, source, part_width);
          values[i] = real;
        } else {
          std::memcpy(&values[i], source, part_width);
        }
      }
    } else {
      cpu_compute::CastConvert(data.bytes() + block_begin * input_width,
                               static_cast<cpu_compute::DataType>(data.data_type), values.data(),
                               cpu_compute::DataType::DOUBLE, block_count);
    }

    cpu_compute::CastConvert(values.data(), cpu_compute::DataType::DOUBLE,
                             output.mutable_bytes() + block_begin * output_width,
                             static_cast<cpu_compute::DataType>(output.data_type), block_count);
  }
}

void ConvertComplex(const Tensor &data, Tensor &output, int64_t count) {
  if (data.data_type == output.data_type) {
    if (data.size_bytes() != 0) {
      std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    }
    return;
  }
  const bool from_complex = Complex(data.data_type);
  const bool to_complex = Complex(output.data_type);
  if ((!from_complex && !Numeric(data.data_type)) || (!to_complex && !Numeric(output.data_type))) {
    Invalid("unsupported complex cast pair.");
  }
  cpu_compute::ExecuteRanges(count, 1.0, 256, [&](int64_t begin, int64_t end) {
    ConvertComplexRange(data, output, static_cast<std::size_t>(begin),
                        static_cast<std::size_t>(end));
  });
}

void Convert(const rt_ns::KernelContext &ctx, const Tensor &data, int32_t to, bool saturate,
             Tensor &output, const CastPlan &plan) {
  ValidateOutput(data, to, output, plan);
  if (Complex(data.data_type) || Complex(to)) {
    ConvertComplex(data, output, plan.count);
    return;
  }
  if (Numeric(data.data_type) && Numeric(to)) {
    ConvertNumeric(data, output, plan.count);
    return;
  }
  // Call the concrete builtin, never its registration: replacing Cast globally
  // cannot recurse. Stage compatibility results so parser/pair errors cannot
  // partially modify preallocated output, and builtin typed stores stay aligned.
  Tensor normalized;
  const Tensor *source = &data;
  if (data.data_type == DataType::STRING) {
    normalized = Tensor::FromStrings(data.name, data.shape, data.AsStrings());
    source = &normalized;
  } else if (data.size_bytes() != 0 &&
             reinterpret_cast<std::uintptr_t>(data.bytes()) % alignof(double) != 0) {
    normalized = Tensor(data.name, data.data_type, data.shape,
                        std::vector<std::uint8_t>(data.bytes(), data.bytes() + data.size_bytes()));
    source = &normalized;
  }
  BuiltinCast builtin(ctx);
  if (data.data_type == DataType::STRING && Numeric(to)) {
    // Preserve the builtin parser (including prefix/locale behavior), but do
    // not inherit its undefined NaN/out-of-range floating-to-integer casts.
    const Tensor parsed = builtin(*source, DataType::DOUBLE, saturate);
    ConvertNumeric(parsed, output, plan.count);
    return;
  }
  Tensor converted = builtin(*source, to, saturate);
  if (to == DataType::STRING) {
    output.AsStrings() = std::move(converted.string_data);
  } else if (plan.output_bytes != 0) {
    std::memcpy(output.mutable_bytes(), converted.bytes(), plan.output_bytes);
  }
}

} // namespace

CastKernel::CastKernel(const rt_ns::KernelContext &ctx) : KernelBase(ctx) {}

CastKernel::CastKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node, const rt_ns::KernelContext &ctx)
    : KernelBase(ctx) {
  bool has_to = false;
  bool has_saturate = false;
  for (const auto &attribute : node.attribute()) {
    if (attribute.name() != "to" && attribute.name() != "saturate") {
      continue;
    }
    if (attribute.type() != ONNX_LIGHT_NAMESPACE::AttributeProto::INT) {
      Invalid("to and saturate must be INT attributes.");
    }
    const int64_t value = attribute.i();
    if (attribute.name() == "to") {
      if (has_to || value < 0 || value > std::numeric_limits<int32_t>::max() ||
          !Supported(static_cast<int32_t>(value))) {
        Invalid("invalid or duplicate to attribute.");
      }
      has_to = true;
      to_ = static_cast<int32_t>(value);
    } else {
      if (has_saturate || (value != 0 && value != 1)) {
        Invalid("saturate must be 0 or 1 and cannot be duplicated.");
      }
      has_saturate = true;
      saturate_ = value != 0;
    }
  }
  if (!has_to) {
    Invalid("required to attribute is missing.");
  }
  set_node(node);
}

Tensor CastKernel::operator()(const Tensor &data, int32_t to, rt_ns::RuntimeContext *rt) const {
  const RuntimeExecutionScope execution_scope;
  return (*this)(data, to, true, rt);
}

Tensor CastKernel::operator()(const Tensor &data, int32_t to, bool saturate,
                              rt_ns::RuntimeContext *rt) const {
  const RuntimeExecutionScope execution_scope;
  const CastPlan plan = PrepareCast(data, to);
  Tensor output = rt ? rt->MakeOutputTensor(0, to, data.shape, plan.output_bytes)
                     : rt_ns::MakeOutputTensor(to, data.shape, plan.output_bytes, ctx_.allocator);
  if (to == rt_ns::DataType::STRING) {
    output.string_data.resize(static_cast<std::size_t>(plan.count));
  }
  Convert(ctx_, data, to, saturate, output, plan);
  if (rt != nullptr && rt->kernel_usage_enabled()) {
    rt->RecordKernelUsage(
        Complex(data.data_type) || Complex(to) ? "Cast.complex"
        : Numeric(data.data_type) && Numeric(to)
            ? cpu_compute::CastConversionPath(static_cast<cpu_compute::DataType>(data.data_type),
                                              static_cast<cpu_compute::DataType>(to),
                                              static_cast<std::size_t>(plan.count))
            : "Cast.builtin");
  }
  return output;
}

void CastKernel::operator()(const Tensor &data, int32_t to, Tensor &output) const {
  const RuntimeExecutionScope execution_scope;
  (*this)(data, to, true, output);
}

void CastKernel::operator()(const Tensor &data, int32_t to, bool saturate, Tensor &output) const {
  const RuntimeExecutionScope execution_scope;
  const CastPlan plan = PrepareCast(data, to);
  Convert(ctx_, data, to, saturate, output, plan);
}

void CastKernel::Run(rt_ns::RuntimeContext &rt) {
  const RuntimeExecutionScope execution_scope;
  rt.RecordKernelUsage(kName);
  if (node_ == nullptr) {
    Invalid("Run requires a node.");
  }
  const auto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &data = rt_ns::GetInput(node, 0, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(data, to_, saturate_, &rt), rt);
}

void RegisterCastKernel() {
  using rt_ns::DataType;
  rt_ns::NodeKernelFn factory =
      [](const ONNX_LIGHT_NAMESPACE::NodeProto &node, rt_ns::RuntimeContext &,
         const rt_ns::KernelContext &kernel_context) -> std::unique_ptr<rt_ns::KernelBase> {
    return std::make_unique<CastKernel>(node, kernel_context);
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "Cast";
  info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
  info.kernel_name = CastKernel::kName;
  info.types = {DataType::FLOAT,          DataType::DOUBLE,     DataType::FLOAT16,
                DataType::BFLOAT16,       DataType::INT8,       DataType::UINT8,
                DataType::INT16,          DataType::UINT16,     DataType::INT32,
                DataType::UINT32,         DataType::INT64,      DataType::UINT64,
                DataType::BOOL,           DataType::STRING,     DataType::FLOAT8E4M3FN,
                DataType::FLOAT8E4M3FNUZ, DataType::FLOAT8E5M2, DataType::FLOAT8E5M2FNUZ,
                DataType::FLOAT8E8M0,     DataType::INT4,       DataType::UINT4,
                DataType::INT2,           DataType::UINT2,      DataType::FLOAT4E2M1,
                DataType::FLOAT6E2M3,     DataType::FLOAT6E3M2, DataType::COMPLEX64,
                DataType::COMPLEX128};
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
