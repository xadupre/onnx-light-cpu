// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/attention/rotary_embedding_kernel.h"

#include "onnx_light_cpu/impl/checked_arithmetic.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/math/normalization_helpers.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/node_helpers.h"

#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
using rt_ns::DataType;
using rt_ns::Tensor;

[[noreturn]] void Invalid(const char *message) {
  throw std::invalid_argument(std::string(RotaryEmbeddingKernel::kName) + ": " + message);
}

std::size_t Count(const Tensor &tensor) {
  const auto count = CheckedShapeIndexProduct(tensor.shape, RotaryEmbeddingKernel::kName, "shape");
  const auto bytes = CheckedByteSize(static_cast<std::size_t>(count), tensor.element_size(),
                                     RotaryEmbeddingKernel::kName, "buffer");
  if (tensor.size_bytes() != bytes || (bytes && tensor.bytes() == nullptr)) {
    Invalid("buffer size does not match shape.");
  }
  return static_cast<std::size_t>(count);
}

std::int64_t Position(const Tensor &positions, std::size_t index) {
  std::int64_t value;
  std::memcpy(&value, positions.bytes() + index * sizeof(value), sizeof(value));
  return value;
}

template <DataType Type>
void Rotate(const Tensor &input, const Tensor &cos, const Tensor &sin, Tensor &output,
            const Tensor *positions, std::size_t batch, std::size_t seq, std::size_t heads,
            std::size_t head_size, std::size_t rotate_dim, bool interleaved) {
  using Traits = normalization::TypeTraits<Type>;
  using Storage = normalization::StorageType<Type>;
  const auto *x = input.bytes();
  const auto *c = cos.bytes();
  const auto *s = sin.bytes();
  auto *y = output.mutable_bytes();
  const std::size_t half = rotate_dim / 2;
  const bool rank4 = input.shape.size() == 4;
  ExecuteRanges(
      static_cast<std::int64_t>(batch * seq), static_cast<double>(heads * head_size),
      [&](std::int64_t begin, std::int64_t end) {
        for (auto row = static_cast<std::size_t>(begin); row < static_cast<std::size_t>(end);
             ++row) {
          const std::size_t b = row / seq;
          const std::size_t t = row % seq;
          const std::size_t cache_row = positions == nullptr ? row
                                        : positions->shape.size() <= 1
                                            ? static_cast<std::size_t>(Position(*positions, 0)) + t
                                            : static_cast<std::size_t>(Position(*positions, row));
          for (std::size_t h = 0; h < heads; ++h) {
            const std::size_t base =
                rank4 ? ((b * heads + h) * seq + t) * head_size : (row * heads + h) * head_size;
            const std::size_t cache_base = cache_row * half;
            for (std::size_t i = 0; i < half; ++i) {
              const std::size_t first = base + (interleaved ? i * 2 : i);
              const std::size_t second = first + (interleaved ? 1 : half);
              Storage a, z, cosine, sine;
              std::memcpy(&a, x + first * sizeof(Storage), sizeof(Storage));
              std::memcpy(&z, x + second * sizeof(Storage), sizeof(Storage));
              std::memcpy(&cosine, c + (cache_base + i) * sizeof(Storage), sizeof(Storage));
              std::memcpy(&sine, s + (cache_base + i) * sizeof(Storage), sizeof(Storage));
              const float v1 = Traits::Load(&a, 0);
              const float v2 = Traits::Load(&z, 0);
              const float cv = Traits::Load(&cosine, 0);
              const float sv = Traits::Load(&sine, 0);
              Storage real, imag;
              Traits::Store(&real, 0, cv * v1 - sv * v2);
              Traits::Store(&imag, 0, sv * v1 + cv * v2);
              std::memcpy(y + first * sizeof(Storage), &real, sizeof(Storage));
              std::memcpy(y + second * sizeof(Storage), &imag, sizeof(Storage));
            }
          }
        }
      });
}

} // namespace

Tensor RotaryEmbeddingKernel::operator()(const Tensor &input, const Tensor &cos, const Tensor &sin,
                                         const Tensor *positions, bool interleaved,
                                         std::int64_t rotary_dim, std::int64_t num_heads,
                                         bool microsoft, rt_ns::RuntimeContext *rt) const {
  const auto type = static_cast<DataType>(input.data_type);
  if (type != DataType::FLOAT && type != DataType::FLOAT16 && type != DataType::BFLOAT16) {
    Invalid("input must be FLOAT, FLOAT16, or BFLOAT16.");
  }
  if (cos.data_type != input.data_type || sin.data_type != input.data_type) {
    Invalid("input and caches must have the same type.");
  }
  if (input.shape.size() != 3 && input.shape.size() != 4) {
    Invalid("input must have rank 3 or 4.");
  }
  const auto batch = CheckedDimension(input.shape[0], kName, "batch");
  const auto seq =
      CheckedDimension(input.shape[input.shape.size() == 4 ? 2 : 1], kName, "sequence");
  std::size_t heads;
  std::size_t head_size;
  if (input.shape.size() == 3) {
    const auto hidden = CheckedDimension(input.shape[2], kName, "hidden size");
    if (num_heads <= 0 && microsoft && rotary_dim == 0 && cos.shape.size() == 2 &&
        cos.shape[1] > 0 && static_cast<std::uint64_t>(cos.shape[1]) <= hidden / 2) {
      const auto inferred_head = static_cast<std::size_t>(cos.shape[1]) * 2;
      if (hidden % inferred_head != 0) {
        Invalid("hidden size must be divisible by inferred head size.");
      }
      num_heads = static_cast<std::int64_t>(hidden / inferred_head);
    }
    if (num_heads <= 0) {
      Invalid("num_heads must be positive for rank-3 input.");
    }
    heads = static_cast<std::size_t>(num_heads);
    if (hidden % heads != 0) {
      Invalid("hidden size must be divisible by num_heads.");
    }
    head_size = hidden / heads;
  } else {
    heads = CheckedDimension(input.shape[1], kName, "heads");
    head_size = CheckedDimension(input.shape[3], kName, "head size");
  }
  if (rotary_dim < 0 || (rotary_dim != 0 && static_cast<std::uint64_t>(rotary_dim) > head_size)) {
    Invalid("rotary_embedding_dim must be between 0 and head size.");
  }
  const std::size_t rotate_dim = rotary_dim == 0 ? head_size : static_cast<std::size_t>(rotary_dim);
  if (rotate_dim == 0 || rotate_dim % 2 != 0) {
    Invalid("rotary dimension must be positive and even.");
  }
  const std::size_t rows = CheckedProduct({batch, seq}, kName, "rows");
  const std::size_t elements = CheckedProduct({rows, heads, head_size}, kName, "input elements");
  if (rows > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    Invalid("row count exceeds int64_t.");
  }
  if (Count(input) != elements) {
    Invalid("input shape mismatch.");
  }
  if (positions != nullptr) {
    const bool offset = microsoft && (positions->shape.empty() ||
                                      (positions->shape.size() == 1 && positions->shape[0] == 1));
    if (positions->data_type != DataType::INT64 ||
        (!offset && (positions->shape.size() != 2 || positions->shape[0] != input.shape[0] ||
                     positions->shape[1] != static_cast<std::int64_t>(seq))) ||
        cos.shape.size() != 2 || sin.shape.size() != 2) {
      Invalid("position_ids must be INT64 (batch, sequence), or a Microsoft scalar offset.");
    }
    if (Count(*positions) != (offset ? 1 : rows)) {
      Invalid("position_ids shape mismatch.");
    }
  } else if (cos.shape.size() != 3 || sin.shape.size() != 3 || cos.shape[0] != input.shape[0] ||
             sin.shape[0] != input.shape[0] || cos.shape[1] != static_cast<std::int64_t>(seq) ||
             sin.shape[1] != static_cast<std::int64_t>(seq)) {
    Invalid("without position_ids caches must have shape (batch, sequence, rotary_dim/2).");
  }
  if (cos.shape != sin.shape || cos.shape.back() != static_cast<std::int64_t>(rotate_dim / 2)) {
    Invalid("cos_cache and sin_cache must have matching rotary dimensions.");
  }
  const std::size_t cache_count = Count(cos);
  if (Count(sin) != cache_count) {
    Invalid("cache sizes differ.");
  }
  if (positions != nullptr) {
    const auto max_positions = CheckedDimension(cos.shape[0], kName, "cache length");
    for (std::size_t i = 0; i < (positions->shape.size() <= 1 ? 1 : rows); ++i) {
      const auto index = Position(*positions, i);
      if (index < 0 || static_cast<std::uint64_t>(index) >= max_positions ||
          (positions->shape.size() <= 1 && seq > max_positions - static_cast<std::size_t>(index))) {
        Invalid("position_ids contains an out-of-range index.");
      }
    }
  }
  Tensor output = normalization::AllocateOutput(input.data_type, input.shape, 0, rt);
  if (elements == 0) {
    return output;
  }
  std::memcpy(output.mutable_bytes(), input.bytes(), input.size_bytes());
  if (type == DataType::FLOAT) {
    Rotate<DataType::FLOAT>(input, cos, sin, output, positions, batch, seq, heads, head_size,
                            rotate_dim, interleaved);
  } else if (type == DataType::FLOAT16) {
    Rotate<DataType::FLOAT16>(input, cos, sin, output, positions, batch, seq, heads, head_size,
                              rotate_dim, interleaved);
  } else {
    Rotate<DataType::BFLOAT16>(input, cos, sin, output, positions, batch, seq, heads, head_size,
                               rotate_dim, interleaved);
  }
  return output;
}

void RotaryEmbeddingKernel::Run(rt_ns::RuntimeContext &rt) {
  rt.RecordKernelUsage(kName);
  const auto &node = *node_;
  const bool microsoft = node.domain() == kMicrosoftDomain;
  if ((!microsoft && node.input_size() != 3 && node.input_size() != 4) ||
      (microsoft && node.input_size() != 4) || node.output_size() != 1) {
    Invalid("expected three inputs, optional position_ids, and one output.");
  }
  if (microsoft && rt_ns::GetAttributeIntOrDefault(node, "is_packed_batching", 0) != 0) {
    Invalid("is_packed_batching is not supported.");
  }
  const Tensor *positions = microsoft ? &rt_ns::GetInput(node, 1, rt.tensors())
                            : node.input_size() == 4 && !node.input(3).empty()
                                ? &rt_ns::GetInput(node, 3, rt.tensors())
                                : nullptr;
  const auto interleaved = rt_ns::GetAttributeIntOrDefault(node, "interleaved", 0);
  if (interleaved != 0 && interleaved != 1) {
    Invalid("interleaved must be 0 or 1.");
  }
  rt_ns::SetOutput(node, 0,
                   (*this)(rt_ns::GetInput(node, 0, rt.tensors()),
                           rt_ns::GetInput(node, microsoft ? 2 : 1, rt.tensors()),
                           rt_ns::GetInput(node, microsoft ? 3 : 2, rt.tensors()), positions,
                           interleaved != 0,
                           rt_ns::GetAttributeIntOrDefault(node, "rotary_embedding_dim", 0),
                           rt_ns::GetAttributeIntOrDefault(node, "num_heads", 0), microsoft, &rt),
                   rt);
}

void RegisterRotaryEmbeddingKernels() {
  for (const auto &[domain, version] :
       {std::pair{"", std::int64_t{23}}, std::pair{kMicrosoftDomain, std::int64_t{1}}}) {
    rt_ns::NodeKernelFn factory = [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                     rt_ns::RuntimeContext &rt) {
      auto kernel = std::make_unique<RotaryEmbeddingKernel>(rt.kernel_ctx());
      kernel->set_node(node);
      return kernel;
    };
    KernelRegistration info;
    info.domain = domain;
    info.op_type = "RotaryEmbedding";
    info.device = ONNX_LIGHT_NAMESPACE::core::symbolic::Device::kCPU;
    info.kernel_name = RotaryEmbeddingKernel::kName;
    info.types = {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16};
    info.since_version = version;
    RegisterKernel(std::move(info), std::move(factory));
  }
}

} // namespace onnx_light_cpu
