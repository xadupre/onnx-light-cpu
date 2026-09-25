// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/matmul_nbits_kernel.h"

#include "onnx_light_cpu/impl/checked_arithmetic.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using rt_ns::Tensor;
using RuntimeDataType = rt_ns::DataType;

constexpr const char *kParallelThresholdOutputs = "parallel.threshold_outputs";
constexpr const char *kTargetBlockOutputs = "parallel.target_block_outputs";
constexpr const char *kMaxParticipants = "parallel.max_participants";

rt_ns::KernelTuningKey MakeTuningKey(RuntimeDataType data_type, std::int64_t bits) {
  return {"onnx_light_cpu",
          "MatMulNBits",
          "packed_int" + std::to_string(bits),
          static_cast<std::int32_t>(data_type),
          sym_ns::Device::kCPU,
          MatMulNBitsKernel::kTuningAbi};
}

void ValidateTuning(const rt_ns::KernelTuningParameters &parameters) {
  for (const char *name : {kParallelThresholdOutputs, kTargetBlockOutputs, kMaxParticipants}) {
    const std::int64_t value = parameters.Get<std::int64_t>(name);
    if (value <= 0 || static_cast<std::uint64_t>(value) >
                          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      throw std::invalid_argument(std::string("MatMulNBits ") + name +
                                  " must be positive and representable by size_t.");
    }
  }
}

MatMulNBitsAttributes ParseAttributes(const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  MatMulNBitsAttributes attributes;
  attributes.k = rt_ns::GetAttributeIntOrDefault(node, "K", 0);
  attributes.n = rt_ns::GetAttributeIntOrDefault(node, "N", 0);
  attributes.bits = rt_ns::GetAttributeIntOrDefault(node, "bits", 4);
  attributes.block_size = rt_ns::GetAttributeIntOrDefault(node, "block_size", 0);
  attributes.accuracy_level = rt_ns::GetAttributeIntOrDefault(node, "accuracy_level", 0);
  attributes.weight_prepacked = rt_ns::GetAttributeIntOrDefault(node, "weight_prepacked", 0);
  if (attributes.k <= 0 || attributes.n <= 0) {
    throw std::invalid_argument("onnx_light_cpu::MatMulNBits: K and N must be positive.");
  }
  if ((attributes.bits != 2 && attributes.bits != 4 && attributes.bits != 8) ||
      attributes.block_size != 32 ||
      (attributes.accuracy_level != 0 && attributes.accuracy_level != 4) ||
      attributes.weight_prepacked != 0) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulNBits: the kernel requires bits=2, 4, or 8, block_size=32, "
        "accuracy_level 0 or 4, and weight_prepacked=0.");
  }
  return attributes;
}

std::size_t TensorElementCount(const Tensor &tensor, const char *label) {
  return static_cast<std::size_t>(
      tensor.shape.product(0, tensor.shape.size(), std::string("MatMulNBits ") + label));
}

void RequireShape(const Tensor &tensor, std::initializer_list<std::int64_t> expected,
                  const char *label) {
  if (tensor.shape.size() != expected.size() ||
      !std::equal(tensor.shape.begin(), tensor.shape.end(), expected.begin())) {
    throw std::invalid_argument(std::string("onnx_light_cpu::MatMulNBits: invalid ") + label +
                                " shape.");
  }
}

} // namespace

struct MatMulNBitsKernel::PreparedInt4Plan {
  PreparedInt4Plan(const Tensor &a, const Tensor &b, const Tensor &scales,
                   const MatMulNBitsAttributes &attributes,
                   const MatMulNBitsExecutionTuning &tuning)
      : rows(TensorElementCount(a, "A") / static_cast<std::size_t>(attributes.k)),
        k(static_cast<std::size_t>(attributes.k)), n(static_cast<std::size_t>(attributes.n)),
        blocks((k + 31) / 32), source_b(b), source_scales(scales),
        vnni_weights(CheckedMultiply(k, n, "MatMulNBits", "prepared weight count")),
        vnni_scales(CheckedMultiply(blocks, n, "MatMulNBits", "prepared scale count")),
        max_participants(tuning.max_participants) {
    const auto *packed_values = source_b.bytes();
    const auto *scale_values = reinterpret_cast<const float *>(source_scales.bytes());
    const std::size_t column_groups = n / 16;
    for (std::size_t block = 0; block < blocks; ++block) {
      for (std::size_t group = 0; group < column_groups; ++group) {
        for (std::size_t pair = 0; pair < 16; ++pair) {
          for (std::size_t lane = 0; lane < 16; ++lane) {
            const std::size_t column = group * 16 + lane;
            const std::uint8_t byte = packed_values[(column * blocks + block) * 16 + pair];
            const std::size_t target = (((group * blocks + block) * 16 + pair) * 32) + lane * 2;
            vnni_weights[target] = static_cast<std::int8_t>((byte & 15) - 8);
            vnni_weights[target + 1] = static_cast<std::int8_t>((byte >> 4) - 8);
          }
        }
      }
      for (std::size_t column = 0; column < n; ++column) {
        const std::size_t group = column / 16;
        const std::size_t lane = column % 16;
        vnni_scales[(group * blocks + block) * 16 + lane] = scale_values[column * blocks + block];
      }
    }
  }

  bool Matches(const Tensor &a, const Tensor &b, const Tensor &scales) const {
    return TensorElementCount(a, "A") == rows * k &&
           ((b.bytes() == source_b.bytes() && scales.bytes() == source_scales.bytes()) ||
            (std::equal(source_b.bytes(), source_b.bytes() + n * blocks * 16, b.bytes()) &&
             std::equal(reinterpret_cast<const float *>(source_scales.bytes()),
                        reinterpret_cast<const float *>(source_scales.bytes()) + n * blocks,
                        reinterpret_cast<const float *>(scales.bytes()))));
  }

  std::size_t rows;
  std::size_t k;
  std::size_t n;
  std::size_t blocks;
  Tensor source_b;
  Tensor source_scales;
  std::vector<std::int8_t> vnni_weights;
  std::vector<float> vnni_scales;
  std::int64_t max_participants;
};

MatMulNBitsKernel::MatMulNBitsKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                     const rt_ns::KernelContext &ctx)
    : KernelBase(ctx), attributes_(ParseAttributes(node)) {
  set_node(node);
}

MatMulNBitsKernel::~MatMulNBitsKernel() = default;

void MatMulNBitsKernel::RegisterTuningSchemas() {
  static std::once_flag once;
  std::call_once(once, [] {
    for (RuntimeDataType data_type :
         {RuntimeDataType::FLOAT, RuntimeDataType::FLOAT16, RuntimeDataType::BFLOAT16}) {
      for (std::int64_t bits : {2, 4, 8}) {
        rt_ns::RegisterKernelTuningSchema(rt_ns::KernelTuningSchema(
            {MakeTuningKey(data_type, bits),
             {{kParallelThresholdOutputs,
               static_cast<std::int64_t>(
                   kDefaultMatMulNBitsExecutionTuning.parallel_threshold_outputs)},
              {kTargetBlockOutputs,
               static_cast<std::int64_t>(kDefaultMatMulNBitsExecutionTuning.target_block_outputs)},
              {kMaxParticipants, kDefaultMatMulNBitsExecutionTuning.max_participants}}},
            ValidateTuning));
      }
    }
  });
}

rt_ns::KernelTuningKey MatMulNBitsKernel::TuningKey(std::int32_t element_type) const {
  const auto data_type = static_cast<RuntimeDataType>(element_type);
  return data_type == RuntimeDataType::FLOAT || data_type == RuntimeDataType::FLOAT16 ||
                 data_type == RuntimeDataType::BFLOAT16
             ? MakeTuningKey(data_type, attributes_.bits)
             : rt_ns::KernelTuningKey{};
}

void MatMulNBitsKernel::Configure(const rt_ns::KernelTuningParameters &parameters) {
  constexpr std::array kDataTypes{RuntimeDataType::FLOAT, RuntimeDataType::FLOAT16,
                                  RuntimeDataType::BFLOAT16};
  const bool key_matches =
      std::any_of(kDataTypes.begin(), kDataTypes.end(), [&](RuntimeDataType data_type) {
        return parameters.key == MakeTuningKey(data_type, attributes_.bits);
      });
  if (!key_matches) {
    throw std::invalid_argument("MatMulNBits tuning parameters have an incompatible key.");
  }
  ValidateTuning(parameters);
  tuning_ = {
      static_cast<std::size_t>(parameters.Get<std::int64_t>(kParallelThresholdOutputs)),
      static_cast<std::size_t>(parameters.Get<std::int64_t>(kTargetBlockOutputs)),
      parameters.Get<std::int64_t>(kMaxParticipants),
  };
  std::lock_guard<std::mutex> lock(prepared_int4_mutex_);
  prepared_int4_.reset();
}

Tensor MatMulNBitsKernel::operator()(const Tensor &a, const Tensor &b, const Tensor &scales,
                                     const Tensor *bias, rt_ns::RuntimeContext *rt) const {
  const auto data_type = static_cast<RuntimeDataType>(a.data_type);
  if ((data_type != RuntimeDataType::FLOAT && data_type != RuntimeDataType::FLOAT16 &&
       data_type != RuntimeDataType::BFLOAT16) ||
      scales.data_type != a.data_type || (bias != nullptr && bias->data_type != a.data_type) ||
      b.data_type != static_cast<std::int32_t>(RuntimeDataType::UINT8)) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulNBits: A, scales, bias, and Y must have matching FLOAT, FLOAT16, "
        "or BFLOAT16 types, and B must be UINT8.");
  }
  if (a.shape.empty() || a.shape.back() != attributes_.k) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulNBits: A must have positive rank and final dimension K.");
  }
  const std::size_t k = CheckedDimension(attributes_.k, "MatMulNBits", "K");
  const std::size_t n = CheckedDimension(attributes_.n, "MatMulNBits", "N");
  const std::size_t block_size =
      CheckedDimension(attributes_.block_size, "MatMulNBits", "block_size");
  const std::size_t k_blocks =
      CheckedAdd(k, block_size - 1, "MatMulNBits", "K blocks") / block_size;
  const std::size_t blob_size =
      CheckedMultiply(block_size, static_cast<std::size_t>(attributes_.bits), "MatMulNBits",
                      "packed block bits") /
      8;
  const std::size_t rows = TensorElementCount(a, "A") / k;
  RequireShape(
      b, {attributes_.n, static_cast<std::int64_t>(k_blocks), static_cast<std::int64_t>(blob_size)},
      "B");
  const std::size_t expected_scales =
      CheckedMultiply(n, k_blocks, "MatMulNBits", "scale element count");
  if (!((scales.shape.size() == 1 && TensorElementCount(scales, "scales") == expected_scales) ||
        (scales.shape.size() == 2 && scales.shape[0] == attributes_.n &&
         scales.shape[1] == static_cast<std::int64_t>(k_blocks)))) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulNBits: scales must have shape [N, ceil(K/block_size)] or be flat.");
  }
  if (bias != nullptr) {
    RequireShape(*bias, {attributes_.n}, "bias");
  }
  rt_ns::Shape output_shape = a.shape;
  output_shape.back() = attributes_.n;
  const std::size_t output_count = CheckedMultiply(rows, n, "MatMulNBits", "output element count");
  const std::size_t output_bytes = CheckedByteSize(output_count, rt_ns::ElementSize(a.data_type),
                                                   "MatMulNBits", "output byte size");
  Tensor y = rt != nullptr
                 ? rt->MakeOutputTensor(0, a.data_type, output_shape, output_bytes)
                 : rt_ns::MakeOutputTensor(a.data_type, output_shape, output_bytes, nullptr);
  std::shared_ptr<const PreparedInt4Plan> prepared;
  if (attributes_.bits == 4 && attributes_.accuracy_level == 4 &&
      MatMulNBitsAccuracy4Float32Available() && data_type == RuntimeDataType::FLOAT &&
      n % 16 == 0 && k % 32 == 0) {
    std::lock_guard<std::mutex> lock(prepared_int4_mutex_);
    if (prepared_int4_ == nullptr || !prepared_int4_->Matches(a, b, scales)) {
      prepared_int4_ = std::make_shared<PreparedInt4Plan>(a, b, scales, attributes_, tuning_);
    }
    prepared = prepared_int4_;
  }
  if (prepared != nullptr) {
    MatMulNBitsAccuracy4Float32(
        reinterpret_cast<const float *>(a.bytes()), prepared->vnni_weights.data(),
        prepared->vnni_scales.data(),
        bias == nullptr ? nullptr : reinterpret_cast<const float *>(bias->bytes()),
        reinterpret_cast<float *>(y.mutable_bytes()), rows, k, n, prepared->max_participants);
    return y;
  }
  MatMulNBits(a.bytes(), b.bytes(), scales.bytes(), bias == nullptr ? nullptr : bias->bytes(),
              y.mutable_bytes(), static_cast<onnx_light_cpu::DataType>(a.data_type), rows, k, n,
              static_cast<std::size_t>(attributes_.bits), block_size, tuning_);
  return y;
}

void MatMulNBitsKernel::Run(rt_ns::RuntimeContext &rt) {
  rt.RecordKernelUsage(kName);
  const auto &node = *node_;
  if (node.input_size() < 3 || node.input_size() > 6) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulNBits: expected A, B, scales and optional input slots.");
  }
  rt_ns::RequireOutputCount(node, 1);
  if ((node.input_size() > 3 && !node.input(3).empty()) ||
      (node.input_size() > 4 && !node.input(4).empty())) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulNBits: zero_points and g_idx are not supported by the foundation "
        "kernel.");
  }
  const Tensor *bias = rt_ns::GetOptionalInput(node, 5, rt.tensors());
  rt_ns::SetOutput(node, 0,
                   (*this)(rt_ns::GetInput(node, 0, rt.tensors()),
                           rt_ns::GetInput(node, 1, rt.tensors()),
                           rt_ns::GetInput(node, 2, rt.tensors()), bias, &rt),
                   rt);
}

void RegisterMatMulNBitsKernel() {
  MatMulNBitsKernel::RegisterTuningSchemas();
  rt_ns::NodeKernelFn factory = [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                   rt_ns::RuntimeContext &rt) {
    return std::make_unique<MatMulNBitsKernel>(node, rt.kernel_ctx());
  };
  KernelRegistration info;
  info.domain = kMicrosoftDomain;
  info.op_type = "MatMulNBits";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = MatMulNBitsKernel::kName;
  info.types = {RuntimeDataType::FLOAT, RuntimeDataType::FLOAT16, RuntimeDataType::BFLOAT16};
  info.since_version = 1;
  info.until_version = 1;
  RegisterKernel(std::move(info), std::move(factory));
}

} // namespace onnx_light_cpu
