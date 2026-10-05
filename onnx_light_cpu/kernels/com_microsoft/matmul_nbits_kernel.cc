// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/com_microsoft/matmul_nbits_kernel.h"

#include "onnx_light_cpu/impl/checked_arithmetic.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/schemas/com_microsoft/op_schema.h"

#include "onnx_core/compute/prepared_execution.h"
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
#include <optional>
#include <sstream>
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
  PreparedInt4Plan(const Tensor &b, const Tensor &scales, const MatMulNBitsAttributes &attributes,
                   const MatMulNBitsExecutionTuning &tuning)
      : k(static_cast<std::size_t>(attributes.k)), n(static_cast<std::size_t>(attributes.n)),
        blocks((k + 31) / 32), data_type(static_cast<RuntimeDataType>(scales.data_type)),
        vnni_weights(CheckedMultiply(k, n, "MatMulNBits", "prepared weight count") / 2),
        vnni_weight_sums(CheckedMultiply(blocks, n, "MatMulNBits", "prepared weight sum count")),
        vnni_scales(CheckedMultiply(blocks, n, "MatMulNBits", "prepared scale count")),
        max_participants(tuning.max_participants) {
    Pack(b, scales, attributes, vnni_weights.data(), vnni_weight_sums.data(), vnni_scales.data());
  }

  static std::size_t WeightBytes(const MatMulNBitsAttributes &attributes) {
    return CheckedMultiply(static_cast<std::size_t>(attributes.k),
                           static_cast<std::size_t>(attributes.n), "MatMulNBits",
                           "prepared weight count") /
           2;
  }

  static std::size_t WeightSumBytes(const MatMulNBitsAttributes &attributes) {
    const std::size_t blocks = (static_cast<std::size_t>(attributes.k) + 31) / 32;
    return CheckedByteSize(CheckedMultiply(blocks, static_cast<std::size_t>(attributes.n),
                                           "MatMulNBits", "prepared weight sum count"),
                           sizeof(std::int32_t), "MatMulNBits", "prepared weight sum byte size");
  }

  static std::size_t ScaleBytes(const MatMulNBitsAttributes &attributes) {
    const std::size_t blocks = (static_cast<std::size_t>(attributes.k) + 31) / 32;
    return CheckedByteSize(CheckedMultiply(blocks, static_cast<std::size_t>(attributes.n),
                                           "MatMulNBits", "prepared scale count"),
                           sizeof(float), "MatMulNBits", "prepared scale byte size");
  }

  static std::size_t TotalBytes(const MatMulNBitsAttributes &attributes) {
    return CheckedAdd(WeightBytes(attributes),
                      CheckedAdd(WeightSumBytes(attributes), ScaleBytes(attributes), "MatMulNBits",
                                 "prepared metadata byte size"),
                      "MatMulNBits", "prepared plan byte size");
  }

  static void Pack(const Tensor &b, const Tensor &scales, const MatMulNBitsAttributes &attributes,
                   std::uint8_t *vnni_weights, std::int32_t *vnni_weight_sums, float *vnni_scales) {
    const std::size_t k = static_cast<std::size_t>(attributes.k);
    const std::size_t n = static_cast<std::size_t>(attributes.n);
    const std::size_t blocks = (k + 31) / 32;
    const auto data_type = static_cast<RuntimeDataType>(scales.data_type);
    const auto *packed_values = b.bytes();
    const auto *scale_float = reinterpret_cast<const float *>(scales.bytes());
    const auto *scale_half = reinterpret_cast<const std::uint16_t *>(scales.bytes());
    const std::size_t column_groups = n / 16;
    for (std::size_t group = 0; group < column_groups; ++group) {
      for (std::size_t block = 0; block < blocks; ++block) {
        for (std::size_t quad = 0; quad < 8; ++quad) {
          for (std::size_t lane = 0; lane < 16; ++lane) {
            const std::size_t column = group * 16 + lane;
            const std::size_t source = (column * blocks + block) * 16 + quad * 2;
            const std::size_t target = ((group * blocks + block) * 8 + quad) * 32 + lane * 2;
            vnni_weights[target] = packed_values[source];
            vnni_weights[target + 1] = packed_values[source + 1];
          }
        }
        for (std::size_t lane = 0; lane < 16; ++lane) {
          const std::size_t column = group * 16 + lane;
          const std::size_t target = (group * blocks + block) * 16 + lane;
          const std::size_t scale_index = column * blocks + block;
          if (data_type == RuntimeDataType::FLOAT) {
            vnni_scales[target] = scale_float[scale_index];
          } else if (data_type == RuntimeDataType::FLOAT16) {
            vnni_scales[target] = detail::Float16BitsToFloat(scale_half[scale_index]);
          } else {
            vnni_scales[target] = detail::Bfloat16BitsToFloat(scale_half[scale_index]);
          }
          std::int32_t sum = 0;
          for (std::size_t offset = 0; offset < 16; ++offset) {
            const std::uint8_t byte = packed_values[(column * blocks + block) * 16 + offset];
            sum += static_cast<std::int32_t>(byte & 15) - 8;
            sum += static_cast<std::int32_t>(byte >> 4) - 8;
          }
          vnni_weight_sums[target] = sum;
        }
      }
    }
  }

  std::size_t k;
  std::size_t n;
  std::size_t blocks;
  RuntimeDataType data_type;
  std::vector<std::uint8_t> vnni_weights;
  std::vector<std::int32_t> vnni_weight_sums;
  std::vector<float> vnni_scales;
  std::int64_t max_participants;
};

struct MatMulNBitsKernel::PreparedInt4State {
  PreparedInt4State(rt_ns::PreparedExecutionState &execution_,
                    rt_ns::PreparedObjectRequest request_, std::size_t weight_bytes_,
                    std::size_t weight_sum_bytes_, std::size_t total_bytes_,
                    std::int64_t max_participants_)
      : execution(&execution_), request(std::move(request_)), weight_bytes(weight_bytes_),
        weight_sum_bytes(weight_sum_bytes_), total_bytes(total_bytes_),
        max_participants(max_participants_) {}

  rt_ns::PreparedExecutionState *execution;
  rt_ns::PreparedObjectRequest request;
  std::size_t weight_bytes;
  std::size_t weight_sum_bytes;
  std::size_t total_bytes;
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
}

bool MatMulNBitsKernel::HasPreparations(
    const std::unordered_set<std::string> &immutable_inputs) const {
  return attributes_.bits == 4 && attributes_.accuracy_level == 4 &&
         MatMulNBitsAccuracy4Float32Available() && attributes_.n % 16 == 0 &&
         attributes_.k % 32 == 0 && node_ != nullptr && node_->input_size() >= 3 &&
         immutable_inputs.contains(node_->input(1)) && immutable_inputs.contains(node_->input(2));
}

void MatMulNBitsKernel::Prepare(rt_ns::RuntimeContext &rt,
                                const std::unordered_set<std::string> &immutable_inputs,
                                rt_ns::PreparedExecutionState &state) {
  if (!HasPreparations(immutable_inputs)) {
    return;
  }
  const Tensor &b = rt.Get(node_->input(1));
  const Tensor &scales = rt.Get(node_->input(2));
  const std::size_t k = CheckedDimension(attributes_.k, "MatMulNBits", "K");
  const std::size_t n = CheckedDimension(attributes_.n, "MatMulNBits", "N");
  const std::size_t blocks = k / 32;
  RequireShape(b, {attributes_.n, static_cast<std::int64_t>(blocks), 16}, "B");
  const std::size_t expected_scales =
      CheckedMultiply(n, blocks, "MatMulNBits", "scale element count");
  if (!((scales.shape.size() == 1 && TensorElementCount(scales, "scales") == expected_scales) ||
        (scales.shape.size() == 2 && scales.shape[0] == attributes_.n &&
         scales.shape[1] == static_cast<std::int64_t>(blocks)))) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulNBits: scales must have shape [N, ceil(K/block_size)] or be flat.");
  }
  if (b.data_type != static_cast<std::int32_t>(RuntimeDataType::UINT8) ||
      (scales.data_type != static_cast<std::int32_t>(RuntimeDataType::FLOAT) &&
       scales.data_type != static_cast<std::int32_t>(RuntimeDataType::FLOAT16) &&
       scales.data_type != static_cast<std::int32_t>(RuntimeDataType::BFLOAT16))) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulNBits: prepared B must be UINT8 and scales must be FLOAT, FLOAT16, "
        "or BFLOAT16.");
  }
  std::uint64_t digest = 14695981039346656037ULL;
  const auto add_to_digest = [&](const Tensor &tensor) {
    for (std::size_t index = 0; index < tensor.size_bytes(); ++index) {
      digest = (digest ^ tensor.bytes()[index]) * 1099511628211ULL;
    }
  };
  add_to_digest(b);
  add_to_digest(scales);
  std::ostringstream key;
  key << "MatMulNBits:int4-v1:" << node_->input(1) << ':' << node_->input(2)
      << ":K=" << attributes_.k << ":N=" << attributes_.n << ":dtype=" << scales.data_type
      << ":digest=" << digest;
  rt_ns::PreparedObjectRequirement requirement{
      rt_ns::PreparedKey{key.str()}, node_->input(1) + std::string{"/"} + node_->input(2)};
  std::optional<rt_ns::PreparedObjectRequest> request;
  const std::size_t weight_bytes = PreparedInt4Plan::WeightBytes(attributes_);
  const std::size_t weight_sum_bytes = PreparedInt4Plan::WeightSumBytes(attributes_);
  const std::size_t total_bytes = PreparedInt4Plan::TotalBytes(attributes_);
  if (!state.objects().Find(requirement.key).has_value()) {
    rt_ns::AllocationHandle allocation = state.AllocatePrepared(total_bytes);
    request.emplace(state.objects().Request(requirement));
    if (request->producer) {
      state.objects().MarkPreparing(*request);
      std::uint8_t *base = allocation.buffer()->data();
      PreparedInt4Plan::Pack(b, scales, attributes_, base,
                             reinterpret_cast<std::int32_t *>(base + weight_bytes),
                             reinterpret_cast<float *>(base + weight_bytes + weight_sum_bytes));
      state.objects().Publish(*request, std::move(allocation));
    }
  }
  if (!request.has_value()) {
    request.emplace(state.objects().Request(requirement));
  }
  prepared_int4_ =
      std::make_unique<PreparedInt4State>(state, std::move(*request), weight_bytes,
                                          weight_sum_bytes, total_bytes, tuning_.max_participants);
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
  const std::uint8_t *prepared_weights = nullptr;
  const std::int32_t *prepared_weight_sums = nullptr;
  const float *prepared_scales = nullptr;
  std::int64_t prepared_max_participants = tuning_.max_participants;
  std::optional<rt_ns::PreparedObjectView> prepared_view;
  std::unique_ptr<const PreparedInt4Plan> invocation_plan;
  if (prepared_int4_ != nullptr) {
    prepared_int4_->request.completion.Wait();
    prepared_view = prepared_int4_->execution->objects().Find(prepared_int4_->request.key);
    if (prepared_view.has_value()) {
      if (prepared_view->buffer->size() < prepared_int4_->total_bytes) {
        throw std::runtime_error("onnx_light_cpu::MatMulNBits: prepared INT4 plan is truncated.");
      }
      const std::uint8_t *base = prepared_view->buffer->data();
      prepared_weights = base;
      prepared_weight_sums =
          reinterpret_cast<const std::int32_t *>(base + prepared_int4_->weight_bytes);
      prepared_scales = reinterpret_cast<const float *>(base + prepared_int4_->weight_bytes +
                                                        prepared_int4_->weight_sum_bytes);
      prepared_max_participants = prepared_int4_->max_participants;
    }
  }
  if (attributes_.bits == 4 && attributes_.accuracy_level == 4 &&
      MatMulNBitsAccuracy4Float32Available() &&
      (data_type == RuntimeDataType::FLOAT || data_type == RuntimeDataType::FLOAT16 ||
       data_type == RuntimeDataType::BFLOAT16) &&
      n % 16 == 0 && k % 32 == 0) {
    if (prepared_weights == nullptr) {
      invocation_plan = std::make_unique<PreparedInt4Plan>(b, scales, attributes_, tuning_);
      prepared_weights = invocation_plan->vnni_weights.data();
      prepared_weight_sums = invocation_plan->vnni_weight_sums.data();
      prepared_scales = invocation_plan->vnni_scales.data();
      prepared_max_participants = invocation_plan->max_participants;
    }
  }
  if (prepared_weights != nullptr) {
    if (data_type == RuntimeDataType::FLOAT16 || data_type == RuntimeDataType::BFLOAT16) {
      struct HalfScratch {
        std::vector<float> a;
        std::vector<float> bias;
        std::vector<float> y;
      };
      thread_local HalfScratch scratch;
      scratch.a.resize(rows * k);
      scratch.y.resize(output_count);
      const auto *half_a = reinterpret_cast<const std::uint16_t *>(a.bytes());
      if (data_type == RuntimeDataType::FLOAT16) {
        detail::ConvertFloat16ToFloat32(half_a, scratch.a.data(), scratch.a.size());
      } else {
        detail::ConvertBFloat16ToFloat32(half_a, scratch.a.data(), scratch.a.size());
      }
      const float *float_bias = nullptr;
      if (bias != nullptr) {
        scratch.bias.resize(n);
        const auto *half_bias = reinterpret_cast<const std::uint16_t *>(bias->bytes());
        if (data_type == RuntimeDataType::FLOAT16) {
          detail::ConvertFloat16ToFloat32(half_bias, scratch.bias.data(), n);
        } else {
          detail::ConvertBFloat16ToFloat32(half_bias, scratch.bias.data(), n);
        }
        float_bias = scratch.bias.data();
      }
      MatMulNBitsAccuracy4Float32(scratch.a.data(), prepared_weights, prepared_weight_sums,
                                  prepared_scales, float_bias, scratch.y.data(), rows, k, n,
                                  prepared_max_participants);
      auto *half_y = reinterpret_cast<std::uint16_t *>(y.mutable_bytes());
      if (data_type == RuntimeDataType::FLOAT16) {
        detail::ConvertFloat32ToFloat16(scratch.y.data(), half_y, output_count);
      } else {
        detail::ConvertFloat32ToBFloat16(scratch.y.data(), half_y, output_count);
      }
      return y;
    }
    MatMulNBitsAccuracy4Float32(
        reinterpret_cast<const float *>(a.bytes()), prepared_weights, prepared_weight_sums,
        prepared_scales, bias == nullptr ? nullptr : reinterpret_cast<const float *>(bias->bytes()),
        reinterpret_cast<float *>(y.mutable_bytes()), rows, k, n, prepared_max_participants);
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
