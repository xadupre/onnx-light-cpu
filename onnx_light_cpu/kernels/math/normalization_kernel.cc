// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/normalization_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/kernels/math/normalization_helpers.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/normalization_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace onnx_light_cpu {
namespace {

namespace norm = normalization;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::RuntimeContext;
using rt_ns::Shape;
using rt_ns::Tensor;

constexpr std::string_view kBatchOp = "onnx_light_cpu::BatchNormalization";
constexpr std::string_view kGroupOp = "onnx_light_cpu::GroupNormalization";
constexpr std::string_view kInstanceOp = "onnx_light_cpu::InstanceNormalization";
constexpr std::string_view kLayerOp = "onnx_light_cpu::LayerNormalization";
constexpr std::string_view kLpOp = "onnx_light_cpu::LpNormalization";
constexpr std::string_view kMvnOp = "onnx_light_cpu::MeanVarianceNormalization";

void RequireNonNegativeEpsilon(float epsilon, std::string_view op) {
  if (!std::isfinite(epsilon) || epsilon < 0.0F) {
    throw std::invalid_argument(std::string(op) + ": epsilon must be finite and non-negative.");
  }
}

Shape ReducedShape(const Shape &shape, std::size_t axis) {
  Shape result(shape);
  for (std::size_t i = axis; i < result.size(); ++i) {
    result[i] = 1;
  }
  return result;
}

template <typename Kernel>
void RegisterOne(const char *op_type, const char *kernel_name, std::int64_t since_version) {
  rt_ns::NodeKernelFn factory = [](const NodeProto &node,
                                   RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<Kernel>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = op_type;
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = kernel_name;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  info.since_version = since_version;
  RegisterKernel(std::move(info), std::move(factory));
}

template <typename Fn> void ExecuteItems(std::size_t count, double work_per_item, Fn &&fn) {
  if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("normalization work item count exceeds int64_t.");
  }
  constexpr double kParallelWeight = 0.125;
  ExecuteRanges(static_cast<std::int64_t>(count), work_per_item * kParallelWeight,
                [&](std::int64_t begin, std::int64_t end) {
                  fn(static_cast<std::size_t>(begin), static_cast<std::size_t>(end));
                });
}

template <typename Acc> Acc ReadParameter(const norm::TensorReader &reader, std::size_t index) {
  if constexpr (std::is_same_v<Acc, double>) {
    return reader.LoadDouble(index);
  } else {
    return reader.LoadFloat(index);
  }
}

template <DataType Type>
void ApplyAffine(const norm::StorageType<Type> *input, norm::StorageType<Type> *output,
                 std::size_t count, norm::AccumulatorType<Type> multiplier,
                 norm::AccumulatorType<Type> offset) {
  using Traits = norm::TypeTraits<Type>;
  for (std::size_t i = 0; i < count; ++i) {
    Traits::Store(output, i, Traits::Load(input, i) * multiplier + offset);
  }
}

template <DataType Type>
norm::Moments<norm::AccumulatorType<Type>>
ComputeBatchMoments(const norm::StorageType<Type> *input, std::size_t batch, std::size_t channels,
                    std::size_t spatial, std::size_t channel) {
  using Traits = norm::TypeTraits<Type>;
  using Acc = norm::AccumulatorType<Type>;
  Acc sums[4] = {};
  std::size_t position = 0;
  for (std::size_t n = 0; n < batch; ++n) {
    const auto *run = input + (n * channels + channel) * spatial;
    for (std::size_t i = 0; i < spatial; ++i, ++position) {
      sums[position & 3] += Traits::Load(run, i);
    }
  }
  const std::size_t count = batch * spatial;
  const Acc mean = (sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<Acc>(count);
  Acc squared_sums[4] = {};
  position = 0;
  for (std::size_t n = 0; n < batch; ++n) {
    const auto *run = input + (n * channels + channel) * spatial;
    for (std::size_t i = 0; i < spatial; ++i, ++position) {
      const Acc delta = Traits::Load(run, i) - mean;
      squared_sums[position & 3] += delta * delta;
    }
  }
  const Acc variance = (squared_sums[0] + squared_sums[1] + squared_sums[2] + squared_sums[3]) /
                       static_cast<Acc>(count);
  return {mean, variance};
}

template <DataType Type>
void BatchInference(const Tensor &x, Tensor &y,
                    const std::vector<norm::AccumulatorType<Type>> &multipliers,
                    const std::vector<norm::AccumulatorType<Type>> &offsets, std::size_t batch,
                    std::size_t channels, std::size_t spatial) {
  const auto *input = norm::Data<Type>(x);
  auto *output = norm::MutableData<Type>(y);
  const std::size_t slices = batch * channels;
  ExecuteItems(slices, static_cast<double>(spatial) * 3.0, [&](std::size_t begin, std::size_t end) {
    for (std::size_t slice = begin; slice < end; ++slice) {
      const std::size_t channel = slice % channels;
      const std::size_t base = slice * spatial;
      ApplyAffine<Type>(input + base, output + base, spatial, multipliers[channel],
                        offsets[channel]);
    }
  });
}

template <DataType Type>
void BatchTraining(const Tensor &x, Tensor &y, const norm::TensorReader &scale_reader,
                   const norm::TensorReader &bias_reader, const norm::TensorReader &mean_reader,
                   const norm::TensorReader &variance_reader, Tensor *running_mean,
                   Tensor *running_variance, std::size_t batch, std::size_t channels,
                   std::size_t spatial, float epsilon, float momentum) {
  using Acc = norm::AccumulatorType<Type>;
  const auto *input = norm::Data<Type>(x);
  auto *output = norm::MutableData<Type>(y);
  const std::optional<norm::TensorWriter> mean_writer =
      running_mean == nullptr ? std::nullopt
                              : std::optional<norm::TensorWriter>(std::in_place, *running_mean);
  const std::optional<norm::TensorWriter> variance_writer =
      running_variance == nullptr
          ? std::nullopt
          : std::optional<norm::TensorWriter>(std::in_place, *running_variance);
  const bool stats_are_double = mean_reader.type() == DataType::DOUBLE;
  ExecuteItems(
      channels, static_cast<double>(batch * spatial) * 6.0,
      [&](std::size_t begin, std::size_t end) {
        for (std::size_t channel = begin; channel < end; ++channel) {
          const norm::Moments<Acc> moments =
              ComputeBatchMoments<Type>(input, batch, channels, spatial, channel);
          const Acc multiplier = ReadParameter<Acc>(scale_reader, channel) /
                                 std::sqrt(moments.variance + static_cast<Acc>(epsilon));
          const Acc offset = ReadParameter<Acc>(bias_reader, channel) - moments.mean * multiplier;
          if (mean_writer) {
            if (stats_are_double) {
              mean_writer->StoreDouble(
                  channel,
                  mean_reader.LoadDouble(channel) * static_cast<double>(momentum) +
                      static_cast<double>(moments.mean) * static_cast<double>(1.0F - momentum));
            } else {
              mean_writer->StoreFloat(channel,
                                      mean_reader.LoadFloat(channel) * momentum +
                                          static_cast<float>(moments.mean) * (1.0F - momentum));
            }
          }
          if (variance_writer) {
            if (stats_are_double) {
              variance_writer->StoreDouble(
                  channel,
                  variance_reader.LoadDouble(channel) * static_cast<double>(momentum) +
                      static_cast<double>(moments.variance) * static_cast<double>(1.0F - momentum));
            } else {
              variance_writer->StoreFloat(channel, variance_reader.LoadFloat(channel) * momentum +
                                                       static_cast<float>(moments.variance) *
                                                           (1.0F - momentum));
            }
          }
          for (std::size_t n = 0; n < batch; ++n) {
            const std::size_t base = (n * channels + channel) * spatial;
            ApplyAffine<Type>(input + base, output + base, spatial, multiplier, offset);
          }
        }
      });
}

template <DataType Type>
void InstanceNormalize(const Tensor &x, const Tensor &scale, const Tensor &bias, Tensor &y,
                       std::size_t batch, std::size_t channels, std::size_t spatial,
                       float epsilon) {
  using Traits = norm::TypeTraits<Type>;
  using Acc = norm::AccumulatorType<Type>;
  const auto *input = norm::Data<Type>(x);
  const auto *scale_data = norm::Data<Type>(scale);
  const auto *bias_data = norm::Data<Type>(bias);
  auto *output = norm::MutableData<Type>(y);
  const std::size_t slices = batch * channels;
  ExecuteItems(slices, static_cast<double>(spatial) * 6.0, [&](std::size_t begin, std::size_t end) {
    for (std::size_t slice = begin; slice < end; ++slice) {
      const std::size_t channel = slice % channels;
      const std::size_t base = slice * spatial;
      const norm::Moments<Acc> moments =
          norm::ComputeContiguousMoments<Type>(input + base, spatial);
      const Acc multiplier = Traits::Load(scale_data, channel) /
                             std::sqrt(moments.variance + static_cast<Acc>(epsilon));
      const Acc offset = Traits::Load(bias_data, channel) - moments.mean * multiplier;
      ApplyAffine<Type>(input + base, output + base, spatial, multiplier, offset);
    }
  });
}

template <DataType Type>
void GroupNormalizeLegacy(const Tensor &x, const Tensor &scale, const Tensor &bias, Tensor &y,
                          std::size_t batch, std::size_t groups, std::size_t channels_per_group,
                          std::size_t spatial, std::size_t group_size, float epsilon) {
  using Traits = norm::TypeTraits<Type>;
  using Acc = norm::AccumulatorType<Type>;
  const auto *input = norm::Data<Type>(x);
  const auto *scale_data = norm::Data<Type>(scale);
  const auto *bias_data = norm::Data<Type>(bias);
  auto *output = norm::MutableData<Type>(y);
  ExecuteItems(batch * groups, static_cast<double>(group_size) * 6.0,
               [&](std::size_t begin, std::size_t end) {
                 for (std::size_t item = begin; item < end; ++item) {
                   const std::size_t group = item % groups;
                   const std::size_t base = item * group_size;
                   const norm::Moments<Acc> moments =
                       norm::ComputeContiguousMoments<Type>(input + base, group_size);
                   const Acc inverse_std_dev =
                       Acc{1} / std::sqrt(moments.variance + static_cast<Acc>(epsilon));
                   const Acc scale_value = Traits::Load(scale_data, group);
                   const Acc bias_value = Traits::Load(bias_data, group);
                   for (std::size_t local_channel = 0; local_channel < channels_per_group;
                        ++local_channel) {
                     const std::size_t channel_base = base + local_channel * spatial;
                     for (std::size_t i = 0; i < spatial; ++i) {
                       Traits::Store(output, channel_base + i,
                                     (Traits::Load(input, channel_base + i) - moments.mean) *
                                             inverse_std_dev * scale_value +
                                         bias_value);
                     }
                   }
                 }
               });
}

template <DataType Type>
void GroupNormalizeStashed(const Tensor &x, const Tensor &scale, const Tensor &bias, Tensor &y,
                           std::size_t batch, std::size_t groups, std::size_t channels_per_group,
                           std::size_t spatial, std::size_t group_size, float epsilon) {
  using Traits = norm::TypeTraits<Type>;
  const auto *input = norm::Data<Type>(x);
  const auto *scale_data = norm::Data<Type>(scale);
  const auto *bias_data = norm::Data<Type>(bias);
  auto *output = norm::MutableData<Type>(y);
  ExecuteItems(
      batch * groups, static_cast<double>(group_size) * 6.0,
      [&](std::size_t begin, std::size_t end) {
        for (std::size_t item = begin; item < end; ++item) {
          const std::size_t group = item % groups;
          const std::size_t base = item * group_size;
          const norm::Moments<float> moments =
              norm::ComputeContiguousFloatMoments<Type>(input + base, group_size);
          const float inverse_std_dev = 1.0F / std::sqrt(moments.variance + epsilon);
          for (std::size_t local_channel = 0; local_channel < channels_per_group; ++local_channel) {
            const std::size_t channel = group * channels_per_group + local_channel;
            const std::size_t channel_base = base + local_channel * spatial;
            const auto scale_value = Traits::Load(scale_data, channel);
            const auto bias_value = Traits::Load(bias_data, channel);
            for (std::size_t i = 0; i < spatial; ++i) {
              const float normalized =
                  (static_cast<float>(Traits::Load(input, channel_base + i)) - moments.mean) *
                  inverse_std_dev;
              const auto rounded = norm::RoundFloatToType<Type>(normalized);
              if constexpr (Type == DataType::DOUBLE) {
                Traits::Store(output, channel_base + i, rounded * scale_value + bias_value);
              } else {
                const float scaled =
                    norm::RoundFloatToType<Type>(rounded * static_cast<float>(scale_value));
                Traits::Store(output, channel_base + i, scaled + static_cast<float>(bias_value));
              }
            }
          }
        }
      });
}

bool ShapeMatchesSuffix(const Shape &parameter_shape, const Shape &input_shape, std::size_t axis) {
  if (parameter_shape.size() != input_shape.size() - axis) {
    return false;
  }
  return std::equal(parameter_shape.begin(), parameter_shape.end(),
                    input_shape.begin() + static_cast<std::ptrdiff_t>(axis));
}

template <DataType Type>
void LayerNormalize(const Tensor &x, const Tensor &scale, const Tensor *bias, Tensor &y,
                    float *mean_output, float *inv_output, std::size_t outer, std::size_t inner,
                    float epsilon, const norm::BroadcastIndexer &scale_index,
                    const norm::BroadcastIndexer *bias_index, bool scale_by_inner,
                    bool bias_by_inner) {
  using Traits = norm::TypeTraits<Type>;
  const auto *input = norm::Data<Type>(x);
  const auto *scale_data = norm::Data<Type>(scale);
  const auto *bias_data = bias == nullptr ? nullptr : norm::Data<Type>(*bias);
  auto *output = norm::MutableData<Type>(y);
  ExecuteItems(outer, static_cast<double>(inner) * 7.0, [&](std::size_t begin, std::size_t end) {
    for (std::size_t row = begin; row < end; ++row) {
      const std::size_t base = row * inner;
      const norm::Moments<float> moments = [&]() {
        if constexpr (Type == DataType::FLOAT) {
          const Float32NormalizationMoments shared =
              ComputeNormalizationMomentsFloat32(input + base, inner);
          return norm::Moments<float>{shared.mean, shared.variance};
        } else {
          return norm::ComputeContiguousFloatMoments<Type>(input + base, inner);
        }
      }();
      const float inverse_std_dev = 1.0F / std::sqrt(moments.variance + epsilon);
      if (mean_output != nullptr) {
        mean_output[row] = static_cast<float>(moments.mean);
      }
      if (inv_output != nullptr) {
        inv_output[row] = static_cast<float>(inverse_std_dev);
      }
      if constexpr (Type == DataType::FLOAT) {
        if (scale_by_inner && (bias_data == nullptr || bias_by_inner)) {
          ApplyNormalizationAffineFloat32(input + base, scale_data, bias_data, output + base, inner,
                                          moments.mean, inverse_std_dev);
          continue;
        }
      }
      for (std::size_t i = 0; i < inner; ++i) {
        const std::size_t flat = base + i;
        const std::size_t scale_position =
            scale_index.identity() ? flat : (scale_by_inner ? i : scale_index.Index(flat));
        const float normalized =
            (static_cast<float>(Traits::Load(input, flat)) - moments.mean) * inverse_std_dev;
        const auto rounded = norm::RoundFloatToType<Type>(normalized);
        if constexpr (Type == DataType::DOUBLE) {
          double value = rounded * Traits::Load(scale_data, scale_position);
          if (bias_data != nullptr) {
            const std::size_t bias_position =
                bias_index->identity() ? flat : (bias_by_inner ? i : bias_index->Index(flat));
            value += Traits::Load(bias_data, bias_position);
          }
          Traits::Store(output, flat, value);
        } else {
          float value = norm::RoundFloatToType<Type>(
              rounded * static_cast<float>(Traits::Load(scale_data, scale_position)));
          if (bias_data != nullptr) {
            const std::size_t bias_position =
                bias_index->identity() ? flat : (bias_by_inner ? i : bias_index->Index(flat));
            value += static_cast<float>(Traits::Load(bias_data, bias_position));
          }
          Traits::Store(output, flat, value);
        }
      }
    }
  });
}

template <DataType Type>
void LpNormalize(const Tensor &x, Tensor &y, std::size_t outer, std::size_t dimension,
                 std::size_t inner, std::int64_t p) {
  using Traits = norm::TypeTraits<Type>;
  using Acc = norm::AccumulatorType<Type>;
  const auto *input = norm::Data<Type>(x);
  auto *output = norm::MutableData<Type>(y);
  const std::size_t vectors = outer * inner;
  ExecuteItems(
      vectors, static_cast<double>(dimension) * 4.0, [&](std::size_t begin, std::size_t end) {
        for (std::size_t vector = begin; vector < end; ++vector) {
          const std::size_t prefix = vector / inner;
          const std::size_t suffix = vector % inner;
          const std::size_t first = prefix * dimension * inner + suffix;
          double sums[4] = {};
          for (std::size_t d = 0; d < dimension; ++d) {
            const double value = static_cast<double>(Traits::Load(input, first + d * inner));
            sums[d & 3] += p == 1 ? std::abs(value) : value * value;
          }
          double norm_value = sums[0] + sums[1] + sums[2] + sums[3];
          if (p == 2) {
            norm_value = std::sqrt(norm_value);
          }
          const Acc inverse = norm_value == 0.0 ? Acc{} : static_cast<Acc>(1.0 / norm_value);
          for (std::size_t d = 0; d < dimension; ++d) {
            const std::size_t index = first + d * inner;
            Traits::Store(output, index, Traits::Load(input, index) * inverse);
          }
        }
      });
}

template <DataType Type>
void MvnContiguous(const Tensor &x, Tensor &y, std::size_t lanes, std::size_t reduced_size) {
  using Traits = norm::TypeTraits<Type>;
  using Acc = norm::AccumulatorType<Type>;
  const auto *input = norm::Data<Type>(x);
  auto *output = norm::MutableData<Type>(y);
  ExecuteItems(lanes, static_cast<double>(reduced_size) * 6.0,
               [&](std::size_t begin, std::size_t end) {
                 for (std::size_t lane = begin; lane < end; ++lane) {
                   const std::size_t base = lane * reduced_size;
                   const norm::Moments<Acc> moments =
                       norm::ComputeContiguousMoments<Type>(input + base, reduced_size);
                   const Acc denominator = std::sqrt(moments.variance) + static_cast<Acc>(1.0e-9F);
                   for (std::size_t i = 0; i < reduced_size; ++i) {
                     Traits::Store(output, base + i,
                                   (Traits::Load(input, base + i) - moments.mean) / denominator);
                   }
                 }
               });
}

template <DataType Type>
void MvnRetainedAxis(const Tensor &x, Tensor &y, std::size_t outer, std::size_t lanes,
                     std::size_t inner) {
  using Traits = norm::TypeTraits<Type>;
  using Acc = norm::AccumulatorType<Type>;
  const auto *input = norm::Data<Type>(x);
  auto *output = norm::MutableData<Type>(y);
  const std::size_t count = outer * inner;
  ExecuteItems(lanes, static_cast<double>(count) * 6.0, [&](std::size_t begin, std::size_t end) {
    for (std::size_t lane = begin; lane < end; ++lane) {
      Acc sums[4] = {};
      Acc square_sums[4] = {};
      std::size_t position = 0;
      for (std::size_t prefix = 0; prefix < outer; ++prefix) {
        const auto *run = input + (prefix * lanes + lane) * inner;
        for (std::size_t i = 0; i < inner; ++i, ++position) {
          const Acc value = Traits::Load(run, i);
          sums[position & 3] += value;
          square_sums[position & 3] += value * value;
        }
      }
      const Acc mean = (sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<Acc>(count);
      const Acc second_moment =
          (square_sums[0] + square_sums[1] + square_sums[2] + square_sums[3]) /
          static_cast<Acc>(count);
      Acc variance = second_moment - mean * mean;
      const Acc cancellation_floor = norm::RawMomentsCancellationFloor(second_moment, count);
      if (!(variance > cancellation_floor)) {
        Acc centered_sums[4] = {};
        position = 0;
        for (std::size_t prefix = 0; prefix < outer; ++prefix) {
          const auto *run = input + (prefix * lanes + lane) * inner;
          for (std::size_t i = 0; i < inner; ++i, ++position) {
            const Acc delta = Traits::Load(run, i) - mean;
            centered_sums[position & 3] += delta * delta;
          }
        }
        variance = (centered_sums[0] + centered_sums[1] + centered_sums[2] + centered_sums[3]) /
                   static_cast<Acc>(count);
      }
      const Acc denominator = std::sqrt(variance) + static_cast<Acc>(1.0e-9F);
      for (std::size_t prefix = 0; prefix < outer; ++prefix) {
        const std::size_t base = (prefix * lanes + lane) * inner;
        for (std::size_t i = 0; i < inner; ++i) {
          Traits::Store(output, base + i, (Traits::Load(input, base + i) - mean) / denominator);
        }
      }
    }
  });
}

template <DataType Type>
void MvnGrouped(const Tensor &x, Tensor &y, const std::vector<std::size_t> &offsets,
                const std::vector<std::size_t> &positions, std::size_t reduced_size) {
  using Traits = norm::TypeTraits<Type>;
  using Acc = norm::AccumulatorType<Type>;
  const auto *input = norm::Data<Type>(x);
  auto *output = norm::MutableData<Type>(y);
  const std::size_t lanes = offsets.size() - 1;
  ExecuteItems(
      lanes, static_cast<double>(reduced_size) * 6.0, [&](std::size_t begin, std::size_t end) {
        for (std::size_t lane = begin; lane < end; ++lane) {
          const std::size_t first = offsets[lane];
          const std::size_t last = offsets[lane + 1];
          Acc sums[4] = {};
          for (std::size_t i = first; i < last; ++i) {
            sums[(i - first) & 3] += Traits::Load(input, positions[i]);
          }
          const Acc mean = (sums[0] + sums[1] + sums[2] + sums[3]) / static_cast<Acc>(reduced_size);
          Acc squared_sums[4] = {};
          for (std::size_t i = first; i < last; ++i) {
            const Acc delta = Traits::Load(input, positions[i]) - mean;
            squared_sums[(i - first) & 3] += delta * delta;
          }
          const Acc stddev =
              std::sqrt((squared_sums[0] + squared_sums[1] + squared_sums[2] + squared_sums[3]) /
                        static_cast<Acc>(reduced_size));
          const Acc denominator = stddev + static_cast<Acc>(1.0e-9F);
          for (std::size_t i = first; i < last; ++i) {
            const std::size_t index = positions[i];
            Traits::Store(output, index, (Traits::Load(input, index) - mean) / denominator);
          }
        }
      });
}

} // namespace

Tensor BatchNormalizationKernel::operator()(const Tensor &x, const Tensor &scale,
                                            const Tensor &bias, const Tensor &mean,
                                            const Tensor &variance, float epsilon,
                                            RuntimeContext *rt) const {
  return Compute(x, scale, bias, mean, variance, false, epsilon, 0.9F, rt).y;
}

BatchNormalizationResult
BatchNormalizationKernel::Compute(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                  const Tensor &mean, const Tensor &variance, bool training_mode,
                                  float epsilon, float momentum, RuntimeContext *rt,
                                  bool output_running_mean, bool output_running_variance) const {
  norm::RequireSupportedFloatType(x, kBatchOp, "X");
  norm::RequireSupportedFloatType(scale, kBatchOp, "scale");
  norm::RequireSupportedFloatType(bias, kBatchOp, "B");
  norm::RequireSupportedFloatType(mean, kBatchOp, "input_mean");
  norm::RequireSupportedFloatType(variance, kBatchOp, "input_var");
  norm::RequireSameType(scale, bias, kBatchOp, "scale and B");
  norm::RequireSameType(mean, variance, kBatchOp, "input_mean and input_var");
  RequireNonNegativeEpsilon(epsilon, kBatchOp);
  if (!std::isfinite(momentum)) {
    throw std::invalid_argument(std::string(kBatchOp) + ": momentum must be finite.");
  }
  if (x.shape.empty()) {
    throw std::invalid_argument(std::string(kBatchOp) + ": X must have rank at least 1.");
  }
  norm::Product(x.shape, 0, x.shape.size(), kBatchOp);
  const std::size_t channels = x.shape.size() == 1 ? 1 : static_cast<std::size_t>(x.shape[1]);
  if (channels == 0) {
    throw std::invalid_argument(std::string(kBatchOp) + ": channel count must be positive.");
  }
  norm::RequireVector(scale, channels, kBatchOp, "scale");
  norm::RequireVector(bias, channels, kBatchOp, "B");
  norm::RequireVector(mean, channels, kBatchOp, "input_mean");
  norm::RequireVector(variance, channels, kBatchOp, "input_var");

  BatchNormalizationResult result{norm::AllocateOutput(x.data_type, x.shape, 0, rt), std::nullopt,
                                  std::nullopt};
  if (training_mode && output_running_mean) {
    result.running_mean.emplace(norm::AllocateOutput(mean.data_type, mean.shape, 1, rt));
  }
  if (training_mode && output_running_variance) {
    result.running_variance.emplace(
        norm::AllocateOutput(variance.data_type, variance.shape, 2, rt));
  }
  const norm::TensorReader scale_reader(scale, kBatchOp, "scale");
  const norm::TensorReader bias_reader(bias, kBatchOp, "B");
  const norm::TensorReader mean_reader(mean, kBatchOp, "input_mean");
  const norm::TensorReader variance_reader(variance, kBatchOp, "input_var");
  const std::size_t batch = static_cast<std::size_t>(x.shape[0]);
  const std::size_t spatial =
      x.shape.size() == 1 ? 1 : norm::Product(x.shape, 2, x.shape.size(), kBatchOp);
  if (training_mode && batch * spatial == 0) {
    throw std::invalid_argument(std::string(kBatchOp) +
                                ": training reduction dimensions must contain elements.");
  }

  norm::DispatchFloatType(x.data_type, [&]<DataType Type>() {
    using Acc = norm::AccumulatorType<Type>;
    if (training_mode) {
      BatchTraining<Type>(x, result.y, scale_reader, bias_reader, mean_reader, variance_reader,
                          result.running_mean ? &*result.running_mean : nullptr,
                          result.running_variance ? &*result.running_variance : nullptr, batch,
                          channels, spatial, epsilon, momentum);
      return;
    }
    std::vector<Acc> multipliers(channels);
    std::vector<Acc> offsets(channels);
    for (std::size_t channel = 0; channel < channels; ++channel) {
      const Acc variance_value = ReadParameter<Acc>(variance_reader, channel);
      if (!std::isfinite(variance_value) || variance_value < Acc{}) {
        throw std::invalid_argument(std::string(kBatchOp) +
                                    ": input_var must be finite and non-negative.");
      }
      multipliers[channel] = ReadParameter<Acc>(scale_reader, channel) /
                             std::sqrt(variance_value + static_cast<Acc>(epsilon));
      offsets[channel] = ReadParameter<Acc>(bias_reader, channel) -
                         ReadParameter<Acc>(mean_reader, channel) * multipliers[channel];
    }
    BatchInference<Type>(x, result.y, multipliers, offsets, batch, channels, spatial);
  });
  return result;
}

void BatchNormalizationKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 5);
  const std::int64_t training_mode = rt_ns::GetAttributeIntOrDefault(node, "training_mode", 0);
  if (training_mode != 0 && training_mode != 1) {
    throw std::invalid_argument(
        "onnx_light_cpu::BatchNormalization: training_mode must be 0 or 1.");
  }
  if (node.output_size() != 1 && node.output_size() != 3) {
    throw std::invalid_argument(std::string(kBatchOp) +
                                ": expected one output or three optional output slots.");
  }
  if (node.output(0).empty()) {
    throw std::invalid_argument(std::string(kBatchOp) + ": output Y cannot be omitted.");
  }
  const bool output_running_mean = node.output_size() == 3 && !node.output(1).empty();
  const bool output_running_variance = node.output_size() == 3 && !node.output(2).empty();
  if (training_mode == 0 && (output_running_mean || output_running_variance)) {
    throw std::invalid_argument(std::string(kBatchOp) +
                                ": inference mode cannot request running statistics.");
  }
  BatchNormalizationResult result =
      Compute(rt_ns::GetInput(node, 0, rt.tensors()), rt_ns::GetInput(node, 1, rt.tensors()),
              rt_ns::GetInput(node, 2, rt.tensors()), rt_ns::GetInput(node, 3, rt.tensors()),
              rt_ns::GetInput(node, 4, rt.tensors()), training_mode == 1,
              rt_ns::GetAttributeFloatOrDefault(node, "epsilon", 1.0e-5F),
              rt_ns::GetAttributeFloatOrDefault(node, "momentum", 0.9F), &rt, output_running_mean,
              output_running_variance);
  rt_ns::SetOutput(node, 0, std::move(result.y), rt);
  if (result.running_mean) {
    rt_ns::SetOutput(node, 1, std::move(*result.running_mean), rt);
  }
  if (result.running_variance) {
    rt_ns::SetOutput(node, 2, std::move(*result.running_variance), rt);
  }
}

Tensor GroupNormalizationKernel::operator()(const Tensor &x, const Tensor &scale,
                                            const Tensor &bias, std::int64_t num_groups,
                                            float epsilon, std::int64_t stash_type,
                                            RuntimeContext *rt) const {
  norm::RequireSupportedFloatType(x, kGroupOp, "X");
  norm::RequireSameType(x, scale, kGroupOp, "X, scale, and bias");
  norm::RequireSameType(x, bias, kGroupOp, "X, scale, and bias");
  const bool uses_float_stash = ctx_.opset.version >= 21;
  if (uses_float_stash && stash_type != 1) {
    throw std::invalid_argument(std::string(kGroupOp) + ": only stash_type=1 is supported.");
  }
  RequireNonNegativeEpsilon(epsilon, kGroupOp);
  if (x.shape.size() < 2) {
    throw std::invalid_argument(std::string(kGroupOp) + ": X must have rank at least 2.");
  }
  norm::Product(x.shape, 0, x.shape.size(), kGroupOp);
  if (num_groups <= 0) {
    throw std::invalid_argument(std::string(kGroupOp) + ": num_groups must be positive.");
  }
  const std::size_t batch = static_cast<std::size_t>(x.shape[0]);
  const std::size_t channels = static_cast<std::size_t>(x.shape[1]);
  if (channels == 0 || channels % static_cast<std::size_t>(num_groups) != 0) {
    throw std::invalid_argument(std::string(kGroupOp) +
                                ": num_groups must divide the positive channel count.");
  }
  const std::size_t affine_size =
      uses_float_stash ? channels : static_cast<std::size_t>(num_groups);
  norm::RequireVector(scale, affine_size, kGroupOp, "scale");
  norm::RequireVector(bias, affine_size, kGroupOp, "bias");
  const std::size_t spatial = norm::Product(x.shape, 2, x.shape.size(), kGroupOp);
  const std::size_t channels_per_group = channels / static_cast<std::size_t>(num_groups);
  const std::size_t group_size = channels_per_group * spatial;
  if (group_size == 0) {
    throw std::invalid_argument(std::string(kGroupOp) +
                                ": each normalization group must contain elements.");
  }

  Tensor output = norm::AllocateOutput(x.data_type, x.shape, 0, rt);
  norm::DispatchFloatType(x.data_type, [&]<DataType Type>() {
    if (uses_float_stash) {
      GroupNormalizeStashed<Type>(x, scale, bias, output, batch,
                                  static_cast<std::size_t>(num_groups), channels_per_group, spatial,
                                  group_size, epsilon);
    } else {
      GroupNormalizeLegacy<Type>(x, scale, bias, output, batch,
                                 static_cast<std::size_t>(num_groups), channels_per_group, spatial,
                                 group_size, epsilon);
    }
  });
  return output;
}

void GroupNormalizationKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 3);
  rt_ns::RequireOutputCount(node, 1);
  rt_ns::SetOutput(node, 0,
                   (*this)(rt_ns::GetInput(node, 0, rt.tensors()),
                           rt_ns::GetInput(node, 1, rt.tensors()),
                           rt_ns::GetInput(node, 2, rt.tensors()),
                           rt_ns::GetAttributeIntOrDefault(node, "num_groups", 0),
                           rt_ns::GetAttributeFloatOrDefault(node, "epsilon", 1.0e-5F),
                           rt_ns::GetAttributeIntOrDefault(node, "stash_type", 1), &rt),
                   rt);
}

Tensor InstanceNormalizationKernel::operator()(const Tensor &x, const Tensor &scale,
                                               const Tensor &bias, float epsilon,
                                               RuntimeContext *rt) const {
  norm::RequireSupportedFloatType(x, kInstanceOp, "input");
  norm::RequireSameType(x, scale, kInstanceOp, "input, scale, and B");
  norm::RequireSameType(x, bias, kInstanceOp, "input, scale, and B");
  RequireNonNegativeEpsilon(epsilon, kInstanceOp);
  if (x.shape.size() < 2) {
    throw std::invalid_argument(std::string(kInstanceOp) + ": input must have rank at least 2.");
  }
  norm::Product(x.shape, 0, x.shape.size(), kInstanceOp);
  const std::size_t batch = static_cast<std::size_t>(x.shape[0]);
  const std::size_t channels = static_cast<std::size_t>(x.shape[1]);
  norm::RequireVector(scale, channels, kInstanceOp, "scale");
  norm::RequireVector(bias, channels, kInstanceOp, "B");
  const std::size_t spatial = norm::Product(x.shape, 2, x.shape.size(), kInstanceOp);
  if (spatial == 0) {
    throw std::invalid_argument(std::string(kInstanceOp) +
                                ": each channel slice must contain elements.");
  }

  Tensor output = norm::AllocateOutput(x.data_type, x.shape, 0, rt);
  norm::DispatchFloatType(x.data_type, [&]<DataType Type>() {
    InstanceNormalize<Type>(x, scale, bias, output, batch, channels, spatial, epsilon);
  });
  return output;
}

void InstanceNormalizationKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 3);
  rt_ns::RequireOutputCount(node, 1);
  rt_ns::SetOutput(node, 0,
                   (*this)(rt_ns::GetInput(node, 0, rt.tensors()),
                           rt_ns::GetInput(node, 1, rt.tensors()),
                           rt_ns::GetInput(node, 2, rt.tensors()),
                           rt_ns::GetAttributeFloatOrDefault(node, "epsilon", 1.0e-5F), &rt),
                   rt);
}

LayerNormalizationResult LayerNormalizationKernel::operator()(
    const Tensor &x, const Tensor &scale, const Tensor *bias, std::int64_t axis, float epsilon,
    std::int64_t stash_type, bool output_mean, bool output_inv_std_dev, RuntimeContext *rt) const {
  norm::RequireSupportedFloatType(x, kLayerOp, "X");
  norm::RequireSameType(x, scale, kLayerOp, "X, Scale, and B");
  if (bias != nullptr) {
    norm::RequireSameType(x, *bias, kLayerOp, "X, Scale, and B");
  }
  if (stash_type != 1) {
    throw std::invalid_argument(std::string(kLayerOp) + ": only stash_type=1 is supported.");
  }
  RequireNonNegativeEpsilon(epsilon, kLayerOp);
  if (x.shape.empty()) {
    throw std::invalid_argument(std::string(kLayerOp) + ": X must have rank at least 1.");
  }
  norm::Product(x.shape, 0, x.shape.size(), kLayerOp);
  const std::size_t normalized_axis =
      static_cast<std::size_t>(norm::NormalizeAxis(axis, x.shape.size(), kLayerOp));
  const norm::BroadcastIndexer scale_index(x.shape, scale.shape, kLayerOp, "Scale");
  const std::optional<norm::BroadcastIndexer> bias_index =
      bias != nullptr ? std::optional<norm::BroadcastIndexer>(std::in_place, x.shape, bias->shape,
                                                              kLayerOp, "B")
                      : std::nullopt;
  const std::size_t outer = norm::Product(x.shape, 0, normalized_axis, kLayerOp);
  const std::size_t inner = norm::Product(x.shape, normalized_axis, x.shape.size(), kLayerOp);
  if (inner == 0) {
    throw std::invalid_argument(std::string(kLayerOp) +
                                ": the normalized region must contain elements.");
  }

  LayerNormalizationResult result{norm::AllocateOutput(x.data_type, x.shape, 0, rt), std::nullopt,
                                  std::nullopt};
  const Shape reduced_shape = ReducedShape(x.shape, normalized_axis);
  if (output_mean) {
    result.mean.emplace(
        norm::AllocateOutput(static_cast<std::int32_t>(DataType::FLOAT), reduced_shape, 1, rt));
  }
  if (output_inv_std_dev) {
    result.inv_std_dev.emplace(
        norm::AllocateOutput(static_cast<std::int32_t>(DataType::FLOAT), reduced_shape, 2, rt));
  }

  float *mean_output =
      result.mean ? reinterpret_cast<float *>(result.mean->mutable_bytes()) : nullptr;
  float *inv_output =
      result.inv_std_dev ? reinterpret_cast<float *>(result.inv_std_dev->mutable_bytes()) : nullptr;
  const bool scale_by_inner = ShapeMatchesSuffix(scale.shape, x.shape, normalized_axis);
  const bool bias_by_inner =
      bias != nullptr && ShapeMatchesSuffix(bias->shape, x.shape, normalized_axis);
  norm::DispatchFloatType(x.data_type, [&]<DataType Type>() {
    LayerNormalize<Type>(x, scale, bias, result.y, mean_output, inv_output, outer, inner, epsilon,
                         scale_index, bias_index ? &*bias_index : nullptr, scale_by_inner,
                         bias_by_inner);
  });
  return result;
}

void LayerNormalizationKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  if (node.input_size() < 2 || node.input_size() > 3) {
    throw std::invalid_argument(
        "onnx_light_cpu::LayerNormalization: expected two or three inputs.");
  }
  if (node.output_size() < 1 || node.output_size() > 3 || node.output(0).empty()) {
    throw std::invalid_argument(
        "onnx_light_cpu::LayerNormalization: expected one to three outputs.");
  }
  const Tensor *bias = rt_ns::GetOptionalInput(node, 2, rt.tensors());
  const bool output_mean = node.output_size() > 1 && !node.output(1).empty();
  const bool output_inv_std_dev = node.output_size() > 2 && !node.output(2).empty();
  LayerNormalizationResult result = (*this)(
      rt_ns::GetInput(node, 0, rt.tensors()), rt_ns::GetInput(node, 1, rt.tensors()), bias,
      rt_ns::GetAttributeIntOrDefault(node, "axis", -1),
      rt_ns::GetAttributeFloatOrDefault(node, "epsilon", 1.0e-5F),
      rt_ns::GetAttributeIntOrDefault(node, "stash_type", 1), output_mean, output_inv_std_dev, &rt);
  rt_ns::SetOutput(node, 0, std::move(result.y), rt);
  if (result.mean) {
    rt_ns::SetOutput(node, 1, std::move(*result.mean), rt);
  }
  if (result.inv_std_dev) {
    rt_ns::SetOutput(node, 2, std::move(*result.inv_std_dev), rt);
  }
}

Tensor LpNormalizationKernel::operator()(const Tensor &x, std::int64_t axis, std::int64_t p,
                                         RuntimeContext *rt) const {
  norm::RequireSupportedFloatType(x, kLpOp, "input");
  if (x.shape.empty()) {
    throw std::invalid_argument(std::string(kLpOp) + ": input must have rank at least 1.");
  }
  norm::Product(x.shape, 0, x.shape.size(), kLpOp);
  if (p != 1 && p != 2) {
    throw std::invalid_argument(std::string(kLpOp) + ": p must be 1 or 2.");
  }
  const std::size_t normalized_axis =
      static_cast<std::size_t>(norm::NormalizeAxis(axis, x.shape.size(), kLpOp));
  const std::size_t outer = norm::Product(x.shape, 0, normalized_axis, kLpOp);
  const std::size_t dimension = static_cast<std::size_t>(x.shape[normalized_axis]);
  const std::size_t inner = norm::Product(x.shape, normalized_axis + 1, x.shape.size(), kLpOp);
  if (dimension == 0) {
    throw std::invalid_argument(std::string(kLpOp) + ": normalized axis must be non-empty.");
  }

  Tensor output = norm::AllocateOutput(x.data_type, x.shape, 0, rt);
  norm::DispatchFloatType(x.data_type, [&]<DataType Type>() {
    LpNormalize<Type>(x, output, outer, dimension, inner, p);
  });
  return output;
}

void LpNormalizationKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  rt_ns::SetOutput(node, 0,
                   (*this)(rt_ns::GetInput(node, 0, rt.tensors()),
                           rt_ns::GetAttributeIntOrDefault(node, "axis", -1),
                           rt_ns::GetAttributeIntOrDefault(node, "p", 2), &rt),
                   rt);
}

Tensor MeanVarianceNormalizationKernel::operator()(const Tensor &x,
                                                   const std::vector<std::int64_t> &axes,
                                                   RuntimeContext *rt) const {
  norm::RequireSupportedFloatType(x, kMvnOp, "X");
  if (x.shape.empty()) {
    throw std::invalid_argument(std::string(kMvnOp) + ": X must have rank at least 1.");
  }
  norm::Product(x.shape, 0, x.shape.size(), kMvnOp);
  std::vector<bool> reduced(x.shape.size(), false);
  if (axes.empty()) {
    std::fill(reduced.begin(), reduced.end(), true);
  } else {
    for (std::int64_t axis : axes) {
      const std::size_t normalized_axis =
          static_cast<std::size_t>(norm::NormalizeAxis(axis, x.shape.size(), kMvnOp));
      reduced[normalized_axis] = true;
    }
  }

  std::size_t reduced_size = 1;
  std::size_t lane_count = 1;
  std::vector<std::size_t> lane_strides(x.shape.size(), 0);
  for (std::size_t reverse = 0; reverse < x.shape.size(); ++reverse) {
    const std::size_t axis = x.shape.size() - reverse - 1;
    const std::size_t dimension = static_cast<std::size_t>(x.shape[axis]);
    if (reduced[axis]) {
      reduced_size *= dimension;
    } else {
      lane_strides[axis] = lane_count;
      lane_count *= dimension;
    }
  }
  if (reduced_size == 0) {
    throw std::invalid_argument(std::string(kMvnOp) +
                                ": reduction dimensions must contain elements.");
  }
  const std::size_t total = norm::Product(x.shape, 0, x.shape.size(), kMvnOp);
  Tensor output = norm::AllocateOutput(x.data_type, x.shape, 0, rt);
  if (total == 0) {
    return output;
  }

  std::size_t suffix_begin = x.shape.size();
  while (suffix_begin > 0 && reduced[suffix_begin - 1]) {
    --suffix_begin;
  }
  bool contiguous_suffix = suffix_begin < x.shape.size();
  for (std::size_t axis = 0; axis < suffix_begin; ++axis) {
    contiguous_suffix = contiguous_suffix && !reduced[axis];
  }
  if (contiguous_suffix) {
    const std::size_t lanes = norm::Product(x.shape, 0, suffix_begin, kMvnOp);
    norm::DispatchFloatType(
        x.data_type, [&]<DataType Type>() { MvnContiguous<Type>(x, output, lanes, reduced_size); });
    return output;
  }

  std::size_t retained_axes = 0;
  std::size_t retained_axis = 0;
  for (std::size_t axis = 0; axis < reduced.size(); ++axis) {
    if (!reduced[axis]) {
      ++retained_axes;
      retained_axis = axis;
    }
  }
  if (retained_axes == 1) {
    const std::size_t outer = norm::Product(x.shape, 0, retained_axis, kMvnOp);
    const std::size_t lanes = static_cast<std::size_t>(x.shape[retained_axis]);
    const std::size_t inner = norm::Product(x.shape, retained_axis + 1, x.shape.size(), kMvnOp);
    norm::DispatchFloatType(x.data_type, [&]<DataType Type>() {
      MvnRetainedAxis<Type>(x, output, outer, lanes, inner);
    });
    return output;
  }

  std::vector<std::size_t> lane_counts(lane_count, 0);
  std::vector<std::size_t> lanes(total);
  for (std::size_t i = 0; i < total; ++i) {
    std::size_t flat = i;
    std::size_t lane = 0;
    for (std::size_t reverse = 0; reverse < x.shape.size(); ++reverse) {
      const std::size_t axis = x.shape.size() - reverse - 1;
      const std::size_t dimension = static_cast<std::size_t>(x.shape[axis]);
      const std::size_t coordinate = flat % dimension;
      flat /= dimension;
      lane += coordinate * lane_strides[axis];
    }
    lanes[i] = lane;
    ++lane_counts[lane];
  }
  std::vector<std::size_t> offsets(lane_count + 1, 0);
  for (std::size_t lane = 0; lane < lane_count; ++lane) {
    offsets[lane + 1] = offsets[lane] + lane_counts[lane];
  }
  std::vector<std::size_t> cursors(offsets.begin(), offsets.end() - 1);
  std::vector<std::size_t> positions(total);
  for (std::size_t i = 0; i < total; ++i) {
    positions[cursors[lanes[i]]++] = i;
  }
  norm::DispatchFloatType(x.data_type, [&]<DataType Type>() {
    MvnGrouped<Type>(x, output, offsets, positions, reduced_size);
  });
  return output;
}

void MeanVarianceNormalizationKernel::Run(RuntimeContext &rt) {
  RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireInputCount(node, 1);
  rt_ns::RequireOutputCount(node, 1);
  const rt_ns::ParamInts axes = rt_ns::GetAttributeIntsOrDefault(node, "axes", {0, 2, 3});
  rt_ns::SetOutput(node, 0,
                   (*this)(rt_ns::GetInput(node, 0, rt.tensors()),
                           std::vector<std::int64_t>(axes.begin(), axes.end()), &rt),
                   rt);
}

void RegisterNormalizationKernels() {
  RegisterOne<BatchNormalizationKernel>("BatchNormalization", BatchNormalizationKernel::kName, 15);
  RegisterOne<GroupNormalizationKernel>("GroupNormalization", GroupNormalizationKernel::kName, 18);
  RegisterOne<InstanceNormalizationKernel>("InstanceNormalization",
                                           InstanceNormalizationKernel::kName, 22);
  RegisterOne<LayerNormalizationKernel>("LayerNormalization", LayerNormalizationKernel::kName, 17);
  RegisterOne<LpNormalizationKernel>("LpNormalization", LpNormalizationKernel::kName, 22);
  RegisterOne<MeanVarianceNormalizationKernel>("MeanVarianceNormalization",
                                               MeanVarianceNormalizationKernel::kName, 13);
}

} // namespace onnx_light_cpu
