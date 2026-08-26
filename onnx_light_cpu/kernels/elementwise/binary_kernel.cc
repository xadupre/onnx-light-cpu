// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/elementwise/binary_kernel.h"

#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/symbolic/sym_tensor.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <stdexcept>
#include <string>

namespace onnx_light_cpu {
namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

constexpr const char *kBulkThresholdBytes = "parallel.bulk_threshold_bytes";
constexpr const char *kBlockThresholdBytes = "parallel.block_threshold_bytes";
constexpr const char *kScalarThresholdBytes = "parallel.scalar_threshold_bytes";
constexpr const char *kTargetBlockBytes = "parallel.target_block_bytes";
constexpr const char *kMaxParticipants = "parallel.max_participants";

std::string KernelName(std::string_view op_type) {
  return std::string("onnx_light_cpu::") + std::string(op_type);
}

rt_ns::KernelTuningKey MakeTuningKey(std::string_view op_type, int32_t element_type) {
  return {"onnx_light_cpu", std::string(op_type), "broadcast_plan",
          element_type,     sym_ns::Device::kCPU, BinaryElementwiseKernel::kTuningAbi};
}

void ValidateTuning(const rt_ns::KernelTuningParameters &parameters) {
  for (const char *name :
       {kBulkThresholdBytes, kBlockThresholdBytes, kScalarThresholdBytes, kTargetBlockBytes}) {
    const int64_t value = parameters.Get<int64_t>(name);
    if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(std::string("Binary ") + name +
                                  " must be zero or a positive value representable by size_t.");
    }
  }
  if (parameters.Get<int64_t>(kTargetBlockBytes) == 0) {
    throw std::invalid_argument("Binary parallel.target_block_bytes must be positive.");
  }
  if (parameters.Get<int64_t>(kMaxParticipants) < 0) {
    throw std::invalid_argument("Binary parallel.max_participants must be non-negative.");
  }
}

rt_ns::KernelTuningParameters MakeTuningDefaults(std::string_view op_type, int32_t element_type) {
  return {
      MakeTuningKey(op_type, element_type),
      {{kBulkThresholdBytes,
        static_cast<int64_t>(kDefaultBinaryExecutionTuning.bulk_parallel_threshold_bytes)},
       {kBlockThresholdBytes,
        static_cast<int64_t>(kDefaultBinaryExecutionTuning.block_parallel_threshold_bytes)},
       {kScalarThresholdBytes,
        static_cast<int64_t>(kDefaultBinaryExecutionTuning.scalar_parallel_threshold_bytes)},
       {kTargetBlockBytes, static_cast<int64_t>(kDefaultBinaryExecutionTuning.target_block_bytes)},
       {kMaxParticipants, int64_t{0}}}};
}

struct BinaryCalibrationProfile {
  const char *name;
  int64_t bulk_threshold;
  int64_t block_threshold;
  int64_t scalar_threshold;
  int64_t target_block;
  int64_t max_participants;
};

constexpr std::array<BinaryCalibrationProfile, 7> kBinaryCalibrationProfiles = {{
    {"portable", 1024 * 1024, 1024 * 1024, 256 * 1024, 1024 * 1024, 0},
    {"serial", std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(),
     std::numeric_limits<int64_t>::max(), 1024 * 1024, 1},
    {"coarse", 4 * 1024 * 1024, 16 * 1024 * 1024, 4 * 1024 * 1024, 4 * 1024 * 1024, 0},
    {"balanced", 1024 * 1024, 4 * 1024 * 1024, 1024 * 1024, 1024 * 1024, 32},
    {"fine", 256 * 1024, 1024 * 1024, 256 * 1024, 256 * 1024, 32},
    {"compute", 64 * 1024, 256 * 1024, 64 * 1024, 256 * 1024, 32},
    {"limited", 1024 * 1024, 4 * 1024 * 1024, 1024 * 1024, 1024 * 1024, 8},
}};

rt_ns::KernelTuningParameters CalibrationParameters(const rt_ns::KernelTuningKey &key,
                                                    const BinaryCalibrationProfile &profile) {
  return {key,
          {{kBulkThresholdBytes, profile.bulk_threshold},
           {kBlockThresholdBytes, profile.block_threshold},
           {kScalarThresholdBytes, profile.scalar_threshold},
           {kTargetBlockBytes, profile.target_block},
           {kMaxParticipants, profile.max_participants}}};
}

template <typename T>
rt_ns::Tensor MakeIntegerCalibrationTensor(int32_t element_type, const rt_ns::Shape &shape,
                                           uint64_t seed) {
  const std::size_t count = static_cast<std::size_t>(shape.product());
  std::vector<T> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = static_cast<T>(1 + ((i + seed) % 4));
  }
  rt_ns::Tensor tensor = rt_ns::Tensor::From("", shape, values);
  if (tensor.data_type != element_type) {
    throw std::invalid_argument("Binary calibration integer element type mismatch.");
  }
  return tensor;
}

rt_ns::Tensor MakeCalibrationTensor(int32_t element_type, const rt_ns::Shape &shape, uint64_t seed,
                                    bool integral_values = false) {
  const std::size_t count = static_cast<std::size_t>(shape.product());
  switch (static_cast<rt_ns::DataType>(element_type)) {
  case rt_ns::DataType::FLOAT: {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = 1.0f + static_cast<float>((i + seed) % 4) * (integral_values ? 1.0f : 0.125f);
    }
    return rt_ns::Tensor::FromFloat("", shape, values);
  }
  case rt_ns::DataType::DOUBLE: {
    std::vector<double> values(count);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = 1.0 + static_cast<double>((i + seed) % 4) * (integral_values ? 1.0 : 0.125);
    }
    return rt_ns::Tensor::FromDouble("", shape, values);
  }
  case rt_ns::DataType::FLOAT16:
  case rt_ns::DataType::BFLOAT16: {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = 1.0f + static_cast<float>((i + seed) % 4) * (integral_values ? 1.0f : 0.125f);
    }
    return element_type == static_cast<int32_t>(rt_ns::DataType::FLOAT16)
               ? rt_ns::MakeFloat16Tensor("", shape, values)
               : rt_ns::MakeBfloat16Tensor("", shape, values);
  }
  case rt_ns::DataType::BOOL: {
    std::vector<std::uint8_t> values(count);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = static_cast<std::uint8_t>((i + seed) & 1U);
    }
    return rt_ns::Tensor::FromBool("", shape, values);
  }
  case rt_ns::DataType::INT8:
    return MakeIntegerCalibrationTensor<std::int8_t>(element_type, shape, seed);
  case rt_ns::DataType::INT16:
    return MakeIntegerCalibrationTensor<std::int16_t>(element_type, shape, seed);
  case rt_ns::DataType::INT32:
    return MakeIntegerCalibrationTensor<std::int32_t>(element_type, shape, seed);
  case rt_ns::DataType::INT64:
    return MakeIntegerCalibrationTensor<std::int64_t>(element_type, shape, seed);
  case rt_ns::DataType::UINT8:
    return MakeIntegerCalibrationTensor<std::uint8_t>(element_type, shape, seed);
  case rt_ns::DataType::UINT16:
    return MakeIntegerCalibrationTensor<std::uint16_t>(element_type, shape, seed);
  case rt_ns::DataType::UINT32:
    return MakeIntegerCalibrationTensor<std::uint32_t>(element_type, shape, seed);
  case rt_ns::DataType::UINT64:
    return MakeIntegerCalibrationTensor<std::uint64_t>(element_type, shape, seed);
  default:
    throw std::invalid_argument("Binary calibration received an unsupported element type.");
  }
}

ONNX_LIGHT_NAMESPACE::NodeProto MakeCalibrationNode(const BinaryManifestEntry &entry,
                                                    int32_t element_type) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type(std::string(entry.op_type));
  node.add_input("left");
  node.add_input("right");
  node.add_output("output");
  if (entry.op == BinaryOperator::kMod) {
    const auto type = element_type;
    const bool floating = type == BinaryDataType::FLOAT || type == BinaryDataType::DOUBLE ||
                          type == BinaryDataType::FLOAT16 || type == BinaryDataType::BFLOAT16;
    auto *attribute = node.add_attribute();
    attribute->set_name("fmod");
    attribute->set_i(floating ? 1 : 0);
    attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::INT);
  } else if (entry.op == BinaryOperator::kBitShift) {
    auto *attribute = node.add_attribute();
    attribute->set_name("direction");
    attribute->set_s("LEFT");
    attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::STRING);
  }
  return node;
}

int64_t Median(std::array<int64_t, 3> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[1];
}

int64_t Measure(const std::function<void()> &run) {
  const auto begin = std::chrono::steady_clock::now();
  run();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                              begin)
      .count();
}

struct BinaryCalibrationCaseData {
  std::string name;
  rt_ns::Tensor left;
  rt_ns::Tensor right;
  rt_ns::Tensor reference_output;
  rt_ns::Tensor candidate_output;
};

struct BinaryCalibrationCaseSpec {
  std::string name;
  std::int32_t left_type;
  std::int32_t right_type;
  std::int32_t output_type;
  rt_ns::Shape left_shape;
  rt_ns::Shape right_shape;
  uint64_t output_elements;
};

std::size_t CalibrationElementSize(std::int32_t type) {
  switch (type) {
  case BinaryDataType::BOOL:
  case BinaryDataType::INT8:
  case BinaryDataType::UINT8:
    return 1;
  case BinaryDataType::FLOAT16:
  case BinaryDataType::BFLOAT16:
  case BinaryDataType::INT16:
  case BinaryDataType::UINT16:
    return 2;
  case BinaryDataType::FLOAT:
  case BinaryDataType::INT32:
  case BinaryDataType::UINT32:
    return 4;
  case BinaryDataType::DOUBLE:
  case BinaryDataType::INT64:
  case BinaryDataType::UINT64:
    return 8;
  default:
    throw std::invalid_argument("Binary calibration received an unsupported element type.");
  }
}

uint64_t CalibrationConstructionBytes(std::int32_t type, const rt_ns::Shape &shape) {
  const uint64_t elements = static_cast<uint64_t>(shape.product());
  const uint64_t temporary_size =
      type == BinaryDataType::FLOAT16 || type == BinaryDataType::BFLOAT16
          ? sizeof(float)
          : CalibrationElementSize(type);
  return elements * temporary_size;
}

rt_ns::KernelTuningParameters CalibrateBinary(const rt_ns::KernelTuningKey &key,
                                              const rt_ns::CpuExecutionDescriptor &execution,
                                              const rt_ns::CalibrationOptions &options,
                                              rt_ns::CalibrationReporter &reporter) {
  const uint64_t duration_ms =
      options.maximum_duration_ms == 0 ? uint64_t{250} : options.maximum_duration_ms;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(duration_ms);
  const auto keep_portable = [&](std::string reason) {
    reporter.AddDiagnostic(key.kernel + " calibration " + std::move(reason) +
                           "; kept portable parameters.");
    reporter.FinalizeCandidateDiagnostics();
    return MakeTuningDefaults(key.kernel, key.element_type);
  };
  const BinaryManifestEntry &entry = GetBinaryManifestEntry(key.kernel);
  const ONNX_LIGHT_NAMESPACE::NodeProto node = MakeCalibrationNode(entry, key.element_type);
  const rt_ns::KernelContext context(rt_ns::OpsetId(std::string(), entry.since_version));
  BinaryElementwiseKernel reference(node, context);
  BinaryElementwiseKernel candidate(node, context);
  reference.Configure(CalibrationParameters(key, kBinaryCalibrationProfiles[1]));

  const std::int32_t input_type = key.element_type;
  const auto same_type_signature = std::find_if(
      entry.signatures.begin(), entry.signatures.end(), [input_type](const auto &candidate) {
        return candidate.left == input_type && candidate.right == input_type;
      });
  if (same_type_signature == entry.signatures.end()) {
    throw std::invalid_argument("Binary calibration requires a same-type signature.");
  }
  std::vector<BinaryCalibrationCaseSpec> specs = {
      {"equal", input_type, input_type, same_type_signature->output, {65536}, {65536}, 65536},
      {"scalar", input_type, input_type, same_type_signature->output, {1048576}, {}, 1048576},
      {"rank4",
       input_type,
       input_type,
       same_type_signature->output,
       {512, 1, 8, 1},
       {1, 4, 1, 64},
       1048576},
  };
  if (entry.op == BinaryOperator::kPow) {
    for (const BinaryTypeSignature &signature : entry.signatures) {
      if (signature.left == input_type && signature.right != input_type) {
        specs.push_back({"mixed_" + std::to_string(static_cast<int32_t>(signature.right)),
                         signature.left,
                         signature.right,
                         signature.output,
                         {65536},
                         {65536},
                         65536});
      }
    }
  }
  const uint64_t memory_budget =
      options.maximum_memory_bytes == 0 ? uint64_t{64} << 20 : options.maximum_memory_bytes;
  if (memory_budget < (uint64_t{2} << 20)) {
    return keep_portable("memory budget is too small");
  }
  if (execution.effective_threads != static_cast<uint32_t>(ExecutionThreadCount())) {
    throw std::invalid_argument(
        "Binary calibration execution descriptor does not match the active executor.");
  }
  std::vector<BinaryCalibrationCaseData> cases;
  uint64_t retained_memory = 0;
  uint64_t peak_memory = 0;
  for (std::size_t index = 0; index < specs.size(); ++index) {
    if (std::chrono::steady_clock::now() >= deadline) {
      break;
    }
    const BinaryCalibrationCaseSpec &spec = specs[index];
    const uint64_t left_bytes =
        static_cast<uint64_t>(spec.left_shape.product()) * CalibrationElementSize(spec.left_type);
    const uint64_t right_bytes =
        static_cast<uint64_t>(spec.right_shape.product()) * CalibrationElementSize(spec.right_type);
    const uint64_t output_bytes = spec.output_elements * CalibrationElementSize(spec.output_type);
    const uint64_t case_storage = left_bytes + right_bytes + 2 * output_bytes;
    const uint64_t construction_peak = std::max(
        {case_storage, left_bytes + CalibrationConstructionBytes(spec.left_type, spec.left_shape),
         left_bytes + right_bytes +
             CalibrationConstructionBytes(spec.right_type, spec.right_shape)});
    if (construction_peak > memory_budget - retained_memory) {
      continue;
    }
    rt_ns::Tensor left =
        MakeCalibrationTensor(static_cast<int32_t>(spec.left_type), spec.left_shape, 17 + index);
    const bool integral_pow_exponents =
        key.kernel == "Pow" &&
        (spec.left_type == BinaryDataType::INT32 || spec.left_type == BinaryDataType::INT64);
    rt_ns::Tensor right =
        MakeCalibrationTensor(static_cast<int32_t>(spec.right_type), spec.right_shape, 29 + index,
                              integral_pow_exponents);
    if (std::chrono::steady_clock::now() >= deadline) {
      break;
    }
    rt_ns::Tensor reference_output = reference(left, right);
    rt_ns::Tensor candidate_output = rt_ns::MakeOutputTensor(
        reference_output.data_type, reference_output.shape, reference_output.size_bytes(), nullptr);
    cases.push_back({spec.name, std::move(left), std::move(right), std::move(reference_output),
                     std::move(candidate_output)});
    peak_memory = std::max(peak_memory, retained_memory + construction_peak);
    retained_memory += case_storage;
  }
  if (cases.size() != specs.size()) {
    return keep_portable("resource limits do not admit every benchmark case");
  }

  const BinaryCalibrationProfile *best = &kBinaryCalibrationProfiles[0];
  double best_score = std::numeric_limits<double>::infinity();
  std::vector<int64_t> serial_times(cases.size());
  std::vector<int64_t> serial_p90(cases.size());
  for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
    auto &benchmark_case = cases[case_index];
    std::array<int64_t, 3> samples{};
    for (int repetition = 0; repetition < 3; ++repetition) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return keep_portable("duration limit expired before serial sampling completed");
      }
      samples[repetition] = Measure([&] {
        reference(benchmark_case.left, benchmark_case.right, benchmark_case.reference_output);
      });
    }
    serial_times[case_index] = Median(samples);
    serial_p90[case_index] = *std::max_element(samples.begin(), samples.end());
  }

  bool duration_exhausted = false;
  for (const BinaryCalibrationProfile &profile : kBinaryCalibrationProfiles) {
    if (std::chrono::steady_clock::now() >= deadline) {
      break;
    }
    candidate.Configure(CalibrationParameters(key, profile));
    double score = 0.0;
    bool valid = true;
    bool small_regression = false;
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
      auto &benchmark_case = cases[case_index];
      if (std::chrono::steady_clock::now() >= deadline) {
        duration_exhausted = true;
        break;
      }
      candidate(benchmark_case.left, benchmark_case.right, benchmark_case.candidate_output);
      if (benchmark_case.reference_output.size_bytes() !=
              benchmark_case.candidate_output.size_bytes() ||
          std::memcmp(benchmark_case.reference_output.bytes(),
                      benchmark_case.candidate_output.bytes(),
                      benchmark_case.reference_output.size_bytes()) != 0) {
        valid = false;
        break;
      }
      std::array<int64_t, 3> samples{};
      uint64_t measured_ns = 0;
      for (int repetition = 0; repetition < 3; ++repetition) {
        if (std::chrono::steady_clock::now() >= deadline) {
          duration_exhausted = true;
          break;
        }
        samples[repetition] = Measure([&] {
          candidate(benchmark_case.left, benchmark_case.right, benchmark_case.candidate_output);
        });
        measured_ns += static_cast<uint64_t>(samples[repetition]);
      }
      if (duration_exhausted) {
        break;
      }
      reporter.RecordBenchmark(peak_memory, measured_ns);
      const int64_t candidate_time = Median(samples);
      score += static_cast<double>(candidate_time) /
               static_cast<double>(std::max<int64_t>(serial_times[case_index], 1));
      if (specs[case_index].output_elements == 65536 &&
          static_cast<double>(*std::max_element(samples.begin(), samples.end())) >
              static_cast<double>(serial_p90[case_index]) * 1.02) {
        small_regression = true;
      }
    }
    if (!duration_exhausted && valid && !small_regression && score < best_score) {
      best = &profile;
      best_score = score;
    }
    if (duration_exhausted) {
      break;
    }
  }

  candidate.Configure(CalibrationParameters(key, *best));
  if (std::chrono::steady_clock::now() < deadline) {
    reporter.ProfileCandidate([&] {
      auto &benchmark_case = cases.back();
      candidate(benchmark_case.left, benchmark_case.right, benchmark_case.candidate_output);
    });
  }
  reporter.AddDiagnostic(key.kernel + " selected binary profile '" + best->name + "'.");
  reporter.FinalizeCandidateDiagnostics();
  return CalibrationParameters(key, *best);
}

bool SupportsElementType(const BinaryManifestEntry &entry, int32_t element_type) {
  return std::any_of(entry.signatures.begin(), entry.signatures.end(),
                     [element_type](const BinaryTypeSignature &signature) {
                       return static_cast<int32_t>(signature.left) == element_type;
                     });
}

BinaryKernelDescriptor::Attributes
ParseBinaryAttributes(const ONNX_LIGHT_NAMESPACE::NodeProto &node) {
  BinaryKernelDescriptor::Attributes attributes;
  if (node.op_type() == "Mod") {
    attributes.mod_fmod = rt_ns::GetAttributeIntOrDefault(node, "fmod", 0);
  } else if (node.op_type() == "BitShift") {
    const std::string direction = rt_ns::GetRequiredAttributeString(node, "direction");
    if (direction == "RIGHT") {
      attributes.bitshift_direction = BinaryKernelDescriptor::Attributes::BitShiftDirection::kRight;
    } else if (direction != "LEFT") {
      throw std::invalid_argument(
          "onnx_light_cpu::BinaryElementwiseKernel: invalid BitShift direction.");
    }
  }
  return attributes;
}

} // namespace

BinaryElementwiseKernel::BinaryElementwiseKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node,
                                                 const rt_ns::KernelContext &ctx)
    : rt_ns::KernelBase(ctx),
      descriptor_(node.op_type(), ctx.opset.version, ParseBinaryAttributes(node)) {
  set_node(node);
}

void BinaryElementwiseKernel::RegisterTuningSchemas() {
  static std::once_flag once;
  std::call_once(once, [] {
    for (const BinaryManifestEntry &entry : GetBinaryManifest()) {
      std::set<int32_t> registered_types;
      for (const BinaryTypeSignature &signature : entry.signatures) {
        const int32_t element_type = static_cast<int32_t>(signature.left);
        if (registered_types.insert(element_type).second) {
          const rt_ns::KernelTuningKey key = MakeTuningKey(entry.op_type, element_type);
          rt_ns::RegisterKernelTuningSchema(rt_ns::KernelTuningSchema(
              MakeTuningDefaults(entry.op_type, element_type), ValidateTuning));
          rt_ns::RegisterKernelCalibrationFunction(key, CalibrateBinary);
        }
      }
    }
  });
}

rt_ns::KernelTuningKey BinaryElementwiseKernel::TuningKey(int32_t element_type) const {
  return SupportsElementType(descriptor_.manifest_entry(), element_type)
             ? MakeTuningKey(descriptor_.op_type(), element_type)
             : rt_ns::KernelTuningKey{};
}

void BinaryElementwiseKernel::Configure(const rt_ns::KernelTuningParameters &parameters) {
  if (!SupportsElementType(descriptor_.manifest_entry(), parameters.key.element_type) ||
      parameters.key != MakeTuningKey(descriptor_.op_type(), parameters.key.element_type)) {
    throw std::invalid_argument("Binary tuning parameters have an incompatible key.");
  }
  ValidateTuning(parameters);
  tuning_ = {
      static_cast<std::size_t>(parameters.Get<int64_t>(kBulkThresholdBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kBlockThresholdBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kScalarThresholdBytes)),
      static_cast<std::size_t>(parameters.Get<int64_t>(kTargetBlockBytes)),
      parameters.Get<int64_t>(kMaxParticipants) == 0 ? std::numeric_limits<std::int64_t>::max()
                                                     : parameters.Get<int64_t>(kMaxParticipants),
  };
}

rt_ns::Tensor BinaryElementwiseKernel::operator()(const rt_ns::Tensor &left,
                                                  const rt_ns::Tensor &right,
                                                  rt_ns::RuntimeContext *rt) const {
  const auto output_type =
      static_cast<rt_ns::DataType>(descriptor_.ResolveOutputType(left.data_type, right.data_type));
  const std::shared_ptr<const BinaryBroadcastPlan> plan =
      plan_cache_.GetOrCreate(descriptor_, left.data_type, right.data_type,
                              static_cast<std::int32_t>(output_type), left.shape, right.shape);
  const rt_ns::Shape output_shape(
      std::vector<std::int64_t>(plan->output_shape().begin(), plan->output_shape().end()));
  const std::size_t out_bytes =
      plan->inner_loop_elements() * plan->outer_block_count() * plan->adapter().output_size;
  rt_ns::Tensor output =
      rt != nullptr ? rt->MakeOutputTensor(0, output_type, output_shape, out_bytes)
                    : rt_ns::MakeOutputTensor(output_type, output_shape, out_bytes, nullptr);
  (*this)(left, right, output);
  return output;
}

void BinaryElementwiseKernel::operator()(const rt_ns::Tensor &left, const rt_ns::Tensor &right,
                                         rt_ns::Tensor &output) const {
  const auto output_type =
      static_cast<rt_ns::DataType>(descriptor_.ResolveOutputType(left.data_type, right.data_type));
  const std::shared_ptr<const BinaryBroadcastPlan> plan =
      plan_cache_.GetOrCreate(descriptor_, left.data_type, right.data_type,
                              static_cast<std::int32_t>(output_type), left.shape, right.shape);
  const rt_ns::Shape output_shape(
      std::vector<std::int64_t>(plan->output_shape().begin(), plan->output_shape().end()));
  if (output.data_type != static_cast<int32_t>(output_type) || output.shape != output_shape) {
    throw std::invalid_argument(
        "onnx_light_cpu::BinaryElementwiseKernel: output tensor metadata mismatch.");
  }
  plan->Execute(left.bytes(), right.bytes(), output.mutable_bytes(), tuning_);
}

void BinaryElementwiseKernel::Run(rt_ns::RuntimeContext &rt) {
  const auto &node = *node_;
  RecordKernelUsage(KernelName(descriptor_.op_type()));
  rt_ns::RequireInputCount(node, 2);
  rt_ns::RequireOutputCount(node, 1);
  const rt_ns::Tensor &left = rt_ns::GetInput(node, 0, rt.tensors());
  const rt_ns::Tensor &right = rt_ns::GetInput(node, 1, rt.tensors());
  rt_ns::SetOutput(node, 0, (*this)(left, right, &rt), rt);
}

void RegisterBinaryKernels() {
  BinaryElementwiseKernel::RegisterTuningSchemas();
  for (const BinaryManifestEntry &entry : GetBinaryManifest()) {
    rt_ns::NodeKernelFn factory =
        [](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
           rt_ns::RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
      return std::make_unique<BinaryElementwiseKernel>(node, rt.kernel_ctx());
    };
    KernelRegistration info;
    info.domain = "";
    info.op_type = std::string(entry.op_type);
    info.device = sym_ns::Device::kCPU;
    info.kernel_name = KernelName(entry.op_type);
    for (const BinaryTypeSignature &signature : entry.signatures) {
      const auto type = static_cast<rt_ns::DataType>(signature.left);
      if (std::find(info.types.begin(), info.types.end(), type) == info.types.end()) {
        info.types.push_back(type);
      }
    }
    info.since_version = entry.since_version;
    RegisterKernel(std::move(info), std::move(factory));
  }
}

} // namespace onnx_light_cpu
