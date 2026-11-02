// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/math/matmul_kernel.h"
#include "onnx_light_cpu/kernels/execution.h"

#include "ml_cpu_compute/checked_arithmetic.h"
#include "ml_cpu_compute/math/gemm/gemm_plan.h"
#include "ml_cpu_compute/math/half_conversion.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/value_store.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace onnx_light_cpu {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using NodeProto = ONNX_LIGHT_NAMESPACE::NodeProto;
using DataType = rt_ns::DataType;
using RuntimeContext = rt_ns::RuntimeContext;
using Shape = rt_ns::Shape;
using Tensor = rt_ns::Tensor;

namespace {

bool CanPrepareWideWeights(const Tensor &b) {
  if (b.data_type != static_cast<int>(DataType::FLOAT) || b.shape.size() != 2 ||
      b.shape[0] != 512 || b.shape[1] < 32000 || b.shape[1] % 16 != 0) {
    return false;
  }
  if (!std::in_range<std::size_t>(b.shape[1])) {
    throw std::invalid_argument(
        "onnx_light_cpu::MatMul: prepared weight dimension overflows size_t.");
  }
  const std::size_t elements = ml_cpu_compute::CheckedMultiply(
      static_cast<std::size_t>(b.shape[0]), static_cast<std::size_t>(b.shape[1]), "MatMul",
      "prepared weight elements");
  const std::size_t bytes =
      ml_cpu_compute::CheckedByteSize(elements, sizeof(float), "MatMul", "prepared weights");
  if (b.size_bytes() != bytes) {
    throw std::invalid_argument("onnx_light_cpu::MatMul: prepared weight storage size mismatch.");
  }
  return true;
}

std::string PreparedWeightsKey(const std::string &name, const Tensor &b) {
  std::uint64_t digest = 14695981039346656037ULL;
  for (std::size_t index = 0; index < b.size_bytes(); ++index) {
    digest = (digest ^ b.bytes()[index]) * 1099511628211ULL;
  }
  return "MatMul:fp32-panel16-v1:" + std::to_string(name.size()) + ":" + name +
         ":N=" + std::to_string(b.shape[1]) + ":digest=" + std::to_string(digest);
}

std::vector<std::size_t> ShapeAsSize(const Shape &shape) {
  std::vector<std::size_t> result;
  result.reserve(shape.size());
  for (const std::int64_t dimension : shape) {
    if (dimension < 0) {
      throw std::invalid_argument(
          "onnx_light_cpu::MatMulKernel: dynamic dimensions are unsupported.");
    }
    result.push_back(static_cast<std::size_t>(dimension));
  }
  return result;
}

Shape OutputShape(std::span<const std::size_t> shape) {
  Shape result;
  result.reserve(shape.size());
  for (const std::size_t dimension : shape) {
    result.push_back(static_cast<std::int64_t>(dimension));
  }
  return result;
}

std::size_t ElementCount(std::span<const std::size_t> shape) {
  std::size_t count = 1;
  for (const std::size_t dimension : shape) {
    count *= dimension;
  }
  return count;
}

template <typename T>
Tensor ComputeFloating(const Tensor &a, const Tensor &b, RuntimeContext *rt,
                       const ml_cpu_compute::MatMulPlan<T> &plan) {
  const Shape output_shape = OutputShape(plan.output_shape());
  const std::size_t output_elements = ElementCount(plan.output_shape());
  const std::size_t bytes = output_elements * sizeof(T);
  Tensor y = rt != nullptr ? rt->MakeOutputTensor(0, a.data_type, output_shape, bytes)
                           : rt_ns::MakeOutputTensor(a.data_type, output_shape, bytes, nullptr);
  const auto *a_data = reinterpret_cast<const T *>(a.bytes());
  const auto *b_data = reinterpret_cast<const T *>(b.bytes());
  auto *y_data = reinterpret_cast<T *>(y.mutable_bytes());
  plan.Execute(a_data, b_data, y_data);
  return y;
}

Tensor ComputeHalf(const Tensor &a, const Tensor &b, RuntimeContext *rt, bool bfloat16,
                   const ml_cpu_compute::MatMulPlan<float> &plan) {
  const auto *a_bits = reinterpret_cast<const std::uint16_t *>(a.bytes());
  const auto *b_bits = reinterpret_cast<const std::uint16_t *>(b.bytes());
  const Shape output_shape = OutputShape(plan.output_shape());
  const std::size_t output_elements = ElementCount(plan.output_shape());
  Tensor y = rt != nullptr
                 ? rt->MakeOutputTensor(0, a.data_type, output_shape,
                                        output_elements * sizeof(std::uint16_t))
                 : rt_ns::MakeOutputTensor(a.data_type, output_shape,
                                           output_elements * sizeof(std::uint16_t), nullptr);
  const ml_cpu_compute::GemmPlan<float> &float_plan = plan.gemm_plan();
  const ml_cpu_compute::GemmHalfPlan half_plan(ml_cpu_compute::GemmHalfPlanOptions{
      bfloat16, float_plan.trans_a(), float_plan.trans_b(), float_plan.m(), float_plan.n(),
      float_plan.k(), float_plan.alpha()});
  ml_cpu_compute::GemmEpilogue<float> epilogue;
  epilogue.output_conversion = bfloat16 ? ml_cpu_compute::GemmOutputConversion::kBFloat16
                                        : ml_cpu_compute::GemmOutputConversion::kFloat16;
  epilogue.converted_output = reinterpret_cast<std::uint16_t *>(y.mutable_bytes());
  rt_ns::detail::TemporaryTypedBuffer<float> output(
      output_elements, rt != nullptr ? rt->execution_allocator() : nullptr,
      "MatMul FP32 workspace");
  plan.ExecuteHalf(half_plan, a_bits, b_bits, epilogue, output.data());
  return y;
}

} // namespace

struct MatMulKernel::PreparedWeights {
  rt_ns::ValueStore::Slot slot;
  const rt_ns::ValueStore *store;
  const std::uint8_t *source;
  Shape shape;

  bool Matches(const Tensor &b, const RuntimeContext &rt) const {
    return store == &rt.value_store() && source == b.bytes() && shape == b.shape;
  }
};

struct MatMulKernel::MatMulPlanCache {
  int data_type = 0;
  std::vector<std::size_t> a_shape;
  std::vector<std::size_t> b_shape;
  std::unique_ptr<ml_cpu_compute::MatMulPlan<float>> plan_f32;
  std::unique_ptr<ml_cpu_compute::MatMulPlan<double>> plan_f64;

  template <typename T>
  const ml_cpu_compute::MatMulPlan<T> &GetOrBuild(int type,
                                                  std::span<const std::size_t> current_a_shape,
                                                  std::span<const std::size_t> current_b_shape) {
    std::unique_ptr<ml_cpu_compute::MatMulPlan<T>> &slot = Slot<T>();
    const bool shapes_match = a_shape.size() == current_a_shape.size() &&
                              b_shape.size() == current_b_shape.size() &&
                              std::equal(a_shape.begin(), a_shape.end(), current_a_shape.begin()) &&
                              std::equal(b_shape.begin(), b_shape.end(), current_b_shape.begin());
    if (slot == nullptr || data_type != type || !shapes_match) {
      slot = std::make_unique<ml_cpu_compute::MatMulPlan<T>>(current_a_shape, current_b_shape);
      data_type = type;
      a_shape.assign(current_a_shape.begin(), current_a_shape.end());
      b_shape.assign(current_b_shape.begin(), current_b_shape.end());
    }
    return *slot;
  }

private:
  template <typename T> std::unique_ptr<ml_cpu_compute::MatMulPlan<T>> &Slot() {
    if constexpr (std::is_same_v<T, float>) {
      return plan_f32;
    } else {
      static_assert(std::is_same_v<T, double>);
      return plan_f64;
    }
  }
};

MatMulKernel::MatMulKernel(const rt_ns::KernelContext &ctx)
    : KernelBase(ctx), plan_cache_(std::make_unique<MatMulPlanCache>()) {}

MatMulKernel::~MatMulKernel() = default;

bool MatMulKernel::HasPreparations(const std::unordered_set<std::string> &immutable_inputs) const {
  return ml_cpu_compute::EffectiveGemmSimdLevel() >= ml_cpu_compute::SimdLevel::kAVX2 &&
         node_ != nullptr && node_->input_size() == 2 && immutable_inputs.contains(node_->input(1));
}

void MatMulKernel::DeclarePreparations(const std::unordered_set<std::string> &immutable_inputs,
                                       std::vector<rt_ns::KernelPreparation> &preparations) {
  if (!HasPreparations(immutable_inputs)) {
    return;
  }
  const std::string name = node_->input(1);
  rt_ns::KernelPreparation preparation;
  preparation.info.source_inputs = {name};
  preparation.info.representation = "MatMul:fp32-panel16-v1";
  preparation.info.output_slots = {preparation.info.representation + ":" +
                                   std::to_string(name.size()) + ":" + name};
  preparation.estimate_memory = [this, name](const RuntimeContext &rt) {
    const Tensor &b = rt.Get(name);
    std::size_t prepared_bytes = 0;
    if (CanPrepareWideWeights(b) &&
        !(prepared_weights_ != nullptr && prepared_weights_->Matches(b, rt))) {
      const auto slot = rt.value_store().Find(PreparedWeightsKey(name, b));
      if (!slot || !rt.value_store().IsReady(*slot)) {
        prepared_bytes = b.size_bytes();
      }
    }
    return rt_ns::KernelPreparationMemory{b.size_bytes(), 0, prepared_bytes};
  };
  preparation.execute = [this, name](RuntimeContext &rt) {
    const Tensor &b = rt.Get(name);
    if (!CanPrepareWideWeights(b)) {
      prepared_weights_.reset();
      return;
    }
    if (prepared_weights_ != nullptr && prepared_weights_->Matches(b, rt)) {
      return;
    }
    auto &store = rt.value_store();
    const auto slot = store.Bind(PreparedWeightsKey(name, b));
    if (!store.IsReady(slot)) {
      Tensor packed = store.AllocateTensor(b.data_type, b.shape, b.size_bytes());
      ml_cpu_compute::detail::PackGemmWideB(static_cast<std::size_t>(b.shape[1]),
                                            static_cast<std::size_t>(b.shape[0]),
                                            reinterpret_cast<const float *>(b.bytes()),
                                            reinterpret_cast<float *>(packed.mutable_bytes()));
      store.Publish(slot, rt_ns::RuntimeValue(std::move(packed)));
    }
    prepared_weights_ =
        std::make_unique<PreparedWeights>(PreparedWeights{slot, &store, b.bytes(), b.shape});
  };
  preparations.push_back(std::move(preparation));
}

Tensor MatMulKernel::Compute(const Tensor &a, const Tensor &b, RuntimeContext *rt,
                             MatMulPlanCache *cache, const PreparedWeights *prepared) {
  if (a.data_type != b.data_type) {
    throw std::invalid_argument("onnx_light_cpu::MatMulKernel: inputs must share the same dtype.");
  }
  const auto a_shape = ShapeAsSize(a.shape);
  const auto b_shape = ShapeAsSize(b.shape);
  switch (static_cast<DataType>(a.data_type)) {
  case DataType::FLOAT: {
    std::optional<ml_cpu_compute::MatMulPlan<float>> transient;
    const ml_cpu_compute::MatMulPlan<float> *plan;
    if (cache == nullptr) {
      transient.emplace(a_shape, b_shape);
      plan = &*transient;
    } else {
      plan = &cache->GetOrBuild<float>(a.data_type, a_shape, b_shape);
    }
    if (prepared != nullptr && rt != nullptr && prepared->Matches(b, *rt) &&
        plan->batch_count() == 1 && plan->gemm_plan().m() >= 16 && plan->gemm_plan().m() <= 512) {
      const auto &gemm = plan->gemm_plan();
      const Shape output_shape = OutputShape(plan->output_shape());
      const std::size_t bytes = ElementCount(plan->output_shape()) * sizeof(float);
      Tensor y = rt->MakeOutputTensor(0, a.data_type, output_shape, bytes);
      const Tensor &packed = rt->value_store().Get(prepared->slot).tensor;
      rt->RecordKernelUsage(ml_cpu_compute::EffectiveGemmSimdLevel() >=
                                    ml_cpu_compute::SimdLevel::kAVX512
                                ? "MatMul.prepared_b.avx512"
                                : "MatMul.prepared_b.avx2");
      ml_cpu_compute::detail::GemmWidePreparedB(
          gemm.m(), gemm.n(), gemm.k(), 1.0f, reinterpret_cast<const float *>(a.bytes()),
          reinterpret_cast<const float *>(packed.bytes()), 0.0f, nullptr,
          reinterpret_cast<float *>(y.mutable_bytes()), gemm.blocking().mr);
      return y;
    }
    return ComputeFloating<float>(a, b, rt, *plan);
  }
  case DataType::DOUBLE: {
    std::optional<ml_cpu_compute::MatMulPlan<double>> transient;
    const ml_cpu_compute::MatMulPlan<double> *plan;
    if (cache == nullptr) {
      transient.emplace(a_shape, b_shape);
      plan = &*transient;
    } else {
      plan = &cache->GetOrBuild<double>(a.data_type, a_shape, b_shape);
    }
    return ComputeFloating<double>(a, b, rt, *plan);
  }
  case DataType::FLOAT16:
  case DataType::BFLOAT16: {
    std::optional<ml_cpu_compute::MatMulPlan<float>> transient;
    const ml_cpu_compute::MatMulPlan<float> *plan;
    if (cache == nullptr) {
      transient.emplace(a_shape, b_shape);
      plan = &*transient;
    } else {
      plan = &cache->GetOrBuild<float>(a.data_type, a_shape, b_shape);
    }
    return ComputeHalf(a, b, rt, static_cast<DataType>(a.data_type) == DataType::BFLOAT16, *plan);
  }
  default:
    throw std::invalid_argument(
        "onnx_light_cpu::MatMulKernel: unsupported data type; expected FLOAT, DOUBLE, FLOAT16 or "
        "BFLOAT16.");
  }
}

Tensor MatMulKernel::operator()(const Tensor &a, const Tensor &b, RuntimeContext *rt) const {
  const RuntimeExecutionScope execution_scope;
  return Compute(a, b, rt, nullptr, nullptr);
}

void MatMulKernel::Run(RuntimeContext &rt) {
  const RuntimeExecutionScope execution_scope;
  rt.RecordKernelUsage(kName);
  const NodeProto &node = *node_;
  rt_ns::RequireOutputCount(node, 1);
  const Tensor &a = rt_ns::GetInput(node, 0, rt.tensors());
  const Tensor &b = rt_ns::GetInput(node, 1, rt.tensors());
  rt_ns::SetOutput(node, 0, Compute(a, b, &rt, plan_cache_.get(), prepared_weights_.get()), rt);
}

void RegisterMatMulKernel() {
  rt_ns::NodeKernelFn matmul =
      [](const NodeProto &node, RuntimeContext &,
         const rt_ns::KernelContext &kernel_context) -> std::unique_ptr<rt_ns::KernelBase> {
    auto kernel = std::make_unique<MatMulKernel>(kernel_context);
    kernel->set_node(node);
    return kernel;
  };
  KernelRegistration info;
  info.domain = "";
  info.op_type = "MatMul";
  info.device = sym_ns::Device::kCPU;
  info.kernel_name = MatMulKernel::kName;
  info.types = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16};
  RegisterKernel(std::move(info), std::move(matmul));
}

} // namespace onnx_light_cpu
