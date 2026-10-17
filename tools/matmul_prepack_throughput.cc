// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/gemm/gemm_common.h"
#include "onnx_light_cpu/kernels/math/matmul_kernel.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/value_store.h"
#include "onnx_core/runtime/tuning/cpu_executor.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;
using Clock = std::chrono::steady_clock;

rt::Tensor MakeTensor(rt::DataType dtype, const std::string &name, const rt::Shape &shape,
                      const std::vector<float> &values) {
  if (dtype == rt::DataType::FLOAT16) {
    return rt::MakeFloat16Tensor(name, shape, values);
  }
  if (dtype == rt::DataType::BFLOAT16) {
    return rt::MakeBfloat16Tensor(name, shape, values);
  }
  return rt::Tensor::FromFloat(name, shape, values);
}

void Measure(rt::DataType dtype, std::int64_t k, std::int64_t n, bool persistent) {
  const rt::KernelContext context{rt::DefaultOpset(18)};
  rt::RuntimeContext runtime(context);
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("MatMul");
  node.add_input("A");
  node.add_input("B");
  node.add_output("Y");
  onnx_light_cpu::MatMulKernel kernel(context);
  kernel.set_node(node);
  std::vector<float> weights(static_cast<std::size_t>(k * n));
  for (std::size_t index = 0; index < weights.size(); ++index) {
    weights[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 128.0f;
  }
  runtime.Put("B", MakeTensor(dtype, "B", {k, n}, weights));
  const auto begin_prepare = Clock::now();
  if (persistent) {
    kernel.Prepare(runtime, {"B"});
  }
  const double preparation_seconds =
      persistent ? std::chrono::duration<double>(Clock::now() - begin_prepare).count() : 0.0;
  for (const std::int64_t m : {1, 128}) {
    std::vector<float> values(static_cast<std::size_t>(m * k));
    for (std::size_t index = 0; index < values.size(); ++index) {
      values[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 32.0f;
    }
    runtime.Put("A", MakeTensor(dtype, "A", {m, k}, values));
    for (int warmup = 0; warmup < 5; ++warmup) {
      kernel.Run(runtime);
    }
    const auto expected = runtime.Get("Y");
    std::array<double, 11> seconds;
    for (double &sample : seconds) {
      const auto start = Clock::now();
      kernel.Run(runtime);
      sample = std::chrono::duration<double>(Clock::now() - start).count();
      if (std::memcmp(expected.bytes(), runtime.Get("Y").bytes(), expected.size_bytes()) != 0) {
        throw std::runtime_error("MatMul output changed between repetitions.");
      }
    }
    std::sort(seconds.begin(), seconds.end());
    const char *type = dtype == rt::DataType::FLOAT     ? "fp32"
                       : dtype == rt::DataType::FLOAT16 ? "fp16"
                                                        : "bf16";
    std::printf("%s,%lld,%lld,%lld,%s,%.9g,%.9g,%.9g,%zu\n", type, static_cast<long long>(m),
                static_cast<long long>(k), static_cast<long long>(n),
                persistent ? "prepared" : "dynamic", preparation_seconds, seconds[5], seconds[9],
                runtime.value_store().prepared_bytes());
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 2) {
    throw std::invalid_argument("Usage: matmul_prepack_throughput [threads]");
  }
  const int threads = argc > 1 ? std::stoi(argv[1]) : 1;
  if (threads <= 0) {
    throw std::invalid_argument("Thread count must be positive.");
  }
  const auto executor =
      rt::GlobalCpuExecutorRegistry().Acquire(rt::CpuExecutionPolicy{.num_threads = threads});
  const rt::CpuExecutorScope scope(executor.get());
  std::printf("# effective_isa=%d effective_threads=%u\n",
              static_cast<int>(onnx_light_cpu::EffectiveGemmSimdLevel()),
              executor->effective_threads());
  std::puts("dtype,M,K,N,mode,preparation_seconds,median_seconds,p90_seconds,prepared_bytes");
  for (const auto dtype : {rt::DataType::FLOAT, rt::DataType::FLOAT16, rt::DataType::BFLOAT16}) {
    for (const auto [k, n] : {std::pair{512, 32000}, std::pair{512, 1376}, std::pair{1376, 512},
                              std::pair{512, 1024}, std::pair{512, 512}}) {
      for (const bool persistent : {false, true}) {
        Measure(dtype, k, n, persistent);
      }
    }
  }
}
