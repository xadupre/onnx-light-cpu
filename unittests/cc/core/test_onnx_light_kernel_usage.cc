// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/kernel_usage.h"

#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"
#include "onnx_light_cpu/kernels/attention/attention_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/bias_gelu_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/cdist_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/group_query_attention_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_bias_gelu_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_cdist_kernel.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_group_query_attention_kernel.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/logical/not_kernel.h"
#include "onnx_light_cpu/kernels/math/abs_kernel.h"
#include "onnx_light_cpu/kernels/math/exp_log_kernel.h"
#include "onnx_light_cpu/kernels/math/gemm_kernel.h"
#include "onnx_light_cpu/kernels/math/integer_matmul_kernel.h"
#include "onnx_light_cpu/kernels/math/matmul_kernel.h"
#include "onnx_light_cpu/kernels/math/normalization_kernel.h"
#include "onnx_light_cpu/kernels/math/rms_normalization_kernel.h"
#include "onnx_light_cpu/kernels/math/sigmoid_softmax_kernel.h"
#include "onnx_light_cpu/kernels/math/swiglu_kernel.h"
#include "onnx_light_cpu/kernels/register_kernels.h"
#include "onnx_light_cpu/kernels/traditionalml/tree_ensemble_kernel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <barrier>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

rt_ns::KernelContext MakeContext() { return rt_ns::KernelContext(rt_ns::DefaultOpset(18)); }

ONNX_LIGHT_NAMESPACE::NodeProto MakeNode(const std::string &op_type) {
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type(op_type);
  node.add_input("x");
  node.add_output("y");
  return node;
}

void RunAbs(rt_ns::RuntimeContext &runtime) {
  const auto node = MakeNode("Abs");
  onnx_light_cpu::AbsKernel kernel(node, MakeContext());
  runtime.Clear();
  runtime.Set("x", rt_ns::Tensor::FromFloat("x", {2}, {-2.0f, 3.0f}));
  kernel.Run(runtime);
  EXPECT_FLOAT_EQ(runtime.Get("y").AsFloat()[0], 2.0f);
  EXPECT_FLOAT_EQ(runtime.Get("y").AsFloat()[1], 3.0f);
}

// Every onnx-light-cpu kernel exposes a unique, library-qualified name so the
// kernel that a model dispatches to can be told apart from onnx-light's
// built-in one.
TEST(OnnxLightKernelUsage, KernelNamesAreLibraryQualified) {
  EXPECT_STREQ(onnx_light_cpu::AbsKernel::kName, "onnx_light_cpu::Abs");
  EXPECT_STREQ(onnx_light_cpu::AttentionKernel::kName, "onnx_light_cpu::Attention");
  EXPECT_STREQ(onnx_light_cpu::BatchNormalizationKernel::kName,
               "onnx_light_cpu::BatchNormalization");
  EXPECT_STREQ(onnx_light_cpu::BiasGeluKernel::kName, "onnx_light_cpu::BiasGelu");
  EXPECT_STREQ(onnx_light_cpu::CDistKernel::kName, "onnx_light_cpu::CDist");
  EXPECT_STREQ(onnx_light_cpu::GroupQueryAttentionKernel::kName,
               "onnx_light_cpu::GroupQueryAttention");
  EXPECT_STREQ(onnx_light_cpu::NaiveBiasGeluKernel::kName, "onnx_light_cpu::NaiveBiasGelu");
  EXPECT_STREQ(onnx_light_cpu::NaiveCDistKernel::kName, "onnx_light_cpu::NaiveCDist");
  EXPECT_STREQ(onnx_light_cpu::NaiveGroupQueryAttentionKernel::kName,
               "onnx_light_cpu::NaiveGroupQueryAttention");
  EXPECT_STREQ(onnx_light_cpu::ExpKernel::kName, "onnx_light_cpu::Exp");
  EXPECT_STREQ(onnx_light_cpu::LogKernel::kName, "onnx_light_cpu::Log");
  EXPECT_STREQ(onnx_light_cpu::GemmKernel::kName, "onnx_light_cpu::Gemm");
  EXPECT_STREQ(onnx_light_cpu::GroupNormalizationKernel::kName,
               "onnx_light_cpu::GroupNormalization");
  EXPECT_STREQ(onnx_light_cpu::InstanceNormalizationKernel::kName,
               "onnx_light_cpu::InstanceNormalization");
  EXPECT_STREQ(onnx_light_cpu::LayerNormalizationKernel::kName,
               "onnx_light_cpu::LayerNormalization");
  EXPECT_STREQ(onnx_light_cpu::LpNormalizationKernel::kName, "onnx_light_cpu::LpNormalization");
  EXPECT_STREQ(onnx_light_cpu::MatMulKernel::kName, "onnx_light_cpu::MatMul");
  EXPECT_STREQ(onnx_light_cpu::MatMulIntegerKernel::kName, "onnx_light_cpu::MatMulInteger");
  EXPECT_STREQ(onnx_light_cpu::QLinearMatMulKernel::kName, "onnx_light_cpu::QLinearMatMul");
  EXPECT_STREQ(onnx_light_cpu::NotKernel::kName, "onnx_light_cpu::Not");
  EXPECT_STREQ(onnx_light_cpu::MeanVarianceNormalizationKernel::kName,
               "onnx_light_cpu::MeanVarianceNormalization");
  EXPECT_STREQ(onnx_light_cpu::RmsNormalizationKernel::kName, "onnx_light_cpu::RMSNormalization");
  EXPECT_STREQ(onnx_light_cpu::SigmoidKernel::kName, "onnx_light_cpu::Sigmoid");
  EXPECT_STREQ(onnx_light_cpu::SoftmaxKernel::kName, "onnx_light_cpu::Softmax");
  EXPECT_STREQ(onnx_light_cpu::SwiGLUKernel::kName, "onnx_light_cpu::SwiGLU");
  EXPECT_STREQ(onnx_light_cpu::TreeEnsembleKernel::kName, "onnx_light_cpu::TreeEnsemble");
}

// ``RegisteredKernelNames`` maps every overridden ONNX op_type to the
// library-qualified name of the accelerated kernel installed for it. It is
// derived from ``CollectRegisteredKernels()``'s structured inventory, so
// entries are ordered by ``domain`` and then ``op_type`` for the public name map.
TEST(OnnxLightKernelUsage, RegisteredKernelNames) {
  std::vector<std::pair<std::string, std::string>> expected;
  for (const onnx_light_cpu::KernelRegistration &record :
       onnx_light_cpu::CollectRegisteredKernels()) {
    expected.emplace_back(record.op_type, record.kernel_name);
  }
  EXPECT_EQ(onnx_light_cpu::RegisteredKernelNames(), expected);
}

TEST(OnnxLightKernelUsage, RecordingIsDisabledByDefault) {
  rt_ns::RuntimeContext runtime(MakeContext());
  EXPECT_FALSE(runtime.kernel_usage_enabled());
  for (std::size_t i = 0; i < 2 * rt_ns::RuntimeContext::kKernelUsageLimit; ++i) {
    RunAbs(runtime);
  }
  EXPECT_TRUE(runtime.GetKernelUsage().empty());
}

class OnnxLightKernelUsageRecording : public ::testing::Test {
protected:
  rt_ns::RuntimeContext runtime{MakeContext()};
  void SetUp() override { runtime.set_kernel_usage_enabled(true); }
};

// The usage log records names in invocation order and can be cleared.
TEST_F(OnnxLightKernelUsageRecording, RecordAndClear) {
  EXPECT_TRUE(runtime.GetKernelUsage().empty());

  RunAbs(runtime);
  const auto node = MakeNode("Exp");
  onnx_light_cpu::ExpKernel kernel(node, MakeContext());
  kernel.Run(runtime);
  const std::vector<std::string> expected = {"onnx_light_cpu::Abs", "onnx_light_cpu::Exp"};
  EXPECT_EQ(runtime.GetKernelUsage(), expected);

  runtime.Clear();
  EXPECT_EQ(runtime.GetKernelUsage(), expected);
  runtime.ClearKernelUsage();
  EXPECT_TRUE(runtime.GetKernelUsage().empty());
  EXPECT_TRUE(runtime.kernel_usage_enabled());
  RunAbs(runtime);
  EXPECT_EQ(runtime.GetKernelUsage(), std::vector<std::string>({"onnx_light_cpu::Abs"}));
}

TEST_F(OnnxLightKernelUsageRecording, RecordingCanBeDisabled) {
  runtime.set_kernel_usage_enabled(false);
  RunAbs(runtime);
  EXPECT_TRUE(runtime.GetKernelUsage().empty());

  runtime.set_kernel_usage_enabled(true);
  RunAbs(runtime);
  EXPECT_EQ(runtime.GetKernelUsage(), std::vector<std::string>({"onnx_light_cpu::Abs"}));

  runtime.set_kernel_usage_enabled(false);
  for (std::size_t i = 0; i < 2 * rt_ns::RuntimeContext::kKernelUsageLimit; ++i) {
    RunAbs(runtime);
  }
  EXPECT_EQ(runtime.GetKernelUsage(), std::vector<std::string>({"onnx_light_cpu::Abs"}));
  runtime.ClearKernelUsage();
  EXPECT_FALSE(runtime.kernel_usage_enabled());
  RunAbs(runtime);
  EXPECT_TRUE(runtime.GetKernelUsage().empty());
}

TEST_F(OnnxLightKernelUsageRecording, StorageIsBoundedAndClearRestartsRecording) {
  EXPECT_EQ(rt_ns::RuntimeContext::kKernelUsageLimit, 1024u);
  std::vector<std::string> expected;
  for (std::size_t i = 0; i < 2 * rt_ns::RuntimeContext::kKernelUsageLimit; ++i) {
    const std::string name = "kernel_" + std::to_string(i);
    runtime.RecordKernelUsage(name);
    if (i < rt_ns::RuntimeContext::kKernelUsageLimit) {
      expected.push_back(name);
    }
  }
  EXPECT_EQ(runtime.GetKernelUsage(), expected);
  runtime.set_kernel_usage_enabled(false);
  runtime.set_kernel_usage_enabled(true);
  RunAbs(runtime);
  EXPECT_EQ(runtime.GetKernelUsage(), expected);

  runtime.ClearKernelUsage();
  RunAbs(runtime);
  EXPECT_EQ(runtime.GetKernelUsage(), std::vector<std::string>({"onnx_light_cpu::Abs"}));
}

TEST_F(OnnxLightKernelUsageRecording, SnapshotOwnsNamesAndDoesNotConsumeLog) {
  std::string name = "original";
  runtime.RecordKernelUsage(name);
  name.assign("changed");
  auto snapshot = runtime.GetKernelUsage();
  ASSERT_EQ(snapshot, std::vector<std::string>({"original"}));
  snapshot[0] = "independent";
  EXPECT_EQ(runtime.GetKernelUsage(), std::vector<std::string>({"original"}));
  runtime.ClearKernelUsage();
  EXPECT_EQ(snapshot, std::vector<std::string>({"independent"}));
}

TEST_F(OnnxLightKernelUsageRecording, ConcurrentRecording) {
  constexpr int threads = 4;
  constexpr int invocations = 128;
  std::barrier start(threads);
  std::vector<std::thread> workers;
  for (int thread = 0; thread < threads; ++thread) {
    workers.emplace_back([&, thread] {
      start.arrive_and_wait();
      for (int i = 0; i < invocations; ++i) {
        runtime.RecordKernelUsage(std::to_string(thread) + ":" + std::to_string(i));
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  const auto names = runtime.GetKernelUsage();
  ASSERT_EQ(names.size(), threads * invocations);
  for (int thread = 0; thread < threads; ++thread) {
    const std::string prefix = std::to_string(thread) + ":";
    int invocation = 0;
    for (const auto &name : names) {
      if (name.starts_with(prefix)) {
        EXPECT_EQ(name, prefix + std::to_string(invocation++));
      }
    }
    EXPECT_EQ(invocation, invocations);
  }
}

TEST_F(OnnxLightKernelUsageRecording, ConcurrentRecordingIsBounded) {
  constexpr int threads = 4;
  std::barrier start(threads);
  std::vector<std::thread> workers;
  for (int thread = 0; thread < threads; ++thread) {
    workers.emplace_back([&] {
      start.arrive_and_wait();
      for (std::size_t i = 0; i < rt_ns::RuntimeContext::kKernelUsageLimit; ++i) {
        runtime.RecordKernelUsage("kernel");
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  EXPECT_EQ(runtime.GetKernelUsage(),
            std::vector<std::string>(rt_ns::RuntimeContext::kKernelUsageLimit, "kernel"));
}

TEST_F(OnnxLightKernelUsageRecording, ConcurrentResetRetrievalAndDisable) {
  constexpr int threads = 4;
  std::barrier phase(threads + 1);
  std::vector<std::thread> workers;
  for (int thread = 0; thread < threads; ++thread) {
    workers.emplace_back([&] {
      phase.arrive_and_wait();
      for (std::size_t i = 0; i < rt_ns::RuntimeContext::kKernelUsageLimit; ++i) {
        runtime.RecordKernelUsage("kernel");
      }
      phase.arrive_and_wait();
      for (int i = 0; i < 128; ++i) {
        runtime.RecordKernelUsage("disabled");
      }
    });
  }
  phase.arrive_and_wait();
  for (int i = 0; i < 128; ++i) {
    const auto snapshot = runtime.GetKernelUsage();
    EXPECT_LE(snapshot.size(), rt_ns::RuntimeContext::kKernelUsageLimit);
    EXPECT_TRUE(std::all_of(snapshot.begin(), snapshot.end(),
                            [](const auto &name) { return name == "kernel"; }));
    runtime.ClearKernelUsage();
  }
  runtime.set_kernel_usage_enabled(false);
  const auto stopped = runtime.GetKernelUsage();
  phase.arrive_and_wait();
  for (auto &worker : workers) {
    worker.join();
  }
  EXPECT_EQ(runtime.GetKernelUsage(), stopped);
  runtime.ClearKernelUsage();
  EXPECT_TRUE(runtime.GetKernelUsage().empty());
}

TEST(OnnxLightKernelUsage, ConcurrentSessionsAreIsolated) {
  rt_ns::RuntimeContext abs_runtime(MakeContext());
  rt_ns::RuntimeContext exp_runtime(MakeContext());
  rt_ns::RuntimeContext disabled_runtime(MakeContext());
  onnx_light_cpu::RegisterKernelForSession(abs_runtime, "", "Abs");
  onnx_light_cpu::RegisterKernelForSession(exp_runtime, "", "Exp");
  onnx_light_cpu::RegisterKernelForSession(disabled_runtime, "", "Abs");
  abs_runtime.set_kernel_usage_enabled(true);
  exp_runtime.set_kernel_usage_enabled(true);

  ONNX_LIGHT_NAMESPACE::GraphProto abs_graph;
  abs_graph.add_input()->set_name("x");
  abs_graph.add_output()->set_name("y");
  *abs_graph.add_node() = MakeNode("Abs");
  ONNX_LIGHT_NAMESPACE::GraphProto exp_graph;
  exp_graph.add_input()->set_name("x");
  exp_graph.add_output()->set_name("y");
  *exp_graph.add_node() = MakeNode("Exp");
  rt_ns::SubgraphSession abs_session(abs_runtime, abs_graph);
  rt_ns::SubgraphSession exp_session(exp_runtime, exp_graph);
  rt_ns::SubgraphSession disabled_session(disabled_runtime, abs_graph);
  constexpr int invocations = 128;
  std::barrier start(3);
  auto run = [&](rt_ns::SubgraphSession &session, rt_ns::RuntimeContext &runtime, float expected) {
    start.arrive_and_wait();
    for (int i = 0; i < invocations; ++i) {
      std::vector<std::pair<std::string, rt_ns::Tensor>> inputs;
      inputs.emplace_back("x", rt_ns::Tensor::FromFloat("x", {1}, {0.0f}));
      const auto outputs = session.Run(std::move(inputs), runtime);
      ASSERT_EQ(outputs.size(), 1u);
      EXPECT_FLOAT_EQ(outputs[0].AsFloat()[0], expected);
    }
  };
  std::thread abs_worker([&] { run(abs_session, abs_runtime, 0.0f); });
  std::thread exp_worker([&] { run(exp_session, exp_runtime, 1.0f); });
  std::thread disabled_worker([&] { run(disabled_session, disabled_runtime, 0.0f); });
  abs_worker.join();
  exp_worker.join();
  disabled_worker.join();

  EXPECT_EQ(abs_runtime.GetKernelUsage(),
            std::vector<std::string>(invocations, onnx_light_cpu::AbsKernel::kName));
  EXPECT_EQ(exp_runtime.GetKernelUsage(),
            std::vector<std::string>(invocations, onnx_light_cpu::ExpKernel::kName));
  EXPECT_TRUE(disabled_runtime.GetKernelUsage().empty());
  abs_runtime.ClearKernelUsage();
  abs_runtime.set_kernel_usage_enabled(false);
  EXPECT_TRUE(exp_runtime.kernel_usage_enabled());
  EXPECT_EQ(exp_runtime.GetKernelUsage().size(), invocations);
}

TEST(OnnxLightKernelUsage, NestedContextsAndCopiesShareTheirOwnersRecorder) {
  rt_ns::RuntimeContext runtime(MakeContext());
  rt_ns::RuntimeContext other(MakeContext());
  auto subgraph = runtime.MakeSubgraphContext("then_branch");
  auto function = subgraph.MakeFunctionContext();
  auto nested = function.MakeSubgraphContext("body");
  auto copy = runtime;
  rt_ns::RuntimeContext assigned(MakeContext());
  assigned = function;
  runtime.set_kernel_usage_enabled(true);
  other.set_kernel_usage_enabled(true);
  EXPECT_TRUE(subgraph.kernel_usage_enabled());
  EXPECT_TRUE(function.kernel_usage_enabled());
  EXPECT_TRUE(copy.kernel_usage_enabled());
  EXPECT_TRUE(assigned.kernel_usage_enabled());

  RunAbs(subgraph);
  RunAbs(function);
  RunAbs(nested);
  RunAbs(copy);
  RunAbs(assigned);
  const std::vector<std::string> expected(5, onnx_light_cpu::AbsKernel::kName);
  EXPECT_EQ(runtime.GetKernelUsage(), expected);
  EXPECT_EQ(subgraph.GetKernelUsage(), expected);
  EXPECT_EQ(function.GetKernelUsage(), expected);
  EXPECT_EQ(copy.GetKernelUsage(), expected);
  EXPECT_EQ(assigned.GetKernelUsage(), expected);
  EXPECT_TRUE(other.GetKernelUsage().empty());

  nested.set_kernel_usage_enabled(false);
  EXPECT_FALSE(runtime.kernel_usage_enabled());
  EXPECT_FALSE(copy.kernel_usage_enabled());
  RunAbs(runtime);
  EXPECT_EQ(runtime.GetKernelUsage(), expected);
  EXPECT_TRUE(other.kernel_usage_enabled());
  copy.ClearKernelUsage();
  EXPECT_TRUE(runtime.GetKernelUsage().empty());
  EXPECT_TRUE(nested.GetKernelUsage().empty());
  function.set_kernel_usage_enabled(true);
  RunAbs(nested);
  EXPECT_EQ(runtime.GetKernelUsage(), std::vector<std::string>({onnx_light_cpu::AbsKernel::kName}));
  EXPECT_TRUE(other.GetKernelUsage().empty());
}

TEST(OnnxLightKernelUsage, FunctionInsideSubgraphRecordsOnOwningRuntime) {
  ONNX_LIGHT_NAMESPACE::ModelProto model;
  auto *graph = model.add_graph();
  graph->add_input()->set_name("x");
  graph->add_output()->set_name("y");
  auto *call = graph->add_node();
  *call = MakeNode("LocalAbs");
  call->set_domain("test.kernel_usage");
  auto *function = model.add_functions();
  function->set_name("LocalAbs");
  function->set_domain("test.kernel_usage");
  function->add_input("x");
  function->add_output("y");
  function->add_opset_import()->set_version(18);
  *function->add_node() = MakeNode("Abs");

  rt_ns::RuntimeContext runtime(MakeContext());
  rt_ns::RuntimeContext other(MakeContext());
  onnx_light_cpu::RegisterKernelForSession(runtime, "", "Abs");
  rt_ns::RegisterModelFunctions(model, runtime);
  rt_ns::SubgraphSession session(runtime, *graph);
  auto run = [&] {
    const auto outputs =
        session.Run({{"x", rt_ns::Tensor::FromFloat("x", {2}, {-2.0f, 3.0f})}}, runtime, "body");
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_FLOAT_EQ(outputs[0].AsFloat()[0], 2.0f);
    EXPECT_FLOAT_EQ(outputs[0].AsFloat()[1], 3.0f);
  };
  run();
  EXPECT_TRUE(runtime.GetKernelUsage().empty());
  runtime.set_kernel_usage_enabled(true);
  other.set_kernel_usage_enabled(true);
  run();
  const auto snapshot = runtime.GetKernelUsage();
  EXPECT_EQ(snapshot, std::vector<std::string>({onnx_light_cpu::AbsKernel::kName}));
  EXPECT_TRUE(other.GetKernelUsage().empty());
  runtime.set_kernel_usage_enabled(false);
  run();
  EXPECT_EQ(runtime.GetKernelUsage(), snapshot);
  runtime.ClearKernelUsage();
  runtime.set_kernel_usage_enabled(true);
  run();
  EXPECT_EQ(runtime.GetKernelUsage(), snapshot);
}

} // namespace
