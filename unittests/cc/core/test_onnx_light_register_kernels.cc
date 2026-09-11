// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_light_cpu/kernels/com_microsoft/naive_bias_gelu_kernel.h"
#include "onnx_light_cpu/kernels/kernel_registration.h"
#include "onnx_light_cpu/kernels/kernel_usage.h"

#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

struct PreparationCounts {
  std::atomic<int> plans{0};
  std::atomic<int> live{0};
  std::atomic<int> runs{0};
};

// This test kernel prepares immutable input metadata at construction, just as
// a factory preparing an execution plan would. Run must never rebuild it.
class PreparedCopyKernel : public rt_ns::KernelBase {
public:
  PreparedCopyKernel(const ONNX_LIGHT_NAMESPACE::NodeProto &node, rt_ns::RuntimeContext &rt,
                     std::shared_ptr<PreparationCounts> counts)
      : KernelBase(rt.kernel_ctx()), counts_(std::move(counts)),
        shape_(rt.Get(node.input(0)).shape), dtype_(rt.Get(node.input(0)).data_type) {
    set_node(node);
    ++counts_->plans;
    ++counts_->live;
  }

  ~PreparedCopyKernel() override { --counts_->live; }

  void Run(rt_ns::RuntimeContext &rt) override {
    EXPECT_EQ(active_.fetch_add(1), 0);
    std::this_thread::yield();
    const auto &input = rt.Get(node_->input(0));
    EXPECT_EQ(input.shape, shape_);
    EXPECT_EQ(input.data_type, dtype_);
    rt.Put(node_->output(0), input.ToOwned());
    ++counts_->runs;
    EXPECT_EQ(active_.fetch_sub(1), 1);
  }

private:
  std::shared_ptr<PreparationCounts> counts_;
  const rt_ns::Shape shape_;
  const int dtype_;
  std::atomic<int> active_{0};
};

rt_ns::CustomKernelFn PreparedCopyCallback(const std::shared_ptr<PreparationCounts> &counts) {
  return onnx_light_cpu::detail::AsSessionKernel(
      [counts](const ONNX_LIGHT_NAMESPACE::NodeProto &node, rt_ns::RuntimeContext &rt) {
        return std::make_unique<PreparedCopyKernel>(node, rt, counts);
      });
}

TEST(OnnxLightRegisterKernels, SessionPlansReuseOnlyMatchingMetadata) {
  auto counts = std::make_shared<PreparationCounts>();
  auto callback = PreparedCopyCallback(counts);
  rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Copy");
  node.add_input("x");
  node.add_input("optional");
  node.add_output("y");

  for (int i = 0; i < 5; ++i) {
    runtime.Put("x", rt_ns::Tensor::FromFloat("x", {2}, {float(i), 2.0f}));
    callback(node, runtime);
    EXPECT_EQ(runtime.Get("y").AsFloat()[0], float(i));
  }
  EXPECT_EQ(counts->plans, 1);

  runtime.Put("x", rt_ns::Tensor::FromFloat("x", {1, 2}, {3.0f, 4.0f}));
  callback(node, runtime);
  EXPECT_EQ(counts->plans, 2);
  runtime.Put("x", rt_ns::Tensor::FromFloat("x", {2}, {5.0f, 6.0f}));
  callback(node, runtime);
  EXPECT_EQ(counts->plans, 2);
  runtime.Put("x", rt_ns::Tensor::FromDouble("x", {1, 2}, {3.0, 4.0}));
  callback(node, runtime);
  EXPECT_EQ(counts->plans, 3);
  runtime.Set("optional", rt_ns::Tensor::FromFloat("optional", {}, {1.0f}));
  callback(node, runtime);
  EXPECT_EQ(counts->plans, 4);
  runtime.kernel_ctx().opset.version = 19;
  callback(node, runtime);
  EXPECT_EQ(counts->plans, 5);

  // A second node must not inherit the first node's output names or attributes,
  // even when the caller reuses the same NodeProto storage.
  node.clear_output();
  node.add_output("z");
  callback(node, runtime);
  EXPECT_TRUE(runtime.Has("z"));
  EXPECT_EQ(counts->plans, 6);
  auto *attribute = node.add_attribute();
  attribute->set_name("version");
  attribute->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::INT);
  attribute->set_i(1);
  callback(node, runtime);
  EXPECT_EQ(counts->plans, 7);
  callback(node, runtime);
  EXPECT_EQ(counts->plans, 7);
}

TEST(OnnxLightRegisterKernels, SessionPlanCopiesOwnNodesAndReleaseWithCallbacks) {
  auto counts = std::make_shared<PreparationCounts>();
  rt_ns::CustomKernelFn resolved;
  ONNX_LIGHT_NAMESPACE::NodeProto saved_node;
  {
    rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
    runtime.RegisterCustomKernel("", "Copy", PreparedCopyCallback(counts));
    runtime.Set("x", rt_ns::Tensor::FromFloat("x", {1}, {4.0f}));
    ONNX_LIGHT_NAMESPACE::NodeProto node;
    node.set_op_type("Copy");
    node.add_input("x");
    node.add_output("y");
    saved_node.CopyFrom(node);
    resolved = runtime.custom_kernels().at("ai.onnx:Copy");
    resolved(node, runtime);
    EXPECT_EQ(counts->plans, 1);
    EXPECT_EQ(counts->live, 1);

    runtime.RegisterCustomKernel("", "Copy", PreparedCopyCallback(counts));
    runtime.custom_kernels().at("ai.onnx:Copy")(node, runtime);
    EXPECT_EQ(counts->plans, 2);
    EXPECT_EQ(counts->live, 2);
  }
  EXPECT_EQ(counts->live, 1);
  rt_ns::RuntimeContext next(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
  next.Set("x", rt_ns::Tensor::FromFloat("x", {1}, {7.0f}));
  resolved(saved_node, next);
  EXPECT_EQ(next.Get("y").AsFloat()[0], 7.0f);
  EXPECT_EQ(counts->plans, 2);
  resolved = {};
  EXPECT_EQ(counts->live, 0);
}

TEST(OnnxLightRegisterKernels, ConcurrentSessionCallbacksPrepareEachSignatureOnce) {
  auto counts = std::make_shared<PreparationCounts>();
  auto callback = PreparedCopyCallback(counts);
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Copy");
  node.add_input("x");
  node.add_output("y");
  std::barrier start(4);
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&, callback, i] {
      rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
      start.arrive_and_wait();
      for (int run = 0; run < 32; ++run) {
        const int64_t size = 1 + run % 2;
        runtime.Put("x", rt_ns::Tensor::FromFloat("x", {size}, std::vector<float>(size, float(i))));
        callback(node, runtime);
        EXPECT_EQ(runtime.Get("y").AsFloat()[0], float(i));
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  EXPECT_EQ(counts->plans, 2);
  EXPECT_EQ(counts->runs, 128);
}

TEST(OnnxLightRegisterKernels, RuntimeSessionRetainsNativeKernelsAcrossRuns) {
  ONNX_LIGHT_NAMESPACE::ModelProto model;
  auto *graph = model.mutable_graph();
  graph->add_input()->set_name("a");
  graph->add_input()->set_name("b");
  graph->add_output()->set_name("y");
  auto *node = graph->add_node();
  node->set_op_type("Gemm");
  node->add_input("a");
  node->add_input("b");
  node->add_output("y");
  for (bool register_all : {false, true}) {
    rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
    if (register_all) {
      ASSERT_GT(onnx_light_cpu::RegisterAllKernelsForSession(runtime), 0u);
    } else {
      ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(runtime, "", "Gemm"));
    }
    rt_ns::RuntimeSession session(model);
    runtime.Set("a", rt_ns::Tensor::FromFloat("a", {1, 2}, {2.0f, 3.0f}));
    runtime.Set("b", rt_ns::Tensor::FromFloat("b", {2, 1}, {4.0f, 5.0f}));
    session.Run(runtime);
    EXPECT_FLOAT_EQ(runtime.Get("y").AsFloat()[0], 23.0f);
    const auto constructed = rt_ns::KernelBase::ConstructionCountForTesting();
    for (int run = 0; run < 3; ++run) {
      runtime.Put("a", rt_ns::Tensor::FromFloat("a", {1, 2}, {3.0f, 2.0f}));
      session.Run(runtime);
      EXPECT_FLOAT_EQ(runtime.Get("y").AsFloat()[0], 22.0f);
    }
    EXPECT_EQ(rt_ns::KernelBase::ConstructionCountForTesting(), constructed);
  }
}

TEST(OnnxLightRegisterKernels, FailedSessionPreparationCanBeRetried) {
  auto counts = std::make_shared<PreparationCounts>();
  int attempts = 0;
  auto callback = onnx_light_cpu::detail::AsSessionKernel(
      [&](const ONNX_LIGHT_NAMESPACE::NodeProto &node,
          rt_ns::RuntimeContext &rt) -> std::unique_ptr<rt_ns::KernelBase> {
        if (++attempts == 1) {
          return nullptr;
        }
        return std::make_unique<PreparedCopyKernel>(node, rt, counts);
      });
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Copy");
  node.add_input("x");
  node.add_output("y");
  rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
  runtime.Set("x", rt_ns::Tensor::FromFloat("x", {1}, {4.0f}));
  EXPECT_THROW(callback(node, runtime), std::runtime_error);
  callback(node, runtime);
  callback(node, runtime);
  EXPECT_EQ(attempts, 2);
  EXPECT_EQ(counts->plans, 1);
  EXPECT_EQ(runtime.Get("y").AsFloat()[0], 4.0f);
}

// ``RegisterAllKernels`` installs every onnx-light-cpu kernel class into
// onnx-light's shared ``KernelDispatchTable``. The call must succeed and be
// safe to invoke more than once (re-registration overrides the existing
// entries with the same factories).
TEST(OnnxLightRegisterKernels, RegisterAllKernels) {
  EXPECT_NO_THROW(onnx_light_cpu::RegisterAllKernels());
  EXPECT_NO_THROW(onnx_light_cpu::RegisterAllKernels());
}

TEST(OnnxLightRegisterKernels, NaivePolicyInstallsNaiveMicrosoftFactories) {
  namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
  onnx_light_cpu::RegisterAllKernels(onnx_light_cpu::MicrosoftKernelImplementation::NAIVE);
  const auto &table = rt_ns::KernelDispatchTable();
  const auto factory = table.find("com.microsoft:BiasGelu");
  ASSERT_NE(factory, table.end());

  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_domain("com.microsoft");
  node.set_op_type("BiasGelu");
  node.add_input("a");
  node.add_input("b");
  node.add_output("c");
  rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::OpsetId("com.microsoft", 1)));
  runtime.Set("a", rt_ns::Tensor::FromFloat("a", {1, 2}, {-1.0f, 2.0f}));
  runtime.Set("b", rt_ns::Tensor::FromFloat("b", {2}, {0.25f, -0.5f}));

  std::unique_ptr<rt_ns::KernelBase> kernel = factory->second(node, runtime);
  EXPECT_NE(dynamic_cast<onnx_light_cpu::NaiveBiasGeluKernel *>(kernel.get()), nullptr);

  onnx_light_cpu::RegisterAllKernels();
}

TEST(OnnxLightRegisterKernels, RegisteredFactoriesConstructWithoutSessionExecutor) {
  namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
  onnx_light_cpu::RegisterAllKernels();
  const auto &table = rt_ns::KernelDispatchTable();
  const auto factory = table.find("ai.onnx:Abs");
  ASSERT_NE(factory, table.end());
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Abs");
  node.add_input("x");
  node.add_output("y");
  rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
  runtime.Set("x", rt_ns::Tensor::FromFloat("x", {4}, {-1.0f, 2.0f, -3.5f, 4.0f}));

  std::unique_ptr<rt_ns::KernelBase> kernel;
  ASSERT_NO_THROW(kernel = factory->second(node, runtime));
  ASSERT_NE(kernel, nullptr);
  EXPECT_NO_THROW(kernel->Run(runtime));
  const float *y = runtime.Get("y").AsFloat();
  EXPECT_FLOAT_EQ(y[0], 1.0f);
  EXPECT_FLOAT_EQ(y[1], 2.0f);
  EXPECT_FLOAT_EQ(y[2], 3.5f);
  EXPECT_FLOAT_EQ(y[3], 4.0f);
}

TEST(OnnxLightRegisterKernels, VariadicFactoryUsesOneCommonBroadcastPlan) {
  namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
  onnx_light_cpu::RegisterAllKernels();
  const auto factory = rt_ns::KernelDispatchTable().find("ai.onnx:Sum");
  ASSERT_NE(factory, rt_ns::KernelDispatchTable().end());
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Sum");
  node.add_input("a");
  node.add_input("b");
  node.add_input("c");
  node.add_output("y");
  rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
  runtime.Set("a", rt_ns::Tensor::FromFloat("a", {2, 1}, {1.0f, 2.0f}));
  runtime.Set("b", rt_ns::Tensor::FromFloat("b", {1, 3}, {10.0f, 20.0f, 30.0f}));
  runtime.Set("c", rt_ns::Tensor::FromFloat("c", {}, {100.0f}));

  std::unique_ptr<rt_ns::KernelBase> kernel = factory->second(node, runtime);
  ASSERT_NE(kernel, nullptr);
  ASSERT_NO_THROW(kernel->Run(runtime));
  const rt_ns::Tensor &output = runtime.Get("y");
  EXPECT_EQ(output.shape, (rt_ns::Shape{2, 3}));
  const float *values = output.AsFloat();
  EXPECT_EQ(std::vector<float>(values, values + 6),
            (std::vector<float>{111.0f, 121.0f, 131.0f, 112.0f, 122.0f, 132.0f}));
}

TEST(OnnxLightRegisterKernels, RegisterSelectedKernelGlobalRunsNativeKernel) {
  namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
  namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
  rt_ns::RegisterKernelFn("", "Abs", sym_ns::Device::kCPU,
                          [](const ONNX_LIGHT_NAMESPACE::NodeProto &, rt_ns::RuntimeContext &)
                              -> std::unique_ptr<rt_ns::KernelBase> { return nullptr; });

  EXPECT_FALSE(onnx_light_cpu::RegisterKernelGlobal("", "Abs", false));
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Abs");
  node.add_input("x");
  node.add_output("y");
  rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
  runtime.Set("x", rt_ns::Tensor::FromFloat("x", {2}, {-2.0f, 3.0f}));
  EXPECT_EQ(rt_ns::KernelDispatchTable().at("ai.onnx:Abs")(node, runtime), nullptr);

  ASSERT_TRUE(onnx_light_cpu::RegisterKernelGlobal("ai.onnx", "Abs"));
  std::unique_ptr<rt_ns::KernelBase> kernel =
      rt_ns::KernelDispatchTable().at("ai.onnx:Abs")(node, runtime);
  ASSERT_NE(kernel, nullptr);
  onnx_light_cpu::ClearUsedKernelNames();
  kernel->Run(runtime);
  EXPECT_EQ(onnx_light_cpu::UsedKernelNames(), (std::vector<std::string>{"onnx_light_cpu::Abs"}));
  EXPECT_FLOAT_EQ(runtime.Get("y").AsFloat()[0], 2.0f);
  EXPECT_FLOAT_EQ(runtime.Get("y").AsFloat()[1], 3.0f);
}

TEST(OnnxLightRegisterKernels, SessionRegistrationIsIsolatedAndReplaceable) {
  namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
  namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;
  rt_ns::RegisterKernelFn("", "Abs", sym_ns::Device::kCPU,
                          [](const ONNX_LIGHT_NAMESPACE::NodeProto &, rt_ns::RuntimeContext &)
                              -> std::unique_ptr<rt_ns::KernelBase> { return nullptr; });
  const std::size_t global_size = rt_ns::KernelDispatchTable().size();

  rt_ns::RuntimeContext selected(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
  rt_ns::RuntimeContext other(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
  bool sentinel_ran = false;
  selected.RegisterCustomKernel("", "Abs",
                                [&sentinel_ran](const ONNX_LIGHT_NAMESPACE::NodeProto &,
                                                rt_ns::RuntimeContext &) { sentinel_ran = true; });
  EXPECT_FALSE(onnx_light_cpu::RegisterKernelForSession(selected, "", "Abs", false));

  ONNX_LIGHT_NAMESPACE::NodeProto node;
  node.set_op_type("Abs");
  node.add_input("x");
  node.add_output("y");
  selected.custom_kernels().at("ai.onnx:Abs")(node, selected);
  EXPECT_TRUE(sentinel_ran);

  ASSERT_TRUE(onnx_light_cpu::RegisterKernelForSession(selected, "", "Abs"));
  EXPECT_EQ(other.custom_kernels().find("ai.onnx:Abs"), other.custom_kernels().end());
  EXPECT_EQ(rt_ns::KernelDispatchTable().size(), global_size);
  EXPECT_EQ(rt_ns::KernelDispatchTable().at("ai.onnx:Abs")(node, other), nullptr);

  selected.Set("x", rt_ns::Tensor::FromFloat("x", {2}, {-4.0f, 5.0f}));
  onnx_light_cpu::ClearUsedKernelNames();
  rt_ns::RunNode(node, selected);
  EXPECT_EQ(onnx_light_cpu::UsedKernelNames(), (std::vector<std::string>{"onnx_light_cpu::Abs"}));
  EXPECT_FLOAT_EQ(selected.Get("y").AsFloat()[0], 4.0f);
  EXPECT_FLOAT_EQ(selected.Get("y").AsFloat()[1], 5.0f);

  EXPECT_TRUE(onnx_light_cpu::RegisterKernelGlobal("", "Abs"));
}

TEST(OnnxLightRegisterKernels, RegisterAllForSessionIsIdempotentWithoutReplacement) {
  namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
  rt_ns::RuntimeContext selected(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
  std::set<std::pair<std::string, std::string>> expected;
  for (const auto &entry : onnx_light_cpu::CollectRegisteredKernels()) {
    expected.emplace(entry.domain, entry.op_type);
  }

  EXPECT_EQ(onnx_light_cpu::RegisterAllKernelsForSession(selected, false), expected.size());
  EXPECT_EQ(onnx_light_cpu::RegisterAllKernelsForSession(selected, false), 0u);
  EXPECT_EQ(selected.custom_kernels().size(), expected.size());
}

TEST(OnnxLightRegisterKernels, UnknownKernelFailsClearly) {
  namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
  rt_ns::RuntimeContext selected(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
  EXPECT_THROW(onnx_light_cpu::RegisterKernelGlobal("unknown.domain", "Abs"),
               std::invalid_argument);
  EXPECT_THROW(onnx_light_cpu::RegisterKernelGlobal("", "UnknownOperator"), std::invalid_argument);
  EXPECT_THROW(onnx_light_cpu::RegisterKernelForSession(selected, "", "UnknownOperator"),
               std::invalid_argument);
}

} // namespace
