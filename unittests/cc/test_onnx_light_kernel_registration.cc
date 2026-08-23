// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/kernel_registration.h"

#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym_ns = ONNX_LIGHT_NAMESPACE::core::symbolic;

using onnx_light_cpu::KernelRegistration;

std::vector<rt_ns::DataType> UniqueLeftTypes(const onnx_light_cpu::BinaryManifestEntry &entry) {
  std::vector<rt_ns::DataType> types;
  for (const auto &signature : entry.signatures) {
    const auto type = static_cast<rt_ns::DataType>(signature.left);
    if (std::find(types.begin(), types.end(), type) == types.end()) {
      types.push_back(type);
    }
  }
  return types;
}

// ``CollectRegisteredKernels`` must report one record for every ``op_type``
// that ``RegisterAllKernels`` installs into onnx-light's shared dispatch
// table (``kernel_usage.cc``'s ``RegisteredKernelNames`` is derived from this
// same inventory).
TEST(KernelRegistration, CollectionReportsEveryActualRegistration) {
  const std::vector<KernelRegistration> records = onnx_light_cpu::CollectRegisteredKernels();

  const std::vector<KernelRegistration> expected = {
      {"ai.onnx",
       "Abs",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::Abs",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::INT32,
        rt_ns::DataType::INT64, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16,
        rt_ns::DataType::INT8, rt_ns::DataType::INT16},
       std::nullopt,
       std::nullopt},
      {"ai.onnx",
       "Exp",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::Exp",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       std::nullopt,
       std::nullopt},
      {"ai.onnx",
       "Log",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::Log",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       std::nullopt,
       std::nullopt},
      {"ai.onnx",
       "Gemm",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::Gemm",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       std::nullopt,
       std::nullopt},
      {"ai.onnx",
       "MatMul",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::MatMul",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       std::nullopt,
       std::nullopt},
      {"ai.onnx",
       "MatMulInteger",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::MatMulInteger",
       {rt_ns::DataType::INT8, rt_ns::DataType::UINT8},
       std::nullopt,
       std::nullopt},
      {"ai.onnx",
       "QLinearMatMul",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::QLinearMatMul",
       {rt_ns::DataType::INT8, rt_ns::DataType::UINT8},
       std::nullopt,
       std::nullopt},
      {"ai.onnx",
       "Not",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::Not",
       {rt_ns::DataType::BOOL},
       std::nullopt,
       std::nullopt},
  };
  std::vector<KernelRegistration> all_expected = expected;
  for (const auto &entry : onnx_light_cpu::GetBinaryManifest()) {
    all_expected.push_back({"ai.onnx", std::string(entry.op_type), sym_ns::Device::kCPU,
                            std::string("onnx_light_cpu::") + std::string(entry.op_type),
                            UniqueLeftTypes(entry), entry.since_version, std::nullopt});
  }
  ASSERT_EQ(records.size(), all_expected.size());
  for (const KernelRegistration &expected_record : all_expected) {
    const auto actual = std::find_if(records.begin(), records.end(),
                                     [&expected_record](const KernelRegistration &record) {
                                       return record.op_type == expected_record.op_type;
                                     });
    ASSERT_NE(actual, records.end()) << expected_record.op_type;
    EXPECT_EQ(actual->domain, expected_record.domain);
    EXPECT_EQ(actual->op_type, expected_record.op_type);
    EXPECT_EQ(actual->device, expected_record.device);
    EXPECT_EQ(actual->kernel_name, expected_record.kernel_name);
    EXPECT_EQ(actual->types, expected_record.types);
    EXPECT_EQ(actual->since_version, expected_record.since_version);
    EXPECT_EQ(actual->until_version, expected_record.until_version);
  }
}

// Collection returns records already sorted by
// ``(domain, op_type, device, kernel_name)``, so callers never need to sort
// (or otherwise depend on registration order) themselves.
TEST(KernelRegistration, CollectionIsDeterministicallyOrdered) {
  const std::vector<KernelRegistration> first = onnx_light_cpu::CollectRegisteredKernels();
  const std::vector<KernelRegistration> second = onnx_light_cpu::CollectRegisteredKernels();

  ASSERT_EQ(first.size(), second.size());
  for (std::size_t i = 0; i < first.size(); ++i) {
    EXPECT_EQ(first[i].domain, second[i].domain);
    EXPECT_EQ(first[i].op_type, second[i].op_type);
    EXPECT_EQ(first[i].device, second[i].device);
    EXPECT_EQ(first[i].kernel_name, second[i].kernel_name);
  }

  std::vector<std::tuple<std::string, std::string, sym_ns::Device, std::string>> sort_keys;
  sort_keys.reserve(first.size());
  for (const KernelRegistration &record : first) {
    sort_keys.emplace_back(record.domain, record.op_type, record.device, record.kernel_name);
  }
  EXPECT_TRUE(std::is_sorted(sort_keys.begin(), sort_keys.end()));
}

// Collecting the inventory must never install, replace, or execute a kernel:
// it must not mutate onnx-light's shared ``KernelDispatchTable`` at all.
TEST(KernelRegistration, CollectionDoesNotMutateDispatchTable) {
  rt_ns::RegisterKernelFn("ai.onnx", "Abs", sym_ns::Device::kCPU,
                          [](const ONNX_LIGHT_NAMESPACE::NodeProto &, rt_ns::RuntimeContext &)
                              -> std::unique_ptr<rt_ns::KernelBase> { return nullptr; });
  const std::size_t before = rt_ns::KernelDispatchTable().size();

  const std::vector<KernelRegistration> records = onnx_light_cpu::CollectRegisteredKernels();
  EXPECT_FALSE(records.empty());

  EXPECT_EQ(rt_ns::KernelDispatchTable().size(), before);
  const auto factory = rt_ns::KernelDispatchTable().find("ai.onnx:Abs");
  ASSERT_NE(factory, rt_ns::KernelDispatchTable().end());
  ONNX_LIGHT_NAMESPACE::NodeProto node;
  rt_ns::RuntimeContext runtime(rt_ns::KernelContext(rt_ns::DefaultOpset(18)));
  EXPECT_EQ(factory->second(node, runtime), nullptr);
}

// A repeated ``(domain, op_type, device)`` registration within a single pass
// must fail explicitly rather than silently overwrite the earlier entry or
// the collected inventory. Uses ``KernelRegistrationScope`` directly (as
// ``RegisterAllKernels``/``CollectRegisteredKernels`` do internally) to bound
// both calls to the same pass; two independent top-level ``RegisterKernel``
// calls made outside any pass are, by contrast, each checked in isolation.
TEST(KernelRegistration, RejectsDuplicateRegistrationWithinOnePass) {
  std::vector<KernelRegistration> inventory;
  onnx_light_cpu::KernelRegistrationScope scope(&inventory);

  KernelRegistration first;
  first.domain = "";
  first.op_type = "Abs";
  first.device = sym_ns::Device::kCPU;
  first.kernel_name = "onnx_light_cpu::Abs";
  onnx_light_cpu::RegisterKernel(
      first,
      [](const ONNX_LIGHT_NAMESPACE::NodeProto &,
         rt_ns::RuntimeContext &) -> std::unique_ptr<rt_ns::KernelBase> { return nullptr; });
  ASSERT_EQ(inventory.size(), 1u);

  KernelRegistration duplicate = first;
  EXPECT_THROW(onnx_light_cpu::RegisterKernel(
                   duplicate,
                   [](const ONNX_LIGHT_NAMESPACE::NodeProto &, rt_ns::RuntimeContext &)
                       -> std::unique_ptr<rt_ns::KernelBase> { return nullptr; }),
               std::invalid_argument);
  // The failed duplicate must not have been appended to the inventory.
  EXPECT_EQ(inventory.size(), 1u);
}

// Optional opset bounds (``since_version``/``until_version``) must round-trip
// exactly through ``RegisterKernel`` into the collected inventory: a bounded
// record must keep its bounds, and an unbounded record collected in the same
// pass must keep reporting ``std::nullopt`` for both, so the two never bleed
// into each other.
TEST(KernelRegistration, OpsetBoundsRoundTripThroughCollection) {
  std::vector<KernelRegistration> inventory;
  onnx_light_cpu::KernelRegistrationScope scope(&inventory);

  auto noop_factory = [](const ONNX_LIGHT_NAMESPACE::NodeProto &,
                         rt_ns::RuntimeContext &) -> std::unique_ptr<rt_ns::KernelBase> {
    return nullptr;
  };

  KernelRegistration bounded;
  bounded.domain = "";
  bounded.op_type = "BoundedOp";
  bounded.device = sym_ns::Device::kCPU;
  bounded.kernel_name = "onnx_light_cpu::BoundedOp";
  bounded.types = {rt_ns::DataType::FLOAT};
  bounded.since_version = 9;
  bounded.until_version = 13;
  onnx_light_cpu::RegisterKernel(bounded, noop_factory);

  KernelRegistration unbounded;
  unbounded.domain = "";
  unbounded.op_type = "UnboundedOp";
  unbounded.device = sym_ns::Device::kCPU;
  unbounded.kernel_name = "onnx_light_cpu::UnboundedOp";
  unbounded.types = {rt_ns::DataType::FLOAT};
  onnx_light_cpu::RegisterKernel(unbounded, noop_factory);

  ASSERT_EQ(inventory.size(), 2u);
  const auto found_bounded =
      std::find_if(inventory.begin(), inventory.end(),
                   [](const KernelRegistration &record) { return record.op_type == "BoundedOp"; });
  ASSERT_NE(found_bounded, inventory.end());
  EXPECT_EQ(found_bounded->since_version, 9);
  EXPECT_EQ(found_bounded->until_version, 13);

  const auto found_unbounded =
      std::find_if(inventory.begin(), inventory.end(), [](const KernelRegistration &record) {
        return record.op_type == "UnboundedOp";
      });
  ASSERT_NE(found_unbounded, inventory.end());
  EXPECT_EQ(found_unbounded->since_version, std::nullopt);
  EXPECT_EQ(found_unbounded->until_version, std::nullopt);
}

// Normal-mode registration (``RegisterAllKernels``) must keep installing the
// same kernels into the shared dispatch table exactly as before, so runtime
// dispatch is unaffected by routing registrations through the new helper.
TEST(KernelRegistration, NormalModeStillInstallsKernelsIntoDispatchTable) {
  onnx_light_cpu::RegisterAllKernels();
  const auto &table = rt_ns::KernelDispatchTable();
  for (const std::string &key :
       {"ai.onnx:Abs", "ai.onnx:Exp", "ai.onnx:Log", "ai.onnx:Gemm", "ai.onnx:MatMul",
        "ai.onnx:MatMulInteger", "ai.onnx:QLinearMatMul", "ai.onnx:Not"}) {
    EXPECT_NE(table.find(key), table.end()) << key;
  }
}

// ``RegisterAllKernels`` itself must not trip its own duplicate check: it is
// expected to be callable more than once (e.g. re-registration during tests
// or repeated Python imports).
TEST(KernelRegistration, RegisterAllKernelsCanBeCalledRepeatedly) {
  EXPECT_NO_THROW(onnx_light_cpu::RegisterAllKernels());
  EXPECT_NO_THROW(onnx_light_cpu::RegisterAllKernels());
}

} // namespace
