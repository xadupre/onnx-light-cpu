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

std::vector<KernelRegistration>
BinaryRegistrations(const onnx_light_cpu::BinaryManifestEntry &entry) {
  std::vector<std::int64_t> versions = {entry.minimum_version};
  for (const auto &signature : entry.signatures) {
    if (signature.minimum_version > entry.minimum_version &&
        std::find(versions.begin(), versions.end(), signature.minimum_version) == versions.end()) {
      versions.push_back(signature.minimum_version);
    }
  }
  std::sort(versions.begin(), versions.end());
  std::vector<KernelRegistration> records;
  for (std::size_t index = 0; index < versions.size(); ++index) {
    std::vector<rt_ns::DataType> types;
    for (const auto &signature : entry.signatures) {
      const std::int64_t minimum =
          signature.minimum_version == 0 ? entry.minimum_version : signature.minimum_version;
      const auto type = static_cast<rt_ns::DataType>(signature.left);
      if (minimum <= versions[index] &&
          std::find(types.begin(), types.end(), type) == types.end()) {
        types.push_back(type);
      }
    }
    records.push_back({"ai.onnx", std::string(entry.op_type), sym_ns::Device::kCPU,
                       std::string("onnx_light_cpu::") + std::string(entry.op_type),
                       std::move(types), versions[index],
                       index + 1 < versions.size()
                           ? std::optional<std::int64_t>(versions[index + 1] - 1)
                           : std::nullopt});
  }
  return records;
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
       "Attention",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::Attention",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16},
       23,
       std::nullopt},
      {"ai.onnx",
       "BatchNormalization",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::BatchNormalization",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       15,
       std::nullopt},
      {"com.microsoft",
       "BiasGelu",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::BiasGelu",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       1,
       std::nullopt},
      {"com.microsoft",
       "CDist",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::CDist",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE},
       1,
       std::nullopt},
      {"com.microsoft",
       "GroupQueryAttention",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::GroupQueryAttention",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::FLOAT16, rt_ns::DataType::BFLOAT16},
       1,
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
       "GroupNormalization",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::GroupNormalization",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       18,
       std::nullopt},
      {"ai.onnx",
       "InstanceNormalization",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::InstanceNormalization",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       22,
       std::nullopt},
      {"ai.onnx",
       "LayerNormalization",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::LayerNormalization",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       17,
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
       "LpNormalization",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::LpNormalization",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       22,
       std::nullopt},
      {"ai.onnx",
       "MatMulInteger",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::MatMulInteger",
       {rt_ns::DataType::INT8, rt_ns::DataType::UINT8},
       std::nullopt,
       std::nullopt},
      {"ai.onnx",
       "Max",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::Max",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::INT8, rt_ns::DataType::INT16, rt_ns::DataType::INT32,
        rt_ns::DataType::INT64, rt_ns::DataType::UINT8, rt_ns::DataType::UINT16,
        rt_ns::DataType::UINT32, rt_ns::DataType::UINT64},
       13,
       std::nullopt},
      {"ai.onnx",
       "Mean",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::Mean",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE},
       13,
       std::nullopt},
      {"ai.onnx",
       "MeanVarianceNormalization",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::MeanVarianceNormalization",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       13,
       std::nullopt},
      {"ai.onnx",
       "Min",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::Min",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::INT8, rt_ns::DataType::INT16, rt_ns::DataType::INT32,
        rt_ns::DataType::INT64, rt_ns::DataType::UINT8, rt_ns::DataType::UINT16,
        rt_ns::DataType::UINT32, rt_ns::DataType::UINT64},
       13,
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
      {"ai.onnx",
       "RMSNormalization",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::RMSNormalization",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       23,
       std::nullopt},
      {"ai.onnx",
       "Sigmoid",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::Sigmoid",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       std::nullopt,
       std::nullopt},
      {"ai.onnx",
       "Softmax",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::Softmax",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       std::nullopt,
       std::nullopt},
      {"ai.onnx",
       "SwiGLU",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::SwiGLU",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16,
        rt_ns::DataType::BFLOAT16},
       28,
       std::nullopt},
      {"ai.onnx",
       "Sum",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::Sum",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE},
       13,
       std::nullopt},
      {"ai.onnx.ml",
       "TreeEnsemble",
       sym_ns::Device::kCPU,
       "onnx_light_cpu::TreeEnsemble",
       {rt_ns::DataType::FLOAT, rt_ns::DataType::DOUBLE, rt_ns::DataType::FLOAT16},
       5,
       std::nullopt},
  };
  std::vector<KernelRegistration> all_expected = expected;
  for (const auto &entry : onnx_light_cpu::GetBinaryManifest()) {
    std::vector<KernelRegistration> binary_records = BinaryRegistrations(entry);
    all_expected.insert(all_expected.end(), std::make_move_iterator(binary_records.begin()),
                        std::make_move_iterator(binary_records.end()));
  }
  ASSERT_EQ(records.size(), all_expected.size());
  for (const KernelRegistration &expected_record : all_expected) {
    const auto actual = std::find_if(
        records.begin(), records.end(), [&expected_record](const KernelRegistration &record) {
          return record.op_type == expected_record.op_type &&
                 record.since_version == expected_record.since_version &&
                 record.until_version == expected_record.until_version;
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

TEST(KernelRegistration, MicrosoftImplementationPolicySelectsOneCompleteFamily) {
  const auto optimized = onnx_light_cpu::CollectRegisteredKernels(
      onnx_light_cpu::MicrosoftKernelImplementation::OPTIMIZED);
  const auto naive = onnx_light_cpu::CollectRegisteredKernels(
      onnx_light_cpu::MicrosoftKernelImplementation::NAIVE);

  auto microsoft_names = [](const std::vector<KernelRegistration> &records) {
    std::vector<std::pair<std::string, std::string>> names;
    for (const auto &record : records) {
      if (record.domain == "com.microsoft") {
        names.emplace_back(record.op_type, record.kernel_name);
      }
    }
    return names;
  };

  EXPECT_EQ(microsoft_names(optimized),
            (std::vector<std::pair<std::string, std::string>>{
                {"BiasGelu", "onnx_light_cpu::BiasGelu"},
                {"CDist", "onnx_light_cpu::CDist"},
                {"GroupQueryAttention", "onnx_light_cpu::GroupQueryAttention"}}));
  EXPECT_EQ(microsoft_names(naive),
            (std::vector<std::pair<std::string, std::string>>{
                {"BiasGelu", "onnx_light_cpu::NaiveBiasGelu"},
                {"CDist", "onnx_light_cpu::NaiveCDist"},
                {"GroupQueryAttention", "onnx_light_cpu::NaiveGroupQueryAttention"}}));
  EXPECT_EQ(microsoft_names(onnx_light_cpu::CollectRegisteredKernels()),
            microsoft_names(optimized));
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
        "ai.onnx:MatMulInteger", "ai.onnx:Max", "ai.onnx:Mean", "ai.onnx:Min",
        "ai.onnx:QLinearMatMul", "ai.onnx:Not", "ai.onnx:RMSNormalization", "ai.onnx:Sum",
        "ai.onnx:SwiGLU"}) {
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
