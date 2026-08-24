// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/backend_test/cases/elementwise/include_elementwise_cases.h"

#include "onnx_light_cpu/impl/math/binary/binary_broadcast_plan.h"
#include "onnx_light_cpu/impl/math/binary/binary_kernel_descriptor.h"
#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"
#include "onnx_light_cpu/kernels/elementwise/binary_kernel.h"

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_cpu::backend_test {
namespace {

namespace bt_ns = ONNX_LIGHT_NAMESPACE::core::backend_test;
namespace rt_ns = ONNX_LIGHT_NAMESPACE::core::runtime;

using bt_ns::Expect;
using bt_ns::IoData;
using bt_ns::TestCase;
using bt_ns::TestMode;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using rt_ns::DataType;
using rt_ns::KernelContext;
using rt_ns::OpsetId;
using rt_ns::Tensor;

struct ShapePair {
  std::string tag;
  rt_ns::Shape left;
  rt_ns::Shape right;
};

const std::vector<ShapePair> &BinaryShapePairs() {
  static const std::vector<ShapePair> kPairs = {
      {"contiguous", {2, 3}, {2, 3}},
      {"left_scalar", {}, {2, 3}},
      {"right_scalar", {2, 3}, {}},
      {"repeated_block", {2, 3, 4, 5}, {3, 1, 1}},
      {"inner_vector", {2, 3, 4, 5}, {1, 3, 1, 5}},
      {"outer_broadcast", {2, 1, 4, 5}, {1, 3, 4, 5}},
      {"general", {2, 1, 4, 1, 8, 1}, {1, 3, 1, 5, 1, 7}},
  };
  return kPairs;
}

const std::vector<ShapePair> &BinaryBenchmarkShapePairs() {
  static const std::vector<ShapePair> kPairs = {
      {"contiguous_1d", {1024}, {1024}},
      {"contiguous_2d", {128, 256}, {128, 256}},
      {"contiguous_4d", {3, 5, 17, 257}, {3, 5, 17, 257}},
      {"contiguous_3d", {4, 16, 1024}, {4, 16, 1024}},
      {"repeated_block_4d", {2, 8, 128, 64}, {8, 1, 1}},
      {"inner_vector_4d", {4, 16, 256, 64}, {1, 16, 1, 64}},
      {"general_5d", {4, 1, 16, 1, 64}, {1, 8, 1, 128, 1}},
      {"general_6d", {4, 1, 8, 1, 16, 1}, {1, 4, 1, 8, 1, 256}},
  };
  return kPairs;
}

bool IsNonCommutative(BinaryOperator op) {
  switch (op) {
  case BinaryOperator::kSub:
  case BinaryOperator::kDiv:
  case BinaryOperator::kPow:
  case BinaryOperator::kGreater:
  case BinaryOperator::kGreaterOrEqual:
  case BinaryOperator::kLess:
  case BinaryOperator::kLessOrEqual:
  case BinaryOperator::kBitShift:
    return true;
  default:
    return false;
  }
}

const char *DataTypeSuffix(BinaryDataType data_type) {
  switch (data_type) {
  case BinaryDataType::BOOL:
    return "bool";
  case BinaryDataType::FLOAT:
    return "float32";
  case BinaryDataType::DOUBLE:
    return "float64";
  case BinaryDataType::FLOAT16:
    return "float16";
  case BinaryDataType::BFLOAT16:
    return "bfloat16";
  case BinaryDataType::INT8:
    return "int8";
  case BinaryDataType::INT16:
    return "int16";
  case BinaryDataType::INT32:
    return "int32";
  case BinaryDataType::INT64:
    return "int64";
  case BinaryDataType::UINT8:
    return "uint8";
  case BinaryDataType::UINT16:
    return "uint16";
  case BinaryDataType::UINT32:
    return "uint32";
  case BinaryDataType::UINT64:
    return "uint64";
  default:
    throw std::invalid_argument("Unsupported binary benchmark type.");
  }
}

std::size_t ElementCount(const rt_ns::Shape &shape) {
  std::size_t value = 1;
  for (std::int64_t dim : shape) {
    value *= static_cast<std::size_t>(std::max<std::int64_t>(dim, 0));
  }
  return value;
}

std::size_t BroadcastElementCount(const rt_ns::Shape &left, const rt_ns::Shape &right) {
  const std::size_t rank = std::max(left.size(), right.size());
  std::size_t value = 1;
  for (std::size_t axis = 0; axis < rank; ++axis) {
    const std::int64_t left_dim = axis < rank - left.size() ? 1 : left[axis - (rank - left.size())];
    const std::int64_t right_dim =
        axis < rank - right.size() ? 1 : right[axis - (rank - right.size())];
    if (left_dim != right_dim && left_dim != 1 && right_dim != 1) {
      throw std::invalid_argument("Incompatible binary benchmark shapes.");
    }
    value *= static_cast<std::size_t>(std::max(left_dim, right_dim));
  }
  return value;
}

NodeProto MakeNode(std::string_view op_type, const BinaryKernelDescriptor::Attributes &attributes) {
  NodeProto node;
  node.set_op_type(std::string(op_type));
  node.add_input("a");
  node.add_input("b");
  node.add_output("c");
  if (op_type == "Mod") {
    auto *attr = node.add_attribute();
    attr->set_name("fmod");
    attr->set_i(attributes.mod_fmod);
    attr->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::INT);
  } else if (op_type == "BitShift") {
    auto *attr = node.add_attribute();
    attr->set_name("direction");
    attr->set_s(attributes.bitshift_direction ==
                        BinaryKernelDescriptor::Attributes::BitShiftDirection::kLeft
                    ? "LEFT"
                    : "RIGHT");
    attr->set_type(ONNX_LIGHT_NAMESPACE::AttributeProto::AttributeType::STRING);
  }
  return node;
}

std::vector<float> MakeFloatValues(std::size_t count, bool positive_only, bool nonzero_only,
                                   bool small_values) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    float value = static_cast<float>((static_cast<int>(i % 7) - 3));
    if (small_values) {
      value = static_cast<float>((i % 4) + 1);
    }
    if (positive_only) {
      value = std::fabs(value) + 1.0f;
    }
    if (nonzero_only && value == 0.0f) {
      value = 1.0f;
    }
    values[i] = value;
  }
  return values;
}

Tensor MakeTypedTensor(BinaryDataType type, const rt_ns::Shape &shape,
                       const std::vector<float> &values, bool positive_only, bool nonzero_only,
                       bool small_values) {
  const std::size_t count = ElementCount(shape);
  std::vector<float> generated = values;
  if (generated.size() != count) {
    generated = MakeFloatValues(count, positive_only, nonzero_only, small_values);
  }
  switch (type) {
  case BinaryDataType::BOOL: {
    std::vector<std::uint8_t> raw(count);
    for (std::size_t i = 0; i < count; ++i)
      raw[i] = static_cast<std::uint8_t>((i + 1) & 1U);
    return Tensor::FromBool("", shape, raw);
  }
  case BinaryDataType::FLOAT:
    return Tensor::FromFloat("", shape, generated);
  case BinaryDataType::DOUBLE:
    return Tensor::FromDouble("", shape, std::vector<double>(generated.begin(), generated.end()));
  case BinaryDataType::FLOAT16:
    return rt_ns::MakeFloat16Tensor("", shape, generated);
  case BinaryDataType::BFLOAT16:
    return rt_ns::MakeBfloat16Tensor("", shape, generated);
  case BinaryDataType::INT8: {
    std::vector<std::int8_t> raw(count);
    for (std::size_t i = 0; i < count; ++i)
      raw[i] = static_cast<std::int8_t>(generated[i]);
    return Tensor::FromInt8("", shape, raw);
  }
  case BinaryDataType::INT16: {
    std::vector<std::int16_t> raw(count);
    for (std::size_t i = 0; i < count; ++i)
      raw[i] = static_cast<std::int16_t>(generated[i]);
    return Tensor::FromInt16("", shape, raw);
  }
  case BinaryDataType::INT32: {
    std::vector<std::int32_t> raw(count);
    for (std::size_t i = 0; i < count; ++i)
      raw[i] = static_cast<std::int32_t>(generated[i]);
    return Tensor::FromInt32("", shape, raw);
  }
  case BinaryDataType::INT64: {
    std::vector<std::int64_t> raw(count);
    for (std::size_t i = 0; i < count; ++i)
      raw[i] = static_cast<std::int64_t>(generated[i]);
    return Tensor::FromInt64("", shape, raw);
  }
  case BinaryDataType::UINT8: {
    std::vector<std::uint8_t> raw(count);
    for (std::size_t i = 0; i < count; ++i)
      raw[i] = static_cast<std::uint8_t>(std::max(1.0f, std::fabs(generated[i])));
    return Tensor::FromUint8("", shape, raw);
  }
  case BinaryDataType::UINT16: {
    std::vector<std::uint16_t> raw(count);
    for (std::size_t i = 0; i < count; ++i)
      raw[i] = static_cast<std::uint16_t>(std::max(1.0f, std::fabs(generated[i])));
    return Tensor::FromUint16("", shape, raw);
  }
  case BinaryDataType::UINT32: {
    std::vector<std::uint32_t> raw(count);
    for (std::size_t i = 0; i < count; ++i)
      raw[i] = static_cast<std::uint32_t>(std::max(1.0f, std::fabs(generated[i])));
    return Tensor::FromUint32("", shape, raw);
  }
  case BinaryDataType::UINT64: {
    std::vector<std::uint64_t> raw(count);
    for (std::size_t i = 0; i < count; ++i)
      raw[i] = static_cast<std::uint64_t>(std::max(1.0f, std::fabs(generated[i])));
    return Tensor::FromUint64("", shape, raw);
  }
  default:
    throw std::invalid_argument("Unsupported binary tensor type.");
  }
}

BinaryKernelDescriptor::Attributes DefaultAttributes(const BinaryManifestEntry &entry,
                                                     BinaryDataType left_type) {
  BinaryKernelDescriptor::Attributes attributes;
  if (entry.op == BinaryOperator::kMod) {
    const bool is_float =
        left_type == BinaryDataType::FLOAT || left_type == BinaryDataType::DOUBLE ||
        left_type == BinaryDataType::FLOAT16 || left_type == BinaryDataType::BFLOAT16;
    attributes.mod_fmod = is_float ? 1 : 0;
  } else if (entry.op == BinaryOperator::kBitShift) {
    attributes.bitshift_direction = BinaryKernelDescriptor::Attributes::BitShiftDirection::kLeft;
  }
  return attributes;
}

IoData MakeBinaryIoData(const BinaryManifestEntry &entry, const BinaryTypeSignature &signature,
                        const ShapePair &shape_pair,
                        const BinaryKernelDescriptor::Attributes &attributes, bool swap_operands) {
  const bool division_like = entry.op == BinaryOperator::kDiv || entry.op == BinaryOperator::kMod;
  const bool pow_like = entry.op == BinaryOperator::kPow;
  const bool shift_like = entry.op == BinaryOperator::kBitShift;
  Tensor left = MakeTypedTensor(signature.left, swap_operands ? shape_pair.right : shape_pair.left,
                                {}, entry.op == BinaryOperator::kBitShift,
                                division_like || shift_like, pow_like || shift_like);
  Tensor right =
      MakeTypedTensor(signature.right, swap_operands ? shape_pair.left : shape_pair.right, {}, true,
                      division_like || shift_like, true);
  const OpsetId opset(std::string(), entry.since_version);
  const NodeProto node = MakeNode(entry.op_type, attributes);
  const BinaryElementwiseKernel kernel(node, KernelContext{opset});
  Tensor output = kernel(left, right);
  return IoData{{std::move(left), std::move(right)}, {std::move(output)}};
}

std::string CaseName(const BinaryManifestEntry &entry, const BinaryTypeSignature &signature,
                     const ShapePair &shape_pair,
                     const BinaryKernelDescriptor::Attributes &attributes, bool swap_operands,
                     bool benchmark, std::int64_t size_override = -1) {
  std::string op = std::string(entry.op_type);
  for (char &c : op) {
    if (c >= 'A' && c <= 'Z')
      c = static_cast<char>(c - 'A' + 'a');
  }
  std::string name = "test_cpu_" + op + "_v" + std::to_string(entry.since_version) + "_" +
                     shape_pair.tag + "_" + DataTypeSuffix(signature.left) + "x" +
                     DataTypeSuffix(signature.right) + "_to_" + DataTypeSuffix(signature.output);
  if (entry.op == BinaryOperator::kMod) {
    name += attributes.mod_fmod == 0 ? "_fmod0" : "_fmod1";
  }
  if (entry.op == BinaryOperator::kBitShift) {
    name += attributes.bitshift_direction ==
                    BinaryKernelDescriptor::Attributes::BitShiftDirection::kLeft
                ? "_left"
                : "_right";
  }
  if (swap_operands)
    name += "_swapped";
  if (benchmark) {
    name += "_n" + std::to_string(size_override) + "_benchmark";
  }
  return name;
}

void RegisterBenchmarksForSignature(std::vector<TestCase> &registry,
                                    const BinaryManifestEntry &entry,
                                    const BinaryTypeSignature &signature,
                                    bool cover_priority_shapes) {
  const std::vector<ShapePair> &shape_pairs = BinaryBenchmarkShapePairs();
  const std::size_t shape_count = cover_priority_shapes ? shape_pairs.size() : 1;
  for (std::size_t shape_index = 0; shape_index < shape_count; ++shape_index) {
    const ShapePair &shape_pair = shape_pairs[shape_index];
    const std::int64_t left_count = static_cast<std::int64_t>(ElementCount(shape_pair.left));
    const std::int64_t right_count = static_cast<std::int64_t>(ElementCount(shape_pair.right));
    const std::int64_t output_count =
        static_cast<std::int64_t>(BroadcastElementCount(shape_pair.left, shape_pair.right));
    BinaryKernelDescriptor::Attributes attributes = DefaultAttributes(entry, signature.left);
    const NodeProto node = MakeNode(entry.op_type, attributes);
    const OpsetId opset(std::string(), entry.since_version);
    const std::string name =
        CaseName(entry, signature, shape_pair, attributes, false, true, output_count);
    Expect(registry, node, name, {opset}, {left_count, right_count}, {output_count},
           [entry, signature, shape_pair, attributes]() -> IoData {
             return MakeBinaryIoData(entry, signature, shape_pair, attributes, false);
           });
    if (IsNonCommutative(entry.op)) {
      const std::string swapped_name =
          CaseName(entry, signature, shape_pair, attributes, true, true, output_count);
      Expect(registry, node, swapped_name, {opset}, {right_count, left_count}, {output_count},
             [entry, signature, shape_pair, attributes]() -> IoData {
               return MakeBinaryIoData(entry, signature, shape_pair, attributes, true);
             });
    }
    if (entry.op == BinaryOperator::kBitShift) {
      attributes.bitshift_direction = BinaryKernelDescriptor::Attributes::BitShiftDirection::kRight;
      const NodeProto right_node = MakeNode(entry.op_type, attributes);
      const std::string right_name =
          CaseName(entry, signature, shape_pair, attributes, false, true, output_count);
      Expect(registry, right_node, right_name, {opset}, {left_count, right_count}, {output_count},
             [entry, signature, shape_pair, attributes]() -> IoData {
               return MakeBinaryIoData(entry, signature, shape_pair, attributes, false);
             });
    }
  }
}

} // namespace

void RegisterCpuBinaryCases(std::vector<TestCase> &registry, const std::string &op_type,
                            TestMode mode) {
  const BinaryManifestEntry &entry = GetBinaryManifestEntry(op_type);
  if (mode == TestMode::BENCHMARK) {
    const std::size_t priority_signature = entry.op == BinaryOperator::kEqual ? 1 : 0;
    for (std::size_t signature_index = 0; signature_index < entry.signatures.size();
         ++signature_index) {
      RegisterBenchmarksForSignature(registry, entry, entry.signatures[signature_index],
                                     signature_index == priority_signature);
    }
    return;
  }

  for (const BinaryTypeSignature &signature : entry.signatures) {
    for (const ShapePair &shape_pair : BinaryShapePairs()) {
      BinaryKernelDescriptor::Attributes attributes = DefaultAttributes(entry, signature.left);
      const NodeProto node = MakeNode(entry.op_type, attributes);
      const OpsetId opset(std::string(), entry.since_version);
      const std::string name = CaseName(entry, signature, shape_pair, attributes, false, false);
      Expect(registry, node, name, {opset}, [entry, signature, shape_pair, attributes]() -> IoData {
        return MakeBinaryIoData(entry, signature, shape_pair, attributes, false);
      });
      if (IsNonCommutative(entry.op)) {
        const std::string swapped_name =
            CaseName(entry, signature, shape_pair, attributes, true, false);
        Expect(registry, node, swapped_name, {opset},
               [entry, signature, shape_pair, attributes]() -> IoData {
                 return MakeBinaryIoData(entry, signature, shape_pair, attributes, true);
               });
      }
      if (entry.op == BinaryOperator::kBitShift) {
        attributes.bitshift_direction =
            BinaryKernelDescriptor::Attributes::BitShiftDirection::kRight;
        const NodeProto right_node = MakeNode(entry.op_type, attributes);
        const std::string right_name =
            CaseName(entry, signature, shape_pair, attributes, false, false);
        Expect(registry, right_node, right_name, {opset},
               [entry, signature, shape_pair, attributes]() -> IoData {
                 return MakeBinaryIoData(entry, signature, shape_pair, attributes, false);
               });
      }
      if (entry.op == BinaryOperator::kMod &&
          !(signature.left == BinaryDataType::FLOAT || signature.left == BinaryDataType::DOUBLE ||
            signature.left == BinaryDataType::FLOAT16 ||
            signature.left == BinaryDataType::BFLOAT16)) {
        attributes.mod_fmod = 1;
        const NodeProto fmod_node = MakeNode(entry.op_type, attributes);
        const std::string fmod_name =
            CaseName(entry, signature, shape_pair, attributes, false, false);
        Expect(registry, fmod_node, fmod_name, {opset},
               [entry, signature, shape_pair, attributes]() -> IoData {
                 return MakeBinaryIoData(entry, signature, shape_pair, attributes, false);
               });
      }
    }
  }
}

} // namespace onnx_light_cpu::backend_test
