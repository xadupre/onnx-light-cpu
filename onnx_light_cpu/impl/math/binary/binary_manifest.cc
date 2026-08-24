// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/math/binary/binary_manifest.h"

#include <array>
#include <stdexcept>
#include <string>

namespace onnx_light_cpu {
namespace {

using DT = BinaryDataType;
using Signature = BinaryTypeSignature;

constexpr std::array<Signature, 12> kArithmeticSignatures = {
    {{DT::FLOAT, DT::FLOAT, DT::FLOAT},
     {DT::DOUBLE, DT::DOUBLE, DT::DOUBLE},
     {DT::FLOAT16, DT::FLOAT16, DT::FLOAT16},
     {DT::BFLOAT16, DT::BFLOAT16, DT::BFLOAT16},
     {DT::INT8, DT::INT8, DT::INT8},
     {DT::INT16, DT::INT16, DT::INT16},
     {DT::INT32, DT::INT32, DT::INT32},
     {DT::INT64, DT::INT64, DT::INT64},
     {DT::UINT8, DT::UINT8, DT::UINT8},
     {DT::UINT16, DT::UINT16, DT::UINT16},
     {DT::UINT32, DT::UINT32, DT::UINT32},
     {DT::UINT64, DT::UINT64, DT::UINT64}}};

constexpr std::array<Signature, 11> kComparisonSignatures = {
    {{DT::FLOAT, DT::FLOAT, DT::BOOL},
     {DT::FLOAT16, DT::FLOAT16, DT::BOOL},
     {DT::BFLOAT16, DT::BFLOAT16, DT::BOOL},
     {DT::INT8, DT::INT8, DT::BOOL},
     {DT::INT16, DT::INT16, DT::BOOL},
     {DT::INT32, DT::INT32, DT::BOOL},
     {DT::INT64, DT::INT64, DT::BOOL},
     {DT::UINT8, DT::UINT8, DT::BOOL},
     {DT::UINT16, DT::UINT16, DT::BOOL},
     {DT::UINT32, DT::UINT32, DT::BOOL},
     {DT::UINT64, DT::UINT64, DT::BOOL}}};

constexpr std::array<Signature, 13> kEqualSignatures = {{{DT::BOOL, DT::BOOL, DT::BOOL},
                                                         {DT::FLOAT, DT::FLOAT, DT::BOOL},
                                                         {DT::DOUBLE, DT::DOUBLE, DT::BOOL},
                                                         {DT::FLOAT16, DT::FLOAT16, DT::BOOL},
                                                         {DT::BFLOAT16, DT::BFLOAT16, DT::BOOL},
                                                         {DT::INT8, DT::INT8, DT::BOOL},
                                                         {DT::INT16, DT::INT16, DT::BOOL},
                                                         {DT::INT32, DT::INT32, DT::BOOL},
                                                         {DT::INT64, DT::INT64, DT::BOOL},
                                                         {DT::UINT8, DT::UINT8, DT::BOOL},
                                                         {DT::UINT16, DT::UINT16, DT::BOOL},
                                                         {DT::UINT32, DT::UINT32, DT::BOOL},
                                                         {DT::UINT64, DT::UINT64, DT::BOOL}}};

constexpr std::array<Signature, 1> kLogicalSignatures = {{{DT::BOOL, DT::BOOL, DT::BOOL}}};

constexpr std::array<Signature, 8> kBitwiseSignatures = {{{DT::INT8, DT::INT8, DT::INT8},
                                                          {DT::INT16, DT::INT16, DT::INT16},
                                                          {DT::INT32, DT::INT32, DT::INT32},
                                                          {DT::INT64, DT::INT64, DT::INT64},
                                                          {DT::UINT8, DT::UINT8, DT::UINT8},
                                                          {DT::UINT16, DT::UINT16, DT::UINT16},
                                                          {DT::UINT32, DT::UINT32, DT::UINT32},
                                                          {DT::UINT64, DT::UINT64, DT::UINT64}}};

constexpr std::array<Signature, 4> kBitShiftSignatures = {{{DT::UINT8, DT::UINT8, DT::UINT8},
                                                           {DT::UINT16, DT::UINT16, DT::UINT16},
                                                           {DT::UINT32, DT::UINT32, DT::UINT32},
                                                           {DT::UINT64, DT::UINT64, DT::UINT64}}};

constexpr std::array<Signature, 8> kPReluSignatures = {{{DT::FLOAT, DT::FLOAT, DT::FLOAT},
                                                        {DT::DOUBLE, DT::DOUBLE, DT::DOUBLE},
                                                        {DT::FLOAT16, DT::FLOAT16, DT::FLOAT16},
                                                        {DT::BFLOAT16, DT::BFLOAT16, DT::BFLOAT16},
                                                        {DT::INT32, DT::INT32, DT::INT32},
                                                        {DT::INT64, DT::INT64, DT::INT64},
                                                        {DT::UINT32, DT::UINT32, DT::UINT32},
                                                        {DT::UINT64, DT::UINT64, DT::UINT64}}};

constexpr std::array<Signature, 29> kPowSignatures = {{
    {DT::FLOAT, DT::FLOAT, DT::FLOAT},          {DT::FLOAT, DT::INT32, DT::FLOAT},
    {DT::FLOAT, DT::INT64, DT::FLOAT},          {DT::FLOAT, DT::UINT32, DT::FLOAT},
    {DT::FLOAT, DT::UINT64, DT::FLOAT},         {DT::FLOAT16, DT::FLOAT, DT::FLOAT16},
    {DT::FLOAT16, DT::FLOAT16, DT::FLOAT16},    {DT::FLOAT16, DT::BFLOAT16, DT::FLOAT16},
    {DT::FLOAT16, DT::INT32, DT::FLOAT16},      {DT::FLOAT16, DT::INT64, DT::FLOAT16},
    {DT::FLOAT16, DT::UINT32, DT::FLOAT16},     {DT::FLOAT16, DT::UINT64, DT::FLOAT16},
    {DT::BFLOAT16, DT::FLOAT, DT::BFLOAT16},    {DT::BFLOAT16, DT::FLOAT16, DT::BFLOAT16},
    {DT::BFLOAT16, DT::BFLOAT16, DT::BFLOAT16}, {DT::BFLOAT16, DT::INT32, DT::BFLOAT16},
    {DT::BFLOAT16, DT::INT64, DT::BFLOAT16},    {DT::BFLOAT16, DT::UINT32, DT::BFLOAT16},
    {DT::BFLOAT16, DT::UINT64, DT::BFLOAT16},   {DT::INT32, DT::FLOAT, DT::INT32},
    {DT::INT32, DT::INT32, DT::INT32},          {DT::INT32, DT::INT64, DT::INT32},
    {DT::INT32, DT::UINT32, DT::INT32},         {DT::INT32, DT::UINT64, DT::INT32},
    {DT::INT64, DT::FLOAT, DT::INT64},          {DT::INT64, DT::INT32, DT::INT64},
    {DT::INT64, DT::INT64, DT::INT64},          {DT::INT64, DT::UINT32, DT::INT64},
    {DT::INT64, DT::UINT64, DT::INT64},
}};

constexpr std::array<BinaryManifestEntry, 19> kManifest = {{
    {BinaryOperator::kAdd, "Add", 14, kArithmeticSignatures},
    {BinaryOperator::kSub, "Sub", 14, kArithmeticSignatures},
    {BinaryOperator::kMul, "Mul", 14, kArithmeticSignatures},
    {BinaryOperator::kDiv, "Div", 14, kArithmeticSignatures},
    {BinaryOperator::kMod, "Mod", 13, kArithmeticSignatures},
    {BinaryOperator::kPow, "Pow", 15, kPowSignatures},
    {BinaryOperator::kEqual, "Equal", 19, kEqualSignatures},
    {BinaryOperator::kGreater, "Greater", 13, kComparisonSignatures},
    {BinaryOperator::kGreaterOrEqual, "GreaterOrEqual", 16, kComparisonSignatures},
    {BinaryOperator::kLess, "Less", 13, kComparisonSignatures},
    {BinaryOperator::kLessOrEqual, "LessOrEqual", 16, kComparisonSignatures},
    {BinaryOperator::kAnd, "And", 7, kLogicalSignatures},
    {BinaryOperator::kOr, "Or", 7, kLogicalSignatures},
    {BinaryOperator::kXor, "Xor", 7, kLogicalSignatures},
    {BinaryOperator::kBitwiseAnd, "BitwiseAnd", 18, kBitwiseSignatures},
    {BinaryOperator::kBitwiseOr, "BitwiseOr", 18, kBitwiseSignatures},
    {BinaryOperator::kBitwiseXor, "BitwiseXor", 18, kBitwiseSignatures},
    {BinaryOperator::kBitShift, "BitShift", 11, kBitShiftSignatures},
    {BinaryOperator::kPRelu, "PRelu", 16, kPReluSignatures},
}};

} // namespace

std::span<const BinaryManifestEntry> GetBinaryManifest() noexcept { return kManifest; }

const BinaryManifestEntry &GetBinaryManifestEntry(BinaryOperator op) {
  for (const BinaryManifestEntry &entry : kManifest) {
    if (entry.op == op) {
      return entry;
    }
  }
  throw std::invalid_argument("onnx_light_cpu::GetBinaryManifestEntry: unknown operator id.");
}

const BinaryManifestEntry &GetBinaryManifestEntry(std::string_view op_type) {
  for (const BinaryManifestEntry &entry : kManifest) {
    if (entry.op_type == op_type) {
      return entry;
    }
  }
  throw std::invalid_argument("onnx_light_cpu::GetBinaryManifestEntry: unsupported operator '" +
                              std::string(op_type) + "'.");
}

std::string_view BinaryOperatorName(BinaryOperator op) noexcept {
  for (const BinaryManifestEntry &entry : kManifest) {
    if (entry.op == op) {
      return entry.op_type;
    }
  }
  return "<unknown>";
}

} // namespace onnx_light_cpu
