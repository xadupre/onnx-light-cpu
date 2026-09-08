// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/impl/tensor/cast_kernel.h"

#include "onnx_light_cpu/impl/execution.h"
#include "onnx_light_cpu/impl/math/half_conversion.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace onnx_light_cpu {
namespace {

template <DataType Type, typename T> struct Codec {
  using Storage = T;
  static auto Decode(T value) {
    if constexpr (Type == DataType::FLOAT16) {
      return detail::Float16BitsToFloat(value);
    } else if constexpr (Type == DataType::BFLOAT16) {
      return detail::Bfloat16BitsToFloat(value);
    } else if constexpr (Type == DataType::BOOL) {
      return value != 0;
    } else {
      return value;
    }
  }

  template <typename Value> static T Encode(Value value) {
    if constexpr (Type == DataType::FLOAT16) {
      return detail::FloatToFloat16Bits(static_cast<float>(value));
    } else if constexpr (Type == DataType::BFLOAT16) {
      return detail::FloatToBFloat16Bits(static_cast<float>(value));
    } else if constexpr (Type == DataType::BOOL) {
      return static_cast<T>(value != 0);
    } else if constexpr (std::is_integral_v<T> && std::is_floating_point_v<Value>) {
      if (std::isnan(value)) {
        return 0;
      }
      if constexpr (std::is_same_v<T, std::uint64_t>) {
        if (value <= 0) {
          return 0;
        }
        if (value >= static_cast<Value>(0x1p64)) {
          return std::numeric_limits<T>::max();
        }
        return static_cast<T>(value);
      } else {
        // Powers of two are exactly representable even when long double is
        // only DOUBLE (MSVC). Comparing against rounded INT64_MAX is unsafe.
        if (value < static_cast<Value>(-0x1p63)) {
          return std::numeric_limits<T>::lowest();
        }
        if (value >= static_cast<Value>(0x1p63)) {
          return std::numeric_limits<T>::max();
        }
        return static_cast<T>(static_cast<std::int64_t>(value));
      }
    } else {
      return static_cast<T>(value);
    }
  }
};

template <typename Fn> decltype(auto) Dispatch(DataType type, Fn &&fn) {
  switch (type) {
  case DataType::FLOAT:
    return fn(Codec<DataType::FLOAT, float>{});
  case DataType::DOUBLE:
    return fn(Codec<DataType::DOUBLE, double>{});
  case DataType::FLOAT16:
    return fn(Codec<DataType::FLOAT16, std::uint16_t>{});
  case DataType::BFLOAT16:
    return fn(Codec<DataType::BFLOAT16, std::uint16_t>{});
  case DataType::BOOL:
    return fn(Codec<DataType::BOOL, std::uint8_t>{});
  case DataType::INT8:
    return fn(Codec<DataType::INT8, std::int8_t>{});
  case DataType::UINT8:
    return fn(Codec<DataType::UINT8, std::uint8_t>{});
  case DataType::INT16:
    return fn(Codec<DataType::INT16, std::int16_t>{});
  case DataType::UINT16:
    return fn(Codec<DataType::UINT16, std::uint16_t>{});
  case DataType::INT32:
    return fn(Codec<DataType::INT32, std::int32_t>{});
  case DataType::UINT32:
    return fn(Codec<DataType::UINT32, std::uint32_t>{});
  case DataType::INT64:
    return fn(Codec<DataType::INT64, std::int64_t>{});
  case DataType::UINT64:
    return fn(Codec<DataType::UINT64, std::uint64_t>{});
  default:
    throw std::invalid_argument("onnx_light_cpu::Cast: unsupported numeric type.");
  }
}

using ConvertRange = void (*)(const std::uint8_t *, std::uint8_t *, std::size_t, std::size_t);

template <typename From, typename To>
void Convert(const std::uint8_t *input, std::uint8_t *output, std::size_t begin, std::size_t end) {
  for (std::size_t i = begin; i < end; ++i) {
    typename From::Storage value;
    std::memcpy(&value, input + i * sizeof(value), sizeof(value));
    const auto converted = To::Encode(From::Decode(value));
    std::memcpy(output + i * sizeof(converted), &converted, sizeof(converted));
  }
}

} // namespace

bool IsCastNumericType(DataType type) noexcept {
  const auto value = static_cast<std::int32_t>(type);
  return (value >= 1 && value <= 13 && type != DataType::STRING) || type == DataType::BFLOAT16;
}

std::size_t CastElementSize(DataType type) {
  return Dispatch(type, [](auto codec) { return sizeof(typename decltype(codec)::Storage); });
}

void CastConvert(const void *input, DataType from, void *output, DataType to, std::size_t count) {
  const std::size_t source_width = CastElementSize(from);
  const std::size_t output_width = CastElementSize(to);
  const auto limit = std::numeric_limits<std::size_t>::max();
  if (count > limit / source_width || count > limit / output_width) {
    throw std::invalid_argument("onnx_light_cpu::Cast: byte size overflow.");
  }
  if (count == 0) {
    return;
  }
  const auto source_address = reinterpret_cast<std::uintptr_t>(input);
  const auto output_address = reinterpret_cast<std::uintptr_t>(output);
  if (input == nullptr || output == nullptr ||
      (source_address <= output_address ? output_address - source_address < count * source_width
                                        : source_address - output_address < count * output_width)) {
    throw std::invalid_argument("onnx_light_cpu::Cast: null or overlapping buffers.");
  }
  const auto *source = static_cast<const std::uint8_t *>(input);
  auto *destination = static_cast<std::uint8_t *>(output);
  ConvertRange convert = nullptr;
  if (from != to) {
    convert = Dispatch(from, [&](auto source_codec) {
      return Dispatch(to, [&](auto output_codec) -> ConvertRange {
        return &Convert<decltype(source_codec), decltype(output_codec)>;
      });
    });
  }
  const auto run = [&](std::size_t begin, std::size_t end) {
    if (convert != nullptr) {
      convert(source, destination, begin, end);
    } else {
      std::memcpy(destination + begin * output_width, source + begin * source_width,
                  (end - begin) * source_width);
    }
  };
  constexpr std::size_t kTileElements = 8192;
  const auto *executor = CurrentExecutionExecutor();
  if (count < 131072 || executor == nullptr || executor->run_blocks == nullptr ||
      ExecutionThreadCount() <= 1 || ExecutionInParallelRegion()) {
    run(0, count);
    return;
  }
  const auto tiles = static_cast<int64_t>(count / kTileElements + (count % kTileElements != 0));
  ExecutionSchedule schedule;
  schedule.min_parallel_size = 16;
  schedule.min_block_size = 4;
  schedule.max_participants = ExecutionThreadCount();
  ExecuteRanges(tiles, schedule, [&](int64_t begin, int64_t end) {
    run(static_cast<std::size_t>(begin) * kTileElements,
        end == tiles ? count : static_cast<std::size_t>(end) * kTileElements);
  });
}

} // namespace onnx_light_cpu
