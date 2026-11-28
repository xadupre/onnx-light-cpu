# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

execute_process(
    COMMAND "${OBJDUMP}" -d -C --no-show-raw-insn -M intel "${LIBRARY}"
    RESULT_VARIABLE status OUTPUT_VARIABLE assembly ERROR_VARIABLE error)
if(NOT status STREQUAL "0")
  message(FATAL_ERROR "Cannot disassemble ${LIBRARY}: ${status}\n${error}")
endif()

function(require_instruction isa kernel opcode register)
  if(NOT "${isa}" IN_LIST ISAS)
    return()
  endif()
  string(TOUPPER "${register}" register_upper)
  string(REGEX MATCH
      "\n[0-9a-f]+ <ml_cpu_compute::${kernel}\\([^\n]*\\)>:[ \t]*\n([ \t]+[0-9a-f]+:[^\n]*\n)*"
      body "${assembly}")
  if(NOT body)
    message(SEND_ERROR "${isa}: ${kernel} not found in ${LIBRARY}")
  elseif(NOT body MATCHES
         ":[ \t]+(\\{vex\\}[ \t]+)?${opcode}[ \t]+[^\n]*(${register}[0-9]+|${register_upper}WORD)")
    message(SEND_ERROR "${isa}: ${kernel} must contain ${opcode} operating on ${register}")
  endif()
endfunction()

foreach(op IN ITEMS Add Sub Mul Div)
  string(TOLOWER "${op}" mnemonic)
  foreach(layout IN ITEMS "" Left Right)
    require_instruction(AVX Binary${op}Float32${layout}_AVX v${mnemonic}ps ymm)
    require_instruction(AVX Binary${op}Float64${layout}_AVX v${mnemonic}pd ymm)
    require_instruction(AVX512F Binary${op}Float32${layout}_AVX512 v${mnemonic}ps zmm)
    require_instruction(AVX512F Binary${op}Float64${layout}_AVX512 v${mnemonic}pd zmm)
  endforeach()
endforeach()
require_instruction(AVX GemmMicroTile_AVX_F32 vmulps ymm)
require_instruction(AVX2 AbsInt32_AVX2 vpabsd ymm)
require_instruction(AVX2_FMA ExpFloat32_AVX2_FMA "vfmadd[0-9]+ps" ymm)
require_instruction(F16C detail::CastFloat32ToFloat16_F16C vcvtps2ph ymm)
require_instruction(AVXVNNI MatMulNBitsInt8TileAvxVnni vpdpbusd ymm)
require_instruction(AVX512F GemmMicroTile_AVX512_F32_MR12_NR16 "vfmadd[0-9]+ps" zmm)
require_instruction(AVX512BW AbsInt8_AVX512 vpabsb zmm)
require_instruction(AVX512BF16 GemmMicroTile_AVX512BF16 vdpbf16ps zmm)
require_instruction(AVX512FP16 BinaryAddFloat16_AVX512FP16 vaddph zmm)
require_instruction(AVX512VNNI MatMulNBitsInt8TileAvx512Vnni vpdpbusd zmm)
