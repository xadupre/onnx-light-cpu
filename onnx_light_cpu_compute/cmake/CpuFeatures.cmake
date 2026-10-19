include_guard(GLOBAL)

set(ONNX_LIGHT_CPU_MAX_SIMD_LEVEL "AUTO" CACHE STRING
    "Maximum x86 SIMD level compiled and reported at runtime: AUTO, AVX2, or AVX512.")
set_property(CACHE ONNX_LIGHT_CPU_MAX_SIMD_LEVEL PROPERTY STRINGS AUTO AVX2 AVX512)
string(TOUPPER "${ONNX_LIGHT_CPU_MAX_SIMD_LEVEL}" _onnx_light_cpu_max_simd_level)
if(NOT _onnx_light_cpu_max_simd_level MATCHES "^(AUTO|AVX2|AVX512)$")
  message(FATAL_ERROR
      "ONNX_LIGHT_CPU_MAX_SIMD_LEVEL must be AUTO, AVX2, or AVX512; "
      "got '${ONNX_LIGHT_CPU_MAX_SIMD_LEVEL}'.")
endif()

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
# The AVX intrinsics live behind __x86_64__/_M_X64 guards, so the AVX compile
# flags only make sense on x86/x86-64 targets. On other architectures (for
# example Apple Silicon arm64) the kernels fall back to scalar code and passing
# flags such as -mavx2 would break the build, so detect the target CPU family.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86|x86|X86)$")
  set(ONNX_LIGHT_CPU_X86 TRUE)
else()
  set(ONNX_LIGHT_CPU_X86 FALSE)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(ONNX_LIGHT_CPU_ARM64 TRUE)
else()
  set(ONNX_LIGHT_CPU_ARM64 FALSE)
endif()

# Dispatch, validation, and scalar code must keep the compiler's baseline ISA.
# ISA flags belong exclusively to the implementation source groups below.
if(ONNX_LIGHT_CPU_SIMD_FLAGS)
  message(FATAL_ERROR
      "ONNX_LIGHT_CPU_SIMD_FLAGS is no longer supported: target-wide ISA flags "
      "make runtime fallbacks unsafe. Remove this entry from the CMake cache; "
      "use ONNX_LIGHT_CPU_MAX_SIMD_LEVEL to limit specialized kernels.")
endif()

if(ONNX_LIGHT_CPU_X86 AND _onnx_light_cpu_max_simd_level STREQUAL "AVX2")
  if(CMAKE_CXX_FLAGS MATCHES "[Aa][Vv][Xx]512|[Aa][Mm][Xx]")
    message(FATAL_ERROR
        "ONNX_LIGHT_CPU_MAX_SIMD_LEVEL=AVX2 is incompatible with explicit "
        "AVX-512 or AMX options in CMAKE_CXX_FLAGS.")
  endif()
  set(_onnx_light_cpu_max_simd_level_value 3)
else()
  set(_onnx_light_cpu_max_simd_level_value 4)
endif()

if(ONNX_LIGHT_CPU_ARM64)
  if(MSVC)
    set(_arm_neon_header "arm64_neon.h")
  else()
    set(_arm_neon_header "arm_neon.h")
  endif()
  check_cxx_source_compiles(
      "#include <${_arm_neon_header}>
       int main() {
         float32x4_t value = vdupq_n_f32(1.0f);
         value = vfmaq_n_f32(value, value, 2.0f);
         return static_cast<int>(vgetq_lane_f32(value, 0));
       }"
      ONNX_LIGHT_CPU_HAVE_NEON_INTRINSICS)
else()
  set(ONNX_LIGHT_CPU_HAVE_NEON_INTRINSICS FALSE)
endif()

# The FP16 -> FP32 packing conversion uses the NEON ``vcvt_f32_f16`` (``FCVTL``)
# instruction, which needs the FP16 vector load/convert intrinsics. These are
# baseline on AArch64 but require a working ``float16_t`` storage type; probe for
# them so the FLOAT16 packing path stays scalar when they are unavailable.
if(ONNX_LIGHT_CPU_HAVE_NEON_INTRINSICS)
  check_cxx_source_compiles(
      "#include <${_arm_neon_header}>
       int main() {
         const float16_t values[8] = {};
         float16x8_t half = vld1q_f16(values);
         float32x4_t wide = vcvt_f32_f16(vget_low_f16(half));
         return static_cast<int>(vgetq_lane_f32(wide, 0));
       }"
      ONNX_LIGHT_CPU_HAVE_NEON_FP16_INTRINSICS)
else()
  set(ONNX_LIGHT_CPU_HAVE_NEON_FP16_INTRINSICS FALSE)
endif()

if(ONNX_LIGHT_CPU_ARM64 AND NOT MSVC)
  set(_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -march=armv8-a+sve")
  check_cxx_source_compiles(
      "#include <arm_sve.h>
       int main() { return static_cast<int>(svcntb() == 0); }"
      ONNX_LIGHT_CPU_HAVE_SVE_INTRINSICS)
  set(CMAKE_REQUIRED_FLAGS "${_saved_required_flags}")
else()
  set(ONNX_LIGHT_CPU_HAVE_SVE_INTRINSICS FALSE)
endif()

# The INT8 MatMulInteger fast path uses the NEON dot-product (``UDOT``, ARMv8.2-A
# ``+dotprod``) instruction. Probe for the intrinsics so the integer path keeps
# its portable scalar reduction on toolchains or targets without them.
if(ONNX_LIGHT_CPU_HAVE_NEON_INTRINSICS AND NOT MSVC)
  set(_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -march=armv8.2-a+dotprod")
  check_cxx_source_compiles(
      "#include <${_arm_neon_header}>
       int main() {
         uint32x4_t acc = vdupq_n_u32(0);
         uint8x16_t bytes = vdupq_n_u8(1);
         acc = vdotq_u32(acc, bytes, bytes);
         return static_cast<int>(vgetq_lane_u32(acc, 0));
       }"
      ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD_INTRINSICS)
  set(CMAKE_REQUIRED_FLAGS "${_saved_required_flags}")
else()
  set(ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD_INTRINSICS FALSE)
endif()

# Dedicated SIMD translation units use feature-specific flags. The remaining
# sources keep the baseline ISA, including DetectSimdLevel() and all fallbacks.
# Unsupported sources are excluded below.
if(ONNX_LIGHT_CPU_X86)
  if(NOT MSVC)
    check_cxx_compiler_flag("-mssse3" ONNX_LIGHT_CPU_HAVE_SSSE3_FLAG)
    check_cxx_compiler_flag("-mavx" ONNX_LIGHT_CPU_HAVE_AVX_FLAG)
    check_cxx_compiler_flag("-mavx2" ONNX_LIGHT_CPU_HAVE_AVX2_FLAG)
  else()
    check_cxx_source_compiles(
        "#include <tmmintrin.h>
         int main() {
           __m128i x = _mm_setzero_si128();
           return _mm_cvtsi128_si32(_mm_abs_epi8(x));
         }"
        ONNX_LIGHT_CPU_HAVE_SSSE3_FLAG)
    check_cxx_compiler_flag("/arch:AVX" ONNX_LIGHT_CPU_HAVE_AVX_FLAG)
    check_cxx_compiler_flag("/arch:AVX2" ONNX_LIGHT_CPU_HAVE_AVX2_FLAG)
  endif()
else()
  set(ONNX_LIGHT_CPU_HAVE_SSSE3_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX2_FLAG FALSE)
endif()

if(NOT ONNX_LIGHT_CPU_X86)
  # Non-x86 targets (e.g. arm64) have no AVX support; keep the SIMD sources
  # excluded and let the scalar fallbacks handle every operator.
  set(ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_F16C_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512F_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512DQ_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512BW_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512FP16_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512BF16_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512VNNI_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AMX_TILE_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AMX_BF16_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AMX_INT8_FLAG FALSE)
elseif(_onnx_light_cpu_max_simd_level STREQUAL "AVX2")
  if(NOT MSVC)
    check_cxx_compiler_flag("-mavx2 -mfma" ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG)
    check_cxx_compiler_flag("-mavx -mf16c" ONNX_LIGHT_CPU_HAVE_F16C_FLAG)
  else()
    check_cxx_compiler_flag("/arch:AVX2" ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG)
    check_cxx_compiler_flag("/arch:AVX" ONNX_LIGHT_CPU_HAVE_F16C_FLAG)
  endif()
  set(ONNX_LIGHT_CPU_HAVE_AVX512F_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512DQ_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512BW_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512FP16_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512BF16_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512VNNI_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AMX_TILE_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AMX_BF16_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AMX_INT8_FLAG FALSE)
elseif(NOT MSVC)
  check_cxx_compiler_flag("-mavx2 -mfma" ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG)
  check_cxx_compiler_flag("-mavx -mf16c" ONNX_LIGHT_CPU_HAVE_F16C_FLAG)
  check_cxx_compiler_flag("-mavx512f" ONNX_LIGHT_CPU_HAVE_AVX512F_FLAG)
  check_cxx_compiler_flag("-mavx512f -mavx512dq"
      ONNX_LIGHT_CPU_HAVE_AVX512DQ_FLAG)
  check_cxx_compiler_flag("-mavx512f -mavx512bw"
      ONNX_LIGHT_CPU_HAVE_AVX512BW_FLAG)
  check_cxx_compiler_flag("-mavx512fp16" ONNX_LIGHT_CPU_HAVE_AVX512FP16_FLAG)
  check_cxx_compiler_flag("-mavx512f -mavx512bw -mavx512bf16"
      ONNX_LIGHT_CPU_HAVE_AVX512BF16_FLAG)
  # The native INT8 VNNI dot-product needs -mavx512vnni for the vpdpbusd
  # intrinsic; it is only compiled when the compiler accepts the flag.
  check_cxx_compiler_flag("-mavx512f -mavx512vnni"
      ONNX_LIGHT_CPU_HAVE_AVX512VNNI_FLAG)
  # The AMX tile-state lifecycle (LDTILECFG/TILERELEASE) needs -mamx-tile for the
  # tile intrinsics; the module keeps a safe fallback when the flag is absent.
  check_cxx_compiler_flag("-mamx-tile" ONNX_LIGHT_CPU_HAVE_AMX_TILE_FLAG)
  # The native AMX-BF16 GEMM kernel additionally needs -mamx-bf16 for the
  # tdpbf16ps tile dot-product; it is only compiled when both flags are present.
  check_cxx_compiler_flag("-mamx-tile -mamx-bf16" ONNX_LIGHT_CPU_HAVE_AMX_BF16_FLAG)
  check_cxx_compiler_flag("-mamx-tile -mamx-int8" ONNX_LIGHT_CPU_HAVE_AMX_INT8_FLAG)
else()
  check_cxx_compiler_flag("/arch:AVX2" ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG)
  check_cxx_compiler_flag("/arch:AVX" ONNX_LIGHT_CPU_HAVE_F16C_FLAG)
  check_cxx_compiler_flag("/arch:AVX512" ONNX_LIGHT_CPU_HAVE_AVX512F_FLAG)
  set(ONNX_LIGHT_CPU_HAVE_AVX512DQ_FLAG
      ${ONNX_LIGHT_CPU_HAVE_AVX512F_FLAG})
  set(ONNX_LIGHT_CPU_HAVE_AVX512BW_FLAG
      ${ONNX_LIGHT_CPU_HAVE_AVX512F_FLAG})
  # MSVC has no native AVX-512FP16 (_Float16 / __m512h) support; keep the
  # native FLOAT16 kernel out of the build there and rely on the converting
  # float32 fallback path.
  set(ONNX_LIGHT_CPU_HAVE_AVX512FP16_FLAG FALSE)
  # MSVC has no -mavx512bf16 equivalent for the __m512bh / _mm512_dpbf16_ps
  # intrinsics used by the native BFLOAT16 kernel; keep it out of the build and
  # rely on the converting float32 fallback path.
  set(ONNX_LIGHT_CPU_HAVE_AVX512BF16_FLAG FALSE)
  # MSVC has no -mavx512vnni equivalent for the _mm512_dpbusd_epi32 intrinsic
  # used by the native INT8 VNNI dot-product; keep it out of the build and rely
  # on the portable scalar fallback path.
  set(ONNX_LIGHT_CPU_HAVE_AVX512VNNI_FLAG FALSE)
  # MSVC has no -mamx-tile equivalent for the tile intrinsics used by the AMX
  # tile-state lifecycle; keep the native path out of the build and rely on the
  # safe fallback.
  set(ONNX_LIGHT_CPU_HAVE_AMX_TILE_FLAG FALSE)
  # MSVC has no -mamx-bf16 equivalent for the tdpbf16ps intrinsic used by the
  # native AMX-BF16 kernel; keep it out of the build and rely on the AVX-512BF16
  # or converting float32 fallback path.
  set(ONNX_LIGHT_CPU_HAVE_AMX_BF16_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AMX_INT8_FLAG FALSE)
endif()

if(ONNX_LIGHT_CPU_X86 AND NOT MSVC)
  # Check the intrinsic as well as the target flag; AVX-VNNI is independent
  # of AVX-512 and remains available under the AVX2 build ceiling.
  set(_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -mavx2 -mavxvnni")
  check_cxx_source_compiles(
      "#include <immintrin.h>
       int main() {
         __m256i x = _mm256_setzero_si256();
         __m128i y = _mm_setzero_si128();
         return _mm256_movemask_epi8(_mm256_dpbusd_epi32(x, x, x)) +
                _mm_cvtsi128_si32(_mm_dpbusd_epi32(y, y, y));
       }"
      ONNX_LIGHT_CPU_HAVE_AVXVNNI_FLAG)
  set(CMAKE_REQUIRED_FLAGS "${_saved_required_flags}")
else()
  set(ONNX_LIGHT_CPU_HAVE_AVXVNNI_FLAG FALSE)
endif()
