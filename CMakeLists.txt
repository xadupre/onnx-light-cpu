cmake_minimum_required(VERSION 3.15)

# Default to a Release build when no build type is specified.
get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _is_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

project(onnx_light_cpu VERSION 0.1.16 LANGUAGES CXX)

# CMake 4.1 introduced CMP0178, which stops silently discarding empty test
# launcher / crosscompiling-emulator arguments. With the NEW behaviour,
# gtest_discover_tests passes an empty executor (``-- ""``) to its discovery
# script, so the discovery run tries to execute ``""`` before the test binary.
# On Windows this fails and yields an empty JSON test list, breaking the build
# during POST_BUILD test discovery. Keep the OLD behaviour so empty executor
# arguments are dropped and discovery runs the test binary directly.
if(POLICY CMP0178)
  cmake_policy(SET CMP0178 OLD)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(ONNX_LIGHT_CPU_BUILD_TESTS "Build C++ unit tests." OFF)
option(ONNX_LIGHT_CPU_BUILD_BENCHMARKS
       "Build the isolated C++ throughput benchmarks (not run by ctest)." OFF)
option(ONNX_LIGHT_CPU_BUILD_PYTHON "Build Python extension." ON)
option(ONNX_LIGHT_CPU_INSTALL "Install the C++ library and headers." ON)
option(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT
       "Build the onnx-light kernel registration library (requires onnx-light)." OFF)
set(ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR "" CACHE PATH
    "Path to a local onnx-light source tree whose headers match the already-built \
Python runtime.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY "" CACHE FILEPATH
    "Exact onnx-light core shared library loaded by the Python runtime.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_LIBRARY "" CACHE FILEPATH
    "Exact onnx-light proto shared library loaded by the Python runtime.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_IMPLIB "" CACHE FILEPATH
    "Windows import library for ONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_IMPLIB "" CACHE FILEPATH
    "Windows import library for ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_LIBRARY.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_KERNELS_IMPLIB "" CACHE FILEPATH
    "Windows import library for the onnx-light kernels runtime.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_BACKEND_TEST_IMPLIB "" CACHE FILEPATH
    "Windows import library for the onnx-light backend-test runtime.")
option(ONNX_LIGHT_CPU_ENABLE_COVERAGE
       "Instrument the C++ targets for code coverage (GCC/Clang only)." OFF)

# A local onnx-light runtime target is not part of an export set, so
# lib_onnx_light_cpu_kernels cannot be installed as part of onnx_light_cpuTargets
# (its PUBLIC link dependency onnx_light::lib_onnx_core would need to be
# exported too). Disable the C++ install step in that mode; it is normally
# only used for local/dev builds (e.g. --cpp-tests) anyway.
if(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT AND ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR
   AND ONNX_LIGHT_CPU_INSTALL)
  message(STATUS
      "onnx-light-cpu: disabling ONNX_LIGHT_CPU_INSTALL because "
      "ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR is set (onnx-light built from "
      "source is not part of an installable export set).")
  set(ONNX_LIGHT_CPU_INSTALL OFF CACHE BOOL
      "Install the C++ library and headers." FORCE)
endif()
option(ONNX_HARDENING
    "Enable OpenSSF Compiler Hardening Guide flags for C/C++ targets. \
See cmake/Hardening.cmake and \
https://best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++.html"
    OFF)

if(ONNX_LIGHT_CPU_ENABLE_COVERAGE AND
   NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(FATAL_ERROR
      "ONNX_LIGHT_CPU_ENABLE_COVERAGE requires a GCC or Clang "
      "compiler; got '${CMAKE_CXX_COMPILER_ID}'.")
endif()

if(MSVC)
  if(NOT CMAKE_GENERATOR MATCHES "Ninja")
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/MP>)
  endif()
endif()

include(GNUInstallDirs)

# Thread topology detection uses std::thread::hardware_concurrency().
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Optional OpenSSF compiler hardening flags.
if(ONNX_HARDENING)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(Hardening)
else()
  function(onnx_light_apply_hardening target)
  endfunction()
endif()

# ---------------------------------------------------------------------------
# SIMD / AVX compile flags
# ---------------------------------------------------------------------------
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

# On x86 platforms, enable AVX2 by default for the kernel sources so the
# compiler can emit AVX/AVX2 intrinsics. The runtime dispatch still falls
# back to SSE2/scalar on older CPUs. AVX-512 codepaths are guarded by
# __AVX512F__ and only compiled when the user explicitly passes -mavx512f.
if(NOT ONNX_LIGHT_CPU_X86)
  set(ONNX_LIGHT_CPU_SIMD_FLAGS "" CACHE STRING
      "SIMD compiler flags for kernel sources (e.g. -mavx2, -mavx512f).")
elseif(NOT MSVC)
  set(ONNX_LIGHT_CPU_SIMD_FLAGS "-mavx2" CACHE STRING
      "SIMD compiler flags for kernel sources (e.g. -mavx2, -mavx512f).")
else()
  set(ONNX_LIGHT_CPU_SIMD_FLAGS "/arch:AVX2" CACHE STRING
      "SIMD compiler flags for kernel sources (e.g. /arch:AVX2, /arch:AVX512).")
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

# The dedicated AVX-512 translation units are compiled with feature-specific
# flags on top of ONNX_LIGHT_CPU_SIMD_FLAGS. This lets a binary built with the
# default -mavx2 still carry wider kernels selected by DetectSimdLevel() at
# runtime. Exp, Log, Abs and Gemm only need AVX-512F; Not additionally needs
# AVX-512BW for byte comparisons. Unsupported sources are excluded below.
if(NOT ONNX_LIGHT_CPU_X86)
  # Non-x86 targets (e.g. arm64) have no AVX support; keep the SIMD sources
  # excluded and let the scalar fallbacks handle every operator.
  set(ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_F16C_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512F_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512BW_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512FP16_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512BF16_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AVX512VNNI_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AMX_TILE_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AMX_BF16_FLAG FALSE)
  set(ONNX_LIGHT_CPU_HAVE_AMX_INT8_FLAG FALSE)
elseif(NOT MSVC)
  check_cxx_compiler_flag("-mavx2 -mfma" ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG)
  check_cxx_compiler_flag("-mavx2 -mfma -mf16c" ONNX_LIGHT_CPU_HAVE_F16C_FLAG)
  check_cxx_compiler_flag("-mavx512f" ONNX_LIGHT_CPU_HAVE_AVX512F_FLAG)
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
  # /arch:AVX2 already enables the F16C intrinsics on MSVC.
  set(ONNX_LIGHT_CPU_HAVE_F16C_FLAG ${ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG})
  check_cxx_compiler_flag("/arch:AVX512" ONNX_LIGHT_CPU_HAVE_AVX512F_FLAG)
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

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
# Discover the kernel and binding sources with globbing so new files are picked
# up without editing this list. CONFIGURE_DEPENDS re-runs the glob when the set
# of matching files changes.
file(GLOB CPU_KERNEL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/*/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/*/*/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/*/*/*/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/reference/*.cc"
)

set(_gemm_avx512_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/gemm/avx512/gemm_kernel_avx512.cc")
set(_gemm_avx512fp16_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/gemm/avx512fp16/gemm_kernel_avx512fp16.cc")
set(_gemm_avx512bf16_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/gemm/avx512bf16/gemm_kernel_avx512bf16.cc")
set(_gemm_avx512vnni_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_avx512vnni.cc")
set(_gemm_avx2_integer_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/gemm/vnni/integer_gemm_avx2.cc")
set(_gemm_amx_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/gemm/amx/gemm_amx_tile.cc")
set(_gemm_amx_bf16_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/gemm/amx/gemm_amx_bf16.cc")
set(_gemm_amx_int8_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/gemm/amx/gemm_amx_int8.cc")
set(_gemm_avx2_fma_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/gemm/avx2/gemm_kernel_avx2_fma.cc")
set(_gemm_neon_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/gemm/arm/gemm_kernel_neon.cc")
set(_gemm_sve_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/gemm/arm/gemm_kernel_sve.cc")
set(_gemm_neon_dotprod_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/gemm/arm/gemm_kernel_neon_dotprod.cc")
set(_abs_avx512_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/avx512/abs_kernel_avx512.cc")
set(_binary_arithmetic_avx512_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/avx512/binary_arithmetic_kernel_avx512.cc")
set(_binary_arithmetic_sve_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/binary/arm/binary_arithmetic_kernel_sve.cc")
set(_exp_log_avx512_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/avx512/exp_log_kernel_avx512.cc")
set(_exp_log_avx2_fma_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/avx2/exp_log_kernel_avx2_fma.cc")
set(_not_avx512_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/logical/avx512/not_kernel_avx512.cc")
set(_compute_avx512_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/compute/avx512/compute_kernel_avx512.cc")
set(_compute_avx512fp16_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/compute/avx512fp16/compute_kernel_avx512fp16.cc")
set(_compute_avx512bf16_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/compute/avx512bf16/compute_kernel_avx512bf16.cc")
set(_compute_avx512vnni_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/compute/avx512vnni/compute_kernel_avx512vnni.cc")
set(_compute_neon_dotprod_source
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/compute/arm/compute_kernel_neon_dotprod.cc")
if(NOT ONNX_LIGHT_CPU_HAVE_AVX512F_FLAG)
  # Exclude AVX-512F translation units rather than fail on unsupported targets.
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_gemm_avx512_source}")
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_abs_avx512_source}")
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_exp_log_avx512_source}")
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_compute_avx512_source}")
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_binary_arithmetic_avx512_source}")
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_gemm_avx2_fma_source}")
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_gemm_avx2_integer_source}")
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_exp_log_avx2_fma_source}")
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX512BW_FLAG)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_not_avx512_source}")
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX512FP16_FLAG)
  # Exclude the native AVX-512FP16 GEMM kernel when the compiler cannot target
  # the ISA; the FLOAT16 path keeps its converting float32 fallback.
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_gemm_avx512fp16_source}")
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_compute_avx512fp16_source}")
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX512BF16_FLAG)
  # Exclude the native AVX-512BF16 GEMM kernel when the compiler cannot target
  # the ISA; the BFLOAT16 path keeps its converting float32 fallback.
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_gemm_avx512bf16_source}")
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_compute_avx512bf16_source}")
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AVX512VNNI_FLAG)
  # Exclude the native AVX-512 VNNI INT8 dot-product when the compiler cannot
  # target the ISA; the integer matmul keeps its AVX2/scalar fallbacks.
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_gemm_avx512vnni_source}")
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_compute_avx512vnni_source}")
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AMX_BF16_FLAG)
  # Exclude the native AMX-BF16 GEMM kernel when the compiler cannot target the
  # ISA; the BFLOAT16 path keeps its AVX-512BF16 / converting float32 fallback.
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_gemm_amx_bf16_source}")
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_AMX_INT8_FLAG)
  # Exclude the native AMX-INT8 GEMM kernel when the compiler cannot target the
  # ISA; the integer matmul keeps its AVX-512 VNNI / AVX2 / scalar fallbacks.
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_gemm_amx_int8_source}")
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_NEON_INTRINSICS)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_gemm_neon_source}")
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_SVE_INTRINSICS)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_gemm_sve_source}")
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_binary_arithmetic_sve_source}")
endif()
if(NOT ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD_INTRINSICS)
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_gemm_neon_dotprod_source}")
  list(REMOVE_ITEM CPU_KERNEL_SOURCES "${_compute_neon_dotprod_source}")
endif()

file(GLOB PY_KERNELS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/onnx_py/*.cc"
)
# The kernel-registration binding (_cpuregister) links onnx-light and is built
# as a separate module below; keep it out of the _cpukernels module, which only
# exposes the SIMD detection helpers.
list(REMOVE_ITEM PY_KERNELS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/onnx_py/_cpuregister.cc")

# ---------------------------------------------------------------------------
# C++ library: lib_onnx_light_cpu
# ---------------------------------------------------------------------------
if(ONNX_LIGHT_CPU_BUILD_PYTHON AND UNIX AND NOT APPLE)
  set(_cpu_lib_type SHARED)
else()
  set(_cpu_lib_type STATIC)
endif()

add_library(lib_onnx_light_cpu ${_cpu_lib_type} ${CPU_KERNEL_SOURCES})
target_include_directories(lib_onnx_light_cpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_property(TARGET lib_onnx_light_cpu PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(lib_onnx_light_cpu PUBLIC cxx_std_20)
if(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
  target_compile_definitions(lib_onnx_light_cpu PRIVATE ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=1)
endif()

# Thread topology detection and public affinity helpers use the platform
# threading library, but kernel scheduling is owned by onnx-light.
target_link_libraries(lib_onnx_light_cpu PUBLIC Threads::Threads)

# Apply SIMD flags to the kernel library.
if(ONNX_LIGHT_CPU_SIMD_FLAGS)
  separate_arguments(_simd_flags NATIVE_COMMAND "${ONNX_LIGHT_CPU_SIMD_FLAGS}")
  target_compile_options(lib_onnx_light_cpu PRIVATE ${_simd_flags})
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX512F_FLAG)
  if(NOT MSVC)
    set_source_files_properties("${_gemm_avx512_source}" PROPERTIES
        COMPILE_OPTIONS "-mavx512f")
    set_source_files_properties("${_abs_avx512_source}" PROPERTIES
        COMPILE_OPTIONS "-mavx512f")
    set_source_files_properties("${_exp_log_avx512_source}" PROPERTIES
        COMPILE_OPTIONS "-mavx512f")
    set_source_files_properties("${_compute_avx512_source}" PROPERTIES
        COMPILE_OPTIONS "-mavx512f")
    set_source_files_properties("${_binary_arithmetic_avx512_source}" PROPERTIES
        COMPILE_OPTIONS "-mavx512f")
  else()
    set_source_files_properties("${_gemm_avx512_source}" PROPERTIES
        COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties("${_abs_avx512_source}" PROPERTIES
        COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties("${_exp_log_avx512_source}" PROPERTIES
        COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties("${_compute_avx512_source}" PROPERTIES
        COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties("${_binary_arithmetic_avx512_source}" PROPERTIES
        COMPILE_OPTIONS "/arch:AVX512")
  endif()
  # Told gemm_kernel.cc (compiled at the baseline ONNX_LIGHT_CPU_SIMD_FLAGS)
  # that GemmMicroKernel_AVX512_F32/F64 are available to dispatch to. The
  # register-resident compute-profile FP32/FP64 kernel (Processor Profile
  # PR03) and the binary arithmetic SIMD kernels (Binary PR02) dispatch on the
  # same definition.
  target_compile_definitions(lib_onnx_light_cpu PRIVATE ONNX_LIGHT_CPU_HAVE_AVX512=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX512FP16_FLAG)
  # The native FLOAT16 GEMM kernel needs -mavx512fp16 for the _Float16 type and
  # the vcvtph2psx conversion; the rest of the library keeps the baseline flags
  # and only dispatches here once CpuSupportsAvx512Fp16() reports the ISA. The
  # register-resident compute-profile FP16 throughput kernel (Processor
  # Profile PR03) needs the same flag for native ``vfmadd*ph`` arithmetic.
  set_source_files_properties("${_gemm_avx512fp16_source}" "${_compute_avx512fp16_source}"
      PROPERTIES COMPILE_OPTIONS "-mavx512fp16")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX512FP16=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX512BF16_FLAG)
  # The native BFLOAT16 GEMM kernel needs -mavx512bf16 for the __m512bh type and
  # the vdpbf16ps dot-product (plus -mavx512bw for the 512-bit zero-extend used
  # to build the pair layout); the rest of the library keeps the baseline flags
  # and only dispatches here once CpuSupportsAvx512Bf16() reports the ISA. The
  # register-resident compute-profile BFLOAT16 dot-product kernel (Processor
  # Profile PR03) needs the same flags.
  set_source_files_properties("${_gemm_avx512bf16_source}" "${_compute_avx512bf16_source}"
      PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512bf16")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX512BF16=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG)
  if(NOT MSVC)
    set_source_files_properties("${_exp_log_avx2_fma_source}"
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  else()
    set_source_files_properties("${_exp_log_avx2_fma_source}"
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endif()
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX2_FMA=1)

  if(NOT MSVC)
    set_source_files_properties("${_gemm_avx2_integer_source}" PROPERTIES
        COMPILE_OPTIONS "-mavx2")
  else()
    set_source_files_properties("${_gemm_avx2_integer_source}" PROPERTIES
        COMPILE_OPTIONS "/arch:AVX2")
  endif()
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX512VNNI_FLAG)
  # The native INT8 VNNI dot-product needs -mavx512vnni for the vpdpbusd
  # (_mm512_dpbusd_epi32) intrinsic and -mavx512f for the ZMM load / reduce; the
  # rest of the library keeps the baseline flags and only dispatches here once
  # CpuSupportsAvx512Vnni() reports the ISA. The register-resident
  # compute-profile INT8 dot-product kernel (Processor Profile PR03) needs the
  # same flags.
  set_source_files_properties("${_gemm_avx512vnni_source}" "${_compute_avx512vnni_source}"
      PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512vnni")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX512VNNI=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AMX_TILE_FLAG)
  # The AMX tile-state lifecycle needs -mamx-tile for the LDTILECFG/TILERELEASE
  # tile intrinsics; the rest of the library keeps the baseline flags and the
  # module only touches the tiles once AmxTileStateAvailable() reports the ISA.
  set_source_files_properties("${_gemm_amx_source}" PROPERTIES
      COMPILE_OPTIONS "-mamx-tile")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AMX_TILE=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AMX_BF16_FLAG)
  # The native AMX-BF16 GEMM kernel needs -mamx-tile for the tile load/store and
  # -mamx-bf16 for the tdpbf16ps dot-product; the rest of the library keeps the
  # baseline flags and only dispatches here once CpuSupportsAmxBf16() and
  # AmxTileStateAvailable() report the ISA and OS-enabled tile state.
  set_source_files_properties("${_gemm_amx_bf16_source}" PROPERTIES
      COMPILE_OPTIONS "-mamx-tile;-mamx-bf16")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AMX_BF16=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AMX_INT8_FLAG)
  # The native AMX-INT8 GEMM kernel needs -mamx-tile for the tile load/store and
  # -mamx-int8 for the tdpbuud dot-product; dispatch additionally checks the ISA
  # and OS-enabled tile state at runtime.
  set_source_files_properties("${_gemm_amx_int8_source}" PROPERTIES
      COMPILE_OPTIONS "-mamx-tile;-mamx-int8")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AMX_INT8=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG)
  if(NOT MSVC)
    # Add -mf16c when supported so the packing loops can use the F16C
    # ``vcvtph2ps`` conversion; the AVX2/FMA micro-kernels only require -mavx2.
    if(ONNX_LIGHT_CPU_HAVE_F16C_FLAG)
      set_source_files_properties("${_gemm_avx2_fma_source}" PROPERTIES
          COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    else()
      set_source_files_properties("${_gemm_avx2_fma_source}" PROPERTIES
          COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
  else()
    set_source_files_properties("${_gemm_avx2_fma_source}" PROPERTIES
        COMPILE_OPTIONS "/arch:AVX2")
  endif()

  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX2_FMA=1)
  if(ONNX_LIGHT_CPU_HAVE_F16C_FLAG)
    target_compile_definitions(lib_onnx_light_cpu PRIVATE
        ONNX_LIGHT_CPU_HAVE_F16C=1)
  endif()
endif()

if(ONNX_LIGHT_CPU_HAVE_NEON_INTRINSICS)
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_NEON=1)
  if(ONNX_LIGHT_CPU_HAVE_NEON_FP16_INTRINSICS)
    target_compile_definitions(lib_onnx_light_cpu PRIVATE
        ONNX_LIGHT_CPU_HAVE_NEON_FP16=1)
  endif()
endif()

if(ONNX_LIGHT_CPU_HAVE_SVE_INTRINSICS)
  set_source_files_properties("${_gemm_sve_source}" "${_binary_arithmetic_sve_source}" PROPERTIES
      COMPILE_OPTIONS "-march=armv8-a+sve")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_SVE=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD_INTRINSICS)
  set_source_files_properties("${_gemm_neon_dotprod_source}" "${_compute_neon_dotprod_source}"
      PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD=1)
endif()

if(ONNX_LIGHT_CPU_HAVE_AVX512BW_FLAG)
  if(NOT MSVC)
    set_source_files_properties("${_not_avx512_source}" PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512bw")
  else()
    set_source_files_properties("${_not_avx512_source}" PROPERTIES
        COMPILE_OPTIONS "/arch:AVX512")
  endif()
  target_compile_definitions(lib_onnx_light_cpu PRIVATE
      ONNX_LIGHT_CPU_HAVE_AVX512BW=1)
endif()


if(ONNX_LIGHT_CPU_ENABLE_COVERAGE)
  target_compile_options(lib_onnx_light_cpu PUBLIC --coverage -O0 -g)
  target_link_options(lib_onnx_light_cpu PUBLIC --coverage)
endif()

# ---------------------------------------------------------------------------
# onnx-light kernel registration library: lib_onnx_light_cpu_kernels
# ---------------------------------------------------------------------------
# Wraps the SIMD kernels in onnx-light ``KernelBase`` subclasses and registers
# them into onnx-light's shared ``KernelDispatchTable`` so any ONNX model using
# ``Abs`` runs the accelerated kernel. Requires the onnx-light C++ package and
# is therefore opt-in via ONNX_LIGHT_CPU_WITH_ONNX_LIGHT.
if(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
  set(_onnx_light_imported_library_type UNKNOWN)
  if(WIN32)
    set(_onnx_light_imported_library_type SHARED)
  endif()
  if(ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR)
    if(NOT ONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY OR
       NOT ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_LIBRARY)
      message(FATAL_ERROR
          "ONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY and "
          "ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_LIBRARY are required with "
          "ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR. They must name the exact "
          "shared libraries loaded by onnx-light's Python runtime.")
    endif()
    add_library(onnx_light::lib_onnx_proto
        ${_onnx_light_imported_library_type} IMPORTED GLOBAL)
    set_target_properties(onnx_light::lib_onnx_proto PROPERTIES
        IMPORTED_LOCATION "${ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_LIBRARY}")
    add_library(onnx_light::lib_onnx_core
        ${_onnx_light_imported_library_type} IMPORTED GLOBAL)
    set_target_properties(onnx_light::lib_onnx_core PROPERTIES
        IMPORTED_LOCATION "${ONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES
            "${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/onnx_light;${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/onnx_light/onnx_helpers/include;${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/onnx_light/onnx_proto"
        INTERFACE_LINK_LIBRARIES onnx_light::lib_onnx_proto)
    if(ONNX_LIGHT_CPU_ONNX_LIGHT_IMPLIB)
      if(NOT ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_IMPLIB)
        message(FATAL_ERROR
            "ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_IMPLIB is required with "
            "ONNX_LIGHT_CPU_ONNX_LIGHT_IMPLIB.")
      endif()
      set_property(TARGET onnx_light::lib_onnx_core PROPERTY
          IMPORTED_IMPLIB "${ONNX_LIGHT_CPU_ONNX_LIGHT_IMPLIB}")
      set_property(TARGET onnx_light::lib_onnx_proto PROPERTY
          IMPORTED_IMPLIB "${ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_IMPLIB}")
    endif()
    # The backend-test registry (lib_onnx_backend_test) and its kernel
    # dependency (lib_onnx_kernels) live in the same directory as the core
    # runtime library loaded by onnx-light's Python runtime. They are not
    # reported by get_cpp_build_info(), so locate them next to the core library
    # and expose them as imported targets when present. This lets the
    # backend-test-driven C++ test link against CollectTestCases(); editable
    # builds whose onnx-light predates these libraries simply skip that test.
    get_filename_component(_onnx_light_libdir
        "${ONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY}" DIRECTORY)
    file(GLOB _onnx_light_kernels_lib
        "${_onnx_light_libdir}/liblib_onnx_kernels.*"
        "${_onnx_light_libdir}/lib_onnx_kernels.*")
    file(GLOB _onnx_light_op_lib
        "${_onnx_light_libdir}/liblib_onnx_op.*"
        "${_onnx_light_libdir}/lib_onnx_op.*")
    file(GLOB _onnx_light_backend_test_lib
        "${_onnx_light_libdir}/liblib_onnx_backend_test.*"
        "${_onnx_light_libdir}/lib_onnx_backend_test.*")
    if(_onnx_light_op_lib)
      list(GET _onnx_light_op_lib 0 _onnx_light_op_lib)
      add_library(onnx_light::lib_onnx_op
          ${_onnx_light_imported_library_type} IMPORTED GLOBAL)
      set_target_properties(onnx_light::lib_onnx_op PROPERTIES
          IMPORTED_LOCATION "${_onnx_light_op_lib}"
          INTERFACE_LINK_LIBRARIES onnx_light::lib_onnx_core)
      if(WIN32)
        if(NOT ONNX_LIGHT_CPU_ONNX_LIGHT_OP_IMPLIB)
          message(FATAL_ERROR
              "The onnx-light op Windows import library is required for "
              "source-runtime integration.")
        endif()
        set_property(TARGET onnx_light::lib_onnx_op PROPERTY
            IMPORTED_IMPLIB "${ONNX_LIGHT_CPU_ONNX_LIGHT_OP_IMPLIB}")
      endif()
    endif()
    if(_onnx_light_kernels_lib AND _onnx_light_backend_test_lib)
      list(GET _onnx_light_kernels_lib 0 _onnx_light_kernels_lib)
      list(GET _onnx_light_backend_test_lib 0 _onnx_light_backend_test_lib)
      add_library(onnx_light::lib_onnx_kernels
          ${_onnx_light_imported_library_type} IMPORTED GLOBAL)
      set_target_properties(onnx_light::lib_onnx_kernels PROPERTIES
          IMPORTED_LOCATION "${_onnx_light_kernels_lib}"
          INTERFACE_LINK_LIBRARIES onnx_light::lib_onnx_core)
      add_library(onnx_light::lib_onnx_backend_test
          ${_onnx_light_imported_library_type} IMPORTED GLOBAL)
      set_target_properties(onnx_light::lib_onnx_backend_test PROPERTIES
          IMPORTED_LOCATION "${_onnx_light_backend_test_lib}"
          INTERFACE_LINK_LIBRARIES
              "onnx_light::lib_onnx_proto;onnx_light::lib_onnx_kernels")
      if(WIN32)
        if(NOT ONNX_LIGHT_CPU_ONNX_LIGHT_KERNELS_IMPLIB OR
           NOT ONNX_LIGHT_CPU_ONNX_LIGHT_BACKEND_TEST_IMPLIB)
          message(FATAL_ERROR
              "The onnx-light kernels and backend-test Windows import "
              "libraries are required for source-runtime integration.")
        endif()
        set_property(TARGET onnx_light::lib_onnx_kernels PROPERTY
            IMPORTED_IMPLIB "${ONNX_LIGHT_CPU_ONNX_LIGHT_KERNELS_IMPLIB}")
        set_property(TARGET onnx_light::lib_onnx_backend_test PROPERTY
            IMPORTED_IMPLIB "${ONNX_LIGHT_CPU_ONNX_LIGHT_BACKEND_TEST_IMPLIB}")
      endif()
    endif()
  else()
    find_package(onnx_light CONFIG REQUIRED)
  endif()

  # When the optional onnx-light integration is enabled, ``execution.cc`` can
  # observe the current session ``CpuExecutor`` directly. Keep that dependency
  # private so standalone builds remain unchanged when the integration is off.
  target_link_libraries(lib_onnx_light_cpu PRIVATE onnx_light::lib_onnx_core)
  if(UNIX AND NOT APPLE AND ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR)
    # lib_onnx_core has no RUNPATH to the adjacent lib_onnx_proto. Retain proto
    # as a direct dependency so standalone test executables can resolve it.
    target_link_options(lib_onnx_light_cpu PRIVATE
        "LINKER:--no-as-needed")
  endif()

  file(GLOB CPU_KERNEL_REGISTRATION_SOURCES CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/*.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/*/*.cc")

  add_library(lib_onnx_light_cpu_kernels ${_cpu_lib_type}
      ${CPU_KERNEL_REGISTRATION_SOURCES})
  target_include_directories(lib_onnx_light_cpu_kernels PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  set_property(TARGET lib_onnx_light_cpu_kernels
      PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_features(lib_onnx_light_cpu_kernels PUBLIC cxx_std_20)
  target_link_libraries(lib_onnx_light_cpu_kernels PUBLIC
      lib_onnx_light_cpu
      onnx_light::lib_onnx_core)
  if(UNIX AND NOT APPLE AND ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR)
    # Keep lib_onnx_proto as a direct dependency. The local Python runtime
    # libraries have no transitive RUNPATH, so relying on lib_onnx_core alone
    # would leave its proto dependency undiscoverable outside Python.
    target_link_options(lib_onnx_light_cpu_kernels PRIVATE
        "LINKER:--no-as-needed")
  endif()

  if(ONNX_LIGHT_CPU_ENABLE_COVERAGE)
    target_compile_options(lib_onnx_light_cpu_kernels PUBLIC --coverage -O0 -g)
    target_link_options(lib_onnx_light_cpu_kernels PUBLIC --coverage)
  endif()

  # -------------------------------------------------------------------------
  # onnx-light-cpu backend-test library: lib_onnx_light_cpu_backend_test
  # -------------------------------------------------------------------------
  # Mirrors onnx-light's own ``lib_onnx_backend_test`` pattern: the backend test
  # cases live in their own library that takes a dependency on the kernels. It
  # does not run or check anything itself -- it merely *registers* the
  # onnx-light-cpu backend test cases (the ``test_cpu_*`` cases covering every
  # dtype each kernel implements) into onnx-light's shared backend test case
  # registry, so onnx-light's regular ``CollectTestCases`` (C++) and
  # ``collect_test_cases`` (Python) APIs return them. The unit tests register
  # these cases, register the kernels, and drive them through onnx-light's
  # regular runtime / ``ReferenceEvaluator`` API. Requires onnx-light's
  # ``lib_onnx_backend_test`` registry, so it is only built when that imported
  # target is available (always via find_package; best-effort in the
  # --onnx-light-source path).
  if(TARGET onnx_light::lib_onnx_backend_test)
    add_library(lib_onnx_light_cpu_backend_test ${_cpu_lib_type}
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/collect_test_cases.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/elementwise/collect_elementwise_cases.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/elementwise/cases_binary.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/math/collect_math_cases.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/math/cases_abs.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/math/cases_exp.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/math/cases_log.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/math/cases_gemm.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/math/cases_matmul.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/math/cases_matmul_integer.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/logical/collect_logical_cases.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/logical/cases_not.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/traditionalml/collect_traditionalml_cases.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/traditionalml/cases_tree_ensemble.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/attention/collect_attention_cases.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/cases/attention/cases_attention.cc")
    target_include_directories(lib_onnx_light_cpu_backend_test PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    set_property(TARGET lib_onnx_light_cpu_backend_test
        PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_compile_features(lib_onnx_light_cpu_backend_test PUBLIC cxx_std_20)
    target_link_libraries(lib_onnx_light_cpu_backend_test PUBLIC
        lib_onnx_light_cpu_kernels
        onnx_light::lib_onnx_backend_test)
    if(ONNX_LIGHT_CPU_SIMD_FLAGS)
      target_compile_options(lib_onnx_light_cpu_backend_test PRIVATE ${_simd_flags})
    endif()
    if(ONNX_LIGHT_CPU_ENABLE_COVERAGE)
      target_compile_options(lib_onnx_light_cpu_backend_test PUBLIC --coverage -O0 -g)
      target_link_options(lib_onnx_light_cpu_backend_test PUBLIC --coverage)
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# Python extension: _cpukernels
# ---------------------------------------------------------------------------
if(ONNX_LIGHT_CPU_BUILD_PYTHON)
  if(CMAKE_CROSSCOMPILING)
    find_package(Python3 3.12 REQUIRED COMPONENTS Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter)
    if(NOT TARGET Python::Module)
      add_library(Python::Module ALIAS Python3::Module)
    endif()
    if(TARGET Python3::SABIModule AND NOT TARGET Python::SABIModule)
      add_library(Python::SABIModule ALIAS Python3::SABIModule)
    endif()
  else()
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
  endif()

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
  find_package(nanobind CONFIG REQUIRED)

  if(TARGET lib_onnx_light_cpu_kernels AND _cpu_lib_type STREQUAL "SHARED"
      AND TARGET onnx_light::lib_onnx_core)
    # lib_onnx_light_cpu_kernels links onnx-light's lib_onnx_core (and
    # transitively lib_onnx_proto), which live in the installed onnx-light
    # Python package's own onnx_py directory -- a different directory tree
    # than this package's onnx_light_cpu/onnx_py. $ORIGIN (used on
    # _cpuregister/_cpukernels below) only resolves NEEDED entries relative
    # to a .so's own directory, so it cannot reach across packages; without
    # an explicit RPATH entry pointing at onnx-light's library directory,
    # importing _cpuregister fails at runtime with "liblib_onnx_core.so:
    # cannot open shared object file". Resolve that directory from the
    # onnx_light::lib_onnx_core IMPORTED target's own location via a
    # generator expression, resolved at build time so it does not depend on
    # onnx-light being importable from Python at CMake configure time (pip's
    # isolated build environment for this package does not have onnx-light
    # installed).
    set(ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR "$<TARGET_FILE_DIR:onnx_light::lib_onnx_core>")
    set_target_properties(lib_onnx_light_cpu_kernels PROPERTIES
        INSTALL_RPATH "$ORIGIN;${ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR}"
        BUILD_RPATH "$ORIGIN;${ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR}"
    )

    # The backend-test registration library, when linked into _cpuregister, is
    # installed next to it and loaded as one of its runtime dependencies. Like
    # lib_onnx_light_cpu_kernels above it directly NEEDs onnx-light's
    # lib_onnx_core/lib_onnx_proto (and lib_onnx_backend_test), so it needs the
    # same extra RPATH entry pointing at onnx-light's Python library directory.
    if(TARGET lib_onnx_light_cpu_backend_test)
      set_target_properties(lib_onnx_light_cpu_backend_test PROPERTIES
          INSTALL_RPATH "$ORIGIN;${ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR}"
          BUILD_RPATH "$ORIGIN;${ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR}"
      )
    endif()
  endif()

  nanobind_add_module(_cpukernels STABLE_ABI ${PY_KERNELS_SOURCES})
  target_include_directories(_cpukernels PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(_cpukernels PRIVATE lib_onnx_light_cpu)

  set_target_properties(_cpukernels PROPERTIES
      INSTALL_RPATH "$ORIGIN"
      BUILD_RPATH "$ORIGIN"
  )

  install(TARGETS _cpukernels
      LIBRARY DESTINATION onnx_light_cpu/onnx_py
      RUNTIME DESTINATION onnx_light_cpu/onnx_py)

  if(_cpu_lib_type STREQUAL "SHARED")
    install(TARGETS lib_onnx_light_cpu
        LIBRARY DESTINATION onnx_light_cpu/onnx_py
        RUNTIME DESTINATION onnx_light_cpu/onnx_py)
  endif()

  # -------------------------------------------------------------------------
  # Kernel-registration Python extension: _cpuregister
  # -------------------------------------------------------------------------
  # Exposes ``register_all_kernels()``, which installs the onnx-light ``Abs``,
  # ``Exp``, ``Log`` and ``Not`` kernel classes into onnx-light's shared
  # ``KernelDispatchTable``. It links the onnx-light kernel-registration
  # library and is therefore only built when the integration is enabled.
  if(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
    nanobind_add_module(_cpuregister STABLE_ABI
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/onnx_py/_cpuregister.cc")
    target_include_directories(_cpuregister PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(_cpuregister PRIVATE lib_onnx_light_cpu_kernels)

    # When onnx-light's backend test registry is available, link the
    # onnx-light-cpu backend-test *registration* library so ``_cpuregister``
    # can expose ``register_backend_test_cases()`` -- which registers the
    # ``test_cpu_*`` cases into that shared registry so Python's
    # ``collect_test_cases`` returns them alongside onnx-light's own cases.
    if(TARGET lib_onnx_light_cpu_backend_test)
      target_link_libraries(_cpuregister PRIVATE lib_onnx_light_cpu_backend_test)
      target_compile_definitions(_cpuregister PRIVATE ONNX_LIGHT_CPU_HAS_BACKEND_TEST)
    endif()

    # _cpuregister also directly NEEDs onnx-light's lib_onnx_core/lib_onnx_proto
    # (transitively, via lib_onnx_light_cpu_kernels' PUBLIC link), so it needs
    # the same extra RPATH entry as lib_onnx_light_cpu_kernels above -- see the
    # comment near ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR for why $ORIGIN alone is
    # not enough.
    if(ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR)
      set_target_properties(_cpuregister PROPERTIES
          INSTALL_RPATH "$ORIGIN;${ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR}"
          BUILD_RPATH "$ORIGIN;${ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR}"
      )
    else()
      set_target_properties(_cpuregister PROPERTIES
          INSTALL_RPATH "$ORIGIN"
          BUILD_RPATH "$ORIGIN"
      )
    endif()

    install(TARGETS _cpuregister
        LIBRARY DESTINATION onnx_light_cpu/onnx_py
        RUNTIME DESTINATION onnx_light_cpu/onnx_py)

    # _cpuregister is built with INSTALL_RPATH "$ORIGIN" and therefore looks
    # for lib_onnx_light_cpu_kernels.so next to itself in onnx_light_cpu/onnx_py
    # at runtime. That is independent of ONNX_LIGHT_CPU_INSTALL (the C++
    # library/headers install step, which is disabled entirely when onnx-light
    # is built from a local source tree -- see the comment near the top of
    # this file), so install the shared library into onnx_py unconditionally
    # here rather than only inside the ONNX_LIGHT_CPU_INSTALL block below.
    # Without this, register_kernels() fails with an ImportError at runtime
    # whenever ONNX_LIGHT_CPU_INSTALL is off.
    if(_cpu_lib_type STREQUAL "SHARED")
      install(TARGETS lib_onnx_light_cpu_kernels
          LIBRARY DESTINATION onnx_light_cpu/onnx_py
          RUNTIME DESTINATION onnx_light_cpu/onnx_py)

      # When linked into _cpuregister, the backend-test registration library is
      # a runtime dependency too, so it must sit next to _cpuregister (found via
      # its "$ORIGIN" RPATH) just like lib_onnx_light_cpu_kernels above.
      if(TARGET lib_onnx_light_cpu_backend_test)
        install(TARGETS lib_onnx_light_cpu_backend_test
            LIBRARY DESTINATION onnx_light_cpu/onnx_py
            RUNTIME DESTINATION onnx_light_cpu/onnx_py)
      endif()
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# C++ install
# ---------------------------------------------------------------------------
if(ONNX_LIGHT_CPU_INSTALL)
  install(TARGETS lib_onnx_light_cpu
      EXPORT onnx_light_cpuTargets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/simd_level.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/impl")

  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/execution.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/thread_topology.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/cpu_cache_topology.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/memory_traffic_profile.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/compute_arithmetic_profile.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/impl")

  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/math_kernels.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/impl/math")

  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/gemm/gemm_plan.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/impl/math/gemm")

  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/logical/logical_kernels.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/impl/logical")

  if(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
    install(TARGETS lib_onnx_light_cpu_kernels
        EXPORT onnx_light_cpuTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/kernel_registration.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/register_kernels.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/kernels")

    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/math/abs_kernel.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/math/exp_log_kernel.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/math/gemm_kernel.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/math/matmul_kernel.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/kernels/math")

    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/logical/not_kernel.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/kernels/logical")
  endif()

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_light_cpuConfigVersion.cmake"
      COMPATIBILITY SameMajorVersion)

  configure_package_config_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnx_light_cpuConfig.cmake.in"
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_light_cpuConfig.cmake"
      INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_light_cpu")

  install(EXPORT onnx_light_cpuTargets
      NAMESPACE onnx_light_cpu::
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_light_cpu")

  install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_light_cpuConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_light_cpuConfigVersion.cmake"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_light_cpu")
endif()

# ---------------------------------------------------------------------------
# Isolated C++ throughput benchmarks (opt-in, not run by ctest)
# ---------------------------------------------------------------------------
if(ONNX_LIGHT_CPU_BUILD_BENCHMARKS)
  add_executable(gemm_throughput
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/gemm_throughput.cc")
  target_link_libraries(gemm_throughput PRIVATE lib_onnx_light_cpu)
  if(ONNX_LIGHT_CPU_SIMD_FLAGS)
    target_compile_options(gemm_throughput PRIVATE ${_simd_flags})
  endif()
  onnx_light_apply_hardening(gemm_throughput)

  add_executable(compact_gemm_throughput
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/compact_gemm_throughput.cc")
  target_link_libraries(compact_gemm_throughput PRIVATE lib_onnx_light_cpu)
  if(ONNX_LIGHT_CPU_SIMD_FLAGS)
    target_compile_options(compact_gemm_throughput PRIVATE ${_simd_flags})
  endif()
  onnx_light_apply_hardening(compact_gemm_throughput)

  add_executable(exp_log_throughput
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/exp_log_throughput.cc")
  target_link_libraries(exp_log_throughput PRIVATE lib_onnx_light_cpu)
  if(ONNX_LIGHT_CPU_SIMD_FLAGS)
    target_compile_options(exp_log_throughput PRIVATE ${_simd_flags})
  endif()
  onnx_light_apply_hardening(exp_log_throughput)

  add_executable(tree_ensemble_throughput
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/tree_ensemble_throughput.cc")
  target_link_libraries(tree_ensemble_throughput PRIVATE lib_onnx_light_cpu)
  onnx_light_apply_hardening(tree_ensemble_throughput)
endif()

# ---------------------------------------------------------------------------
# C++ unit tests (GTest)
# ---------------------------------------------------------------------------
if(ONNX_LIGHT_CPU_BUILD_TESTS)
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  # Do not add GoogleTest's own install rules to this project. Otherwise a
  # `cmake --install` into the inplace prefix (e.g. `setup.py build_ext
  # --inplace --cpp-tests`) would drop libgtest.a/libgmock.a, the gtest/gmock
  # headers, their pkg-config files and CMake package config into the source
  # tree next to the onnx_light_cpu Python package, instead of installing only
  # this project's artifacts.
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  include(GoogleTest)

  # Declare a GTest executable from a single source file, linking it against the
  # requested kernel library. Used by the globbing loop below so new test files
  # are picked up without editing this list.
  function(onnx_light_cpu_add_test _src _link_lib)
    get_filename_component(_name "${_src}" NAME_WE)
    add_executable(${_name} "${_src}")
    target_link_libraries(${_name} PRIVATE ${_link_lib} GTest::gtest_main)
    target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    if(_name STREQUAL "test_integer_gemm_vnni")
      # This differential binary names and calls the optional native integer
      # kernels directly. The library keeps availability definitions private,
      # so mirror only the definitions whose translation units were built.
      if(ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG)
        target_compile_definitions(${_name} PRIVATE ONNX_LIGHT_CPU_HAVE_AVX2_INTEGER=1)
      endif()
      if(ONNX_LIGHT_CPU_HAVE_AVX512VNNI_FLAG)
        target_compile_definitions(${_name} PRIVATE ONNX_LIGHT_CPU_HAVE_AVX512VNNI=1)
      endif()
      if(ONNX_LIGHT_CPU_HAVE_AMX_INT8_FLAG)
        target_compile_definitions(${_name} PRIVATE ONNX_LIGHT_CPU_HAVE_AMX_INT8=1)
      endif()
      if(ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD_INTRINSICS)
        target_compile_definitions(${_name} PRIVATE ONNX_LIGHT_CPU_HAVE_NEON_DOTPROD=1)
      endif()
    endif()
    if(_name STREQUAL "test_gemm_kernel" AND ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG)
      # This differential test calls optional AVX2 conversion helpers directly.
      # Mirror the private library definitions only when those helpers were built.
      target_compile_definitions(${_name} PRIVATE ONNX_LIGHT_CPU_HAVE_AVX2_FMA=1)
      if(ONNX_LIGHT_CPU_HAVE_F16C_FLAG)
        target_compile_definitions(${_name} PRIVATE ONNX_LIGHT_CPU_HAVE_F16C=1)
      endif()
    endif()
    if(_name STREQUAL "test_exp_log_kernel" AND ONNX_LIGHT_CPU_HAVE_AVX2_FMA_FLAG)
      target_compile_definitions(${_name} PRIVATE ONNX_LIGHT_CPU_HAVE_AVX2_FMA=1)
    endif()
    if(_name STREQUAL "test_exp_log_kernel" AND ONNX_LIGHT_CPU_HAVE_AVX512F_FLAG)
      target_compile_definitions(${_name} PRIVATE ONNX_LIGHT_CPU_HAVE_AVX512=1)
    endif()
    if("lib_onnx_light_cpu_kernels" IN_LIST _link_lib AND
       ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR)
      set_target_properties(${_name} PROPERTIES
          BUILD_RPATH "${ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR}")
    endif()
    # Apply SIMD flags to the test target too so it can use intrinsics if needed.
    if(ONNX_LIGHT_CPU_SIMD_FLAGS)
      target_compile_options(${_name} PRIVATE ${_simd_flags})
    endif()
    # Discover the test cases when ctest runs instead of at build time. The
    # POST_BUILD discovery run executes the freshly linked binary, which is
    # flaky on macOS (a just-written, ad-hoc signed executable is sometimes
    # killed by the kernel), and a failed discovery run aborts the whole build.
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
      gtest_discover_tests(${_name} DISCOVERY_MODE PRE_TEST)
    else()
      gtest_discover_tests(${_name})
    endif()
    onnx_light_apply_hardening(${_name})
  endfunction()

  file(GLOB _test_sources CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/test_*.cc")
  foreach(_test_src IN LISTS _test_sources)
    get_filename_component(_test_name "${_test_src}" NAME_WE)
    if(_test_name MATCHES "^test_onnx_light_")
      # onnx-light kernel-class tests link the registration library against
      # onnx-light and are only built when the integration is enabled.
      if(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
        if(_test_name STREQUAL "test_onnx_light_backend_kernels")
          # Registers the onnx-light-cpu backend test cases (via the dedicated
          # registration library lib_onnx_light_cpu_backend_test, which depends
          # on the kernels) and the kernels, then drives the collected
          # ``test_cpu_*`` cases through onnx-light's regular runtime API. Built
          # only when that library is available (it requires onnx-light's
          # lib_onnx_backend_test registry).
          if(TARGET lib_onnx_light_cpu_backend_test)
            onnx_light_cpu_add_test("${_test_src}"
                "lib_onnx_light_cpu_kernels;lib_onnx_light_cpu_backend_test")
          endif()
        elseif(_test_name STREQUAL "test_onnx_light_binary_manifest")
          if(TARGET onnx_light::lib_onnx_op)
            onnx_light_cpu_add_test("${_test_src}"
                "lib_onnx_light_cpu_kernels;onnx_light::lib_onnx_op")
          endif()
        else()
          onnx_light_cpu_add_test("${_test_src}" lib_onnx_light_cpu_kernels)
        endif()
      endif()
    else()
      onnx_light_cpu_add_test("${_test_src}" lib_onnx_light_cpu)
    endif()
  endforeach()
endif()

# ---------------------------------------------------------------------------
# Apply OpenSSF compiler hardening flags to the non-test targets when
# ONNX_HARDENING=ON. When the option is OFF the helper is a no-op. Test targets
# are hardened as they are declared in onnx_light_cpu_add_test above.
# ---------------------------------------------------------------------------
set(_onnx_light_cpu_hardening_targets lib_onnx_light_cpu)
if(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
  list(APPEND _onnx_light_cpu_hardening_targets lib_onnx_light_cpu_kernels)
endif()
if(ONNX_LIGHT_CPU_BUILD_PYTHON)
  list(APPEND _onnx_light_cpu_hardening_targets _cpukernels)
  if(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
    list(APPEND _onnx_light_cpu_hardening_targets _cpuregister)
  endif()
endif()
foreach(_t IN LISTS _onnx_light_cpu_hardening_targets)
  if(TARGET ${_t})
    onnx_light_apply_hardening(${_t})
  endif()
endforeach()
