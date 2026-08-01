cmake_minimum_required(VERSION 3.15)

# Default to a Release build when no build type is specified.
get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _is_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

project(onnx_light_cpu VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(ONNX_LIGHT_CPU_BUILD_TESTS "Build C++ unit tests." OFF)
option(ONNX_LIGHT_CPU_BUILD_PYTHON "Build Python extension." ON)
option(ONNX_LIGHT_CPU_INSTALL "Install the C++ library and headers." ON)
option(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT
       "Build the onnx-light kernel registration library (requires onnx-light)." OFF)
option(ONNX_LIGHT_CPU_ENABLE_COVERAGE
       "Instrument the C++ targets for code coverage (GCC/Clang only)." OFF)
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
# On x86 platforms, enable AVX2 by default for the kernel sources so the
# compiler can emit AVX/AVX2 intrinsics. The runtime dispatch still falls
# back to SSE2/scalar on older CPUs. AVX-512 codepaths are guarded by
# __AVX512F__ and only compiled when the user explicitly passes -mavx512f.
if(NOT MSVC)
  set(ONNX_LIGHT_CPU_SIMD_FLAGS "-mavx2" CACHE STRING
      "SIMD compiler flags for kernel sources (e.g. -mavx2, -mavx512f).")
else()
  set(ONNX_LIGHT_CPU_SIMD_FLAGS "/arch:AVX2" CACHE STRING
      "SIMD compiler flags for kernel sources (e.g. /arch:AVX2, /arch:AVX512).")
endif()

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
set(CPU_KERNEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/simd_level.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/abs_kernel.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/exp_log_kernel.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/logical/not_kernel.cc"
)

set(PY_KERNELS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/onnx_py/_cpupy_math_kernels.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/onnx_py/_cpupy_logical_kernels.cc"
)

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

# Apply SIMD flags to the kernel library.
if(ONNX_LIGHT_CPU_SIMD_FLAGS)
  separate_arguments(_simd_flags NATIVE_COMMAND "${ONNX_LIGHT_CPU_SIMD_FLAGS}")
  target_compile_options(lib_onnx_light_cpu PRIVATE ${_simd_flags})
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
  find_package(onnx_light CONFIG REQUIRED)

  add_library(lib_onnx_light_cpu_kernels ${_cpu_lib_type}
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/math/abs_kernel.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/math/exp_log_kernel.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/logical/not_kernel.cc")
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

  if(ONNX_LIGHT_CPU_ENABLE_COVERAGE)
    target_compile_options(lib_onnx_light_cpu_kernels PUBLIC --coverage -O0 -g)
    target_link_options(lib_onnx_light_cpu_kernels PUBLIC --coverage)
  endif()
endif()

# ---------------------------------------------------------------------------
# Python extension: _cpukernels
# ---------------------------------------------------------------------------
if(ONNX_LIGHT_CPU_BUILD_PYTHON)
  if(CMAKE_CROSSCOMPILING)
    find_package(Python3 3.10 REQUIRED COMPONENTS Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)
    if(NOT TARGET Python::Module)
      add_library(Python::Module ALIAS Python3::Module)
    endif()
    if(TARGET Python3::SABIModule AND NOT TARGET Python::SABIModule)
      add_library(Python::SABIModule ALIAS Python3::SABIModule)
    endif()
  else()
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
  endif()

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
  find_package(nanobind CONFIG REQUIRED)

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
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/simd_level.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/kernels")

  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/math/math_kernels.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/kernels/math")

  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/logical/logical_kernels.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/kernels/logical")

  if(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
    install(TARGETS lib_onnx_light_cpu_kernels
        EXPORT onnx_light_cpuTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/math/abs_kernel.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/math/exp_log_kernel.h"
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
  FetchContent_MakeAvailable(googletest)

  add_executable(test_abs_kernel
      "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/test_abs_kernel.cc")
  target_link_libraries(test_abs_kernel PRIVATE
      lib_onnx_light_cpu
      GTest::gtest_main)
  target_include_directories(test_abs_kernel PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")

  # Apply SIMD flags to the test target too so it can use intrinsics if needed.
  if(ONNX_LIGHT_CPU_SIMD_FLAGS)
    target_compile_options(test_abs_kernel PRIVATE ${_simd_flags})
  endif()

  include(GoogleTest)
  gtest_discover_tests(test_abs_kernel)

  add_executable(test_exp_log_kernel
      "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/test_exp_log_kernel.cc")
  target_link_libraries(test_exp_log_kernel PRIVATE
      lib_onnx_light_cpu
      GTest::gtest_main)
  target_include_directories(test_exp_log_kernel PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
  if(ONNX_LIGHT_CPU_SIMD_FLAGS)
    target_compile_options(test_exp_log_kernel PRIVATE ${_simd_flags})
  endif()
  gtest_discover_tests(test_exp_log_kernel)

  add_executable(test_not_kernel
      "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/test_not_kernel.cc")
  target_link_libraries(test_not_kernel PRIVATE
      lib_onnx_light_cpu
      GTest::gtest_main)
  target_include_directories(test_not_kernel PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
  if(ONNX_LIGHT_CPU_SIMD_FLAGS)
    target_compile_options(test_not_kernel PRIVATE ${_simd_flags})
  endif()
  gtest_discover_tests(test_not_kernel)

  # onnx-light kernel-class test: only built when the onnx-light integration
  # is enabled (it links the registration library against onnx-light).
  if(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
    add_executable(test_onnx_light_abs_kernel
        "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/test_onnx_light_abs_kernel.cc")
    target_link_libraries(test_onnx_light_abs_kernel PRIVATE
        lib_onnx_light_cpu_kernels
        GTest::gtest_main)
    target_include_directories(test_onnx_light_abs_kernel PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")
    if(ONNX_LIGHT_CPU_SIMD_FLAGS)
      target_compile_options(test_onnx_light_abs_kernel PRIVATE ${_simd_flags})
    endif()
    gtest_discover_tests(test_onnx_light_abs_kernel)

    add_executable(test_onnx_light_exp_log_kernel
        "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/test_onnx_light_exp_log_kernel.cc")
    target_link_libraries(test_onnx_light_exp_log_kernel PRIVATE
        lib_onnx_light_cpu_kernels
        GTest::gtest_main)
    target_include_directories(test_onnx_light_exp_log_kernel PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")
    if(ONNX_LIGHT_CPU_SIMD_FLAGS)
      target_compile_options(test_onnx_light_exp_log_kernel PRIVATE ${_simd_flags})
    endif()
    gtest_discover_tests(test_onnx_light_exp_log_kernel)

    add_executable(test_onnx_light_not_kernel
        "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/test_onnx_light_not_kernel.cc")
    target_link_libraries(test_onnx_light_not_kernel PRIVATE
        lib_onnx_light_cpu_kernels
        GTest::gtest_main)
    target_include_directories(test_onnx_light_not_kernel PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")
    if(ONNX_LIGHT_CPU_SIMD_FLAGS)
      target_compile_options(test_onnx_light_not_kernel PRIVATE ${_simd_flags})
    endif()
    gtest_discover_tests(test_onnx_light_not_kernel)
  endif()
endif()

# ---------------------------------------------------------------------------
# Apply OpenSSF compiler hardening flags to this project's targets when
# ONNX_HARDENING=ON. When the option is OFF the helper is a no-op.
# ---------------------------------------------------------------------------
set(_onnx_light_cpu_hardening_targets lib_onnx_light_cpu)
if(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
  list(APPEND _onnx_light_cpu_hardening_targets lib_onnx_light_cpu_kernels)
endif()
if(ONNX_LIGHT_CPU_BUILD_PYTHON)
  list(APPEND _onnx_light_cpu_hardening_targets _cpukernels)
endif()
if(ONNX_LIGHT_CPU_BUILD_TESTS)
  list(APPEND _onnx_light_cpu_hardening_targets test_abs_kernel)
  list(APPEND _onnx_light_cpu_hardening_targets test_exp_log_kernel)
  list(APPEND _onnx_light_cpu_hardening_targets test_not_kernel)
  if(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
    list(APPEND _onnx_light_cpu_hardening_targets test_onnx_light_abs_kernel)
    list(APPEND _onnx_light_cpu_hardening_targets test_onnx_light_exp_log_kernel)
    list(APPEND _onnx_light_cpu_hardening_targets test_onnx_light_not_kernel)
  endif()
endif()
foreach(_t IN LISTS _onnx_light_cpu_hardening_targets)
  if(TARGET ${_t})
    onnx_light_apply_hardening(${_t})
  endif()
endforeach()
