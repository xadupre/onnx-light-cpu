cmake_minimum_required(VERSION 3.15)

# Default to a Release build when no build type is specified.
get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _is_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

project(onnx_light_cpu VERSION 0.1.11 LANGUAGES CXX)

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
option(ONNX_LIGHT_CPU_BUILD_PYTHON "Build Python extension." ON)
option(ONNX_LIGHT_CPU_INSTALL "Install the C++ library and headers." ON)
option(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT
       "Build the onnx-light kernel registration library (requires onnx-light)." OFF)
set(ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR "" CACHE PATH
    "Path to a local onnx-light source tree. When set (and \
ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON), onnx-light is built from these sources with \
add_subdirectory instead of being located with find_package(onnx_light). Use \
this when the same sources were used to build the installed onnx-light Python \
package but its onnx_lightConfig.cmake is not available.")
option(ONNX_LIGHT_CPU_ENABLE_COVERAGE
       "Instrument the C++ targets for code coverage (GCC/Clang only)." OFF)

# When onnx-light is built from a local source tree (add_subdirectory with
# ONNX_LIGHT_INSTALL=OFF), its targets are never added to any export set, so
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

# Threads: required by the header-only ParallelFor helper (std::thread).
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
# Discover the kernel and binding sources with globbing so new files are picked
# up without editing this list. CONFIGURE_DEPENDS re-runs the glob when the set
# of matching files changes.
file(GLOB CPU_KERNEL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/*/*.cc"
)

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

# The kernels dispatch large ranges through the header-only ParallelFor helper,
# which uses std::thread; link the platform threading library.
target_link_libraries(lib_onnx_light_cpu PUBLIC Threads::Threads)

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
  if(ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR)
    # Build onnx-light from a local source tree instead of locating a
    # previously installed onnx-light with find_package. This assumes the same
    # sources were used to build the onnx-light Python package that is used at
    # runtime. Only the C++ library is needed here, so disable onnx-light's own
    # Python extensions, tests, benchmarks and install rules. These are set as
    # plain (non-cache) variables so onnx-light's option() calls honour them
    # (policy CMP0077).
    set(ONNX_LIGHT_BUILD_PYTHON OFF)
    set(ONNX_LIGHT_BUILD_TESTS OFF)
    set(ONNX_LIGHT_BUILD_BENCHMARKS OFF)
    set(ONNX_LIGHT_INSTALL OFF)
    add_subdirectory("${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/_onnx_light_src" EXCLUDE_FROM_ALL)
    # find_package(onnx_light) exposes the namespaced onnx_light::lib_onnx_core
    # target; provide the same alias for the in-tree target so the code below is
    # identical for both build modes.
    if(NOT TARGET onnx_light::lib_onnx_core)
      add_library(onnx_light::lib_onnx_core ALIAS lib_onnx_core)
    endif()
  else()
    find_package(onnx_light CONFIG REQUIRED)
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

    set_target_properties(_cpuregister PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_RPATH "$ORIGIN"
    )

    install(TARGETS _cpuregister
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
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/simd_level.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/impl")

  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/parallel_for.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/impl")

  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/impl/math/math_kernels.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/impl/math")

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
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/register_kernels.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu/kernels")

    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/math/abs_kernel.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/math/exp_log_kernel.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/math/gemm_kernel.h"
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

  include(GoogleTest)

  # Declare a GTest executable from a single source file, linking it against the
  # requested kernel library. Used by the globbing loop below so new test files
  # are picked up without editing this list.
  function(onnx_light_cpu_add_test _src _link_lib)
    get_filename_component(_name "${_src}" NAME_WE)
    add_executable(${_name} "${_src}")
    target_link_libraries(${_name} PRIVATE ${_link_lib} GTest::gtest_main)
    target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    # Apply SIMD flags to the test target too so it can use intrinsics if needed.
    if(ONNX_LIGHT_CPU_SIMD_FLAGS)
      target_compile_options(${_name} PRIVATE ${_simd_flags})
    endif()
    gtest_discover_tests(${_name})
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
        onnx_light_cpu_add_test("${_test_src}" lib_onnx_light_cpu_kernels)
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
