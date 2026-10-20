cmake_minimum_required(VERSION 3.15)

# Default to a Release build when no build type is specified.
get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _is_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

project(onnx_light_cpu VERSION 0.2.0 LANGUAGES CXX)

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
option(ONNX_LIGHT_CPU_PYTHON_STABLE_ABI
       "Build Python extensions with CPython's stable ABI." ON)
option(ONNX_LIGHT_CPU_INSTALL "Install the C++ library and headers." ON)
option(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT
       "Build the onnx-light kernel registration library (requires onnx-light)." OFF)
option(ONNX_LIGHT_CPU_RELEASE_WHEEL
       "Link release wheels to the sibling installed onnx-light runtime." OFF)
set(ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR "" CACHE PATH
    "Path to a local onnx-light source tree whose headers match the already-built \
Python runtime.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY "" CACHE FILEPATH
    "Exact onnx-light core shared library loaded by the Python runtime.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_LIBRARY "" CACHE FILEPATH
    "Exact onnx-light proto shared library loaded by the Python runtime.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_THREADS_LIBRARY "" CACHE FILEPATH
    "Exact onnx-light threading shared library loaded by the Python runtime.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_KERNELS_LIBRARY "" CACHE FILEPATH
    "Exact onnx-light kernels shared library (Cast compatibility dependency).")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_IMPLIB "" CACHE FILEPATH
    "Windows import library for ONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_IMPLIB "" CACHE FILEPATH
    "Windows import library for ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_LIBRARY.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_THREADS_IMPLIB "" CACHE FILEPATH
    "Windows import library for ONNX_LIGHT_CPU_ONNX_LIGHT_THREADS_LIBRARY.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_KERNELS_IMPLIB "" CACHE FILEPATH
    "Windows import library for the onnx-light kernels runtime.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_BACKEND_TEST_IMPLIB "" CACHE FILEPATH
    "Windows import library for the onnx-light backend-test runtime.")
set(ONNX_LIGHT_CPU_ONNX_LIGHT_SHAPE_IMPLIB "" CACHE FILEPATH
    "Windows import library for the onnx-light shape-inference runtime.")
if(ONNX_LIGHT_CPU_RELEASE_WHEEL)
  set(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT ON)
  get_filename_component(ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/../onnx-light" ABSOLUTE)
  find_package(Python COMPONENTS Interpreter REQUIRED)
  execute_process(
      COMMAND "${Python_EXECUTABLE}" -c
          "from onnx_light import get_cpp_build_info; info = get_cpp_build_info(); print(info['core_library']); print(info['proto_library'])"
      OUTPUT_VARIABLE _onnx_light_runtime_libraries
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _onnx_light_runtime_status)
  if(NOT _onnx_light_runtime_status EQUAL 0)
    message(FATAL_ERROR "Could not locate the installed onnx-light runtime.")
  endif()
  string(REPLACE "\n" ";" _onnx_light_runtime_libraries
      "${_onnx_light_runtime_libraries}")
  list(GET _onnx_light_runtime_libraries 0 ONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY)
  list(GET _onnx_light_runtime_libraries 1 ONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_LIBRARY)
  if(WIN32)
    set(_onnx_light_components CORE PROTO KERNELS BACKEND_TEST OP SHAPE)
    if(EXISTS "${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/onnx_threads/include/onnx_threads/config.h")
      list(APPEND _onnx_light_components THREADS)
    endif()
    foreach(_component ${_onnx_light_components})
      string(TOLOWER "${_component}" _name)
      file(GLOB_RECURSE _import_lib
          "${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/build/lib_onnx_${_name}.lib")
      list(GET _import_lib 0 _import_lib_path)
      if(_component STREQUAL "CORE")
        set(ONNX_LIGHT_CPU_ONNX_LIGHT_IMPLIB "${_import_lib_path}")
      else()
        set(ONNX_LIGHT_CPU_ONNX_LIGHT_${_component}_IMPLIB
            "${_import_lib_path}")
      endif()
    endforeach()
  endif()
endif()
option(ONNX_LIGHT_CPU_ENABLE_COVERAGE
       "Instrument the C++ targets for code coverage (GCC/Clang only)." OFF)
if(DEFINED ONNX_LIGHT_CPU_MAX_SIMD_LEVEL)
  set(ML_CPU_COMPUTE_MAX_SIMD_LEVEL "${ONNX_LIGHT_CPU_MAX_SIMD_LEVEL}"
      CACHE STRING "Maximum SIMD level forwarded by the runtime project." FORCE)
endif()
if(ONNX_LIGHT_CPU_SIMD_FLAGS)
  message(FATAL_ERROR
      "ONNX_LIGHT_CPU_SIMD_FLAGS is no longer supported. Use ML_CPU_COMPUTE_MAX_SIMD_LEVEL.")
endif()
if(ONNX_LIGHT_CPU_ENABLE_COVERAGE)
  set(ML_CPU_COMPUTE_ENABLE_COVERAGE ON)
endif()
include(ml_cpu_compute/cmake/CpuFeatures.cmake)

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
See ml_cpu_compute/cmake/Hardening.cmake and \
https://best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++.html"
    OFF)
if(ONNX_HARDENING)
  set(ML_CPU_COMPUTE_HARDENING ON)
endif()

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
if(ML_CPU_COMPUTE_HARDENING)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ml_cpu_compute/cmake")
  include(Hardening)
else()
  function(ml_cpu_compute_apply_hardening target)
  endfunction()
endif()

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
# Discover the kernel and binding sources with globbing so new files are picked
# up without editing this list. CONFIGURE_DEPENDS re-runs the glob when the set
# of matching files changes.
file(GLOB_RECURSE CPU_BACKEND_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/backend_test/*.cc")
set(CPU_BACKEND_CORPUS_SOURCES ${CPU_BACKEND_TEST_SOURCES})
list(FILTER CPU_BACKEND_CORPUS_SOURCES INCLUDE REGEX "_corpus[.]cc$")

file(GLOB PY_KERNELS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/onnx_py/*.cc"
)
file(GLOB PY_REGISTER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/onnx_py/_cpuregister*.cc"
)
# Registration sources link onnx-light and form a separate extension below;
# keep the whole family out of the standalone _cpukernels module.
list(REMOVE_ITEM PY_KERNELS_SOURCES ${PY_REGISTER_SOURCES})

# ---------------------------------------------------------------------------
# C++ library: ml_cpu_compute
# ---------------------------------------------------------------------------
if(ONNX_LIGHT_CPU_BUILD_PYTHON AND UNIX AND NOT APPLE)
  set(_cpu_lib_type SHARED)
else()
  set(_cpu_lib_type STATIC)
endif()

if(ONNX_LIGHT_CPU_BUILD_TESTS)
  set(ML_CPU_COMPUTE_BUILD_TESTS ON)
endif()
add_subdirectory(ml_cpu_compute)
add_library(lib_onnx_light_cpu ALIAS ml_cpu_compute)

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
    set(_onnx_light_include_dirs
        "${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/onnx_light")
    if(EXISTS "${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/onnx_proto/include/onnx_proto/onnx.h")
      list(APPEND _onnx_light_include_dirs
          "${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/onnx_proto/include"
          "${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/onnx_proto/include/onnx_proto"
          "${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/onnx_proto/helpers/include")
    else()
      list(APPEND _onnx_light_include_dirs
          "${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/onnx_light/onnx_helpers/include"
          "${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/onnx_light/onnx_proto")
    endif()
    if(ONNX_LIGHT_CPU_ONNX_LIGHT_THREADS_LIBRARY OR
       EXISTS "${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/onnx_threads/include/onnx_threads/config.h")
      get_filename_component(_onnx_light_libdir
          "${ONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY}" DIRECTORY)
      if(ONNX_LIGHT_CPU_ONNX_LIGHT_THREADS_LIBRARY)
        set(_onnx_light_threads_lib "${ONNX_LIGHT_CPU_ONNX_LIGHT_THREADS_LIBRARY}")
      else()
        file(GLOB _onnx_light_threads_lib
            "${_onnx_light_libdir}/liblib_onnx_threads${CMAKE_SHARED_LIBRARY_SUFFIX}"
            "${_onnx_light_libdir}/lib_onnx_threads${CMAKE_SHARED_LIBRARY_SUFFIX}")
      endif()
      if(NOT _onnx_light_threads_lib)
        message(FATAL_ERROR
            "The onnx-light threading shared library is required by this source tree. "
            "Set ONNX_LIGHT_CPU_ONNX_LIGHT_THREADS_LIBRARY to the library loaded by Python.")
      endif()
      list(GET _onnx_light_threads_lib 0 _onnx_light_threads_lib)
      add_library(onnx_threads::lib_onnx_threads
          ${_onnx_light_imported_library_type} IMPORTED GLOBAL)
      set_target_properties(onnx_threads::lib_onnx_threads PROPERTIES
          IMPORTED_LOCATION "${_onnx_light_threads_lib}"
          INTERFACE_COMPILE_DEFINITIONS ONNX_THREADS_SHARED
          INTERFACE_LINK_LIBRARIES Threads::Threads)
      if(EXISTS "${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/onnx_threads/include/onnx_threads/config.h")
        set_property(TARGET onnx_threads::lib_onnx_threads PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES
                "${ONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR}/onnx_threads/include")
      endif()
      if(WIN32)
        if(NOT ONNX_LIGHT_CPU_ONNX_LIGHT_THREADS_IMPLIB)
          message(FATAL_ERROR
              "ONNX_LIGHT_CPU_ONNX_LIGHT_THREADS_IMPLIB is required for "
              "source-runtime integration on Windows.")
        endif()
        set_property(TARGET onnx_threads::lib_onnx_threads PROPERTY
            IMPORTED_IMPLIB "${ONNX_LIGHT_CPU_ONNX_LIGHT_THREADS_IMPLIB}")
      endif()
      target_link_libraries(onnx_light::lib_onnx_proto INTERFACE
          onnx_threads::lib_onnx_threads)
    endif()
    add_library(onnx_light::lib_onnx_core
        ${_onnx_light_imported_library_type} IMPORTED GLOBAL)
    set_target_properties(onnx_light::lib_onnx_core PROPERTIES
        IMPORTED_LOCATION "${ONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES
            "${_onnx_light_include_dirs}"
        INTERFACE_LINK_LIBRARIES onnx_light::lib_onnx_proto)
    if(TARGET onnx_threads::lib_onnx_threads)
      target_link_libraries(onnx_light::lib_onnx_core INTERFACE
          onnx_threads::lib_onnx_threads)
    endif()
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
    # Cast's STRING/float8/sub-byte compatibility path calls the concrete
    # builtin Cast. Import its kernels library even without backend tests.
    # The backend-test registry remains optional.
    get_filename_component(_onnx_light_libdir
        "${ONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY}" DIRECTORY)
    if(ONNX_LIGHT_CPU_ONNX_LIGHT_KERNELS_LIBRARY)
      set(_onnx_light_kernels_lib "${ONNX_LIGHT_CPU_ONNX_LIGHT_KERNELS_LIBRARY}")
    else()
      file(GLOB _onnx_light_kernels_lib
          "${_onnx_light_libdir}/liblib_onnx_kernels${CMAKE_SHARED_LIBRARY_SUFFIX}*"
          "${_onnx_light_libdir}/lib_onnx_kernels${CMAKE_SHARED_LIBRARY_SUFFIX}*")
    endif()
    file(GLOB _onnx_light_op_lib
        "${_onnx_light_libdir}/liblib_onnx_op.*"
        "${_onnx_light_libdir}/lib_onnx_op.*")
    # lib_onnx_shape hosts onnx-light's own shape-inference functions (e.g.
    # ``onnx_shapes::shapes::nn::ComputeShapeLinearAttention``), reused below
    # by ``onnx_light_cpu::ComputeShapeLinearAttention`` for the ordinary
    # grouping/no-shared-key-heads case. Same best-effort discovery as the
    # other onnx-light libraries: missing (e.g. statically linked into the
    # Python extension on this platform) simply means the reuse is skipped
    # and onnx_light_cpu falls back to its own bespoke shape computation.
    #
    # Like lib_onnx_kernels (required by Cast), lib_onnx_shape is linked into
    # lib_onnx_light_cpu_kernels itself, which is loaded by the ``_cpuregister``
    # Python extension. On Unix that new NEEDED entry is resolved via the
    # explicit RPATH pointing at onnx-light's Python library directory (see
    # ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR below), verified working in CI. On
    # Windows there is no RPATH equivalent: lib_onnx_core/lib_onnx_proto
    # already work there only because some earlier-imported onnx-light Python
    # extension has already loaded them into the process, and nothing else
    # yet forces lib_onnx_shape.dll to be loaded before ``_cpuregister`` needs
    # it, so linking it in makes ``import _cpuregister`` fail with "DLL load
    # failed ... The specified module could not be found." Restrict the
    # discovery to non-Windows platforms until Windows DLL search/preload is
    # addressed; onnx_light_cpu falls back to its bespoke shape computation
    # there instead.
    if(NOT WIN32)
      file(GLOB _onnx_light_shape_lib
          "${_onnx_light_libdir}/liblib_onnx_shape.*"
          "${_onnx_light_libdir}/lib_onnx_shape.*")
    endif()
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
    if(_onnx_light_shape_lib)
      list(GET _onnx_light_shape_lib 0 _onnx_light_shape_lib)
      add_library(onnx_light::lib_onnx_shape
          ${_onnx_light_imported_library_type} IMPORTED GLOBAL)
      set_target_properties(onnx_light::lib_onnx_shape PROPERTIES
          IMPORTED_LOCATION "${_onnx_light_shape_lib}"
          INTERFACE_LINK_LIBRARIES onnx_light::lib_onnx_core)
      if(WIN32)
        if(NOT ONNX_LIGHT_CPU_ONNX_LIGHT_SHAPE_IMPLIB)
          message(FATAL_ERROR
              "The onnx-light shape Windows import library is required for "
              "source-runtime integration.")
        endif()
        set_property(TARGET onnx_light::lib_onnx_shape PROPERTY
            IMPORTED_IMPLIB "${ONNX_LIGHT_CPU_ONNX_LIGHT_SHAPE_IMPLIB}")
      endif()
    endif()
    if(_onnx_light_kernels_lib)
      list(GET _onnx_light_kernels_lib 0 _onnx_light_kernels_lib)
      add_library(onnx_light::lib_onnx_kernels
          ${_onnx_light_imported_library_type} IMPORTED GLOBAL)
      set_target_properties(onnx_light::lib_onnx_kernels PROPERTIES
          IMPORTED_LOCATION "${_onnx_light_kernels_lib}"
          INTERFACE_LINK_LIBRARIES onnx_light::lib_onnx_core)
      if(WIN32)
        if(NOT ONNX_LIGHT_CPU_ONNX_LIGHT_KERNELS_IMPLIB)
          message(FATAL_ERROR
              "ONNX_LIGHT_CPU_ONNX_LIGHT_KERNELS_IMPLIB is required for "
              "Cast's builtin compatibility path on Windows.")
        endif()
        set_property(TARGET onnx_light::lib_onnx_kernels PROPERTY
            IMPORTED_IMPLIB "${ONNX_LIGHT_CPU_ONNX_LIGHT_KERNELS_IMPLIB}")
      endif()
    endif()
    if(TARGET onnx_light::lib_onnx_kernels AND _onnx_light_backend_test_lib)
      list(GET _onnx_light_backend_test_lib 0 _onnx_light_backend_test_lib)
      add_library(onnx_light::lib_onnx_backend_test
          ${_onnx_light_imported_library_type} IMPORTED GLOBAL)
      set_target_properties(onnx_light::lib_onnx_backend_test PROPERTIES
          IMPORTED_LOCATION "${_onnx_light_backend_test_lib}"
          INTERFACE_LINK_LIBRARIES
              "onnx_light::lib_onnx_proto;onnx_light::lib_onnx_kernels")
      if(WIN32)
        if(NOT ONNX_LIGHT_CPU_ONNX_LIGHT_BACKEND_TEST_IMPLIB)
          message(FATAL_ERROR
              "The onnx-light backend-test Windows import library "
              "is required for source-runtime integration.")
        endif()
        set_property(TARGET onnx_light::lib_onnx_backend_test PROPERTY
            IMPORTED_IMPLIB "${ONNX_LIGHT_CPU_ONNX_LIGHT_BACKEND_TEST_IMPLIB}")
      endif()
    endif()
  else()
    find_package(onnx_light CONFIG REQUIRED)
  endif()
  if(NOT TARGET onnx_light::lib_onnx_kernels)
    message(FATAL_ERROR
        "Cast requires onnx_light::lib_onnx_kernels for STRING and extended "
        "numeric compatibility. Set ONNX_LIGHT_CPU_ONNX_LIGHT_KERNELS_LIBRARY "
        "for source-runtime integration, or use an onnx-light C++ package "
        "exporting that target.")
  endif()

  file(GLOB CPU_KERNEL_REGISTRATION_SOURCES CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/*.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/kernels/*/*.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/gradient/*/*.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/patterns/*/*.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/schemas/*/*.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/shapes/*/*.cc")

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
      ml_cpu_compute
      onnx_light::lib_onnx_core
      onnx_light::lib_onnx_kernels)
  # Standalone consumers inherit the concrete Cast link dependency. Linux/macOS
  # build-tree consumers resolve it via CMake's target RPATH; Windows consumers
  # must place the onnx-light runtime DLL directory on their loader search path.
  # Availability definitions on the implementation library are private. The
  # registered activation dispatchers also need to see the compiled SIMD paths.
  if(ML_CPU_COMPUTE_HAVE_AVX512BF16_FLAG)
    target_compile_definitions(lib_onnx_light_cpu_kernels PRIVATE
        ML_CPU_COMPUTE_HAVE_AVX512BF16=1)
  endif()
  if(ML_CPU_COMPUTE_HAVE_AVX2_FMA_FLAG)
    target_compile_definitions(lib_onnx_light_cpu_kernels PRIVATE
        ML_CPU_COMPUTE_HAVE_AVX2_FMA=1)
    if(ML_CPU_COMPUTE_HAVE_F16C_FLAG)
      target_compile_definitions(lib_onnx_light_cpu_kernels PRIVATE
          ML_CPU_COMPUTE_HAVE_F16C=1)
    endif()
  endif()
  # Reuse onnx-light's own shape-inference functions (e.g.
  # ``onnx_shapes::shapes::nn::ComputeShapeLinearAttention``) for the shape
  # cases they already cover instead of re-deriving them; only available when
  # ``onnx_light::lib_onnx_shape`` was discovered above (always via
  # find_package, best-effort for --onnx-light-source).
  if(TARGET onnx_light::lib_onnx_shape)
    target_link_libraries(lib_onnx_light_cpu_kernels PUBLIC
        onnx_light::lib_onnx_shape)
    target_compile_definitions(lib_onnx_light_cpu_kernels PUBLIC
        ONNX_LIGHT_CPU_HAS_ONNX_SHAPE=1)
  endif()
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
        ${CPU_BACKEND_TEST_SOURCES})
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
  set(_onnx_light_cpu_nanobind_abi)
  if(ONNX_LIGHT_CPU_PYTHON_STABLE_ABI)
    list(APPEND _onnx_light_cpu_nanobind_abi STABLE_ABI)
  endif()

  if(CMAKE_CROSSCOMPILING)
    if(ONNX_LIGHT_CPU_PYTHON_STABLE_ABI)
      find_package(Python3 3.12 REQUIRED COMPONENTS Development.Module
          OPTIONAL_COMPONENTS Development.SABIModule)
    else()
      find_package(Python3 3.12 REQUIRED COMPONENTS Development.Module)
    endif()
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter)
    if(NOT TARGET Python::Module)
      add_library(Python::Module ALIAS Python3::Module)
    endif()
    if(TARGET Python3::SABIModule AND NOT TARGET Python::SABIModule)
      add_library(Python::SABIModule ALIAS Python3::SABIModule)
    endif()
  else()
    if(ONNX_LIGHT_CPU_PYTHON_STABLE_ABI)
      find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module
          OPTIONAL_COMPONENTS Development.SABIModule)
    else()
      find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
    endif()
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
    if(ONNX_LIGHT_CPU_RELEASE_WHEEL)
      set(_onnx_light_cpu_install_libdir "$ORIGIN/../../onnx_light/onnx_py")
    else()
      set(_onnx_light_cpu_install_libdir "${ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR}")
    endif()
    set_target_properties(lib_onnx_light_cpu_kernels PROPERTIES
        INSTALL_RPATH "$ORIGIN;${_onnx_light_cpu_install_libdir}"
        BUILD_RPATH "$ORIGIN;${ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR};$<TARGET_FILE_DIR:onnx_light::lib_onnx_kernels>"
    )

    # The backend-test registration library, when linked into _cpuregister, is
    # installed next to it and loaded as one of its runtime dependencies. Like
    # lib_onnx_light_cpu_kernels above it directly NEEDs onnx-light's
    # lib_onnx_core/lib_onnx_proto (and lib_onnx_backend_test), so it needs the
    # same extra RPATH entry pointing at onnx-light's Python library directory.
    if(TARGET lib_onnx_light_cpu_backend_test)
      set_target_properties(lib_onnx_light_cpu_backend_test PROPERTIES
          INSTALL_RPATH "$ORIGIN;${_onnx_light_cpu_install_libdir}"
          BUILD_RPATH "$ORIGIN;${ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR}"
      )
    endif()
  endif()

  nanobind_add_module(_cpukernels ${_onnx_light_cpu_nanobind_abi} ${PY_KERNELS_SOURCES})
  target_include_directories(_cpukernels PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(_cpukernels PRIVATE ml_cpu_compute)

  set_target_properties(_cpukernels PROPERTIES
      INSTALL_RPATH "$ORIGIN"
      BUILD_RPATH "$ORIGIN")

  install(TARGETS _cpukernels
      LIBRARY DESTINATION onnx_light_cpu/onnx_py
      RUNTIME DESTINATION onnx_light_cpu/onnx_py)

  if(_cpu_lib_type STREQUAL "SHARED")
    install(TARGETS ml_cpu_compute
        LIBRARY DESTINATION onnx_light_cpu/onnx_py
        RUNTIME DESTINATION onnx_light_cpu/onnx_py)
  endif()

  # -------------------------------------------------------------------------
  # Kernel-registration Python extension: _cpuregister
  # -------------------------------------------------------------------------
  # Exposes registration and inventory APIs. It links the onnx-light
  # kernel-registration library and is therefore only built when the
  # integration is enabled.
  if(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
    nanobind_add_module(_cpuregister ${_onnx_light_cpu_nanobind_abi} ${PY_REGISTER_SOURCES})
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
          INSTALL_RPATH "$ORIGIN;${_onnx_light_cpu_install_libdir}"
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
  install(DIRECTORY ml_cpu_compute/include/
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(TARGETS ml_cpu_compute
      EXPORT onnx_light_cpuTargets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  set(_header_install_excludes
      PATTERN "backend_test" EXCLUDE
      PATTERN "logical_kernels.h" EXCLUDE
      PATTERN "math_kernels.h" EXCLUDE
      PATTERN "onnx_py" EXCLUDE)
  if(NOT ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
    list(APPEND _header_install_excludes
        PATTERN "gradient" EXCLUDE
        PATTERN "kernels" EXCLUDE
        PATTERN "patterns" EXCLUDE
        PATTERN "schemas" EXCLUDE
        PATTERN "shapes" EXCLUDE)
  endif()
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_cpu/"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_cpu"
      FILES_MATCHING
      PATTERN "*.h"
      ${_header_install_excludes})

  if(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
    install(TARGETS lib_onnx_light_cpu_kernels
        EXPORT onnx_light_cpuTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

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
# Isolated C++ throughput benchmarks (opt-in; ctest only runs tiny CLI smoke tests)
# ---------------------------------------------------------------------------
if(ONNX_LIGHT_CPU_BUILD_BENCHMARKS)
  file(GLOB _throughput_sources CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/*_throughput.cc")
  foreach(_source IN LISTS _throughput_sources)
    get_filename_component(_target "${_source}" NAME_WE)
    if(_target STREQUAL "batch_normalization_throughput" AND NOT ONNX_LIGHT_CPU_WITH_ONNX_LIGHT)
      continue()
    endif()
    add_executable(${_target} "${_source}")
    target_link_libraries(${_target} PRIVATE ml_cpu_compute)
    if(_target STREQUAL "batch_normalization_throughput")
      target_link_libraries(${_target} PRIVATE lib_onnx_light_cpu_kernels)
    endif()
    if(_target STREQUAL "integer_gemm_throughput")
      if(ML_CPU_COMPUTE_HAVE_AVXVNNI_FLAG)
        target_compile_definitions(${_target} PRIVATE ML_CPU_COMPUTE_HAVE_AVXVNNI=1)
      endif()
      if(ML_CPU_COMPUTE_HAVE_AVX2_FMA_FLAG)
        target_compile_definitions(${_target} PRIVATE ML_CPU_COMPUTE_HAVE_AVX2_INTEGER=1)
      endif()
      if(ML_CPU_COMPUTE_HAVE_AVX512VNNI_FLAG)
        target_compile_definitions(${_target} PRIVATE ML_CPU_COMPUTE_HAVE_AVX512VNNI=1)
      endif()
    endif()
    if(_target STREQUAL "abs_throughput" AND ML_CPU_COMPUTE_HAVE_AVX512F_FLAG)
      target_compile_definitions(${_target} PRIVATE ML_CPU_COMPUTE_HAVE_AVX512=1)
    endif()
    if(_target STREQUAL "exp_log_throughput" AND ML_CPU_COMPUTE_HAVE_AVX2_FMA_FLAG)
      target_compile_definitions(${_target} PRIVATE ML_CPU_COMPUTE_HAVE_AVX2_FMA=1)
    endif()
    if(_target STREQUAL "exp_log_throughput" AND ML_CPU_COMPUTE_HAVE_AVX512F_FLAG)
      target_compile_definitions(${_target} PRIVATE ML_CPU_COMPUTE_HAVE_AVX512=1)
    endif()
    ml_cpu_compute_apply_hardening(${_target})
  endforeach()
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
    target_compile_definitions(${_name} PRIVATE
        ML_CPU_COMPUTE_MAX_SIMD_LEVEL=${_ml_cpu_compute_max_simd_level_value})
    foreach(_corpus_source IN LISTS CPU_BACKEND_CORPUS_SOURCES)
      get_filename_component(_corpus_name "${_corpus_source}" NAME_WE)
      string(REGEX REPLACE "_corpus$" "" _corpus_subject "${_corpus_name}")
      if(_name STREQUAL "test_${_corpus_subject}")
        target_sources(${_name} PRIVATE "${_corpus_source}")
      endif()
    endforeach()
    target_link_libraries(${_name} PRIVATE ${_link_lib} GTest::gtest_main)
    target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    if(_name STREQUAL "test_cast_kernel")
      if(ML_CPU_COMPUTE_HAVE_F16C_FLAG)
        target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_AVX_F16C=1)
      endif()
      if(ML_CPU_COMPUTE_HAVE_AVX2_FLAG)
        target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_AVX2=1)
      endif()
      if(ML_CPU_COMPUTE_HAVE_AVX2_FLAG AND ML_CPU_COMPUTE_HAVE_F16C_FLAG)
        target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_AVX2_F16C=1)
      endif()
      if(ML_CPU_COMPUTE_HAVE_AVX512DQ_FLAG)
        target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_AVX512DQ=1)
      endif()
    endif()
    if(_name STREQUAL "test_matmul_nbits")
      foreach(_isa IN ITEMS AVX2 AVXVNNI AVX512VNNI AVX512BW)
        if(ML_CPU_COMPUTE_HAVE_${_isa}_FLAG)
          target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_${_isa}=1)
        endif()
      endforeach()
    endif()
    if(_name STREQUAL "test_integer_gemm_vnni")
      if(ML_CPU_COMPUTE_HAVE_AVXVNNI_FLAG)
        target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_AVXVNNI=1)
      endif()
      # This differential binary names and calls the optional native integer
      # kernels directly. The library keeps availability definitions private,
      # so mirror only the definitions whose translation units were built.
      if(ML_CPU_COMPUTE_HAVE_AVX2_FMA_FLAG)
        target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_AVX2_INTEGER=1)
      endif()
      if(ML_CPU_COMPUTE_HAVE_AVX512VNNI_FLAG)
        target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_AVX512VNNI=1)
      endif()
      if(ML_CPU_COMPUTE_HAVE_AVX512BW_FLAG)
        target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_AVX512BW=1)
      endif()
      if(ML_CPU_COMPUTE_HAVE_AMX_INT8_FLAG)
        target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_AMX_INT8=1)
      endif()
      if(ML_CPU_COMPUTE_HAVE_NEON_DOTPROD_INTRINSICS)
        target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_NEON_DOTPROD=1)
      endif()
    endif()
    if(_name STREQUAL "test_gemm_kernel" AND ML_CPU_COMPUTE_HAVE_AVX2_FMA_FLAG)
      # This differential test calls optional AVX2 conversion helpers directly.
      # Mirror the private library definitions only when those helpers were built.
      target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_AVX2_FMA=1)
      if(ML_CPU_COMPUTE_HAVE_F16C_FLAG)
        target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_F16C=1)
      endif()
    endif()
    if((_name STREQUAL "test_exp_log_kernel" OR
        _name STREQUAL "test_tanh_kernel" OR
        _name STREQUAL "test_onnx_light_sigmoid_softmax_kernel") AND
       ML_CPU_COMPUTE_HAVE_AVX2_FMA_FLAG)
      target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_AVX2_FMA=1)
      if(_name STREQUAL "test_onnx_light_sigmoid_softmax_kernel" AND
         ML_CPU_COMPUTE_HAVE_F16C_FLAG)
        target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_F16C=1)
      endif()
    endif()
    if((_name STREQUAL "test_exp_log_kernel" OR
        _name STREQUAL "test_tanh_kernel" OR
        _name STREQUAL "test_gemm_kernel") AND
       ML_CPU_COMPUTE_HAVE_AVX512F_FLAG)
      target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_AVX512=1)
    endif()
    if((_name STREQUAL "test_abs_kernel" OR _name STREQUAL "test_abs_avx512_kernel") AND
       ML_CPU_COMPUTE_HAVE_AVX512F_FLAG)
      target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_AVX512=1)
    endif()
    if(_name STREQUAL "test_abs_kernel" AND ML_CPU_COMPUTE_HAVE_SSSE3_FLAG)
      target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_SSSE3=1)
    endif()
    if(_name STREQUAL "test_abs_avx512_kernel" AND ML_CPU_COMPUTE_HAVE_AVX512BW_FLAG)
      target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_AVX512BW=1)
    endif()
    if(_name STREQUAL "test_rms_normalization_kernel" AND ML_CPU_COMPUTE_HAVE_F16C_FLAG)
      # This differential test calls the optional F16C RMSNormalization
      # kernel directly. Mirror the private library definition only when that
      # translation unit was built.
      target_compile_definitions(${_name} PRIVATE ML_CPU_COMPUTE_HAVE_RMS_F16C=1)
    endif()
    if("lib_onnx_light_cpu_kernels" IN_LIST _link_lib AND
       ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR)
      set_target_properties(${_name} PROPERTIES
          BUILD_RPATH "${ONNX_LIGHT_CPU_ONNX_LIGHT_PY_LIBDIR}")
    endif()
    # Discover the test cases when ctest runs instead of at build time. The
    # POST_BUILD discovery run executes the freshly linked binary, which is
    # flaky on macOS (a just-written, ad-hoc signed executable is sometimes
    # killed by the kernel), and a failed discovery run aborts the whole build.
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
      gtest_discover_tests(${_name} DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 60)
    else()
      gtest_discover_tests(${_name})
    endif()
    ml_cpu_compute_apply_hardening(${_name})
  endfunction()

  file(GLOB _test_sources CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/*/test_*.cc")
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
        elseif(_test_name STREQUAL "test_onnx_light_custom_operator_support" OR
               _test_name STREQUAL "test_onnx_light_normalization_kernel" OR
               _test_name STREQUAL "test_onnx_light_cast_kernel" OR
               _test_name STREQUAL "test_onnx_light_gather_kernel" OR
               _test_name STREQUAL "test_onnx_light_reduce_sum_kernel" OR
               _test_name STREQUAL "test_onnx_light_scatter_nd_kernel" OR
               _test_name STREQUAL "test_onnx_light_slice_kernel" OR
               _test_name STREQUAL "test_onnx_light_concat_kernel" OR
               _test_name STREQUAL "test_onnx_light_split_kernel" OR
               _test_name STREQUAL "test_onnx_light_simplified_layer_normalization_kernel" OR
               _test_name STREQUAL "test_onnx_light_simplified_layer_normalization_support")
          if(TARGET onnx_light::lib_onnx_kernels)
            onnx_light_cpu_add_test("${_test_src}"
                "lib_onnx_light_cpu_kernels;onnx_light::lib_onnx_kernels")
          endif()
        else()
          onnx_light_cpu_add_test("${_test_src}" lib_onnx_light_cpu_kernels)
        endif()
      endif()
    else()
      onnx_light_cpu_add_test("${_test_src}" ml_cpu_compute)
    endif()
  endforeach()

  if(TARGET matmul_nbits_throughput)
    foreach(_bits IN ITEMS 4 8)
      add_test(NAME MatMulNBitsThroughput.Bits${_bits}
          COMMAND matmul_nbits_throughput 1 32 9 1 ${_bits})
      set_tests_properties(MatMulNBitsThroughput.Bits${_bits} PROPERTIES
          TIMEOUT 30 PASS_REGULAR_EXPRESSION "1,32,9,${_bits},float32,1,")
      add_test(NAME MatMulNBitsThroughput.Bits${_bits}Generic
          COMMAND matmul_nbits_throughput 1 32 9 1 ${_bits} --generic)
      set_tests_properties(MatMulNBitsThroughput.Bits${_bits}Generic PROPERTIES
          TIMEOUT 30 PASS_REGULAR_EXPRESSION "1,32,9,${_bits},float32,1,.*generic")
    endforeach()
    add_test(NAME MatMulNBitsThroughput.DefaultBits
        COMMAND matmul_nbits_throughput 1 32 9 1)
    set_tests_properties(MatMulNBitsThroughput.DefaultBits PROPERTIES
        TIMEOUT 30 PASS_REGULAR_EXPRESSION "1,32,9,4,float32,1,")
    add_test(NAME MatMulNBitsThroughput.DefaultBitsGeneric
        COMMAND matmul_nbits_throughput 1 32 9 1 --generic)
    set_tests_properties(MatMulNBitsThroughput.DefaultBitsGeneric PROPERTIES
        TIMEOUT 30 PASS_REGULAR_EXPRESSION "1,32,9,4,float32,1,.*generic")
  endif()
endif()

# ---------------------------------------------------------------------------
# Apply OpenSSF compiler hardening flags to the non-test targets when
# ONNX_HARDENING=ON. When the option is OFF the helper is a no-op. Test targets
# are hardened as they are declared in onnx_light_cpu_add_test above.
# ---------------------------------------------------------------------------
set(_onnx_light_cpu_hardening_targets)
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
    ml_cpu_compute_apply_hardening(${_t})
  endif()
endforeach()
