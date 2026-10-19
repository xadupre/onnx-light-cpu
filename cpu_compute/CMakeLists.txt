cmake_minimum_required(VERSION 3.15)
project(cpu_compute VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)

option(CPU_COMPUTE_ENABLE_COVERAGE "Instrument computation code for coverage." OFF)
option(CPU_COMPUTE_HARDENING "Enable compiler hardening." OFF)
option(CPU_COMPUTE_BUILD_TESTS "Build standalone consumer tests." OFF)
if(CPU_COMPUTE_ENABLE_COVERAGE AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(FATAL_ERROR "Coverage requires a GCC or Clang compiler.")
endif()
if(CPU_COMPUTE_HARDENING AND NOT COMMAND cpu_compute_apply_hardening)
  include(cmake/Hardening.cmake)
endif()
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  option(CPU_COMPUTE_INSTALL "Install the standalone computation library." ON)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type." FORCE)
  endif()
else()
  option(CPU_COMPUTE_INSTALL "Install the standalone computation library." OFF)
endif()

include(cmake/CpuFeatures.cmake)
if(NOT DEFINED _cpu_lib_type)
  if(BUILD_SHARED_LIBS)
    set(_cpu_lib_type SHARED)
  else()
    set(_cpu_lib_type STATIC)
  endif()
endif()
include(cmake/ComputeLibrary.cmake)
add_library(cpu_compute::cpu_compute ALIAS cpu_compute)
if(WIN32 AND _cpu_lib_type STREQUAL "SHARED")
  set_target_properties(cpu_compute PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
if(CPU_COMPUTE_HARDENING)
  cpu_compute_apply_hardening(cpu_compute)
endif()

if(CPU_COMPUTE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(CPU_COMPUTE_INSTALL)
  install(TARGETS cpu_compute EXPORT cpu_computeTargets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/cpu_compute")
  configure_package_config_file(
      cmake/cpu_computeConfig.cmake.in
      "${CMAKE_CURRENT_BINARY_DIR}/cpu_computeConfig.cmake"
      INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cpu_compute")
  write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/cpu_computeConfigVersion.cmake"
      COMPATIBILITY SameMajorVersion)
  install(EXPORT cpu_computeTargets NAMESPACE cpu_compute::
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cpu_compute")
  install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/cpu_computeConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/cpu_computeConfigVersion.cmake"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cpu_compute")
endif()
