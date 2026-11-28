cmake_minimum_required(VERSION 3.15)
project(ml_cpu_compute_consumer LANGUAGES CXX)

if(NOT TARGET ml_cpu_compute::ml_cpu_compute)
  find_package(ml_cpu_compute CONFIG REQUIRED)
endif()
enable_testing()
add_executable(compute_consumer consumer.cc)
target_link_libraries(compute_consumer PRIVATE ml_cpu_compute::ml_cpu_compute)
add_test(NAME compute_consumer COMMAND compute_consumer)

if(ML_CPU_COMPUTE_BUILD_TESTS)
  if(NOT TARGET GTest::gtest_main)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP ON
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  include(GoogleTest)
  add_executable(test_execution test_execution.cc)
  target_link_libraries(test_execution PRIVATE ml_cpu_compute::ml_cpu_compute GTest::gtest_main)
  gtest_discover_tests(test_execution)

  if(ML_CPU_COMPUTE_X86 AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_OBJDUMP)
    set(_avx_checks)
    foreach(_isa IN ITEMS AVX AVX2 AVX2_FMA F16C AVXVNNI AVX512F AVX512BW AVX512BF16
                          AVX512FP16 AVX512VNNI)
      if(ML_CPU_COMPUTE_HAVE_${_isa}_FLAG)
        list(APPEND _avx_checks ${_isa})
      endif()
    endforeach()
    if(_avx_checks)
      add_test(NAME compute_avx_instructions
          COMMAND "${CMAKE_COMMAND}"
              "-DOBJDUMP=${CMAKE_OBJDUMP}"
              "-DLIBRARY=$<TARGET_FILE:ml_cpu_compute>"
              "-DISAS=${_avx_checks}"
              "-DCHECK_REGISTER_ALLOCATION=$<AND:$<CONFIG:Release>,$<NOT:$<BOOL:${ML_CPU_COMPUTE_ENABLE_COVERAGE}>>>"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/check_avx_instructions.cmake")
    endif()
  endif()
endif()
