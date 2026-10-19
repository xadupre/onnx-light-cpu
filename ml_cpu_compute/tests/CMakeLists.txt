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
endif()
