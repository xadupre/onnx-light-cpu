cmake_minimum_required(VERSION 3.15)
project(onnx_light_cpu_standalone_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx_light REQUIRED)
find_package(onnx_light_cpu REQUIRED)

add_executable(onnx_light_cpu_inference main.cc)
target_link_libraries(
  onnx_light_cpu_inference
  PRIVATE
    onnx_light_cpu::lib_onnx_light_cpu_kernels
    onnx_light::lib_onnx_kernels)
