cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(chapter01 LANGUAGES C CXX CUDA)  

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87 89 90 100 101 120) 
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

add_executable(helloWorld helloWorld.cu)
target_include_directories(helloWorld PRIVATE ../common)

target_compile_features(helloWorld PRIVATE cxx_std_17 cuda_std_17)

target_compile_options(helloWorld PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-command-line-argument> 
    $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>: -G -O0>>
    $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>: -O3>>
)

install(TARGETS helloWorld DESTINATION chapter01)
