cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(chapter02 LANGUAGES C CXX CUDA)  

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# 定义所有可执行目标列表
set(EXECUTABLE_TARGETS
    sumArrays
    checkDimension
    defineGridBlock
    sumArraysOnGPU-small-case
    sumArraysTimer
    checkThreadIndex
    checkThreadIndexFloat
    sumMatrix
    checkDeviceInfor
)

# 为每个目标创建可执行文件并设置属性
foreach(target IN LISTS EXECUTABLE_TARGETS)
    add_executable(${target} ${target}.cu)
    target_include_directories(${target} PRIVATE ../common)
    target_compile_features(${target} PRIVATE cxx_std_17 cuda_std_17)
    
    # 统一应用编译选项
    target_compile_options(${target} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-command-line-argument>
        $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-G -O0>>
        $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>:-O3>>
    )
    
    install(TARGETS ${target} DESTINATION chapter02)
endforeach()
