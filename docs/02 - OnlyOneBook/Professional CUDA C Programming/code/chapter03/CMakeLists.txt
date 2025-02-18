cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(chapter03 LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87 89 90 100 101 120) 
# 定义不兼容的CUDA架构
set(CUDA_INCOMPATIBLE_ARCHITECTURES 53 62 72 87)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# 定义所有可执行目标列表
set(EXECUTABLE_TARGETS
    simpleDivergence
    simpleDeviceQuery
    sumMatrix2D
    reduceInteger
    reduceUnrolling
    nestedHelloWorld
)

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

    # 特殊处理nestedHelloWorld
    if(target STREQUAL "nestedHelloWorld")
        set(FILTERED_ARCHS ${CMAKE_CUDA_ARCHITECTURES})
        foreach(arch IN LISTS CUDA_INCOMPATIBLE_ARCHITECTURES)
            list(REMOVE_ITEM FILTERED_ARCHS ${arch})
        endforeach()

        # 为特定目标设置过滤后的架构
        set_target_properties(${target} PROPERTIES
            CUDA_ARCHITECTURES "${FILTERED_ARCHS}"
        )

        target_compile_options(${target} PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
        )
        set_target_properties(${target} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
    endif()

    install(TARGETS ${target} DESTINATION chapter03)
endforeach()
