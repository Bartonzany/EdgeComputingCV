cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(chapter04 LANGUAGES C CXX CUDA)  

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87 89 90 100 101 120) 
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# 定义所有可执行目标列表
set(EXECUTABLE_TARGETS
    checkRegisters
    globalVariable
    pinMemTransfer
    sumArrayZeroCopy
    UVA
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
    
    install(TARGETS ${target} DESTINATION chapter04)
endforeach()

# L1 内存实验
set(SUMARRAYOFFSET_TARGETS
    sumArrayOffset_with_L1cache
    sumArrayOffset_without_L1cache
)

foreach(target IN LISTS SUMARRAYOFFSET_TARGETS)
    add_executable(${target} sumArrayOffset.cu)
    target_include_directories(${target} PRIVATE ../common)
    target_compile_features(${target} PRIVATE cxx_std_17 cuda_std_17)
    
    # 根据目标名称设置不同的编译选项
    if(target STREQUAL "sumArrayOffset_with_L1cache")
        target_compile_options(${target} PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-command-line-argument>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=cg>  # 启用一级缓存
        )
    elseif(target STREQUAL "sumArrayOffset_without_L1cache")
        target_compile_options(${target} PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-command-line-argument>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=ca>  # 不启用一级缓存
        )
    endif()

    target_compile_options(${target} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-G -O0>>
        $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>:-O3>>
    )
    install(TARGETS ${target} DESTINATION chapter04)
endforeach()