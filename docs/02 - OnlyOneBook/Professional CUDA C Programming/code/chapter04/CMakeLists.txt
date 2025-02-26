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


# 定义一个宏来简化重复的配置
macro(configure_experiment target_name source_file)
    add_executable(${target_name} ${source_file})
    target_include_directories(${target_name} PRIVATE ../common)
    target_compile_features(${target_name} PRIVATE cxx_std_17 cuda_std_17)
    
    # 根据目标名称设置不同的编译选项
    if(target_name MATCHES "_with_L1cache")
        target_compile_options(${target_name} PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-command-line-argument>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=cg>  # 启用一级缓存
        )
    elseif(target_name MATCHES "_without_L1cache")
        target_compile_options(${target_name} PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-command-line-argument>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=ca>  # 不启用一级缓存
        )
    endif()

    target_compile_options(${target_name} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-G -O0>>
        $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>:-O3>>
    )
    install(TARGETS ${target_name} DESTINATION chapter04)
endmacro()

# L1 内存实验
set(SUMARRAYOFFSET_TARGETS
    sumArrayOffset_with_L1cache
    sumArrayOffset_without_L1cache
)

foreach(target IN LISTS SUMARRAYOFFSET_TARGETS)
    configure_experiment(${target} sumArrayOffset.cu)
endforeach()

# AoS 内存实验
set(AoS_TARGETS
    AoS_with_L1cache
    AoS_without_L1cache
)

foreach(target IN LISTS AoS_TARGETS)
    configure_experiment(${target} AoS.cu)
endforeach()

# SoA 内存实验
set(SoA_TARGETS
    SoA_with_L1cache
    SoA_without_L1cache
)

foreach(target IN LISTS SoA_TARGETS)
    configure_experiment(${target} SoA.cu)
endforeach()

# sumArrayOffsetUnrolling 内存实验
set(sumArrayOffsetUnrolling_TARGETS
    sumArrayOffsetUnrolling_with_L1cache
    sumArrayOffsetUnrolling_without_L1cache
)

foreach(target IN LISTS sumArrayOffsetUnrolling_TARGETS)
    configure_experiment(${target} sumArrayOffsetUnrolling.cu)
endforeach()
