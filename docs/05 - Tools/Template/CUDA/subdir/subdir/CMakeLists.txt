# 模板1：CUDA 项目的 CMakeLists.txt 文件

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(<<工程名>> LANGUAGES C CXX CUDA)  # 必须声明CUDA语言支持

# 设置 CMake 模块路径 list(APPEND CMAKE_MODULE_PATH "<<自定义的 CMake 模块路径>>")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

find_package(CUDAToolkit REQUIRED) # 查找 CUDA Toolkit

set(CMAKE_POSITION_INDEPENDENT_CODE ON) # 设置生成的代码为独立的
set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87 89 90 100 101 120) # 设置支持的 CUDA 架构
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets") # 忽略关于过时 GPU 目标的警告

target_include_directories(helloWorld PRIVATE ../common) # 包含目录和库
target_compile_features(asyncAPI PRIVATE cxx_std_17 cuda_std_17) # 设置编译特性

add_executable(asyncAPI asyncAPI.cu) # 添加源文件，替换为你自己的源文件

# 设置 CUDA 编译选项
target_compile_options(helloWorld PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-command-line-argument>  # 抑制无关警告
    $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>: -G -O0>>
    $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>: -O3>>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

set_target_properties(helloWorld PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_LINK_LIBRARIES_PROPAGATE OFF  # 阻止库传播
    INTERFACE_LINK_LIBRARIES ""  # 清空接口库
)

install(TARGETS helloWorld DESTINATION chapter01)












# 模板2：CUDA 项目的 CMakeLists.txt 文件

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(chapter02 LANGUAGES CXX CUDA)  

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87)
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
