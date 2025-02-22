cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(string 
                VERSION 1.0 
                DESCRIPTION "use of string"
                LANGUAGES C CXX)  

set(CMAKE_CXX_STANDARD 14)              # 设置使用的C++标准，11 14 17
set(CMAKE_CXX_STANDARD_REQUIRED OFF)    # 强制要求必须达到相应的C++标准, ON/OFF
set(CMAKE_CXX_EXTENSIONS OFF)           # 不接受编译器扩展，ON/OFF
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # 生成编译命令文件，ON/OFF

set_property(GLOBAL PROPERTY USE_FOLDERS ON)  # 设置工程文件夹, 要求在build文件夹内仍然保持相应的文件夹结构

add_executable(string string.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Werror)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
endif()

# 确保编译器满足项目的最低版本要求
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    message(FATAL_ERROR "GCC version must be at least 7.0")
endif()

# 禁止在源码目录下生成构建系统（这会污染整个项目），可以避免很多误操作
if(PROJECT_BINARY_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(FATAL_ERROR "The binary directory cannot be the same as source directory")
endif()

install(TARGETS string DESTINATION string)
