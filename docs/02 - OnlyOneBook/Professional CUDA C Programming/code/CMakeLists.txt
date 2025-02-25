cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
Project(CUDA_C LANGUAGES CXX C CUDA)

# Support GTX 1080 and RTX 3090
set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87 89 90 100 101 120) 

add_subdirectory(chapter01)
add_subdirectory(chapter02)
add_subdirectory(chapter03)
add_subdirectory(chapter04)