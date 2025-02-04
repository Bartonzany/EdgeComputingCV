cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
Project(CUDA_C CXX C CUDA)

# Support GTX 1080 and RTX 3090
set(CMAKE_CUDA_ARCHITECTURES 61;80)

add_subdirectory(chapter01)
add_subdirectory(chapter02)
add_subdirectory(chapter03)

# add_subdirectory(3_device_information)
# add_subdirectory(4_divergence)
# add_subdirectory(5_sum_matrix2D)
# add_subdirectory(6_reduceInteger)