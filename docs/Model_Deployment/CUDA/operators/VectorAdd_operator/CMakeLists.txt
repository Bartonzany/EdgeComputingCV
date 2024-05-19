cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
Project(vectoradd_op CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -g -G -O3")
add_executable(vectoradd_op vectoradd_op.cu)