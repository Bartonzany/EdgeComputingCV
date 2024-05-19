cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
Project(mul_op CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -g -G -O3")
add_executable(mul_op mul_op.cu)