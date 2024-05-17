cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
Project(MatMul_op CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -g -G -O3")
add_executable(MatMul_op MatMul_op.cu)