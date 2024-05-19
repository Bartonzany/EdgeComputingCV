cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
Project(div_op CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -g -G -O3")
add_executable(div_op div_op.cu)