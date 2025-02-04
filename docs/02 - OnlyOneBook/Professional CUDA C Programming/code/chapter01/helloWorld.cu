#include "../common/common.h"
#include <stdio.h>

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */

__global__ void hello_world_GPU(void) {
    printf("Hello World from GPU! Thread ID: %d\n", threadIdx.x);
}

int main(int argc, char** argv) {
    printf("Hello World from CPU!\n");
    hello_world_GPU<<<1, 10>>>();
    CHECK(cudaDeviceReset());    // if no this line ,it can not output hello world from gpu
    return 0;
}