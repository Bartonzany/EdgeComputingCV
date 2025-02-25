#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

int main(int argc, char* argv[]) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("Device %d: %s\n", i, prop.name);
        printf("Registers per block: %d\n", prop.regsPerBlock);
        printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("Constant memory per thread: %zu bytes\n", prop.totalConstMem / prop.multiProcessorCount);
        printf("\n");
    }

    return 0;
}
