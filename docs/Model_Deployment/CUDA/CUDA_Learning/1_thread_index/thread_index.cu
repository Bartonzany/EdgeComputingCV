#include <cuda_runtime.h>
#include <stdio.h>

__global__ void thread_index_kernel(float *A, const int num_x, const int num_y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = col * num_x + row;

    // Ensure we are within the bounds of the array
    if (row < num_x && col < num_y) {
        printf("thread_id: (%d, %d) block_id: (%d, %d) coordinate: (%d, %d) global index: %2d val: %f\n",
                threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, row, col, idx, A[idx]);
    }
}

void printMatrix(float * C,const int nx,const int ny) {
    float *ic=C;
    printf("Matrix<%d,%d>:\n",ny,nx);

    for(int i=0;i<ny;i++) {
        for(int j=0;j<nx;j++) {
            printf("%6f ",ic[j]);
        }
        
        ic+=nx;
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int nx = 8, ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // Allocate memory on the host
    float* A_host = (float*)malloc(nBytes);

    // Initialize host array with some values
    for (int i = 0; i < nxy; ++i) {
        A_host[i] = (float)rand() / RAND_MAX; // Assign random float values between 0 and 1
    }
    printMatrix(A_host, nx, ny);

    // Allocate memory on the device
    float *A_dev = NULL;
    cudaMalloc((void**)&A_dev, nBytes);

    // Copy data from host to device
    cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // Launch the kernel
    thread_index_kernel<<<grid, block>>>(A_dev, nx, ny);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete

    // Free device memory
    cudaFree(A_dev);

    // Free host memory
    free(A_host);

    cudaDeviceReset();
    return 0;
}
