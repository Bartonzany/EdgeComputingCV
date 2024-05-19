#include <stdio.h>
#include <cuda_runtime.h>
#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);

}

void sumMatrix2D_CPU(float * MatA, float * MatB, float * MatC, const int num_x, const int num_y) {
    float * a = MatA;
    float * b = MatB;
    float * c = MatC;

    for(int j = 0; j < num_y; j++) {
        for(int i = 0; i < num_x; i++) {
            c[i] = a[i] + b[i];
        }

        c += num_x;
        b += num_x;
        a += num_x;
    }
}

__global__ void sumMatrix(float * MatA, float * MatB, float * MatC, const int num_x, const int num_y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = col * num_x + row;

    if (row < num_x && col < num_y) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc,char** argv) {
    int row = 1<<14; // 2^12, 16384
    int col = 1<<14; // 2^12, 16384
    int sum = row * col;
    int nBytes = sum * sizeof(float);

    //Malloc
    float* A_host = (float*)malloc(nBytes);
    float* B_host = (float*)malloc(nBytes);
    float* C_host = (float*)malloc(nBytes);
    float* C_from_gpu = (float*)malloc(nBytes);

    // 为输入矩阵赋值
    for (int i = 0; i < sum; ++i) {
        A_host[i] = (float)rand() / RAND_MAX;
        B_host[i] = (float)rand() / RAND_MAX;
    }

    // 输出矩阵，分配设备端内存
    float *A_dev = NULL;
    float *B_dev = NULL;
    float *C_dev = NULL;
    cudaMalloc((void**)&A_dev, nBytes);
    cudaMalloc((void**)&B_dev, nBytes);
    cudaMalloc((void**)&C_dev, nBytes);

    // 将输入数据从主机端拷贝到设备端
    cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, nBytes, cudaMemcpyHostToDevice);

    int dim_x = argc>2?atoi(argv[1]):32;
    int dim_y = argc>2?atoi(argv[2]):32;
    double iStart = 0.0;
    double iElaps = 0.0;

    // cpu compute
    cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost);
    iStart=cpuSecond();
    sumMatrix2D_CPU(A_host, B_host, C_host, row, col);
    iElaps=cpuSecond()-iStart;
    printf("CPU Execution Time elapsed %f sec\n",iElaps);

    // 2d block and 2d grid
    dim3 blockDim_0(dim_x, dim_y);
    dim3 gridDim_0((row + blockDim_0.x - 1) / blockDim_0.x, (col + blockDim_0.y - 1) / blockDim_0.y);
    iStart = cpuSecond();
    sumMatrix<<<gridDim_0, blockDim_0>>>(A_dev, B_dev, C_dev, row, col); // 调用 CUDA 核函数
    cudaDeviceSynchronize();
    iElaps=cpuSecond() - iStart;
    printf("GPU Execution configuration<<<(%d, %d),(%d, %d)>>> Time elapsed %f sec\n",
            gridDim_0.x, gridDim_0.y, blockDim_0.x, blockDim_0.y, iElaps);
    cudaMemcpy(C_host, C_dev, nBytes, cudaMemcpyDeviceToHost); 
    
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    free(A_host);
    free(B_host);
    free(C_host);
    free(C_from_gpu);
    cudaDeviceReset();
    return 0;
}