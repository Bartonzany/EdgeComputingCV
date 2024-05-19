#include <stdio.h>
#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif

#define N 3 // 矩阵的大小

// CUDA 核函数，执行矩阵加法
__global__ void MatrixAdd(float * MatA, float * MatB, float * MatC, const int num_x, const int num_y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = col * num_x + row;

    if (row < num_x && col < num_y) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);

}

void printMatrix(float * C, const int nx, const int ny) {
    float *ic = C;

    printf("Matrix<%d,%d>:\n",ny,nx);

    for(int i = 0; i < ny; i++) {
        for(int j = 0; j < nx; j++) {
            printf("%6f ",ic[j]);
        }
        
        ic += nx;
        printf("\n");
    }
}

int main() {
    int row = 1<<12; // 2^12, 4096
    int col = 1<<12; // 2^12, 4096
    // int row = 1<<5; // 用于打印矩阵，32
    // int col = 1<<5; // 用于打印矩阵，32
    int sum = row * col;
    int nBytes = sum * sizeof(float);

    // 输入矩阵，分配主机端内存
    float* A_host = (float*)malloc(nBytes);
    float* B_host = (float*)malloc(nBytes);
    float* C_host = (float*)malloc(nBytes);

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

    // 计算线程块和线程数
    int dim_x = 32;
    int dim_y = 32;
    double iStart = 0.0;
    double iElaps = 0.0;


    /********* 2d block and 2d grid***********/
    dim3 blockDim_0(dim_x, dim_y);
    dim3 gridDim_0((row + blockDim_0.x - 1) / blockDim_0.x, (col + blockDim_0.y - 1) / blockDim_0.y);
    iStart = cpuSecond();
    MatrixAdd<<<gridDim_0, blockDim_0>>>(A_dev, B_dev, C_dev, row, col); // 调用 CUDA 核函数
    cudaDeviceSynchronize();
    iElaps=cpuSecond() - iStart;
    printf("GPU Execution configuration<<<(%d, %d),(%d, %d)>>> Time elapsed %f sec\n",
            gridDim_0.x, gridDim_0.y, blockDim_0.x, blockDim_0.y, iElaps);
    cudaMemcpy(C_host, C_dev, nBytes, cudaMemcpyDeviceToHost); // 将计算结果从设备端拷贝到主机端
    // printMatrix(C_host, row, col);// 打印输出结果, 4096*4096 矩阵太耗费时间


    /********* 1d block and 1d grid***********/
    dim3 blockDim_1(dim_x);
    dim3 gridDim_1((sum + blockDim_1.x - 1) / blockDim_1.x);
    iStart = cpuSecond();
    MatrixAdd<<<gridDim_1, blockDim_1>>>(A_dev, B_dev, C_dev, sum, 1); // 调用 CUDA 核函数
    cudaDeviceSynchronize();
    iElaps=cpuSecond() - iStart;
    printf("GPU Execution configuration<<<(%d, %d),(%d, %d)>>> Time elapsed %f sec\n",
            gridDim_1.x, gridDim_1.y, blockDim_1.x, blockDim_1.y, iElaps);
    cudaMemcpy(C_host, C_dev, nBytes, cudaMemcpyDeviceToHost); // 将计算结果从设备端拷贝到主机端
    // printMatrix(C_host, row, col);// 打印输出结果, 4096*4096 矩阵太耗费时间


    /********* 2d block and 1d grid***********/
    dim3 blockDim_2(dim_x);
    dim3 gridDim_2((row + blockDim_2.x - 1) / blockDim_2.x, col);
    iStart = cpuSecond();
    MatrixAdd<<<gridDim_2, blockDim_2>>>(A_dev, B_dev, C_dev, row, col); // 调用 CUDA 核函数
    cudaDeviceSynchronize();
    iElaps=cpuSecond() - iStart;
    printf("GPU Execution configuration<<<(%d, %d),(%d, %d)>>> Time elapsed %f sec\n",
            gridDim_2.x, gridDim_2.y, blockDim_2.x, blockDim_2.y, iElaps);
    cudaMemcpy(C_host, C_dev, nBytes, cudaMemcpyDeviceToHost); // 将计算结果从设备端拷贝到主机端
    // printMatrix(C_host, row, col);// 打印输出结果, 4096*4096 矩阵太耗费时间


    // 释放主机端内存
    free(A_host);
    free(B_host);
    free(C_host);

    // 释放设备端内存
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    return 0;
}
