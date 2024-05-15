#include <stdio.h>

#define N 3 // 矩阵的大小

// CUDA 核函数，执行矩阵加法
__global__ void matrixAdd(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        int idx = row * n + col;
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int *h_a, *h_b, *h_c; // 输入和输出矩阵（主机端）
    int *d_a, *d_b, *d_c; // 输入和输出矩阵（设备端）

    // 分配主机端内存
    h_a = (int*)malloc(N * N * sizeof(int));
    h_b = (int*)malloc(N * N * sizeof(int));
    h_c = (int*)malloc(N * N * sizeof(int));

    // 为输入矩阵赋值
    for (int i = 0; i < N * N; ++i) {
        h_a[i] = i;
        h_b[i] = i * i;
    }

    // 分配设备端内存
    cudaMalloc((void**)&d_a, N * N * sizeof(int));
    cudaMalloc((void**)&d_b, N * N * sizeof(int));
    cudaMalloc((void**)&d_c, N * N * sizeof(int));

    // 将输入数据从主机端拷贝到设备端
    cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // 计算线程块和线程数
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // 调用 CUDA 核函数
    matrixAdd<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);

    // 将计算结果从设备端拷贝到主机端
    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印输出结果
    printf("Matrix C (result):\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", h_c[i * N + j]);
        }
        printf("\n");
    }

    // 释放主机端内存
    free(h_a);
    free(h_b);
    free(h_c);

    // 释放设备端内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
