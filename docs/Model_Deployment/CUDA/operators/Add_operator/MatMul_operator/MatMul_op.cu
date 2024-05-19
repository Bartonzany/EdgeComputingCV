#include <stdio.h>

#define ROWS_A 3 // 矩阵 A 的行数
#define COLS_A 2 // 矩阵 A 的列数
#define ROWS_B 2 // 矩阵 B 的行数
#define COLS_B 4 // 矩阵 B 的列数

// CUDA 核函数，执行矩阵乘法
__global__ void matrixMultiply(int *a, int *b, int *c, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rowsA && col < colsB) {
        int sum = 0;
        for (int i = 0; i < colsA; ++i) {
            sum += a[row * colsA + i] * b[i * colsB + col];
        }
        c[row * colsB + col] = sum;
    }
}

int main() {
    int *h_a, *h_b, *h_c; // 输入和输出矩阵（主机端）
    int *d_a, *d_b, *d_c; // 输入和输出矩阵（设备端）

    // 分配主机端内存
    h_a = (int*)malloc(ROWS_A * COLS_A * sizeof(int));
    h_b = (int*)malloc(ROWS_B * COLS_B * sizeof(int));
    h_c = (int*)malloc(ROWS_A * COLS_B * sizeof(int));

    // 为输入矩阵赋值
    for (int i = 0; i < ROWS_A * COLS_A; ++i) {
        h_a[i] = i;
    }
    for (int i = 0; i < ROWS_B * COLS_B; ++i) {
        h_b[i] = i * i;
    }

    // 分配设备端内存
    cudaMalloc((void**)&d_a, ROWS_A * COLS_A * sizeof(int));
    cudaMalloc((void**)&d_b, ROWS_B * COLS_B * sizeof(int));
    cudaMalloc((void**)&d_c, ROWS_A * COLS_B * sizeof(int));

    // 将输入数据从主机端拷贝到设备端
    cudaMemcpy(d_a, h_a, ROWS_A * COLS_A * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, ROWS_B * COLS_B * sizeof(int), cudaMemcpyHostToDevice);

    // 计算线程块和线程数
    dim3 blockDim(16, 16);
    dim3 gridDim((COLS_B + blockDim.x - 1) / blockDim.x, (ROWS_A + blockDim.y - 1) / blockDim.y);

    // 调用 CUDA 核函数
    matrixMultiply<<<gridDim, blockDim>>>(d_a, d_b, d_c, ROWS_A, COLS_A, COLS_B);

    // 将计算结果从设备端拷贝到主机端
    cudaMemcpy(h_c, d_c, ROWS_A * COLS_B * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印输出结果
    printf("Matrix C (result):\n");
    for (int i = 0; i < ROWS_A; ++i) {
        for (int j = 0; j < COLS_B; ++j) {
            printf("%d ", h_c[i * COLS_B + j]);
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
