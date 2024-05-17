#include <stdio.h>

__global__ void mulKernel(int *a, int *b, int *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] * b[tid];
    }
}

int main() {
    int n = 10; // 数组长度
    int a[n], b[n], c[n]; // 输入和输出数组

    // 为输入数组赋值
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = i * i;
    }

    int *dev_a, *dev_b, *dev_c; // 设备端指针

    // 分配设备端内存
    cudaMalloc((void**)&dev_a, n * sizeof(int));
    cudaMalloc((void**)&dev_b, n * sizeof(int));
    cudaMalloc((void**)&dev_c, n * sizeof(int));

    // 将输入数据从主机端拷贝到设备端
    cudaMemcpy(dev_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    // 计算线程块和线程数
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 调用 CUDA 核函数
    mulKernel<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c, n);

    // 将计算结果从设备端拷贝到主机端
    cudaMemcpy(c, dev_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印输出结果
    printf("Result: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", c[i]);
    }
    printf("\n");

    // 释放设备端内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
